#!/usr/bin/env python3
"""MIT 모드(MOVE M + 0xAD) 첫 실험 — 드라이버 없이 SDK로 직접. 기본은 dry-run(아무것도 보내지 않음).

순서: 위치 모드 정지 확인 → 유지 토크 측정(τ_ff) → MIT 전환 → 6축 '현재 위치 유지' 100 Hz (+ J6 사인파) → 비상 정지로 종료.
자동 중단: 기준 대비 3° 이탈, 관절 속도 0.5 rad/s 초과, 모터 비활성, 상태 이상 → MotionCtrl_1(0x01)(천천히 내려앉음).
사용: python3 tools/mit_probe.py --stage hold [--go]      (--fake: 가짜 팔로 로직 시험)
"""
import argparse
import csv
import math
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "common_pkg"))
from joint_ctrl import JointCtrl, JointCtrlCfg  # noqa: E402
from vib_damp import VibDamp, VibDampCfg  # noqa: E402

DEV_LIMIT_DEG = 3.0       # 기준 대비 이탈 한계
SPEED_LIMIT = 0.5         # [rad/s] 관절 속도 한계(10-06 0xAD 튐은 3~9 rad/s)
SPEED_CONSEC = 3          # 속도 한계는 연속 3주기 초과 시만 — 정지 중 속도 피드백 잡음 최대 0.1 rad/s
SETTLE_SPEED = 0.2        # [rad/s] 위치 모드 정지 판정 속도(잡음 최대의 2배)
TAU_FF_LIMIT = 3.0        # [N·m] τ_ff 상한(시작 자세 유지 토크 최대 1.6)
TAU_CMD_LIMIT = 4.0       # [N·m] --ext-kp/--ext-kd 사용 시 중력+우리 PD 합의 상한
MISMATCH_NM = 1.0         # [N·m] 보낸 τ와 effort 차 — 10-20 200 Hz에서 +4 명령 중 effort −0.3, 반대로 등속 100 ms
TAU_STEP = 16.0 / 255      # [N·m] MIT 토크 필드 8비트(±8) 한 칸 — SDK가 int()로 버려 τ=0도 −0.031로 감, 반 칸 더해 반올림
SETTLE_S = 2.0            # 위치 모드 정지 확인 시간
TORQUE_AVG_S = 0.5        # 유지 토크 평균 시간
MODE_GRACE_S = 0.2        # MIT 전환 후 모드 피드백이 MOVE M(4)이 될 때까지 유예 — 이후 벗어나면 중단
# MOVE M 명령은 진입 때 1회만: 10-06 1초 주기 재전송 직후 J5·J6가 0° 쪽으로 튐(목표 초기화 추정)
LOG_ROOT = os.path.expanduser("~/piper_ws/logs/mit")


def read_state(p):  # → q[rad], qd[rad/s], eff[N·m], enable[6], (arm_status, ctrl_mode, mode_feed)
    js = p.GetArmJointMsgs().joint_state
    q = np.radians([getattr(js, f"joint_{i}") / 1000.0 for i in range(1, 7)])
    hs = p.GetArmHighSpdInfoMsgs()
    m = [getattr(hs, f"motor_{i}") for i in range(1, 7)]
    qd = np.array([x.motor_speed / 1000.0 for x in m])
    eff = np.array([x.effort / 1000.0 for x in m])
    en = p.GetArmEnableStatus()
    st = p.GetArmStatus().arm_status
    return q, qd, eff, np.array(en, bool), (int(st.arm_status), int(st.ctrl_mode), int(st.mode_feed))


class Probe:
    def __init__(self, p, a):
        self.p, self.a = p, a
        self.kp = np.array(a.kp, float); self.kd = np.array(a.kd, float)
        self.ext_kp = None if a.ext_kp is None else np.array(a.ext_kp, float)
        if self.ext_kp is not None:
            self.kp = np.zeros(6)  # 펌웨어 위치 항 끔 — 10-06 목표 위치가 순간 0으로 처리되는 튐 회피, P는 우리 루프가 τ_ff로
        self.ext_kd = None if a.ext_kd is None else np.array(a.ext_kd, float)
        if self.ext_kd is not None:
            self.kd = np.zeros(6)  # 펌웨어 감쇠 항도 끔 — 10-20 펌웨어 Kp 0에서도 J6 튐, 모터엔 우리 토크만
        os.makedirs(LOG_ROOT, exist_ok=True)
        tag = time.strftime("%Y%m%d_%H%M%S") + f"_{a.stage}" + ("_fake" if a.fake else "") + ("" if a.go else "_dry")
        self.path = os.path.join(LOG_ROOT, tag + ".csv")
        self.f = open(self.path, "w", newline="")
        self.w = csv.writer(self.f)
        J = range(1, 7)
        self.w.writerow(["t", "phase"] + [f"q{j}" for j in J] + [f"qd{j}" for j in J] + [f"eff{j}" for j in J]
                        + [f"qref{j}" for j in J] + [f"tff{j}" for j in J] + ["arm_status", "ctrl_mode", "mode_feed", "enabled"]
                        + (["chirp"] if a.chirp_joint else []) + (["tau_vib"] if a.vib_joint else []))
        self.t0 = time.time()
        self.imu = None; self.stream = None; self.imu_f = None
        if a.vib_joint:  # τ_vib: 실시간 수신이 IMU 기록도 맡음(장치는 한 프로세스만 잡음) — 형식은 imu_rec와 같음
            with open(self.path.replace(".csv", ".t0"), "w") as m:
                m.write(f"{self.t0:.6f}\n")
            self.imu_f = open(self.path.replace(".csv", ".imu.csv"), "w", newline="")
            iw = csv.writer(self.imu_f); iw.writerow(["t", "t_dev", "stream", "x", "y", "z"])
            if a.fake:
                self.stream = FakeImu(p, a, iw)
            else:
                from imu_stream import ImuStream
                self.stream = ImuStream(log=iw)
            hz = self.stream.start(); time.sleep(1.0)
            if self.stream.latest_gyro()[0] is None:
                raise SystemExit("★ 실시간 IMU 표본이 안 들어옴 — 손목캠 확인")
            print(f"실시간 IMU: 가속도 {hz[0]} Hz, 자이로 {hz[1]} Hz → {self.path.replace('.csv', '.imu.csv')}")
            self.vw_f = open(self.path.replace(".csv", ".vib.csv"), "w", newline="")
            self.vw = csv.writer(self.vw_f)
            self.vw.writerow(["t", "t_gyro", "td_ms", "age_ms", "n_new", "v", "wy", "ey", "vf", "vhat", "raw", "tau_vib", "enabled", "tilt",
                              "rejects", "stale", "k"])
        elif a.imu:  # 손목 D455 IMU 동시 기록(VIB D0) — 시각은 epoch, .t0로 맞춤
            with open(self.path.replace(".csv", ".t0"), "w") as m:
                m.write(f"{self.t0:.6f}\n")
            self.imu = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "imu_rec.py"),
                                         self.path.replace(".csv", ".imu.csv")])
            time.sleep(2.0)
            if self.imu.poll() is not None:
                raise SystemExit("★ IMU 기록이 시작되지 않음 — --imu 없이 하거나 손목캠 확인")
        self.dump = None
        if a.go and not a.fake:  # CAN 원본 기록 — candump 시각(epoch) − t0 = CSV t
            with open(self.path.replace(".csv", ".t0"), "w") as m:
                m.write(f"{self.t0:.6f}\n")
            self.dump = subprocess.Popen(["candump", "-L", a.can], stdout=open(self.path.replace(".csv", ".candump"), "w"),
                                         stderr=subprocess.DEVNULL)
        self.fast_n = 0
        self.spike_n = 0; self.spikes = 0
        self.tdc = None
        if a.tdc:  # 한 축 TDC(TDE + 극점 배치) — 체크리스트 2단계, 시험 관절은 펌웨어 Kp·Kd 0, 토크는 전부 우리 계산
            if not a.hybrid:  # 하이브리드는 시험 관절도 펌웨어 MIT PD 유지
                self.kp[a.joint - 1] = 0.0; self.kd[a.joint - 1] = 0.0  # 나머지 관절은 --ext 없으면 펌웨어 PD로 고정(10-20 손목 튐 대비)
            if self.ext_kp is not None:
                self.ext_kp[a.joint - 1] = 0.0
            if self.ext_kd is not None:
                self.ext_kd[a.joint - 1] = 0.0
            self.tdc = open(self.path.replace(".csv", ".tdc.csv"), "w", newline="")
            self.tw = csv.writer(self.tdc)
            self.tw.writerow(["t", "e", "ed", "qdd_f", "ar", "tau_prev", "tde", "pd", "sw", "tau", "K", "eta"])

    vib_now = {"v": 0.0}

    def log(self, phase, s, qref, tff, chirp_v=None):
        q, qd, eff, en, st = s
        self.w.writerow([f"{time.time() - self.t0:.4f}", phase] + list(np.degrees(q).round(4)) + list(qd.round(4))
                        + list(eff.round(4)) + list(np.degrees(qref).round(4)) + list(np.round(tff, 4))
                        + list(st) + [int(en.all())] + ([] if not self.a.chirp_joint else [f"{chirp_v or 0.0:.5f}"])
                        + ([] if not self.a.vib_joint else [f"{self.vib_now['v']:.5f}"]))

    def check(self, s, qref, phase, t_mit=None, vref=None):  # 중단 사유 문자열 또는 None
        q, qd, eff, en, st = s
        if self.a.speed_err and vref is not None:
            qd = qd - vref  # 빠른 사인파: 목표 속도와의 차로 판정
        if t_mit is not None and t_mit > MODE_GRACE_S and st[2] != 0x04:
            return f"모드 피드백 {st[2]} (MOVE M=4 아님)"
        dev = np.degrees(np.abs(q - qref))
        if dev.max() > self.a.dev_limit:
            return f"기준 이탈 J{int(dev.argmax()) + 1} {dev.max():.2f}° > {self.a.dev_limit}°"
        self.fast_n = self.fast_n + 1 if np.abs(qd).max() > SPEED_LIMIT else 0
        if self.fast_n >= self.a.speed_consec:
            return f"관절 속도 J{int(np.abs(qd).argmax()) + 1} {np.abs(qd).max():.2f} rad/s > {SPEED_LIMIT} ({self.a.speed_consec}주기 연속)"
        if not en.all():
            return f"모터 비활성 J{[i + 1 for i in range(6) if not en[i]]}"
        if st[0] != 0:
            return f"arm_status={st[0]:#x}"
        return None

    def run(self):
        a, p = self.a, self.p
        dt = 1.0 / a.rate
        # 1) 위치 모드 정지 확인 — Enable·처짐과 MIT 전환을 분리
        s = read_state(p)
        q, qd, eff, en, st = s
        print(f"현재 q[deg] {np.degrees(q).round(1)} | enable {en.astype(int)} | status {st}")
        if not en.all() or st[0] != 0 or st[1] != 1:
            raise SystemExit("★ 팔이 위치 모드(CAN 제어)로 Enable된 상태가 아님 — traj_test로 시작 자세에 둔 뒤 실행")
        if a.go:
            p.MotionCtrl_2(0x01, 0x01, 10, 0x00)
        t_end = time.time() + SETTLE_S
        q_ref = q.copy()
        slow_n = 0
        while time.time() < t_end:
            if a.go:
                p.JointCtrl(*[round(v * 1000) for v in np.degrees(q_ref)])
            s = read_state(p)
            self.log("settle", s, q_ref, np.zeros(6))
            slow_n = slow_n + 1 if np.abs(s[1]).max() > SETTLE_SPEED else 0  # 위치 서보 한 주기 움찔(10-20 J5 0.32 rad/s, 0.08°)은 통과
            if np.degrees(np.abs(s[0] - q_ref)).max() > 0.5 or slow_n >= SPEED_CONSEC:
                raise SystemExit(f"★ 위치 모드에서 정지 상태가 아님(이탈 {np.degrees(np.abs(s[0]-q_ref)).max():.2f}°, "
                                 f"속도 {np.abs(s[1]).max():.2f} rad/s) — 중단")
            time.sleep(dt)
        # 2) 유지 토크 측정 → τ_ff (중력 보상 근사)
        qs, es = [], []
        t_end = time.time() + TORQUE_AVG_S
        while time.time() < t_end:
            s = read_state(p); qs.append(s[0]); es.append(s[2])
            self.log("measure", s, q_ref, np.zeros(6))
            time.sleep(dt)
        q_hold = np.mean(qs, axis=0)
        tff = np.clip(np.mean(es, axis=0), -TAU_FF_LIMIT, TAU_FF_LIMIT)
        print(f"유지 자세 q[deg] {np.degrees(q_hold).round(2)}\nτ_ff[N·m] {tff.round(3)}\nKp(펌웨어) {self.kp} Kd {self.kd}"
              + (f"\nKp(우리 루프, τ_ff로) {self.ext_kp} — 상한 ±{TAU_CMD_LIMIT} N·m" if self.ext_kp is not None else "")
              + (f"\nKd(우리 루프, τ_ff로) {self.ext_kd}" if self.ext_kd is not None else "")
              + ((f"\n하이브리드 J{a.joint}: 펌웨어 PD Kp {self.kp[a.joint - 1]} Kd {self.kd[a.joint - 1]} + τ_ff(TDE, M̄ {a.mbar}), q̈ 필터 {a.acc_fc} Hz"
                  if a.hybrid else
                  f"\nTDC J{a.joint}: M̄ {a.mbar}, Ks {a.ks}, Kg {a.kg} → 등가 Kp {a.mbar * a.ks * a.kg:.2f}, "
                  f"Kd {a.mbar * (a.ks + a.kg):.3f}, q̈ 필터 {a.acc_fc} Hz")
                 + (f" | 스위칭 K {a.sw_k} (최대 {a.mbar * a.sw_k:.3f} N·m), φ {a.sw_phi}" if a.sw_k and not a.asmc else "")
                 + (f" | SS-ASMC δ {a.delta}, ε̄ {a.eps}, γ {a.gamma}, a {a.a_exp}, η [{a.eta_lo}, {a.eta_hi}], μ̄ {a.mu}, "
                    f"K̂ 상한 {a.k_max} (M̄K̂ {a.mbar * a.k_max:.3f} N·m), " + (f"sat φ {a.sw_phi}" if a.sw_phi else "sgn")
                    if a.asmc else "")
                 + (f" | ASMC-2016 φ {a.phi}, α {a.alpha}, ε {a.eps}, K̂ 상한 {a.k_max} (M̄K̂ {a.mbar * a.k_max:.3f} N·m), "
                    + (f"sat φ {a.sw_phi}" if a.sw_phi else "sgn") if a.asmc16 else "") if a.tdc else ""))
        if not a.go:
            print(f"[dry-run] MIT 전환 시 보낼 첫 프레임 (관절, q_ref[rad], q̇_ref, Kp, Kd, τ_ff):")
            for j in range(6):
                print(f"  JointMitCtrl({j + 1}, {q_hold[j]:+.4f}, 0.0, {self.kp[j]:.1f}, {self.kd[j]:.2f}, {tff[j]:+.3f})")
            if a.stage == "sine":
                print(f"[dry-run] J{a.joint} 사인파: ±{a.amp}° {a.hz} Hz, {a.cycles}주기, 진폭 2 s 램프 → q_ref 범위 "
                      f"{np.degrees(q_hold[a.joint - 1]) - a.amp:.1f} ~ {np.degrees(q_hold[a.joint - 1]) + a.amp:.1f}°")
            if a.chirp_joint:
                path = (f"dist_comp({a.dist_mode})" if a.tdc and a.chirp_joint == a.joint else "τ_ff 직접")
                print(f"[dry-run] 처프 J{a.chirp_joint}: ±{a.chirp_amp} N·m, {a.chirp_f0}–{a.chirp_f1} Hz 로그 스윕 {a.chirp_dur} s, "
                      f"MIT {a.chirp_start} s부터, 경로 {path}"
                      + (" — 실제 토크는 TDE 반응으로 처프와 달라짐(10-24 가짜 팔: accel 최대 7배, torque 저주파 1/4·중간 3.3배)" if path != "τ_ff 직접" else " — 보낸 토크 = 처프"))
            if a.vib_joint:
                from vib_damp import bp_phase_deg
                print(f"[dry-run] τ_vib J{a.vib_joint}: {a.vib_law} k {a.vib_k}, 대역 {a.vib_band[0]}–{a.vib_band[1]} Hz(8 Hz 위상 "
                      f"{bp_phase_deg(tuple(a.vib_band), 8.0, a.rate):+.1f}°), 게이트 기울기 ≥ {a.vib_gate}°, 포화 ±{a.vib_max} N·m"
                      + (", 데드밴드 반 칸" if a.vib_deadband else "") + (f", PPF {a.ppf_f} Hz ζ {a.ppf_zeta}" if a.vib_law == "ppf" else "")
                      + ", 경로 " + ("dist_comp(torque)" if a.vib_joint == a.joint and a.tdc else "τ_ff 직접"))
            if a.end_pos:
                print("[dry-run] 정상 종료 시 비상 정지 대신 위치 모드로 전환(유지 자세 목표, 2 s 감시)")
            print("[dry-run] 명령은 하나도 보내지 않음. 실제 실행은 --go")
            return "dry-run"
        # 3) MIT 전환 + 6축 유지(+사인파)
        p.MotionCtrl_2(0x01, 0x04, 0, 0xAD)
        t_start = time.time()
        hold_s = a.hold
        dur = hold_s + (a.cycles / a.hz + 2.0 if a.stage == "sine" else 0.0)
        j6 = a.joint - 1
        reason = None
        t_prev = None
        ctrl = None
        chirp_now = {"v": 0.0}  # 이번 주기 처프 토크 — TDC 관절이면 dist_comp(τ_vib와 같은 경로), 아니면 τ_ff에 직접
        if self.tdc is not None:
            sw = "asmc16" if a.asmc16 else "asmc" if a.asmc else "fixed" if a.sw_k else "none"
            ctrl = JointCtrl(JointCtrlCfg(mbar=a.mbar, ks=a.ks, kg=a.kg, acc_fc=a.acc_fc, hybrid=a.hybrid,
                                          kp_fw=self.kp[j6], kd_fw=self.kd[j6], switch=sw, sw_k=a.sw_k, sw_phi=a.sw_phi,
                                          k_max=a.k_max, eps=a.eps, phi=a.phi, alpha=a.alpha, delta=a.delta, gamma=a.gamma,
                                          a_exp=a.a_exp, eta_lo=a.eta_lo, eta_hi=a.eta_hi, mu=a.mu, dist_mode=a.dist_mode), tff[j6],
                             dist_comp=(lambda o: dist_now["v"]) if a.joint in (a.chirp_joint, a.vib_joint) else None)
        same = bool(a.chirp_joint and a.chirp_joint == a.vib_joint)  # 처프 + τ_vib 같은 관절: 합에 포화(8절 10/26 밤 2)
        dist_now = {"v": 0.0}; self.n_sat = 0; k_switched = False
        vd = None; tds = []
        if a.vib_joint:
            from imu_frame import ImuToBase
            from piper_kin import PiperKin
            kin = PiperKin()
            vd = VibDamp(kin, ImuToBase(kin), VibDampCfg(law=a.vib_law, k=a.vib_k, band=tuple(a.vib_band), gate_deg=a.vib_gate,
                                                         deadband=a.vib_deadband, tau_max=a.vib_max, ppf_f=a.ppf_f, ppf_zeta=a.ppf_zeta,
                                                         pred_f=a.vib_pred, age_max=a.vib_age_max / 1000.0, pred_mode=a.vib_pred_mode,
                                                         res_q=a.vib_q, lead_deg=a.vib_lead_deg, gate_hi=a.vib_gate_hi, hf_max=a.vib_hf_max))
        sent = None; mm_t = np.zeros(6); t_last = None
        beeped = not a.beep
        while True:
            now = time.time(); tt = now - t_start
            if tt > dur:
                break
            qr = q_hold.copy(); vr = np.zeros(6); ar = 0.0
            if a.stage == "sine" and tt > hold_s:  # 진폭을 2 s 동안 0→A로 올림(급출발 방지), --sine-joints면 여러 관절 같은 위상(a′)
                ts = tt - hold_s; ramp = min(1.0, ts / 2.0)
                w = 2 * math.pi * a.hz; A = math.radians(a.amp) * ramp
                for jj in ([j - 1 for j in a.sine_joints] if a.sine_joints else [j6]):
                    qr[jj] = q_hold[jj] + A * math.sin(w * ts)
                    vr[jj] = A * w * math.cos(w * ts)
                ar = -A * w * w * math.sin(w * ts) if (not a.sine_joints or a.joint in a.sine_joints) else 0.0
            if vd is not None and a.vib_k_switch and not k_switched and tt >= a.vib_k_switch:  # 한 MIT 시행 안 k 짝(같은 처짐)
                vd.c.k = a.vib_k2; k_switched = True
                print(f"k 전환 {a.vib_k} → {a.vib_k2} ({tt:.1f} s)", flush=True)
            if not beeped and tt >= a.beep:
                print("\a▶ 지금 툭", flush=True); beeped = True
            chirp_now["v"] = chirp(a, tt) if a.chirp_joint else 0.0
            s = read_state(p)  # 읽기 → 계산 → 송신 (우리 P가 최신 위치를 쓰도록)
            tau = tff.copy()
            vo = None
            if vd is not None:  # τ_vib — 새 자이로 표본 전부를 표본 시각 기준으로 처리, 송신 직전 시각으로 예측
                gs, ac = self.stream.gyro_after(vd.t_g)
                vo = vd.step(tt, time.time(), s[0], s[1], gs, ac); self.vib_now["v"] = vo["tau"]
            if same:
                c = chirp_now["v"] + self.vib_now["v"]; cc = float(np.clip(c, -a.vib_sum_max, a.vib_sum_max))
                self.n_sat += int(cc != c); dist_now["v"] = cc
            else:
                dist_now["v"] = chirp_now["v"] if a.chirp_joint == a.joint else self.vib_now["v"]
            if self.ext_kp is not None:
                tau = tau + self.ext_kp * (qr - s[0])
            if self.ext_kd is not None:
                tau = tau + self.ext_kd * (vr - s[1])
            if self.tdc is not None:  # τ = τ(t−L) − M̄·q̈(t−L) + M̄(q̈_d + (Ks+Kg)ė + Ks·Kg·e) + 스위칭 — src/common_pkg/joint_ctrl.py
                h = max(now - t_prev, 1e-3) if t_prev is not None else 0.0
                t_prev = now
                co = ctrl.step(s[0][j6], s[1][j6], qr[j6], vr[j6], ar, h)
                tau[j6] = co["tau"]
            if same and (a.chirp_joint != a.joint or self.tdc is None):
                tau[a.chirp_joint - 1] += dist_now["v"]
            elif not same:
                if a.chirp_joint and a.chirp_joint != a.joint or (a.chirp_joint and self.tdc is None):
                    tau[a.chirp_joint - 1] += chirp_now["v"]
                if a.vib_joint and (a.vib_joint != a.joint or self.tdc is None):
                    tau[a.vib_joint - 1] += self.vib_now["v"]
            if self.ext_kp is not None or self.ext_kd is not None or self.tdc is not None or a.chirp_joint or a.vib_joint:
                tau = np.clip(tau, -TAU_CMD_LIMIT, TAU_CMD_LIMIT)
            if self.tdc is not None:
                self.tw.writerow([f"{now - self.t0:.4f}", f"{co['e']:.6f}", f"{co['ed']:.5f}", f"{co['qdd_f']:.4f}", f"{ar:.5f}",
                                  f"{co['tau_prev']:.4f}", f"{co['tde']:.4f}", f"{co['pd']:.4f}", f"{co['sw']:.4f}", f"{tau[j6]:.4f}",
                                  f"{co['K']:.3f}", f"{co['eta']:.2f}"])
                ctrl.commit(tau[j6])  # 포화 후 값 — 실제 보낸 토크로 다음 TDE
            for j in range(6):
                p.JointMitCtrl(j + 1, float(qr[j]), float(vr[j]), float(self.kp[j]), float(self.kd[j]), float(tau[j]) + TAU_STEP / 2)
            if vo is not None and vo["t_g"] is not None:  # Td = 이 주기 JointMitCtrl 송신 완료 − 마지막 자이로 표본 시각(장치 시각 + 최소 오프셋 → 하한)
                td = (time.time() - vo["t_g"]) * 1000; tds.append(td)
                self.vw.writerow([f"{now - self.t0:.4f}", f"{vo['t_g'] - self.t0:.4f}", f"{td:.2f}", f"{vo['age'] * 1000:.2f}", vo["n_new"],
                                  f"{vo['v']:.5f}", f"{vo['wy']:.5f}", f"{vo['ey']:.5f}", f"{vo['vf']:.5f}", f"{vo['vhat']:.5f}",
                                  f"{vo['raw']:.5f}", f"{vo['tau']:.5f}", int(vo["enabled"]), "" if vo["tilt"] is None else f"{vo['tilt']:.2f}",
                                  vo["rejects"], vo["stale"], vd.c.k])
                if vo["tripped"] is not None and not getattr(self, "_trip_said", False):
                    self._trip_said = True
                    print(f"★ τ_vib 감시: {vo['trip_why']}({vo['tripped']:.1f} s) → τ_vib 끔, 시행은 k = 0으로 계속", flush=True)
                if vo["tilt"] is not None and not getattr(self, "_gate_said", False):
                    self._gate_said = True
                    print(f"기울기 게이트: {vo['tilt']:.2f}° → " + ("τ_vib 켬" if vo["enabled"] else f"★ {a.vib_gate}–{a.vib_gate_hi}° 밖, τ_vib 끔(k = 0으로 진행 — 37° 초과면 그 상태 ∠H 기준선)"), flush=True)
            self.log("mit", s, qr, tau, chirp_now["v"] if a.chirp_joint else None)
            if np.abs(s[1] - (vr if a.speed_err else 0.0)).max() > SPEED_LIMIT and self.fast_n == 0:
                self.spike_n += 1
            reason = self.check(s, qr, "mit", tt, vr)
            if sent is not None and not reason:  # 우리 토크만 쓰는 관절: 직전에 보낸 τ를 모터가 안 따르면 중단
                pure = (self.kp == 0) & (self.kd == 0)
                bad = pure & (np.abs(sent - s[2]) > MISMATCH_NM)
                mm_t = np.where(bad, mm_t + (now - t_last), 0.0)
                if (mm_t > a.mismatch_ms / 1000.0).any():
                    j = int(mm_t.argmax())
                    reason = f"명령 무시 J{j + 1}: 보낸 τ {sent[j]:+.2f} vs effort {s[2][j]:+.2f} N·m, {mm_t[j] * 1000:.0f} ms 지속"
            sent = tau.copy()
            if t_last is not None and now - t_last > 0.03:  # 루프 멈춤 — 그동안 τ 유지라 예측 보상 안 됨(8 Hz에서 70 ms = 200°)
                self.n_gap30 = getattr(self, "n_gap30", 0) + 1; self.gap_max = max(getattr(self, "gap_max", 0.0), now - t_last)
            t_last = now
            if reason:
                break
            time.sleep(max(0.0, dt - (time.time() - now)))
        self.spikes = self.spike_n  # 순간 튐: 1주기라도 속도 한계 초과
        print(f"루프 멈춤 30 ms 초과 {getattr(self, 'n_gap30', 0)}회 (최대 {getattr(self, 'gap_max', 0.0) * 1000:.0f} ms) — 수십 회면 원인 확인(8절 10/26 밤 7)")
        if tds:
            td = np.array(tds); self.td_stats = (f"Td(자이로 표본 → τ_vib 송신, 하한) 중앙 {np.median(td):.1f} ms · 95% {np.percentile(td, 95):.1f} · 최대 {td.max():.1f}"
                                                 f" | 엔코더 튐 거부 {vd.n_reject}회 · 묵은 표본(> {a.vib_age_max:.0f} ms) {vd.n_stale}/{vd.n_cycles}주기"
                                                 f" · 자이로 표본 {vd.n_samples} · 시각 이상 버림 {vd.n_badtime}"
                                                 + (f" | 처프+τ_vib 합 포화 {self.n_sat}/{vd.n_cycles}주기" if same else "")
                                                 + (f" | ★ IMU 시각 도메인 전환 {self.stream.n_domain}회 — k > 0 구간이 있으면 이 회 무효"
                                                    if getattr(self.stream, "n_domain", 0) else ""))
        # 4) 종료 — 정상 종료 + --end-pos면 위치 모드로 전환(처짐 상태 유지 시험), 아니면·문제가 있으면 비상 정지(천천히 내려앉힘)
        if a.end_pos and not reason:
            r2 = self.end_position(q_hold, dt)
            if r2 is None:
                return f"정상 종료 → 위치 모드 전환 | 순간 튐(1주기라도 {SPEED_LIMIT} rad/s 초과) {self.spikes}회"
            reason = f"위치 모드 전환 중 {r2}"
        p.MotionCtrl_1(0x01, 0, 0)
        for _ in range(int(1.0 / dt)):
            self.log("estop", read_state(p), q_hold, np.zeros(6)); time.sleep(dt)
        return (f"중단: {reason}" if reason else "정상 종료") + f" | 순간 튐(1주기라도 {SPEED_LIMIT} rad/s 초과) {self.spikes}회"

    def end_position(self, q_hold, dt, watch_s=2.0):  # MIT → 위치 모드(유지 자세 목표, 10%) 후 watch_s 감시 → 문제 사유 또는 None
        p = self.p; tgt = [round(v * 1000) for v in np.degrees(q_hold)]
        p.JointCtrl(*tgt); p.MotionCtrl_2(0x01, 0x01, 10, 0x00)
        t_end = time.time() + watch_s
        while time.time() < t_end:
            p.JointCtrl(*tgt)
            s = read_state(p); self.log("endpos", s, q_hold, np.zeros(6))
            r = self.check(s, q_hold, "endpos")
            if r:
                return r
            time.sleep(dt)
        st = read_state(p)[4]  # V1.7-3: MIT를 빠져나오면 대기(ctrl_mode 0, Enable 유지·자세 유지)로 감, MotionCtrl_2 반복으로도 CAN 제어 복귀 안 됨(10-26)
        return None if st[2] != 0x04 else f"전환 미완료(ctrl_mode {st[1]}, mode_feed {st[2]})"


def chirp(a, tt):  # MIT 진입 뒤 --chirp-start s부터 로그 스윕 f0→f1, --chirp-dur 동안, 앞뒤 0.5 s 진폭 램프 — --vib-k-switch면 전환 뒤 한 번 더
    if getattr(a, "vib_k_switch", 0) and tt >= a.vib_k_switch:
        tt -= a.vib_k_switch
    t = tt - a.chirp_start; T = a.chirp_dur
    if t < 0 or t > T:
        return 0.0
    k = a.chirp_f1 / a.chirp_f0
    ph = 2 * math.pi * a.chirp_f0 * T / math.log(k) * (k ** (t / T) - 1)
    return a.chirp_amp * min(1.0, t / 0.5, (T - t) / 0.5) * math.sin(ph)


def bus_type(p):  # SDK 내부 CAN 버스 객체의 클래스 이름(ThreadSafeBus 적용 확인용)
    seen = set()
    def walk(o, depth):
        if depth > 3 or id(o) in seen:
            return None
        seen.add(id(o))
        for v in getattr(o, "__dict__", {}).values():
            n = type(v).__name__
            if n.endswith("Bus"):
                return n
            r = walk(v, depth + 1)
            if r:
                return r
        return None
    return walk(p, 0) or "확인 불가"


def driver_running():  # 실제 드라이버 프로세스만(python 실행 + 설치 경로의 piper_driver_node) — 셸 명령줄 오탐 방지
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        if len(argv) > 1 and argv[0].endswith(b"python3") and argv[1].endswith(b"/piper_driver_node"):
            return True
    return False


def make_fake(kick=False, modedrop=False, refzero=False, friction=False, inertia=0.05):  # 가짜 팔 — 위치 모드는 목표 유지, MIT는 τ=PD+ff로 단순 2차 동역학(중력 = 시작 유지 토크)
    from types import SimpleNamespace as NS

    class Fake:
        def __init__(s):
            s.q = np.radians([0, 103.6, -54.8, 0, 46.3, -15.0]); s.qd = np.zeros(6)
            s.g = np.array([0.087, -1.582, -0.904, 0.013, -0.08, 0.008])  # 실측 유지 토크
            s.mode = 1; s.mit = False; s.cmd = {}; s.t = time.time(); s.kicked = False; s.en = True; s.sent = []
        def _step(s):
            now = time.time(); h = min(now - s.t, 0.02); s.t = now
            if s.mit:
                tau = np.zeros(6)
                glitch = refzero and int(now * 2) % 3 == 0 and (now % 0.5) < 0.012  # 가끔 한 주기 동안 목표 위치를 0으로
                for j, (qr, vr, kp, kd, tf) in s.cmd.items():
                    tau[j] = kp * ((0.0 if (glitch and j == 2) else qr) - s.q[j]) + kd * (vr - s.qd[j]) + tf
                acc = (tau - s.g) / inertia - 2.0 * s.qd
                if friction:  # 정지마찰 0.25·운동마찰 0.15 N·m — 10-20 J1·J3 헌팅 흉내
                    net = tau - s.g
                    for j in range(6):
                        if abs(s.qd[j]) < 0.01 and abs(net[j]) < 0.25:
                            acc[j] = 0.0; s.qd[j] = 0.0
                        else:
                            acc[j] -= np.sign(s.qd[j] if abs(s.qd[j]) >= 0.01 else net[j]) * 0.15 / inertia
                if kick and not s.kicked:
                    acc[2] -= 400.0; s.kicked = True  # 10-06 같은 진입 튐 흉내
                s.qd = s.qd + acc * h; s.q = s.q + s.qd * h
        def MotionCtrl_2(s, c, m, spd, flag): s.sent.append(("Motion", m, flag)); s.mode = m; s.mit = (m == 0x04 and flag == 0xAD)
        def MotionCtrl_1(s, e, a, b): s.sent.append(("Stop", e)); s.mit = False; s.qd[:] = 0
        def JointCtrl(s, *c): s.sent.append(("Joint",))
        def JointMitCtrl(s, j, qr, vr, kp, kd, tf): s.cmd[j - 1] = (qr, vr, kp, kd, tf)
        def GetArmJointMsgs(s):
            s._step(); return NS(joint_state=NS(**{f"joint_{i + 1}": int(math.degrees(s.q[i]) * 1000) for i in range(6)}))
        def GetArmHighSpdInfoMsgs(s):
            tau = s.g if not s.mit else np.array([s.cmd.get(j, (0, 0, 0, 0, 0))[4] for j in range(6)])
            return NS(**{f"motor_{i + 1}": NS(motor_speed=int(s.qd[i] * 1000), effort=int(tau[i] * 1000)) for i in range(6)})
        def GetArmEnableStatus(s): return [s.en] * 6
        def GetArmStatus(s):
            if modedrop and s.mit and not hasattr(s, "t_mit"):
                s.t_mit = time.time()
            m = 1 if (modedrop and s.mit and time.time() - s.t_mit > 1.0) else s.mode
            return NS(arm_status=NS(arm_status=0, ctrl_mode=1, mode_feed=m))
    return Fake()


class FakeImu:  # 가짜 팔용 IMU — 8 Hz 피치 모드(J2 τ로 구동, 툭 충격) + 강체 J_ω q̇, 전달 지연, 정지 기울기 — 부호·게이트 로직 시험용(모델 아님)
    def __init__(s, arm, a, log, f0=8.0, zeta=0.03, b=6.0):
        from imu_frame import ImuToBase
        from link_est import jacobians
        from piper_kin import PiperKin
        s.arm, s.a, s.log = arm, a, log; s.kin = PiperKin(); s.tb = ImuToBase(s.kin); s.jac = jacobians
        s.w = 2 * math.pi * f0; s.z = zeta; s.b = b; s.x = s.xd = 0.0; s.hist = []; s.t = None; s.tap_done = False; s.t_mit = None
        s.tau0 = None

    def start(s):
        s.t = time.time(); return (63, 200)

    def stop(s):
        pass

    def gap_stats(s):
        return {"a": (0, 0, 0, 0), "g": (0, 0, 0, 0)}

    def _adv(s):  # 5 ms 표본으로 모드 적분(J2 τ − 진입 τ로 구동), 기록 — 호출이 뜸하면 밀린 구간은 건너뜀
        now = time.time()
        if s.arm.mit and s.t_mit is None:
            s.t_mit = now
        s.t = max(s.t, now - 0.05)
        while s.t + 0.005 <= now:
            s.t += 0.005; arm = s.arm
            tau2 = arm.cmd.get(1, (0, 0, 0, 0, 0))[4] if arm.mit else 0.0
            if arm.mit and s.tau0 is None:
                s.tau0 = tau2
            u = (tau2 - s.tau0) if s.tau0 is not None else 0.0
            if s.a.fake_tap and s.t_mit is not None and not s.tap_done and s.t - s.t_mit >= s.a.fake_tap:
                s.xd += 0.3; s.tap_done = True
            s.xd += 0.005 * (s.b * u - 2 * s.z * s.w * s.xd - s.w * s.w * s.x); s.x += 0.005 * s.xd
            Jw, _ = s.jac(s.kin, arm.q, s.tb.R_l6)
            wb = Jw @ arm.qd + np.array([0.0, s.xd, 0.0]); R = s.tb.rot(arm.q)
            g_raw = R.T @ wb; tl = math.radians(s.a.fake_tilt); a_raw = 9.67 * np.array([0.0, math.sin(tl), -math.cos(tl)])
            s.hist.append((s.t, g_raw, a_raw))
            if s.log is not None:
                s.log.writerow((f"{s.t:.6f}", f"{s.t * 1000:.3f}", "g", *[f"{v:.5f}" for v in g_raw]))
                s.log.writerow((f"{s.t:.6f}", f"{s.t * 1000:.3f}", "a", *[f"{v:.5f}" for v in a_raw]))
        s.hist = s.hist[-400:]

    def latest_gyro(s):
        s._adv(); due = time.time() - s.a.fake_delay / 1000.0
        for t, g, ac in reversed(s.hist):
            if t <= due:
                return t, g, ac
        return None, None, None

    def gyro_after(s, t0):  # 표본 나이: --fake-td-file이면 실기 td_ms 시퀀스를 호출마다 차례로, 아니면 고정 --fake-delay
        s._adv()
        if s.a.fake_td_file and not hasattr(s, "tds"):
            import pandas as pd
            s.tds = pd.read_csv(s.a.fake_td_file).td_ms.values / 1000.0; s.k = 0
        if hasattr(s, "tds"):
            d = float(s.tds[s.k % len(s.tds)]); s.k += 1
        else:
            d = s.a.fake_delay / 1000.0
        due = time.time() - d
        g = [(t, gr) for t, gr, _ in s.hist if t <= due and (t0 is None or t > t0)]
        return g[-40:], (s.hist[-1][2] if s.hist else None)


def print_tilt(path):  # 링크 처짐 상태(VIB_E3·8절 10/26): MIT 시작·끝 1.5 s의 IMU 정지 기울기 — 엔코더가 같아도 회마다 2–4° 다름
    try:
        from imu_frame import static_tilt
        with open(path) as f:
            rows = [(float(r["t"]), r["phase"]) for r in csv.DictReader(f)]
        tm = [t for t, ph in rows if ph == "mit"]; ts = [t for t, ph in rows if ph == "settle"]
        out = []
        if ts:
            out.append(f"정지 확인 {static_tilt(path, ts[0], ts[-1]):.2f}°")
        if tm:
            out.append(f"MIT 시작 {static_tilt(path, tm[0] + 0.3, tm[0] + 1.8):.2f}°")
            out.append(f"MIT 끝 {static_tilt(path, tm[-1] - 1.5, tm[-1]):.2f}°")
        print("링크 정지 기울기(IMU atan2(y,−z), 33° 이상이 E3 '켜진' 쪽): " + " | ".join(out))
    except Exception as e:  # 기울기는 참고값 — 실패해도 실험 결과에는 영향 없음
        print(f"링크 정지 기울기 계산 실패: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["hold", "sine"], default="hold")
    ap.add_argument("--go", action="store_true", help="실제로 명령 송신(없으면 dry-run)")
    ap.add_argument("--fake", action="store_true", help="가짜 팔로 로직 시험(CAN 미사용)")
    ap.add_argument("--fake-kick", action="store_true", help="가짜 팔에 진입 튐 주입(중단 로직 시험)")
    ap.add_argument("--fake-modedrop", action="store_true", help="가짜 팔 모드가 1 s 뒤 풀림(중단 로직 시험)")
    ap.add_argument("--fake-friction", action="store_true", help="가짜 팔에 정지·운동 마찰(헌팅 흉내)")
    ap.add_argument("--fake-refzero", action="store_true", help="가짜 팔이 가끔 J3 목표 위치를 0으로 처리(실기 튐 흉내)")
    ap.add_argument("--kp", type=float, nargs=6, default=[10, 10, 10, 10, 10, 10])
    ap.add_argument("--kd", type=float, nargs=6, default=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    ap.add_argument("--ext-kp", type=float, nargs=6, default=None,
                    help="펌웨어 Kp=0으로 두고 우리 루프가 τ_ff = 중력 + ext_kp·(q_ref−q)로 위치를 잡음")
    ap.add_argument("--ext-kd", type=float, nargs=6, default=None,
                    help="펌웨어 Kd=0으로 두고 우리 루프가 τ_ff에 ext_kd·(q̇_ref−q̇)를 더함")
    ap.add_argument("--tdc", action="store_true", help="--joint 한 축을 TDC로(나머지는 --ext-kp/--ext-kd PD)")
    ap.add_argument("--mbar", type=float, default=0.005, help="TDC 관성 추정 M̄[kg·m²] — 작게 시작해서 키움")
    ap.add_argument("--ks", type=float, default=20.0, help="TDC 극점 Ks[1/s]")
    ap.add_argument("--kg", type=float, default=20.0, help="TDC 극점 Kg[1/s]")
    ap.add_argument("--sw-k", type=float, default=0.0, help="TDC 스위칭 게인 K[rad/s²] — 0이면 TDC만, M̄·K가 최대 보정 토크")
    ap.add_argument("--sw-phi", type=float, default=0.2, help="스위칭 경계층 φ[rad/s] — sat(s/φ)로 채터링 완화, --asmc에서 0이면 논문대로 sgn")
    ap.add_argument("--asmc", action="store_true", help="TDC + SS-ASMC(Baek & Kwon 2020 식 12–15) 적응 스위칭")
    ap.add_argument("--hybrid", action="store_true", help="--tdc 관절도 펌웨어 MIT PD(--kp/--kd) 유지, 우리는 TDE 보정 + M̄q̈_d(+ 스위칭)만 τ_ff로")
    ap.add_argument("--asmc16", action="store_true", help="TDC + Baek·Jin·Han 2016 IEEE TIE 식 (11) 적응 스위칭")
    ap.add_argument("--phi", type=float, default=200.0, help="ASMC-2016 φ — 적응 속도")
    ap.add_argument("--alpha", type=float, default=1.0, help="ASMC-2016 α — 증가 φ|s|/α, 감소 φα/|s|")
    ap.add_argument("--delta", type=float, default=1.0, help="SS-ASMC δ̄ — K̂ 적응률")
    ap.add_argument("--eps", type=float, default=0.05, help="SS-ASMC ε̄[rad/s] — |s|가 이보다 크면 K̂ 증가, 작으면 감소")
    ap.add_argument("--gamma", type=float, default=2e4, help="SS-ASMC γ̄ — η̂ 적응률")
    ap.add_argument("--a-exp", type=float, default=1.5, help="SS-ASMC a — child 법칙 지수")
    ap.add_argument("--eta-lo", type=float, default=100.0, help="SS-ASMC η* 하한")
    ap.add_argument("--eta-hi", type=float, default=2000.0, help="SS-ASMC η̄* 상한")
    ap.add_argument("--mu", type=float, default=200.0, help="SS-ASMC μ̄ — |K̂ 변화율| 상한[1/s²·s]")
    ap.add_argument("--k-max", type=float, default=20.0, help="K̂ 안전 상한(논문에 없음) — M̄K̂가 최대 스위칭 토크")
    ap.add_argument("--acc-fc", type=float, default=15.0, help="q̈ 추정 저역통과 차단 주파수[Hz]")
    ap.add_argument("--dev-limit", type=float, default=DEV_LIMIT_DEG, help="MIT 중 기준 이탈 중단 한계[deg] — 내부 튐(최대 4.75°) 관찰 시 6")
    ap.add_argument("--speed-err", action="store_true", help="MIT 중 속도 한계를 |q̇ − q̇_ref|로 판정(빠른 사인파용, 기본은 |q̇|)")
    ap.add_argument("--mismatch-ms", type=float, default=60.0, help="우리 토크 관절에서 |보낸 τ − effort| > 1 N·m 지속 시 중단[ms]")
    ap.add_argument("--speed-consec", type=int, default=SPEED_CONSEC, help="MIT 중 속도 한계 연속 주기 — 내부 튐(20–30 ms) 관찰 시 10")
    ap.add_argument("--hold", type=float, default=5.0, help="유지 시간[s]")
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--joint", type=int, default=6)
    ap.add_argument("--amp", type=float, default=5.0, help="사인파 진폭[deg]")
    ap.add_argument("--hz", type=float, default=0.25)
    ap.add_argument("--cycles", type=float, default=3.0)
    ap.add_argument("--can", default="can_piper")
    ap.add_argument("--imu", action="store_true", help="손목 D455 IMU 동시 기록(tools/imu_rec.py)")
    ap.add_argument("--dist-mode", choices=["accel", "torque"], default="accel", help="dist_comp(처프·τ_vib)를 다음 TDE에 넘길지")
    ap.add_argument("--chirp-joint", type=int, default=0, help="처프 토크를 더할 관절(0이면 없음) — --joint와 같으면 dist_comp 경로")
    ap.add_argument("--chirp-amp", type=float, default=0.12, help="처프 진폭[N·m] — MIT 토크 한 칸 0.063, 최소 두 칸")
    ap.add_argument("--chirp-f0", type=float, default=2.0); ap.add_argument("--chirp-f1", type=float, default=20.0)
    ap.add_argument("--chirp-dur", type=float, default=20.0); ap.add_argument("--chirp-start", type=float, default=2.0)
    ap.add_argument("--beep", type=float, default=0.0, help="MIT 진입 뒤 이 시각[s]에 툭 신호 표시(VIB D0 조건 b), 0이면 없음")
    ap.add_argument("--vib-joint", type=int, default=0, help="τ_vib를 넣을 관절(0이면 없음) — --tdc --joint와 같으면 dist_comp(torque)")
    ap.add_argument("--vib-law", choices=["vel", "ppf"], default="vel")
    ap.add_argument("--vib-k", type=float, default=0.0, help="vel: N·m/(rad/s), ppf: N·m/rad — 0이면 계산·기록만(기준선)")
    ap.add_argument("--vib-band", type=float, nargs=2, default=[4.0, 16.0], help="변형 속도 대역통과[Hz]")
    ap.add_argument("--vib-gate", type=float, default=35.0, help="MIT 시작 기울기가 이보다 작으면 τ_vib 끔[deg]")
    ap.add_argument("--vib-deadband", action="store_true", help="|τ_vib| < 반 칸이면 0")
    ap.add_argument("--vib-max", type=float, default=0.3, help="τ_vib 포화[N·m]")
    ap.add_argument("--vib-pred", type=float, default=8.0, help="조화 예측 주파수[Hz] — 표본 나이만큼 앞당김, 0이면 예측 안 함(처프 피크 8.0, 8절 10/26 밤 2)")
    ap.add_argument("--vib-k2", type=float, default=0.0, help="--vib-k-switch 뒤 k")
    ap.add_argument("--vib-pred-mode", choices=["res", "old"], default="res", help="res: 공진기 뒤진 직교(Q, 1 Hz HP), old: 대역통과 + 앞선 직교(10-26 리밋 사이클)")
    ap.add_argument("--vib-q", type=float, default=1.0, help="공진기 Q")
    ap.add_argument("--vib-lead-deg", type=float, default=0.0, help="φ_lead[deg] — θ = ω₀a + φ_lead, 60 넘기지 말 것")
    ap.add_argument("--vib-gate-hi", type=float, default=37.0, help="기울기 게이트 상한[deg]")
    ap.add_argument("--vib-hf-max", type=float, default=0.02, help="감시: τ_vib 16 Hz 이상 rms(0.25 s) 상한[N·m]")
    ap.add_argument("--fake-inertia", type=float, default=0.05, help="가짜 팔 관절 관성[kg·m²] — 실기 고주파 실효 관성은 TDE M̄ 0.01 수준")
    ap.add_argument("--fake-enc-delay", type=int, default=0, help="가짜 팔: 엔코더(q, q̇) 보고를 N번 읽기 늦춤 — 자이로·엔코더 강체 몫 고주파 불일치")
    ap.add_argument("--vib-k-switch", type=float, default=0.0, help="MIT 진입 뒤 이 시각[s]에 k → --vib-k2, 처프는 전환 뒤 한 번 더(같은 처짐에서 짝 비교)")
    ap.add_argument("--vib-sum-max", type=float, default=0.3, help="처프 + τ_vib 합 포화[N·m]")
    ap.add_argument("--sine-joints", type=int, nargs="*", default=[], help="--stage sine을 여러 관절 같은 위상으로(a′: 2 3)")
    ap.add_argument("--vib-age-max", type=float, default=60.0, help="표본 나이 상한[ms] — 넘으면 직전 τ 유지 후 반씩 줄임")
    ap.add_argument("--fake-td-file", default="", help="가짜 IMU: 실기 .vib.csv의 td_ms 시퀀스를 표본 나이로 재생(수신 몰림 재현)")
    ap.add_argument("--fake-encspike", action="store_true", help="가짜 팔: 1.5 s마다 J3 엔코더 속도만 한 주기 튐")
    ap.add_argument("--ppf-f", type=float, default=8.0); ap.add_argument("--ppf-zeta", type=float, default=0.3)
    ap.add_argument("--end-pos", action="store_true", help="정상 종료 시 비상 정지 대신 위치 모드로 전환(처짐 상태 유지 시험)")
    ap.add_argument("--fake-tilt", type=float, default=36.0, help="가짜 IMU 정지 기울기[deg]")
    ap.add_argument("--fake-tap", type=float, default=5.0, help="가짜 IMU 모드에 툭(속도 충격)을 넣을 MIT 시각[s], 0이면 없음")
    ap.add_argument("--fake-delay", type=float, default=10.0, help="가짜 IMU 자이로 전달 지연[ms]")
    a = ap.parse_args()
    if a.vib_lead_deg > 60:
        raise SystemExit("φ_lead는 60° 이하(8절 10/26 밤 4)")
    if a.vib_joint and ((a.chirp_joint and a.chirp_joint != a.vib_joint) or a.vib_max > 0.5 or a.vib_sum_max > 0.5
                        or (a.vib_joint == a.joint and a.tdc and a.dist_mode != "torque")):
        raise SystemExit("τ_vib: 처프는 같은 관절만, 포화 0.5 N·m 이하, TDC 관절이면 --dist-mode torque(8절 10/26)")
    if a.vib_k_switch and (not a.vib_joint or (a.chirp_joint and (a.hold < 2 * a.vib_k_switch or a.chirp_start + a.chirp_dur > a.vib_k_switch))):
        raise SystemExit("--vib-k-switch T: τ_vib 필요, 처프면 --hold ≥ 2T이고 처프는 한 구간(chirp-start + chirp-dur ≤ T) 안에")
    if a.amp > 10:
        raise SystemExit("사인파 진폭은 10° 이하로")
    if a.chirp_joint and (a.chirp_amp > 0.25 or a.stage != "hold" or a.hold < a.chirp_start + a.chirp_dur):
        raise SystemExit("처프는 --stage hold, 진폭 0.25 N·m 이하, --hold ≥ chirp-start + chirp-dur")
    if a.fake:
        p = make_fake(kick=a.fake_kick, modedrop=a.fake_modedrop, refzero=a.fake_refzero, friction=a.fake_friction, inertia=a.fake_inertia)
        if a.fake_enc_delay:  # 엔코더 보고 지연(읽기 N번) — 실기의 자이로·엔코더 고주파 불일치 흉내
            import collections
            jm0, hs0d = p.GetArmJointMsgs, p.GetArmHighSpdInfoMsgs; qj = collections.deque(maxlen=a.fake_enc_delay + 1); qh = collections.deque(maxlen=a.fake_enc_delay + 1)
            def jm_d():
                qj.append(jm0()); return qj[0]
            def hs_d():
                qh.append(hs0d()); return qh[0]
            p.GetArmJointMsgs = jm_d; p.GetArmHighSpdInfoMsgs = hs_d
        if a.fake_encspike:  # 1.5 s마다 한 번 읽기에서 J3 q̇만 +2.5 rad/s(팔·가짜 IMU는 그대로) — 엔코더만 튐 거부 시험
            hs0 = p.GetArmHighSpdInfoMsgs; p.spk_t = time.time()
            def hs_spiky():
                r = hs0()
                if time.time() - p.spk_t > 1.5:
                    p.spk_t = time.time(); r.motor_3.motor_speed += 2500
                return r
            p.GetArmHighSpdInfoMsgs = hs_spiky
    else:
        if driver_running():
            raise SystemExit("★ piper_driver_node가 실행 중 — CAN 명령이 섞이지 않게 먼저 종료할 것")
        import can
        def _ts_bus(*args, bustype=None, **kw):  # SDK가 옛 인자명 bustype을 넘김 → interface로 바꿔 ThreadSafeBus 생성
            kw.setdefault("interface", bustype)
            kw["local_loopback"] = True  # SDK는 False로 끔 → 우리 송신 프레임이 candump에 안 남음(10-20)
            return can.ThreadSafeBus(*args, **kw)
        can.interface.Bus = _ts_bus  # SDK 수신 스레드와 우리 송신이 같은 버스를 잠금 없이 공유 → MIT 순간 튐(커뮤니티 포크 수정과 동일)
        from piper_sdk import C_PiperInterface_V2
        p = C_PiperInterface_V2(a.can); p.ConnectPort(); time.sleep(0.5)
        print(f"CAN 버스: {bus_type(p)}")
        p.SearchPiperFirmwareVersion(); time.sleep(0.3)
        print(f"펌웨어: {p.GetPiperFirmwareVersion()} | piper_sdk {__import__('importlib.metadata').metadata.version('piper_sdk')}")
    pr = Probe(p, a)
    try:
        res = pr.run()
    except KeyboardInterrupt:
        if a.go:
            p.MotionCtrl_1(0x01, 0, 0)
        res = "사용자 중단(Ctrl-C) — 비상 정지 송신"
    finally:
        pr.f.close()
        if pr.dump:
            pr.dump.terminate()
        if pr.stream is not None:
            pr.stream.stop(); pr.imu_f.close(); pr.vw_f.close()
            print("IMU 빈틈(표본 수, 빈틈 수, 합 ms, 최대 ms):", pr.stream.gap_stats())
        if pr.imu:
            pr.imu.terminate()
            try:
                pr.imu.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pr.imu.kill()
        if pr.tdc:
            pr.tdc.close()
    print(f"결과: {res}\n기록: {pr.path}" + (f"\nIMU: {pr.path.replace('.csv', '.imu.csv')}" if pr.imu else "") + (f"\nCAN 원본: {pr.path.replace('.csv', '.candump')}" if pr.dump else ""))
    if getattr(pr, "td_stats", None):
        print(pr.td_stats)
    if pr.imu or pr.stream is not None:
        print_tilt(pr.path)
    if a.go and not a.fake:
        if "위치 모드 전환 |" in res:
            print("팔은 대기 모드(Enable 유지)로 유지 자세에 서 있습니다 — 내려앉지 않음. 다음 시행 전 clear_arm_fault → traj_test 필요.")
        else:
            print("팔은 비상 정지(천천히 내려앉음) 상태입니다. 받친 채로 clear_arm_fault.py로 복구한 뒤 다음 실험을 하세요.")


if __name__ == "__main__":
    main()
