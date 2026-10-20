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
                        + [f"qref{j}" for j in J] + [f"tff{j}" for j in J] + ["arm_status", "ctrl_mode", "mode_feed", "enabled"])
        self.t0 = time.time()
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

    def log(self, phase, s, qref, tff):
        q, qd, eff, en, st = s
        self.w.writerow([f"{time.time() - self.t0:.4f}", phase] + list(np.degrees(q).round(4)) + list(qd.round(4))
                        + list(eff.round(4)) + list(np.degrees(qref).round(4)) + list(np.round(tff, 4))
                        + list(st) + [int(en.all())])

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
            print("[dry-run] 명령은 하나도 보내지 않음. 실제 실행은 --go")
            return "dry-run"
        # 3) MIT 전환 + 6축 유지(+사인파)
        p.MotionCtrl_2(0x01, 0x04, 0, 0xAD)
        t_start = time.time()
        hold_s = a.hold
        dur = hold_s + (a.cycles / a.hz + 2.0 if a.stage == "sine" else 0.0)
        j6 = a.joint - 1
        reason = None
        tau_prev = tff[j6]; qd_prev = None; qdd_f = 0.0; t_prev = None
        pd_fw_prev = 0.0
        k_hat = 0.0; eta = a.eta_lo  # SS-ASMC 스위칭 게인 K̂, 갱신 파라미터 η̂(하한에서 시작)
        sent = None; mm_t = np.zeros(6); t_last = None
        while True:
            now = time.time(); tt = now - t_start
            if tt > dur:
                break
            qr = q_hold.copy(); vr = np.zeros(6); ar = 0.0
            if a.stage == "sine" and tt > hold_s:  # 진폭을 2 s 동안 0→A로 올림(급출발 방지)
                ts = tt - hold_s; ramp = min(1.0, ts / 2.0)
                w = 2 * math.pi * a.hz; A = math.radians(a.amp) * ramp
                qr[j6] = q_hold[j6] + A * math.sin(w * ts)
                vr[j6] = A * w * math.cos(w * ts)
                ar = -A * w * w * math.sin(w * ts)
            s = read_state(p)  # 읽기 → 계산 → 송신 (우리 P가 최신 위치를 쓰도록)
            tau = tff.copy()
            if self.ext_kp is not None:
                tau = tau + self.ext_kp * (qr - s[0])
            if self.ext_kd is not None:
                tau = tau + self.ext_kd * (vr - s[1])
            if self.tdc is not None:  # τ = τ(t−L) − M̄·q̈(t−L) + M̄(q̈_d + (Ks+Kg)ė + Ks·Kg·e)
                h = 0.0
                if qd_prev is not None:
                    h = max(now - t_prev, 1e-3)
                    al = h / (h + 1.0 / (2 * math.pi * a.acc_fc))
                    qdd_f += al * ((s[1][j6] - qd_prev) / h - qdd_f)
                qd_prev = s[1][j6]; t_prev = now
                e = qr[j6] - s[0][j6]; ed = vr[j6] - s[1][j6]
                tde = tau_prev - a.mbar * qdd_f
                if a.hybrid:  # τ_ff = [τ_ff + 펌웨어 PD](t−L) − M̄q̈(t−L) + M̄q̈_d — PD는 드라이버가, 우리는 TDE 보정만(10-20 고정 PD는 매끈, TDC는 속도 변화에 강함)
                    tde += pd_fw_prev
                    pd = a.mbar * ar
                else:
                    pd = a.mbar * (ar + (a.ks + a.kg) * ed + a.ks * a.kg * e)
                sv = ed + a.ks * e  # 논문 (6) s = ė + Ks·e
                if a.asmc16:  # Baek·Jin·Han 2016 TIE 식 (11): |s| ≥ ε면 φ|s|/α로 증가, 작으면 −φα/|s|로 급감
                    th = 1.0 if abs(sv) - a.eps >= 0 else -1.0
                    if k_hat > 1e-4:  # 논문 실험의 k̂ > 0.0001 처리
                        kdot = a.phi * (max(abs(sv), 1e-6) / a.alpha) ** th * th  # s = 0이면 감소항 발산 방지
                    else:
                        kdot = a.phi * abs(sv) / a.alpha
                    k_hat = float(np.clip(k_hat + kdot * h, 0.0, a.k_max))
                    sgn = (1.0 if sv >= 0 else -1.0) if not a.sw_phi else float(np.clip(sv / a.sw_phi, -1.0, 1.0))
                    sw = a.mbar * k_hat * sgn
                elif a.asmc:  # 논문 (12)–(15) SS-ASMC — parent (13) K̂, child (14) η̂, 오일러 적분
                    chi = 1.0 if abs(sv) - a.eps >= 0 else -1.0
                    kdot = a.delta * eta * abs(sv) * (chi if k_hat > 0 and sv != 0 else 1.0)
                    sat_k = abs(kdot) >= a.mu
                    kdot = float(np.clip(kdot, -a.mu, a.mu))
                    if sat_k:
                        etadot = 0.0
                    elif eta >= a.eta_hi:
                        etadot = -a.gamma * abs(sv)
                    elif eta <= a.eta_lo:
                        etadot = a.gamma * abs(sv)
                    else:
                        etadot = -a.gamma * abs(sv) ** a.a_exp * chi
                    k_hat = float(np.clip(k_hat + kdot * h, 0.0, a.k_max))
                    eta = float(np.clip(eta + etadot * h, a.eta_lo, a.eta_hi))
                    sgn = (1.0 if sv >= 0 else -1.0) if not a.sw_phi else float(np.clip(sv / a.sw_phi, -1.0, 1.0))
                    sw = a.mbar * k_hat * sgn
                else:  # 3단계 고정 스위칭 M̄·K·sat(s/φ)
                    sw = a.mbar * a.sw_k * float(np.clip(sv / a.sw_phi, -1.0, 1.0)) if a.sw_k else 0.0
                tau[j6] = tde + pd + sw
            if self.ext_kp is not None or self.ext_kd is not None or self.tdc is not None:
                tau = np.clip(tau, -TAU_CMD_LIMIT, TAU_CMD_LIMIT)
            if self.tdc is not None:
                self.tw.writerow([f"{now - self.t0:.4f}", f"{e:.6f}", f"{ed:.5f}", f"{qdd_f:.4f}", f"{ar:.5f}",
                                  f"{tau_prev:.4f}", f"{tde:.4f}", f"{pd:.4f}", f"{sw:.4f}", f"{tau[j6]:.4f}",
                                  f"{k_hat:.3f}", f"{eta:.2f}"])
                tau_prev = tau[j6]  # 포화 후 값 — 실제 보낸 토크로 다음 TDE
                pd_fw_prev = self.kp[j6] * e + self.kd[j6] * ed if a.hybrid else 0.0  # 펌웨어 PD 추정(내부 게인 배율 1 가정)
            for j in range(6):
                p.JointMitCtrl(j + 1, float(qr[j]), float(vr[j]), float(self.kp[j]), float(self.kd[j]), float(tau[j]) + TAU_STEP / 2)
            self.log("mit", s, qr, tau)
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
            sent = tau.copy(); t_last = now
            if reason:
                break
            time.sleep(max(0.0, dt - (time.time() - now)))
        self.spikes = self.spike_n  # 순간 튐: 1주기라도 속도 한계 초과
        # 4) 종료 — 첫 실험은 위치 모드로 되돌리지 않고 비상 정지(천천히 내려앉힘)
        p.MotionCtrl_1(0x01, 0, 0)
        for _ in range(int(1.0 / dt)):
            self.log("estop", read_state(p), q_hold, np.zeros(6)); time.sleep(dt)
        return (f"중단: {reason}" if reason else "정상 종료") + f" | 순간 튐(1주기라도 {SPEED_LIMIT} rad/s 초과) {self.spikes}회"


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


def make_fake(kick=False, modedrop=False, refzero=False, friction=False):  # 가짜 팔 — 위치 모드는 목표 유지, MIT는 τ=PD+ff로 단순 2차 동역학(중력 = 시작 유지 토크)
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
                acc = (tau - s.g) / 0.05 - 2.0 * s.qd
                if friction:  # 정지마찰 0.25·운동마찰 0.15 N·m — 10-20 J1·J3 헌팅 흉내
                    net = tau - s.g
                    for j in range(6):
                        if abs(s.qd[j]) < 0.01 and abs(net[j]) < 0.25:
                            acc[j] = 0.0; s.qd[j] = 0.0
                        else:
                            acc[j] -= np.sign(s.qd[j] if abs(s.qd[j]) >= 0.01 else net[j]) * 0.15 / 0.05
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
    a = ap.parse_args()
    if a.amp > 10:
        raise SystemExit("사인파 진폭은 10° 이하로")
    if a.fake:
        p = make_fake(kick=a.fake_kick, modedrop=a.fake_modedrop, refzero=a.fake_refzero, friction=a.fake_friction)
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
        if pr.tdc:
            pr.tdc.close()
    print(f"결과: {res}\n기록: {pr.path}" + (f"\nCAN 원본: {pr.path.replace('.csv', '.candump')}" if pr.dump else ""))
    if a.go and not a.fake:
        print("팔은 비상 정지(천천히 내려앉음) 상태입니다. 받친 채로 clear_arm_fault.py로 복구한 뒤 다음 실험을 하세요.")


if __name__ == "__main__":
    main()
