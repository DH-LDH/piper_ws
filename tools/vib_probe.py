#!/usr/bin/env python3
"""VIB D0 진단 — 펌웨어 위치 모드(MIT 아님)에서 떨림이 구조 모드인지 본다. 기본은 dry-run(아무것도 보내지 않음).

stage tap : 현재 자세 유지 + 사용자가 EE를 툭 침 (조건 a 가진 ①)
stage step: --joint에 --step° 계단 → 절반 지나 원위치 (조건 a 가진 ②)
stage slow: --joints 사인파 저속 추종 (조건 a′, E5 느린 사인파 ±5° 0.25 Hz와 같은 궤적)
stage goto: --target 관절각까지 코사인 속도 프로파일로 이동(최고 --vmax °/s) — 자세 B 이동용
stage ramp: 지금 자세(멈추는 자세)에서 --joint를 −dist만큼 천천히 뺀 뒤 --speed °/s 등속으로 돌아와 정지, --hold s 유지 (단계 C·D)
기록 200 Hz(q, q̇, effort, SDK 수신 시각), JointCtrl 송신 100 Hz, --imu면 손목 D455 IMU, --imu2면 3DM-GQ7 동시 기록.
기록 옆에 .meta.json(인자, 펌웨어, SDK, 시작 자세)을 남긴다.
정상 종료는 위치 모드 유지(다음 시행 바로 가능), 이상 시 MotionCtrl_1(0x01)(천천히 내려앉음).
사용: python3 tools/vib_probe.py --stage tap --imu [--go]      (--fake: 가짜 팔로 로직 시험)
"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

from mit_probe import SETTLE_S, SETTLE_SPEED, SPEED_CONSEC, bus_type, driver_running, make_fake, read_state

LOG_ROOT = os.path.expanduser("~/piper_ws/logs/vib")
TOOLS = os.path.dirname(os.path.abspath(__file__))
STEP_MAX_DEG = 3.0
AMP_MAX_DEG = 10.0
GOTO_MAX_DEG = 40.0       # goto 한 번에 관절당 최대 변화
RAMP_MAX_DEG = 15.0       # ramp 이동 거리 상한
RAMP_VMAX = 20.0          # ramp 등속 상한[deg/s] — 기존 실기 최고는 사인 0.5 Hz의 15.7


def stamps(p):  # SDK가 관절·고속 피드백을 받은 시각 — 200 Hz 로그의 중복 행 구분용
    return getattr(p.GetArmJointMsgs(), "time_stamp", 0.0), getattr(p.GetArmHighSpdInfoMsgs(), "time_stamp", 0.0)


def fake_follow(p):  # 가짜 팔이 JointCtrl 목표를 1차 지연(τ 50 ms)으로 따르게 — 원래 가짜 팔은 위치 모드 목표를 무시
    goal = {"q": None}
    p.JointCtrl = lambda *c: goal.__setitem__("q", np.radians(np.array(c) / 1000.0))
    step = p._step

    def _step():
        h = min(time.time() - p.t, 0.02)
        step()
        if goal["q"] is not None and not p.mit:
            dq = (goal["q"] - p.q) * min(1.0, h / 0.05)
            p.qd = dq / max(h, 1e-3); p.q = p.q + dq
    p._step = _step
    return p


def fake_imu_file(path, f0=7.0, zeta=0.05, tilt=33.0):  # 가짜 팔용 IMU 파일 — 관절 가속이 7 Hz 모드를 흔드는 흉내(분석 도구 시험용, 모델 아님)
    import pandas as pd
    d = pd.read_csv(path); t0 = float(open(path.replace(".csv", ".t0")).read())
    j = int(np.argmax([np.ptp(d[f"qref{k}"].values) for k in range(1, 7)])) + 1
    h = 1 / 400.0; t = np.arange(d.t.values[0], d.t.values[-1], h); q = np.radians(np.interp(t, d.t.values, d[f"q{j}"].values))
    from scipy import signal
    q = signal.sosfiltfilt(signal.butter(2, 15, fs=400, output="sos"), q); v = np.gradient(q, h); acc = np.gradient(v, h)
    w = 2 * math.pi * f0  # η̈ + 2ζω η̇ + ω² η = −q̈ (받침 가진), 출력 η̇
    ed = signal.lsim(([-1, 0], [1, 2 * zeta * w, w * w]), acc, t - t[0])[1]
    rng = np.random.default_rng(0); edd = np.gradient(ed, h); g = 9.81
    with open(path.replace(".csv", ".imu.csv"), "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["t", "t_dev", "stream", "x", "y", "z"])
        for i in range(len(t)):
            te = t[i] + t0
            wr.writerow((f"{te:.6f}", f"{t[i] * 1000:.3f}", "g", f"{v[i] + ed[i] + rng.normal(0, 2e-3):.5f}", f"{rng.normal(0, 2e-3):.5f}", f"{rng.normal(0, 2e-3):.5f}"))
            if i % 2 == 0:
                wr.writerow((f"{te:.6f}", f"{t[i] * 1000:.3f}", "a", f"{rng.normal(0, 0.02):.5f}", f"{g * math.sin(math.radians(tilt)) + rng.normal(0, 0.02):.5f}",
                             f"{-g * math.cos(math.radians(tilt)) + 0.37 * edd[i] + rng.normal(0, 0.02):.5f}"))


class Vib:
    def __init__(self, p, a):
        self.p, self.a = p, a
        os.makedirs(LOG_ROOT, exist_ok=True)
        tag = time.strftime("%Y%m%d_%H%M%S") + f"_{a.stage}" + (f"_j{a.joint}" if a.stage == "step" else "") \
            + (f"_j{''.join(map(str, a.joints))}" if a.stage == "slow" else "") \
            + (f"_j{a.joint}_v{a.speed:g}_d{a.dist:+g}" if a.stage == "ramp" else "") + (f"_{a.note}" if a.note else "") \
            + ("_fake" if a.fake else "") + ("" if a.go else "_dry")
        self.path = os.path.join(LOG_ROOT, tag + ".csv")
        self.f = open(self.path, "w", newline="")
        self.w = csv.writer(self.f)
        J = range(1, 7)
        self.w.writerow(["t", "phase", "ts_j", "ts_h"] + [f"q{j}" for j in J] + [f"qd{j}" for j in J]
                        + [f"eff{j}" for j in J] + [f"qref{j}" for j in J] + ["arm_status", "ctrl_mode", "mode_feed", "enabled"])
        self.t0 = time.time()
        with open(self.path.replace(".csv", ".t0"), "w") as m:  # 모든 기록의 epoch 기준 — CSV t + t0 = candump·IMU 시각
            m.write(f"{self.t0:.6f}\n")
        self.dump = self.imu = self.imu2 = None
        if a.imu2 and not a.fake:
            self.imu2 = subprocess.Popen([sys.executable, os.path.join(TOOLS, "gq7_rec.py"), self.path.replace(".csv", ".gq7.csv")])
            time.sleep(1.5)
            if self.imu2.poll() is not None:
                raise SystemExit("★ GQ7 기록이 시작되지 않음 — --imu2 없이 하거나 USB 확인")
        if a.imu and not a.fake:  # candump보다 먼저 — IMU 실패로 끝날 때 candump가 남지 않게
            self.imu = subprocess.Popen([sys.executable, os.path.join(TOOLS, "imu_rec.py"), self.path.replace(".csv", ".imu.csv")])
            time.sleep(2.0)  # 스트림 시작 대기
            if self.imu.poll() is not None:
                if self.imu2:
                    self.imu2.terminate()
                raise SystemExit("★ IMU 기록이 시작되지 않음 — --imu 없이 하거나 손목캠 확인")
        if a.go and not a.fake:
            self.dump = subprocess.Popen(["candump", "-L", a.can], stdout=open(self.path.replace(".csv", ".candump"), "w"),
                                         stderr=subprocess.DEVNULL)
        self.fast_n = 0; self.segs = []

    def meta(self, **kw):  # 기록마다 조건·버전을 남김(FW191_PLAN 0절)
        a = self.a
        if not hasattr(self, "_meta"):
            self._meta = {"stage": a.stage, "fw": a.fw, "sdk": a.sdk, "go": a.go, "fake": a.fake, "note": a.note, "spd": a.spd, "t0": self.t0,
                          "args": {k: v for k, v in vars(a).items() if k not in ("fw", "sdk")}}
        m = self._meta; m.update(kw)
        with open(self.path.replace(".csv", ".meta.json"), "w") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)

    def ramp_plan(self, q0):  # [(이름, 시간, 시작각, 끝각, 모양)] — 관절 a.joint만, 각도 rad
        a = self.a; j = a.joint - 1; A = q0[j]; S = A - math.radians(a.dist); d = abs(a.dist)
        Tc = max(0.5, math.pi / 2 * d / a.pre_vmax); Tl = d / a.speed
        segs = [("lead", a.lead, A, A, "hold"), ("pre", Tc, A, S, "cos"), ("wait", a.pre_hold, S, S, "hold")]
        for r in range(a.reps):
            segs += [("ramp", Tl, S, A, "lin"), ("hold", a.hold, A, A, "hold")]
            if a.round:
                segs += [("back", Tl, A, S, "lin"), ("bhold", a.hold, S, S, "hold")]
            elif r < a.reps - 1:
                segs += [("pre", Tc, A, S, "cos"), ("wait", a.pre_hold, S, S, "hold")]
        if a.round:
            segs.append(("post", Tc, S, A, "cos"))
        return segs

    def ramp_at(self, tt):  # → (구간 이름, 관절각, 관절 속도)
        for name, T, qa, qb, kind in self.segs:
            if tt < T:
                u = tt / T if T > 0 else 1.0
                if kind == "lin":
                    return name, qa + (qb - qa) * u, (qb - qa) / T
                if kind == "cos":
                    return name, qa + (qb - qa) * (1 - math.cos(math.pi * u)) / 2, (qb - qa) * math.pi / (2 * T) * math.sin(math.pi * u)
                return name, qa, 0.0
            tt -= T
        return self.segs[-1][0], self.segs[-1][3], 0.0

    def log(self, phase, s, qref):
        q, qd, eff, en, st = s
        tj, th = stamps(self.p)
        self.w.writerow([f"{time.time() - self.t0:.4f}", phase, f"{tj:.4f}", f"{th:.4f}"] + list(np.degrees(q).round(4))
                        + list(qd.round(4)) + list(eff.round(4)) + list(np.degrees(qref).round(4)) + list(st) + [int(en.all())])

    def check(self, s, qref, vref):  # 중단 사유 문자열 또는 None
        q, qd, eff, en, st = s
        dev = np.degrees(np.abs(q - qref))
        if dev.max() > self.a.dev_limit:
            return f"기준 이탈 J{int(dev.argmax()) + 1} {dev.max():.2f}° > {self.a.dev_limit}°"
        v = np.abs(qd - vref)
        self.fast_n = self.fast_n + 1 if v.max() > self.a.speed_limit else 0
        if self.fast_n >= self.a.speed_consec:
            return f"관절 속도 J{int(v.argmax()) + 1} {v.max():.2f} rad/s > {self.a.speed_limit} ({self.a.speed_consec}주기 연속)"
        if not en.all():
            return f"모터 비활성 J{[i + 1 for i in range(6) if not en[i]]}"
        if st[0] != 0 or st[1] != 1:
            return f"상태 이상 arm_status={st[0]:#x} ctrl_mode={st[1]}"
        return None

    def ref(self, q0, tt):  # 시각 tt의 (q_ref, q̇_ref)
        a = self.a; qr = q0.copy(); vr = np.zeros(6)
        if a.stage == "step":
            j = a.joint - 1; t1 = a.lead; t2 = a.lead + (a.sec - a.lead) / 2
            if t1 <= tt < t2:
                qr[j] += math.radians(a.step)
        elif a.stage == "slow" and tt > a.lead:  # mit_probe sine과 같은 모양: 진폭 2 s 램프
            ts = tt - a.lead
            if ts < a.cycles / a.hz:
                ramp = min(1.0, ts / 2.0); w = 2 * math.pi * a.hz; A = math.radians(a.amp) * ramp
                for j in a.joints:
                    qr[j - 1] = q0[j - 1] + A * math.sin(w * ts); vr[j - 1] = A * w * math.cos(w * ts)
        elif a.stage == "goto" and tt > a.lead:  # q0 + Δ·(1 − cos πu)/2, u = (t − lead)/T — 시작·끝 속도 0
            dq = np.radians(a.target) - q0; T = self.goto_T(q0); u = min(1.0, (tt - a.lead) / T)
            qr = q0 + dq * (1 - math.cos(math.pi * u)) / 2
            vr = dq * math.pi / (2 * T) * math.sin(math.pi * u) if u < 1.0 else np.zeros(6)
        elif a.stage == "ramp":
            _, qr[a.joint - 1], vr[a.joint - 1] = self.ramp_at(tt)
        return qr, vr

    def goto_T(self, q0):  # 최고 관절 속도가 vmax가 되는 이동 시간
        return max(0.5, math.pi / 2 * np.degrees(np.abs(np.radians(self.a.target) - q0)).max() / self.a.vmax)

    def duration(self):
        a = self.a
        if a.stage == "goto":
            return a.lead + self.goto_T(self.q0) + a.tail
        if a.stage == "ramp":
            return sum(sg[1] for sg in self.segs)
        return a.lead + a.cycles / a.hz + a.tail if a.stage == "slow" else a.sec

    def run(self):
        a, p = self.a, self.p
        s = read_state(p)
        q, qd, eff, en, st = s
        hj = getattr(p.GetArmJointMsgs(), "Hz", float("nan")); hh = getattr(p.GetArmHighSpdInfoMsgs(), "Hz", float("nan"))
        print(f"현재 q[deg] {np.degrees(q).round(1)} | enable {en.astype(int)} | status {st} | 피드백 관절 {hj:.0f} Hz, 고속 {hh:.0f} Hz")
        if a.resume:  # 10-26 실기: 대기에서 0x02 → 6축 Enable 해제, 팔 낙하(J2 121→167°) — 다시 쓰지 않음
            raise SystemExit("★ --resume 사용 금지: 대기 모드에서 0x02는 모터를 해제해 팔이 떨어짐(VIB_E4 5절)")
        if False:  # 8절 10/26 저녁 11번: 대기(자세 유지)에서 0x02 복구 1회 → 아래 MotionCtrl_2
            print("대기 모드 → 0x02(복구) 1회 송신, 1 s 대기")
            p.EmergencyStop(0x02); time.sleep(1.0)
            s = read_state(p); q, qd, eff, en, st = s
            print(f"0x02 뒤 q[deg] {np.degrees(q).round(1)} | enable {en.astype(int)} | status {st}")
            if not en.all() or st[0] != 0:
                p.MotionCtrl_1(0x01, 0, 0)
                raise SystemExit("★ 0x02 뒤 Enable·상태 이상 — 비상 정지 송신")
        elif not en.all() or st[0] != 0 or st[1] != 1:  # 대기(0)는 MotionCtrl_2만으로 CAN 제어에 안 돌아옴(10-26 실기) → 복구 절차 필요
            raise SystemExit("★ 팔이 위치 모드(CAN 제어)로 Enable된 상태가 아님 — clear_arm_fault → traj_test로 자세를 잡은 뒤 실행")
        q0 = self.q0 = q.copy()
        self.meta(q0_deg=[round(float(v), 3) for v in np.degrees(q0)])
        if a.expect is not None:  # 멈추는 자세가 시행마다 같은지
            e = np.abs(np.degrees(q0) - np.array(a.expect))
            if e.max() > a.expect_tol:
                raise SystemExit(f"★ 시작 자세가 기대와 다름: J{int(e.argmax()) + 1} {e.max():.2f}° > {a.expect_tol}° — goto로 자세를 맞춘 뒤 실행")
        if a.stage == "ramp":
            self.segs = self.ramp_plan(q0); j = a.joint; A = math.degrees(q0[j - 1])
            print(f"J{j} 등속 후 정지: 멈추는 각 {A:.2f}°, 출발 각 {A - a.dist:.2f}°, {a.speed:g}°/s × {abs(a.dist):g}° ({abs(a.dist) / a.speed:.2f} s), "
                  f"정지 후 {a.hold:g} s, {a.reps}회{' 왕복' if a.round else ''} → 총 {self.duration():.1f} s")
            print("  구간: " + " → ".join(f"{n} {T:.1f}s" for n, T, *_ in self.segs))
            try:  # 끝단 이동 거리와 올림·내림(FK) — 표에 적을 값, 실패해도 시행에는 영향 없음
                sys.path.insert(0, os.path.join(TOOLS, "..", "src", "common_pkg"))
                from piper_kin import PiperKin
                kin = PiperKin(); qs = q0.copy(); qs[j - 1] -= math.radians(a.dist); d = kin.fk(q0)[0] - kin.fk(qs)[0]
                print(f"  끝단 이동 {np.linalg.norm(d) * 100:.1f} cm, 높이 {d[2] * 100:+.1f} cm ({'올림' if d[2] > 0.002 else '내림' if d[2] < -0.002 else '수평'})")
                self.meta(ee_move_cm=round(float(np.linalg.norm(d)) * 100, 2), ee_dz_cm=round(float(d[2]) * 100, 2))
            except Exception as e:
                print(f"  끝단 이동 거리 계산 생략({type(e).__name__}) — source install/setup.bash 뒤에 실행하면 나옴")
        if a.stage == "goto":
            d = np.array(a.target) - np.degrees(q0)
            if np.abs(d).max() > GOTO_MAX_DEG:
                raise SystemExit(f"★ goto 변화 J{int(np.abs(d).argmax()) + 1} {d[np.abs(d).argmax()]:+.1f}° > {GOTO_MAX_DEG}° — 나눠서 이동")
            print(f"이동 목표 q[deg] {np.round(a.target, 1)}\n변화 {d.round(1)} | 최고 {a.vmax}°/s, {self.goto_T(q0):.1f} s")
        elif a.stage == "step":
            print(f"J{a.joint} 계단 {a.step:+.1f}°: {a.lead:.0f} s에 올리고 {a.lead + (a.sec - a.lead) / 2:.1f} s에 원위치, 총 {a.sec:.0f} s")
        elif a.stage == "slow":
            print(f"J{a.joints} 사인파 ±{a.amp}° {a.hz} Hz {a.cycles}주기 (램프 2 s), 앞 {a.lead} s·뒤 {a.tail} s 정지 → 총 {self.duration():.1f} s")
            for j in a.joints:
                print(f"  J{j}: {math.degrees(q0[j - 1]) - a.amp:.1f} – {math.degrees(q0[j - 1]) + a.amp:.1f}°")
        elif a.stage == "tap":
            print(f"자세 유지 {a.sec:.0f} s — 시작 {a.lead:.0f} s 뒤 신호음이 나면 EE를 가볍게 한 번 툭 (방향: {a.note or '기록 안 함'})")
        print(f"위치 모드 속도 {a.spd}% | 이탈 한계 {a.dev_limit}° | 속도 한계 |q̇ − q̇_ref| {a.speed_limit} rad/s × {a.speed_consec}주기")
        if not a.go:
            print("[dry-run] 명령은 하나도 보내지 않음. 실제 실행은 --go")
            return "dry-run"
        p.MotionCtrl_2(0x01, 0x01, a.spd, 0x00)
        dt = 1.0 / a.rate; t_end = time.time() + SETTLE_S; slow_n = 0  # 1) 위치 모드 정지 확인
        while time.time() < t_end:
            p.JointCtrl(*[round(v * 1000) for v in np.degrees(q0)])
            s = read_state(p); self.log("settle", s, q0)
            slow_n = slow_n + 1 if np.abs(s[1]).max() > SETTLE_SPEED else 0
            if np.degrees(np.abs(s[0] - q0)).max() > 0.5 or slow_n >= SPEED_CONSEC:
                raise SystemExit(f"★ 위치 모드에서 정지 상태가 아님(이탈 {np.degrees(np.abs(s[0] - q0)).max():.2f}°, "
                                 f"속도 J{int(np.abs(s[1]).argmax()) + 1} {np.abs(s[1]).max():.2f} rad/s) — 중단, 다시 실행하면 됨")
            time.sleep(dt)
        t_start = time.time(); dur = self.duration(); k = 0; beeped = False; reason = None  # 2) 본 시행
        while True:
            now = time.time(); tt = now - t_start
            if tt > dur:
                break
            qr, vr = self.ref(q0, tt)
            if k % max(1, round(a.rate / a.send_rate)) == 0:
                p.JointCtrl(*[round(v * 1000) for v in np.degrees(qr)])
            s = read_state(p); self.log(self.ramp_at(tt)[0] if a.stage == "ramp" else a.stage, s, qr)
            if a.stage == "tap" and not beeped and tt >= a.lead:
                print("\a▶ 지금 툭", flush=True); beeped = True
            reason = self.check(s, qr, vr)
            if reason:
                break
            k += 1
            time.sleep(max(0.0, dt - (time.time() - now)))
        if reason:
            p.MotionCtrl_1(0x01, 0, 0)
            for _ in range(int(a.rate)):
                self.log("estop", read_state(p), q0); time.sleep(dt)
            return f"중단: {reason} — 비상 정지 송신"
        q_end = self.ref(q0, dur)[0]  # goto는 목표 자세, 나머지는 시작 자세
        for _ in range(int(a.rate)):  # 3) 끝난 뒤 1 s 더 기록(위치 모드 유지)
            p.JointCtrl(*[round(v * 1000) for v in np.degrees(q_end)])
            self.log("after", read_state(p), q_end); time.sleep(dt)
        return "정상 종료 — 위치 모드로 " + ("목표" if a.stage == "goto" else "시작") + " 자세 유지 중"


def print_end_tilt(path):  # 끝 정지 1.5 s의 IMU 기울기 — 자세 A 진입 방향별 처짐 상태 비교(8절 10/26 붙이기 절차)
    try:
        from imu_frame import static_tilt
        with open(path) as f:
            t = [float(r["t"]) for r in csv.DictReader(f)]
        print(f"링크 정지 기울기(끝 1.5 s, IMU atan2(y,−z), 33° 이상이 E3 '켜진' 쪽): {static_tilt(path, t[-1] - 1.5, t[-1]):.2f}°")
    except Exception as e:  # 참고값 — 실패해도 실험에는 영향 없음
        print(f"링크 정지 기울기 계산 실패: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tap", "step", "slow", "goto", "ramp"], required=True)
    ap.add_argument("--go", action="store_true", help="실제로 명령 송신(없으면 dry-run)")
    ap.add_argument("--fake", action="store_true", help="가짜 팔로 로직 시험(CAN 미사용)")
    ap.add_argument("--imu", action="store_true", help="손목 D455 IMU 동시 기록(tools/imu_rec.py)")
    ap.add_argument("--imu2", action="store_true", help="3DM-GQ7 동시 기록(tools/gq7_rec.py → .gq7.csv)")
    ap.add_argument("--note", default="", help="파일명에 붙일 메모 — 예: poseA_x (자세·툭 방향)")
    ap.add_argument("--sec", type=float, default=12.0, help="tap·step 총 시간[s]")
    ap.add_argument("--lead", type=float, default=2.0, help="가진·궤적 시작 전 정지[s]")
    ap.add_argument("--tail", type=float, default=3.0, help="slow 끝난 뒤 정지[s]")
    ap.add_argument("--joint", type=int, default=1, help="step·ramp 관절")
    ap.add_argument("--speed", type=float, default=5.0, help="ramp 등속[deg/s]")
    ap.add_argument("--dist", type=float, default=5.0, help="ramp 이동 거리[deg], 부호 = 멈추는 자세로 들어오는 방향(+면 관절각이 커지며 도착, J2는 +가 내림)")
    ap.add_argument("--hold", type=float, default=3.0, help="ramp 정지 후 유지[s]")
    ap.add_argument("--reps", type=int, default=1, help="ramp 반복 수")
    ap.add_argument("--round", action="store_true", help="ramp 왕복: 도착 → 유지 → 같은 속도로 출발점 → 유지 (단계 D 시나리오)")
    ap.add_argument("--pre-vmax", type=float, default=5.0, help="ramp 출발점까지 빼는 최고 속도[deg/s] (코사인)")
    ap.add_argument("--pre-hold", type=float, default=3.0, help="ramp 출발점에서 기다리는 시간[s]")
    ap.add_argument("--expect", type=float, nargs=6, default=None, help="기대 시작 자세[deg] — 벗어나면 시작하지 않음")
    ap.add_argument("--expect-tol", type=float, default=0.5, help="--expect 허용 오차[deg]")
    ap.add_argument("--step", type=float, default=2.0, help="계단 크기[deg]")
    ap.add_argument("--joints", type=int, nargs="+", default=[1], help="slow 관절들")
    ap.add_argument("--amp", type=float, default=5.0, help="slow 진폭[deg]")
    ap.add_argument("--hz", type=float, default=0.25)
    ap.add_argument("--cycles", type=float, default=3.0)
    ap.add_argument("--target", type=float, nargs=6, default=None, help="goto 목표 관절각[deg]")
    ap.add_argument("--vmax", type=float, default=5.0, help="goto 최고 관절 속도[deg/s]")
    ap.add_argument("--spd", type=int, default=10, help="위치 모드 속도 비율[%%] (mit_probe 정지 확인과 같은 10)")
    ap.add_argument("--dev-limit", type=float, default=3.0, help="기준 이탈 중단 한계[deg]")
    ap.add_argument("--speed-limit", type=float, default=0.5, help="|q̇ − q̇_ref| 중단 한계[rad/s]")
    ap.add_argument("--speed-consec", type=int, default=6, help="속도 한계 연속 주기(200 Hz에서 30 ms) — 툭 친 순간은 통과")
    ap.add_argument("--rate", type=float, default=200.0, help="기록 주기[Hz]")
    ap.add_argument("--send-rate", type=float, default=100.0, help="JointCtrl 송신 주기[Hz]")
    ap.add_argument("--can", default="can_piper")
    ap.add_argument("--resume", action="store_true", help="사용 금지 — 10-26 대기 모드에서 0x02가 모터를 해제해 팔이 떨어짐")
    a = ap.parse_args()
    if a.stage == "goto" and (a.target is None or not 0 < a.vmax <= 10):
        raise SystemExit("goto는 --target 6개 필요, --vmax는 10°/s 이하")
    if abs(a.step) > STEP_MAX_DEG or a.amp > AMP_MAX_DEG:
        raise SystemExit(f"계단은 {STEP_MAX_DEG}°, 진폭은 {AMP_MAX_DEG}° 이하로")
    if a.stage == "ramp" and not (0 < abs(a.dist) <= RAMP_MAX_DEG and 0 < a.speed <= RAMP_VMAX and 0 < a.pre_vmax <= 10 and a.reps >= 1):
        raise SystemExit(f"ramp는 거리 {RAMP_MAX_DEG}° 이하, 속도 {RAMP_VMAX}°/s 이하, --pre-vmax 10°/s 이하")
    a.fw, a.sdk = "fake", "fake"
    if a.fake:
        p = fake_follow(make_fake())
    else:
        if driver_running():
            raise SystemExit("★ piper_driver_node가 실행 중 — CAN 명령이 섞이지 않게 먼저 종료할 것")
        import can
        def _ts_bus(*args, bustype=None, **kw):  # mit_probe와 같은 ThreadSafeBus + loopback(송신 프레임 candump 기록)
            kw.setdefault("interface", bustype)
            kw["local_loopback"] = True
            return can.ThreadSafeBus(*args, **kw)
        can.interface.Bus = _ts_bus
        from piper_sdk import C_PiperInterface_V2
        p = C_PiperInterface_V2(a.can); p.ConnectPort(); time.sleep(0.5)
        print(f"CAN 버스: {bus_type(p)}")
        p.SearchPiperFirmwareVersion(); time.sleep(0.3)
        a.fw, a.sdk = str(p.GetPiperFirmwareVersion()), __import__("importlib.metadata").metadata.version("piper_sdk")
        print(f"펌웨어: {a.fw} | piper_sdk {a.sdk}")
    v = Vib(p, a)
    try:
        res = v.run()
    except KeyboardInterrupt:
        if a.go:
            p.MotionCtrl_1(0x01, 0, 0)
        res = "사용자 중단(Ctrl-C) — 비상 정지 송신"
    finally:
        v.f.close()
        for sp in (v.dump, v.imu, v.imu2):
            if sp:
                sp.terminate()
                try:
                    sp.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    sp.kill()
    v.meta(result=res, **({"ramp_segs": [[n, round(T, 3)] for n, T, *_ in v.segs]} if v.segs else {}))
    if a.fake and a.imu and a.go:
        fake_imu_file(v.path); v.imu = True
    print(f"결과: {res}\n기록: {v.path}" + (f"\nIMU: {v.path.replace('.csv', '.imu.csv')}" if v.imu else "")
          + (f"\nGQ7: {v.path.replace('.csv', '.gq7.csv')}" if v.imu2 else "")
          + (f"\nCAN 원본: {v.path.replace('.csv', '.candump')}" if v.dump else ""))
    if v.imu and a.go:
        print_end_tilt(v.path)


if __name__ == "__main__":
    main()
