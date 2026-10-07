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
import sys
import time

import numpy as np

DEV_LIMIT_DEG = 3.0       # 기준 대비 이탈 한계
SPEED_LIMIT = 0.5         # [rad/s] 관절 속도 한계(10-06 0xAD 튐은 3~9 rad/s)
SPEED_CONSEC = 3          # 속도 한계는 연속 3주기 초과 시만 — 정지 중 속도 피드백 잡음 최대 0.1 rad/s
SETTLE_SPEED = 0.2        # [rad/s] 위치 모드 정지 판정 속도(잡음 최대의 2배)
TAU_FF_LIMIT = 3.0        # [N·m] τ_ff 상한(시작 자세 유지 토크 최대 1.6)
TAU_CMD_LIMIT = 4.0       # [N·m] --ext-kp 사용 시 중력+우리 P 합의 상한
SETTLE_S = 2.0            # 위치 모드 정지 확인 시간
TORQUE_AVG_S = 0.5        # 유지 토크 평균 시간
MODE_GRACE_S = 0.2        # MIT 전환 후 모드 피드백이 MOVE M(4)이 될 때까지 유예 — 이후 벗어나면 중단
# MOVE M 명령은 진입 때 1회만: 10-06 1초 주기 재전송 직후 J5·J6가 0° 쪽으로 튐(목표 초기화 추정)
LOG_ROOT = os.path.expanduser("~/IsaacSim_ranger_piper/logs/mit")


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
        os.makedirs(LOG_ROOT, exist_ok=True)
        tag = time.strftime("%Y%m%d_%H%M%S") + f"_{a.stage}" + ("_fake" if a.fake else "") + ("" if a.go else "_dry")
        self.path = os.path.join(LOG_ROOT, tag + ".csv")
        self.f = open(self.path, "w", newline="")
        self.w = csv.writer(self.f)
        J = range(1, 7)
        self.w.writerow(["t", "phase"] + [f"q{j}" for j in J] + [f"qd{j}" for j in J] + [f"eff{j}" for j in J]
                        + [f"qref{j}" for j in J] + [f"tff{j}" for j in J] + ["arm_status", "ctrl_mode", "mode_feed", "enabled"])
        self.t0 = time.time()
        self.fast_n = 0
        self.spike_n = 0; self.spikes = 0

    def log(self, phase, s, qref, tff):
        q, qd, eff, en, st = s
        self.w.writerow([f"{time.time() - self.t0:.4f}", phase] + list(np.degrees(q).round(4)) + list(qd.round(4))
                        + list(eff.round(4)) + list(np.degrees(qref).round(4)) + list(np.round(tff, 4))
                        + list(st) + [int(en.all())])

    def check(self, s, qref, phase, t_mit=None):  # 중단 사유 문자열 또는 None
        q, qd, eff, en, st = s
        if t_mit is not None and t_mit > MODE_GRACE_S and st[2] != 0x04:
            return f"모드 피드백 {st[2]} (MOVE M=4 아님)"
        dev = np.degrees(np.abs(q - qref))
        if dev.max() > DEV_LIMIT_DEG:
            return f"기준 이탈 J{int(dev.argmax()) + 1} {dev.max():.2f}° > {DEV_LIMIT_DEG}°"
        self.fast_n = self.fast_n + 1 if np.abs(qd).max() > SPEED_LIMIT else 0
        if self.fast_n >= SPEED_CONSEC:
            return f"관절 속도 J{int(np.abs(qd).argmax()) + 1} {np.abs(qd).max():.2f} rad/s > {SPEED_LIMIT} ({SPEED_CONSEC}주기 연속)"
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
        while time.time() < t_end:
            if a.go:
                p.JointCtrl(*[round(v * 1000) for v in np.degrees(q_ref)])
            s = read_state(p)
            self.log("settle", s, q_ref, np.zeros(6))
            if np.degrees(np.abs(s[0] - q_ref)).max() > 0.5 or np.abs(s[1]).max() > SETTLE_SPEED:
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
              + (f"\nKp(우리 루프, τ_ff로) {self.ext_kp} — 상한 ±{TAU_CMD_LIMIT} N·m" if self.ext_kp is not None else ""))
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
        while True:
            now = time.time(); tt = now - t_start
            if tt > dur:
                break
            qr = q_hold.copy(); vr = np.zeros(6)
            if a.stage == "sine" and tt > hold_s:  # 진폭을 2 s 동안 0→A로 올림(급출발 방지)
                ts = tt - hold_s; ramp = min(1.0, ts / 2.0)
                w = 2 * math.pi * a.hz; A = math.radians(a.amp) * ramp
                qr[j6] = q_hold[j6] + A * math.sin(w * ts)
                vr[j6] = A * w * math.cos(w * ts)
            s = read_state(p)  # 읽기 → 계산 → 송신 (우리 P가 최신 위치를 쓰도록)
            tau = tff if self.ext_kp is None else np.clip(tff + self.ext_kp * (qr - s[0]), -TAU_CMD_LIMIT, TAU_CMD_LIMIT)
            for j in range(6):
                p.JointMitCtrl(j + 1, float(qr[j]), float(vr[j]), float(self.kp[j]), float(self.kd[j]), float(tau[j]))
            self.log("mit", s, qr, tau)
            if np.abs(s[1]).max() > SPEED_LIMIT and self.fast_n == 0:
                self.spike_n += 1
            reason = self.check(s, qr, "mit", tt)
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


def make_fake(kick=False, modedrop=False, refzero=False):  # 가짜 팔 — 위치 모드는 목표 유지, MIT는 τ=PD+ff로 단순 2차 동역학(중력 = 시작 유지 토크)
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
    ap.add_argument("--fake-refzero", action="store_true", help="가짜 팔이 가끔 J3 목표 위치를 0으로 처리(실기 튐 흉내)")
    ap.add_argument("--kp", type=float, nargs=6, default=[10, 10, 10, 10, 10, 10])
    ap.add_argument("--kd", type=float, nargs=6, default=[0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
    ap.add_argument("--ext-kp", type=float, nargs=6, default=None,
                    help="펌웨어 Kp=0으로 두고 우리 루프가 τ_ff = 중력 + ext_kp·(q_ref−q)로 위치를 잡음")
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
        p = make_fake(kick=a.fake_kick, modedrop=a.fake_modedrop, refzero=a.fake_refzero)
    else:
        if driver_running():
            raise SystemExit("★ piper_driver_node가 실행 중 — CAN 명령이 섞이지 않게 먼저 종료할 것")
        import can
        def _ts_bus(*args, bustype=None, **kw):  # SDK가 옛 인자명 bustype을 넘김 → interface로 바꿔 ThreadSafeBus 생성
            kw.setdefault("interface", bustype)
            return can.ThreadSafeBus(*args, **kw)
        can.interface.Bus = _ts_bus  # SDK 수신 스레드와 우리 송신이 같은 버스를 잠금 없이 공유 → MIT 순간 튐(커뮤니티 포크 수정과 동일)
        from piper_sdk import C_PiperInterface_V2
        p = C_PiperInterface_V2(a.can); p.ConnectPort(); time.sleep(0.5)
        print(f"CAN 버스: {bus_type(p)}")
    pr = Probe(p, a)
    try:
        res = pr.run()
    except KeyboardInterrupt:
        if a.go:
            p.MotionCtrl_1(0x01, 0, 0)
        res = "사용자 중단(Ctrl-C) — 비상 정지 송신"
    finally:
        pr.f.close()
    print(f"결과: {res}\n기록: {pr.path}")
    if a.go and not a.fake:
        print("팔은 비상 정지(천천히 내려앉음) 상태입니다. 받친 채로 clear_arm_fault.py로 복구한 뒤 다음 실험을 하세요.")


if __name__ == "__main__":
    main()
