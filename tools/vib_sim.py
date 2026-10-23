#!/usr/bin/env python3
"""D2-2 사전 검증 — 2질량 관절(flex_plant) + 펌웨어 PD(1 kHz) + 우리 하이브리드 TDE(100 Hz, joint_ctrl) + τ_vib, 지연 포함, 고정 시간 간격.

τ_vib: vel = s·D·δ̇,  ppf = s·g·ω_f²·ξ (ξ̈ + 2ζ_f ω_f ξ̇ + ω_f² ξ = ω_f² δ, δ = ∫δ̇ 누설 적분),  δ̇ = HP_1.5Hz(ω_gyro − ω_enc)
링크에 5 ms 충격(툭) → 링크 각속도로 모드 주파수·ζ, 발산 여부. dist_mode accel/torque, 부호 s = ±1, 지연(센서·명령·자이로)을 바꿔 가며 비교.
사용: python3 tools/vib_sim.py calib | sweep [--extra-delay 0 10 20]
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "common_pkg"))
from flex_plant import FlexJoint  # noqa: E402
from joint_ctrl import JointCtrl, JointCtrlCfg  # noqa: E402

H = 1e-3                 # 플랜트·펌웨어 PD 주기
CTRL_DIV = 10            # 우리 루프 100 Hz
GYRO_DIV = 5             # 자이로 200 Hz
TAU_LIMIT = 4.0


class Delay:  # 1 kHz 표본의 고정 지연(ms 단위 정수)
    def __init__(self, ms, init):
        self.buf = [init] * (int(ms) + 1)

    def push(self, v):
        self.buf.append(v); self.buf.pop(0)

    def get(self):
        return self.buf[0]


class Vib:  # τ_vib 계산기 — 100 Hz
    def __init__(self, law, gain, sign, wf, zf, dt=CTRL_DIV * H, hp_hz=1.5):
        self.law, self.gain, self.sign, self.wf, self.zf, self.dt = law, gain, sign, wf, zf, dt
        self.a_hp = 1.0 / (1.0 + 2 * math.pi * hp_hz * dt); self.w_hp = 2 * math.pi * hp_hz
        self.prev_in = 0.0; self.dd = 0.0; self.delta = 0.0; self.xi = 0.0; self.xid = 0.0

    def __call__(self, d_raw):
        self.dd = self.a_hp * (self.dd + d_raw - self.prev_in); self.prev_in = d_raw  # 1차 고역통과 δ̇
        if self.law == "vel":
            return self.sign * self.gain * self.dd
        self.delta += self.dt * (self.dd - self.w_hp * self.delta)  # 누설 적분 δ
        acc = self.wf ** 2 * (self.delta - self.xi) - 2 * self.zf * self.wf * self.xid
        self.xid += self.dt * acc; self.xi += self.dt * self.xid  # 반암시 오일러
        return self.sign * self.gain * self.wf ** 2 * self.xi


def run(p, vib=None, dist_mode="accel", delays=(5, 10, 6), tap=0.05, T=4.0, kp=10.0, kd=0.8, mbar=0.01, hybrid=True):
    pl = FlexJoint(p["Jm"], p["Jl"], p["K"], p["c"], p.get("bm", 0.0), p.get("Fs", 0.0), p.get("Fc", 0.0))
    ds, dc, dg = delays  # 센서(엔코더) · 명령 · 자이로 지연[ms]
    sens = Delay(ds, (0.0, 0.0)); cmd = Delay(dc, 0.0); gyro = Delay(dg, 0.0)
    cfg = JointCtrlCfg(mbar=mbar, hybrid=hybrid, kp_fw=kp if hybrid else 0.0, kd_fw=kd if hybrid else 0.0, dist_mode=dist_mode)
    holder = {"g": 0.0, "m": (0.0, 0.0)}
    ctrl = JointCtrl(cfg, 0.0, dist_comp=(lambda o: vib(holder["g"] - holder["m"][1])) if vib else None)
    kpf, kdf = (kp, kd) if hybrid else (0.0, 0.0)
    n = int(T / H); wl = np.zeros(n); wm = np.zeros(n); tau_ff = 0.0; g_s = 0.0; first = True; taus = np.zeros(n)
    for k in range(n):
        t = k * H
        th_m, w_m, th_l, w_l = pl.x
        sens.push((th_m, w_m)); gyro.push(w_l if k % GYRO_DIV == 0 else gyro.buf[-1])
        if k % CTRL_DIV == 0:
            holder["m"] = sens.get(); holder["g"] = gyro.get()
            out = ctrl.step(holder["m"][0], holder["m"][1], 0.0, 0.0, 0.0, 0.0 if first else CTRL_DIV * H); first = False
            tau_ff = float(np.clip(out["tau"], -TAU_LIMIT, TAU_LIMIT)); ctrl.commit(tau_ff)
        cmd.push(tau_ff)
        tau_m = cmd.get() + kpf * (0.0 - th_m) + kdf * (0.0 - w_m)  # 펌웨어 PD는 최신 엔코더로 1 kHz
        ext = tap if 1.0 <= t < 1.005 else 0.0
        pl.step(tau_m, ext, H)
        wl[k], wm[k], taus[k] = pl.x[3], pl.x[1], tau_m
        if not np.all(np.isfinite(pl.x)) or abs(pl.x[2]) > 1.0:
            return dict(unstable=True, wl=wl[:k + 1], wm=wm[:k + 1])
    return dict(unstable=False, wl=wl, wm=wm, tau=taus)


def modal(w, t0=1.01, T=None):  # 충격 뒤 링크 각속도 → (주파수, ζ, 이어진 주기, 발산 여부, 모터/링크 진폭비 자리)
    x = w[int(t0 / H):]
    pk = [i for i in range(1, len(x) - 1) if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] > 0]
    pk = [i for j, i in enumerate(pk) if j == 0 or i - pk[j - 1] > 40]  # 25 Hz 이상 극대 무시
    if len(pk) < 3:
        return float("nan"), float("nan"), len(pk)
    a = np.array([x[i] for i in pk]); a0 = a.max(); i0 = int(np.argmax(a)); a = a[i0:]; pk = pk[i0:]
    keep = a >= 0.05 * a0; n = int(np.argmin(keep)) if not keep.all() else len(a)
    n = max(n, 3) if len(a) >= 3 else len(a)
    f = 1.0 / (np.mean(np.diff(pk[:n])) * H)
    dl = -np.polyfit(np.arange(n), np.log(np.maximum(a[:n], 1e-12)), 1)[0]
    return f, dl / math.sqrt(4 * math.pi ** 2 + dl ** 2), n


def ratio_motor_link(r, f):  # 모드 주파수에서 모터/링크 속도 진폭비·위상(모터가 앞서면 +)
    a = r["wl"][int(1.05 / H):int(2.5 / H)]; b = r["wm"][int(1.05 / H):int(2.5 / H)]
    tt = np.arange(len(a)) * H; e = np.exp(-2j * math.pi * f * tt)
    A = (a * e).sum(); B = (b * e).sum()
    return abs(B) / abs(A), math.degrees(np.angle(B / A))


PLANT = dict(Jm=0.004, Jl=0.011, K=21.0, c=0.004)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["calib", "sweep"])
    ap.add_argument("--Jm", type=float, default=PLANT["Jm"]); ap.add_argument("--Jl", type=float, default=PLANT["Jl"])
    ap.add_argument("--K", type=float, default=PLANT["K"]); ap.add_argument("--c", type=float, default=PLANT["c"])
    ap.add_argument("--extra-delay", type=float, nargs="+", default=[0, 10, 20], help="명령 지연에 더할 ms")
    a = ap.parse_args()
    p = dict(Jm=a.Jm, Jl=a.Jl, K=a.K, c=a.c)
    if a.mode == "calib":
        r = run(p); f, z, n = modal(r["wl"]); rm = ratio_motor_link(r, f)
        print(f"기준(하이브리드 TDE, τ_vib 없음): 링크 모드 {f:.2f} Hz, ζ {z:.3f}, {n}주기, 모터/링크 속도비 {rm[0]:.2f} 위상 {rm[1]:+.0f}°")
        return
    fn = 2 * math.pi * modal(run(p)["wl"])[0]
    for ex in a.extra_delay:
        dl = (5, int(10 + ex), 6)
        print(f"\n### 명령 지연 +{ex:.0f} ms (센서 {dl[0]}·명령 {dl[1]}·자이로 {dl[2]} ms)")
        print("| 법칙 | 모드 | 부호 | 이득 | 주파수 [Hz] | ζ | 판정 |\n|---|---|---|---|---|---|---|")
        r = run(p, delays=dl); f, z, n = modal(r["wl"]); print(f"| 없음 | — | — | 0 | {f:.2f} | {z:.3f} | {'발산' if r['unstable'] else ''} |")
        for law, gains in (("vel", [0.005, 0.01, 0.02, 0.04, 0.08]), ("ppf", [0.002, 0.005, 0.01, 0.02, 0.04])):
            for dm in ("accel", "torque"):
                for s in (+1, -1):
                    for g in gains:
                        v = Vib(law, g, s, 1.2 * fn, 0.4)
                        r = run(p, vib=v, dist_mode=dm, delays=dl)
                        f, z, n = modal(r["wl"]) if not r["unstable"] else (float("nan"), float("nan"), 0)
                        print(f"| {law} | {dm} | {s:+d} | {g} | {f:.2f} | {z:.3f} | {'발산' if r['unstable'] or (z == z and z < 0) else ''} |")


if __name__ == "__main__":
    main()
