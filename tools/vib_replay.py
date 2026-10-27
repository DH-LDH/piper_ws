#!/usr/bin/env python3
"""기록 재생 — 시행의 엔코더(.csv)와 실시간 IMU(.imu.csv)를 실제 도착 순서대로 VibDamp에 넣어 k별 τ_vib·감시를 본다(개루프: τ_vib는 기록된 운동에 영향 없음).

출력: k별 τ_vib 최대(클립 전)·클립 비율(전체·충격 창 밖)·감시 발동(①a·①b·②, 지금 코드 그대로)·합 포화.
충격 창 = 원시 자이로 |ω| > 0.8 rad/s 시각부터 −0.05 – +0.3 s.
사용: python3 tools/vib_replay.py logs/mit/20261026_172544_hold.csv … --k 0.5 1.0 2.0 --tmax 20(앞 20 s만, k 0 구간)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "common_pkg"))
from imu_frame import ImuToBase  # noqa: E402
from piper_kin import PiperKin  # noqa: E402
from vib_damp import VibDamp, VibDampCfg  # noqa: E402


def load(p):
    m = pd.read_csv(p); t0 = float(open(p.replace(".csv", ".t0")).read()); m = m[m.phase == "mit"]
    g = pd.read_csv(p.replace(".csv", ".imu.csv")); med = np.median(g.t_dev); g = g[np.abs(g.t_dev - med) < 1e6].copy()
    g["tg"] = g.t_dev / 1000.0 + np.quantile(g.t - g.t_dev / 1000.0, 0.01)
    gg = g[g.stream == "g"]; n = np.linalg.norm(gg[["x", "y", "z"]].values - np.median(gg[["x", "y", "z"]].values, 0), axis=1)
    shocks = []
    for tt in gg.t.values[n > 0.8] - t0:
        if not shocks or tt - shocks[-1] > 0.3:
            shocks.append(tt)
    return m, t0, g, shocks


def run(data, k, clip, notch, lead, tmax, kin, tb):
    m, t0, g, shocks = data
    vd = VibDamp(kin, tb, VibDampCfg(k=k, gate_deg=-90, gate_hi=90, tau_max=clip, notch_f=notch, lead_deg=lead))
    gv = g[["t", "x", "y", "z", "stream", "tg"]].values; gi = 0; acc = None; T = m.t.values; tm0 = T[0]
    Q = np.radians(m[[f"q{j}" for j in range(1, 7)]].values); QD = m[[f"qd{j}" for j in range(1, 7)]].values
    CH = m.chirp.values if "chirp" in m else np.zeros(len(T)); TF = m.tff2.values; raws = []; nsum = 0
    for i in range(len(T)):
        if tmax and T[i] - tm0 > tmax:
            break
        now = T[i] + t0; smp = []
        while gi < len(gv) and gv[gi][0] <= now:
            r = gv[gi]
            if r[4] == "g":
                smp.append((float(r[5]), np.array(r[1:4], float)))
            else:
                acc = np.array(r[1:4], float)
            gi += 1
        smp = [s for s in smp if vd.t_g is None or s[0] > vd.t_g]
        o = vd.step(T[i] - tm0, now, Q[i], QD[i], smp, acc); raws.append(o["raw"])
        sat = abs(CH[i] + o["tau"]) >= 0.3 or (abs(TF[i] + o["tau"]) >= 4.0 and abs(TF[i]) < 4.0); nsum += sat; vd.note_sum_sat(sat)
    raws = np.array(raws); tt = T[:len(raws)] - tm0; win = np.zeros(len(tt), bool)
    for s0 in shocks:
        win |= (tt >= s0 - tm0 - 0.05) & (tt <= s0 - tm0 + 0.3)
    cl = np.abs(raws) > clip
    return vd.tripped, vd.trip_why, np.abs(raws).max(), 100 * cl.mean(), 100 * cl[~win].mean(), nsum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--k", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--notch", type=float, default=29.6)
    ap.add_argument("--lead", type=float, default=0.0)
    ap.add_argument("--tmax", type=float, default=0.0, help="MIT 앞 이 시각까지만(0이면 전체)")
    a = ap.parse_args()
    kin = PiperKin(); tb = ImuToBase(kin)
    for p in a.csv:
        data = load(p); print(f"== {os.path.basename(p)[:15]} 충격 {len(data[3])}회" + (f", MIT {a.tmax:.0f} s까지" if a.tmax else ""))
        for k in a.k:
            tr, why, mx, cla, clo, ns = run(data, k, a.clip, a.notch, a.lead, a.tmax, kin, tb)
            print(f"  k {k} | τ 최대 {mx:.3f} 클립 전체 {cla:.1f}% 충격 창 밖 {clo:.1f}% 합 포화 {ns}주기 | 감시 {'안 걸림' if tr is None else f'{tr:.1f} s {why}'}")


if __name__ == "__main__":
    main()
