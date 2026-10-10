#!/usr/bin/env python3
"""bag 구간 → EE 위치 x, y, z(지령/실측) + 합성 |v_EE| 그림과 구간별 수치. --vel이면 축별 속도도 그린다.

사용: python3 tools/plot_bag_xyz.py logs/bags/<bag> --t 77 92 --hl 78.5 85 --after 85.5 89 --out docs/img/a_zoom_place_descend_xyz.png
좌표: /arm/ee_pose_body(body_link), 속도: 5샘플 이동평균 후 중앙차분(plot_run.deriv)
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from plot_run import read_bag, deriv
from traj_analyze import _font, hp


def stats(name, v, vn, s):  # 구간 평균과 고역 RMS(축별, 합성)
    print(name, "평균 |v| %.1f" % vn[s].mean(),
          " ".join("v%s 평균 %.1f 고역RMS %.2f" % (a, v[s, i].mean(), np.sqrt((hp(v[s, i:i + 1]) ** 2).mean()))
                   for i, a in enumerate("xyz")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--t", nargs=2, type=float, required=True)   # 그릴 구간 [s, bag 시작 기준]
    ap.add_argument("--hl", nargs=2, type=float, default=None)   # 음영 구간(목표 갱신 중)
    ap.add_argument("--after", nargs=2, type=float, default=None)  # 비교 구간(목표 멈춘 뒤)
    ap.add_argument("--vel", action="store_true")  # 축별 속도 v_x, v_y, v_z 줄 추가
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _font()
    d = read_bag(a.bag)
    t0 = min(v[0][0] for v in d.values() if v)
    te = np.array([t for t, _ in d["/arm/ee_pose_body"]]) - t0
    pe = np.array([[m.x, m.y, m.z] for _, m in d["/arm/ee_pose_body"]]) * 1000
    tc = np.array([t for t, _ in d["/arm/cartesian_target"]]) - t0
    pc = np.array([[m.x, m.y, m.z] for _, m in d["/arm/cartesian_target"]]) * 1000
    v = deriv(te, pe)
    vn = np.linalg.norm(v, axis=1)
    k = (te >= a.t[0]) & (te <= a.t[1])
    kc = (tc >= a.t[0]) & (tc <= a.t[1])
    if a.hl:
        stats("음영 구간", v, vn, (te >= a.hl[0]) & (te < a.hl[1]))
    if a.after:
        stats("비교 구간", v, vn, (te >= a.after[0]) & (te < a.after[1]))

    C = ("tab:red", "tab:green", "tab:blue")
    rows = 4 + (3 if a.vel else 0)
    fig, axs = plt.subplots(rows, 1, figsize=(11, 2.2 * rows), sharex=True)
    for i, n in enumerate("xyz"):
        axs[i].plot(tc[kc], pc[kc, i], ".", ms=2, color="tab:orange", label=f"cmd {n}")
        axs[i].plot(te[k], pe[k, i], color=C[i], lw=1, label=f"actual {n}")
        axs[i].set_ylabel(f"{n} [mm]"); axs[i].legend(fontsize=8, loc="upper right")
        if a.vel:
            axs[3 + i].plot(te[k], v[k, i], color=C[i], lw=0.8); axs[3 + i].set_ylabel(f"v_{n} = d{n}/dt\n[mm/s]")
    axs[-1].plot(te[k], vn[k], color="k", lw=0.8); axs[-1].set_ylabel("|v_EE| [mm/s]\n=√(vx²+vy²+vz²)")
    axs[-1].set_xlabel("t [s]")
    for x in axs:
        x.grid(alpha=0.3)
        if a.hl:
            x.axvspan(*a.hl, color="tab:purple", alpha=0.07, lw=0)
    fig.suptitle("EE 위치 x, y, z와 합성 속도 (body_link 좌표, 60 Hz, |v_EE|는 각 축 위치를 5샘플 이동평균 후 중앙차분해 합성)", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=110)
    print("저장:", a.out)


if __name__ == "__main__":
    main()
