#!/usr/bin/env python3
"""D-1 (가) 입력 성형 묶음 그림 — 시행별 정지 창 5–10 Hz 자이로 rms를 성형 조건별 점으로, 중앙값과 성형 없음 대비 비를 같이 (FW191_D_LOG 5절).

입력: vib_stop_report.py --trial-out 표(docs/data/d1a_trials.csv). 행 = 정지 쪽(자세 A 도착이 주 판정), 열 = 속도.
사용: python3 tools/vib_d1a_fig.py docs/data/d1a_trials.csv docs/img/p_fw191_d1a_shaping.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"none": "#2a78d6", "zv": "#eb6834", "zvd": "#1baf7a"}  # 범주 색 1–3번(고정 순서) — 조건은 x 위치와 눈금 이름으로도 구분됨
LAB = {"none": "no shaping", "zv": "ZV", "zvd": "ZVD"}


def main(src, out):
    d = pd.read_csv(src); speeds = sorted(d.speed_dps.unique())
    rows = [("hold", "arrive at pose A (lowering) — primary"), ("bhold", "arrive at 111.1° (raising) — secondary")]
    ymax = 1.18 * d.gyro_rms.max()
    fig, ax = plt.subplots(len(rows), len(speeds), figsize=(9.2, 7.2), sharey=True, facecolor=SURF)
    for r, (st, rname) in enumerate(rows):
        for c, v in enumerate(speeds):
            x = ax[r, c]; g = d[(d.stop == st) & (d.speed_dps == v)]; base = g[g["shape"] == "none"].gyro_rms.median()
            x.set_facecolor(SURF)
            for k, sh in enumerate(("none", "zv", "zvd")):
                y = np.sort(g[g["shape"] == sh].gyro_rms.values); med = float(np.median(y))
                x.scatter(k + np.linspace(-0.16, 0.16, len(y)), y, s=62, color=COL[sh], edgecolor=SURF, linewidth=1.6, zorder=3)
                x.plot([k - 0.3, k + 0.3], [med, med], color=INK, lw=2, solid_capstyle="round", zorder=4)
                x.text(k, y.max() + 0.035 * ymax, f"{med:.3f}" + ("" if sh == "none" else f"\n{med / base:.2f}×"), ha="center", va="bottom", fontsize=10, color=INK, linespacing=1.25)
            x.set_xticks(range(3)); x.set_xticklabels([LAB[s] for s in ("none", "zv", "zvd")], fontsize=10, color=INK2)
            x.set_xlim(-0.6, 2.6); x.set_ylim(0, ymax); x.grid(axis="y", color=GRID, lw=0.8); x.set_axisbelow(True)
            for sp in ("top", "right", "left"):
                x.spines[sp].set_visible(False)
            x.spines["bottom"].set_color(GRID); x.tick_params(axis="both", length=0, labelcolor=INK2)
            x.set_title(f"{v:g} deg/s, {rname}", fontsize=10, color=INK, loc="left")
            if c == 0:
                x.set_ylabel("5–10 Hz gyro rms, 2 s after stop [rad/s]", fontsize=9.5, color=INK2)
    fig.suptitle("Input shaping reduces residual vibration after stop (position mode, baseline comparison)", fontsize=12.5, color=INK, x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.945, "J2 10 deg constant-speed round trips, 5 trials per condition (dot = trial median of 6 stops, bar = median of trials, × = ratio to no shaping).\n"
             "Shaper fixed at 7.0 Hz, zeta 0.10. Camera-site IMU, firmware S-V1.7-3, piper_sdk 0.6.2, 2026-11-18.", fontsize=8.8, color=INK2, va="top", linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(out, dpi=140, facecolor=SURF); print("그림:", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
