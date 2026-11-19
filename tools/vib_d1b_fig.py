#!/usr/bin/env python3
"""D-1 MIT 묶음 그림 — 시행별 정지 창 5–10 Hz 자이로 rms(튐이 겹친 창 제외)를 제어 조건별 점으로, 중앙값과 위치 모드 대비 비를 같이 (FW191_D_LOG 10절).

입력: vib_stop_report.py --trial-out 표(docs/data/d1b_trials.csv)와 시행 목록(재측정 표시용). 왼쪽 = 자세 A 도착(주 판정), 오른쪽 = 111.1° 도착(보조).
점: 채운 점 = 원래 시행, 빈 점 = 재측정. 굵은 선과 숫자 = 모든 시행의 중앙값과 위치 모드 대비 비.
사용: python3 tools/vib_d1b_fig.py docs/data/d1b_trials.csv docs/data/trials_d1b5_v10.csv docs/img/p_fw191_d1b_mit.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ORDER = ["pos", "pd", "tde", "asmc16", "asmc16_j123"]
COL = {"pos": "#2a78d6", "pd": "#eb6834", "tde": "#1baf7a", "asmc16": "#8a5cd6", "asmc16_j123": "#c9930e"}
LAB = {"pos": "position\nmode", "pd": "MIT\nfirmware PD", "tde": "MIT + TDE\n(J2)", "asmc16": "MIT + TDE\n+ ASMC (J2)", "asmc16_j123": "MIT + TDE\n+ ASMC (J1–3)"}


def main(src, plan, out):
    d = pd.read_csv(src).merge(pd.read_csv(plan)[["file", "retake_of"]], on="file"); d["re"] = d.retake_of.notna()
    cols = [("hold", "arrive at pose A (lowering) — primary"), ("bhold", "arrive at 111.1° (raising) — secondary")]
    ymax = 1.22 * d.gyro_rms.max()
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.6), sharey=True, facecolor=SURF)
    for x, (st, name) in zip(ax, cols):
        g = d[d.stop == st]; base = g[g.cond == "pos"].gyro_rms.median(); x.set_facecolor(SURF)
        for k, c in enumerate(ORDER):
            m = g[g.cond == c].sort_values("gyro_rms"); xs = k + np.linspace(-0.2, 0.2, len(m)); med = float(m.gyro_rms.median())
            x.scatter(xs[~m.re.values], m.gyro_rms[~m.re], s=62, color=COL[c], edgecolor=SURF, linewidth=1.6, zorder=3)
            x.scatter(xs[m.re.values], m.gyro_rms[m.re], s=46, facecolor=SURF, edgecolor=COL[c], linewidth=2, zorder=3)
            x.plot([k - 0.32, k + 0.32], [med, med], color=INK, lw=2, solid_capstyle="round", zorder=4)
            x.text(k, m.gyro_rms.max() + 0.035 * ymax, f"{med:.3f}" + ("" if c == "pos" else f"\n{med / base:.2f}×"), ha="center", va="bottom", fontsize=10, color=INK, linespacing=1.25)
        x.set_xticks(range(len(ORDER))); x.set_xticklabels([LAB[c] for c in ORDER], fontsize=9, color=INK2)
        x.set_xlim(-0.6, len(ORDER) - 0.4); x.set_ylim(0, ymax); x.grid(axis="y", color=GRID, lw=0.8); x.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            x.spines[sp].set_visible(False)
        x.spines["bottom"].set_color(GRID); x.tick_params(axis="both", length=0, labelcolor=INK2)
        x.set_title(name, fontsize=10, color=INK, loc="left")
    ax[0].set_ylabel("5–10 Hz gyro rms, 2 s after stop [rad/s]", fontsize=9.5, color=INK2)
    fig.suptitle("Residual vibration after stop by controller (J2 10 deg/s × 10 deg round trips, fixed base)", fontsize=12.5, color=INK, x=0.02, ha="left", y=0.985)
    fig.text(0.02, 0.935, "Dot = trial median of valid stop windows (windows overlapping an MIT jump removed); open dot = retake; bar = median of all trials; × = ratio to position mode.\n"
             "J2 torque limit 5 N·m, rate limit 0.3 N·m/cycle. Camera-site IMU, firmware S-V1.7-3, piper_sdk 0.6.2, 2026-11-19.", fontsize=8.8, color=INK2, va="top", linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.87)); fig.savefig(out, dpi=140, facecolor=SURF); print("그림:", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
