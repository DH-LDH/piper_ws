#!/usr/bin/env python3
"""MIT 순간 튐 한 건을 CAN 원본으로 그림 — 고속 피드백(0x251–0x256: 속도·전류·위치, 200 Hz)과 우리 MIT 명령 프레임(0x15A–0x15F)을 푼 값을 같은 시간축에. 위고 문의 첨부용(9절 11/18 14:10 2번).

사용: python3 tools/mit_jump_fig.py logs/mit/<기록>.csv <t_mit[s]> <처음 튄 관절> [--also 5] [--out 폴더] [--tag 이름]
      그림 PNG와 그 구간(전후 100 ms) candump 발췌 txt를 낸다.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C1, C2 = "#2a78d6", "#eb6834"


def s16(b):
    v = int.from_bytes(b, "big"); return v - 65536 if v > 32767 else v


def s32(b):
    v = int.from_bytes(b, "big"); return v - (1 << 32) if v >= (1 << 31) else v


def parse(path, t1, t2):  # → 피드백 {관절: [t, 속도, 전류, 위치]}, 명령 {관절: [t, pos, vel, kp, kd, tau]}, 원본 줄
    fb = {j: [] for j in range(1, 7)}; cmd = {j: [] for j in range(1, 7)}; raw = []
    for ln in open(path):
        p = ln.split(); t = float(p[0][1:-1])
        if t < t1 or t > t2:
            continue
        cid, dat = p[2].split("#"); c = int(cid, 16); raw.append(ln)
        if 0x251 <= c <= 0x256 and len(dat) == 16:
            b = bytes.fromhex(dat); fb[c - 0x250].append((t, s16(b[0:2]) / 1000, s16(b[2:4]) / 1000, np.degrees(s32(b[4:8]) / 1000)))
        elif 0x15A <= c <= 0x15F and len(dat) == 16:  # pos 16비트 ±12.5 rad, vel 12비트 ±45, Kp 12비트 0–500, Kd 12비트 ±5, τ 8비트 ±8 N·m, CRC 4비트
            d = bytes.fromhex(dat)
            cmd[c - 0x159].append((t, np.degrees(((d[0] << 8) | d[1]) / 65535 * 25 - 12.5), ((d[2] << 4) | (d[3] >> 4)) / 4095 * 90 - 45, (((d[3] & 0x0F) << 8) | d[4]) / 4095 * 500,
                                   ((d[5] << 4) | (d[6] >> 4)) / 4095 * 10 - 5, (((d[6] & 0x0F) << 4) | (d[7] >> 4)) / 255 * 16 - 8, dat))
    return {j: np.array(v) for j, v in fb.items()}, cmd, raw


def style(x, ylab):
    x.set_facecolor(SURF); x.grid(color=GRID, lw=0.8); x.set_axisbelow(True); x.set_ylabel(ylab, fontsize=9, color=INK2)
    for sp in ("top", "right"):
        x.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        x.spines[sp].set_color(GRID)
    x.tick_params(length=0, labelcolor=INK2, labelsize=8.5); x.axvline(0, color=INK2, lw=0.8, ls=":")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv"); ap.add_argument("t_mit", type=float); ap.add_argument("joint", type=int)
    ap.add_argument("--also", type=int, default=0, help="같이 그릴 관절(따라 움직인 관절)")
    ap.add_argument("--out", default="."); ap.add_argument("--tag", default="")
    ap.add_argument("--span", type=float, default=0.1, help="전후 시간[s]")
    ap.add_argument("--title", default="", help="그림 제목(없으면 기본 문장)")
    a = ap.parse_args()
    d = pd.read_csv(a.csv); t0 = float(open(a.csv.replace(".csv", ".t0")).read()); te = t0 + d[d.phase == "mit"].t.values[0] + a.t_mit
    fb, cmd, raw = parse(a.csv.replace(".csv", ".candump"), te - a.span - 0.05, te + a.span + 0.05)
    j = a.joint; F = fb[j]; i0 = int(np.argmax(np.abs(F[:, 1]) > 0.5)) if (np.abs(F[:, 1]) > 0.5).any() else len(F) // 2  # 속도가 처음 0.5 rad/s를 넘은 피드백 프레임 = 시각 0
    tz = F[i0, 0]; ms = lambda t: (np.asarray(t) - tz) * 1000
    name = os.path.basename(a.csv)[:15]; tag = a.tag or f"{name}_J{j}_{a.t_mit:.2f}s"
    with open(os.path.join(a.out, tag + "_candump.txt"), "w") as f:
        f.write(f"# candump -L excerpt, {name}, +/-{a.span * 1000:.0f} ms around the J{j} jump (t = 0 at epoch {tz:.6f}), firmware S-V1.7-3, piper_sdk 0.6.2\n"
                "# 0x251-0x256: high-speed feedback J1-J6 (speed int16 0.001 rad/s, current int16 0.001 A, position int32 0.001 rad); 0x15A-0x15F: our MIT command J1-J6\n")
        f.writelines(ln for ln in raw if te - a.span <= float(ln.split()[0][1:-1]) <= te + a.span)
    js = [j] + ([a.also] if a.also else []); col = {js[0]: C1}
    if a.also:
        col[a.also] = C2
    C = cmd[j]; tc = ms([c[0] for c in C]); n_unique = len({c[6] for c in C if abs((c[0] - tz)) <= a.span})
    fig, ax = plt.subplots(5, 1, figsize=(8.6, 9.4), sharex=True, facecolor=SURF)
    for k, (idx, lab) in enumerate(((2, "motor current [A]"), (1, "joint speed [rad/s]"))):
        for jj in js:
            ax[k].plot(ms(fb[jj][:, 0]), fb[jj][:, idx], color=col[jj], lw=2, marker="o", ms=3.5, label=f"J{jj}")
        style(ax[k], lab)
    for jj in js:
        b = fb[jj][fb[jj][:, 0] < tz - 0.02, 3].mean(); ax[2].plot(ms(fb[jj][:, 0]), fb[jj][:, 3] - b, color=col[jj], lw=2, marker="o", ms=3.5, label=f"J{jj} (from {b:+.1f} deg)")
    style(ax[2], "position change [deg]")
    ax[3].plot(tc, [c[1] for c in C], color=INK, lw=2, marker="s", ms=3); style(ax[3], f"J{j} command:\nposition ref [deg]")
    fit = lambda v, h: (min(v) - h, max(v) + h)  # 명령 값이 변하면 그 범위가 다 보이게
    ax[3].set_ylim(*fit([c[1] for c in C], 3.0))
    ax[4].plot(tc, [c[5] for c in C], color=INK, lw=2, marker="s", ms=3); style(ax[4], f"J{j} command:\ntorque ff [N·m]")
    ax[4].set_ylim(*fit([c[5] for c in C], 0.8)); ax[4].set_xlabel("time from first feedback frame with |speed| > 0.5 rad/s [ms]", fontsize=9, color=INK2)
    ax[3].set_title(f"Kp {C[0][3]:.1f}, Kd {C[0][4]:.2f} in every frame; {n_unique} distinct J{j} command frame(s) in this window", loc="right", fontsize=8.5, color=INK2, pad=3)
    if a.also:
        ax[0].legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2); ax[2].legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
    ax[0].set_xlim(-a.span * 1000, a.span * 1000)
    fig.suptitle(a.title or f"MIT-mode jump of J{j}: motor current moves first, our command frames do not change", fontsize=12, color=INK, x=0.02, ha="left", y=0.99)
    fig.text(0.02, 0.957, f"Raw CAN (candump): feedback 0x25{j}" + (f", 0x25{a.also}" if a.also else "") + f" at 200 Hz and our MIT command 0x{0x159 + j:X} at 100 Hz.\nRecord {name}, {a.t_mit:.2f} s after MIT entry. "
             "PiPER firmware S-V1.7-3, piper_sdk 0.6.2, all joints Kp 10 / Kd 0.8.", fontsize=8.6, color=INK2, va="top", linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.925)); out = os.path.join(a.out, tag + ".png"); fig.savefig(out, dpi=140, facecolor=SURF)
    cur = F[:, 2]; pre = cur[F[:, 0] < tz - 0.03].mean(); pk = cur[np.abs(cur - pre).argmax()]
    print(f"그림: {out}\n발췌: {os.path.join(a.out, tag + '_candump.txt')}\nJ{j}: 전류 {pre:+.2f} → {pk:+.2f} A, 위치 변화 최대 {np.abs(F[:, 3] - F[F[:, 0] < tz - 0.02, 3].mean()).max():.2f}°, 명령 프레임 종류 {n_unique}개(전후 {a.span * 1000:.0f} ms)")


if __name__ == "__main__":
    main()
