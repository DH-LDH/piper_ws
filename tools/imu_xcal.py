#!/usr/bin/env python3
"""두 IMU 기록(imu_rec / gq7_rec CSV) 맞대조 — 같은 강체에 붙였을 때 기준(ref) 대비 대상(dut)의 지연·진폭 비·위상.

순서: 장치 시각으로 등간격 재표본 → 수신 시각으로 대략 맞춤 → 자이로 크기 상호상관으로 지연 → 자이로로 축 회전(SVD) → 축별 H(f)·코히어런스
사용: python3 tools/imu_xcal.py ref.gq7.csv dut.imu.csv [--band 2 20] [--fs 200] [--png out.png]
"""
import argparse
import csv

import numpy as np
from scipy import signal


def load(path, fs):
    s = {"a": [], "g": []}
    with open(path) as f:
        for r in csv.DictReader(f):
            s[r["stream"]].append((float(r["t"]), float(r["t_dev"]) / 1000.0, float(r["x"]), float(r["y"]), float(r["z"])))
    a, g = np.array(s["a"]), np.array(s["g"])
    off = np.min(np.r_[a[:, 0] - a[:, 1], g[:, 0] - g[:, 1]])  # 수신 − 장치 시각의 최솟값 = 전송 지연이 가장 짧았던 표본
    t0, t1 = max(a[0, 1], g[0, 1]), min(a[-1, 1], g[-1, 1])
    td = np.arange(t0, t1, 1.0 / fs)
    acc = np.column_stack([np.interp(td, a[:, 1], a[:, k]) for k in (2, 3, 4)])
    gyr = np.column_stack([np.interp(td, g[:, 1], g[:, k]) for k in (2, 3, 4)])
    return td + off, acc, gyr  # 호스트 시각 축


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref"); ap.add_argument("dut")
    ap.add_argument("--band", type=float, nargs=2, default=[2.0, 20.0])
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--max-lag", type=float, default=0.3, help="지연 탐색 범위[s]")
    ap.add_argument("--png", default="")
    a = ap.parse_args()
    fs = a.fs
    tr, ar, gr = load(a.ref, fs); tdu, ad, gd = load(a.dut, fs)
    t0, t1 = max(tr[0], tdu[0]) + a.max_lag, min(tr[-1], tdu[-1]) - a.max_lag
    if t1 - t0 < 5:
        raise SystemExit("★ 겹치는 구간이 5 s 미만")
    t = np.arange(t0, t1, 1.0 / fs)
    rs = lambda tt, x, sh=0.0: np.column_stack([np.interp(t + sh, tt, x[:, k]) for k in range(3)])
    sos = signal.butter(4, a.band, "bandpass", fs=fs, output="sos")
    bp = lambda x: signal.sosfiltfilt(sos, x, axis=0)
    # 지연: 축 방향과 무관한 자이로 크기로 상호상관, 포물선 보간으로 표본 사이 값
    mr, md = np.linalg.norm(bp(rs(tr, gr)), axis=1), np.linalg.norm(bp(rs(tdu, gd)), axis=1)
    mr -= mr.mean(); md -= md.mean()
    L = int(a.max_lag * fs)
    cc = np.array([np.dot(mr[L:-L], md[L + k:len(md) - L + k]) for k in range(-L, L + 1)])
    k = int(np.argmax(cc))
    frac = 0.5 * (cc[k - 1] - cc[k + 1]) / (cc[k - 1] - 2 * cc[k] + cc[k + 1]) if 0 < k < 2 * L else 0.0
    lag = (k - L + frac) / fs  # dut가 ref보다 늦으면 양수
    peak = cc[k] / np.sqrt(np.dot(mr, mr) * np.dot(md, md))
    Gr, Ar = bp(rs(tr, gr)), bp(rs(tr, ar))
    Gd, Ad = bp(rs(tdu, gd, lag)), bp(rs(tdu, ad, lag))
    # 축 회전: dut ≈ R·ref 를 자이로로 (같은 강체면 각속도가 같음)
    U, _, Vt = np.linalg.svd(Gd.T @ Gr)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    Gr, Ar = Gr @ R.T, Ar @ R.T  # ref를 dut 축으로
    print(f"겹침 {t1 - t0:.1f} s, 대역 {a.band[0]:g}–{a.band[1]:g} Hz")
    print(f"지연(dut − ref): {lag * 1000:+.1f} ms (상관 {peak:.2f})")
    print("축 회전 R(ref→dut):"); print(np.round(R, 3))
    nper = int(fs * 4)
    rows = []
    for name, X, Y in (("자이로", Gr, Gd), ("가속도", Ar, Ad)):
        for i, ax in enumerate("xyz"):
            f, Pxx = signal.welch(X[:, i], fs, nperseg=nper)
            _, Pxy = signal.csd(X[:, i], Y[:, i], fs, nperseg=nper)
            _, coh = signal.coherence(X[:, i], Y[:, i], fs, nperseg=nper)
            H = Pxy / Pxx
            m = (f >= a.band[0]) & (f <= a.band[1]) & (coh >= 0.8)
            ratio = np.std(Y[:, i]) / np.std(X[:, i])
            if m.any():
                wgt = Pxx[m] / Pxx[m].sum()
                rows.append((name, ax, ratio, float(np.sum(np.abs(H[m]) * wgt)), float(np.degrees(np.angle(np.sum(H[m] * wgt)))), float(m.sum() / ((f >= a.band[0]) & (f <= a.band[1])).sum())))
            else:
                rows.append((name, ax, ratio, np.nan, np.nan, 0.0))
            if a.png:
                rows[-1] += ((f, H, coh),)
    print("신호  축  rms 비   |H| 가중평균  위상[°]  코히어런스≥0.8 비율")
    for r in rows:
        print(f"{r[0]} {r[1]}   {r[2]:.3f}    {r[3]:.3f}        {r[4]:+6.1f}   {r[5]:.2f}")
    print("※ 가속도는 두 센서 위치가 다르면 회전 성분(ω̇×r)만큼 달라짐 — 자이로 결과를 기준으로")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
        for j, name in enumerate(("자이로", "가속도")):
            for r in [r for r in rows if r[0] == name]:
                f, H, coh = r[6]
                axs[0, j].plot(f, np.abs(H), label=r[1]); axs[1, j].plot(f, np.degrees(np.angle(H))); axs[2, j].plot(f, coh)
            axs[0, j].set_title("gyro" if j == 0 else "accel"); axs[0, j].set_ylim(0, 2); axs[0, j].legend()
            axs[2, j].set_xlim(0, a.band[1] * 1.5); axs[2, j].set_xlabel("Hz")
        axs[0, 0].set_ylabel("|H| dut/ref"); axs[1, 0].set_ylabel("phase [deg]"); axs[2, 0].set_ylabel("coherence")
        for x in axs.flat:
            x.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(a.png, dpi=120); print("그림:", a.png)


if __name__ == "__main__":
    main()
