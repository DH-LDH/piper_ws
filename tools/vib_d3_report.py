#!/usr/bin/env python3
"""D3 지표(8절 10/27 오후 16) — 운동 외란 사인(vib_probe slow·mit_probe sine) 한 회씩.

창: 사인 시작 + 램프 2 s ~ 시작 + cycles/hz(꼬리 없음), MIT 순간 튐(엔코더 한 주기 > 0.7 rad/s) ±0.5 s 제외.
지표: IMU 5–10 Hz rms(카메라 축 acc z·gyro x·각가속도 = gyro x 미분), 사인 관절별 위치·각속도 추종 오차 rms(q̇_ref는 q_ref 미분), 시작 기울기(사인 전 유지 구간, 같은 자세).
(8절 10/29 오전 2) 둘째 줄: 3축 크기 |acc|·|gyro|, base 요 각속도 gyro z(유지 자세 회전), 2–20 Hz 자이로 봉우리 주파수. 사인 주파수는 q_ref에서 자동.
사용: python3 tools/vib_d3_report.py logs/vib/…_slow_….csv logs/mit/…_sine.csv …
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "common_pkg"))
from imu_frame import ImuToBase, static_tilt  # noqa: E402
from piper_kin import PiperKin  # noqa: E402


def bp(t, x, keep, fs, deriv=False):  # 5–10 Hz 대역 신호와 남길 표본 — deriv면 주파수 영역 미분(jω)
    tu = np.arange(t[0], t[-1], 1 / fs); xu = np.interp(tu, t, x); X = np.fft.rfft(xu - xu.mean()); f = np.fft.rfftfreq(len(xu), 1 / fs)
    X[(f < 5) | (f > 10)] = 0
    if deriv:
        X = X * 2j * np.pi * f
    return np.fft.irfft(X, len(xu)), np.interp(tu, t, keep.astype(float)) > 0.99


def bp_rms(t, x, keep, fs, deriv=False):
    y, k = bp(t, x, keep, fs, deriv)
    return float(np.sqrt(np.mean(y[k] ** 2))) if k.sum() > 50 else float("nan")


def vec_rms(t, X, keep, fs):  # 3축 크기 rms = √(Σ축 대역 신호² 평균)
    ys = [bp(t, X[:, c], keep, fs) for c in range(3)]; k = ys[0][1]
    return float(np.sqrt(np.mean(sum(y[k] ** 2 for y, _ in ys)))) if k.sum() > 50 else float("nan")


def peak_hz(t, X, keep, fs, lo=2.0, hi=20.0):  # 제외 표본을 0으로 둔 3축 합 스펙트럼의 lo–hi Hz 최대 봉우리
    tu = np.arange(t[0], t[-1], 1 / fs); k = np.interp(tu, t, keep.astype(float)) > 0.99; P = 0
    for c in range(3):
        xu = np.interp(tu, t, X[:, c]); xu = np.where(k, xu - xu[k].mean(), 0.0); P = P + np.abs(np.fft.rfft(xu * np.hanning(len(xu)))) ** 2
    f = np.fft.rfftfreq(len(tu), 1 / fs); b = (f >= lo) & (f <= hi)
    return float(f[b][np.argmax(P[b])])


def report(p, tb):
    d = pd.read_csv(p); ph = "mit" if "mit" in set(d.phase) else "slow"; m = d[d.phase == ph]; t = m.t.values
    J = [j for j in range(1, 7) if np.ptp(m[f"qref{j}"].values) > 1.0]  # 사인 관절
    mv = np.abs(m[f"qref{J[0]}"].values - m[f"qref{J[0]}"].values[0]) > 0.05; ts = t[mv][0]
    tq = np.arange(ts, t[mv][-1], 0.005); qr = np.interp(tq, t, m[f"qref{J[0]}"].values); F = np.abs(np.fft.rfft(qr - qr.mean(), 8 * len(qr)))
    hz = round(20 * float(np.fft.rfftfreq(8 * len(qr), 0.005)[np.argmax(F[1:]) + 1])) / 20  # 사인 주파수(0.25·0.5 Hz), 0.05 Hz 격자로 맞춤 — FFT 오차가 창 길이를 바꾸지 않게
    cyc = round((t[mv][-1] - ts) * hz) / hz  # 실제 주기 수(꼬리 버그 회도 주기 끝에서 자름)
    t0, t1 = ts + 2.0, ts + cyc; w = (t >= t0) & (t <= t1); mm = m[w]; tt = mm.t.values
    qd = np.abs(np.diff(np.radians(mm[[f"q{j}" for j in range(1, 7)]].values), axis=0)) / np.maximum(np.diff(tt)[:, None], 1e-3)
    sp = tt[1:][(qd > 0.7).any(1)] if ph == "mit" else np.array([])
    ok = np.ones(len(tt), bool)
    for s in sp:
        ok &= np.abs(tt - s) > 0.5
    err = {}
    for j in J:
        e = (mm[f"q{j}"] - mm[f"qref{j}"]).values; qref = np.radians(mm[f"qref{j}"].values)
        vref = np.gradient(qref, tt); ve = mm[f"qd{j}"].values - vref
        err[j] = (np.sqrt(np.mean(e[ok] ** 2)), np.sqrt(np.mean(ve[ok] ** 2)))
    i = pd.read_csv(p.replace(".csv", ".imu.csv")); i["t"] = i.t - float(open(p.replace(".csv", ".t0")).read()); i = i[(i.t >= t0 - 1) & (i.t <= t1 + 1)]
    res = []
    for st, ax in (("a", "z"), ("g", "x")):
        s = i[i.stream == st]; tsv = s.t.values; fs = 1.0 / np.median(np.diff(tsv)); keep = (tsv >= t0) & (tsv <= t1)
        for q in sp:
            keep &= np.abs(tsv - q) > 0.5
        res.append(bp_rms(tsv, s[ax].values, keep, fs))
        if st == "g":
            res.append(bp_rms(tsv, s[ax].values, keep, fs, deriv=True))
    s = i[i.stream == "g"]; tsv = s.t.values; fs = 1.0 / np.median(np.diff(tsv)); keep = (tsv >= t0) & (tsv <= t1)
    for q in sp:
        keep &= np.abs(tsv - q) > 0.5
    G = s[["x", "y", "z"]].values; R = tb.rot(np.radians(m[[f"q{j}" for j in range(1, 7)]].values[t < ts].mean(0)))
    gz = bp_rms(tsv, (G @ R.T)[:, 2], keep, fs); g_vec = vec_rms(tsv, G, keep, fs); f_pk = peak_hz(tsv, G, keep, fs)
    a = i[i.stream == "a"]; tav = a.t.values; fsa = 1.0 / np.median(np.diff(tav)); ka = (tav >= t0) & (tav <= t1)
    for q in sp:
        ka &= np.abs(tav - q) > 0.5
    a_vec = vec_rms(tav, a[["x", "y", "z"]].values, ka, fsa)
    tilt = static_tilt(p, ts - 1.5, ts - 0.1)
    print(f"{os.path.basename(p)[:15]} {ph:4s} J{J} | 창 {t1 - t0:.1f} s 튐 {len(sp)}(제외 {100 * (1 - ok.mean()):.0f}%) | IMU 5–10 Hz acc z {res[0]:.3f} gyro x {res[1]:.4f} 각가속도 {res[2]:.2f} rad/s² | "
          + " ".join(f"J{j} 위치 {err[j][0]:.3f}° 각속도 {err[j][1]:.4f} rad/s" for j in J) + f" | 시작 기울기 {tilt:.2f}°")
    print(f"{'':15s} {hz:.2f} Hz | |acc| {a_vec:.3f} |gyro| {g_vec:.4f} gyro z(base 요) {gz:.4f} | 자이로 2–20 Hz 봉우리 {f_pk:.2f} Hz")


if __name__ == "__main__":
    tb = ImuToBase(PiperKin())
    for p in sys.argv[1:]:
        report(p, tb)
