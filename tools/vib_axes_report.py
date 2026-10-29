#!/usr/bin/env python3
"""D3 보조 지표(8절 10/29 오전 8·9) — 회마다 base 축별 IMU 대역 rms와 10–13 Hz 성분.

창: vib_d3_report와 같음(사인 시작 + 2 s ~ 주기 끝, MIT 튐 ±0.5 s 제외). 계획 회(_plan.csv)는 처프가 없는 on/off a′ 구간(MIT 80–104 s).
출력: base 성분(gy 피치 = J2 평면, gz 요 = J1, gx 롤, ax·ay 수평, az 수직) 5–10 Hz·2–20 Hz rms와 크기,
자이로 2–20 Hz 봉우리, |gyro| 10–13 Hz rms, J2 토크 명령(tff2) 10–13 Hz rms와 2–20 Hz 봉우리.
사용: python3 tools/vib_axes_report.py [--csv out.csv] logs/…_slow_….csv logs/mit/…_sine.csv …
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "common_pkg"))
from imu_frame import ImuToBase  # noqa: E402
from piper_kin import PiperKin  # noqa: E402


def band(t, x, keep, fs, lo, hi):  # 균일 재표본 → FFT 대역 → 남길 표본만 rms
    tu = np.arange(t[0], t[-1], 1 / fs); xu = np.interp(tu, t, x); X = np.fft.rfft(xu - xu.mean()); f = np.fft.rfftfreq(len(xu), 1 / fs)
    X[(f < lo) | (f > hi)] = 0; y = np.fft.irfft(X, len(xu)); k = np.interp(tu, t, keep.astype(float)) > 0.99
    return float(np.sqrt(np.mean(y[k] ** 2))) if k.sum() > 50 else float("nan")


def peak(t, X, keep, fs, lo=2.0, hi=20.0):  # 제외 표본 0, 열 합 스펙트럼의 lo–hi Hz 최대 봉우리
    tu = np.arange(t[0], t[-1], 1 / fs); k = np.interp(tu, t, keep.astype(float)) > 0.99; P = 0
    for c in range(X.shape[1]):
        xu = np.interp(tu, t, X[:, c]); xu = np.where(k, xu - xu[k].mean(), 0.0); P = P + np.abs(np.fft.rfft(xu * np.hanning(len(xu)))) ** 2
    f = np.fft.rfftfreq(len(tu), 1 / fs); b = (f >= lo) & (f <= hi)
    return float(f[b][np.argmax(P[b])])


def window(m):  # (t0, t1, 유지 자세 끝) — 사인 회는 vib_d3_report 창, 계획 회는 on/off a′ 구간(처프 4–12 Hz가 10–13 Hz 토크에 섞이지 않게)
    t = m.t.values; J = [j for j in range(1, 7) if np.ptp(m[f"qref{j}"].values) > 1.0]
    if "chirp" in m:
        return t[0] + 80.0, t[0] + 104.0, t[0] + 8.3
    mv = np.abs(m[f"qref{J[0]}"].values - m[f"qref{J[0]}"].values[0]) > 0.05; ts = t[mv][0]
    tq = np.arange(ts, t[mv][-1], 0.005); qr = np.interp(tq, t, m[f"qref{J[0]}"].values); F = np.abs(np.fft.rfft(qr - qr.mean(), 8 * len(qr)))
    hz = round(20 * float(np.fft.rfftfreq(8 * len(qr), 0.005)[np.argmax(F[1:]) + 1])) / 20
    return ts + 2.0, ts + round((t[mv][-1] - ts) * hz) / hz, ts


def report(p, tb):
    d = pd.read_csv(p); ph = "mit" if "mit" in set(d.phase) else "slow"; m = d[d.phase == ph]; t = m.t.values
    t0, t1, ts = window(m); w = (t >= t0) & (t <= t1); mm = m[w]; tt = mm.t.values
    qd = np.abs(np.diff(np.radians(mm[[f"q{j}" for j in range(1, 7)]].values), axis=0)) / np.maximum(np.diff(tt)[:, None], 1e-3)
    sp = tt[1:][(qd > 0.7).any(1)] if ph == "mit" else np.array([])
    R = tb.rot(np.radians(m[[f"q{j}" for j in range(1, 7)]].values[t <= ts + 0.2].mean(0)))
    ip = p.replace(".csv", ".imu.csv")
    i = pd.read_csv(ip); i["t"] = i.t - float(open(p.replace(".csv", ".t0")).read()); i = i[(i.t >= t0 - 1) & (i.t <= t1 + 1)]
    o = {"run": os.path.basename(p)[:15], "mode": ph, "spikes": len(sp)}
    for st in "ga":
        s = i[i.stream == st]; tv = s.t.values; fs = 1 / np.median(np.diff(tv)); keep = (tv >= t0) & (tv <= t1)
        for q in sp:
            keep &= np.abs(tv - q) > 0.5
        B = s[["x", "y", "z"]].values @ R.T
        for lo, hi in ((5, 10), (2, 20)):
            c = [band(tv, B[:, a], keep, fs, lo, hi) for a in range(3)]
            for a, n in enumerate("xyz"):
                o[f"{st}{n}_{lo}_{hi}"] = c[a]
            o[f"{st}|_{lo}_{hi}"] = float(np.sqrt(sum(v * v for v in c)))
        if st == "g":
            o["g_pk"] = peak(tv, B, keep, fs)
            o["g|_10_13"] = float(np.sqrt(sum(band(tv, B[:, a], keep, fs, 10, 13) ** 2 for a in range(3))))
    keep = np.ones(len(tt), bool)
    for q in sp:
        keep &= np.abs(tt - q) > 0.5
    fs = 1 / np.median(np.diff(tt)); tq = mm.tff2.values if "tff2" in mm else None  # 위치 모드 회는 토크 명령 없음
    o["tau_10_13"] = band(tt, tq, keep, fs, 10, 13) if tq is not None else float("nan")
    o["tau_pk"] = peak(tt, tq[:, None], keep, fs) if tq is not None else float("nan")
    return o


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--csv", default="")
    a = ap.parse_args(); tb = ImuToBase(PiperKin()); rows = []
    for p in a.files:
        o = report(p, tb); rows.append(o)
        print(f"{o['run']} {o['mode']:4s} 튐 {o['spikes']:2d} | 5–10 Hz gy {o['gy_5_10']:.4f} gz {o['gz_5_10']:.4f} gx {o['gx_5_10']:.4f} "
              f"ax {o['ax_5_10']:.3f} ay {o['ay_5_10']:.3f} az {o['az_5_10']:.3f} |a| {o['a|_5_10']:.3f} |g| {o['g|_5_10']:.4f} | "
              f"2–20 Hz |a| {o['a|_2_20']:.3f} |g| {o['g|_2_20']:.4f} | 봉우리 {o['g_pk']:.2f} Hz | 10–13 Hz |g| {o['g|_10_13']:.4f} τ2 {o['tau_10_13']:.4f} (τ2 봉우리 {o['tau_pk']:.2f} Hz)")
    if a.csv:
        pd.DataFrame(rows).to_csv(a.csv, index=False)
