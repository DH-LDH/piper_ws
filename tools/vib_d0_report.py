#!/usr/bin/env python3
"""VIB D0 판정표·그림 — vib_probe 기록(+ .imu.csv)에서 구조 모드(IMU, 20 Hz 저역)와 a′ 저속 궤적 떨림을 정리.

모드 표: 툭(IMU 크기 최대) 또는 계단 도달(q̇ < 0.05 rad/s) 뒤 1.5 s, IMU 가속도 z 20 Hz 저역 → 극대 간격 주파수, 로그 감소율 ζ, 첫 극대 20% 이상 주기 수
a′ 표: 1 s 창마다 IMU z 5–10 Hz 진폭과 목표 속도, 정지·궤적 구간 2–20 / 30–50 Hz 대역 rms
사용: python3 tools/vib_d0_report.py logs/vib/20261023_1* [--exclude poseA_x2] [--img docs/img/p_vib_e1]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vib_analyze as V  # noqa: E402


def load(f):
    t0 = float(open(f.replace(".csv", ".t0")).read())
    d = pd.read_csv(f)
    i = pd.read_csv(f.replace(".csv", ".imu.csv"))
    i = i[i.stream == "a"].copy(); i["t"] -= t0
    return d, i


def ring(i, t_from, base_mean, c="z"):  # (주파수, ζ, 주기 수, 첫 진폭, 시각, 신호)
    w = i[(i.t > t_from) & (i.t < t_from + 1.5)]
    fs = 1 / np.median(np.diff(w.t.values))
    y = V.lp(w[c].values - base_mean, fs, 20); y = y - y.mean()
    f0, _ = V.peak(y, fs)
    gap = int(0.6 * fs / f0); pk = []
    for k in range(1, len(y) - 1):
        if y[k] > y[k - 1] and y[k] >= y[k + 1] and y[k] > 0:
            if pk and k - pk[-1] < gap:
                if y[k] > y[pk[-1]]:
                    pk[-1] = k
            else:
                pk.append(k)
    a = np.array([y[k] for k in pk])
    if len(a) < 3:
        return np.nan, np.nan, len(a), (a[0] if len(a) else np.nan), w.t.values, y
    i0 = int(np.argmax(a[:3])); a = a[i0:]; pk = pk[i0:]
    keep = a >= 0.2 * a[0]; n = int(np.argmin(keep)) if not keep.all() else len(a)
    if n < 3:
        return np.nan, np.nan, n, a[0], w.t.values, y
    f = 1 / np.mean(np.diff(w.t.values[pk[:n]]))
    dl = -np.polyfit(np.arange(n), np.log(a[:n]), 1)[0]
    return f, dl / np.sqrt(4 * np.pi ** 2 + dl ** 2), n, a[0], w.t.values, y


def events(f, d, i):  # [(분석 시작 시각, 기준 구간 IMU, 라벨)]
    if "_tap_" in f:
        tap = d[d.phase == "tap"]; w = i[(i.t > tap.t.iloc[0]) & (i.t < tap.t.iloc[0] + 7)]
        m = np.sqrt(((w[["x", "y", "z"]] - w[["x", "y", "z"]].mean()) ** 2).sum(axis=1)).values
        return [(w.t.values[int(np.argmax(m))] + 0.05, w, f"툭 {m.max():.1f} m/s²")]
    j = int(f.split("_step_j")[1][0]); body = d[d.phase == "step"]
    dq = np.abs(np.diff(body[f"qref{j}"].values)); ev = body.t.values[1:][dq > 0.5]
    ev = [e for k, e in enumerate(ev) if k == 0 or e - ev[k - 1] > 0.1]
    out = []
    for n, e in enumerate(ev):
        b = body[(body.t > e) & (body.t < e + 1.5)]; mv = b.t.values[np.abs(b[f"qd{j}"].values) > 0.05]
        out.append((mv.max() if len(mv) else e, i[(i.t > e - 1.5) & (i.t < e - 0.1)], "올림" if n == 0 else "내림"))
    return out


def band_rms(x, fs, lo, hi):
    X = np.fft.rfft((x - x.mean()) * np.hanning(len(x))); f = np.fft.rfftfreq(len(x), 1 / fs); P = np.abs(X) ** 2
    return np.sqrt(P[(f >= lo) & (f < hi)].sum() / P.sum()) * x.std()


def slow_table(f, d, i):
    js = [int(c) for c in f.split("_slow_j")[1].split("_")[0]]
    s = d[d.phase == "slow"]; ts = s.t.iloc[0]
    err = max(np.abs(s[f"q{j}"] - s[f"qref{j}"]).max() for j in js)
    print(f"\n### {os.path.basename(f)} — 추종 오차 최대 {err:.2f}°")
    print("| 구간 | ax 2–20 Hz | ay 2–20 Hz | az 2–20 Hz | ax 30–50 Hz | ay 30–50 Hz | az 30–50 Hz |\n|---|---|---|---|---|---|---|")
    for k, (a, b) in {"정지(앞)": (ts + 0.2, ts + 1.9), "궤적": (ts + 4.0, ts + 14.0), "정지(뒤)": (ts + 14.5, ts + 16.8)}.items():
        w = i[(i.t > a) & (i.t < b)]; fs = 1 / np.median(np.diff(w.t.values))
        x = {c: V.hp(w[c].values, fs, 1.5) for c in "xyz"}
        print(f"| {k} | " + " | ".join(f"{band_rms(x[c], fs, 2, 20):.3f}" for c in "xyz") + " | "
              + " | ".join(f"{band_rms(x[c], fs, 30, 50):.3f}" for c in "xyz") + " |")
    j = js[0]; rows = []
    for k in range(16):
        a, b = ts + k, ts + k + 1; w = i[(i.t > a) & (i.t < b)]; x = V.hp(w.z.values, 250, 1.5)
        X = np.fft.rfft(x * np.hanning(len(x))); fr = np.fft.rfftfreq(len(x), 1 / 250); m = (fr >= 5) & (fr < 10)
        v = s[(s.t > a) & (s.t < b)]; vr = np.abs(np.diff(v[f"qref{j}"].values)).sum() / max(np.ptp(v.t.values), 1e-3)  # 창 평균 |속도| — 샘플 시각 흔들림에 둔감
        rows.append((k, vr, np.abs(X[m]).max() * 4 / len(x)))
    print("\n| 구간[s] | 목표 속도 J%d [°/s] | az 5–10 Hz 진폭 [m/s²] |\n|---|---|---|" % j)
    for k, vr, amp in rows:
        print(f"| {k}–{k + 1} | {vr:.1f} | {amp:.2f} |")
    return ts, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--exclude", nargs="*", default=[], help="파일명에 이 문자열이 있으면 제외(예: 밀어 버린 시행)")
    ap.add_argument("--img", default="", help="그림 경로 앞부분 — _modes.png, _slow.png가 붙음")
    a = ap.parse_args()
    files = sorted(f for f in a.csv if f.endswith(".csv") and not f.endswith((".imu.csv", ".tdc.csv"))
                   and "_dry" not in f and "_fake" not in f and os.path.exists(f.replace(".csv", ".imu.csv"))
                   and not any(x in f for x in a.exclude))
    print("| 시행 | 가진 | 주파수 [Hz] | 감쇠비 ζ | 주기 수 | 첫 진폭 [m/s²] |\n|---|---|---|---|---|---|")
    traces = {}
    for f in files:
        if "_tap_" not in f and "_step_" not in f:
            continue
        d, i = load(f); name = os.path.basename(f)[16:-4]
        for t_from, base, lab in events(f, d, i):
            fr, z, n, a0, tt, y = ring(i, t_from, base["z"].mean())
            print(f"| {name} | {lab} | {fr:.2f} | {z:.3f} | {n} | {a0:.2f} |")
            if "_step_j3_" in f and lab == "올림":
                traces[name] = (tt - tt[0], y)
    slow = None
    for f in files:
        if "_slow_" in f:
            d, i = load(f); r = slow_table(f, d, i)
            if "_slow_j23_" in f:
                slow = (d, i, r)
    if a.img:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from traj_analyze import _font
        _font()
        fig, ax = plt.subplots(figsize=(8, 3.2))
        for name, (tt, y) in traces.items():
            ax.plot(tt, y, lw=1, label=name)
        ax.set_xlabel("J3 2° 계단 도달 뒤 시간 [s]"); ax.set_ylabel("IMU az (20 Hz 저역) [m/s²]"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(a.img + "_modes.png", dpi=120); print(f"\n그림: {a.img}_modes.png")
        if slow:
            d, i, (ts, rows) = slow
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
            s = d[d.phase.isin(["settle", "slow", "after"])]
            a1.plot(s.t - ts, s.qref2, lw=1, label="J2 목표 [°]"); a1.plot(s.t - ts, s.qref3 + 194, lw=1, label="J3 목표 + 194 [°]")
            a1.legend(fontsize=8); a1.grid(alpha=0.3); a1.set_title("a′ 위치 모드 저속 궤적(J2·J3 ±5° 0.25 Hz, 자세 A)")
            w = i[(i.t > ts - 2) & (i.t < ts + 18)]
            a2.plot(w.t - ts, V.lp(V.hp(w.z.values, 250, 1.5), 250, 20), lw=0.6, color="C1", label="IMU az 1.5–20 Hz")
            a2.step([r[0] + 0.5 for r in rows], [r[2] for r in rows], where="mid", color="k", lw=1.2, label="5–10 Hz 진폭(1 s 창)")
            a2.set_xlabel("궤적 시작 기준 시간 [s]"); a2.set_ylabel("m/s²"); a2.legend(fontsize=8); a2.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(a.img + "_slow.png", dpi=120); print(f"그림: {a.img}_slow.png")


if __name__ == "__main__":
    main()
