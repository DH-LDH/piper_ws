#!/usr/bin/env python3
"""VIB D0 분석 — vib_probe/mit_probe 기록(+ .imu.csv)에서 떨림 주파수·감쇠비·IMU 지연을 뽑아 판정표 한 줄씩.

tap/step: 가진 순간(속도 최대 / 계단 시각) 뒤 --win 초 — q̇ FFT 피크, 로그 감소율 감쇠비, 10% 이상 이어진 주기 수
slow·mit 기록: 궤적 구간 q̇에서 --hp Hz 이상 성분의 FFT 피크
IMU: 가속도 축 중 에너지 최대 축의 FFT 피크, q̈(q̇ 미분)과의 상호상관 지연(+면 IMU가 먼저)
사용: python3 tools/vib_analyze.py logs/vib/*.csv [--plot docs/img/p_vib_e1_x.png]
"""
import argparse
import os

import numpy as np
import pandas as pd

FMIN, FMAX = 2.0, 45.0


def uniform(t, x, fs):  # 불균일 표본 → 등간격(선형 보간)
    tu = np.arange(t[0], t[-1], 1.0 / fs)
    return tu, np.interp(tu, t, x)


def hp(x, fs, fc):  # 1차 고역통과(앞뒤로 한 번씩 → 위상 0)
    if fc <= 0:
        return x - x.mean()
    a = 1.0 / (1.0 + 2 * np.pi * fc / fs)
    def one(v):
        y = np.zeros_like(v)
        for i in range(1, len(v)):
            y[i] = a * (y[i - 1] + v[i] - v[i - 1])
        return y
    return one(one(x)[::-1])[::-1]


def peak(x, fs):  # (피크 Hz, 진폭) — 해닝 창, FMIN–FMAX
    if len(x) < 16:
        return float("nan"), 0.0
    X = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))) * 4 / len(x)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    m = (f >= FMIN) & (f <= FMAX)
    i = int(np.argmax(X[m]))
    return float(f[m][i]), float(X[m][i])


def spectrum(x, fs):
    X = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)))) * 4 / len(x)
    return np.fft.rfftfreq(len(x), 1.0 / fs), X


def decay(x, fs, f):  # 양의 극대값들로 (감쇠비 ζ, 첫 피크의 10% 이상 이어진 주기 수) — 극대는 0.7주기 이상 떨어진 것만(잡음 극대 제외)
    gap = int(0.7 * fs / f) if f > 0 else 1
    pk = []
    for i in range(1, len(x) - 1):
        if x[i] > x[i - 1] and x[i] >= x[i + 1] and x[i] > 0:
            if pk and i - pk[-1] < gap:
                if x[i] > x[pk[-1]]:
                    pk[-1] = i
            else:
                pk.append(i)
    if len(pk) < 2:
        return float("nan"), len(pk)
    i0 = int(np.argmax([x[i] for i in pk[:5]]))  # 가진 직후 가장 큰 극대부터
    a = np.array([x[i] for i in pk[i0:]])
    keep = a >= 0.1 * a[0]
    n = int(np.argmin(keep)) if not keep.all() else len(a)
    if n < 2:
        return float("nan"), n
    dl = -np.polyfit(np.arange(n), np.log(a[:n]), 1)[0]  # 극대 로그의 기울기 — 끝점 둘만 쓰면 꼬리 잡음에 끌림
    return float(dl / np.sqrt(4 * np.pi ** 2 + dl ** 2)), n


def lag_ms(x, y, fs, max_ms=60):  # x 대비 y의 지연 — +면 y가 늦음
    x = (x - x.mean()) / (x.std() + 1e-12); y = (y - y.mean()) / (y.std() + 1e-12)
    m = int(max_ms / 1000 * fs)
    c = [np.dot(x[max(0, -k):len(x) - max(0, k)], y[max(0, k):len(y) - max(0, -k)]) for k in range(-m, m + 1)]
    return (int(np.argmax(np.abs(c))) - m) / fs * 1000


def load_imu(path, t0):
    p = path.replace(".csv", ".imu.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    d = d[d.stream == "a"].copy()
    d["t"] = d.t - t0
    return d


def analyze(path, a):
    d = pd.read_csv(path)
    t0p = path.replace(".csv", ".t0")
    t0 = float(open(t0p).read()) if os.path.exists(t0p) else None
    name = os.path.basename(path)
    stage = d.phase.iloc[-1] if "phase" in d else ""
    body = d[~d.phase.isin(["settle", "measure", "estop", "after"])] if "phase" in d else d
    if "ts_h" in d and (d.ts_h > 0).any():  # SDK 수신 시각이 같은 행은 같은 피드백 — 중복 제거
        body = body.drop_duplicates("ts_h")
    t = body.t.values
    fs = a.fs
    qd = {j: uniform(t, body[f"qd{j}"].values, fs) for j in range(1, 7)}
    qref = {j: uniform(t, body[f"qref{j}"].values, fs)[1] for j in range(1, 7)}
    tu = qd[1][0]
    kind = a.kind if a.kind != "auto" else "step" if "_step" in name else "tap" if "_tap" in name else "traj"
    if kind == "step":
        dq = np.abs(np.diff(sum(qref.values())))
        ev = [tu[i + 1] for i in np.where(dq > 0.5)[0]]
    elif kind == "tap":
        mag = np.max([np.abs(qd[j][1]) for j in range(1, 7)], axis=0)
        ev = [tu[int(np.argmax(mag))]]
    else:
        ev = [None]
    imu = load_imu(path, t0) if t0 is not None else None
    rows = []
    for e in ev:
        if e is None:  # 궤적 전체, 저주파(궤적 자체) 제거
            sel = slice(None)
        else:
            sel = (tu >= e) & (tu < e + a.win)
        j = a.joint or max(range(1, 7), key=lambda j: np.std(hp(qd[j][1][sel], fs, a.hp)))
        x = hp(qd[j][1][sel], fs, a.hp)
        f_q, amp_q = peak(x, fs)
        z, n = decay(x, fs, f_q) if e is not None else (float("nan"), 0)
        f_i = lag = z_i = float("nan"); n_i = 0; ax = "-"
        if imu is not None and len(imu) > 50:
            ts = tu[sel]
            fi = 1.0 / np.median(np.diff(imu.t.values))
            w = (imu.t >= ts[0]) & (imu.t < ts[-1])
            if w.sum() > 32:
                ax = max("xyz", key=lambda c: np.std(hp(imu[c][w].values, fi, a.hp)))
                ti, ai = uniform(imu.t[w].values, imu[ax][w].values, fs)
                ai = hp(ai, fs, a.hp)
                f_i, _ = peak(ai, fs)
                if e is not None:  # q̇ 정지 잡음(최대 0.1 rad/s)에 묻히는 약한 가진은 IMU 감쇠가 기준
                    z_i, n_i = decay(ai, fs, f_i)
                xx = np.gradient(np.interp(ti, ts, x), 1.0 / fs)  # 가속도끼리 비교 — q̇와 바로 비교하면 미분의 90° 위상이 지연으로 보임
                lag = lag_ms(ai, xx, fs)  # +면 IMU가 q̈보다 먼저
        rows.append(dict(file=name, kind=kind, t_ev=e, joint=j, f_qd=f_q, amp_qd=amp_q, zeta=z, cycles=n,
                         imu_axis=ax, f_imu=f_i, zeta_imu=z_i, cycles_imu=n_i, imu_lead_ms=lag))
    return rows, (d, qd, imu, ev, rows)


def plot(results, out, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from traj_analyze import _font
    _font()
    n = len(results)
    fig, axs = plt.subplots(n, 2, figsize=(12, 3 * n), squeeze=False)
    for k, (rows, (d, qd, imu, ev, _)) in enumerate(results):
        r = rows[0]; j = r["joint"]; tu, v = qd[j]
        ax = axs[k][0]
        ax.plot(tu, v, lw=0.8, label=f"q̇{j} [rad/s]")
        for e in ev:
            if e is not None:
                ax.axvline(e, color="k", lw=0.5, ls="--")
        if imu is not None and r["imu_axis"] != "-":
            a2 = ax.twinx(); c = r["imu_axis"]
            a2.plot(imu.t, imu[c] - imu[c].mean(), lw=0.5, color="C1", alpha=0.7, label=f"IMU a{c} [m/s²]")
            a2.legend(loc="upper right", fontsize=8)
        ax.set_xlim(tu[0], tu[-1]); ax.set_title(r["file"], fontsize=9); ax.legend(loc="upper left", fontsize=8)
        ax = axs[k][1]
        sel = slice(None) if ev[0] is None else (tu >= ev[0]) & (tu < ev[0] + a.win)
        f, X = spectrum(hp(v[sel], a.fs, a.hp), a.fs)
        ax.semilogy(f, X + 1e-9, lw=0.8, label=f"q̇{j} 피크 {r['f_qd']:.1f} Hz, ζ {r['zeta']:.3f}")
        ax.set_xlim(0, FMAX); ax.set_xlabel("Hz"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(f"그림: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--win", type=float, default=2.0, help="가진 뒤 분석 창[s]")
    ap.add_argument("--hp", type=float, default=1.5, help="고역통과[Hz] — 궤적·자세 성분 제거")
    ap.add_argument("--fs", type=float, default=200.0, help="재표본 주기[Hz]")
    ap.add_argument("--plot", default="")
    ap.add_argument("--kind", choices=["auto", "tap", "step", "traj"], default="auto", help="가진 종류 — mit_probe 기록(조건 b 툭)은 tap 지정")
    ap.add_argument("--joint", type=int, default=0, help="주 관절 고정(0이면 응답이 가장 큰 관절)")
    a = ap.parse_args()
    results = [analyze(p, a) for p in a.csv if not p.endswith((".imu.csv", ".tdc.csv"))]
    print("| 파일 | 가진 | 시각[s] | 주 관절 | 주 피크 Hz(q̇) | 진폭 | 피크 Hz(IMU, 축) | 감쇠비(q̇/IMU) | 지속 주기 수(q̇/IMU) | IMU 선행[ms] |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for rows, _ in results:
        for r in rows:
            te = "-" if r["t_ev"] is None else f"{r['t_ev']:.2f}"
            print(f"| {r['file']} | {r['kind']} | {te} | J{r['joint']} | {r['f_qd']:.1f} | {r['amp_qd']:.4f} | "
                  f"{r['f_imu']:.1f} ({r['imu_axis']}) | {r['zeta']:.3f} / {r['zeta_imu']:.3f} | {r['cycles']} / {r['cycles_imu']} | {r['imu_lead_ms']:.0f} |")
    if a.plot:
        plot(results, a.plot, a)


if __name__ == "__main__":
    main()
