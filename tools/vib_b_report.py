#!/usr/bin/env python3
"""단계 B 요약 — 정지(B-3)·손 툭(B-4)·사인(B-4) 기록에서 잡음 바닥, 두 IMU 대역별 비, 지연(B-5), 0–100 Hz 스펙트럼(B-6).

GQ7(.gq7.csv)을 기준으로 카메라 IMU(.imu.csv)를 본다. GQ7 축은 손 툭 기록의 자이로로 카메라 축에 돌려 맞춘다.
사용: source install/setup.bash; python3 tools/vib_b_report.py --still S.csv --tap T.csv --sine N.csv [--png docs/img/p_x.png]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "common_pkg")); sys.path.insert(0, HERE)
from vib_stop_report import band, load_imu, rms  # noqa: E402

FS = 400.0
BANDS = [(2, 5), (5, 10), (10, 20)]


def both(path, t1, t2):  # → (t, 카메라 자이로·가속도, GQ7 자이로·가속도) 같은 시각 축
    t0 = float(open(path.replace(".csv", ".t0")).read())
    c = load_imu(path.replace(".csv", ".imu.csv"), t0); q = load_imu(path.replace(".csv", ".gq7.csv"), t0)
    t1 = max(t1, c["g"][0][0], q["g"][0][0], c["a"][0][0]) + 0.2; t2 = min(t2, c["g"][0][-1], q["g"][0][-1], c["a"][0][-1]) - 0.2
    t = np.arange(t1, t2, 1 / FS); rs = lambda s: np.column_stack([np.interp(t, s[0], s[1][:, k]) for k in range(3)])
    return t, rs(c["g"]), rs(c["a"]), rs(q["g"]), rs(q["a"])


def bp(x, lo, hi):
    return signal.sosfiltfilt(signal.butter(4, [lo, hi], "bandpass", fs=FS, output="sos"), x, axis=0)


def lag_ms(x, y, maxlag=0.2):  # y가 x보다 늦으면 양수
    L = int(maxlag * FS); x = x - x.mean(); y = y - y.mean()
    cc = np.array([np.dot(x[L:-L], y[L + k:len(y) - L + k]) for k in range(-L, L + 1)]); k = int(np.argmax(cc))
    fr = 0.5 * (cc[k - 1] - cc[k + 1]) / (cc[k - 1] - 2 * cc[k] + cc[k + 1]) if 0 < k < 2 * L else 0.0
    return (k - L + fr) / FS * 1000, cc[k] / np.sqrt(np.dot(x[L:-L], x[L:-L]) * np.dot(y[L:-L], y[L:-L]))


def tf(x, y, nper):  # → f, H = S_xy / S_xx, 코히어런스
    f, Pxx = signal.welch(x, FS, nperseg=nper); _, Pxy = signal.csd(x, y, FS, nperseg=nper); _, coh = signal.coherence(x, y, FS, nperseg=nper)
    return f, Pxy / Pxx, coh, Pxx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", required=True); ap.add_argument("--tap", required=True); ap.add_argument("--sine", required=True)
    ap.add_argument("--png", default="")
    a = ap.parse_args()
    # 축 회전과 지연: 손 툭 기록
    t, Gc, Ac, Gq, Aq = both(a.tap, 2.5, 1e9)
    lag, pk = lag_ms(np.linalg.norm(bp(Gq, 2, 20), axis=1), np.linalg.norm(bp(Gc, 2, 20), axis=1))
    U, _, Vt = np.linalg.svd(bp(Gc, 2, 20).T @ bp(Gq, 2, 20)); R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    print(f"[B-5] 카메라 − GQ7 지연(손 툭, 2–20 Hz 자이로 크기): {lag:+.1f} ms (상관 {pk:.2f})")
    print("축 회전 R(GQ7 → 카메라):"); print(np.round(R, 3))
    # B-3 잡음 바닥
    ts, Gcs, Acs, Gqs, Aqs = both(a.still, 4.0, 1e9); Gqs, Aqs = Gqs @ R.T, Aqs @ R.T
    seg = lambda X, lo, hi: [rms(band_fs(X[:, k], lo, hi)) for k in range(3)]
    print(f"[B-3] 정지 {ts[-1] - ts[0]:.1f} s — 카메라 축 x/y/z")
    for nm, X, u in (("카메라 자이로", Gcs, "rad/s"), ("GQ7 자이로", Gqs, "rad/s"), ("카메라 가속도", Acs, "m/s²"), ("GQ7 가속도", Aqs, "m/s²")):
        print(f"  {nm:8s} 5–10 Hz rms " + " / ".join(f"{v:.5f}" for v in seg(X, 5, 10)) + "  2–20 Hz " + " / ".join(f"{v:.5f}" for v in seg(X, 2, 20))
              + "  30–50 Hz " + " / ".join(f"{v:.5f}" for v in seg(X, 30, 50)) + f" {u}")
    k = int(0.25 * FS); env = lambda x: np.convolve(np.abs(signal.hilbert(bp(x, 2, 20))), np.ones(k) / k, "same")[k:-k]
    print(f"  포락선 바닥(2–20 Hz, 0.25 s 평활, x축 중앙값): 카메라 {np.median(env(Gcs[:, 0])):.5f}, GQ7 {np.median(env(Gqs[:, 0])):.5f} rad/s")
    # B-4 대역별 비
    Gq, Aq = Gq @ R.T, Aq @ R.T; mag = np.linalg.norm(bp(Gq, 2, 20), axis=1)
    pki, _ = signal.find_peaks(mag, height=0.25 * mag.max(), distance=int(1.5 * FS)); tt = t[pki]
    print(f"[B-4] 손 툭 {len(tt)}회 검출, 시각[s]: " + " ".join(f"{x:.1f}" for x in tt))
    parts = [("전체", np.ones(len(t), bool))]
    if len(tt) >= 8:
        cut = 0.5 * (tt[5] + tt[6]); parts += [("위→아래(앞 6회)", t < cut), ("옆(뒤)", t >= cut)]
    for rec, T, GC, GQ in [("손 툭 " + n, t[m], Gc[m], Gq[m]) for n, m in parts] + [("사인",) + both(a.sine, 2.5, 1e9)[:2] + (both(a.sine, 2.5, 1e9)[3] @ R.T,)]:
        print(f"  {rec}: 자이로 축별 [|H| 위상° 코히어런스] (카메라 / GQ7)")
        for i, ax in enumerate("xyz"):
            f, H, coh, Pxx = tf(GQ[:, i], GC[:, i], int(FS * 2)); s = f"    {ax}:"
            for lo, hi in BANDS:
                m = (f >= lo) & (f <= hi); w = Pxx[m] / Pxx[m].sum(); Hm = np.sum(H[m] * w)
                s += f"  {lo}–{hi} Hz [{np.sum(np.abs(H[m]) * w):.2f} {np.degrees(np.angle(Hm)):+.0f} {np.sum(coh[m] * w):.2f}]"
            print(s)
    xq, xc = bp(Gq[:, 0], 5, 10), bp(Gc[:, 0], 5, 10); r = []
    for x in tt:
        m = (t > x + 0.1) & (t < x + 1.2); r.append(rms(xc[m]) / rms(xq[m]))
    print("  툭마다 x축 5–10 Hz rms 비(카메라 / GQ7, 툭 0.1–1.2 s 뒤): " + " ".join(f"{v:.2f}" for v in r))
    # B-5 엔코더 지연: 사인 기록, J2 속도와 GQ7 x축(0.1–2 Hz)
    d = pd.read_csv(a.sine); tn, Gcn, _, Gqn, _ = both(a.sine, 0.5, 1e9); Gqn = Gqn @ R.T
    qd = np.radians(signal.savgol_filter(np.interp(tn, d.t.values, d.q2.values), 41, 2, deriv=1, delta=1 / FS))
    lo = lambda x: signal.sosfiltfilt(signal.butter(2, 3, fs=FS, output="sos"), x)
    for nm, G in (("GQ7", Gqn), ("카메라", Gcn)):
        i = int(np.argmax([abs(np.dot(lo(G[:, k]), qd)) for k in range(3)])); sg = np.sign(np.dot(lo(G[:, i]), qd))
        lg, pk2 = lag_ms(qd, sg * lo(G[:, i])); print(f"[B-5] {nm} − 엔코더(J2 속도, 사인, 3 Hz 저역, 축 {'xyz'[i]}): {lg:+.1f} ms (상관 {pk2:.3f}), 진폭 비 {np.std(lo(G[:, i])) / np.std(qd):.3f}")
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(11, 7))
        for nm, (T, GC, AC, GQ, AQ), st in (("still", (ts, Gcs, Acs, Gqs, Aqs), ":"), ("tap", (t, Gc, Ac, Gq, Aq), "-"), ("sine", both(a.sine, 2.5, 1e9), "--")):
            if nm == "sine":
                GQ, AQ = GQ @ R.T, AQ @ R.T
            for j, (C, Q) in enumerate(((GC, GQ), (AC, AQ))):
                f, Pc = signal.welch(C[:, [0, 2][j]], FS, nperseg=int(FS * 2)); _, Pq = signal.welch(Q[:, [0, 2][j]], FS, nperseg=int(FS * 2))
                axs[0, j].semilogy(f, np.sqrt(Pc), "C0" + st, lw=1, label=f"camera {nm}"); axs[0, j].semilogy(f, np.sqrt(Pq), "C3" + st, lw=1, label=f"GQ7 {nm}")
        for j, nm in enumerate(("gyro x (pitch) ASD [rad/s/√Hz]", "accel z ASD [m/s²/√Hz]")):
            axs[0, j].set_xlim(0, 100); axs[0, j].axvspan(5, 10, color="C1", alpha=0.12); axs[0, j].set_title(nm, fontsize=10); axs[0, j].legend(fontsize=7, ncol=3); axs[0, j].grid(alpha=0.3)
        for i, ax in enumerate("xyz"):
            f, H, coh, _ = tf(Gq[:, i], Gc[:, i], int(FS * 2)); axs[1, 0].plot(f, np.abs(H), label=f"gyro {ax}"); axs[1, 1].plot(f, coh, label=f"gyro {ax}")
        axs[1, 0].set_ylim(0, 3); axs[1, 0].set_ylabel("|H| camera / GQ7 (tap)"); axs[1, 1].set_ylabel("coherence (tap)"); axs[1, 1].axhline(0.9, color="gray", ls=":")
        for x in axs[1]:
            x.set_xlim(0, 30); x.set_xlabel("Hz"); x.grid(alpha=0.3); x.legend(fontsize=8); x.axvspan(5, 10, color="C1", alpha=0.12)
        fig.tight_layout(); fig.savefig(a.png, dpi=120); print("그림:", a.png)


def band_fs(x, lo, hi):  # vib_stop_report.band의 FS = 400 판
    x = signal.detrend(x); X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / FS); X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


if __name__ == "__main__":
    main()
