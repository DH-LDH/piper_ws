#!/usr/bin/env python3
"""전달률 분석(단계 E, EXPERIMENTS 2.4절) — 팔 베이스 IMU(입력 x, GQ7)와 끝단 IMU(출력 y, 손목 D455)의 H(f) = S_xy / S_xx와 코히어런스.

여러 기록을 주면 스펙트럼을 합쳐 평균한다(10회를 한 번에). 코히어런스가 --coh(0.8) 미만인 주파수는 가린다.
좌표: 끝단은 FK로 팔 베이스 좌표로 돌린다(기록의 평균 관절각). 베이스 IMU는 --base-axes로 센서 축 → 팔 베이스 축(예: "+y -x +z" = 베이스 x는 센서 +y).
쌍: 병진 x→x, y→y, z→z(가속도), 회전 x→x, y→y, z→z(자이로), 끄덕임 → 끝단 위아래(베이스 자이로 y 미분 → 끝단 가속도 z, [m/s² per rad/s²]).
가속도는 중력 섞임 때문에 2 Hz 아래를 버린다. 자체 검증: 2–3 Hz 병진 |H|가 1 근처인지 같이 찍는다.
사용: python3 tools/vib_transmit.py logs/vib/*_e3_*.csv [--base-axes +x +y +z] [--gain 1.0] [--lag-ms 0] [--png out.png] [--out curves.csv]
      (각 기록 옆에 .imu.csv와 .gq7.csv가 있어야 함)   --selftest: 합성 신호로 계산 확인
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "common_pkg"))
FS = 200.0
PAIRS = [("acc x→x", "ax", "ax"), ("acc y→y", "ay", "ay"), ("acc z→z", "az", "az"), ("gyro x→x", "gx", "gx"), ("gyro y→y", "gy", "gy"), ("gyro z→z", "gz", "gz"),
         ("pitch acc→acc z", "dgy", "az")]


def load(path):  # → (호스트 시각 축 t, acc[n,3], gyro[n,3]) FS 등간격 — 장치 시각 + 수신 시각 최소 지연
    d = pd.read_csv(path); off = d.t.values - d.t_dev.values / 1000; ok = np.abs(off - np.median(off)) < 1.0; d = d[ok]
    th = d.t_dev.values / 1000 + np.percentile(off[ok], 1); a = (d.stream == "a").values; g = ~a
    t = np.arange(max(th[a][0], th[g][0]), min(th[a][-1], th[g][-1]), 1 / FS)
    return t, np.column_stack([np.interp(t, th[a], d[c].values[a]) for c in "xyz"]), np.column_stack([np.interp(t, th[g], d[c].values[g]) for c in "xyz"])


def axes_matrix(spec):  # "+y -x +z" → 센서 벡터를 팔 베이스 벡터로 바꾸는 행렬
    R = np.zeros((3, 3))
    for i, s in enumerate(spec):
        R[i, "xyz".index(s[-1])] = -1.0 if s[0] == "-" else 1.0
    if abs(abs(np.linalg.det(R)) - 1) > 1e-9:
        raise SystemExit("★ --base-axes는 x, y, z를 한 번씩")
    return R


def channels(acc, gyr):  # 이름 → 신호
    ch = {f"a{c}": acc[:, i] for i, c in enumerate("xyz")}; ch.update({f"g{c}": gyr[:, i] for i, c in enumerate("xyz")})
    ch["dgy"] = np.gradient(gyr[:, 1], 1 / FS)
    return ch


class Acc:  # 기록들에 걸친 스펙트럼 합
    def __init__(self, nper):
        self.nper = nper; self.S = {}; self.sec = 0.0

    def add(self, X, Y):
        n = min(len(X["ax"]), len(Y["ax"]))
        if n < self.nper:
            return False
        nseg = (n - self.nper // 2) // (self.nper // 2)  # 50% 겹침 구간 수 — 기록 길이만큼 가중
        for name, kx, ky in PAIRS:
            x, y = X[kx][:n], Y[ky][:n]
            f, Pxx = signal.welch(x, FS, nperseg=self.nper); _, Pyy = signal.welch(y, FS, nperseg=self.nper); _, Pxy = signal.csd(x, y, FS, nperseg=self.nper)
            s = self.S.setdefault(name, [0, 0, 0]); s[0] = s[0] + Pxx * nseg; s[1] = s[1] + Pyy * nseg; s[2] = s[2] + Pxy * nseg
        self.f = f; self.sec += n / FS
        return True

    def result(self, name):  # → f, H, 코히어런스
        Pxx, Pyy, Pxy = self.S[name]
        return self.f, Pxy / Pxx, np.abs(Pxy) ** 2 / (Pxx * Pyy)


def pair_record(path, a, tb):  # 한 기록 → (입력 채널, 출력 채널) 팔 베이스 좌표
    gp, ip = path.replace(".csv", ".gq7.csv"), path.replace(".csv", ".imu.csv")
    if not (os.path.exists(gp) and os.path.exists(ip)):
        print(f"{os.path.basename(path)}: .gq7.csv/.imu.csv 없음 — 건너뜀"); return None
    tx, ax, gx = load(gp); ty, ay, gy = load(ip); ty = ty - a.lag_ms / 1000.0  # 끝단이 늦는 만큼 당김
    t1, t2 = max(tx[0], ty[0]) + a.trim, min(tx[-1], ty[-1]) - a.trim
    if t2 - t1 < a.seg:
        print(f"{os.path.basename(path)}: 겹치는 구간 {t2 - t1:.1f} s — 건너뜀"); return None
    t = np.arange(t1, t2, 1 / FS); rs = lambda tt, X: np.column_stack([np.interp(t, tt, X[:, c]) for c in range(3)])
    Rb = axes_matrix(a.base_axes); ax, gx = rs(tx, ax) @ Rb.T, rs(tx, gx) @ Rb.T
    d = pd.read_csv(path); q = np.radians(d[[f"q{k}" for k in range(1, 7)]].values.mean(0))
    Re = tb.rot(q) if tb is not None else np.eye(3)
    ay, gy = rs(ty, ay) @ Re.T * a.gain, rs(ty, gy) @ Re.T * a.gain
    return channels(ax, gx), channels(ay, gy)


def report(acc, a):
    rows = []; curves = {}
    print(f"합친 길이 {acc.sec:.0f} s, 주파수 칸 {FS / acc.nper:.2f} Hz, 코히어런스 문턱 {a.coh}")
    print("쌍 | 봉우리 Hz  |H|  위상° | 2–3 Hz |H| | 5–10 Hz |H| 평균 | 코히어런스 통과 비율(2–20 Hz)")
    for name, _, _ in PAIRS:
        f, H, coh = acc.result(name); b = (f >= a.band[0]) & (f <= a.band[1]); ok = b & (coh >= a.coh); curves[name] = (f, H, coh)
        if not ok.any():
            print(f"{name} | 통과한 주파수 없음"); rows.append((name,) + (float("nan"),) * 6); continue
        i = np.where(ok)[0][np.argmax(np.abs(H[ok]))]; lo = ok & (f >= 2) & (f <= 3); mid = ok & (f >= 5) & (f <= 10)
        r = (name, float(f[i]), float(abs(H[i])), float(np.degrees(np.angle(H[i]))), float(np.abs(H[lo]).mean()) if lo.any() else float("nan"),
             float(np.abs(H[mid]).mean()) if mid.any() else float("nan"), float(ok.sum() / b.sum()))
        rows.append(r); print(f"{r[0]} | {r[1]:.2f}  {r[2]:.2f}  {r[3]:+.0f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]:.2f}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="") as fo:
            w = csv.writer(fo); w.writerow(["pair", "f", "H_abs", "H_deg", "coh"])
            for name, (f, H, coh) in curves.items():
                for k in np.where((f >= a.band[0]) & (f <= a.band[1]))[0]:
                    w.writerow([name, f"{f[k]:.3f}", f"{abs(H[k]):.4f}", f"{np.degrees(np.angle(H[k])):.1f}", f"{coh[k]:.3f}"])
        print("곡선:", a.out)
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
        for col, keys in enumerate((PAIRS[:3], PAIRS[3:6])):
            for name, _, _ in keys:
                f, H, coh = curves[name]; b = (f >= a.band[0]) & (f <= a.band[1]); Hm = np.where(coh >= a.coh, np.abs(H), np.nan)
                axs[0, col].semilogy(f[b], Hm[b], label=name); axs[1, col].plot(f[b], coh[b])
            axs[0, col].axhline(1, color="gray", lw=0.6); axs[0, col].legend(fontsize=8); axs[1, col].axhline(a.coh, color="gray", ls=":"); axs[1, col].set_xlabel("Hz")
            axs[0, col].set_title("translation (accel)" if col == 0 else "rotation (gyro)", fontsize=9)
        axs[0, 0].set_ylabel(f"|H| end-effector / base (coh >= {a.coh})"); axs[1, 0].set_ylabel("coherence")
        for x in axs.flat:
            x.grid(alpha=0.3, which="both")
        fig.tight_layout(); fig.savefig(a.png, dpi=130); print("그림:", a.png)
    return rows


def selftest(a):  # 받침 가진 1자유도(7 Hz, ζ 0.05) — 통과한 칸의 |H|가 해석값과 맞는지, 입력이 없는 대역(16 Hz 위)이 가려지는지
    rng = np.random.default_rng(1); n = int(120 * FS); w = 2 * np.pi * 7.0; z = 0.05
    num, den = [2 * z * w, w * w], [1, 2 * z * w, w * w]; sd = signal.cont2discrete((num, den), 1 / FS, method="bilinear")
    acc = Acc(int(a.seg * FS))
    for _ in range(3):
        x = signal.sosfilt(signal.butter(8, [1, 14], "bandpass", fs=FS, output="sos"), rng.normal(0, 1, n))
        y = signal.lfilter(sd[0].ravel(), sd[1], x) + rng.normal(0, 0.05, n); xm = x + rng.normal(0, 0.002, n)
        acc.add(channels(np.column_stack([xm] * 3), np.column_stack([xm] * 3)), channels(np.column_stack([y] * 3), np.column_stack([y] * 3)))
    r = report(acc, a)[0]; f, H, coh = acc.result("acc x→x")
    m = (f >= 2) & (f <= 12) & (coh >= a.coh) & (np.abs(f - 7) > 0.6)  # 봉우리 칸은 칸 폭 평균으로 낮게 나옴
    Ht = np.abs(signal.freqs(num, den, 2 * np.pi * f[m])[1]); err = float(np.max(np.abs(np.abs(H[m]) / Ht - 1)))
    hi = float(np.mean(coh[(f >= 17) & (f <= 20)] >= a.coh))
    ok = abs(r[1] - 7.0) < 0.3 and err < 0.1 and hi < 0.2
    print("selftest:", "통과" if ok else "실패", f"(봉우리 {r[1]:.2f} Hz, 해석값 대비 |H| 최대 오차 {err * 100:.1f}%, 17–20 Hz 통과 비율 {hi:.2f})")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--base-axes", nargs=3, default=["+x", "+y", "+z"], help="팔 베이스 x y z가 베이스 IMU의 어느 축인지")
    ap.add_argument("--gain", type=float, default=1.0, help="끝단(카메라) IMU 보정 배율(B-4)")
    ap.add_argument("--lag-ms", type=float, default=0.0, help="끝단 IMU가 베이스 IMU보다 늦는 시간(B-5)")
    ap.add_argument("--seg", type=float, default=4.0, help="Welch 구간[s] — 주파수 칸 = 1/seg")
    ap.add_argument("--trim", type=float, default=0.5, help="기록 앞뒤로 버리는 시간[s]")
    ap.add_argument("--band", type=float, nargs=2, default=[2.0, 20.0])
    ap.add_argument("--coh", type=float, default=0.8)
    ap.add_argument("--out", default=""); ap.add_argument("--png", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest(a))
    try:
        from imu_frame import ImuToBase
        from piper_kin import PiperKin
        tb = ImuToBase(PiperKin())
    except Exception as e:
        raise SystemExit(f"★ FK 없음({type(e).__name__}) — source install/setup.bash 뒤 실행(끝단을 베이스 좌표로 돌려야 함)")
    acc = Acc(int(a.seg * FS)); n = 0
    for p in a.paths:
        if p.endswith((".imu.csv", ".gq7.csv")):
            continue
        pr = pair_record(p, a, tb)
        if pr and acc.add(*pr):
            n += 1
    if n == 0:
        raise SystemExit("★ 쓸 수 있는 기록 없음")
    print(f"기록 {n}개"); report(acc, a)


if __name__ == "__main__":
    main()
