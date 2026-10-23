#!/usr/bin/env python3
"""D2-0 오프라인 확인 — vib_probe 기록의 IMU를 base 좌표로 바꿔 축별 5–10 Hz 성분, 자이로 피치 vs 엔코더(FK) link6 피치 속도 비교.

정지 중력 → base +z 각도(장착 회전 확인), 1 Hz 아래 자이로·엔코더 일치(변환·단위 확인), 5–8 Hz 결맞음·위상·크기비, 변형 속도(자이로 − 엔코더) rms
사용: source install/setup.bash; python3 tools/vib_d2_imu_check.py logs/vib/20261023_144309_slow_j23_poseA.csv [--pitch 30.75 --roll -88.25]
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "common_pkg")); sys.path.insert(0, HERE)
from imu_frame import MOUNT_PITCH_DEG, MOUNT_ROLL_DEG, ImuToBase  # noqa: E402
from piper_kin import PiperKin  # noqa: E402
import vib_analyze as V  # noqa: E402
from vib_d0_report import band_rms  # noqa: E402


def so3log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    return np.zeros(3) if th < 1e-9 else th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--pitch", type=float, default=MOUNT_PITCH_DEG)
    ap.add_argument("--roll", type=float, default=MOUNT_ROLL_DEG)
    a = ap.parse_args()
    k = PiperKin(); tb = ImuToBase(k, a.pitch, a.roll); fs = 200
    for f in a.csv:
        d = pd.read_csv(f).drop_duplicates("ts_h"); t0 = float(open(f.replace(".csv", ".t0")).read())
        i = pd.read_csv(f.replace(".csv", ".imu.csv")); i["t"] -= t0
        st = d[d.phase == "settle"]; acc = i[i.stream == "a"]
        g0 = acc[(acc.t > st.t.iloc[0]) & (acc.t < st.t.iloc[-1])][["x", "y", "z"]].mean().values
        gb = tb.rot(np.radians(st[[f"q{j}" for j in range(1, 7)]].mean().values)) @ g0
        print(f"\n## {os.path.basename(f)}\n정지 중력 → base {gb.round(3)}, 수직에서 {math.degrees(math.acos(gb[2] / np.linalg.norm(gb))):.2f}°")
        body = d[~d.phase.isin(["settle", "after", "estop"])]; ts = body.t.iloc[0]
        span = (ts + 4, ts + 14) if "_slow_" in f else (body.t.iloc[0], body.t.iloc[-1])
        res = {}
        for sname, fn in (("a", tb.accel), ("g", tb.gyro)):
            w = i[(i.stream == sname) & (i.t > span[0]) & (i.t < span[1])]
            Q = np.radians(np.column_stack([np.interp(w.t.values, d.t.values, d[f"q{j}"].values) for j in range(1, 7)]))
            v = np.array([fn(q, x) for q, x in zip(Q, w[["x", "y", "z"]].values)])
            tu = np.arange(w.t.values[0], w.t.values[-1], 1 / fs)
            res[sname] = (tu, np.column_stack([np.interp(tu, w.t.values, v[:, c]) for c in range(3)]))
        tu = res["g"][0]
        Q = np.radians(np.column_stack([np.interp(tu, d.t.values, d[f"q{j}"].values) for j in range(1, 7)])); Rs = [k.fk(q)[1] for q in Q]
        we = np.array([so3log(Rs[n + 1] @ Rs[n].T) * fs for n in range(len(Rs) - 1)] + [np.zeros(3)])
        print("| 신호 | 5–10 Hz rms x | y | z | 2–20 Hz 피크 x/y/z [Hz] |\n|---|---|---|---|---|")
        for nm, (t_, vu) in (("가속도 base [m/s²]", res["a"]), ("자이로 base [rad/s]", res["g"]), ("엔코더 link6 각속도 [rad/s]", (tu, we))):
            r = [band_rms(V.hp(vu[:, c], fs, 1.5), fs, 5, 10) for c in range(3)]; pk = []
            for c in range(3):
                fr, X = V.spectrum(V.hp(vu[:, c], fs, 1.5), fs); m = (fr > 2) & (fr < 20); pk.append(f"{fr[m][np.argmax(X[m])]:.1f}")
            print(f"| {nm} | {r[0]:.4f} | {r[1]:.4f} | {r[2]:.4f} | {'/'.join(pk)} |")
        gy = res["g"][1][:, 1]; G = V.hp(gy, fs, 1.5); E = V.hp(we[:, 1], fs, 1.5)
        n = len(tu); fq = np.fft.rfftfreq(n, 1 / fs); FG = np.fft.rfft(G * np.hanning(n)); FE = np.fft.rfft(E * np.hanning(n))
        m = (fq > 5) & (fq < 8); cr = (FG[m] * np.conj(FE[m])).sum()
        coh = abs(cr) ** 2 / ((abs(FG[m]) ** 2).sum() * (abs(FE[m]) ** 2).sum())
        gl = V.lp(gy, fs, 1.0); el = V.lp(we[:, 1], fs, 1.0)
        print(f"피치(base y): 1 Hz 아래 상관 {np.corrcoef(gl, el)[0, 1]:.3f}, rms 자이로 {gl.std():.4f} / 엔코더 {el.std():.4f} rad/s | "
              f"5–8 Hz 결맞음 {coh:.2f}, 위상 {np.degrees(np.angle(cr)):+.0f}°(자이로 앞서면 +), 크기비 {np.sqrt((abs(FG[m]) ** 2).sum() / (abs(FE[m]) ** 2).sum()):.2f} | "
              f"5–10 Hz 변형 속도(자이로 − 엔코더) {band_rms(G - E, fs, 5, 10):.4f} rad/s")


if __name__ == "__main__":
    main()
