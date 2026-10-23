#!/usr/bin/env python3
"""D2-1 오프라인 — 관절별 변형 비중(가속도+자이로) 추정, 자이로 링크 속도 추정기(link_est) 검증.

shares  : 툭·계단 뒤 4–10 Hz에서 [ω_res; v_res] = [J_ω; J_v]·δ̇ 를 최소제곱으로 풀어 관절별 변형 속도 rms → W
          ω_res = 자이로(base) − J_ω q̇_m,  v_res = ∫가속도(base, 중력 제거) − J_v q̇_m (IMU 원점)
validate: a′ 등에서 추정 q̇̂로 예측한 IMU 원점 속도 J_v q̇̂ vs 가속도 적분(추정에 안 쓴 센서) — 모터만(J_v q̇_m)과 비교
사용: source install/setup.bash; python3 tools/vib_d2_estimator.py shares logs/vib/*_tap_pose*.csv
      python3 tools/vib_d2_estimator.py validate logs/vib/20261023_144309_slow_j23_poseA.csv --w 0 1 1 0 1 0
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "common_pkg")); sys.path.insert(0, HERE)
from imu_frame import ImuToBase  # noqa: E402
from link_est import LinkVelEst, jacobians  # noqa: E402
from piper_kin import PiperKin  # noqa: E402
import vib_analyze as V  # noqa: E402
from vib_d0_report import events  # noqa: E402

FS = 200.0
DEV_TIME = True  # IMU 시각을 장치 타임스탬프로 복원


def band(x, lo, hi, integrate=False):  # 주파수 영역 대역통과(+적분) — 영위상
    n = len(x); X = np.fft.rfft(x - x.mean()); f = np.fft.rfftfreq(n, 1 / FS)
    X[(f < lo) | (f > hi)] = 0
    if integrate:
        X[1:] = X[1:] / (2j * np.pi * f[1:]); X[0] = 0
    return np.fft.irfft(X, n)


def load_aligned(f, t1, t2, kin, tb):  # 200 Hz 등간격: q_m, q̇_m(엔코더), 자이로·가속도(base)
    d = pd.read_csv(f)
    if "ts_h" in d:
        d = d.drop_duplicates("ts_h")
    t0 = float(open(f.replace(".csv", ".t0")).read()); i = pd.read_csv(f.replace(".csv", ".imu.csv"))
    if DEV_TIME:  # PC 수신 시각은 몰려 와서 중앙 23 ms·95% 46 ms 늦음 → 장치 시각 + 최소 오프셋(가장 빨리 온 표본 기준)
        off = i.t.values - i.t_dev.values / 1000; ok = np.abs(off - np.median(off)) < 1.0
        i = i[ok].copy(); i["t"] = i.t_dev / 1000 + np.percentile(off[ok], 1)
    i["t"] -= t0
    tu = np.arange(t1, t2, 1 / FS)
    Q = np.radians(np.column_stack([np.interp(tu, d.t.values, d[f"q{j}"].values) for j in range(1, 7)]))
    QD = np.column_stack([np.interp(tu, d.t.values, d[f"qd{j}"].values) for j in range(1, 7)])
    out = {}
    for s, fn in (("g", tb.gyro), ("a", tb.accel)):
        w = i[(i.stream == s) & (i.t > t1 - 0.2) & (i.t < t2 + 0.2)]
        Qi = np.radians(np.column_stack([np.interp(w.t.values, d.t.values, d[f"q{j}"].values) for j in range(1, 7)]))
        v = np.array([fn(q, x) for q, x in zip(Qi, w[["x", "y", "z"]].values)])
        out[s] = np.column_stack([np.interp(tu, w.t.values, v[:, c]) for c in range(3)])
    return tu, Q, QD, out["g"], out["a"]


def residuals(kin, tb, Q, QD, G, A, lo=4.0, hi=10.0):
    Jw_all, Jv_all = zip(*[jacobians(kin, q, tb.R_l6) for q in Q])
    Jw_all = np.array(Jw_all); Jv_all = np.array(Jv_all)
    wm = np.einsum("nij,nj->ni", Jw_all, QD); vm = np.einsum("nij,nj->ni", Jv_all, QD)
    w_res = np.column_stack([band(G[:, c] - wm[:, c], lo, hi) for c in range(3)])
    v_imu = np.column_stack([band(A[:, c], lo, hi, integrate=True) for c in range(3)])
    v_res = v_imu - np.column_stack([band(vm[:, c], lo, hi) for c in range(3)])
    return Jw_all, Jv_all, w_res, v_res, v_imu


def cmd_shares(a, kin, tb):
    rows = []
    for f in a.csv:
        if f.endswith((".imu.csv", ".tdc.csv")) or "_dry" in f or not os.path.exists(f.replace(".csv", ".imu.csv")) or any(x in f for x in a.exclude):
            continue
        from vib_d0_report import load
        d, i = load(f)
        for t_from, _, lab in events(f, d, i):
            tu, Q, QD, G, A = load_aligned(f, t_from - 0.3, t_from + 1.5, kin, tb)
            Jw, Jv, w_res, v_res, _ = residuals(kin, tb, Q, QD, G, A)
            sel = (tu > t_from) & (tu < t_from + 1.2)
            dd = np.array([np.linalg.lstsq(np.vstack([Jw[k], Jv[k]]), np.concatenate([w_res[k], v_res[k]]), rcond=1e-3)[0] for k in np.where(sel)[0]])
            r = np.sqrt((dd ** 2).mean(axis=0)); fit = np.vstack([Jw[sel][0], Jv[sel][0]])
            rows.append((os.path.basename(f)[16:-4], lab, r, np.linalg.cond(fit)))
    print("| 시행 | 가진 | δ̇ rms J1 | J2 | J3 | J4 | J5 | J6 [rad/s] | 자코비안 조건수 |\n|---|---|---|---|---|---|---|---|---|")
    allr = []
    for name, lab, r, cn in rows:
        print(f"| {name} | {lab} | " + " | ".join(f"{x:.4f}" for x in r) + f" | {cn:.0f} |"); allr.append(r / r.sum())
    m = np.mean(allr, axis=0); print("\n평균 비중(시행마다 합 1로 정규화): " + ", ".join(f"J{j + 1} {m[j]:.2f}" for j in range(6)))


def cmd_validate(a, kin, tb):
    for f in a.csv:
        d = pd.read_csv(f); body = d[~d.phase.isin(["settle", "after", "estop", "measure"])]
        ts = body.t.iloc[0]; t1, t2 = (ts + 4, ts + 14) if "_slow_" in f else (body.t.iloc[0], body.t.iloc[-1])
        tu, Q, QD, G, A = load_aligned(f, t1, t2, kin, tb)
        G = G - np.array([band(G[:, c], 0, 1.5) for c in range(3)]).T  # 1.5 Hz 고역통과 = 바이어스·저주파 오차 제거
        est = LinkVelEst(kin, tb.R_l6, w=a.w, axes=a.axes)
        WM = np.einsum("nij,nj->ni", np.array([jacobians(kin, q, tb.R_l6)[0] for q in Q]), QD)
        G = G + np.array([band(WM[:, c], 0, 1.5) for c in range(3)]).T  # 저주파 궤적 성분은 엔코더 값으로 채움
        QH = np.array([est.step(q, qd, g, 1 / FS)[1] for q, qd, g in zip(Q, QD, G)])
        Jv = np.array([jacobians(kin, q, tb.R_l6)[1] for q in Q])
        v_imu = np.column_stack([band(A[:, c], 4, 10, integrate=True) for c in range(3)])
        v_m = np.column_stack([band(x, 4, 10) for x in np.einsum("nij,nj->ni", Jv, QD).T])
        v_h = np.column_stack([band(x, 4, 10) for x in np.einsum("nij,nj->ni", Jv, QH).T])
        print(f"\n## {os.path.basename(f)} — W {a.w}, 자이로 축 {a.axes}\n| 축 | 가속도 적분 rms [mm/s] | 모터만 오차 rms | 추정 오차 rms | 오차 비(추정/모터) | 추정–측정 결맞음 |\n|---|---|---|---|---|---|")
        for c, nm in enumerate("xyz"):
            e_m = v_m[:, c] - v_imu[:, c]; e_h = v_h[:, c] - v_imu[:, c]
            n = len(tu); F1 = np.fft.rfft(v_h[:, c] * np.hanning(n)); F2 = np.fft.rfft(v_imu[:, c] * np.hanning(n)); fq = np.fft.rfftfreq(n, 1 / FS); m = (fq > 5) & (fq < 8)
            coh = abs((F1[m] * np.conj(F2[m])).sum()) ** 2 / ((abs(F1[m]) ** 2).sum() * (abs(F2[m]) ** 2).sum() + 1e-30)
            print(f"| {nm} | {v_imu[:, c].std() * 1000:.2f} | {e_m.std() * 1000:.2f} | {e_h.std() * 1000:.2f} | {e_h.std() / e_m.std():.2f} | {coh:.2f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["shares", "validate"])
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--exclude", nargs="*", default=["poseA_x2"])
    ap.add_argument("--w", type=float, nargs=6, default=[1, 1, 1, 1, 1, 1])
    ap.add_argument("--axes", type=int, nargs="+", default=[0, 1, 2], help="쓸 자이로 base 축 — 피치만이면 1")
    ap.add_argument("--host-time", action="store_true", help="IMU를 PC 수신 시각 그대로(비교용)")
    a = ap.parse_args()
    global DEV_TIME
    DEV_TIME = not a.host_time
    kin = PiperKin(); tb = ImuToBase(kin)
    cmd_shares(a, kin, tb) if a.cmd == "shares" else cmd_validate(a, kin, tb)


if __name__ == "__main__":
    main()
