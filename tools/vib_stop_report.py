#!/usr/bin/env python3
"""정지 창 분석(단계 C·D·E) — vib_probe 기록(+ .imu.csv)에서 "엔코더는 멈췄는데 끝단은 흔들린다"를 정지마다 한 줄로.

정지 시점: 명령 정지 t_cmd(ramp→hold 전환, 다른 단계는 qref가 멈춘 때) 뒤, 엔코더 속도(위치 미분, 55 ms 평활)가 문턱 아래로 0.1 s 이어진 첫 시각 t_enc.
창: t_enc부터 --win s. 자이로는 카메라 축, 주축은 창 안 2–20 Hz 에너지가 가장 큰 축(--axis로 고정 가능).
지표: 첫 봉우리(20 Hz 저역, 창 끝 0.5 s 평균을 뺀 값), 5–10 Hz rms(자이로 주축·3축 크기·각가속도, 가속도 주축·3축 크기),
      모드 주파수·감쇠비(감쇠 사인 맞춤, 로그 감소율 병기), 정착 시간(2–20 Hz 포락선이 잡음 바닥 2배 아래로 0.3 s 이어지는 첫 시각, 유지 구간 끝까지 봄), 엔코더 몫(같은 창 엔코더 각속도 rms / 끝단).
잡음 바닥: --floor[rad/s] 없으면 같은 기록의 움직이기 전 정지 구간 포락선 중앙값.
사용: python3 tools/vib_stop_report.py logs/vib/*_ramp_*.csv [--out stops.csv] [--plot docs/img/p_x.png] [--all]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import optimize, signal

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "common_pkg"))
from vib_analyze import decay  # noqa: E402

FS = 200.0
COLS = ["file", "fw", "sdk", "joint", "speed_dps", "dist_deg", "stop", "tilt_deg", "v_enc_dps", "t_stop_ms", "axis", "peak1_rads", "t_peak1_ms", "max_rads",
        "gyro_rms", "gyro_vec_rms", "angacc_rms", "acc_axis", "acc_rms", "acc_vec_rms", "f_fit", "zeta_fit", "r2", "f_fft", "zeta_log", "n_cyc",
        "settle_s", "floor_rads", "enc_rms", "enc_share"]


def load_imu(path, t0, lag=0.0):  # → {"a": (t, xyz), "g": (t, xyz)} — 장치 시각 + 수신 시각 최소 지연으로 기록 시각 축에
    d = pd.read_csv(path); off = d.t.values - d.t_dev.values / 1000; ok = np.abs(off - np.median(off)) < 1.0
    d = d[ok]; tt = d.t_dev.values / 1000 + np.percentile(off[ok], 1) - t0 - lag; out = {}
    for k in "ag":
        m = (d.stream == k).values; out[k] = (tt[m], d[["x", "y", "z"]].values[m])
    return out


def resamp(t, X, t1, t2):  # 등간격 FS로
    tu = np.arange(t1, t2, 1 / FS)
    return tu, np.column_stack([np.interp(tu, t, X[:, c]) for c in range(X.shape[1])])


def band(x, lo, hi, deriv=False):  # 창 안 FFT 벽돌 필터(vib_d3_report와 같은 방식) — 선형 추세 제거 뒤
    x = signal.detrend(x); X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / FS); X[(f < lo) | (f > hi)] = 0
    if deriv:
        X = X * 2j * np.pi * f
    return np.fft.irfft(X, len(x))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def fit_decay(t, y):  # y ≈ A e^{−ζωt} cos(ω_d t + φ) — (f, ζ, R²)
    X = np.abs(np.fft.rfft(y * np.hanning(len(y)), 8 * len(y))); f = np.fft.rfftfreq(8 * len(y), 1 / FS); b = (f >= 2) & (f <= 20)
    f0 = float(f[b][np.argmax(X[b])]); best = None

    def model(t, A, z, fn, ph):
        w = 2 * np.pi * fn
        return A * np.exp(-z * w * t) * np.cos(w * np.sqrt(max(1 - z * z, 1e-6)) * t + ph)
    for ph in (0.0, np.pi / 2, np.pi, -np.pi / 2):
        try:
            p, _ = optimize.curve_fit(model, t, y, p0=[np.abs(y).max(), 0.05, f0, ph], bounds=([0, 0.001, 2, -2 * np.pi], [np.inf, 0.7, 20, 2 * np.pi]), maxfev=4000)
        except (RuntimeError, ValueError):
            continue
        r2 = 1 - np.sum((y - model(t, *p)) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-18)
        if best is None or r2 > best[2]:
            best = (float(p[2]), float(p[1]), float(r2))
    return (best or (float("nan"),) * 3) + (f0,)


def stops(d, meta, all_stops):  # → [(이름, t_cmd, 관절, 창 끝 한계)]
    t = d.t.values; ph = d.phase.values
    if "ramp" in set(ph):
        j = int(meta.get("args", {}).get("joint", 0)) or int(np.argmax([np.ptp(d[f"qref{k}"].values) for k in range(1, 7)])) + 1
        out = []; names = ("hold", "bhold") if all_stops else ("hold",)
        for i in range(1, len(ph)):
            if ph[i] in names and ph[i - 1] != ph[i]:
                e = i
                while e < len(ph) and ph[e] in (ph[i], "after"):
                    e += 1
                out.append((ph[i], t[i], j, t[e - 1]))
        return out
    j = int(np.argmax([np.ptp(d[f"qref{k}"].values) for k in range(1, 7)])) + 1
    r = d[f"qref{j}"].values; ch = np.where(np.abs(np.diff(r)) > 1e-4)[0] + 1; out = []
    for n, i in enumerate(ch):  # qref가 바뀐 뒤 1 s 넘게 그대로면 정지 하나
        nxt = t[ch[n + 1]] if n + 1 < len(ch) else t[-1]
        if nxt - t[i] > 1.0:
            out.append(("stop", t[i], j, nxt))
    return out


def enc_rate_cam(d, kin_tb, tu):  # 엔코더로 계산한 끝단 각속도(카메라 축 3) — FK 없으면 None
    if kin_tb is None:
        return None
    from link_est import jacobians
    kin, tb = kin_tb
    Q = np.radians(np.column_stack([np.interp(tu, d.t.values, d[f"q{k}"].values) for k in range(1, 7)]))
    Qd = signal.savgol_filter(Q, 11, 2, deriv=1, delta=1 / FS, axis=0); qm = Q[len(Q) // 2]
    return Qd @ (tb.rot(qm).T @ jacobians(kin, qm, tb.R_l6)[0]).T


def analyze(path, a, kin_tb):
    d = pd.read_csv(path); t0 = float(open(path.replace(".csv", ".t0")).read()); mp = path.replace(".csv", ".meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}; ip = path.replace(".csv", ".imu.csv")
    if not os.path.exists(ip):
        print(f"{os.path.basename(path)}: IMU 기록 없음 — 건너뜀"); return [], None
    imu = load_imu(ip, t0, a.imu_lag / 1000.0); t = d.t.values; rows = []; fig = None
    mv = np.where(np.abs(np.diff(d[[f"qref{k}" for k in range(1, 7)]].values, axis=0)).max(1) > 1e-4)[0]
    t_still = (t[0] + 0.3, t[mv[0]] - 0.1 if len(mv) else t[-1])  # 움직이기 전 정지 구간 — 기울기·잡음 바닥
    args = meta.get("args", {}); sl = stops(d, meta, a.all)
    if not sl and not len(mv):  # 툭 기록(qref 고정) — 2–20 Hz 자이로 크기가 가장 큰 시각을 정지 시각으로 둠
        tg0, G0 = resamp(*imu["g"], t[0] + 1.0, min(t[-1], imu["g"][0][-1]))
        t_tap = float(tg0[np.argmax(np.linalg.norm(np.column_stack([band(G0[:, c], 2, 20) for c in range(3)]), axis=1))])
        sl = [("tap", t_tap, 2, t[-1])]; t_still = (t[0] + 0.3, t_tap - 0.3)
    ga = imu["a"]; w = (ga[0] > t_still[0]) & (ga[0] < t_still[1])
    g = ga[1][w].mean(0) if w.sum() > 20 else np.full(3, np.nan); tilt = float(np.degrees(np.arctan2(g[1], -g[2])))
    for name, t_cmd, j, t_lim in sl:
        tu = np.arange(max(t[0], t_cmd - 1.5), min(t[-1], t_cmd + a.win + 1.0), 1 / FS); q = np.interp(tu, t, d[f"q{j}"].values)
        v = signal.savgol_filter(q, 11, 2, deriv=1, delta=1 / FS)  # 엔코더 속도[deg/s] — 속도 피드백은 잡음(0.1 rad/s)이 커서 위치 미분
        pre = (tu > t_cmd - 0.5) & (tu < t_cmd + 0.1); v_run = float(v[pre][np.argmax(np.abs(v[pre]))]) if pre.sum() > 5 else 0.0  # 정지 직전 실제로 도달한 최고 속도
        thr = max(0.05 * abs(v_run), a.v_thr); n = int(0.1 * FS); below = np.abs(v) < thr; i0 = int(np.searchsorted(tu, t_cmd - 0.05)); t_enc = t_cmd + 0.1 if name == "tap" else None
        for i in range(i0, len(tu) - n if t_enc is None else 0):
            if below[i:i + n].all():
                t_enc = tu[i]; break
        if t_enc is None:
            print(f"{os.path.basename(path)} {name}: 엔코더 정지 시점을 찾지 못함"); continue
        t2 = min(t_enc + a.win, t_lim, imu["g"][0][-1])
        if t2 - t_enc < 1.0:
            print(f"{os.path.basename(path)} {name}: 정지 창이 {t2 - t_enc:.2f} s뿐 — 건너뜀"); continue
        tg, G = resamp(*imu["g"], t_enc, t2); ta, A = resamp(*imu["a"], t_enc, t2)
        ax = "xyz".index(a.axis) if a.axis in "xyz" else int(np.argmax([rms(band(G[:, c], 2, 20)) for c in range(3)]))
        aax = int(np.argmax([rms(band(A[:, c], 2, 20)) for c in range(3)]))
        bias = G[tg > t2 - 0.5, ax].mean(); sos = signal.butter(4, 20, fs=FS, output="sos"); gl = signal.sosfiltfilt(sos, G[:, ax] - bias)
        ext = np.where((np.diff(np.sign(np.diff(gl))) != 0))[0] + 1; big = [i for i in ext if abs(gl[i]) >= 0.3 * np.abs(gl).max()]
        i1 = big[0] if big else int(np.argmax(np.abs(gl)))
        gb = band(G[:, ax], 2, 20); tt = tg - tg[0]; fw = tt >= tt[i1]
        ff, zf, r2, f_fft = fit_decay(tt[fw][:int(1.2 * FS)] - tt[i1], gb[fw][:int(1.2 * FS)])
        zl, ncyc = decay(gb[i1:] * np.sign(gb[i1]), FS, f_fft)
        sb = signal.butter(2, [2, 20], "bandpass", fs=FS, output="sos"); k = int(0.25 * FS)
        envf = lambda x: np.convolve(np.abs(signal.hilbert(signal.sosfiltfilt(sb, x))), np.ones(k) / k, "same")  # 2–20 Hz 포락선, 0.25 s 평활
        if a.floor > 0:
            floor = a.floor
        elif t_still[1] - t_still[0] > 1.2:  # 같은 기록의 움직이기 전 정지 구간
            floor = float(np.median(envf(resamp(*imu["g"], *t_still)[1][:, ax])[k:-k]))
        else:
            floor = float("nan")
        te, Ge = resamp(*imu["g"], t_enc - 1.0, min(t_lim, imu["g"][0][-1])); env = envf(Ge[:, ax]); m = te >= t_enc; env, te = env[m][:-k], te[m][:-k] - t_enc
        settle = float("nan")
        if np.isfinite(floor) and len(env) > k:  # 포락선이 바닥 2배 아래로 0.3 s 이어지는 첫 시각(유지 구간 끝까지 봄)
            lowrun = np.convolve((env < 2 * floor).astype(float), np.ones(int(0.3 * FS)), "valid") >= int(0.3 * FS) - 0.5
            settle = float(te[np.argmax(lowrun)]) if lowrun.any() else float("inf")
        E = enc_rate_cam(d, kin_tb, tg)
        e5 = rms(band(E[:, ax], 5, 10)) if E is not None else rms(band(np.radians(signal.savgol_filter(np.interp(tg, t, d[f"q{j}"].values), 11, 2, deriv=1, delta=1 / FS)), 5, 10))
        g5 = rms(band(G[:, ax], 5, 10))
        row = dict(file=os.path.basename(path), fw=meta.get("fw", ""), sdk=meta.get("sdk", ""), joint=j, speed_dps=args.get("speed", ""), dist_deg=args.get("dist", ""),
                   stop=name, tilt_deg=round(tilt, 2), v_enc_dps=round(v_run, 2), t_stop_ms=round((t_enc - t_cmd) * 1000), axis="xyz"[ax],
                   peak1_rads=round(float(gl[i1]), 4), t_peak1_ms=round(float(tt[i1]) * 1000), max_rads=round(float(np.abs(gl).max()), 4),
                   gyro_rms=round(g5, 5), gyro_vec_rms=round(float(np.sqrt(sum(rms(band(G[:, c], 5, 10)) ** 2 for c in range(3)))), 5),
                   angacc_rms=round(rms(band(G[:, ax], 5, 10, deriv=True)), 4), acc_axis="xyz"[aax], acc_rms=round(rms(band(A[:, aax], 5, 10)), 4),
                   acc_vec_rms=round(float(np.sqrt(sum(rms(band(A[:, c], 5, 10)) ** 2 for c in range(3)))), 4),
                   f_fit=round(ff, 2), zeta_fit=round(zf, 3), r2=round(r2, 2), f_fft=round(f_fft, 2), zeta_log=round(zl, 3), n_cyc=ncyc,
                   settle_s=round(settle, 2), floor_rads=round(floor, 5), enc_rms=round(e5, 5), enc_share=round(e5 / g5, 3) if g5 > 0 else float("nan"))
        rows.append(row)
        if fig is None:  # 현상 그림용: 첫 정지의 엔코더 속도와 끝단 각속도
            tp, Gp = resamp(*imu["g"], tu[0], min(tu[-1], imu["g"][0][-1])); gp = signal.sosfiltfilt(sos, Gp[:, ax] - bias)
            vp = np.interp(tp, tu, v); mrun = (tp > t_cmd - 0.5) & (tp < t_cmd + 0.1)
            sgn = np.sign(np.dot(gp[mrun], vp[mrun])) or 1.0  # 자이로 축 부호를 관절 방향에 맞춤
            fig = (tp - t_enc, vp, np.degrees(gp) * sgn, t_cmd - t_enc, t2 - t_enc, row)
    return rows, fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--win", type=float, default=2.0, help="정지 창 길이[s]")
    ap.add_argument("--v-thr", type=float, default=0.3, help="엔코더 정지 판정 속도 하한[deg/s] (등속의 5%%와 큰 쪽)")
    ap.add_argument("--axis", default="auto", help="자이로 주축 x/y/z 또는 auto")
    ap.add_argument("--floor", type=float, default=0.0, help="잡음 바닥 포락선[rad/s] (B-3 값) — 0이면 기록 안 정지 구간")
    ap.add_argument("--imu-lag", type=float, default=0.0, help="IMU가 엔코더보다 늦는 시간[ms] (B-5 값)")
    ap.add_argument("--all", action="store_true", help="왕복의 돌아간 쪽 정지(bhold)도 분석")
    ap.add_argument("--out", default="", help="결과 CSV")
    ap.add_argument("--plot", default="", help="현상 그림 PNG(첫 기록의 첫 정지)")
    a = ap.parse_args()
    try:
        from imu_frame import ImuToBase
        from piper_kin import PiperKin
        kin = PiperKin(); kin_tb = (kin, ImuToBase(kin))
    except Exception as e:
        kin_tb = None; print(f"※ FK 없음({type(e).__name__}) — 엔코더 몫은 움직인 관절 속도만으로 계산(source install/setup.bash 뒤 실행 권장)")
    rows = []; fig = None
    for p in a.paths:
        if p.endswith((".imu.csv", ".gq7.csv")):
            continue
        r, f = analyze(p, a, kin_tb); rows += r; fig = fig or f
    print("기록 | 관절 속도 거리 | 기울기 | 도달 속도[°/s] 정지 지연[ms] | 축 첫 봉우리[rad/s]@ms 최대 | 5–10 Hz 자이로·각가속도·가속도 | f[Hz] ζ(R²) | 정착[s] | 엔코더 몫")
    for r in rows:
        print(f"{r['file'][:15]} {r['stop']} | J{r['joint']} {r['speed_dps']} {r['dist_deg']} | {r['tilt_deg']:.2f}° | {r['v_enc_dps']:+.1f} {r['t_stop_ms']} | "
              f"{r['axis']} {r['peak1_rads']:+.4f}@{r['t_peak1_ms']} {r['max_rads']:.4f} | {r['gyro_rms']:.4f} {r['angacc_rms']:.3f} {r['acc_rms']:.3f}({r['acc_axis']}) | "
              f"{r['f_fit']:.2f} {r['zeta_fit']:.3f}({r['r2']:.2f}) | {r['settle_s']} | {r['enc_share']}")
    if a.out and rows:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(rows)
        print("표:", a.out)
    if a.plot and fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        tt, v, gdeg, tc, tw, r = fig
        fg, x = plt.subplots(figsize=(8, 4))
        x.plot(tt, v, label="encoder joint speed", color="C0"); x.plot(tt, gdeg, label=f"end-effector gyro {r['axis']} (20 Hz LP)", color="C3", lw=1)
        x.axvline(tc, color="gray", ls=":", label="command stop"); x.axvline(0, color="k", ls="--", lw=0.8, label="encoder stop"); x.axvspan(0, tw, color="C1", alpha=0.08)
        x.set_xlabel("time from encoder stop [s]"); x.set_ylabel("angular velocity [deg/s]"); x.grid(alpha=0.3); x.legend(fontsize=8)
        x.set_title(f"J{r['joint']} {r['speed_dps']} deg/s, {r['dist_deg']} deg  (fw {r['fw']})", fontsize=9)
        fg.tight_layout(); fg.savefig(a.plot, dpi=130); print("그림:", a.plot)


if __name__ == "__main__":
    main()
