#!/usr/bin/env python3
"""정지 창 분석(단계 C·D·E) — vib_probe 기록(+ .imu.csv)에서 "엔코더는 멈췄는데 끝단은 흔들린다"를 정지마다 한 줄로.

정지 시점: 명령 정지 t_cmd(ramp→hold 전환, 다른 단계는 qref가 멈춘 때) 뒤, 엔코더 속도(위치 미분, 55 ms 평활)가 문턱 아래로 처음 내려간 시각 t_enc(9절 11/17 정의 (나)).
          --sustained면 문턱 아래로 0.1 s 이어진 첫 시각(11/16 정의 (가) — 멈춘 직후 되돌아가는 시행에서 85 ms쯤 늦게 잡힘, FW191_C_LOG 2.3절).
창: t_enc부터 --win s. 자이로는 카메라 축, 주축은 창 안 2–20 Hz 에너지가 가장 큰 축(--axis로 고정 가능).
MIT 기록(mit_probe --stage ramp, logs/mit)도 같은 식으로 읽는다: ctrl 열(pos / pd / tde / asmc16 / asmc16_j123), jump 열(창이 MIT 순간 튐의 제외 구간 −0.1–+1.2 s와 겹치면 1 — 시행 집계에서 버림).
열: shape(.meta.json의 성형 종류), rep(왕복 회차), arrive(도착 방향, +면 관절각이 커지며 도착 — J2는 내림), stop_deg(멈춘 각), gap_*(창 안 자이로 표본 빈틈 수·시간 비율·최대 간격).
집계: --trial-out 시행(기록)별 중앙값, --judge 성형 없음 대비 비와 판정(자이로·가속도 둘 다 0.83 이하 감소, 하나라도 1.2 이상 증가, 시행 범위 겹치면 경향).
지표: 창 최대(진폭 지표, 20 Hz 저역·창 끝 0.5 s 평균을 뺀 값. 첫 봉우리는 참고), 5–10 Hz rms(자이로 주축·3축 크기·각가속도, 가속도 주축·3축 크기),
      모드 주파수·감쇠비(감쇠 사인 맞춤, 로그 감소율 병기 — 통계는 R² 0.8 이상 시행만), 정착 시간(2–20 Hz 포락선이 잡음 바닥 2배 아래로 0.3 s 이어지는 첫 시각, 유지 구간 끝까지 봄),
      명령 정지부터 정착까지(settle_cmd_s = 정지 지연 + 정착 시간), 엔코더 몫(같은 창 엔코더 각속도 rms / 끝단), 명령 이동 시간(t_move_s = 거리 / 속도).
잡음 바닥: --floor[rad/s] 값 하나로 고정(기본 0.0099 = C-1 90시행 기록별 바닥의 중앙값, 카메라 자리 자이로). --floor 0이면 같은 기록의 움직이기 전 정지 구간 포락선 중앙값(floor_own_rads 열에는 항상 적음).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "common_pkg"))
from vib_analyze import decay  # noqa: E402

FS = 200.0
COLS = ["file", "fw", "sdk", "ctrl", "joint", "speed_dps", "dist_deg", "shape", "t_move_s", "stop", "rep", "arrive", "stop_deg", "tilt_deg", "v_enc_dps", "t_stop_ms", "axis", "peak1_rads", "t_peak1_ms", "max_rads",
        "gyro_rms", "gyro_vec_rms", "angacc_rms", "acc_axis", "acc_rms", "acc_vec_rms", "f_fit", "zeta_fit", "r2", "f_fft", "zeta_log", "n_cyc",
        "settle_s", "settle_cmd_s", "floor_rads", "floor_own_rads", "enc_rms", "enc_share", "gap_n", "gap_frac", "gap_max_ms", "jump"]
UPDOWN = {"+": "내림", "-": "올림"}  # J2: 관절각이 커지며 도착(+)이 내림


def load_imu(path, t0, lag=0.0):  # → {"a": (t, xyz), "g": (t, xyz)} — 장치 시각 + 수신 시각 최소 지연으로 기록 시각 축에
    d = pd.read_csv(path); off = d.t.values - d.t_dev.values / 1000; ok = np.abs(off - np.median(off)) < 1.0
    d = d[ok]; late = d.t.values > d.t.values[0] + 4.0  # D455는 시작 3.5 s 동안 장치 시각이 45–65 ms 흔들림(FW191_B 1절) — 그 뒤 표본으로 맞춤
    tt = d.t_dev.values / 1000 + np.percentile(off[ok][late] if late.sum() > 200 else off[ok], 1) - t0 - lag; out = {}
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


def gaps(t, t1, t2):  # 창 안 표본 빈틈(간격이 기록 중앙값의 1.5배 초과) → (개수, 빈틈이 차지하는 시간 비율, 최대 간격 ms)
    dt = np.diff(t); med = float(np.median(dt)); m = (t[1:] > t1) & (t[:-1] < t2); g = dt[m][dt[m] > 1.5 * med]
    return len(g), round(float(g.sum()) / max(t2 - t1, 1e-9), 3), round(float(dt[m].max()) * 1000, 1) if m.any() else float("nan")


def fit_decay(t, y):  # y ≈ A e^{−ζωt} cos(ω_d t + φ) — (f, ζ, R²)
    if len(y) < int(0.3 * FS):  # 첫 봉우리가 창 끝에 있으면(튐이 창 끝에 걸림) 맞출 구간이 없음
        return (float("nan"),) * 4
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
    meta = json.load(open(mp)) if os.path.exists(mp) else {}; ip = path.replace(".csv", ".imu.csv"); jw = []
    if "seg" in d.columns:  # mit_probe --stage ramp 기록: MIT 구간의 구간 이름(seg)을 phase로 쓰고, MIT 순간 튐의 제외 구간(−0.1–+1.2 s, 9절 11/17 22:04)을 구해 둠
        from mit_jump_report import events
        tm0 = d.t.values[d.phase.values == "mit"][0]; jw = [(tm0 + e["t_mit"] - 0.1, tm0 + e["t_mit"] + 1.2) for e in events(path, 0.5)[0]]
        d["phase"] = np.where(d.phase == "mit", d.seg, d.phase)
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
    nrep = {}
    for name, t_cmd, j, t_lim in sl:
        nrep[name] = nrep.get(name, 0) + 1
        tu = np.arange(max(t[0], t_cmd - 1.5), min(t[-1], t_cmd + a.win + 1.0), 1 / FS); q = np.interp(tu, t, d[f"q{j}"].values)
        v = signal.savgol_filter(q, 11, 2, deriv=1, delta=1 / FS)  # 엔코더 속도[deg/s] — 속도 피드백은 잡음(0.1 rad/s)이 커서 위치 미분
        pre = (tu > t_cmd - 0.5) & (tu < t_cmd + 0.1); v_run = float(v[pre][np.argmax(np.abs(v[pre]))]) if pre.sum() > 5 else 0.0  # 정지 직전 실제로 도달한 최고 속도
        thr = max(0.05 * abs(v_run), a.v_thr); n = int(0.1 * FS); below = np.abs(v) < thr; i0 = int(np.searchsorted(tu, t_cmd - 0.05)); t_enc = t_cmd + 0.1 if name == "tap" else None
        if not a.sustained and t_enc is None:  # 정의 (나): 문턱 아래로 처음 내려간 시각 — 아직 안 움직인 상태면(계단 기록) 움직이기 시작한 뒤부터 찾음
            n = 1
            while i0 < len(tu) - 1 and below[i0]:
                i0 += 1
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
        floor_own = float(np.median(envf(resamp(*imu["g"], *t_still)[1][:, ax])[k:-k])) if t_still[1] - t_still[0] > 1.2 else float("nan")  # 같은 기록의 움직이기 전 정지 구간
        floor = a.floor if a.floor > 0 else floor_own
        te, Ge = resamp(*imu["g"], t_enc - 1.0, min(t_lim, imu["g"][0][-1])); env = envf(Ge[:, ax]); m = te >= t_enc; env, te = env[m][:-k], te[m][:-k] - t_enc
        settle = float("nan")
        if np.isfinite(floor) and len(env) > k:  # 포락선이 바닥 2배 아래로 0.3 s 이어지는 첫 시각(유지 구간 끝까지 봄)
            lowrun = np.convolve((env < 2 * floor).astype(float), np.ones(int(0.3 * FS)), "valid") >= int(0.3 * FS) - 0.5
            settle = float(te[np.argmax(lowrun)]) if lowrun.any() else float("inf")
        E = enc_rate_cam(d, kin_tb, tg)
        e5 = rms(band(E[:, ax], 5, 10)) if E is not None else rms(band(np.radians(signal.savgol_filter(np.interp(tg, t, d[f"q{j}"].values), 11, 2, deriv=1, delta=1 / FS)), 5, 10))
        g5 = rms(band(G[:, ax], 5, 10))
        t_move = round(abs(float(args["dist"])) / float(args["speed"]), 4) if "ramp" in set(d.phase.values) and args.get("speed") else ""  # 명령 이동 시간
        arrive = ""  # 도착 방향: hold는 dist 부호, bhold(왕복의 돌아간 쪽)는 반대
        if args.get("dist") not in (None, "") and name in ("hold", "bhold"):
            arrive = "+" if (float(args["dist"]) > 0) == (name == "hold") else "-"
        gn, gf, gm = gaps(imu["g"][0], t_enc, t2)
        row = dict(file=os.path.basename(path), fw=meta.get("fw", ""), sdk=meta.get("sdk", ""), ctrl=meta.get("ctrl", "pos"), jump=int(any(a1 < t2 and b1 > t_enc for a1, b1 in jw)), joint=j, speed_dps=args.get("speed", ""), dist_deg=args.get("dist", ""),
                   shape=(meta.get("shape") or {}).get("kind", "none"), t_move_s=t_move, stop=name, rep=nrep[name], arrive=arrive,
                   stop_deg=round(float(np.interp(t_cmd + 0.5, t, d[f"qref{j}"].values)), 2), gap_n=gn, gap_frac=gf, gap_max_ms=gm, tilt_deg=round(tilt, 2), v_enc_dps=round(v_run, 2), t_stop_ms=round((t_enc - t_cmd) * 1000), axis="xyz"[ax],
                   peak1_rads=round(float(gl[i1]), 4), t_peak1_ms=round(float(tt[i1]) * 1000), max_rads=round(float(np.abs(gl).max()), 4),
                   gyro_rms=round(g5, 5), gyro_vec_rms=round(float(np.sqrt(sum(rms(band(G[:, c], 5, 10)) ** 2 for c in range(3)))), 5),
                   angacc_rms=round(rms(band(G[:, ax], 5, 10, deriv=True)), 4), acc_axis="xyz"[aax], acc_rms=round(rms(band(A[:, aax], 5, 10)), 4),
                   acc_vec_rms=round(float(np.sqrt(sum(rms(band(A[:, c], 5, 10)) ** 2 for c in range(3)))), 4),
                   f_fit=round(ff, 2), zeta_fit=round(zf, 3), r2=round(r2, 2), f_fft=round(f_fft, 2), zeta_log=round(zl, 3), n_cyc=ncyc,
                   settle_s=round(settle, 2), settle_cmd_s=round(settle + (t_enc - t_cmd), 2), floor_rads=round(floor, 5), floor_own_rads=round(floor_own, 5), enc_rms=round(e5, 5), enc_share=round(e5 / g5, 3) if g5 > 0 else float("nan"))
        rows.append(row)
        if fig is None:  # 현상 그림용: 첫 정지의 엔코더 속도와 끝단 각속도
            tp, Gp = resamp(*imu["g"], tu[0], min(tu[-1], imu["g"][0][-1])); gp = signal.sosfiltfilt(sos, Gp[:, ax] - bias)
            vp = np.interp(tp, tu, v); mrun = (tp > t_cmd - 0.5) & (tp < t_cmd + 0.1)
            sgn = np.sign(np.dot(gp[mrun], vp[mrun])) or 1.0  # 자이로 축 부호를 관절 방향에 맞춤
            fig = (tp - t_enc, vp, np.degrees(gp) * sgn, t_cmd - t_enc, t2 - t_enc, row)
    return rows, fig


def summary(rows):  # 조건별 표 — 중앙값(최소–최대), f·ζ는 R² 0.8 이상 시행만(개수 병기)
    d = pd.DataFrame(rows); d = d[d.speed_dps != ""].copy()
    if d.empty:
        return
    d["d"] = d.dist_deg.astype(float).abs(); d["dir"] = np.where(d.arrive == "", np.where(d.dist_deg.astype(float) > 0, "+", "-"), d.arrive); d["v"] = d.speed_dps.astype(float)
    mr = lambda x, n: f"{x.median():.{n}f} ({x.min():.{n}f}–{x.max():.{n}f})"
    print("| 속도 [°/s] | 거리 [°] | 방향 | 정지 | n | 명령 이동 시간 [s] | 도달 속도 [°/s] | 5–10 Hz 자이로 rms [rad/s] | 창 최대 [rad/s] | 가속도 rms [m/s²] | f [Hz] | ζ | R² ≥ 0.8 | 정착 [s] | 명령 정지부터 정착 [s] | 엔코더 몫 |")
    print("|" + "---|" * 16)
    for (v, dd, dr, st), m in d.groupby(["v", "d", "dir", "stop"], sort=True):
        ok = m[m.r2 >= 0.8]; fz = (f"{ok.f_fit.median():.2f} | {ok.zeta_fit.median():.3f}" if len(ok) else "— | —")
        print(f"| {v:g} | {dd:g} | {dr} | {st} | {len(m)} | {dd / v:.3f} | {m.v_enc_dps.abs().median():.1f} | {mr(m.gyro_rms, 4)} | {mr(m.max_rads, 3)} | {m.acc_rms.median():.3f} | {fz} | {len(ok)}/{len(m)} | "
              f"{mr(m.settle_s, 2)} | {mr(m.settle_cmd_s, 2)} | {m.enc_share.median():.2f} |")


def trials(rows):  # 시행(기록)별 중앙값 — 왕복이면 정지 종류(hold·bhold)마다 한 줄, 통계 단위는 시행(9절 11/17 6번). MIT 기록은 튐이 겹친 창을 버린 중앙값(n_win = 남은 창 수)과 버리지 않은 값(_inc)을 같이
    d = pd.DataFrame(rows); out = []
    for (f, st), m0 in d.groupby(["file", "stop"], sort=False):
        m = m0[m0.jump == 0]; ok = m[m.r2 >= 0.8]; r0 = m0.iloc[0]; med = lambda x, n: round(float(x.median()), n) if len(x) else float("nan")
        out.append(dict(file=f, cond=r0.ctrl + ("" if r0["shape"] == "none" else "+" + r0["shape"]), ctrl=r0.ctrl, speed_dps=r0.speed_dps, dist_deg=r0.dist_deg, shape=r0["shape"], stop=st, arrive=r0.arrive, stop_deg=r0.stop_deg,
                        n_win=len(m), n_jump=int(m0.jump.sum()), tilt_deg=r0.tilt_deg, gyro_rms=med(m.gyro_rms, 5), acc_rms=med(m.acc_rms, 4), max_rads=med(m.max_rads, 4),
                        settle_s=med(m.settle_s, 2), settle_cmd_s=med(m.settle_cmd_s, 2), t_stop_ms=med(m.t_stop_ms, 0),
                        f_fit=round(float(ok.f_fit.median()), 2) if len(ok) else "", zeta_fit=round(float(ok.zeta_fit.median()), 3) if len(ok) else "", n_r2=len(ok),
                        enc_share=med(m.enc_share, 3), gap_max_ms=round(float(m0.gap_max_ms.max()), 1), gap_frac=round(float(m0.gap_frac.median()), 3),
                        gyro_rms_inc=med(m0.gyro_rms, 5), acc_rms_inc=med(m0.acc_rms, 4)))
    return out


def judge(tr):  # 기준(위치 모드, 성형 없음 = cond "pos") 대비 비와 판정 — 자이로·가속도 둘 다 0.83 이하면 감소, 하나라도 1.2 이상이면 증가, 시행 범위가 겹치면 경향(9절 11/17 6번). MIT 조건은 튐 제외 값으로 판정하고 포함 값의 비를 병기
    d = pd.DataFrame(tr); d = d[d.speed_dps != ""]
    if d.empty or d.cond.nunique() < 2 or "pos" not in set(d.cond):
        return
    mr = lambda x, n: f"{x.median():.{n}f} ({x.min():.{n}f}–{x.max():.{n}f})"
    print("\n| 속도 [°/s] | 정지 | 조건 | 시행 수(유효 창 3개 미만) | 5–10 Hz 자이로 rms [rad/s] | 비 | 가속도 rms [m/s²] | 비 | 튐 포함 자이로 비 | 버린 창 | 창 최대 [rad/s] | 정착 [s] | f [Hz] | 판정 |")
    print("|" + "---|" * 14)
    order = ["pos"] + [c for c in dict.fromkeys(d.cond) if c != "pos"]
    for (v, st), g in d.groupby(["speed_dps", "stop"], sort=True):
        b = g[g.cond == "pos"]; r0 = g.iloc[0]
        if b.empty:
            continue
        where = f"{st} ({UPDOWN.get(r0.arrive, r0.arrive)}, {r0.stop_deg:g}°)"
        for c in order:
            m = g[(g.cond == c) & g.gyro_rms.notna()]
            if m.empty:
                continue
            rg = m.gyro_rms.median() / b.gyro_rms.median(); ra = m.acc_rms.median() / b.acc_rms.median(); ri = m.gyro_rms_inc.median() / b.gyro_rms_inc.median()
            lap = m.gyro_rms.max() >= b.gyro_rms.min() and m.gyro_rms.min() <= b.gyro_rms.max()  # 시행 범위가 겹치는지(자이로)
            if c == "pos":
                verdict = "기준"
            elif rg >= 1.2 or ra >= 1.2:
                verdict = "증가" + ("(범위 겹침 → 경향)" if lap else "")
            elif rg <= 0.83 and ra <= 0.83:
                verdict = "감소" + ("(범위 겹침 → 경향)" if lap else "")
            else:
                verdict = "차이 없음"
            ff = pd.to_numeric(m.f_fit, errors="coerce").dropna()
            print(f"| {float(v):g} | {where} | {c} | {len(m)}({int((m.n_win < 3).sum())}) | {mr(m.gyro_rms, 4)} | {rg:.2f} | {mr(m.acc_rms, 3)} | {ra:.2f} | {ri:.2f} | {int(m.n_jump.sum())} | {mr(m.max_rads, 3)} | {mr(m.settle_s, 2)} | "
                  + (f"{ff.median():.2f}" if len(ff) else "—") + f" | {verdict} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--win", type=float, default=2.0, help="정지 창 길이[s]")
    ap.add_argument("--v-thr", type=float, default=0.3, help="엔코더 정지 판정 속도 하한[deg/s] (등속의 5%%와 큰 쪽)")
    ap.add_argument("--axis", default="auto", help="자이로 주축 x/y/z 또는 auto")
    ap.add_argument("--floor", type=float, default=0.0099, help="정착 판정 잡음 바닥 포락선[rad/s] — 기본 0.0099(C-1 90시행 중앙값, 9절 11/17), 0이면 기록 안 정지 구간")
    ap.add_argument("--imu-lag", type=float, default=0.0, help="IMU가 엔코더보다 늦는 시간[ms] (B-5 값)")
    ap.add_argument("--sustained", action="store_true", help="엔코더 정지 = 문턱 아래로 0.1 s 이어진 첫 시각(11/16 정의 (가)) — 기본은 처음 내려간 시각(정의 (나))")
    ap.add_argument("--first", action="store_true", help="(기본이 됨, 호환용으로 남김)")
    ap.add_argument("--summary", action="store_true", help="조건(속도·거리·방향)별 중앙값(최소–최대) 표를 마크다운으로")
    ap.add_argument("--all", action="store_true", help="왕복의 돌아간 쪽 정지(bhold)도 분석")
    ap.add_argument("--out", default="", help="결과 CSV")
    ap.add_argument("--trial-out", default="", help="시행(기록)별 중앙값 CSV — 왕복 기록의 정지 창들을 hold·bhold마다 하나로")
    ap.add_argument("--judge", action="store_true", help="성형 조건별 시행 중앙값(범위)과 성형 없음 대비 비·판정 표를 마크다운으로")
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
    print("기록 | 관절 속도 거리 | 기울기 | 도달 속도[°/s] 정지 지연[ms] | 축 창 최대[rad/s] (첫 봉우리@ms) | 5–10 Hz 자이로·각가속도·가속도 | f[Hz] ζ(R²) | 정착[s] 명령 정지부터[s] | 엔코더 몫 | IMU 빈틈(자이로, 창 안)")
    for r in rows:
        print(f"{r['file'][:15]} {r['ctrl']} {r['stop']}{r['rep']}{'★튐' if r['jump'] else ''} {r['shape']} {UPDOWN.get(r['arrive'], '')} {r['stop_deg']:g}° | J{r['joint']} {r['speed_dps']} {r['dist_deg']} | {r['tilt_deg']:.2f}° | {r['v_enc_dps']:+.1f} {r['t_stop_ms']} | "
              f"{r['axis']} {r['max_rads']:.4f} ({r['peak1_rads']:+.4f}@{r['t_peak1_ms']}) | {r['gyro_rms']:.4f} {r['angacc_rms']:.3f} {r['acc_rms']:.3f}({r['acc_axis']}) | "
              f"{r['f_fit']:.2f} {r['zeta_fit']:.3f}({r['r2']:.2f}) | {r['settle_s']} {r['settle_cmd_s']} | {r['enc_share']} | 빈틈 {r['gap_n']}개 {100 * r['gap_frac']:.0f}% 최대 {r['gap_max_ms']:.0f} ms")
    if rows:
        ok = [r for r in rows if r["r2"] >= 0.8]  # 주파수·감쇠비 통계는 맞춤이 된 시행만
        if ok:
            print(f"f·ζ 통계(R² ≥ 0.8인 {len(ok)}/{len(rows)}개): f 중앙 {np.median([r['f_fit'] for r in ok]):.2f} Hz ({min(r['f_fit'] for r in ok):.2f}–{max(r['f_fit'] for r in ok):.2f}), "
                  f"ζ 중앙 {np.median([r['zeta_fit'] for r in ok]):.3f} ({min(r['zeta_fit'] for r in ok):.3f}–{max(r['zeta_fit'] for r in ok):.3f})")
        print(f"엔코더 정지 정의: {'(가) 0.1 s 이어진 첫 시각' if a.sustained else '(나) 처음 내려간 시각'} | 정착 바닥: {('고정 %.4f rad/s' % a.floor) if a.floor > 0 else '기록별'}")
    if a.summary and rows:
        summary(rows)
    if rows and (a.trial_out or a.judge):
        tr = trials(rows)
        if a.trial_out:
            os.makedirs(os.path.dirname(os.path.abspath(a.trial_out)), exist_ok=True); pd.DataFrame(tr).to_csv(a.trial_out, index=False); print("시행별 표:", a.trial_out)
        if a.judge:
            judge(tr)
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
