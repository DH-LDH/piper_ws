#!/usr/bin/env python3
"""D2-2 툭 시행 요약 — mit_probe --vib-joint 기록(.csv, .vib.csv)에서 회별 지표 (8절 10/26 저녁).

툭 시각 t0 = --beep 뒤 첫 |v|(변형 속도) 피크, 툭 세기 = 그 피크값. 모드 = t0+0.25–1.5 s v 스펙트럼 피크, ζ·1/e 시간 = 4–16 Hz 포락선.
MIT 튐 = 엔코더만 튐: 한 주기 |Δq̇| > 0.5 rad/s인 관절 표본 중, 같은 시각 자이로 피치(wy)가 그 점프의 피치 몫을 따라가지 않은 것.
재시행 판정: [t0 − 0.1, t0 + 1.5 s]에 MIT 튐이 있으면 다시.
τ 비(판단 9): 실제 보낸 τ_J(4–16 Hz)를 τ_vib에 투영한 이득 Σ τ_J·τ_vib / Σ τ_vib² (k > 0, 게이트 켬일 때만) — 0.8–1.2 벗어나면 τ_ff 직접 비교군.
사용: source install/setup.bash; python3 tools/vib_tap_report.py logs/mit/20261027_*_hold.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "common_pkg")); sys.path.insert(0, HERE)
from imu_frame import ImuToBase, static_tilt  # noqa: E402
from link_est import jacobians  # noqa: E402
from piper_kin import PiperKin  # noqa: E402

FS = 200.0


def enc_only_spikes(m, v, kin, tb, t_mit):  # → MIT 튐 시각 목록(MIT 기준 s)
    qd = m[[f"qd{j}" for j in range(1, 7)]].values; q = np.radians(m[[f"q{j}" for j in range(1, 7)]].values)
    t = m.t.values - t_mit; tv = v.t_gyro.values - t_mit
    out = []
    for i in np.where((np.abs(np.diff(qd, axis=0)) > 0.5).any(1))[0] + 1:
        Jw, _ = jacobians(kin, q[i], tb.R_l6)
        jump = Jw[1] @ (qd[i] - qd[i - 1])  # 엔코더 점프의 피치 몫
        w = np.abs(tv - t[i]) < 0.015
        dwy = np.ptp(v.wy.values[w]) if w.sum() > 1 else 0.0
        if abs(jump) < 0.1 or dwy < 0.5 * abs(jump):  # 피치 몫이 없는 관절 점프(J1 등)도 엔코더만 튐으로 셈
            out.append(t[i])
    return np.unique(np.round(out, 2))


def ring(tv, x, t0):
    s = (tv > t0 + 0.03) & (tv < t0 + 1.5)
    if s.sum() < 50:
        return np.nan, np.nan, np.nan
    tu = np.arange(tv[s][0], tv[s][-1], 1 / FS); y = np.interp(tu, tv[s], x[s])
    b, a = signal.butter(2, [4, 16], btype="band", fs=FS); y = signal.filtfilt(b, a, y - y.mean())
    w = tu > t0 + 0.25
    ff, P = signal.welch(y[w], FS, nperseg=min(w.sum(), 128)); bb = (ff > 3) & (ff < 15); f0 = ff[bb][np.argmax(P[bb])]
    env = np.abs(signal.hilbert(y)); e0 = env[:int(0.1 * FS)].max()
    k = np.where(env[int(0.05 * FS):] < e0 / np.e)[0]; te = (k[0] / FS + 0.05) if len(k) else np.nan
    sel = (env > 0.05 * e0) & (tu < tu[0] + 1.2)
    sl = np.polyfit(tu[sel] - tu[0], np.log(env[sel]), 1)[0] if sel.sum() > 20 else np.nan
    return f0, -sl / (2 * np.pi * f0), te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--beep", type=float, default=5.0)
    a = ap.parse_args()
    kin = PiperKin(); tb = ImuToBase(kin)
    print("시행 | 기울기 시작/끝 | 게이트 | 툭 t0 | 첫 v 피크 | 모드 Hz | ζ | 1/e s | MIT 튐(툭 창/전체) | 재시행 | τ 비 | 나이 중앙/95% ms | 거부 | 묵음")
    for f in a.csv:
        d = pd.read_csv(f); v = pd.read_csv(f.replace(".csv", ".vib.csv")); m = d[d.phase == "mit"]
        t_mit = m.t.iloc[0]; tv = v.t.values - t_mit
        w = (tv > a.beep - 0.6) & (tv < a.beep + 4); i0 = np.where(w)[0][np.argmax(np.abs(v.v.values[w]))]; t0 = tv[i0]
        f0, z, te = ring(tv, v.v.values, t0)
        sp = enc_only_spikes(m, v, kin, tb, t_mit); win = sp[(sp > t0 - 0.1) & (sp < t0 + 1.5)]
        en = v.enabled.values.astype(bool); tilt = v.tilt.dropna().iloc[0] if v.tilt.notna().any() else np.nan
        J = 2
        ratio = np.nan
        if en.any() and np.abs(v.tau_vib.values).max() > 0:
            tu = np.arange(tv[0], tv[-1], 1 / FS)
            tj = np.interp(tu, m.t.values - t_mit, m[f"tff{J}"].values); tvb = np.interp(tu, tv, v.tau_vib.values)
            b, aa = signal.butter(2, [4, 16], btype="band", fs=FS)
            tj = signal.filtfilt(b, aa, tj - tj.mean()); tvb = signal.filtfilt(b, aa, tvb - tvb.mean())
            ratio = float((tj * tvb).sum() / max((tvb * tvb).sum(), 1e-12))
        end = static_tilt(f, m.t.iloc[-1] - 1.5, m.t.iloc[-1])
        age = v.age_ms.values[tv > 2]
        print(f"{os.path.basename(f)[:15]} | {tilt:.2f}/{end:.2f}° | {'켬' if en.any() else '끔'} | {t0:.2f} | {abs(v.v.values[i0]):.3f} | "
              f"{f0:.2f} | {z:.3f} | {te:.2f} | {len(win)}/{len(sp)} | {'다시' if len(win) else '-'} | "
              f"{'' if np.isnan(ratio) else f'{ratio:.2f}'} | {np.median(age):.1f}/{np.percentile(age, 95):.1f} | {v.rejects.iloc[-1]} | {v.stale.iloc[-1]}")


if __name__ == "__main__":
    main()
