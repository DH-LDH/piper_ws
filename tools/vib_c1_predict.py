#!/usr/bin/env python3
"""C-1 예측 시험 — 엔코더로 잰 관절 운동이 끝단 모드를 얼마나 흔드는지로 정지 후 진동의 조건 간 차이를 설명하는지 본다(FW191_C_LOG 2.2절, 9절 11/17 12번).

방법: 움직인 관절의 각가속도 q̈를 1자유도 모드(받침 가진) θ̈ + 2ζωθ̇ + ω²θ = −q̈ 에 넣고, 정지 창(엔코더 정지부터 2 s)의 θ̇ 5–10 Hz rms를 측정 자이로 rms와 비교.
입력 구간: cut = 엔코더 정지 시각까지만 넣고 그 뒤는 0(정지 창 안 엔코더 값이 들어가면 끝단 진동이 엔코더에 비친 성분으로 끝단 진동을 예측하는 순환), full = 기록 전체(11/17 첫 계산).
모드 값: 7.0 Hz, ζ 0.11 — C-1에 맞춘 값이 아니라 10/23 계단 기록(자세 A 6.6–7.3 Hz, ζ 0.09–0.13, FW191_A_LOG 0.3절)에서 온 값. 민감도는 --sens.
예측 모형이 아니다: 배율(측정 / 계산)을 하나 맞춰도 R²가 낮고 같은 조건 안 시행 차이는 맞추지 못한다. "조건 간 차이를 실제 관절 가감속이 설명한다"까지만.
사용: python3 tools/vib_c1_predict.py docs/data/c1_stops_first.csv [--write] [--sens] [--plot docs/img/p_fw191_c1_summary.png]
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import signal

FS = 200.0
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "vib")


def band(x, lo, hi):  # vib_stop_report.band와 같은 FFT 벽돌 필터
    x = signal.detrend(x); X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / FS); X[(f < lo) | (f > hi)] = 0
    return np.fft.irfft(X, len(x))


def load(rows, logdir):  # → 시행마다 (시각, 엔코더 각가속도, 명령 각가속도, 엔코더 정지 시각)
    out = []
    for r in rows.itertuples():
        d = pd.read_csv(os.path.join(logdir, r.file)); t = d.t.values; ph = d.phase.values
        i = np.where((ph == "hold") & (np.r_[True, ph[:-1] != "hold"]))[0][0]; te = t[i] + r.t_stop_ms / 1000.0
        tu = np.arange(t[0], t[-1], 1 / FS); acc = []
        for col in (f"q{r.joint}", f"qref{r.joint}"):
            acc.append(signal.savgol_filter(np.radians(np.interp(tu, t, d[col].values)), 11, 3, deriv=2, delta=1 / FS))
        out.append((tu, acc[0], acc[1], te))
    return out


def predict(data, fn, z, src, cut, win=2.0):  # → 시행별 정지 창 θ̇ 5–10 Hz rms
    w = 2 * np.pi * fn; sysd = signal.lti([-1.0, 0.0], [1.0, 2 * z * w, w * w]); out = []  # θ̇ / q̈ = −s / (s² + 2ζωs + ω²)
    for tu, a_enc, a_ref, te in data:
        a = (a_enc if src == "enc" else a_ref).copy()
        if cut:
            a[tu > te] = 0.0
        _, y, _ = signal.lsim(sysd, a, tu - tu[0]); m = (tu >= te) & (tu < te + win)
        out.append(np.sqrt(np.mean(band(y[m], 5, 10) ** 2)))
    return np.array(out)


def stats(p, y, cond):  # → 시행 상관, 조건 평균 상관, 조건 중앙값 상관, 같은 조건 안 상관, 배율, 배율 1개 R²
    df = pd.DataFrame(dict(p=p, y=y, c=cond)); g = df.groupby("c")
    k = float(np.dot(p, y) / np.dot(p, p)); r2 = 1 - ((y - k * p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rp = df.p - g.p.transform("mean"); ry = df.y - g.y.transform("mean")
    return (np.corrcoef(p, y)[0, 1], np.corrcoef(g.p.mean(), g.y.mean())[0, 1], np.corrcoef(g.p.median(), g.y.median())[0, 1],
            np.corrcoef(rp, ry)[0, 1], k, r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stops", help="vib_stop_report --out 표(정의 (나))")
    ap.add_argument("--logdir", default=LOG)
    ap.add_argument("--f", type=float, default=7.0); ap.add_argument("--zeta", type=float, default=0.11)
    ap.add_argument("--sens", action="store_true", help="모드 값 민감도(7.0·0.08, 6.8·0.12, 6.8·0.11, 7.2·0.10)")
    ap.add_argument("--write", action="store_true", help="표에 pred_sdof(전체 입력)·pred_sdof_cut(정지 시각까지) 열을 적음")
    ap.add_argument("--plot", default="", help="조건별 rms와 예측 대 측정 그림 PNG")
    a = ap.parse_args()
    R = pd.read_csv(a.stops); y = R.gyro_rms.values
    cond = R.speed_dps.astype(str) + "_" + R.dist_deg.astype(str); data = load(R, a.logdir)
    sets = [(a.f, a.zeta)] + ([(7.0, 0.08), (6.8, 0.12), (6.8, 0.11), (7.2, 0.10)] if a.sens else [])
    print(f"시행 {len(R)}개, 조건 {cond.nunique()}개 | 모드 f·ζ | 입력 | 구간 | 시행 상관 | 조건 평균 상관 | 조건 중앙값 상관 | 조건 안 상관 | 배율 | 배율 1개 R²")
    keep = {}
    for fn, z in sets:
        for src in ("enc", "ref"):
            for cut in (True, False):
                p = predict(data, fn, z, src, cut); s = stats(p, y, cond)
                print(f"  {fn:.1f} Hz ζ {z:.2f} | {'엔코더' if src == 'enc' else '명령'} | {'정지 시각까지' if cut else '기록 전체'} | "
                      f"{s[0]:.2f} | {s[1]:.2f} | {s[2]:.2f} | {s[3]:.2f} | {s[4]:.2f} | {s[5]:.2f}")
                if (fn, z) == sets[0] and src == "enc":
                    keep[cut] = p
    R["pred_sdof"] = np.round(keep[False], 5); R["pred_sdof_cut"] = np.round(keep[True], 5)
    if a.write:
        R.to_csv(a.stops, index=False); print("표:", a.stops)
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        R["d"] = R.dist_deg.abs(); R["down"] = R.dist_deg > 0
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        for k, (nm, dn) in enumerate((("lowering", True), ("raising", False))):
            for j, (d, c) in enumerate(((2.5, "C0"), (5, "C1"), (10, "C2"))):
                m = R[(R.down == dn) & (R.d == d)]; ax[k].scatter(m.speed_dps + (j - 1) * 0.35, m.gyro_rms, color=c, s=14, alpha=0.6)
                g = m.groupby("speed_dps").gyro_rms.median(); ax[k].plot(g.index + (j - 1) * 0.35, g.values, "-", color=c, label=f"{d:g} deg")
            ax[k].set_title(f"J2 {nm}: 5 trials each, line = median", fontsize=10); ax[k].set_xlabel("commanded speed [deg/s]")
            ax[k].set_ylabel("5–10 Hz gyro rms, 2 s after stop [rad/s]"); ax[k].set_ylim(0, 0.13); ax[k].set_xticks(sorted(R.speed_dps.unique())); ax[k].grid(alpha=0.3)
            ax[k].legend(title="distance", fontsize=8)
        for dn, mk in ((True, "o"), (False, "s")):
            m = R[R.down == dn]; sc = ax[2].scatter(m.pred_sdof_cut, m.gyro_rms, c=m.speed_dps, marker=mk, cmap="viridis", s=22, label="lowering" if dn else "raising")
        plt.colorbar(sc, ax=ax[2], label="speed [deg/s]"); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)
        ax[2].set_xlabel(f"{a.f:g} Hz mode driven by encoder motion up to stop [rad/s]"); ax[2].set_ylabel("measured rms [rad/s]")
        s = stats(keep[True], y, cond); ax[2].set_title(f"trials r = {s[0]:.2f} (n = {len(R)}), condition medians r = {s[2]:.2f}", fontsize=10)
        fig.suptitle("C-1: residual vibration after stop, J2, pose A, position mode (S-V1.7-3), camera-site gyro", fontsize=10)
        fig.tight_layout(); fig.savefig(a.plot, dpi=120); print("그림:", a.plot)


if __name__ == "__main__":
    main()
