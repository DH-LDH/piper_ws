#!/usr/bin/env python3
"""처프 식별 — 실제 보낸 τ(CSV tff<J>) → 자이로 피치·엔코더 피치·관절 속도 주파수 응답, 처프 순간 주파수를 따라가는 lock-in.

입력은 chirp 컬럼이 아니라 tff<J>: TDE 경로(dist_comp)에선 둘이 다름(VIB_E2 6-1). τ/처프 비도 함께 출력.
lock-in: 처프 위상 φ(t)로 복조, 창 max(4주기, 0.3 s) 해닝 — STFT 칸(1.6 Hz)보다 촘촘해 6–8 Hz 공진을 0.25 Hz로 봄.
순간 튐(MIT, 0.5 rad/s 초과)은 엔코더에서 보간으로 메우고, 튐 −0.1…+1 s가 걸친 창은 뺌(튐이 6–7 Hz 모드를 때려 자유진동이 남음).
정지 기울기 = 처프 시작 1.5 s의 IMU 가속도 atan2(y, −z) — 엔코더가 같아도 회마다 2–4° 다름(백래시·처짐), 응답 크기와 연관.
사용: source install/setup.bash; python3 tools/vib_chirp_frf.py logs/mit/20261026_*_hold.csv [--label ...] [--out docs/img/p_vib_e3_chirp.png]
"""
import argparse
import math
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
import vib_d2_estimator as E  # noqa: E402

REPORT_F = [3, 4, 5, 6, 6.5, 7, 7.5, 8, 9, 10, 12, 15]


def despike(t, x, thr=0.5, pad=0.03):  # |x − 중앙값 필터| > thr 인 표본 ±pad s를 선형 보간
    med = signal.medfilt(x, 9); bad = np.abs(x - med) > thr
    m = np.zeros_like(bad)
    for ti in t[bad]:
        m |= np.abs(t - ti) <= pad
    y = x.copy()
    if m.any() and (~m).sum() > 2:
        y[m] = np.interp(t[m], t[~m], x[~m])
    return y, t[bad]


def lockin(tu, c, u, ys, spikes, fmin=2.5, fmax=19.0, step=10):  # 창마다 처프 위상으로 복조 → [(f, τ/처프, {이름: y/τ})]
    ph = np.unwrap(np.angle(signal.hilbert(c))); fi = np.gradient(ph, tu) / (2 * np.pi); z = np.exp(-1j * ph)
    rows = []
    for k in range(0, len(tu), step):
        fk = fi[k]
        if not fmin <= fk <= fmax:
            continue
        W = max(4 / fk, 0.3); s = np.abs(tu - tu[k]) < W / 2
        if np.any((spikes > tu[k] - W / 2 - 1.0) & (spikes < tu[k] + W / 2 + 0.1)):
            continue
        w = np.hanning(s.sum()); C = (w * c[s] * z[s]).sum(); U = (w * u[s] * z[s]).sum()
        rows.append((fk, U / C, {n: (w * y[s] * z[s]).sum() / U for n, y in ys.items()}))
    return rows


def binned(rows, key, width=0.25):  # 0.25 Hz 칸 중앙값(크기)·원형 평균(위상)
    df = pd.DataFrame([(r[0], abs(r[2][key]), np.angle(r[2][key])) for r in rows], columns=["f", "mag", "ph"])
    df["fb"] = (df.f / width).round() * width
    return df.groupby("fb").agg(mag=("mag", "median"), ph=("ph", lambda p: np.angle(np.exp(1j * p).mean())))


def analyze(f, kin, tb):
    d = pd.read_csv(f); m = d[(d.phase == "mit") & (d.chirp != 0)]
    t1, t2 = m.t.iloc[0], m.t.iloc[-1]
    J = 1 + int(np.argmax([m[f"tff{j}"].std() for j in range(1, 7)]))
    tu, Q, QD, G, _ = E.load_aligned(f, t1, t2, kin, tb)
    sp = []
    for j in range(6):
        QD[:, j], ts = despike(tu, QD[:, j]); sp += list(ts)
    sp = np.array(sp)
    u = E.band(np.interp(tu, m.t.values, m[f"tff{J}"].values), 1.5, 25); c = np.interp(tu, m.t.values, m.chirp.values)
    enc = np.array([jacobians(kin, q, tb.R_l6)[0][1] @ qd for q, qd in zip(Q, QD)])
    ys = {"gyro": E.band(G[:, 1], 1.5, 25), "enc": E.band(enc, 1.5, 25), "qdJ": E.band(QD[:, J - 1], 1.5, 25)}
    ys["defl"] = ys["gyro"] - ys["enc"]  # 변형 속도 = τ_vib가 줄일 대상(강체 회전 몫 제외)
    rows = lockin(tu, c, u, ys, sp)
    tr = pd.DataFrame([(r[0], abs(r[1])) for r in rows], columns=["f", "r"])
    return dict(name=os.path.basename(f)[:15], J=J, tilt=static_tilt(f, t1, t1 + 1.5), nsp=len(sp), amp=np.abs(m.chirp).max(),
                gyro=binned(rows, "gyro"), enc=binned(rows, "enc"), qdJ=binned(rows, "qdJ"), defl=binned(rows, "defl"),
                tau=tr.groupby((tr.f * 4).round() / 4).r.median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--label", nargs="*", default=[], help="파일별 표시 이름(같은 순서)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    kin = PiperKin(); tb = ImuToBase(kin); res = []
    for n, f in enumerate(a.csv):
        r = analyze(f, kin, tb); r["label"] = a.label[n] if n < len(a.label) else r["name"]; res.append(r)
        g = r["gyro"]; b = g[(g.index >= 5) & (g.index <= 9)]; ip = b.mag.idxmax(); r["peak"] = (ip, b.mag[ip])
        dfl = r["defl"]; bd = dfl[(dfl.index >= 5) & (dfl.index <= 9)]; idp = bd.mag.idxmax(); r["dpeak"] = (idp, bd.mag[idp], bd.ph[idp])
        print(f"\n{r['label']} | J{r['J']} ±{r['amp']:.2f} | 정지 기울기 {r['tilt']:.2f}° | 튐 {r['nsp']}표본"
              f" | 자이로 피크 {ip:.2f} Hz |H| {b.mag[ip]:.3f} ∠{np.degrees(b.ph[ip]):+.0f}°"
              f" | 변형 피크 {idp:.2f} Hz |H| {bd.mag[idp]:.3f} ∠{np.degrees(bd.ph[idp]):+.0f}°")
        print("  f[Hz]  τ/처프  gyro |H| ∠      enc |H|  qdJ |H|  defl |H| ∠    (rad/s per N·m)")
        for fq in REPORT_F:
            if fq in g.index:
                print(f"  {fq:5.2f}  {r['tau'].get(fq, np.nan):5.2f}  {g.mag[fq]:6.3f} {np.degrees(g.ph[fq]):+5.0f}°"
                      f"  {r['enc'].mag.get(fq, np.nan):6.3f}  {r['qdJ'].mag.get(fq, np.nan):6.3f}"
                      f"  {r['defl'].mag.get(fq, np.nan):6.3f} {np.degrees(r['defl'].ph.get(fq, np.nan)):+5.0f}°")
    for J in sorted({r["J"] for r in res}):
        rr = [r for r in res if r["J"] == J]
        if len(rr) > 2:
            tl = pd.Series([r["tilt"] for r in rr])
            print(f"\nJ{J} 정지 기울기–피크 |H| 순위상관(Spearman) 자이로 {tl.corr(pd.Series([r['peak'][1] for r in rr]), method='spearman'):.2f}"
                  f" · 변형 {tl.corr(pd.Series([r['dpeak'][1] for r in rr]), method='spearman'):.2f} ({len(rr)}회)")
    if a.out:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        joints = sorted({r["J"] for r in res})
        fig, ax = plt.subplots(2, len(joints) + 1, figsize=(5 * (len(joints) + 1), 7))
        for k, J in enumerate(joints):
            for r in [r for r in res if r["J"] == J]:
                g = r["gyro"]; s = (g.index >= 3) & (g.index <= 15)
                ln, = ax[0, k].semilogy(g.index[s], g.mag[s], label=f"{r['label']} ({r['tilt']:.1f}°)")
                ax[1, k].plot(g.index[s], np.degrees(g.ph[s]), color=ln.get_color())
            ax[0, k].set_title(f"J{J} chirp τ → gyro pitch rate"); ax[0, k].set_ylabel("|H| [rad/s per N·m]")
            ax[1, k].set_ylabel("phase [deg]"); ax[1, k].set_xlabel("Hz"); ax[0, k].legend(fontsize=7)
            [x.grid(alpha=.3) for x in ax[:, k]]
        for r in res:
            ax[0, -1].plot(r["tilt"], r["dpeak"][1], "x", color="C0" if r["J"] == 2 else "C3")
            ax[0, -1].plot(r["tilt"], r["peak"][1], "o" if r["J"] == 2 else "s", color="C0" if r["J"] == 2 else "C3")
            ax[0, -1].annotate(r["label"], (r["tilt"], r["peak"][1]), fontsize=7)
        ax[0, -1].set_xlabel("static tilt at chirp start [deg] (IMU atan2(y,−z))"); ax[0, -1].set_ylabel("peak |H| 5–9 Hz")
        ax[0, -1].set_title("response vs link tilt (o/□ gyro, x deflection; blue J2, red J5)"); ax[0, -1].grid(alpha=.3); ax[1, -1].axis("off")
        fig.tight_layout(); fig.savefig(a.out, dpi=110); print("그림:", a.out)


if __name__ == "__main__":
    main()
