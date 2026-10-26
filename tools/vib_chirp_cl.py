#!/usr/bin/env python3
"""D2-2 짝 폐루프 처프 판정 (8절 10/26 밤 2) — 한 MIT 시행 안 k = 0 처프 구간 vs k 처프 구간.

처프 위상에 고정한 lock-in(vib_chirp_frf.lockin)으로 구간마다:
  S = v / 처프 명령   (폐루프 효과 — 판정은 같은 회 안 S_k/S₀ 피크 비)
  H = v / τ_sent      (개루프 플랜트, 처프 고정이라 폐루프 잡음 편향 없음)
  L = −τ_vib / τ_sent (= −H·C, C = τ_vib/v — 실제 루프, S/S₀ ≈ 1/|1 + L|, 감쇠면 ∠L이 0° 근처, 여유 = 90° − |∠L|)
  τ 비 = τ_sent / (처프 + τ_vib)  (TDE가 τ_vib·처프를 그대로 넘기는지, 0.8–1.2)
v는 IMU 파일로 다시 계산(자이로 피치 − J_ω q̇, 장치 시각) — 실시간 값과 독립. τ_sent = CSV tff<J>.
사용: source install/setup.bash; python3 tools/vib_chirp_cl.py logs/mit/2026102*_hold.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "common_pkg")); sys.path.insert(0, HERE)
from imu_frame import ImuToBase, static_tilt  # noqa: E402
from link_est import jacobians  # noqa: E402
from piper_kin import PiperKin  # noqa: E402
import vib_chirp_frf as F  # noqa: E402
import vib_d2_estimator as E  # noqa: E402

BAND = (6.5, 9.0)


def segment(f, d, v, t1, t2, J, kin, tb, fmax=11.0):
    m = d[(d.t >= t1) & (d.t <= t2)]
    tu, Q, QD, G, _ = E.load_aligned(f, t1, t2, kin, tb)
    sp = []
    for j in range(6):
        QD[:, j], ts = F.despike(tu, QD[:, j]); sp += list(ts)
    ey = np.array([jacobians(kin, q, tb.R_l6)[0][1] @ qd for q, qd in zip(Q, QD)])
    c = np.interp(tu, m.t.values, m.chirp.values)
    tau = np.interp(tu, m.t.values, m[f"tff{J}"].values); tv = np.interp(tu, v.t.values, v.tau_vib.values)
    ys = {"v": E.band(G[:, 1] - ey, 1.5, 45), "tv": E.band(tv, 1.5, 45), "ct": E.band(c + tv, 1.5, 45)}
    rows = F.lockin(tu, c, E.band(tau, 1.5, 45), ys, np.array(sp), fmin=5.0, fmax=fmax)
    df = pd.DataFrame([(r[0], r[1], r[2]["v"], r[2]["tv"], r[2]["ct"]) for r in rows], columns=["f", "Tc", "H", "Ttv", "Tct"])
    df["S"] = df.H * df.Tc; df["L"] = -df.Ttv; df["tr"] = 1 / df.Tct  # τ_sent/(처프+τ_vib)
    df["fb"] = (df.f * 4).round() / 4
    agg = lambda x: np.median(np.abs(x))
    cang = lambda x: np.degrees(np.angle(np.exp(1j * np.angle(x)).mean()))
    df["nC"] = df.L / df.H  # −C = L/H (제어기 잔차: 예측·지연·필터)
    df["fb"] = np.where(df.f > 11, (df.f).round(), df.fb)  # 11 Hz 위는 1 Hz 칸
    g = df.groupby("fb").agg(S=("S", agg), H=("H", agg), L=("L", agg), tr=("tr", agg),
                             Lph=("L", cang), Hph=("H", cang), nCph=("nC", cang))
    return g, len(sp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--fmax", type=float, default=11.0, help="lock-in 상한[Hz] — 고주파 개루프 처프(4–40 Hz)는 40")
    ap.add_argument("--lead", type=float, default=0.0, help="|L| 예측에 쓸 φ_lead[deg]")
    a = ap.parse_args()
    kin = PiperKin(); tb = ImuToBase(kin)
    out = []
    for f in a.csv:
        d = pd.read_csv(f); v = pd.read_csv(f.replace(".csv", ".vib.csv"))
        mit = d[d.phase == "mit"]; c = mit[mit.chirp != 0]
        J = 1 + int(np.argmax([c[f"tff{j}"].std() for j in range(1, 7)]))
        gap = np.where(np.diff(c.t.values) > 1.0)[0]
        segs = [(c.t.iloc[0], c.t.iloc[gap[0]]), (c.t.iloc[gap[0] + 1], c.t.iloc[-1])] if len(gap) else [(c.t.iloc[0], c.t.iloc[-1])]
        res = []
        for t1, t2 in segs:
            k = float(np.median(np.interp(np.linspace(t1, t2, 50), v.t.values, v.k.values))) if "k" in v else np.nan
            g, nsp = segment(f, d, v, t1, t2, J, kin, tb, a.fmax)
            res.append((k, g, nsp, static_tilt(f, t1 - 1.5, t1)))
        print(f"\n{os.path.basename(f)[:15]} | J{J} | 구간 {len(res)}")
        for k, g, nsp, tl in res:
            b = g[(g.index >= BAND[0]) & (g.index <= BAND[1])]; ip = b.S.idxmax()
            print(f"  k {k:.2f} | 기울기 {tl:.2f}° | 튐 {nsp}표본 | S 피크 {ip:.2f} Hz {b.S[ip]:.3f} | H 피크 {b.H.max():.3f} | "
                  f"τ 비 중앙 {b.tr.median():.2f} | |L| {b.L.median():.3f} ∠L(7–8.5 Hz) "
                  + " ".join(f"{fb:.2f}:{ph:+.0f}°" for fb, ph in b.Lph[(b.index >= 7) & (b.index <= 8.5)].items()))
            w = b[(b.index >= 7) & (b.index <= 8.5)]
            print(f"      7–8.5 Hz ∠H(플랜트) " + " ".join(f"{fb:.2f}:{ph:+.0f}°" for fb, ph in w.Hph.items())
                  + (" | ∠(−C)(제어기) " + " ".join(f"{fb:.2f}:{ph:+.0f}°" for fb, ph in w.nCph.items()) if k else ""))
            if a.fmax > 12:  # 고주파 개루프: |L| = k·|H|·|v̂/x| 예측(8절 10/26 밤 5, 20–40 Hz < 0.3)
                from vib_damp import VibDampCfg, pred_gain
                age = float(np.median(v.age_ms.values[v.age_ms.values < 200])) / 1000
                hf = g[g.index >= 12]; pg = np.abs(pred_gain(hf.index.values, age, VibDampCfg(lead_deg=a.lead)))
                print(f"      고주파 |H| (나이 중앙 {age * 1000:.0f} ms, φ_lead {a.lead:.0f}°): " + " ".join(f"{fb:.0f}:{h:.3f}" for fb, h in hf.H.items()))
                for kk in (0.2, 0.5, 1.0):
                    L = kk * hf.H.values * pg; bad = hf.index.values[(L >= 0.3) & (hf.index.values >= 20)]
                    print(f"      k {kk}: |L| 최대 {L.max():.3f} ({hf.index.values[L.argmax()]:.0f} Hz), 20–40 Hz ≥ 0.3인 칸 {list(bad) if len(bad) else '없음'}")
        if len(res) == 2:
            (k0, g0, _, _), (k1, g1, _, _) = sorted(res, key=lambda r: abs(r[0]))  # 기준 = k 0 구간
            b0 = g0[(g0.index >= BAND[0]) & (g0.index <= BAND[1])]; b1 = g1.reindex(b0.index)
            ratio = b1.S.max() / b0.S.max()
            Lc = b1.L * np.exp(1j * np.radians(b1.Lph))
            pred = (b0.S / np.abs(1 + Lc)).max() / b0.S.max()  # 칸마다 S₀/|1+L| → 피크 비(폐루프 피크 이동 반영)
            ip = b1.S.idxmax(); mg = 90 - abs(b1.Lph[ip])
            print(f"  ⇒ S_k/S₀ 피크 비 {ratio:.3f} (k {k1:.2f} vs {k0:.2f}) | 예측 max(S₀/|1+L|)/max(S₀) {pred:.3f}"
                  f" | k 구간 S 피크 {ip:.2f} Hz에서 |L| {b1.L[ip]:.3f} ∠L {b1.Lph[ip]:+.0f}° → 여유 {mg:.0f}°")
            out.append(ratio)
    if len(out) > 1:
        print(f"\nS_k/S₀ 중앙값 {np.median(out):.3f} ({len(out)}회: {np.round(out, 3)})")


if __name__ == "__main__":
    main()
