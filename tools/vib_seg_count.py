#!/usr/bin/env python3
"""판단 33 감쇠 가능 구간 집계(8절 10/29 오전 3 정의) — 기록 파일로 vib_plan._lockin·fit_mode를 그대로 재현.

구간: (나) AB 회(_hold.csv)는 처프 두 번(MIT 2–20 s k 0, 24–42 s k), (가) 계획 회(_plan.csv)는 id·ck 구간(확인용).
구간마다 H = v/τ_sent 처프 위상 lock-in(0.25 Hz bin 6.5–9 Hz, 충격 제외·선형 추세 제거) → fit_mode → ψ·잔차·설명력·절반 ψ·H 피크.
감쇠 가능 = H 피크 ≥ 0.5 · 잔차 ≤ 35° · 설명력 ≥ 0.7 · 절반 ψ 차 ≤ 30° 그리고 다음 구간과 |Δψ| ≤ 30°(마지막 구간은 분모 밖).
v: 기본은 vib.csv 실시간 값(제어 주기마다 마지막 표본, t_gyro), --v imu면 IMU 파일 표본마다 다시 계산(자이로 피치 − J_ω q̇, vib_chirp_cl과 같음). τ_sent는 본 CSV tff2.
사용: source install/setup.bash; python3 tools/vib_seg_count.py [--v imu] logs/mit/20261027_142200_hold.csv … [logs/mit/…_plan.csv]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "common_pkg"))
from imu_frame import static_tilt  # noqa: E402
from vib_plan import BINS, chirp_phase, fit_mode, parse, psi_with, wrap  # noqa: E402

F0, F1, DUR = 4.0, 12.0, 18.0


def segments(p, m):  # [(구간 시작 MIT 시각, 길이, k, 이름)]
    if p.endswith("_plan.csv"):
        j = json.load(open(p.replace(".csv", ".plan.json"))); segs = parse(j["plan"]); out = []; t = 0.0
        for i, (kind, d, kk) in enumerate(segs):
            dur = d / 0.25 if kind in ("sine", "onoff") else d
            if kind in ("id", "ck"):
                out.append((t, dur, kk, f"{kind}{i}"))
            t += dur
        return out
    v = pd.read_csv(p.replace(".csv", ".vib.csv")); k2 = float(v.k.max())
    return [(2.0, DUR, 0.0, "A"), (24.0, DUR, k2, "B")]


def align(mm, a, dur):  # 처프 열 ≈ A·sin(ph(t − a − δ))가 되는 δ[s] — 0–40 ms 격자(송신 지연)
    t = mm.t.values; c = mm.chirp.values; w = (t - a > 0.6) & (t - a < dur - 0.6); best = (np.inf, 0.0)
    for dl in np.arange(-0.01, 0.04, 0.0005):
        s = np.sin(chirp_phase(t[w] - a - dl, dur, F0, F1)[0]); A = np.dot(c[w], s) / np.dot(s, s); e = np.sum((c[w] - A * s) ** 2)
        best = min(best, (e, dl))
    return best[1]


def lockin(tu, ph, fi, vu, tauu):  # vib_plan._lockin과 같은 계산(L 없음)
    z = np.exp(-1j * ph); fs = 200.0
    res = vu - np.convolve(vu, np.ones(5) / 5, "same"); mad = np.median(np.abs(res - np.median(res))) / 0.6745
    bad = np.abs(res) > max(0.15, 8 * mad); keep = np.convolve(bad.astype(float), np.ones(int(0.3 * fs) + 1), "same") == 0
    rows = []
    for fb in BINS:
        w0 = np.abs(fi - fb) <= 0.125; w = w0 & keep
        if w.sum() < 20 or w.sum() < 0.5 * w0.sum():
            rows.append(None); continue
        idx = np.where(w)[0]; out = []
        for part in (idx, idx[:len(idx) // 2], idx[len(idx) // 2:]):
            x = tu[part]; dt = lambda y: y - np.polyval(np.polyfit(x, y, 1), x)  # noqa: E731
            out.append(np.mean(dt(vu[part]) * z[part]) / np.mean(dt(tauu[part]) * z[part]))
        rows.append(out)
    ok = [i for i, r in enumerate(rows) if r is not None]
    if len(ok) < 5:
        return None
    return BINS[ok], np.array([rows[i][0] for i in ok]), np.array([rows[i][1] for i in ok]), np.array([rows[i][2] for i in ok])


def v_imu(p, a, b, ctx):  # IMU 파일로 표본 단위 v(장치 시각 정렬, 엔코더 튐 보간)
    import vib_chirp_frf as F
    import vib_d2_estimator as E
    from link_est import jacobians
    kin, tb = ctx; tu, Q, QD, G, _ = E.load_aligned(p, a - 0.2, b + 0.2, kin, tb)
    for j in range(6):
        QD[:, j], _ = F.despike(tu, QD[:, j])
    ey = np.array([jacobians(kin, q, tb.R_l6)[0][1] @ qd for q, qd in zip(Q, QD)])
    return tu, E.band(G[:, 1] - ey, 1.5, 45)


def run(p, src="vib", ctx=None):
    d = pd.read_csv(p); m = d[d.phase == "mit"]; v = pd.read_csv(p.replace(".csv", ".vib.csv")); tm0 = m.t.values[0]
    out = []
    for s0, dur, kk, name in segments(p, m):
        a, b = tm0 + s0, tm0 + s0 + dur; mm = m[(m.t >= a) & (m.t <= b)]; vv = v[(v.t_gyro >= a - 0.2) & (v.t_gyro <= b + 0.2)]
        if src == "imu":
            ti, vi = v_imu(p, a, b, ctx); vv = pd.DataFrame({"t_gyro": ti, "v": vi})
        a += align(mm, a, dur)  # 처프 위상 기준 = 실시간 판정의 ts(주기 시작) — 본 CSV 처프 열에 맞춰 송신 지연만큼 옮김
        tu = np.arange(max(mm.t.values[0], vv.t_gyro.values[0]), min(mm.t.values[-1], vv.t_gyro.values[-1]), 1 / 200.0)
        vu = np.interp(tu, vv.t_gyro.values, vv.v.values); tauu = np.interp(tu, mm.t.values, mm.tff2.values)
        ph, fi = chirp_phase(tu - a, dur, F0, F1); r = lockin(tu, ph, fi, vu, tauu)
        if r is None:
            out.append(dict(seg=name, k=kk, ok=False)); continue
        f, H, H1, H2 = r; f0, z, c, res, r2 = fit_mode(f, H); psi = float(np.degrees(np.angle(c)))
        p1, p2 = psi_with(f, H1, f0, z), psi_with(f, H2, f0, z); hp = float(np.abs(H).max())
        gate = hp >= 0.5 and res <= 35 and r2 >= 0.7 and abs(wrap(p1 - p2)) <= 30
        out.append(dict(seg=name, k=kk, ok=True, f0=f0, zeta=z, psi=psi, res=res, r2=r2, half=(p1, p2), h=hp, gate=gate))
    for x, y in zip(out, out[1:] + [None]):
        x["dpsi"] = None if (y is None or not x["ok"] or not y["ok"]) else wrap(y["psi"] - x["psi"])
        x["poss"] = None if y is None else bool(x["ok"] and x["gate"] and x["dpsi"] is not None and abs(x["dpsi"]) <= 30)
    s0 = segments(p, m)[0][0]  # 기울기 = 첫 처프 구간 앞 유지(계획 회는 첫 식별, AB 회는 MIT 시작)
    return out, static_tilt(p, tm0 + s0 + 0.3, tm0 + s0 + 1.8) if s0 > 2 else static_tilt(p, tm0 + 0.3, tm0 + 1.8)


if __name__ == "__main__":
    args = sys.argv[1:]; src = "vib"; ctx = None
    if args[:1] == ["--v"]:
        src, args = args[1], args[2:]
    if src == "imu":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from imu_frame import ImuToBase
        from piper_kin import PiperKin
        ctx = (PiperKin(), None); ctx = (ctx[0], ImuToBase(ctx[0]))
    for p in args:
        segs, tl = run(p, src, ctx); cnt = [s for s in segs if s["poss"] is not None]; n = sum(s["poss"] for s in cnt)
        best = cur = 0
        for s in segs:
            cur = cur + 1 if s["poss"] else 0; best = max(best, cur)
        print(f"{os.path.basename(p)[:15]} 기울기 {tl:.2f}° | 가능 {n}/{len(cnt)} 최장 {best} 구간")
        for s in segs:
            if not s["ok"]:
                print(f"  {s['seg']} k {s['k']}: bin 부족"); continue
            print(f"  {s['seg']} k {s['k']}: H {s['h']:.3f} 잔차 {s['res']:.1f}° 설명력 {s['r2']:.2f} 절반 {s['half'][0]:+.0f}/{s['half'][1]:+.0f} → ψ {s['psi']:+.0f}° f₀ {s['f0']:.2f} ζ {s['zeta']:.2f}"
                  f" | {'G' if s['gate'] else '-'}" + ("" if s["dpsi"] is None else f" Δψ {s['dpsi']:+.0f}") + (" ★" if s["poss"] else ""))
