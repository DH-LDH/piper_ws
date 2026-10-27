#!/usr/bin/env python3
"""(가) 계획 시행 보고 — 게이트 | ck별 φ_lead·ψ_L·공진 |L|·|1+L|·클립 | on/off 짝 IMU 5–10 Hz rms 비 | 재식별 Δψ | 기울기 시작·끝 (8절 10/27 오후 10 형식).

on/off 짝: onoff 구간 주기 1·2, 3·4, 5·6(홀수 on, 짝수 off), 주기마다 앞뒤 0.5 s 버리고(k 램프·전환), MIT 순간 튐 ±0.5 s 제외.
IMU는 실시간 기록(.imu.csv, 가속도 63 Hz·자이로 200 Hz) — 장치 시각 + 누적 하위 1% 오프셋, 카메라 축 acc z·gyro x.
사용: python3 tools/vib_plan_report.py logs/mit/2026…_plan.csv
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "common_pkg"))
from imu_frame import static_tilt  # noqa: E402
from vib_plan import parse  # noqa: E402


def bp_rms(t, x, keep, fs):  # 균일 재표본 → 5–10 Hz 대역 → keep 구간 rms
    tu = np.arange(t[0], t[-1], 1 / fs); xu = np.interp(tu, t, x); X = np.fft.rfft(xu - xu.mean()); f = np.fft.rfftfreq(len(xu), 1 / fs)
    X[(f < 5) | (f > 10)] = 0; y = np.fft.irfft(X, len(xu)); k = np.interp(tu, t, keep.astype(float)) > 0.99
    return float(np.sqrt(np.mean(y[k] ** 2))) if k.sum() > 20 else float("nan")


def report(p):
    j = json.load(open(p.replace(".csv", ".plan.json"))); d = pd.read_csv(p); m = d[d.phase == "mit"]; tm0 = m.t.values[0]
    segs = parse(j["plan"]); t0s = np.cumsum([0] + [dd / 0.25 if k in ("sine", "onoff") else dd for k, dd, _ in segs])
    print(f"\n== {os.path.basename(p)[:15]} | 계획 {j['plan']}")
    for r in j["results"]:
        if r["kind"] == "id":
            s = (f"  식별(구간 {r['seg']}): H 피크 {r.get('h_peak')} 잔차 {r.get('res')}° 설명력 {r.get('r2')} 절반 ψ {r.get('psi_half')} → ψ {r.get('psi')}° f₀ {r.get('f0')} Hz ζ {r.get('zeta')}")
            s += (f" | 게이트 {'통과' if r.get('gate') else '불통과'}" + (f", φ_lead {r['lead']:.0f}°, 허용 k ≤ {r['k_cap']}" if "lead" in r else "")) if "dpsi" not in r else f" | 재식별 Δψ {r['dpsi']:+.0f}°" + (" ★ ±30° 넘음 → on/off 짝 주 판정 제외" if abs(r["dpsi"]) > 30 else "")
            print(s)
        else:
            if not r.get("on"):
                print(f"  ck k {r['k']}(구간 {r['seg']}): 안 켬"); continue
            ok = r["passed"]; good = ok and r["one_plus_L"] > 1.2
            print(f"  ck k {r['k']}(구간 {r['seg']}): φ_lead {r['lead_used']:.0f}° ψ_L {r['psi_L']:+.0f}° 공진 |L| {r['L_peak']} |1+L| {r['one_plus_L']} 클립 {100 * r['clip_frac']:.1f}% "
                  f"플랜트 Δψ {r['drift']:+.0f}° → {'통과' if ok else '불통과'}" + (" ★ |ψ_L| ≤ 30°·|1+L| > 1.2" if good else ""))
    print(f"  k_ok {j['k_ok']}, on/off φ_lead {j.get('lead_ok')}")
    oi = [i for i, (k, _, _) in enumerate(segs) if k == "onoff"]
    if oi and j["k_ok"] > 0:
        i = pd.read_csv(p.replace(".csv", ".imu.csv")); t0 = float(open(p.replace(".csv", ".t0")).read())
        off = i.t.values - i.t_dev.values / 1000; okk = np.abs(off - np.median(off)) < 1.0; i = i[okk].copy()
        i["t"] = i.t_dev / 1000 + np.percentile(off[okk], 1) - t0 - tm0
        tt = m.t.values - tm0; qd = np.abs(np.diff(np.radians(m[[f"q{k}" for k in range(1, 7)]].values), axis=0)) / np.maximum(np.diff(tt)[:, None], 1e-3)
        sp = tt[1:][(qd > 1.0).any(1)]  # 순간 튐(a′ 속도 0.14 rad/s보다 훨씬 큼)
        ts0 = t0s[oi[0]]; n = int(segs[oi[0]][1]); rms = []
        for c in range(n):
            a, b = ts0 + 4 * c + 0.5, ts0 + 4 * (c + 1) - 0.5; row = []
            for st, ax, fs in (("a", "z", 63.0), ("g", "x", 200.0)):
                s = i[(i.stream == st) & (i.t >= a - 1) & (i.t <= b + 1)]; tsv = s.t.values
                keep = (tsv >= a) & (tsv <= b)
                for q in sp:
                    keep &= np.abs(tsv - q) > 0.5
                row.append(bp_rms(tsv, s[ax].values, keep, fs))
            rms.append(row)
        rms = np.array(rms)
        for c in range(0, n - 1, 2):
            print(f"  짝 {c // 2 + 1}: on acc z {rms[c, 0]:.3f} / off {rms[c + 1, 0]:.3f} → 비 {rms[c, 0] / rms[c + 1, 0]:.2f} | gyro x 비 {rms[c, 1] / rms[c + 1, 1]:.2f}")
    elif oi:
        print("  on/off: k_ok 0 → 짝 없음(기준선 a′)")
    f = p
    print(f"  기울기: MIT 시작 {static_tilt(f, tm0 + 0.3, tm0 + 1.8):.2f}° | 첫 식별 직전 {static_tilt(f, tm0 + t0s[1] - 1.5, tm0 + t0s[1]) if len(t0s) > 1 else float('nan'):.2f}° | 끝 {static_tilt(f, m.t.values[-1] - 1.5, m.t.values[-1]):.2f}°(모두 유지 자세)")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
