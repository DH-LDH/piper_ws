#!/usr/bin/env python3
"""MIT 순간 튐의 CAN 원본 통계 — logs/mit 실기 기록 중 .candump가 있는 것의 사건마다 전류 방향·선행 시간·명령 프레임 변화·명령 프레임 간격·상태 프레임을 낸다(9절 11/18 14:10). 읽기만 한다.

사건: mit_jump_report.events()(|q̇ − q̇_ref| > 0.5 rad/s). 시각 0(tz) = 처음 튄 관절의 고속 피드백(0x251–0x256)에서 속도가 직전 값보다 0.5 rad/s 넘게 벗어난 첫 프레임.
열: cur_pre·cur_first·cur_peak[A](직전 중앙값 / 문턱을 처음 넘은 프레임 / 그 뒤 5프레임 중 가장 벗어난 값), cur_same_dir(처음 넘은 프레임의 전류 변화 부호 = 움직인 방향), lead_ms(속도 0.2 rad/s 벗어난 첫 프레임 − 전류가 문턱 넘은 첫 프레임, +면 전류가 먼저. 피드백 5 ms 간격),
    cmd_n100(그 관절 명령 프레임 종류 수, 전후 100 ms), cmd_pre_same(tz 전 100 ms 동안 명령이 한 종류), gap_joint_ms·gap_all_ms(tz 직전 50 ms에 걸친 명령 프레임 간격 최댓값, 그 관절 / 여섯 관절),
    st_n(0x2A1 종류 수, 전후 100 ms), low_n·low_status(그 관절 저속 피드백 상태 바이트 종류 수·값, 전후 200 ms).
대조: 같은 기록에서 사건과 1.5 s 넘게 떨어진 시각(50 ms마다)의 gap_all_ms·관절별 간격 분포 → --ctrl-out.
사용: python3 tools/mit_jump_can.py --out docs/data/mit_jumps_can.csv --ctrl-out docs/data/mit_cmd_gap_ctrl.csv
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mit_jump_report import ROOT, SKIP, events  # noqa: E402


def s16(b):
    v = int.from_bytes(b, "big"); return v - 65536 if v > 32767 else v


def load(path):  # → fb {관절: (t, 속도, 전류)}, cmd {관절: (t[], hex[])}, st [(t, hex)], low {관절: [(t, 상태 바이트)]}
    fb = {j: [] for j in range(1, 7)}; ct = {j: [] for j in range(1, 7)}; ch = {j: [] for j in range(1, 7)}; st = []; low = {j: [] for j in range(1, 7)}
    for ln in open(path):
        p = ln.split()
        if len(p) < 3:
            continue
        cid, _, dat = p[2].partition("#")
        if len(cid) != 3 or len(dat) != 16:
            continue
        c = int(cid, 16)
        if 0x251 <= c <= 0x256:
            fb[c - 0x250].append((float(p[0][1:-1]), s16(bytes.fromhex(dat[0:4])) / 1000, s16(bytes.fromhex(dat[4:8])) / 1000))
        elif 0x15A <= c <= 0x15F:
            ct[c - 0x159].append(float(p[0][1:-1])); ch[c - 0x159].append(dat)
        elif c == 0x2A1:
            if not st or st[-1][1] != dat:
                st.append((float(p[0][1:-1]), dat))
        elif 0x261 <= c <= 0x266:
            low[c - 0x260].append((float(p[0][1:-1]), int(dat[10:12], 16)))
    return {j: np.array(v) for j, v in fb.items()}, {j: (np.array(ct[j]), ch[j]) for j in ct}, st, low


def gap_at(tc, tz, w=0.05):  # tz 직전 w에 걸친 명령 프레임 간격 최댓값[ms] — 프레임이 없으면 nan
    if len(tc) < 2:
        return float("nan")
    i1 = int(np.searchsorted(tc, tz - w)); i2 = int(np.searchsorted(tc, tz))
    lo = max(i1 - 1, 0); hi = min(i2, len(tc) - 1)
    if hi <= lo:
        return float("nan")
    d = np.diff(tc[lo:hi + 1]); last = tz - tc[hi] if tc[hi] < tz else 0.0  # 마지막 프레임 뒤로 tz까지 비어 있으면 그 시간도 봄
    return float(max(d.max(), last) * 1000)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=""); ap.add_argument("--ctrl-out", default=""); ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args(); rows = []; ctrl = []; n_rec = n_cd = 0
    for p in sorted(glob.glob(os.path.join(ROOT, "2026*.csv"))):
        b = os.path.basename(p)
        if any(s in b for s in SKIP) or ".nwin" in b:
            continue
        ev, info = events(p, a.thr)
        if info is None:
            continue
        n_rec += 1; cd = p.replace(".csv", ".candump"); tp = p.replace(".csv", ".t0")
        if not (os.path.exists(cd) and os.path.exists(tp)):
            continue
        n_cd += 1; d = pd.read_csv(p, usecols=["t", "phase"]); tm = d.t.values[d.phase.values == "mit"]; tm0 = float(open(tp).read()) + tm[0]; tend = tm0 + tm[-1] - tm[0]
        fb, cmd, st, low = load(cd)
        for e in ev:
            j = e["joint"]; F = fb[j]; te = tm0 + e["t_mit"]; r = dict(file=b, date=b[:8], t_mit=e["t_mit"], joint=j, kind=e["kind"], disp_deg=e["disp_deg"], toward0=e["toward0"], ours_on_joint=e["ours_on_joint"], fw_kp_joint=e["fw_kp_joint"], joint_moving=e["joint_moving"], t_to_end=e["t_to_end"], abort_last=bool(e["abort"] and e["last_event"]), can_ok=False)
            w = (F[:, 0] > te - 0.30) & (F[:, 0] < te + 0.08) if len(F) else np.zeros(0, bool)  # 기록 시각이 CAN보다 늦다(중앙 9 ms, τ_vib·계획 기록은 0.1 s 넘게도)
            if w.sum() > 10:
                G = F[w]; pre = G[:, 0] < te - 0.22
                if pre.sum() >= 3:
                    sp0 = np.median(G[pre, 1]); k = np.where((np.abs(G[:, 1] - sp0) > a.thr) & ~pre)[0]
                    if len(k):
                        tz = G[k[0], 0]; P = F[(F[:, 0] > tz - 0.12) & (F[:, 0] < tz - 0.03)]; W = F[(F[:, 0] >= tz - 0.04) & (F[:, 0] <= tz + 0.03)]
                        if len(P) >= 5 and len(W) >= 3:
                            c0 = float(np.median(P[:, 2])); cs = float(1.4826 * np.median(np.abs(P[:, 2] - c0))); cthr = max(0.2, 6 * cs); s0 = float(np.median(P[:, 1]))
                            ic = np.where(np.abs(W[:, 2] - c0) > cthr)[0]; isp = np.where(np.abs(W[:, 1] - s0) > 0.2)[0]
                            tc_, hx = cmd[j]; m100 = (tc_ > tz - 0.1) & (tc_ < tz + 0.1); mpre = (tc_ > tz - 0.1) & (tc_ <= tz)
                            r.update(can_ok=True, tz_off_ms=round((tz - te) * 1000, 1), cur_pre=round(c0, 3), cur_thr=round(cthr, 3))
                            if len(ic) and len(isp):
                                pk = W[ic[0]:ic[0] + 5, 2]; cp = float(pk[np.abs(pk - c0).argmax()]); sd = float(W[isp[0], 1] - s0)
                                r.update(cur_peak=round(cp, 3), cur_first=round(float(W[ic[0], 2]), 3), cur_same_dir=bool(np.sign(W[ic[0], 2] - c0) == np.sign(G[k[0], 1] - sp0)), cur_peak_same_dir=bool(np.sign(cp - c0) == np.sign(G[k[0], 1] - sp0)), lead_ms=round((W[isp[0], 0] - W[ic[0], 0]) * 1000, 1), spd_first=round(sd, 3))
                            r.update(cmd_n100=len({hx[i] for i in np.where(m100)[0]}), cmd_pre_same=len({hx[i] for i in np.where(mpre)[0]}) <= 1, gap_joint_ms=round(gap_at(tc_, tz), 2),
                                     gap_all_ms=round(float(np.nanmax([gap_at(cmd[x][0], tz) for x in range(1, 7)])), 2), 
                                     low_n=len({s for t_, s in low[j] if tz - 0.2 < t_ < tz + 0.2}), low_status=";".join(sorted({format(s, "08b") for t_, s in low[j] if tz - 0.2 < t_ < tz + 0.2})))
                            r["st_n"] = 1 + sum(1 for t_, h in st if tz - 0.1 < t_ < tz + 0.1)  # st는 값이 바뀐 시각만 담음 → 창 안의 변화 수 + 1
            rows.append(r)
        te_all = np.array([tm0 + e["t_mit"] for e in ev])
        for tq in np.arange(tm0 + 1.0, tend - 0.3, 0.05):  # 대조: 사건과 1.5 s 넘게 떨어진 시각
            if len(te_all) and np.abs(te_all - tq).min() < 1.5:
                continue
            g = [gap_at(cmd[x][0], tq) for x in range(1, 7)]
            if np.isfinite(g).all():
                ctrl.append((b[:8], *g))
        print(f"{b[:15]} 사건 {len(ev)} 대조 시각 누적 {len(ctrl)}", flush=True)
    R = pd.DataFrame(rows); C = pd.DataFrame(ctrl, columns=["date"] + [f"j{x}" for x in range(1, 7)]); C["all"] = C[[f"j{x}" for x in range(1, 7)]].max(1)
    print(f"\n기록 {n_rec}개 중 candump 있는 것 {n_cd}개 | 사건 {len(R)}건, CAN에서 잡힌 것 {int(R.can_ok.sum())}건")
    if a.out:
        R.to_csv(a.out, index=False); print("표:", a.out)
    if a.ctrl_out:
        q = lambda x: dict(n=len(x), med=np.median(x), p90=np.percentile(x, 90), p99=np.percentile(x, 99), max=np.max(x), over15=np.mean(x > 15), over20=np.mean(x > 20), over30=np.mean(x > 30))
        ok = R[R.can_ok]; T = pd.DataFrame([dict(group="event_all_joints", **q(ok.gap_all_ms.dropna().values)), dict(group="ctrl_all_joints", **q(C["all"].values)),
                                             dict(group="event_own_joint", **q(ok.gap_joint_ms.dropna().values)), dict(group="ctrl_per_joint_pooled", **q(C[[f"j{x}" for x in range(1, 7)]].values.ravel()))])
        T = pd.concat([T, pd.DataFrame([dict(group=f"{g}_all_joints@{dt}", **q(x)) for dt in sorted(set(C.date)) for g, x in (("event", ok[ok.date.astype(str) == dt].gap_all_ms.dropna().values), ("ctrl", C[C.date == dt]["all"].values)) if len(x)])])  # 날짜별(기록 구성이 날마다 다름)
        T.round(4).to_csv(a.ctrl_out, index=False); print(T.round(3).to_string()); print("대조 표:", a.ctrl_out)


if __name__ == "__main__":
    main()
