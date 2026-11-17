#!/usr/bin/env python3
"""MIT 순간 튐 정리 — logs/mit 실기 기록의 mit 구간에서 |q̇ − q̇_ref| > 문턱인 사건을 찾아 사건 표와 집계를 낸다. 읽기만 한다(CAN 미사용).

사건: 문턱을 넘은 표본을 0.3 s 안이면 하나로 묶음. 처음 튄 관절 = 사건 첫 표본에서 |q̇ − q̇_ref|가 가장 큰 관절. q̇_ref는 qref의 ±4표본 차분.
위치 튐 / 속도만 튐: 처음 튄 관절의 이탈 변화(사건 전 0.3 s 중앙값 대비)가 0.1° 이상이면 위치 튐, 아니면 엔코더 속도만 튄 것.
유형: jump = 직전 0.12 s가 조용하고(|q̇ − q̇_ref| < 0.25) 0.2 s 안에 끝난 위치 튐, osc = 그 밖의 위치 튐(헌팅·발산·연속 튐), speed_only.
자이로: .imu.csv가 있으면 사건 창(−0.06–+0.2 s, 수신 시각)의 자이로 변화 최대[rad/s] — 실제로 끝단이 움직였는지.
펌웨어 PD: .candump의 첫 MIT 프레임(0x15A–0x15F)에서 Kp·Kd를 읽음. 우리 토크: 그 관절 tff 열이 mit 구간에서 변했는지.
중단 추정: mit 끝 0.15 s 안에 이탈 2.9° 초과, 속도 초과 3주기 연속, 모드·Enable 이상 중 하나(기록에 종료 사유가 없어 추정).
사용: python3 tools/mit_jump_report.py [--out docs/data/mit_jumps.csv] [--png docs/img/p_fw191_d_jumps.png] [--thr 0.5]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "mit")
SKIP = ("fake", "dry", ".imu.", ".tdc", ".vib.", ".gq7.")
BINS = [-11, -4, -2, 0, 2, 4, 11]  # 기준각 − 유지각[°]
VB = [0.3, 2, 4, 6, 8, 30]  # |기준 속도|[°/s]
J = range(1, 7)


def fw_gains(path):  # .candump 첫 MIT 프레임 → (Kp[6], Kd[6]) 또는 None
    cd = path.replace(".csv", ".candump")
    if not os.path.exists(cd):
        return None
    kp = [None] * 6; kd = [None] * 6
    with open(cd) as f:
        for ln in f:
            p = ln.split()
            if len(p) < 3 or "#" not in p[2]:
                continue
            cid, dat = p[2].split("#")
            c = int(cid, 16)
            if 0x15A <= c <= 0x15F and len(dat) == 16 and kp[c - 0x15A] is None:
                d = bytes.fromhex(dat)
                kp[c - 0x15A] = round((((d[3] & 0x0F) << 8) | d[4]) / 4095 * 500, 1)
                kd[c - 0x15A] = round(((d[5] << 4) | (d[6] >> 4)) / 4095 * 10 - 5, 2)
                if None not in kp:
                    break
    return (kp, kd) if None not in kp else None


def events(path, thr, gap=0.3):
    d = pd.read_csv(path); m = d[d.phase == "mit"].reset_index(drop=True)
    if len(m) < 20:
        return [], None
    t = m.t.values - m.t.values[0]; dt = float(np.median(np.diff(t)))
    q = m[[f"q{j}" for j in J]].values; qr = m[[f"qref{j}" for j in J]].values; qd = m[[f"qd{j}" for j in J]].values
    eff = m[[f"eff{j}" for j in J]].values; tff = m[[f"tff{j}" for j in J]].values
    k = 4; n_ = len(t); ia = np.clip(np.arange(n_) - k, 0, n_ - 1); ib = np.clip(np.arange(n_) + k, 0, n_ - 1)  # 기준 속도: ±4표본 차분(기록 시각과 기준 계산 시각이 어긋나 한 표본 차분은 튐)
    vr = np.radians((qr[ib] - qr[ia]) / np.maximum(t[ib] - t[ia], 1e-3)[:, None]); dv = qd - vr; dev = q - qr
    refbad = np.abs(vr).max(1) > 1.0  # 기준 궤적 불연속(10-27 전 사인 꼬리 등) — 그 주변은 사건에서 뺌
    refbad = np.convolve(refbad.astype(float), np.ones(2 * k + 1), "same") > 0
    ours = [bool(np.ptp(tff[:, j]) > 2e-3) for j in range(6)]
    imu = None; ip = path.replace(".csv", ".imu.csv"); tp = path.replace(".csv", ".t0")
    if os.path.exists(ip) and os.path.exists(tp):
        try:
            gi = pd.read_csv(ip); gi = gi[gi.stream == "g"]; imu = (gi.t.values - float(open(tp).read()) - m.t.values[0], gi[["x", "y", "z"]].values)
        except Exception:
            imu = None
    q_hold = qr[0]
    expo = {}  # 사인 관절의 (기준각 − 유지각) 칸별 머문 시간[s] — 관절각 의존을 노출 시간으로 나눠 보기 위함
    for j in (1, 2):
        mvj = np.abs(vr[:, j]) > 0.005
        expo[j + 1] = np.histogram((qr[:, j] - q_hold[j])[mvj], bins=BINS)[0] * dt if mvj.any() else np.zeros(len(BINS) - 1)
        expo[10 + j + 1] = np.histogram(np.abs(np.degrees(vr[:, j])), bins=VB)[0] * dt
    mvs = [float((np.abs(vr[:, j]) > 0.005).sum() * dt) for j in range(6)]  # 관절별 기준이 움직인 시간
    fw = fw_gains(path)
    moving = np.abs(vr).max(1) > 0.005
    name = os.path.basename(path); stage = "plan" if "_plan" in name else "sine" if "_sine" in name else "hold"
    tail = t > t[-1] - 0.15
    over = (np.abs(dv) > thr).max(1) & ~refbad
    run = 0; consec = 0
    for x in over[tail]:
        run = run + 1 if x else 0; consec = max(consec, run)
    abort = bool(np.abs(dev[tail]).max() > 2.9 or consec >= 3 or (m.mode_feed.values[tail] != 4).any() or (m.enabled.values[tail] != 1).any())
    info = dict(refbad_s=round(float(refbad.sum() * dt), 2), file=name, date=name[:8], stage=stage, mit_s=round(float(t[-1]), 2), move_s=round(float(moving.sum() * dt), 2), rate_hz=round(1 / dt),
                ours="".join(str(j + 1) for j in range(6) if ours[j]), fw_kp="" if fw is None else " ".join(f"{v:g}" for v in fw[0]),
                fw_kd="" if fw is None else " ".join(f"{v:g}" for v in fw[1]), abort=abort, chirp="chirp" in m.columns and bool(np.abs(m.chirp.values).max() > 0),
                tau_vib="tau_vib" in m.columns and bool(np.abs(m.tau_vib.values).max() > 0), imu=imu is not None,
                expo2=expo[2], expo3=expo[3], vexpo2=expo[12], vexpo3=expo[13], mvs=mvs)
    idx = np.where(over)[0]; out = []
    if not len(idx):
        return out, info
    groups = [[idx[0]]]
    for i in idx[1:]:
        if t[i] - t[groups[-1][-1]] <= gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    for g in groups:
        i0 = g[0]; j0 = int(np.abs(dv[i0]).argmax()); t0 = t[i0]
        pre = (t > t0 - 0.3) & (t < t0 - 0.025); win = (t >= t0 - 0.03) & (t <= t0 + 0.5)
        base = np.median(dev[pre], axis=0) if pre.sum() >= 3 else dev[max(i0 - 1, 0)]
        dd = dev[win] - base; disp = np.abs(dd).max(0)  # 관절별 이탈 변화 최대[°]
        k = int(np.abs(dd[:, j0]).argmax()); sgn = float(np.sign(dd[k, j0])); q0 = float(q[max(i0 - 1, 0), j0])
        toward0 = "" if abs(q0) < 2.0 or disp[j0] < 0.1 else ("yes" if sgn * np.sign(q0) < 0 else "no")
        follow = sorted({int(np.abs(dv[i]).argmax()) + 1 for i in g} | {j + 1 for j in range(6) if (np.abs(dv[g, j]) > thr).any()} - {j0 + 1})
        rec = float("nan")  # 복귀: 이탈 변화가 0.2° 안으로 0.1 s 이어진 첫 시각
        n = max(int(round(0.1 / dt)), 2); ok = (np.abs(dev[:, j0] - base[j0]) < 0.2)
        for i in range(g[-1], len(t) - n):
            if ok[i:i + n].all():
                rec = float(t[i] - t0); break
        eb = np.median(eff[pre, j0]) if pre.sum() >= 3 else eff[max(i0 - 1, 0), j0]
        mad = np.median(np.abs(eff[pre, j0] - eb)) if pre.sum() >= 3 else 0.0
        ethr = max(0.8 if j0 < 3 else 0.3, 6 * mad); lead = ""  # 손목은 토크가 작아 문턱을 낮춤
        for i in range(max(i0 - 5, 0), min(i0 + 3, len(t))):
            if abs(eff[i, j0] - eb) > ethr:
                lead = i0 - i; break
        epk = float(np.abs(eff[win, j0] - eb).max())
        qb = (t > t0 - 0.12) & (t < t0 - 0.005); quiet = bool(not qb.any() or np.abs(dv[qb, j0]).max() < 0.25)
        sw = (t >= t0 - 0.03) & (t <= t0 + 0.05); step = float(np.abs(np.diff(dev[sw, j0])).max()) if sw.sum() > 1 else 0.0
        dur = float(t[g[-1]] - t0)
        tq = (t > t0 - 0.06) & (t < t0 - 0.005); tpre = float(np.ptp(tff[tq, j0])) if tq.sum() > 1 else 0.0  # 사건 직전 50 ms 우리 토크 변화폭
        kind = "speed_only" if disp[j0] < 0.1 else "jump" if quiet and dur <= 0.2 else "osc"
        gpk = float("nan")
        if imu is not None:
            wg = (imu[0] > t0 - 0.06) & (imu[0] < t0 + 0.2); bg = (imu[0] > t0 - 0.5) & (imu[0] < t0 - 0.08)
            if wg.sum() > 5 and bg.sum() > 5:
                gpk = float(np.abs(imu[1][wg] - np.median(imu[1][bg], axis=0)).max())
        out.append(dict(file=name, date=name[:8], stage=stage, t_mit=round(float(t0), 2), t_to_end=round(float(t[-1] - t0), 2), joint=j0 + 1,
                        followers="".join(str(x) for x in follow), n_samp=len(g), dur_ms=round(float(t[g[-1]] - t0) * 1000),
                        peak_dv=round(float(np.abs(dv[g, j0]).max()), 2), peak_dv_all=round(float(np.abs(dv[g]).max()), 2),
                        disp_deg=round(float(disp[j0]), 3), disp_max_deg=round(float(disp.max()), 3), disp_max_joint=int(disp.argmax()) + 1,
                        kind=kind, step_deg=round(step, 3), gyro_pk=round(gpk, 2), toward0=toward0, q_joint=round(q0, 2),
                        qref_rel=round(float(qr[i0, j0] - q_hold[j0]), 2), tff_pre_ptp=round(tpre, 3), joint_moving=bool(abs(vr[i0, j0]) > 0.005),
                        vref_dps=round(float(np.degrees(vr[i0, j0])), 2), moving=bool(moving[i0]), recover_s=round(rec, 2),
                        eff_lead_samp=lead, eff_peak=round(epk, 2), ours_on_joint=ours[j0], fw_kp_joint="" if fw is None else fw[0][j0],
                        fw_kd_joint="" if fw is None else fw[1][j0], abort=abort, last_event=False))
    if out and abort:
        out[-1]["last_event"] = True
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--thr", type=float, default=0.5, help="|q̇ − q̇_ref| 문턱[rad/s]")
    ap.add_argument("--out", default=""); ap.add_argument("--png", default=""); ap.add_argument("--files-out", default="", help="기록별 요약 CSV")
    a = ap.parse_args()
    fs = sorted(f for f in glob.glob(os.path.join(a.root, "*.csv")) if not any(x in os.path.basename(f) for x in SKIP))
    ev = []; infos = []
    for f in fs:
        try:
            e, info = events(f, a.thr)
        except Exception as ex:  # 형식이 다른 옛 기록은 건너뜀
            print(f"건너뜀 {os.path.basename(f)}: {type(ex).__name__} {ex}"); continue
        if info is None:
            continue
        for k in ("jump", "osc", "speed_only"):
            info["n_" + k] = sum(x["kind"] == k for x in e)
        infos.append(info); ev += e
    E = pd.DataFrame(ev); F = pd.DataFrame(infos)
    tot = F.mit_s.sum(); mv = F.move_s.sum(); P = E[E.kind == "jump"]; O = E[E.kind == "osc"]
    pm = lambda n, s: n / max(s, 1e-9) * 60
    print(f"실기 기록 {len(F)}개, MIT 누적 {tot:.0f} s({tot / 60:.1f}분), 그중 기준이 움직인 시간 {mv:.0f} s | 문턱 {a.thr} rad/s")
    print(f"사건 {len(E)}건: 튐(jump) {len(P)}, 헌팅·발산·연속(osc) {len(O)}, 속도만 {(E.kind == 'speed_only').sum()} → 튐 {pm(len(P), tot):.2f}건/분(평균 {tot / max(len(P), 1):.1f} s에 1건)")
    print(f"튐이 한 번도 없는 기록 {int((F.n_jump == 0).sum())}/{len(F)}개(그 기록들의 MIT 중앙 {F[F.n_jump == 0].mit_s.median():.1f} s)")
    print("\n[날짜별] 기록 수, MIT s, 튐, osc, 속도만, 튐/분, 중단 추정 기록")
    for dte, g in F.groupby("date"):
        print(f"  {dte}: {len(g)}, {g.mit_s.sum():.0f}, {g.n_jump.sum()}, {g.n_osc.sum()}, {g.n_speed_only.sum()}, {pm(g.n_jump.sum(), g.mit_s.sum()):.2f}, {int(g.abort.sum())}")
    print("\n[단계별] 기록 수, MIT s, 튐, /분")
    for st, g in F.groupby("stage"):
        print(f"  {st}: {len(g)}, {g.mit_s.sum():.0f}, {g.n_jump.sum()}, {pm(g.n_jump.sum(), g.mit_s.sum()):.2f}")
    print("\n[정지 대 운동] 튐 — 기준이 움직이는 중 %d건 / %.0f s = %.2f/분, 정지 중 %d건 / %.0f s = %.2f/분"
          % (P.moving.sum(), mv, pm(P.moving.sum(), mv), (~P.moving).sum(), tot - mv, pm((~P.moving).sum(), tot - mv)))
    print("\n[관절별 — 처음 튄 관절] 튐 수(/분), 이탈 변화 중앙/90%/최대[°], 최고 속도 중앙/최대[rad/s], 0° 쪽/판정 가능, 복귀 중앙[s], effort 선행·동시·뒤·없음, 자이로 변화 중앙[rad/s](n)")
    for j in J:
        p = P[P.joint == j]
        if not len(p):
            continue
        tz = p[p.toward0 != ""]; el = p[p.eff_lead_samp != ""]; ls = el.eff_lead_samp.astype(int); gp = p.gyro_pk.dropna()
        print(f"  J{j}: {len(p)}({pm(len(p), tot):.2f}), {p.disp_deg.median():.2f}/{p.disp_deg.quantile(0.9):.2f}/{p.disp_deg.max():.2f}, {p.peak_dv.median():.2f}/{p.peak_dv.max():.2f}, "
              f"{(tz.toward0 == 'yes').sum()}/{len(tz)}, {p.recover_s.median():.2f}, {(ls > 0).sum()}·{(ls == 0).sum()}·{(ls < 0).sum()}·{len(p) - len(el)}, "
              + (f"{gp.median():.2f}({len(gp)})" if len(gp) else "—"))
    print("\n[관절 설정별 노출 시간과 튐 빈도] 관절: 펌웨어 PD만 / 하이브리드(PD + 우리 토크) / 우리 토크만(Kp 0) — 튐 수 / 노출 s = 건/분")
    for j in J:
        row = []
        for nm, f_sel in (("PD만", lambda r: str(j) not in r.ours and r.kp[j - 1] > 0), ("하이브리드", lambda r: str(j) in r.ours and r.kp[j - 1] > 0), ("토크만", lambda r: r.kp[j - 1] == 0)):
            sel = [r.file for r in F.itertuples() if r.fw_kp != "" and f_sel(type("R", (), dict(ours=r.ours, kp=[float(x) for x in r.fw_kp.split()])))]
            sec = F[F.file.isin(sel)].mit_s.sum(); n = int(((P.joint == j) & P.file.isin(sel)).sum())
            row.append(f"{nm} {n}/{sec:.0f} = {pm(n, sec):.2f}" if sec > 0 else f"{nm} —")
        print(f"  J{j}: " + " | ".join(row))
    print("  같은 것을 그 관절이 정지 / 운동 중으로 나눔(건/분, 괄호는 튐 수 / 노출 s)")
    for j in (1, 2, 3, 5, 6):
        row = []
        for nm, f_sel in (("PD만", lambda o, k: str(j) not in o and k[j - 1] > 0), ("하이브리드", lambda o, k: str(j) in o and k[j - 1] > 0)):
            sel = F[[r.fw_kp != "" and f_sel(r.ours, [float(x) for x in r.fw_kp.split()]) for r in F.itertuples()]]
            m_s = float(np.sum([x[j - 1] for x in sel.mvs])); s_s = sel.mit_s.sum() - m_s; pj = P[(P.joint == j) & P.file.isin(sel.file)]
            row.append(f"{nm} 정지 {pm((~pj.joint_moving).sum(), s_s):.2f}({(~pj.joint_moving).sum()}/{s_s:.0f}) 운동 " + (f"{pm(pj.joint_moving.sum(), m_s):.2f}({pj.joint_moving.sum()}/{m_s:.0f})" if m_s > 1 else "—"))
        print(f"  J{j}: " + " | ".join(row))
    po = P[P.ours_on_joint]
    print("[우리 토크가 걸린 관절의 튐] %d건 중 직전 50 ms 토크 명령 변화폭 0.13 N·m(두 칸) 미만 %d건, 중앙 %.3f N·m" % (len(po), (po.tff_pre_ptp < 0.13).sum(), po.tff_pre_ptp.median() if len(po) else 0))
    print("\n[몰림] 진입 0.5 s 안 %d건, 종료 0.5 s 전 %d건(중단 추정 기록의 마지막 사건 %d건) / 튐 %d건. osc는 진입 0.5 s 안 %d건, 중단 추정 기록 안 %d건 / %d건"
          % ((P.t_mit < 0.5).sum(), (P.t_to_end < 0.5).sum(), P.last_event.sum(), len(P), (O.t_mit < 0.5).sum(), O.abort.sum(), len(O)))
    gaps = np.concatenate([np.diff(g.t_mit.values) for _, g in P.groupby("file") if len(g) > 1]) if len(P) else np.array([])
    if len(gaps):
        h, _ = np.histogram(gaps, bins=[0.3, 0.8, 1.4, 2, 3, 5, 10, 1e9])
        print("[튐 사이 간격] n %d, 중앙 %.2f s, 평균 %.2f s | 0.3–0.8: %d, 0.8–1.4: %d, 1.4–2: %d, 2–3: %d, 3–5: %d, 5–10: %d, 10 s 이상: %d" % (len(gaps), np.median(gaps), gaps.mean(), *h))
    print("\n[관절각 의존 — 사인으로 움직이는 중] 기준각 − 유지각 칸 " + " / ".join(f"{BINS[i]:g}–{BINS[i + 1]:g}" for i in range(len(BINS) - 1)) + "° : 튐 수 / 머문 s = 건/분")
    for j in (2, 3):
        ex = np.sum([np.asarray(x, float) for x in F[f"expo{j}"]], axis=0); p = P[(P.joint == j) & P.joint_moving]
        n = np.histogram(p.qref_rel, bins=BINS)[0]
        print(f"  J{j}: " + " | ".join(f"{n[i]}/{ex[i]:.0f} = {pm(n[i], ex[i]):.1f}" if ex[i] > 1 else "—" for i in range(len(n))))
    print("[기준 속도 의존 — 그 관절이 사인으로 움직이는 중] |기준 속도| 칸 " + " / ".join(f"{VB[i]:g}–{VB[i + 1]:g}" for i in range(len(VB) - 1)) + " °/s : 튐 수 / 머문 s = 건/분")
    for j in (2, 3):
        ex = np.sum([np.asarray(x, float) for x in F[f"vexpo{j}"]], axis=0); p = P[(P.joint == j) & P.joint_moving]
        n = np.histogram(p.vref_dps.abs(), bins=VB)[0]
        print(f"  J{j}: " + " | ".join(f"{n[i]}/{ex[i]:.0f} = {pm(n[i], ex[i]):.1f}" if ex[i] > 1 else "—" for i in range(len(n))))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); E.to_csv(a.out, index=False); print("사건 표:", a.out)
    if a.files_out:
        F.drop(columns=["expo2", "expo3", "vexpo2", "vexpo3", "mvs"]).to_csv(a.files_out, index=False); print("기록 표:", a.files_out)
    if a.png and len(P):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 3, figsize=(15, 8))
        lab = [f"J{j}" for j in J]; c1 = [int((P.joint == j).sum()) for j in J]; c2 = [int((O.joint == j).sum()) for j in J]
        ax[0, 0].bar(lab, c1, label="jump (quiet before, < 0.2 s)"); ax[0, 0].bar(lab, c2, bottom=c1, color="0.7", label="hunting / divergence / burst")
        ax[0, 0].set_ylabel("events (first joint)"); ax[0, 0].set_title(f"{len(P)} jumps in {tot / 60:.1f} min of MIT ({len(F)} runs): one per {tot / len(P):.1f} s", fontsize=10); ax[0, 0].legend(fontsize=8)
        for j in J:
            p = P[P.joint == j]
            if len(p):
                ax[0, 1].scatter(p.peak_dv, p.disp_deg, s=14, label=f"J{j}")
        ax[0, 1].set_xlabel("peak |qd - qd_ref| [rad/s]"); ax[0, 1].set_ylabel("deviation change [deg]"); ax[0, 1].set_xscale("log"); ax[0, 1].set_yscale("log")
        ax[0, 1].legend(fontsize=8); ax[0, 1].set_title("jump size by joint", fontsize=10)
        dts = sorted(F.date.unique()); r = [pm(F[F.date == d].n_jump.sum(), F[F.date == d].mit_s.sum()) for d in dts]
        ax[0, 2].bar([d[4:6] + "/" + d[6:] for d in dts], r); ax[0, 2].set_ylabel("jumps per minute")
        ax[0, 2].set_title("by date (moving %.1f /min, holding %.1f /min overall)" % (pm(P.moving.sum(), mv), pm((~P.moving).sum(), tot - mv)), fontsize=10)
        g = P.dropna(subset=["gyro_pk"])
        for j in J:
            p = g[g.joint == j]
            if len(p):
                ax[1, 0].scatter(p.disp_deg, p.gyro_pk, s=14, label=f"J{j}")
        ax[1, 0].set_xlabel("joint deviation change [deg]"); ax[1, 0].set_ylabel("camera gyro change in jump window [rad/s]"); ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log")
        ax[1, 0].legend(fontsize=8); ax[1, 0].set_title("jumps move the end-effector (camera IMU)", fontsize=10)
        if len(gaps):
            ax[1, 1].hist(np.clip(gaps, 0, 20), bins=np.arange(0, 20.5, 0.5)); ax[1, 1].axvline(1.1, color="C3", ls=":", label="1.1 s (MIT_E3)")
            ax[1, 1].set_xlabel("interval between jumps in one run [s] (clipped at 20)"); ax[1, 1].set_ylabel("count"); ax[1, 1].legend(fontsize=8)
        tb = np.arange(0, 62, 2); n = np.histogram(P.t_mit[P.t_mit < 60], bins=tb)[0]; ex = np.array([np.clip(F.mit_s - x, 0, 2).sum() for x in tb[:-1]])
        ax[1, 2].plot(tb[:-1] + 1, n / np.maximum(ex, 1e-9) * 60, "o-", ms=3); ax[1, 2].set_xlabel("time after MIT entry [s]"); ax[1, 2].set_ylabel("jumps per minute (exposure-normalized)")
        ax[1, 2].set_title("no build-up or decay with time in MIT", fontsize=10)
        for x in ax.flat:
            x.grid(alpha=0.3)
        fig.suptitle("MIT-mode internal jumps in existing real-arm records (S-V1.7-3, piper_sdk 0.6.2)", fontsize=11)
        fig.tight_layout(); fig.savefig(a.png, dpi=120); print("그림:", a.png)


if __name__ == "__main__":
    main()
