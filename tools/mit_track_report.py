#!/usr/bin/env python3
"""MIT 추종층 시행 보고(단계 D) — mit_probe 기록 하나마다 관절별 추종 오차, 보낸 토크, 변화율 제한에 걸린 주기, MIT 순간 튐을 낸다. 읽기만 한다.

구간: 기준이 움직인 구간(사인이면 진폭 램프 2 s 포함 전체), 정지 시행이면 MIT 진입 0.5 s 뒤부터.
튐 창: mit_jump_report.events()의 사건(|q̇ − q̇_ref| > 0.5 rad/s) 시각 기준. 새 규칙 −0.1–+1.2 s(9절 11/17 22:04), 옛 창 −0.1–+0.5 s(D_LOG 1절의 0.073–0.077°가 이 창).
추종 오차: q − q_ref의 rms·최대[°] — 전체 / 옛 창 제외 / 새 규칙 제외. 추종층 관절은 tff가 변한 관절.
토크: 보낸 τ 범위, 명령 상한(4 N·m)에 닿은 주기 수, 주기당 변화 최대. 변화율 제한: .tdc*.csv의 lim 열 — 전체 / 새 규칙 창 밖(창 밖에서 걸리면 0.5 N·m로 올림).
사용: python3 tools/mit_track_report.py logs/mit/20261118_*_sine.csv
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mit_jump_report import events  # noqa: E402
from mit_probe import TAU_CMD_LIMIT  # noqa: E402

J = range(1, 7)


def report(p):
    d = pd.read_csv(p); m = d[d.phase == "mit"].reset_index(drop=True); t = m.t.values; tm = t - t[0]
    qr = m[[f"qref{j}" for j in J]].values; dev = m[[f"q{j}" for j in J]].values - qr; tff = m[[f"tff{j}" for j in J]].values
    mv = np.where(np.abs(qr - qr[0]).max(1) > 1e-3)[0]
    sec = (tm >= tm[mv[0]]) & (tm <= tm[mv[-1]]) if len(mv) else tm > 0.5
    ev, _ = events(p, 0.5); te = np.array([e["t_mit"] for e in ev])
    win = lambda x, a, b: np.any([(x >= s - a) & (x <= s + b) for s in te], axis=0) if len(te) else np.zeros(len(x), bool)
    old, new = win(tm, 0.1, 0.5), win(tm, 0.1, 1.2)
    ours = [j for j in J if np.ptp(tff[:, j - 1]) > 2e-3]
    res = str(m.phase.iloc[-1]); last = d.phase.iloc[-1]
    print(f"\n{os.path.basename(p)} | MIT {tm[-1]:.1f} s, 구간 {sec.sum() * np.median(np.diff(tm)):.1f} s({'운동' if len(mv) else '정지'}) | 추종층 J{ours} | 튐 {len(ev)}건 | 새 규칙 제외 {100 * new[sec].mean():.0f}%, 옛 창 {100 * old[sec].mean():.0f}%")
    for e in ev:
        print(f"   튐 {e['t_mit']:6.2f} s J{e['joint']}({e['kind']}) 이탈 변화 {e['disp_deg']:.2f}°, 최고 {e['peak_dv']:.2f} rad/s, 따라 튄 관절 {e['followers'] or '—'}, 복귀 {e['recover_s']} s, 자이로 {e['gyro_pk']} rad/s")
    rms = lambda x: float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")
    print("   관절 | 이탈 rms 전체 / 옛 창 제외 / 새 규칙 제외 [°] | 최대 전체 / 새 규칙 제외 [°] | 보낸 τ 범위 [N·m] | 상한 닿은 주기 | 주기당 τ 변화 최대(창 밖) | 제한 걸린 주기 전체 / 창 밖")
    files = {os.path.basename(f).split(".tdc")[1].split(".")[0]: f for f in glob.glob(p.replace(".csv", ".tdc*.csv"))}
    main_j = [j for j in ours if str(j) not in files]  # .tdc.csv는 --joint
    for j in J:
        x = dev[:, j - 1]; a = f"   J{j}{'*' if j in ours else ' '} | {rms(x[sec]):.3f} / {rms(x[sec & ~old]):.3f} / {rms(x[sec & ~new]):.3f} | {np.abs(x[sec]).max():.2f} / " \
            + (f"{np.abs(x[sec & ~new]).max():.2f}" if (sec & ~new).any() else "—")
        if j in ours:
            tau = tff[:, j - 1]; dt_ = np.abs(np.diff(tau)); k = ~new[1:] & ~new[:-1] & (tm[1:] > 0.5)
            a += f" | {tau.min():+.2f}–{tau.max():+.2f} | {int((np.abs(tau) >= TAU_CMD_LIMIT - 1e-3).sum())}(창 밖 {int(((np.abs(tau) >= TAU_CMD_LIMIT - 1e-3) & ~new).sum())}) | {dt_.max():.3f}({dt_[k].max() if k.any() else float('nan'):.3f})"
            f = files.get(str(j)) or (files.get("") if j in main_j else None)
            if f:
                c = pd.read_csv(f)
                if "lim" in c.columns:
                    tc = c.t.values - t[0]; nw = win(tc, 0.1, 1.2)
                    a += f" | {int(c.lim.sum())} / {int(c.lim.values[~nw].sum())}" + (f" (창 밖 시각 {np.round(tc[(c.lim.values == 1) & ~nw][:6], 2)})" if c.lim.values[~nw].sum() else "")
                else:
                    a += " | 제한 없음(옛 기록)"
                a += f" | K̂ 최대 {c.K.max():.1f}"
        print(a)
    for j in ours:  # 정지 구간에서 토크가 수렴한 값(9절 11/18 22번 — 상한까지의 여유가 처짐 상태에 따라 달라지는지)
        still = np.abs(np.gradient(qr[:, j - 1], tm)) < 1e-3
        if (still & (tm > 1.0)).sum() > 20:
            x = tff[still & (tm > 1.0), j - 1]; print(f"   J{j} 정지 구간 토크 중앙 {np.median(x):+.2f} N·m (범위 {x.min():+.2f}–{x.max():+.2f}), 상한까지 여유 {TAU_CMD_LIMIT - np.abs(x).max():.2f}")
    print(f"   끝: 마지막 구간 {last}, MIT 마지막 표본 이탈 최대 {np.abs(dev[-1]).max():.2f}°")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if not any(x in p for x in (".tdc", ".imu.", ".vib.")):
            report(p)
