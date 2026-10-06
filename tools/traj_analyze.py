#!/usr/bin/env python3
"""traj_test_node 결과(traj.csv) 분석 — 추종 지연, EE 경로 오차, 진동(A단계와 같은 지표), 정착, FK 대조 + 그래프.

사용: python3 tools/traj_analyze.py logs/traj/<run> [<run> ...]   (여러 개면 비교표)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from plot_run import deriv

WIN = 15  # 0.25 s 이동평균을 뺀 성분을 '떨림'으로 — A단계 지표와 동일
J = range(1, 7)
A_STAGE = {"MOVE J 1회": (0.30, 0.45), "MOVE L 1회": (0.49, 0.62), "MOVE L 스트리밍": (0.91, 1.17)}  # J2, J3


def hp(x):
    k = np.ones(WIN) / WIN
    h = WIN // 2
    lo = np.apply_along_axis(lambda c: np.convolve(np.pad(c, h, mode="edge"), k, mode="valid"), 0, x)
    return x - lo


def analyze(run):
    d = pd.read_csv(os.path.join(run, "traj.csv"))
    d["t"] -= d.t.iloc[0]
    r = d[d.state == "run"].reset_index(drop=True)
    s = d[d.state == "settle"].reset_index(drop=True)
    qc = r[[f"qc{j}" for j in J]].values
    q = r[[f"q{j}" for j in J]].values
    qd = r[[f"qd{j}" for j in J]].values
    t = r.t.values
    out = {"run": os.path.basename(run.rstrip("/")), "T": t[-1] - t[0], "sent": int(r.sent.sum())}

    # 추종: 지령 대비 실측 관절 오차, 지령→실측 시간지연(교차상관)
    err = q - qc
    out["q_err_max"] = np.abs(err).max()
    # 지연 τ: 실측 q(t)와 지령 qc(t−τ)의 차이가 최소가 되는 틱 수(0~30틱, J2·J3)
    cost = [np.mean((q[k:, 1:3] - qc[:len(qc) - k, 1:3]) ** 2) for k in range(31)]
    out["lag_ms"] = int(np.argmin(cost)) / 60.0 * 1000

    # 움직임 구간 = run + settle (movel은 목표가 먼저 끝나고 팔이 settle 동안 따라온다)
    m = pd.concat([r, s], ignore_index=True)
    tm = m.t.values
    pe = m[["ex", "ey", "ez"]].values
    qdm = m[[f"qd{j}" for j in J]].values
    pc = r[["pcx", "pcy", "pcz"]].values
    p_goal = pc[-1]
    dpath = np.array([np.min(np.linalg.norm(pc - p, axis=1)) for p in pe]) * 1000  # 계획 경로까지 거리
    out["path_err_max"], out["path_err_mean"] = dpath.max(), dpath.mean()
    ve = np.linalg.norm(deriv(tm, pe), axis=1) * 1000

    # 진동: 움직이는 동안 관절 속도 고역성분 RMS / 평균 속도 (A단계와 동일)
    mv = ve > 5.0
    rip = np.sqrt((hp(qdm)[mv] ** 2).mean(axis=0))
    sp = np.abs(qdm[mv]).mean(axis=0)
    out["rip_J2"], out["rip_J3"] = rip[1] / sp[1], rip[2] / sp[2]
    out["ee_vhp"] = np.sqrt(np.mean(hp(ve[:, None])[mv] ** 2))
    out["v_ee_mean"] = ve[mv].mean()

    # 도달·정착: 목표 끝점까지 0.5 mm 들어온 시각(궤적 시작 기준), 최종 오차, 잔류 흔들림
    dg = np.linalg.norm(pe - p_goal, axis=1) * 1000
    out_idx = np.where(dg > 0.5)[0]
    out["arrive_s"] = (tm[out_idx[-1] + 1] - tm[0]) if len(out_idx) and out_idx[-1] + 1 < len(tm) else (np.nan if len(out_idx) else 0.0)
    out["final_ee_err_mm"] = dg[-1]
    out["final_q_err"] = np.abs(m[[f"q{j}" for j in J]].values[-1] - qc[-1]).max()
    out["resid_mm"] = np.std(pe[-60:], axis=0).max() * 1000
    out["fk_pos_max"], out["fk_rot_max"] = d.fk_pos_err_mm.max(), d.fk_rot_err_deg.max()
    plot(run, m, tm, m[[f"qc{j}" for j in J]].values, m[[f"q{j}" for j in J]].values, qdm, ve)
    return out


def plot(run, r, t, qc, q, qd, ve):
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for j, c in ((1, "tab:blue"), (2, "tab:orange"), (4, "tab:green")):
        axs[0].plot(t, qc[:, j], "--", color=c, lw=1, label=f"J{j+1} 지령")
        axs[0].plot(t, q[:, j], color=c, lw=1, label=f"J{j+1} 실측")
        axs[1].plot(t, q[:, j] - qc[:, j], color=c, lw=1, label=f"J{j+1}")
        axs[2].plot(t, qd[:, j], color=c, lw=0.8, label=f"J{j+1}")
    axs[0].set_ylabel("관절 [deg]"); axs[1].set_ylabel("실측−지령 [deg]"); axs[2].set_ylabel("관절속도 [rad/s]")
    axs[3].plot(t, ve, lw=0.8); axs[3].set_ylabel("|v_EE| [mm/s]"); axs[3].set_xlabel("t [s]")
    for ax in axs:
        ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="upper right", ncol=3)
    fig.suptitle(os.path.basename(run.rstrip("/")))
    fig.tight_layout()
    fig.savefig(os.path.join(run, "traj_plot.png"), dpi=110)
    plt.close(fig)


def _font():  # 한글 라벨용 CJK 폰트(없으면 기본)
    from matplotlib import font_manager as fm
    for p in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
        if os.path.exists(p):
            fm.fontManager.addfont(p)
            plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return


def main():
    _font()
    rows = [analyze(r) for r in sys.argv[1:]]
    df = pd.DataFrame(rows).set_index("run")
    pd.set_option("display.width", 220)
    print(df.round(3).T.to_string())
    print("\nA단계 기준 진동(요동/속도, J2/J3):", "  ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in A_STAGE.items()))


if __name__ == "__main__":
    main()
