#!/usr/bin/env python3
"""traj_test_node 결과(traj.csv) 분석 — 추종 지연, EE 경로 오차, 진동(A단계와 같은 지표), 정착, FK 대조 + 그래프.

사용: python3 tools/traj_analyze.py logs/traj/<run> [<run> ...]   (여러 개면 비교표 + 비교 그림)
좌표: native = PiPER URDF base_link(팔 베이스 원점), EE = 펌웨어 EndPose(ex,ey,ez, link6), 60 Hz
속도: 5샘플(≈83 ms) 이동평균 후 중앙차분(plot_run.deriv)
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
    vxyz = deriv(tm, pe) * 1000
    ve = np.linalg.norm(vxyz, axis=1)
    dxyz = (pe - pc[[np.argmin(np.linalg.norm(pc - p, axis=1)) for p in pe]]) * 1000  # 가장 가까운 계획점 대비 축별 편차

    # 진동: 움직이는 동안 관절 속도 고역성분 RMS / 평균 속도 (A단계와 동일)
    mv = ve > 5.0
    rip = np.sqrt((hp(qdm)[mv] ** 2).mean(axis=0))
    sp = np.abs(qdm[mv]).mean(axis=0)
    out["rip_J2"], out["rip_J3"] = rip[1] / sp[1], rip[2] / sp[2]
    out["ee_vhp"] = np.sqrt(np.mean(hp(ve[:, None])[mv] ** 2))
    out["v_ee_mean"] = ve[mv].mean()
    for i, a in enumerate("xyz"):  # 축별 떨림: 속도 고역성분 RMS [mm/s], 편차 최대 [mm]
        out[f"vhp_{a}"] = np.sqrt(np.mean(hp(vxyz[:, i:i + 1])[mv] ** 2))
        out[f"dev_{a}_max"] = np.abs(dxyz[:, i]).max()

    # 도달·정착: 목표 끝점까지 0.5 mm 들어온 시각(궤적 시작 기준), 최종 오차, 잔류 흔들림
    dg = np.linalg.norm(pe - p_goal, axis=1) * 1000
    out_idx = np.where(dg > 0.5)[0]
    out["arrive_s"] = (tm[out_idx[-1] + 1] - tm[0]) if len(out_idx) and out_idx[-1] + 1 < len(tm) else (np.nan if len(out_idx) else 0.0)
    out["final_ee_err_mm"] = dg[-1]
    out["final_q_err"] = np.abs(m[[f"q{j}" for j in J]].values[-1] - qc[-1]).max()
    out["resid_mm"] = np.std(pe[-60:], axis=0).max() * 1000
    out["fk_pos_max"], out["fk_rot_max"] = d.fk_pos_err_mm.max(), d.fk_rot_err_deg.max()
    plot(run, m, tm, m[[f"qc{j}" for j in J]].values, m[[f"q{j}" for j in J]].values, qdm, ve)
    ax = {"t": tm - tm[0], "p": (pe - pe[0]) * 1000, "pc": (pc - pe[0]) * 1000, "v": vxyz, "d": dxyz}
    plot_axes(run, ax)
    return out, ax


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


AX = (("x", "tab:red"), ("y", "tab:green"), ("z", "tab:blue"))


def plot_axes(run, a):  # 축별 위치·속도·편차 시계열 + 3D/탑뷰/측면뷰 (시작점 기준 mm)
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3)
    for row, (key, lab) in enumerate((("p", "위치 [mm]"), ("v", "속도 [mm/s]"), ("d", "계획 경로 대비 편차 [mm]"))):
        for i, (n, c) in enumerate(AX):
            axx = fig.add_subplot(gs[row, i])
            if key == "p":
                axx.plot(np.linspace(0, a["t"][len(a["pc"]) - 1], len(a["pc"])), a["pc"][:, i], "--", color="gray", lw=1, label="계획")
            axx.plot(a["t"], a[key][:, i], color=c, lw=0.9, label="실측")
            axx.set_title(f"{n} {lab}", fontsize=9); axx.grid(alpha=0.3)
            if row == 2:
                axx.set_xlabel("t [s]")
            if key == "p":
                axx.legend(fontsize=7)
    fig.suptitle(os.path.basename(run.rstrip("/")) + "  (native=base_link, 시작점 기준)")
    fig.tight_layout()
    fig.savefig(os.path.join(run, "traj_axes.png"), dpi=110)
    plt.close(fig)

    fig = plt.figure(figsize=(15, 5))
    a3 = fig.add_subplot(1, 3, 1, projection="3d")
    a3.plot(*a["pc"].T, "--", color="gray", lw=1, label="계획"); a3.plot(*a["p"].T, color="k", lw=0.9, label="실측")
    a3.set_xlabel("x"); a3.set_ylabel("y"); a3.set_zlabel("z"); a3.set_title("3D [mm]"); a3.legend(fontsize=7)
    for k, (i, j, name) in enumerate(((0, 1, "탑뷰 x–y"), (0, 2, "측면뷰 x–z"))):
        axx = fig.add_subplot(1, 3, k + 2)
        axx.plot(a["pc"][:, i], a["pc"][:, j], "--", color="gray", lw=1); axx.plot(a["p"][:, i], a["p"][:, j], color="k", lw=0.9)
        axx.set_xlabel("xyz"[i] + " [mm]"); axx.set_ylabel("xyz"[j] + " [mm]"); axx.set_title(name); axx.grid(alpha=0.3)
    fig.suptitle(os.path.basename(run.rstrip("/")))
    fig.tight_layout()
    fig.savefig(os.path.join(run, "traj_3d.png"), dpi=110)
    plt.close(fig)


def plot_compare(names, axs_data, path):  # 여러 run의 축별 속도·편차 겹쳐 그리기
    fig, axs = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for n, a in zip(names, axs_data):
        for i in range(3):
            axs[0, i].plot(a["t"], a["v"][:, i], lw=0.8, label=n)
            axs[1, i].plot(a["t"], a["d"][:, i], lw=0.8, label=n)
    for i, (n, _) in enumerate(AX):
        axs[0, i].set_title(f"{n} 속도 [mm/s]", fontsize=9); axs[1, i].set_title(f"{n} 편차 [mm]", fontsize=9)
        axs[1, i].set_xlabel("궤적 시작 후 t [s]")
    for axx in axs.flat:
        axx.grid(alpha=0.3)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print("비교 그림:", path)


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
    res = [analyze(r) for r in sys.argv[1:]]
    rows = [o for o, _ in res]
    df = pd.DataFrame(rows).set_index("run")
    if len(res) > 1:
        plot_compare([o["run"] for o in rows], [a for _, a in res], os.path.join(os.path.dirname(sys.argv[1].rstrip("/")), "compare_axes.png"))
    pd.set_option("display.width", 220)
    print(df.round(3).T.to_string())
    print("\nA단계 기준 진동(요동/속도, J2/J3):", "  ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in A_STAGE.items()))


if __name__ == "__main__":
    main()
