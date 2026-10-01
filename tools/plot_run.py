#!/usr/bin/env python3
"""bag 한 개 → PNG 그래프(관절 위치/속도/토크, EE 지령 vs 실측·오차, EE 속도) + 요약 수치.

사용: python3 tools/plot_run.py logs/bags/<bag폴더> [--out 출력폴더]
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

TOPICS = ("/joint_states", "/arm/ee_pose_body", "/arm/cartesian_target", "/arm/status",
          "/piper/gripper_feedback")
CART_PHASES = {"hover", "pre", "grasp", "lift", "place_lower", "place_hover", "place_detect",
               "place_descend", "place_release", "place_retreat"}


def read_bag(path):  # 토픽별 [(t_sec, msg)] — Point엔 header가 없어 bag 수신시각을 공통 시간축으로 쓴다
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),
           rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    out = {t: [] for t in TOPICS}
    while r.has_next():
        topic, data, t_ns = r.read_next()
        if topic in out:
            out[topic].append((t_ns * 1e-9, deserialize_message(data, get_message(types[topic]))))
    return out


def phase_segments(status):  # /arm/status → [(t0, t1, phase)]
    segs = []
    for t, m in status:
        ph = m.data.split(" ")[0].split("=")[-1]
        if not segs or segs[-1][2] != ph:
            if segs:
                segs[-1][1] = t
            segs.append([t, t, ph])
    if segs:
        segs[-1][1] = status[-1][0]
    return segs


def shade(ax, segs, t0, label=False):  # phase 구간 음영(직교 단계=파랑, 관절 단계=회색)
    for a, b, ph in segs:
        ax.axvspan(a - t0, b - t0, color="tab:blue" if ph in CART_PHASES else "0.6", alpha=0.08, lw=0)
        if label and b - a > 1.5:
            ax.text((a + b) / 2 - t0, 1.0, ph, transform=ax.get_xaxis_transform(), rotation=90,
                    ha="center", va="top", fontsize=6, color="0.3")


def deriv(t, x, win=5):  # 이동평균 후 중앙차분 — 60Hz 피드백의 양자화 노이즈 완화
    k = np.ones(win) / win
    h = win // 2
    xs = np.apply_along_axis(lambda c: np.convolve(np.pad(c, h, mode="edge"), k, mode="valid"), 0, x)
    return np.gradient(xs, t, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.bag, "plots")
    os.makedirs(out, exist_ok=True)

    d = read_bag(a.bag)
    segs = phase_segments(d["/arm/status"])
    js = d["/joint_states"]
    t0 = js[0][0]
    tj = np.array([t for t, _ in js])
    q = np.degrees(np.array([m.position for _, m in js]))
    vel = np.array([m.velocity if len(m.velocity) == 6 else [np.nan] * 6 for _, m in js])
    eff = np.array([m.effort if len(m.effort) == 6 else [np.nan] * 6 for _, m in js])
    qd = deriv(tj, np.radians(q))

    # 1) 관절 위치 / 토크 / 속도(피드백 vs 위치미분)
    for name, ys, unit in (("joint_pos", [q], "deg"), ("joint_effort", [eff], "N·m"),
                           ("joint_vel", [vel, qd], "rad/s")):
        fig, axs = plt.subplots(6, 1, figsize=(14, 12), sharex=True)
        for i, ax in enumerate(axs):
            for k, y in enumerate(ys):
                ax.plot(tj - t0, y[:, i], lw=0.8, label=("feedback", "d(pos)/dt")[k] if len(ys) > 1 else None)
            ax.set_ylabel(f"J{i + 1} [{unit}]")
            shade(ax, segs, t0, label=(i == 0))
            ax.grid(alpha=0.3)
        if len(ys) > 1:
            axs[0].legend(loc="upper right", fontsize=8)
        axs[-1].set_xlabel("t [s]")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{name}.png"), dpi=110)
        plt.close(fig)

    # 2) EE 지령(zero-order hold) vs 실측, 오차
    ee = d["/arm/ee_pose_body"]
    te = np.array([t for t, _ in ee])
    pe = np.array([[m.x, m.y, m.z] for _, m in ee]) * 1000
    ct = d["/arm/cartesian_target"]
    tc = np.array([t for t, _ in ct])
    pc = np.array([[m.x, m.y, m.z] for _, m in ct]) * 1000
    idx = np.searchsorted(tc, te, side="right") - 1
    ph_at = np.array([next((p for a_, b_, p in segs if a_ <= t <= b_), "") for t in te])
    valid = (idx >= 0) & np.isin(ph_at, list(CART_PHASES))
    cmd = np.full_like(pe, np.nan)
    cmd[valid] = pc[idx[valid]]
    err = np.linalg.norm(pe - cmd, axis=1)
    fig, axs = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    for i, c in enumerate("xyz"):
        axs[i].plot(te - t0, pe[:, i], lw=0.9, label="actual")
        axs[i].plot(tc - t0, pc[:, i], ".", ms=2, label="cmd (/arm/cartesian_target)")
        axs[i].set_ylabel(f"{c} body [mm]")
    axs[3].plot(te - t0, err, lw=0.8, color="tab:red")
    axs[3].set_ylabel("|cmd − actual| [mm]")
    axs[0].legend(loc="upper right", fontsize=8)
    for i, ax in enumerate(axs):
        shade(ax, segs, t0, label=(i == 0))
        ax.grid(alpha=0.3)
    axs[-1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ee_cmd_vs_actual.png"), dpi=110)
    plt.close(fig)

    # 3) EE 실측 속도 — 직선 구간에서 끊김(속도 dip)이 보이는지
    ve = np.linalg.norm(deriv(te, pe), axis=1)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(te - t0, ve, lw=0.8)
    ax.set_ylabel("|v_EE| [mm/s]")
    ax.set_xlabel("t [s]")
    shade(ax, segs, t0, label=True)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "ee_speed.png"), dpi=110)
    plt.close(fig)

    # 요약: phase별 EE 최종오차, 최대속도, 속도 dip 수 / 속도 피드백 단위 확인
    print(f"bag {a.bag}  {tj[-1] - tj[0]:.1f}s  → {out}")
    print(f"{'phase':<14}{'dur':>6}{'v_max':>8}{'v_mean':>8}{'dips':>6}{'err_end':>9}{'err_max':>9}")
    for s0, s1, ph in segs:
        m = (te >= s0) & (te <= s1)
        if m.sum() < 3:
            continue
        v = ve[m]
        moving = v > 2.0
        dips = int(np.sum(np.diff(moving.astype(int)) == -1))  # 움직이다 2mm/s 밑으로 떨어진 횟수
        e = err[m]
        e_end = e[~np.isnan(e)][-1] if np.any(~np.isnan(e)) else np.nan
        e_max = np.nanmax(e) if np.any(~np.isnan(e)) else np.nan
        print(f"{ph:<14}{s1 - s0:6.1f}{v.max():8.1f}{v.mean():8.1f}{dips:6d}{e_end:9.2f}{e_max:9.2f}")

    # 목표 갱신 중(직전 0.1s 내 새 목표) vs 목표 정지 후의 EE 속도 — 재계획 가설 검증용
    # arm_node는 도착 후에도 같은 목표를 계속 발행하므로 '값이 바뀐 시각' 기준으로 본다
    changed = np.r_[True, np.any(np.abs(np.diff(pc, axis=0)) > 1e-6, axis=1)]
    t_chg = tc[changed]
    k = np.searchsorted(t_chg, te, side="right") - 1
    age = np.where(k >= 0, te - t_chg[np.clip(k, 0, None)], np.inf)
    streaming = valid & (age < 0.1)
    static = valid & (age >= 0.3) & (err > 2.0)  # 목표는 멈췄고 팔은 아직 가는 중
    print(f"\nEE 속도 [mm/s]  목표 갱신 중: mean={np.mean(ve[streaming]):.1f} (n={streaming.sum()})"
          f"   목표 정지 후 이동: mean={np.mean(ve[static]):.1f} (n={static.sum()})")
    mv = np.abs(qd) > 0.02
    if np.any(mv & ~np.isnan(vel)):
        ratio = np.nanmedian(vel[mv] / qd[mv])
        print(f"\nvelocity 피드백 / d(pos)/dt 중앙값 비 = {ratio:.2f} (≈1이면 관절축, 크면 모터축)")


if __name__ == "__main__":
    main()
