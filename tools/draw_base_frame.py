#!/usr/bin/env python3
"""URDF 메시(STL)로 PiPER 자세를 그리고 base_link(native)·body_link 좌표축을 표시 — 발표용 좌표 정의 그림.

사용: python3 tools/draw_base_frame.py --q 0 103.55 -54.85 0 46.26 -15.03 --out docs/img/p_base_frame.png
q: 관절각 [deg] (기본값 = B단계 line 궤적 시작 자세)
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "common_pkg"))
from traj_analyze import _font
from piper_kin import load_urdf_xml
from step22_common import ARM_BASE_YAW_DEG

MESH_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "piper_description", "meshes")
COLORS = {"base_link": "0.35", "gripper_base": "0.25", "link7": "0.25", "link8": "0.25"}


def read_stl(path):  # 바이너리 STL → 삼각형 [N,3,3] m
    with open(path, "rb") as f:
        f.read(80)
        n = int(np.frombuffer(f.read(4), np.uint32)[0])
        rec = np.frombuffer(f.read(n * 50), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return rec["v"].astype(float)


def rpy_mat(r, p, y):
    cx, sx, cy, sy, cz, sz = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def axis_rot(ax, th):  # 로드리게스
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def link_frames(urdf_xml, q_deg):  # 링크별 base_link 기준 4x4 변환
    root = ET.fromstring(urdf_xml)
    T = {"base_link": np.eye(4)}
    joints = root.findall("joint")
    while True:
        added = False
        for j in joints:
            par, ch = j.find("parent").get("link"), j.find("child").get("link")
            if par not in T or ch in T:
                continue
            o = j.find("origin")
            A = np.eye(4)
            A[:3, :3] = rpy_mat(*[float(v) for v in o.get("rpy", "0 0 0").split()])
            A[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
            M = np.eye(4)
            name = j.get("name")
            if j.get("type") == "revolute" and name.startswith("joint") and int(name[5:]) <= 6:
                M[:3, :3] = axis_rot([float(v) for v in j.find("axis").get("xyz").split()], np.radians(q_deg[int(name[5:]) - 1]))
            T[ch] = T[par] @ A @ M
            added = True
        if not added:
            return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", nargs=6, type=float, default=[0, 103.55, -54.85, 0, 46.26, -15.03])
    ap.add_argument("--out", default="docs/img/p_base_frame.png")
    a = ap.parse_args()
    _font()
    T = link_frames(load_urdf_xml(), a.q)
    light = np.array([0.4, -0.5, 0.8]); light /= np.linalg.norm(light)

    fig = plt.figure(figsize=(14, 7))
    for k, (elev, azim, title) in enumerate(((22, -60, "비스듬히 본 모습"), (0, -90, "옆에서 본 모습 (x–z 평면)"))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d", proj_type="persp" if k == 0 else "ortho")
        ax.computed_zorder = False  # 그린 순서대로 — 좌표축이 메시에 가려지지 않게
        for link, M in T.items():
            path = os.path.join(MESH_DIR, f"{link}.STL")
            if not os.path.exists(path):
                continue
            tri = read_stl(path) @ M[:3, :3].T + M[:3, 3]
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            shade = 0.45 + 0.55 * np.clip(nrm @ light, 0, 1)
            base = np.array(matplotlib.colors.to_rgb(COLORS.get(link, "0.8")))
            ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(base[None] * shade[:, None] + 0.1, 0, 1), linewidths=0))
        L = 0.2
        for v, c, n in ((np.eye(3)[0], "red", "x"), (np.eye(3)[1], "green", "y"), (np.eye(3)[2], "blue", "z")):
            if k == 1 and n == "y":
                continue
            ax.quiver(0, 0, 0, *(v * L), color=c, lw=3, arrow_length_ratio=0.12)
            ax.text(*(v * L * 1.12), n, color=c, fontsize=15, weight="bold")
        pe = T["link6"][:3, 3]
        ax.scatter(*pe, color="k", s=30)
        ax.text(*(pe + [0.015, 0, 0.02]), "EE(link6)", fontsize=10, weight="bold")
        ax.quiver(*pe, 0.07, 0, -0.04, color="tab:purple", lw=2.5, arrow_length_ratio=0.2)
        ax.text(*(pe + [0.07, 0, -0.07]), "line: x +70, z −40 mm", color="tab:purple", fontsize=10)
        ax.set_xlim(-0.1, 0.4); ax.set_ylim(-0.25, 0.25); ax.set_zlim(-0.05, 0.45)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        ax.view_init(elev=elev, azim=azim)
        if k == 1:
            ax.set_yticks([]); ax.set_ylabel("")
        ax.set_title(title, fontsize=11, pad=0)
    yaw = ARM_BASE_YAW_DEG
    fig.suptitle("PiPER base_link(native) 좌표 — 원점: 팔 베이스 바닥 중심, x 앞(팔 뻗는 방향), y 왼쪽, z 위", fontsize=13)
    fig.text(0.5, 0.03, f"A단계 그림의 body_link(차체 기준)는 팔이 차체에 {yaw:.0f}° 돌아 장착돼 있어 x_body = −y, y_body = x, z_body = z  "
             "(EE = 펌웨어 EndPose = URDF link6 원점)", ha="center", fontsize=10, color="0.3")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.08, wspace=0.05)
    fig.savefig(a.out, dpi=120)
    print("저장:", a.out, " EE(link6) =", np.round(pe * 1000, 1), "mm")


if __name__ == "__main__":
    main()
