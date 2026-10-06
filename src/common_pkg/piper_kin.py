"""PiPER 기구학 — URDF(xacro) → PyKDL 체인. FK/IK는 펌웨어 EndPose와 같은 기준(base_link→link6, native 좌표).

펌웨어 EndPose 위치 = URDF link6 원점(10-01 bag 17k쌍, 평균 0.1 mm). 자세 rx,ry,rz[deg]는 R = Rz·Ry·Rx 가정.
"""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import PyKDL as kdl

from step22_common import ARM_BASE_YAW_DEG

N_JOINTS = 6


def load_urdf_xml(path=None):  # xacro 처리된 URDF 문자열. path 없으면 설치된 piper_description 사용
    import xacro
    if path is None:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory("piper_description"), "urdf", "piper_description.xacro")
    return xacro.process_file(path).toxml()


def build_chain(urdf_xml, base="base_link", tip="link6"):  # kdl_parser와 같은 규칙으로 체인 생성 + 관절 한계
    root = ET.fromstring(urdf_xml)
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    seq, link = [], tip
    while link != base:
        j = by_child[link]
        seq.append(j)
        link = j.find("parent").get("link")
    chain, lo, hi = kdl.Chain(), [], []
    for j in reversed(seq):
        o = j.find("origin")
        xyz = [float(v) for v in ((o.get("xyz") if o is not None else None) or "0 0 0").split()]
        rpy = [float(v) for v in ((o.get("rpy") if o is not None else None) or "0 0 0").split()]
        f = kdl.Frame(kdl.Rotation.RPY(*rpy), kdl.Vector(*xyz))
        name = j.find("child").get("link")
        if j.get("type") == "revolute":
            ax = [float(v) for v in j.find("axis").get("xyz").split()]
            jt = kdl.Joint(j.get("name"), f.p, f.M * kdl.Vector(*ax), kdl.Joint.RotAxis)
            lim = j.find("limit")
            lo.append(float(lim.get("lower")))
            hi.append(float(lim.get("upper")))
        else:
            jt = kdl.Joint(j.get("name"), kdl.Joint.Fixed)
        chain.addSegment(kdl.Segment(name, jt, f))
    return chain, np.array(lo), np.array(hi)


def rpy_deg_to_R(rpy_deg):  # EndPose rx,ry,rz[deg] → 회전행렬 (Rz·Ry·Rx)
    r = kdl.Rotation.RPY(*np.radians(rpy_deg))
    return np.array([[r[i, k] for k in range(3)] for i in range(3)])


def R_to_rpy_deg(R):  # 위의 역
    r = kdl.Rotation(*[float(R[i, k]) for i in range(3) for k in range(3)])
    return np.degrees(r.GetRPY())


def native_to_body(p):  # 팔 베이스(native) → body_link 위치 (driver의 _native_to_body_xy와 동일)
    a = math.radians(ARM_BASE_YAW_DEG)
    return np.array([p[0] * math.cos(a) - p[1] * math.sin(a), p[0] * math.sin(a) + p[1] * math.cos(a), p[2]])


def body_to_native(p):
    a = math.radians(-ARM_BASE_YAW_DEG)
    return np.array([p[0] * math.cos(a) - p[1] * math.sin(a), p[0] * math.sin(a) + p[1] * math.cos(a), p[2]])


class PiperKin:  # FK/IK. 좌표는 native(팔 베이스), 위치 m, 관절 rad
    def __init__(self, urdf_xml=None):
        self.chain, self.q_min, self.q_max = build_chain(urdf_xml or load_urdf_xml())
        self._fk = kdl.ChainFkSolverPos_recursive(self.chain)
        self._ik = kdl.ChainIkSolverPos_LMA(self.chain, 1e-10, 1000, 1e-15)  # 1e-6이면 매 틱 해가 흔들려 지령에 잔떨림(10-06 실측)

    def fk(self, q):  # → (p[3] m, R[3x3])
        f = kdl.Frame()
        self._fk.JntToCart(_jnt(q), f)
        return (np.array([f.p[i] for i in range(3)]),
                np.array([[f.M[i, k] for k in range(3)] for i in range(3)]))

    def ik(self, p, R, q_seed):  # → (q, 위치오차 m, 자세오차 rad, 한계내 여부). 수렴 실패도 q를 돌려주고 판단은 호출측
        target = kdl.Frame(kdl.Rotation(*[float(R[i, k]) for i in range(3) for k in range(3)]),
                           kdl.Vector(*[float(v) for v in p]))
        out = kdl.JntArray(N_JOINTS)
        self._ik.CartToJnt(_jnt(q_seed), target, out)
        q = np.array([out[i] for i in range(N_JOINTS)])
        q = (q + np.pi) % (2 * np.pi) - np.pi  # LMA가 2π 돌아간 해를 낼 수 있어 정규화
        p_err, R_err = self.pose_error(q, p, R)
        in_lim = bool(np.all(q >= self.q_min - 1e-6) and np.all(q <= self.q_max + 1e-6))
        return q, p_err, R_err, in_lim

    def pose_error(self, q, p, R):  # FK(q)와 목표의 위치[m]·자세[rad] 오차
        pq, Rq = self.fk(q)
        dR = Rq.T @ R
        ang = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))
        return float(np.linalg.norm(pq - p)), ang


def _jnt(q):
    a = kdl.JntArray(N_JOINTS)
    for i in range(N_JOINTS):
        a[i] = float(q[i])
    return a
