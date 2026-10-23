"""손목 D455 IMU → 팔 베이스 좌표 변환 (D2-0). IMU 축 = 컬러 광학 좌표(장치 extrinsics 회전 0.2° 이하), 장착 회전은 정지 중력으로 맞춤.

R_base_imu = R_base_link6(FK) · R_link6_cam,  R_link6_cam = Ry(pitch)·Rz(roll) (piper_eih_camera_node와 같은 정의)
가속도: a_base = R·a_imu − [0, 0, g]  (정지 시 가속도계는 +g 위쪽을 읽음),  각속도: ω_base = R·ω_imu
"""
import math

import numpy as np

# 10-23 정지 중력 맞춤(자세 A·B, 최대 0.36°) — 비전 캘리브 27.5°/−90°와 약 3° 차이, roll은 피치 자세만으로 맞춰 불확실
MOUNT_PITCH_DEG = 30.75
MOUNT_ROLL_DEG = -88.25
G = 9.81


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def r_link6_imu(pitch_deg=MOUNT_PITCH_DEG, roll_deg=MOUNT_ROLL_DEG):
    return Ry(math.radians(pitch_deg)) @ Rz(math.radians(roll_deg))


class ImuToBase:
    def __init__(self, kin, pitch_deg=MOUNT_PITCH_DEG, roll_deg=MOUNT_ROLL_DEG, g=G):
        self.kin = kin; self.R_l6 = r_link6_imu(pitch_deg, roll_deg); self.g = g

    def rot(self, q):  # 관절각[rad] → R_base_imu
        return self.kin.fk(q)[1] @ self.R_l6

    def accel(self, q, a_imu):  # → 중력 뺀 base 가속도[m/s²]
        return self.rot(q) @ np.asarray(a_imu) - np.array([0.0, 0.0, self.g])

    def gyro(self, q, w_imu):  # → base 각속도[rad/s]
        return self.rot(q) @ np.asarray(w_imu)
