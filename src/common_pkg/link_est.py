"""링크 쪽 관절 속도 추정 (D2-1) — Chen·Tomizuka TMech 2014 IV-B절 자이로 버전 (26)–(27).

(26) q̇̂ = q̇_m + J_ω^W† (ω_e − J_ω(q_m) q̇_m),  J_ω^W† = W J_ωᵀ (J_ω W J_ωᵀ)⁻¹ — W = 관절별 변형 비중(논문은 W = I)
(27) 1차 KF: q_{k+1} = q_k + Δt·q̇̂_k + w,  y_k = q_m,k + v — 논문의 거친 링크 위치 q̂° 대신 모터 위치(K_J 모름, 속도 추정은 강성 오차에 강건)
J_ω·J_v는 FK 수치 미분, ω_e는 base 좌표 자이로(imu_frame), 바이어스는 상위에서 고역통과로 뺌.
"""
import math

import numpy as np

# IMU 원점의 link6 좌표: 손목캠 TCP 오프셋(비전 캘리브) + 장착 회전 · (D455 가속도계 → 컬러 extrinsic 역)
CAM_T_L6 = np.array([-0.085, -0.010, 0.026])
IMU_IN_CAM = np.array([0.029, 0.0073, 0.0156])


def jacobians(kin, q, R_l6_imu, eps=1e-6):  # → (J_ω 3×6 base 각속도, J_v 3×6 IMU 원점 base 속도)
    p0, R0 = kin.fk(q)
    r = CAM_T_L6 + R_l6_imu @ IMU_IN_CAM
    x0 = p0 + R0 @ r
    Jw = np.zeros((3, 6)); Jv = np.zeros((3, 6))
    for j in range(6):
        dq = np.zeros(6); dq[j] = eps
        p1, R1 = kin.fk(q + dq)
        dR = R1 @ R0.T
        Jw[:, j] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * eps)
        Jv[:, j] = (p1 + R1 @ r - x0) / eps
    return Jw, Jv


def weighted_pinv(J, w):  # W Jᵀ (J W Jᵀ)⁻¹, 랭크 부족은 감쇠 최소제곱
    W = np.diag(w)
    A = J @ W @ J.T
    return W @ J.T @ np.linalg.inv(A + 1e-9 * np.trace(A) * np.eye(A.shape[0]))


class LinkVelEst:
    def __init__(self, kin, R_l6_imu, w=(1, 1, 1, 1, 1, 1), axes=(0, 1, 2), q_proc=1e-6, r_meas=1e-4):
        self.kin, self.R = kin, R_l6_imu
        self.w = np.asarray(w, float); self.axes = list(axes)  # 쓸 자이로 축(base) — 피치만이면 (1,)
        self.q = None; self.P = np.ones(6) * 1e-4; self.Qp = q_proc; self.Rm = r_meas

    def vel(self, q_m, qd_m, w_base):  # (26)
        Jw, _ = jacobians(self.kin, q_m, self.R)
        J = Jw[self.axes]; res = np.asarray(w_base)[self.axes] - J @ qd_m
        return qd_m + weighted_pinv(J, self.w) @ res, res

    def step(self, q_m, qd_m, w_base, dt):  # (27) 관절별 독립 1차 KF → (q̂, q̇̂, 잔차)
        qd_hat, res = self.vel(q_m, qd_m, w_base)
        if self.q is None:
            self.q = np.array(q_m, float)
        self.q = self.q + dt * qd_hat; self.P = self.P + self.Qp
        Kk = self.P / (self.P + self.Rm)
        self.q = self.q + Kk * (q_m - self.q); self.P = (1 - Kk) * self.P
        return self.q.copy(), qd_hat, res
