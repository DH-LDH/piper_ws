"""D2-2 진동 감쇠 τ_vib — 자이로 피치 − 엔코더 피치 = 링크 변형 속도 → 대역통과 → 속도 감쇠 또는 PPF. SDK·CAN 없음.

변형 속도 v = ω_base,y − (J_ω q̇_m)_y (Chen·Tomizuka (26)의 괄호 안, 피치 한 축) — VIB_E3에서 J2 τ → v 피크 위상 −27–−59°(처짐 ≥ 35°).
vel: τ_vib = −k·v_f (E3: Re(H) > 0 → 음의 속도 되먹임이 감쇠), ppf: ξ = ∫v_f(1 Hz 누설), η̈ + 2ζ_f ω_f η̇ + ω_f² η = ω_f² ξ, τ_vib = +k·η.
기울기 게이트(8절 10/26 오후): MIT 시작 gate 창의 IMU 정지 기울기 atan2(y, −z) < 문턱이면 k = 0 — 처짐이 낮으면 위상이 약 110° 뒤집혀 양의 되먹임 위험.
데드밴드: |τ_vib| < 반 칸(0.031 N·m)이면 0 — 양자화 떨림 비교용. 포화 ±tau_max.
사용: d = VibDamp(kin, tb, cfg); out = d.step(t_mit, q, qd, gyro_raw, accel_raw, h) → out["tau"]
"""
import math
from dataclasses import dataclass

import numpy as np

from link_est import jacobians

TAU_STEP = 16.0 / 255  # MIT 토크 한 칸[N·m]


@dataclass
class VibDampCfg:
    law: str = "vel"           # vel / ppf
    k: float = 0.0             # vel: N·m/(rad/s), ppf: N·m/rad
    band: tuple = (4.0, 16.0)  # 대역통과[Hz] — 기하 중심 8 Hz에서 연속계 위상 0
    gate_deg: float = 35.0     # 기울기 문턱
    gate_win: tuple = (0.3, 1.8)  # MIT 진입 뒤 기울기 평균 창[s] — 이 창이 끝나야 τ_vib 시작
    deadband: bool = False
    tau_max: float = 0.3       # 안전 포화[N·m]
    ppf_f: float = 8.0         # ω_f/2π[Hz] — k=0 툭 자유진동 주파수로 정함
    ppf_zeta: float = 0.3


def bp_phase_deg(band, f, fs):  # 아래 이산 대역통과(1차 HP·1차 LP, 고정 주기 1/fs)의 f에서 위상[deg]
    h = 1.0 / fs; th = 1 / (2 * math.pi * band[0]); tl = 1 / (2 * math.pi * band[1])
    ah = th / (th + h); al = h / (tl + h); z = np.exp(-1j * 2 * math.pi * f * h)
    H = ah * (1 - z) / (1 - ah * z) * al / (1 - (1 - al) * z)
    return math.degrees(np.angle(H))


class VibDamp:
    def __init__(self, kin, tb, cfg):
        self.kin, self.tb, self.c = kin, tb, cfg
        self.hp = 0.0; self.x_prev = None; self.lp = 0.0
        self.xi = 0.0; self.eta = 0.0; self.eta_d = 0.0
        self.acc = []; self.tilt = None; self.enabled = False

    def _bandpass(self, x, h):  # 주기 h가 흔들려도 쓰도록 매 호출 계수 계산
        th = 1 / (2 * math.pi * self.c.band[0]); tl = 1 / (2 * math.pi * self.c.band[1])
        if self.x_prev is None:
            self.x_prev = x
        ah = th / (th + h); self.hp = ah * (self.hp + x - self.x_prev); self.x_prev = x
        self.lp += h / (tl + h) * (self.hp - self.lp)
        return self.lp

    def defl_vel(self, q, qd, gyro_raw):  # → 변형 피치 속도[rad/s], 자이로 피치, 엔코더 피치
        Jw, _ = jacobians(self.kin, q, self.tb.R_l6)
        wy = (self.tb.rot(q) @ np.asarray(gyro_raw))[1]; ey = float(Jw[1] @ qd)
        return wy - ey, wy, ey

    def step(self, t_mit, q, qd, gyro_raw, accel_raw, h):
        c = self.c
        v, wy, ey = self.defl_vel(q, qd, gyro_raw)
        vf = self._bandpass(v, h) if h > 0 else 0.0
        if self.tilt is None:  # 게이트 창: 원시 가속도 모아 기울기 → 문턱 판정, 그 전엔 τ_vib = 0
            if c.gate_win[0] <= t_mit <= c.gate_win[1] and accel_raw is not None:
                self.acc.append(np.asarray(accel_raw, float))
            if t_mit > c.gate_win[1]:
                g = np.mean(self.acc, axis=0) if self.acc else np.array([np.nan] * 3)
                self.tilt = math.degrees(math.atan2(g[1], -g[2]))
                self.enabled = bool(self.tilt >= c.gate_deg)
        if c.law == "ppf" and h > 0:
            self.xi += h * (vf - 2 * math.pi * 1.0 * self.xi)  # 누설 적분 — 변위형 입력
            wf = 2 * math.pi * c.ppf_f
            self.eta_d += h * (wf * wf * (self.xi - self.eta) - 2 * c.ppf_zeta * wf * self.eta_d)
            self.eta += h * self.eta_d
            raw = c.k * self.eta
        else:
            raw = -c.k * vf
        tau = float(np.clip(raw, -c.tau_max, c.tau_max)) if self.enabled else 0.0
        if c.deadband and abs(tau) < TAU_STEP / 2:
            tau = 0.0
        return dict(v=v, wy=wy, ey=ey, vf=vf, raw=raw, tau=tau, enabled=self.enabled, tilt=self.tilt)
