"""D2-2 진동 감쇠 τ_vib — 자이로 피치 − 엔코더 피치 = 링크 변형 속도 → 대역통과 → 속도 감쇠 또는 PPF. SDK·CAN 없음.

변형 속도 v = ω_base,y − (J_ω q̇_m)_y (Chen·Tomizuka (26)의 괄호 안, 피치 한 축) — VIB_E3에서 J2 τ → v 피크 위상 −27–−59°(처짐 ≥ 35°).
시각(8절 10/26 저녁): v는 자이로 표본마다 그 표본 시각 t_g의 q̇(엔코더 이력 보간)로 계산, 대역통과도 새 표본이 올 때만 h = 표본 간격으로 갱신.
엔코더 튐 거부: 표본 시각에 걸친 엔코더 주기 중 한 주기에 0.5 rad/s 넘게 뛴 관절이 있으면, 원값과 직전 정상값 중 자이로를 더 잘 설명하는 쪽을 씀(엔코더만 튐 → 직전값).
조화 예측(lead): v̂ = v_f·cos(ωa) + (v̇_f/ω)·sin(ωa), a = 송신 시각 − 표본 시각, v̇_f는 저역 단 상태로 정확히((hp − lp)/τ_l). a > a_max면 직전 τ 유지 후 반씩 줄임.
vel: τ_vib = −k·v̂ (E3: Re(H) > 0 → 음의 속도 되먹임이 감쇠), ppf: ξ = ∫v_f(1 Hz 누설), η̈ + 2ζ_f ω_f η̇ + ω_f² η = ω_f² ξ, τ_vib = +k·η̂(같은 예측).
기울기 게이트: MIT 시작 gate 창의 IMU 정지 기울기 atan2(y, −z) < 문턱이면 k = 0 — 처짐이 낮으면 위상이 약 110° 뒤집혀 양의 되먹임 위험.
사용: d = VibDamp(kin, tb, cfg); out = d.step(t_mit, now, q, qd, [(t_g, 자이로 원시), …], accel_raw) → out["tau"]
"""
import collections
import math
from dataclasses import dataclass

import numpy as np

from link_est import jacobians

TAU_STEP = 16.0 / 255  # MIT 토크 한 칸[N·m]
SPIKE_JUMP = 0.5       # 엔코더 한 주기 속도 점프 문턱[rad/s]


@dataclass
class VibDampCfg:
    law: str = "vel"           # vel / ppf
    k: float = 0.0             # vel: N·m/(rad/s), ppf: N·m/rad
    band: tuple = (4.0, 16.0)  # 대역통과[Hz] — 기하 중심 8 Hz에서 연속계 위상 0
    gate_deg: float = 35.0     # 기울기 문턱
    gate_win: tuple = (0.3, 1.8)  # MIT 진입 뒤 기울기 평균 창[s] — 이 창이 끝나야 τ_vib 시작
    deadband: bool = False
    tau_max: float = 0.3       # 안전 포화[N·m]
    ppf_f: float = 8.0         # ω_f/2π[Hz] — 처프 강제응답 피크(E3 F1·F3)
    ppf_zeta: float = 0.3
    pred_f: float = 8.0        # 조화 예측 주파수[Hz], 0이면 예측 안 함 — 처프 피크(E3 F1·F3)
    age_max: float = 0.06      # 표본 나이 상한[s]


def bp_phase_deg(band, f, fs):  # 아래 이산 대역통과(1차 HP·1차 LP, 고정 주기 1/fs)의 f에서 위상[deg]
    h = 1.0 / fs; th = 1 / (2 * math.pi * band[0]); tl = 1 / (2 * math.pi * band[1])
    ah = th / (th + h); al = h / (tl + h); z = np.exp(-1j * 2 * math.pi * f * h)
    H = ah * (1 - z) / (1 - ah * z) * al / (1 - (1 - al) * z)
    return math.degrees(np.angle(H))


class VibDamp:
    def __init__(self, kin, tb, cfg):
        self.kin, self.tb, self.c = kin, tb, cfg
        self.tl = 1 / (2 * math.pi * cfg.band[1]); self.th = 1 / (2 * math.pi * cfg.band[0])
        self.hp = 0.0; self.x_prev = None; self.lp = 0.0; self.t_g = None
        self.xi = 0.0; self.eta = 0.0; self.eta_d = 0.0
        self.enc = collections.deque(maxlen=40)   # (t, q, q̇ 원값, 튐 후보 관절 bool[6])
        self.qd_good = None
        self.acc = []; self.tilt = None; self.enabled = False
        self.v = self.vf = self.vfd = 0.0; self.wy = self.ey = 0.0
        self.tau = 0.0; self.stale_n = 0
        self.n_samples = self.n_reject = self.n_stale = self.n_cycles = self.n_badtime = 0

    def _add_encoder(self, t, q, qd):  # 한 주기 점프 > 문턱인 관절을 튐 후보로 — 판정은 자이로 표본 처리 때
        qd = np.asarray(qd, float)
        cand = np.zeros(6, bool) if self.qd_good is None else np.abs(qd - self.qd_good) > SPIKE_JUMP
        self.enc.append((t, np.asarray(q, float), qd, cand))
        self.qd_good = np.where(cand, self.qd_good if self.qd_good is not None else qd, qd)

    def _qd_at(self, t):  # 표본 시각의 (q, q̇ 원값, q̇ 튐 대체값, 후보 있음) — 이력 선형 보간, 범위 밖은 끝값
        e = self.enc
        if t >= e[-1][0] or len(e) == 1:
            i0 = i1 = len(e) - 1; w = 0.0
        elif t <= e[0][0]:
            i0 = i1 = 0; w = 0.0
        else:
            i1 = next(i for i in range(len(e)) if e[i][0] >= t); i0 = i1 - 1
            w = (t - e[i0][0]) / max(e[i1][0] - e[i0][0], 1e-6)
        q = (1 - w) * e[i0][1] + w * e[i1][1]
        raw = (1 - w) * e[i0][2] + w * e[i1][2]
        cand = e[i0][3] | e[i1][3]
        held = raw.copy()
        if cand.any():  # 후보 관절은 후보 직전 정상 주기 값으로
            j0 = i0
            while j0 > 0 and (e[j0][3] & cand).any():
                j0 -= 1
            held[cand] = e[j0][2][cand]
        return q, raw, held, bool(cand.any())

    def _bandpass(self, x, h):
        if self.x_prev is None:
            self.x_prev = x
        ah = self.th / (self.th + h); self.hp = ah * (self.hp + x - self.x_prev); self.x_prev = x
        self.lp += h / (self.tl + h) * (self.hp - self.lp)
        return self.lp, (self.hp - self.lp) / self.tl  # v_f, v̇_f(저역 단 미분, 지연 없음)

    def _sample(self, t_g, g_raw):
        q, raw, held, cand = self._qd_at(t_g)
        Jw, _ = jacobians(self.kin, q, self.tb.R_l6)
        wy = float((self.tb.rot(q) @ np.asarray(g_raw))[1])
        ey = float(Jw[1] @ raw)
        if cand:  # 엔코더만 튐: 직전값이 자이로를 더 잘 설명하면 거부
            ey_h = float(Jw[1] @ held)
            if abs(wy - ey_h) + 0.05 < abs(wy - ey):
                ey = ey_h; self.n_reject += 1
        h = (t_g - self.t_g) if self.t_g is not None else 0.0
        self.t_g = t_g; self.wy, self.ey = wy, ey; self.v = wy - ey
        if 0 < h < 0.2:
            self.vf, self.vfd = self._bandpass(self.v, h)
            if self.c.law == "ppf":
                self.xi += h * (self.vf - 2 * math.pi * 1.0 * self.xi)  # 누설 적분 — 변위형 입력
                wf = 2 * math.pi * self.c.ppf_f
                self.eta_d += h * (wf * wf * (self.xi - self.eta) - 2 * self.c.ppf_zeta * wf * self.eta_d)
                self.eta += h * self.eta_d
        self.n_samples += 1

    def _predict(self, x, xd, a):
        if not self.c.pred_f:
            return x
        w = 2 * math.pi * self.c.pred_f
        return x * math.cos(w * a) + xd / w * math.sin(w * a)

    def step(self, t_mit, now, q, qd, gyro_samples, accel_raw):
        c = self.c
        self._add_encoder(now, q, qd); self.n_cycles += 1
        for t_g, g in gyro_samples:
            if t_g > now + 0.05 or (self.t_g is not None and t_g <= self.t_g):  # 미래·역행 시각(도메인 전환 잔재) 표본은 버림
                self.n_badtime += 1; continue
            self._sample(t_g, g)
        if self.tilt is None:  # 게이트 창: 원시 가속도 모아 기울기 → 문턱 판정, 그 전엔 τ_vib = 0
            if c.gate_win[0] <= t_mit <= c.gate_win[1] and accel_raw is not None:
                self.acc.append(np.asarray(accel_raw, float))
            if t_mit > c.gate_win[1]:
                g = np.mean(self.acc, axis=0) if self.acc else np.array([np.nan] * 3)
                self.tilt = math.degrees(math.atan2(g[1], -g[2]))
                self.enabled = bool(self.tilt >= c.gate_deg)
        age = (now - self.t_g) if self.t_g is not None else float("inf")
        if c.law == "ppf":
            raw = c.k * self._predict(self.eta, self.eta_d, age); vhat = self.vf
        else:
            vhat = self._predict(self.vf, self.vfd, age); raw = -c.k * vhat
        if age > c.age_max:  # 너무 묵은 표본: 직전 τ 유지 → 반씩 줄임
            self.stale_n += 1; self.n_stale += 1
            tau = self.tau if self.stale_n == 1 else self.tau * 0.5
        else:
            self.stale_n = 0
            tau = float(np.clip(raw, -c.tau_max, c.tau_max)) if self.enabled else 0.0
        if c.deadband and abs(tau) < TAU_STEP / 2:
            tau = 0.0
        self.tau = tau
        return dict(v=self.v, wy=self.wy, ey=self.ey, vf=self.vf, vhat=vhat, raw=raw, tau=tau, enabled=self.enabled,
                    tilt=self.tilt, age=age, t_g=self.t_g, n_new=len(gyro_samples), rejects=self.n_reject, stale=self.n_stale)
