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
    pred_mode: str = "res"     # res: 공진기 뒤진 직교(8절 10/26 밤 5), old: 대역통과 + 앞선 직교(10-26 리밋 사이클 재현용)
    res_q: float = 1.0         # 공진기 Q
    hp_f: float = 1.0          # res 입력 1차 고역통과[Hz]
    lead_deg: float = 0.0      # φ_lead — 회전각 θ = ω₀a + φ_lead
    gate_hi: float = 37.0      # 기울기 상한(8절 10/26 밤 5: 위상 확인된 범위 35–37°)
    hf_max: float = 0.02       # 감시: τ_vib 16 Hz 이상 rms(0.25 s) 상한[N·m]
    hf_win: int = 50           # 감시 ② rms 창[주기] — 0.5 s(8절 10/27 오전 3)
    hf_hold: float = 0.5       # 감시 ②: 상한 초과가 이만큼 연속일 때만 끔[s] — 충격 한 번의 27 Hz 울림은 넘김(8절 10/27 오전 3)
    clip_win: int = 100        # 감시 ①b: 클립에 붙은 주기 비율 창[주기] — 1 s(8절 10/27 오전 4)
    clip_frac: float = 0.5     # 감시 ①b: 창 안 클립 비율 상한 — 충격 한 번은 0.13–0.32 s라 넘김
    sum_n: int = 10            # 감시 ①a: 최근 50주기 중 합 포화(처프 + τ_vib 또는 명령 상한) 횟수 상한
    notch_f: float = 0.0       # res 입력 노치 중심[Hz], 0이면 끔(8절 10/27 저녁: 29.6 Hz — 25–35 Hz 국소 모드)
    notch_q: float = 1.5       # 노치 Q — 1.5면 8 Hz −11°


def pred_gain(f, a, cfg):  # 연속계 |v̂/x|·위상(지연 a 뒤 v 대비 아님, 입력 v 대비) — res·old 공통 검토용
    s = 1j * 2 * math.pi * np.asarray(f, float); w0 = 2 * math.pi * cfg.pred_f; th = w0 * a + math.radians(cfg.lead_deg)
    if cfg.pred_mode == "res":
        hp = s / (s + 2 * math.pi * cfg.hp_f); eta = (w0 * w0 / cfg.res_q) / (s * s + (w0 / cfg.res_q) * s + w0 * w0) * hp
        if cfg.notch_f:
            wn = 2 * math.pi * cfg.notch_f; eta = eta * (s * s + wn * wn) / (s * s + (wn / cfg.notch_q) * s + wn * wn)
        return (s * eta / w0) * math.cos(th) - eta * math.sin(th)
    th_h = 1 / (2 * math.pi * cfg.band[0]); tl = 1 / (2 * math.pi * cfg.band[1])
    HP = th_h * s / (1 + th_h * s); LP = 1 / (1 + tl * s)
    return HP * LP * math.cos(th) + (HP - HP * LP) / tl / w0 * math.sin(th)


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
        self.h1 = 0.0; self.x1_prev = None; self.r = 0.0; self.rd = 0.0  # res: 1 Hz 고역통과, 공진기 η·η'
        self.np_ = self.nq = 0.0; self.nx_prev = None  # 노치 상태
        self.enc = collections.deque(maxlen=40)   # (t, q, q̇ 원값, 튐 후보 관절 bool[6])
        self.qd_good = None
        self.acc = []; self.tilt = None; self.enabled = False
        self.v = self.vf = self.vfd = 0.0; self.wy = self.ey = 0.0
        self.tau = 0.0; self.stale_n = 0
        self.hist = collections.deque(maxlen=cfg.clip_win); self.tripped = None; self.trip_why = ""  # 감시 ①b: 최근 1 s 중 τ_vib가 클립에 붙은 비율(사각파 발진)
        self.sum_hist = collections.deque(maxlen=50)  # 감시 ①a: 합 포화 — note_sum_sat()로 mit_probe가 알려 줌
        self.hp4 = [[0.0, 0.0], [0.0, 0.0]]; self.hp4_u = 0.0  # 감시 ②: 4차 Butterworth 20 Hz 고역(2차 두 단), 8 Hz 통과 0.026
        self.hf = collections.deque(maxlen=cfg.hf_win); self.hf_since = None; self.thp = 0.0; self.t_prev_tau = 0.0; self.t_last = None  # 감시 ②: τ_vib 16 Hz 고역 rms(0.25 s) > hf_max — 성장 단계에서 끔
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

    def _resonator(self, x, h):  # 1 Hz 고역통과 → η'' + (ω₀/Q)η' + ω₀²η = (ω₀²/Q)x, RK4(입력은 표본 사이 선형) — 오일러는 8 Hz에서 13° 앞섰음
        if self.x1_prev is None:
            self.x1_prev = x; self.u_prev = 0.0
        a1 = 1 / (1 + 2 * math.pi * self.c.hp_f * h); self.h1 = a1 * (self.h1 + x - self.x1_prev); self.x1_prev = x
        w0 = 2 * math.pi * self.c.pred_f; Q = self.c.res_q; u0, u1 = self.u_prev, self.h1; self.u_prev = u1
        f = lambda r, rd, u: (rd, (w0 * w0 / Q) * u - (w0 / Q) * rd - w0 * w0 * r)
        r, rd = self.r, self.rd; um = (u0 + u1) / 2
        k1 = f(r, rd, u0); k2 = f(r + h / 2 * k1[0], rd + h / 2 * k1[1], um)
        k3 = f(r + h / 2 * k2[0], rd + h / 2 * k2[1], um); k4 = f(r + h * k3[0], rd + h * k3[1], u1)
        self.r = r + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); self.rd = rd + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])

    def _notch(self, x, h):  # (s² + ωn²)/(s² + (ωn/Q)s + ωn²) = 1 − 대역통과, RK4 4 ms 이하 소단계(ωn·h가 커지는 빈틈 대비)
        if self.nx_prev is None:
            self.nx_prev = x
        wn = 2 * math.pi * self.c.notch_f; b = wn / self.c.notch_q; x0 = self.nx_prev; self.nx_prev = x
        f = lambda p, q, u: (q, u - wn * wn * p - b * q)  # q = s/(s² + b s + ωn²)·x
        n = max(1, math.ceil(h / 0.004)); hs = h / n; p, q = self.np_, self.nq
        for i in range(n):
            ua = x0 + (x - x0) * i / n; ub = x0 + (x - x0) * (i + 1) / n; um = (ua + ub) / 2
            k1 = f(p, q, ua); k2 = f(p + hs / 2 * k1[0], q + hs / 2 * k1[1], um)
            k3 = f(p + hs / 2 * k2[0], q + hs / 2 * k2[1], um); k4 = f(p + hs * k3[0], q + hs * k3[1], ub)
            p += hs / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); q += hs / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.np_, self.nq = p, q
        return x - b * q

    def _sample(self, t_g, g_raw, Jw, R):  # Jw·R은 주기당 1회 계산값 재사용(표본마다 0.5 ms × 몰림 최대 40개 → 루프 멈춤, 8절 10/26 밤 7)
        q, raw, held, cand = self._qd_at(t_g)
        wy = float((R @ np.asarray(g_raw))[1])
        ey = float(Jw[1] @ raw)
        if cand:  # 엔코더만 튐: 직전값이 자이로를 더 잘 설명하면 거부
            ey_h = float(Jw[1] @ held)
            if abs(wy - ey_h) + 0.05 < abs(wy - ey):
                ey = ey_h; self.n_reject += 1
        h = (t_g - self.t_g) if self.t_g is not None else 0.0
        self.t_g = t_g; self.wy, self.ey = wy, ey; self.v = wy - ey
        if 0 < h < 0.2:
            if self.c.pred_mode == "res":
                self._resonator(self._notch(self.v, h) if self.c.notch_f else self.v, h)
                self.vf = self.rd / (2 * math.pi * self.c.pred_f)
            else:
                self.vf, self.vfd = self._bandpass(self.v, h)
            if self.c.law == "ppf":
                self.xi += h * (self.vf - 2 * math.pi * 1.0 * self.xi)  # 누설 적분 — 변위형 입력
                wf = 2 * math.pi * self.c.ppf_f
                self.eta_d += h * (wf * wf * (self.xi - self.eta) - 2 * self.c.ppf_zeta * wf * self.eta_d)
                self.eta += h * self.eta_d
        self.n_samples += 1

    def _predict(self, x, xd, a):  # old: x·cos θ + (ẋ/ω)·sin θ
        if not self.c.pred_f:
            return x
        w = 2 * math.pi * self.c.pred_f; th = w * a + math.radians(self.c.lead_deg)
        return x * math.cos(th) + xd / w * math.sin(th)

    def _predict_res(self, a):  # res: v̂ = (η'/ω₀)cos θ − η sin θ — 고주파에서 두 항 다 줄어듦
        w = 2 * math.pi * self.c.pred_f; th = w * a + math.radians(self.c.lead_deg)
        return self.rd / w * math.cos(th) - self.r * math.sin(th)

    def _hp4(self, u, h):  # 4차 Butterworth 20 Hz 고역 = 2차 고역 두 단(Q 0.541·1.307), RK4 4 ms 이하 소단계 — 감시 전용(루프 밖)
        wc = 2 * math.pi * 20.0; u0 = self.hp4_u; self.hp4_u = u; n = max(1, math.ceil(h / 0.004)); hs = h / n
        y = u
        for i in range(n):
            xa = u0 + (u - u0) * i / n; xb = u0 + (u - u0) * (i + 1) / n
            for st, Q in zip(self.hp4, (0.5412, 1.3066)):  # 단마다 입력 = 앞 단 출력(소단계 처음·끝 선형)
                b_ = wc / Q; f = lambda p, v, w: (v, w - b_ * v - wc * wc * p)
                p, v = st; xm = (xa + xb) / 2
                k1 = f(p, v, xa); k2 = f(p + hs / 2 * k1[0], v + hs / 2 * k1[1], xm)
                k3 = f(p + hs / 2 * k2[0], v + hs / 2 * k2[1], xm); k4 = f(p + hs * k3[0], v + hs * k3[1], xb)
                ya = xa - b_ * v - wc * wc * p
                p += hs / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]); v += hs / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
                st[0], st[1] = p, v; xa, xb = ya, xb - b_ * v - wc * wc * p
            y = xb
        return y

    def note_sum_sat(self, sat):  # mit_probe가 송신 직전에 알려 줌 — 처프 + τ_vib 합 포화 또는 명령 상한
        self.sum_hist.append(bool(sat))

    def step(self, t_mit, now, q, qd, gyro_samples, accel_raw):
        c = self.c
        self._add_encoder(now, q, qd); self.n_cycles += 1
        if gyro_samples:
            qn = np.asarray(q, float); Jw, _ = jacobians(self.kin, qn, self.tb.R_l6); R = self.tb.rot(qn)
        for t_g, g in gyro_samples:
            if t_g > now + 0.05 or (self.t_g is not None and t_g <= self.t_g):  # 미래·역행 시각(도메인 전환 잔재) 표본은 버림
                self.n_badtime += 1; continue
            self._sample(t_g, g, Jw, R)
        if self.tilt is None:  # 게이트 창: 원시 가속도 모아 기울기 → 문턱 판정, 그 전엔 τ_vib = 0
            if c.gate_win[0] <= t_mit <= c.gate_win[1] and accel_raw is not None:
                self.acc.append(np.asarray(accel_raw, float))
            if t_mit > c.gate_win[1]:
                g = np.mean(self.acc, axis=0) if self.acc else np.array([np.nan] * 3)
                self.tilt = math.degrees(math.atan2(g[1], -g[2]))
                self.enabled = bool(c.gate_deg <= self.tilt <= c.gate_hi)
        age = (now - self.t_g) if self.t_g is not None else float("inf")
        if c.pred_mode == "res":
            vhat = self._predict_res(age) if c.pred_f else self.vf
            if c.law == "ppf":  # 공진기 η에 +k (PPF 구조) — η̂ = η cos θ + (η'/ω₀) sin θ
                w = 2 * math.pi * c.pred_f; th = w * age + math.radians(c.lead_deg)
                raw = c.k * (self.r * math.cos(th) + self.rd / w * math.sin(th))
            else:
                raw = -c.k * vhat
        elif c.law == "ppf":
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
        self.hist.append(abs(tau) >= 0.9 * c.tau_max)
        dtc = (now - self.t_last) if self.t_last is not None else 0.01; self.t_last = now
        self.hf.append(self._hp4(tau, dtc))
        hf_rms = float(np.sqrt(np.mean(np.square(self.hf)))) if len(self.hf) == self.hf.maxlen else 0.0
        if self.tripped is None and len(self.hist) == self.hist.maxlen and sum(self.hist) >= c.clip_frac * self.hist.maxlen:
            self.tripped = t_mit; self.trip_why = f"클립에 붙은 주기 {sum(self.hist)}/{self.hist.maxlen}(①b)"
        if self.tripped is None and sum(self.sum_hist) >= c.sum_n:
            self.tripped = t_mit; self.trip_why = f"합 포화 {sum(self.sum_hist)}회/0.5 s(①a)"
        self.hf_since = (self.hf_since if self.hf_since is not None else t_mit) if hf_rms > c.hf_max else None
        if self.tripped is None and self.hf_since is not None and t_mit - self.hf_since >= c.hf_hold:
            self.tripped = t_mit; self.trip_why = f"20 Hz 이상 rms {hf_rms:.3f} > {c.hf_max} N·m" + (f" {c.hf_hold:.1f} s 연속" if c.hf_hold else "") + "(②)"
        if self.tripped is not None:
            tau = 0.0
        self.tau = tau
        return dict(v=self.v, wy=self.wy, ey=self.ey, vf=self.vf, vhat=vhat, raw=raw, tau=tau, enabled=self.enabled,
                    tilt=self.tilt, age=age, t_g=self.t_g, n_new=len(gyro_samples), rejects=self.n_reject, stale=self.n_stale,
                    tripped=self.tripped, trip_why=self.trip_why)
