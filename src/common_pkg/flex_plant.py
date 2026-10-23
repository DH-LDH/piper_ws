"""한 관절 2질량 모델(모터 — 감속기 스프링·감쇠 — 링크), 토크는 MIT 명령 단위. 가짜 팔·vib_sim용 (D2-2 사전 검증).

Jm·ω̇m = τm − K(θm − θl) − c(ωm − ωl) − bm·ωm,   Jl·ω̇l = K(θm − θl) + c(ωm − ωl) + τ_ext
엔코더는 θm·ωm(모터 쪽), 자이로는 ωl(링크 쪽)을 본다. 중력은 상수라 TDE가 지우므로 뺌.
모터 쪽 마찰: 운동 Fc·sgn(ωm), 정지 — |ωm| < w_stick이고 걸린 토크 < Fs면 모터를 붙임(E4 J1·J3 헌팅, E2 엔코더에 진동 1/3.6만 보임).
"""
import numpy as np


class FlexJoint:
    def __init__(self, Jm, Jl, K, c, bm=0.0, Fs=0.0, Fc=0.0, w_stick=1e-3):
        self.Jm, self.Jl, self.K, self.c, self.bm = Jm, Jl, K, c, bm
        self.Fs, self.Fc, self.w_stick = Fs, Fc, w_stick
        self.x = np.zeros(4)  # θm, ωm, θl, ωl

    def _f(self, x, tau_m, tau_ext):
        tm, wm, tl, wl = x
        fs = self.K * (tm - tl) + self.c * (wm - wl)
        fr = self.Fc * np.sign(wm) if abs(wm) >= self.w_stick else 0.0
        return np.array([wm, (tau_m - fs - self.bm * wm - fr) / self.Jm, wl, (fs + tau_ext) / self.Jl])

    def step(self, tau_m, tau_ext, h):  # RK4, 입력은 h 동안 상수
        x = self.x
        if self.Fs > 0 and abs(x[1]) < self.w_stick:  # 정지마찰: 모터에 걸린 토크가 Fs 미만이면 모터 고정, 링크만 적분
            net = tau_m - self.K * (x[0] - x[2]) - self.c * (x[1] - x[3])
            if abs(net) < self.Fs:
                self.x = self._stuck(x, tau_ext, h)
                return self.x
        k1 = self._f(x, tau_m, tau_ext); k2 = self._f(x + h / 2 * k1, tau_m, tau_ext)
        k3 = self._f(x + h / 2 * k2, tau_m, tau_ext); k4 = self._f(x + h * k3, tau_m, tau_ext)
        nx = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if self.Fc > 0 and np.sign(nx[1]) != np.sign(x[1]) and x[1] != 0:  # 속도가 0을 지나면 일단 멈춤 → 다음 단계에서 정지마찰 판정
            nx[1] = 0.0
        self.x = nx
        return self.x

    def _stuck(self, x, tau_ext, h):  # 모터 고정(θm 일정, ωm = 0), 링크 RK4
        tm = x[0]

        def g(y):
            fs = self.K * (tm - y[0]) + self.c * (0.0 - y[1])
            return np.array([y[1], (fs + tau_ext) / self.Jl])
        y = x[2:]
        k1 = g(y); k2 = g(y + h / 2 * k1); k3 = g(y + h / 2 * k2); k4 = g(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return np.array([tm, 0.0, y[0], y[1]])
