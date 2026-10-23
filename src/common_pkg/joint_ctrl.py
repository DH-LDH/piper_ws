"""한 관절 토크 제어기 — TDC(TDE + 극점 배치)와 스위칭(고정 / SS-ASMC 2020 / ASMC-2016), 하이브리드. SDK·CAN·파일 I/O 없음.

mit_probe(실기 시험)에서 떼어 냄. 이후 piper_driver_node의 한 모드로 옮겨 홀바디 아래층(관절 토크 층)으로 쓴다.
식: τ = τ(t−L) − M̄·q̈(t−L) + M̄(q̈_d + (Ks+Kg)ė + Ks·Kg·e) + 스위칭, 하이브리드는 PD를 펌웨어가 맡고 TDE 보정 + M̄q̈_d만.
확장 자리: ref_filter(기준 궤적 앞단, input shaper 자리), dist_comp(τ 뒤 외란·감쇠 보상, DOB·τ_vib 자리) — 기본은 통과.
사용: c = JointCtrl(cfg, tau0); out = c.step(q, qd, qr, vr, ar, h); (상위에서 포화) ; c.commit(tau_sent)
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class JointCtrlCfg:
    mbar: float = 0.005          # 관성 추정 M̄[kg·m²]
    ks: float = 20.0             # 극점 Ks[1/s] — 슬라이딩 변수 s = ė + Ks·e
    kg: float = 20.0             # 극점 Kg[1/s]
    acc_fc: float = 15.0         # q̈ 추정 저역통과[Hz]
    hybrid: bool = False         # 펌웨어 MIT PD 유지, 우리는 TDE 보정 + M̄q̈_d(+ 스위칭)만
    kp_fw: float = 0.0           # 하이브리드에서 펌웨어 PD 추정용(내부 배율 1 가정)
    kd_fw: float = 0.0
    switch: str = "none"         # none / fixed / asmc(SS-ASMC 2020) / asmc16(Baek·Jin·Han 2016)
    sw_k: float = 0.0            # 고정 스위칭 K[rad/s²]
    sw_phi: float = 0.2          # 경계층 φ[rad/s], 적응형에서 0이면 sgn
    k_max: float = 20.0          # K̂ 안전 상한
    eps: float = 0.05            # |s| 문턱 ε
    phi: float = 200.0           # ASMC-2016 적응 속도
    alpha: float = 1.0           # ASMC-2016 α
    delta: float = 1.0           # SS-ASMC δ̄
    gamma: float = 2e4           # SS-ASMC γ̄
    a_exp: float = 1.5           # SS-ASMC child 지수
    eta_lo: float = 100.0        # SS-ASMC η 하한
    eta_hi: float = 2000.0       # SS-ASMC η 상한
    mu: float = 200.0            # SS-ASMC |K̂ 변화율| 상한


class JointCtrl:
    def __init__(self, cfg, tau0, ref_filter=None, dist_comp=None):
        self.c = cfg
        self.ref_filter = ref_filter    # (qr, vr, ar, h) → (qr, vr, ar)
        self.dist_comp = dist_comp      # (상태 dict) → 추가 토크[N·m]
        self.tau_prev = float(tau0)     # 직전에 실제로 보낸 토크(포화 후)
        self.qd_prev = None; self.qdd_f = 0.0
        self.pd_fw_prev = 0.0
        self.k_hat = 0.0; self.eta = cfg.eta_lo  # 적응 스위칭 게인 K̂, SS-ASMC 갱신 파라미터 η̂(하한에서 시작)
        self._e = self._ed = 0.0

    def _accel(self, qd, h):  # 속도 차분 + 1차 저역통과
        if self.qd_prev is not None:
            al = h / (h + 1.0 / (2 * math.pi * self.c.acc_fc))
            self.qdd_f += al * ((qd - self.qd_prev) / h - self.qdd_f)
        self.qd_prev = qd

    def _switch(self, sv, h):
        c = self.c
        if c.switch == "asmc16":  # 2016 TIE 식 (11): |s| ≥ ε면 φ|s|/α로 증가, 작으면 −φα/|s|로 급감
            th = 1.0 if abs(sv) - c.eps >= 0 else -1.0
            if self.k_hat > 1e-4:  # 논문 실험의 k̂ > 0.0001 처리
                kdot = c.phi * (max(abs(sv), 1e-6) / c.alpha) ** th * th  # s = 0이면 감소항 발산 방지
            else:
                kdot = c.phi * abs(sv) / c.alpha
            self.k_hat = float(np.clip(self.k_hat + kdot * h, 0.0, c.k_max))
        elif c.switch == "asmc":  # 2020 (12)–(15) — parent (13) K̂, child (14) η̂, 오일러 적분
            chi = 1.0 if abs(sv) - c.eps >= 0 else -1.0
            kdot = c.delta * self.eta * abs(sv) * (chi if self.k_hat > 0 and sv != 0 else 1.0)
            sat_k = abs(kdot) >= c.mu
            kdot = float(np.clip(kdot, -c.mu, c.mu))
            if sat_k:
                etadot = 0.0
            elif self.eta >= c.eta_hi:
                etadot = -c.gamma * abs(sv)
            elif self.eta <= c.eta_lo:
                etadot = c.gamma * abs(sv)
            else:
                etadot = -c.gamma * abs(sv) ** c.a_exp * chi
            self.k_hat = float(np.clip(self.k_hat + kdot * h, 0.0, c.k_max))
            self.eta = float(np.clip(self.eta + etadot * h, c.eta_lo, c.eta_hi))
        elif c.switch == "fixed":  # M̄·K·sat(s/φ)
            return c.mbar * c.sw_k * float(np.clip(sv / c.sw_phi, -1.0, 1.0)) if c.sw_k else 0.0
        else:
            return 0.0
        sgn = (1.0 if sv >= 0 else -1.0) if not c.sw_phi else float(np.clip(sv / c.sw_phi, -1.0, 1.0))
        return c.mbar * self.k_hat * sgn

    def step(self, q, qd, qr, vr, ar, h):  # h: 직전 호출과의 간격[s](첫 호출은 0) → 계산 결과 dict, 'tau'가 보낼 토크(포화 전)
        c = self.c
        if self.ref_filter is not None:
            qr, vr, ar = self.ref_filter(qr, vr, ar, h)
        self._accel(qd, h)
        e = qr - q; ed = vr - qd
        tde = self.tau_prev - c.mbar * self.qdd_f
        if c.hybrid:  # τ_ff = [τ_ff + 펌웨어 PD](t−L) − M̄q̈(t−L) + M̄q̈_d
            tde += self.pd_fw_prev
            pd = c.mbar * ar
        else:
            pd = c.mbar * (ar + (c.ks + c.kg) * ed + c.ks * c.kg * e)
        sv = ed + c.ks * e  # 2020 논문 (6) s = ė + Ks·e
        sw = self._switch(sv, h)
        out = dict(e=e, ed=ed, qdd_f=self.qdd_f, ar=ar, tau_prev=self.tau_prev, tde=tde, pd=pd, sw=sw, s=sv,
                   K=self.k_hat, eta=self.eta)
        extra = self.dist_comp(out) if self.dist_comp is not None else 0.0
        out["dist"] = extra
        out["tau"] = tde + pd + sw + extra
        self._e, self._ed = e, ed
        return out

    def commit(self, tau_sent):  # 상위에서 포화한 뒤 실제로 보낸 토크 → 다음 주기 TDE
        self.tau_prev = float(tau_sent)
        self.pd_fw_prev = self.c.kp_fw * self._e + self.c.kd_fw * self._ed if self.c.hybrid else 0.0
