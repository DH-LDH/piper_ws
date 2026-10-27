"""(가) 한 MIT 시행 안 계획 — 컨디셔닝 a′ → 식별 처프 → 게이트·φ_lead → k 계단 처프 → a′ on/off 짝 → 재식별 (8절 10/27 오전 7·오후 8–9, E4 14-1 판단 28). SDK·CAN 없음.

계획 문자열(쉼표 구분): sine:N(a′ N주기, k 0) · id:T(hold + 처프 T s, k 0 → 식별) · ck:K:T(hold + 처프 T s, k = K → ∠L 확인) · onoff:N(a′ N주기, 주기마다 k_ok/0 교대) · hold:T
식별: 처프 위상 고정 lock-in(6.5–9 Hz, 0.25 Hz bin, 창마다 선형 추세 제거) → 단일 모드 맞춤 H ≈ c·G(f; f₀, ζ), G는 공진에서 1∠0° → ψ = ∠c.
게이트: H 피크 ≥ h_min, 위상 잔차 rms ≤ res_max, 설명력 ≥ r2_min, 앞·뒤 절반 ψ 차 ≤ half_max. φ_lead = 해 of ψ + ∠(−C)_설계(f₀; φ) = 0(0–359° 격자).
ck: 같은 bin에서 L = −τ_vib/τ_sent, ψ_L = ∠(L ≈ c_L·G) — |ψ_L| ≤ psi_l_max이고 그 구간 H의 ψ가 식별값 ±30° 안이면 k_ok = K. onoff는 마지막 통과 k_ok(없으면 0).
"""
import math
import threading
from dataclasses import dataclass

import numpy as np

from vib_damp import VibDampCfg, pred_gain

# k 1.0 고주파·저주파 |H| 실측(101750 4–36 Hz 처프, 141808 1–6 Hz 처프, 1 Hz·0.25 Hz bin 중앙값) — φ_lead별 |L| 확인용
H_HF = {12: .162, 13: .144, 14: .115, 15: .078, 16: .075, 17: .045, 18: .060, 19: .069, 20: .054, 21: .056, 22: .075, 23: .086, 24: .085,
        25: .180, 26: .094, 27: .137, 28: .153, 29: .105, 30: .132, 31: .250, 32: .155, 33: .233, 34: .158, 35: .220, 36: .108}
H_LF = {1.0: .024, 1.25: .026, 1.5: .023, 1.75: .022, 2.0: .023, 2.25: .028, 2.5: .025, 2.75: .038, 3.0: .035, 3.25: .031, 3.5: .036,
        3.75: .046, 4.0: .038, 4.25: .037, 4.5: .043, 4.75: .102, 5.0: .059}
BINS = np.arange(6.5, 9.01, 0.25)


@dataclass
class PlanCfg:
    chirp_amp: float = 0.12
    f0: float = 4.0            # 처프 시작[Hz]
    f1: float = 12.0           # 처프 끝[Hz]
    sine_amp: float = 5.0      # a′ 진폭[deg]
    sine_hz: float = 0.25
    sine_ramp: float = 1.0     # sine 구간 앞뒤 진폭 램프[s]
    k_ramp: float = 0.2        # k 바꿀 때 선형 램프[s] — 전환 경계 τ 튐 방지
    h_min: float = 0.5
    res_max: float = 35.0      # 맞춤 위상 잔차 rms 상한[deg]
    r2_min: float = 0.7
    half_max: float = 30.0     # 앞·뒤 절반 ψ 차 상한[deg]
    psi_l_max: float = 30.0    # ck 구간 ψ_L 상한[deg]
    drift_max: float = 30.0    # ck 구간 ψ가 식별값과 다를 수 있는 한도[deg]
    l_max: float = 0.3         # 고주파·저주파 |L| 상한(허용 k 계산)
    lead_offset: float = 0.0   # φ_lead 계통 몫[deg] — E4 14절: 0


def parse(s):
    segs = []
    for tok in s.split(","):
        p = tok.strip().split(":")
        if p[0] in ("sine", "onoff"):
            segs.append((p[0], float(p[1]), 0.0))
        elif p[0] in ("id", "hold"):
            segs.append((p[0], float(p[1]), 0.0))
        elif p[0] == "ck":
            segs.append(("ck", float(p[2]), float(p[1])))
        else:
            raise ValueError(f"계획 구간 모름: {tok}")
    return segs


def chirp_phase(t, T, f0, f1):  # 로그 스윕 위상[rad]과 순간 주파수[Hz]
    r = f1 / f0
    return 2 * math.pi * f0 * T / math.log(r) * (r ** (t / T) - 1), f0 * r ** (t / T)


def mode_shape(f, f0, z):  # 힘 → 속도 공진 모양, 공진에서 1∠0°
    return 1j * f * 2 * z * f0 / (f0 ** 2 - f ** 2 + 2j * z * f0 * f)


def fit_mode(f, H):  # |H| 가중 최소제곱 H ≈ c·G — (f₀, ζ, c, 위상 잔차 rms[deg], 설명력)
    F0, Z = np.meshgrid(np.arange(6.8, 9.0, 0.025), np.arange(0.02, 0.4, 0.01), indexing="ij")
    G = mode_shape(f[None, None, :], F0[..., None], Z[..., None]); w = np.abs(H)[None, None, :]
    c = np.sum(w * H * np.conj(G), -1) / np.sum(w * np.abs(G) ** 2, -1)
    r = np.sum(w * np.abs(H - c[..., None] * G) ** 2, -1) / np.sum(w * np.abs(H) ** 2, -1)
    i = np.unravel_index(np.argmin(r), r.shape); f0, z, cc = F0[i], Z[i], c[i]; g = mode_shape(f, f0, z)
    pr = np.degrees(np.angle(H / (cc * g))); ww = np.abs(H)
    return f0, z, cc, float(np.sqrt(np.sum(ww * pr ** 2) / np.sum(ww))), float(1 - r[i])


def psi_with(f, Y, f0, z):  # 정해진 모양으로 Y ≈ c·G의 ∠c[deg]
    g = mode_shape(f, f0, z); w = np.abs(Y)
    return float(np.degrees(np.angle(np.sum(w * Y * np.conj(g)) / np.sum(w * np.abs(g) ** 2))))


def wrap(x):
    return (x + 180.0) % 360.0 - 180.0


def lead_for(psi, f0, age, td, vcfg):  # ψ + ∠(−C)_설계(f₀; φ) = 0인 φ_lead[deg] (0–359 격자)
    best = None
    for ph in range(360):
        c = VibDampCfg(pred_f=vcfg.pred_f, res_q=vcfg.res_q, hp_f=vcfg.hp_f, notch_f=vcfg.notch_f, notch_q=vcfg.notch_q, lead_deg=ph)
        e = abs(wrap(psi + np.degrees(np.angle(pred_gain(f0, age, c))) - 360 * f0 * td))
        if best is None or e < best[0]:
            best = (e, ph)
    return float(best[1])


def k_allowed(lead, age, vcfg, l_max):  # k 1.0 고주파·저주파 |L| 최대 → 허용 k(≤ l_max/|L|)
    c = VibDampCfg(pred_f=vcfg.pred_f, res_q=vcfg.res_q, hp_f=vcfg.hp_f, notch_f=vcfg.notch_f, notch_q=vcfg.notch_q, lead_deg=lead)
    hf = max(h * abs(pred_gain(f, age, c)) for f, h in H_HF.items() if f >= 20)
    lf = max(h * abs(pred_gain(f, age, c)) for f, h in H_LF.items())
    return l_max / max(hf, lf, 1e-6), hf, lf


class Plan:
    def __init__(self, s, cfg, vcfg):
        self.c, self.vc = cfg, vcfg; self.segs = parse(s); self.t0s = []; t = 0.0
        for kind, d, k in self.segs:
            self.t0s.append(t); t += d / cfg.sine_hz if kind in ("sine", "onoff") else d
        self.total = t; self.cur = -1; self.k_now = 0.0; self.k_tgt = 0.0; self.t_last = None
        self.lead = None; self.k_ok = 0.0; self.k_cap = 0.0; self.ident = None; self.results = []
        self.buf = None; self.vbuf = None; self.ck_prev = None  # 직전 ck 통과 여부(None = 아직 ck 없음)
        self.busy = None; self.tripped = False  # 구간 끝 판정 스레드(도는 동안 k 0), 감시 발동 여부(mit_probe가 넣음)

    def at(self, tt):  # (구간 번호, 종류, 구간 안 시각, 사인 env·위상, 처프 값, 처프 위상, 목표 k)
        i = max(j for j, t0 in enumerate(self.t0s) if tt >= t0) if tt >= 0 else 0
        kind, d, kk = self.segs[i]; ts = tt - self.t0s[i]; c = self.c
        out = dict(i=i, kind=kind, ts=ts, sine=None, chirp=0.0, k=0.0)
        if kind in ("sine", "onoff"):
            T = d / c.sine_hz; env = min(1.0, ts / c.sine_ramp, max(0.0, T - ts) / c.sine_ramp)
            denv = (1 / c.sine_ramp if ts < c.sine_ramp else -1 / c.sine_ramp if T - ts < c.sine_ramp else 0.0)
            out["sine"] = (env, denv, 2 * math.pi * c.sine_hz * ts)
            if kind == "onoff":
                idle = self.busy is None or not self.busy.is_alive()
                out["k"] = self.k_ok if (int(ts * c.sine_hz) % 2 == 0 and idle and not self.tripped) else 0.0  # 짝: 홀수 번째 주기 on, 짝수 off
        elif kind in ("id", "ck"):
            ph, _ = chirp_phase(ts, d, c.f0, c.f1)
            out["chirp"] = c.chirp_amp * min(1.0, ts / 0.5, max(0.0, d - ts) / 0.5) * math.sin(ph)
            if kind == "ck":
                out["k"] = kk if self.ck_allowed(kk) else 0.0
        return out

    def ck_allowed(self, kk):  # 게이트 통과·허용 k 안·직전 ck 통과(첫 ck는 예외)일 때만 k 켬
        return (self.lead is not None and 0 < kk <= self.k_cap + 1e-9 and self.ck_prev is not False and not self.tripped
                and (self.busy is None or not self.busy.is_alive()))

    def step_k(self, target, now):  # 목표 k로 선형 램프(k_ramp 동안 큰 쪽 크기만큼)
        h = (now - self.t_last) if self.t_last is not None else 0.0; self.t_last = now
        step = h / self.c.k_ramp * max(abs(target), abs(self.k_now), 0.05) if self.c.k_ramp > 0 else abs(target - self.k_now)  # 0.05 바닥 — 0으로 끝까지 내려감
        self.k_now += float(np.clip(target - self.k_now, -step, step))
        return self.k_now

    def begin(self, i):  # 처프 구간 시작 — 버퍼
        self.cur = i
        self.buf = [] if self.segs[i][0] in ("id", "ck") else None
        self.vbuf = [] if self.buf is not None else None

    def record(self, ts, t_send, tau_sent, tau_vib, age, td):  # ts = 구간 안 시각(처프 계산), t_send = 송신 뒤 절대 시각
        if self.buf is not None:
            self.buf.append((ts, t_send, tau_sent, tau_vib, age, td))

    def record_v(self, t_g, v):
        if self.vbuf is not None:
            self.vbuf.append((t_g, v))

    def _lockin(self, kind, d, buf, vbuf):  # bin별 H, (L), 앞·뒤 절반 H — 버퍼는 인자로만(판정 스레드와 기록이 겹치지 않게)
        b = np.array(buf); vb = np.array(vbuf)
        if len(b) < 50 or len(vb) < 50:
            self.why = f"표본 τ {len(b)}·v {len(vb)}"; return None
        ts, tsend, tau, tvib = b[:, 0], b[:, 1], b[:, 2], b[:, 3]; t_off = np.median(tsend - ts)  # 구간 시각 ↔ 절대 시각
        tq = tsend - t_off; tg = vb[:, 0] - t_off  # τ는 송신 시각, v는 표본(장치) 시각 — 같은 물리 시간축(E4 14절)
        fs = 200.0; tu = np.arange(max(tq[0], tg[0]), min(tq[-1], tg[-1]), 1 / fs)
        vu = np.interp(tu, tg, vb[:, 1]); tauu = np.interp(tu, tq, tau); tvu = np.interp(tu, tq, tvib)
        ph, fi = chirp_phase(tu, d, self.c.f0, self.c.f1); z = np.exp(-1j * ph); rows = []
        res = vu - np.convolve(vu, np.ones(5) / 5, "same")  # 25 ms 이동평균 잔차 — 4–12 Hz 처프 응답은 작고 충격(27 Hz 울림·튐)만 큼
        mad = np.median(np.abs(res - np.median(res))) / 0.6745; bad = np.abs(res) > max(0.15, 8 * mad)
        keep = np.convolve(bad.astype(float), np.ones(int(0.3 * fs) + 1), "same") == 0  # 충격(MIT 순간 튐) ±0.15 s 제외
        self.n_shock = int(np.sum(np.diff(bad.astype(int)) == 1))
        for fb in BINS:
            w0 = np.abs(fi - fb) <= 0.125; w = w0 & keep
            if w.sum() < 20 or w.sum() < 0.5 * w0.sum():
                rows.append(None); continue
            idx = np.where(w)[0]; out = []
            for part in (idx, idx[:len(idx) // 2], idx[len(idx) // 2:]):
                x = tu[part]
                dt = lambda y: y - np.polyval(np.polyfit(x, y, 1), x)  # 창 안 선형 추세 제거(TDE τ_ff 평균·처짐 표류)
                V = np.mean(dt(vu[part]) * z[part]); Tq = np.mean(dt(tauu[part]) * z[part]); Tv = np.mean(dt(tvu[part]) * z[part])
                out.append((V / Tq, -Tv / Tq))
            rows.append(out)
        ok = [i for i, r in enumerate(rows) if r is not None]
        if len(ok) < 5:
            self.why = f"bin {len(ok)}개(τ 시각 {tq[0]:.2f}–{tq[-1]:.2f}, v 시각 {tg[0]:.2f}–{tg[-1]:.2f}, 충격 제외 {100 * (1 - keep.mean()):.0f}%)"; return None
        f = BINS[ok]; R = [rows[i] for i in ok]
        return f, np.array([r[0][0] for r in R]), np.array([r[0][1] for r in R]), np.array([r[1][0] for r in R]), np.array([r[2][0] for r in R])

    def end_async(self, i, log=print):  # 구간 끝 판정을 뒤에서 — 제어 주기를 막지 않음(판정 35 ms 안팎)
        if self.busy is not None:
            self.busy.join()
        self.busy = threading.Thread(target=self.end, args=(i, log, self.buf, self.vbuf), daemon=True); self.buf = self.vbuf = None
        self.busy.start()

    def end(self, i, log=print, buf=None, vbuf=None):  # 구간 끝 판정 — buf 없으면 지금 버퍼(동기 호출)
        kind, d, kk = self.segs[i]
        if buf is None:
            buf, vbuf = self.buf, self.vbuf; self.buf = self.vbuf = None
        if kind not in ("id", "ck") or not buf or len(buf) < 50:
            if kind == "id" and self.ident is None:
                self.ident = dict(seg=i, kind=kind, gate=False, psi=0.0)
            return
        b = np.array(buf); age, td = float(np.median(b[:, 4])), float(np.median(b[:, 5]))
        r = self._lockin(kind, d, buf, vbuf)
        if r is None:
            log(f"[계획] {kind} 구간 {i}: 표본·bin 부족({getattr(self, 'why', '')}) — 판정 안 함(게이트 불통과로 봄)", flush=True)
            if kind == "id" and self.ident is None:
                self.ident = dict(seg=i, kind=kind, gate=False, psi=0.0)
            return
        f, H, L, H1, H2 = r; f0, z, c, res, r2 = fit_mode(f, H); psi = float(np.degrees(np.angle(c)))
        p1, p2 = psi_with(f, H1, f0, z), psi_with(f, H2, f0, z); hp = float(np.abs(H).max())
        rec = dict(seg=i, kind=kind, k=kk, f0=round(float(f0), 3), zeta=round(float(z), 3), psi=round(psi, 1), res=round(res, 1), r2=round(r2, 3),
                   psi_half=(round(p1, 1), round(p2, 1)), h_peak=round(hp, 3), age_ms=round(age * 1000, 1), td_ms=round(td * 1000, 1),
                   bins=len(f), shocks=getattr(self, "n_shock", 0))
        if kind == "id":
            gate = hp >= self.c.h_min and res <= self.c.res_max and r2 >= self.c.r2_min and abs(wrap(p1 - p2)) <= self.c.half_max
            rec["gate"] = bool(gate)
            if self.ident is None:  # 첫 식별만 φ_lead·허용 k를 정함, 뒤 식별은 재식별(Δψ)
                if gate:
                    self.lead = (lead_for(psi, f0, age, td, self.vc) + self.c.lead_offset) % 360
                    self.k_cap, hf, lf = k_allowed(self.lead, age, self.vc, self.c.l_max)
                    rec.update(lead=self.lead, k_cap=round(self.k_cap, 2), L_hf_k1=round(hf, 3), L_lf_k1=round(lf, 3))
                self.ident = rec
            else:
                rec["dpsi"] = round(wrap(psi - self.ident["psi"]), 1)
            log(f"[계획] 식별 구간 {i}: f₀ {f0:.2f} Hz ζ {z:.2f} ψ {psi:+.0f}°(절반 {p1:+.0f}/{p2:+.0f}) 잔차 {res:.0f}° 설명력 {r2:.2f} H 피크 {hp:.3f}"
                + (f" → 게이트 {'통과' if gate else '불통과'}" + (f", φ_lead {self.lead:.0f}°, 허용 k ≤ {self.k_cap:.2f}(k 1.0 |L| 고주파 {rec['L_hf_k1']} 저주파 {rec['L_lf_k1']})" if 'lead' in rec else "")
                   if "dpsi" not in rec else f" | 재식별 Δψ {rec['dpsi']:+.0f}°" + (" ★ ±30° 넘음 — on/off 짝 주 판정 제외" if abs(rec['dpsi']) > 30 else "")), flush=True)
        else:
            on = self.lead is not None and 0 < kk <= self.k_cap + 1e-9 and self.ck_prev is not False and not self.tripped
            psi_l = psi_with(f, L, f0, z) if on else None; drift = wrap(psi - self.ident["psi"]) if self.ident else None
            passed = bool(on and abs(psi_l) <= self.c.psi_l_max and drift is not None and abs(drift) <= self.c.drift_max)
            if passed:
                self.k_ok = kk
            if on:
                self.ck_prev = passed
            rec.update(on=on, psi_L=None if psi_l is None else round(psi_l, 1), L_mag=round(float(np.median(np.abs(L))), 3), drift=None if drift is None else round(drift, 1), passed=passed)
            log(f"[계획] k {kk} 처프 구간 {i}: " + (("k 꺼짐(감시 발동)" if self.tripped else "k 안 켬(게이트 불통과·허용 k 초과·직전 ck 불통과)") if not on else
                f"ψ_L {psi_l:+.0f}° |L| 중앙 {rec['L_mag']:.3f}, 플랜트 ψ {psi:+.0f}°(식별 대비 {drift:+.0f}°) → {'통과, k_ok = ' + str(kk) if passed else '불통과, k_ok ' + str(self.k_ok) + ' 유지'}"), flush=True)
        self.results.append(rec)
