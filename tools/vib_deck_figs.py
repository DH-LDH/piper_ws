#!/usr/bin/env python3
"""11/05 덱 그림(8절 10/29 덱 요청) — docs/img/p_vib_e4_*.png, 흰 배경·한글·1600 px 이상.

1 state_scatter: 복구별 상태 산점도(정지 기울기 × 식별 H 피크, 색 = 공진 위상) — 값은 LOG 표(E3 5·7절, E4 14-1·20절)
2 d3_time_spec: a′ 0.25 Hz ①②④-off 대표 회 acc z 5–10 Hz 시간 응답 + 2–20 Hz 스펙트럼
3 td_sens: 실시간 Td 분포(10-29 (가) 5회) + 143718·145621 감도 비 |S_k|/|S_0|
4 limit_cycle: 172154 τ_vib 시간 응답 + 노치(29.6 Hz, Q 1.5) 이득
5 load_limit: 부하 질량 vs a′ 중 J2 명령 하위 1%·클립(21-8·21-10)
6 seg_timeline: 감쇠 가능 구간 — (가) 실시간 22-1, (나) 오프라인 22-5
7 xy_axes: xy 축별 ① 대비 비(①②④-off 각 3회 평균, 21-11)
사용: source install/setup.bash; python3 tools/vib_deck_figs.py [1 … 7]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src", "common_pkg")); sys.path.insert(0, HERE)
IMG = os.path.join(ROOT, "docs", "img"); MIT = os.path.join(ROOT, "logs", "mit"); VIB = os.path.join(ROOT, "logs", "vib")

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"  # 범주 1–3번(파랑·주황·청록)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "Noto Sans CJK JP", "font.size": 13, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "axes.unicode_minus": False, "lines.linewidth": 2.0})


def save(fig, name):
    p = os.path.join(IMG, f"p_vib_e4_{name}.png"); fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); print(p)


def band(t, x, lo, hi, fs=None):  # 균일 재표본 → FFT 대역
    fs = fs or 1 / np.median(np.diff(t)); tu = np.arange(t[0], t[-1], 1 / fs); xu = np.interp(tu, t, x)
    X = np.fft.rfft(xu - xu.mean()); f = np.fft.rfftfreq(len(xu), 1 / fs); X[(f < lo) | (f > hi)] = 0
    return tu, np.fft.irfft(X, len(xu))


def fig_state():  # (출처, 회, 기울기°, H 피크, 공진 위상° 또는 nan) — LOG 값 그대로
    E3 = [("33.6", 33.6, 0.46, -72), ("32.1", 32.1, 0.16, np.nan), ("31.9", 31.9, 0.05, np.nan), ("32.9", 32.9, 0.36, -130),
          ("32.3", 32.3, 0.14, np.nan), ("34.3", 34.3, 0.79, -127), ("F1", 35.95, 0.64, -59), ("F2", 35.02, 0.28, -16), ("F3", 35.81, 0.42, -27)]
    NA = [("142200", 41.24, 0.405, -92), ("142840", 41.09, 0.379, -57), ("143718", 36.14, 0.507, -114), ("145621", 35.62, 0.535, -117),
          ("150714", 40.07, 0.676, -24), ("151607", 35.68, 0.556, -61), ("104801", 40.43, 0.28, -12)]
    GA = [("163243", 32.75, 0.266, -88), ("101912", 34.18, 0.599, -34), ("102416", 32.77, 0.397, -143), ("103052", 38.63, 0.738, -144),
          ("104447", 40.24, 0.494, -83), ("105310", 40.34, 0.778, -144)]
    fig, ax = plt.subplots(figsize=(11, 6.6))
    for lo, hi, lab in ((31.6, 35.3, "약 33° 무리"), (35.3, 37.2, "약 36° 무리"), (39.5, 42.0, "약 40° 무리")):
        ax.axvspan(lo, hi, color="#f0efec", zorder=0); ax.text((lo + hi) / 2, 0.86, lab, ha="center", color=INK2, fontsize=12)
    ax.axhline(0.5, color=INK2, lw=1, ls="--", zorder=1); ax.text(42.0, 0.505, "게이트 H ≥ 0.5", ha="right", va="bottom", color=INK2, fontsize=11)
    cmap = plt.get_cmap("twilight_shifted"); norm = matplotlib.colors.Normalize(-180, 180); sc = None
    for rows, mk, lab in ((E3, "o", "E3 처프 식별(J2, 10-26)"), (NA, "s", "E4 (나) φ_lead 고정 AB(10-27)"), (GA, "^", "E4 (가) 시행 안 식별(10-27·29)")):
        x = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows]); ph = np.array([r[3] for r in rows], float); ok = ~np.isnan(ph)
        sc = ax.scatter(x[ok], y[ok], c=ph[ok], cmap=cmap, norm=norm, marker=mk, s=150, edgecolors="white", linewidths=2, zorder=3, label=lab)
        ax.scatter(x[~ok], y[~ok], facecolors="none", edgecolors=INK2, marker=mk, s=150, linewidths=1.5, zorder=3)
    for r in GA:
        ax.annotate(r[0], (r[1], r[2]), xytext=(7, -4), textcoords="offset points", fontsize=10, color=INK2)
    cb = fig.colorbar(sc, ax=ax, pad=0.02); cb.set_label("공진 위상 ∠H [°] (순환)"); cb.set_ticks([-180, -90, 0, 90, 180])
    ax.set_xlabel("정지 기울기 (IMU atan2(y, −z), 자세 A) [°]"); ax.set_ylabel("식별 H 피크 [rad/s per N·m]")
    ax.set_xlim(31.5, 42.1); ax.set_ylim(0, 0.92)
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False, fontsize=11)
    for h in getattr(leg, "legend_handles", None) or leg.legendHandles:
        h.set_facecolor(INK2); h.set_edgecolor("white")
    ax.set_title("복구마다 바뀌는 링크 상태 — 같은 자세에서 기울기·공진 세기·위상이 회마다 다름", loc="left", fontsize=15, color=INK)
    fig.text(0.01, -0.08, "빈 마커 = 응답이 약해 위상 미정. E3 = 변형 피크·위상, (나) = 앞 절반 H 피크·7–8.5 Hz ∠H 중앙, (가) = 식별 구간 H 피크·단일 모드 맞춤 ψ.",
             fontsize=10, color=INK2)
    save(fig, "state_scatter")


def fig_d3():
    import vib_axes_report as A
    runs = ((os.path.join(VIB, "20261027_105647_slow_j23_d3_pos.csv"), "① 위치 모드 (105647)", C1),
            (os.path.join(MIT, "20261027_105921_sine.csv"), "② 하이브리드 TDE (105921)", C2),
            (os.path.join(MIT, "20261029_095735_sine.csv"), "④-off ASMC16 (095735)", C3))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for p, lab, col in runs:
        d = pd.read_csv(p); ph = "mit" if "mit" in set(d.phase) else "slow"; m = d[d.phase == ph]
        t0, t1, ts = A.window(m); i = pd.read_csv(p.replace(".csv", ".imu.csv")); i["t"] = i.t - float(open(p.replace(".csv", ".t0")).read())
        s = i[(i.stream == "a") & (i.t >= t0 - 1) & (i.t <= t1 + 1)]; tu, y = band(s.t.values, s.z.values, 5, 10, 250)
        k = (tu >= t0) & (tu <= t0 + 8); a1.plot(tu[k] - t0, y[k], color=col, lw=1.4, label=lab)
        mm = m[(m.t >= t0) & (m.t <= t1)]; tq = mm.t.values  # 지표와 같게 MIT 순간 튐 ±0.5 s가 걸친 Welch 조각은 뺌
        qd = np.abs(np.diff(np.radians(mm[[f"q{j}" for j in range(1, 7)]].values), axis=0)) / np.maximum(np.diff(tq)[:, None], 1e-3)
        sp = tq[1:][(qd > 0.7).any(1)] if ph == "mit" else np.array([])
        w = (s.t >= t0) & (s.t <= t1); tt = s.t.values[w]; tu2 = np.arange(tt[0], tt[-1], 1 / 250); x = np.interp(tu2, tt, s.z.values[w])
        n = 500; P = 0; c = 0
        for j in range(0, len(x) - n, n // 2):
            if len(sp) and np.any((sp >= tu2[j] - 0.5) & (sp <= tu2[j + n - 1] + 0.5)):
                continue
            seg = x[j:j + n] - x[j:j + n].mean(); P = P + np.abs(np.fft.rfft(seg * np.hanning(n))) ** 2; c += 1
        f = np.fft.rfftfreq(n, 1 / 250); b = (f >= 2) & (f <= 20); a2.semilogy(f[b], np.sqrt(P[b] / c), color=col, label=f"{lab}, {c}조각")
    a1.set_xlabel("시간 (사인 시작 + 2 s부터) [s]"); a1.set_ylabel("acc z, 5–10 Hz [m/s²]"); a1.set_title("a′ 0.25 Hz 중 손목 진동 (처음 8 s, 큰 봉우리 = MIT 순간 튐 — 지표에선 ±0.5 s 제외)", loc="left", fontsize=13)
    a1.legend(loc="upper right", frameon=False, fontsize=11)
    a2.axvspan(5, 10, color="#f0efec", zorder=0); a2.set_xlabel("주파수 [Hz]"); a2.set_ylabel("acc z 진폭 스펙트럼 [상대]"); a2.set_title("2–20 Hz 스펙트럼 (튐 제외, 회색 = 지표 대역)", loc="left"); a2.legend(loc="lower left", frameon=False, fontsize=10)
    fig.suptitle("추종층 효과 — 같은 a′ 궤적에서 ②·④-off가 ① 대비 5–10 Hz 진동 약 −39%", x=0.01, ha="left", fontsize=15)
    fig.tight_layout(); save(fig, "d3_time_spec")


def fig_td():
    import vib_chirp_cl as CL
    from imu_frame import ImuToBase
    from piper_kin import PiperKin
    td = np.concatenate([pd.read_csv(os.path.join(MIT, f"20261029_{t}_plan.vib.csv")).td_ms.values for t in ("101912", "102416", "103052", "104447", "105310")])
    td = td[(td > 0) & (td < 200)]; med, p95 = np.median(td), np.percentile(td, 95)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6))
    a1.hist(td, bins=np.arange(0, 121, 2.5), color=C1, edgecolor="white", linewidth=1)
    for v, lab in ((med, f"중앙 {med:.1f} ms"), (p95, f"95% {p95:.1f} ms")):
        a1.axvline(v, color=INK, lw=1.5, ls="--"); a1.text(v + 1.5, a1.get_ylim()[1] * 0.92, lab, color=INK, fontsize=12)
    a1.set_xlabel("Td: 자이로 표본 → τ_vib 송신 [ms]"); a1.set_ylabel("주기 수"); a1.set_xlim(0, 120)
    sx = a1.secondary_xaxis("top", functions=(lambda ms: ms * 360 * 8 / 1000, lambda dg: dg * 1000 / (360 * 8))); sx.set_xlabel("8 Hz에서 위상 지연 [°]", color=INK2)
    a1.set_title(f"USB IMU 지연 분포 (10-29 (가) 5회, {len(td)} 주기)", loc="left", pad=34)
    kin = PiperKin(); tb = ImuToBase(kin)
    for tag, lab, col in (("143718", "143718 k 0.2 (36.1°) — 유효(∠H 유지)", C1), ("145621", "145621 k 0.5 (35.6°) — 무효, 측정 L대로 증폭", C2)):
        f = os.path.join(MIT, f"20261027_{tag}_hold.csv"); d = pd.read_csv(f); v = pd.read_csv(f.replace(".csv", ".vib.csv"))
        mit = d[d.phase == "mit"]; c = mit[mit.chirp != 0]; gap = np.where(np.diff(c.t.values) > 1.0)[0]
        segs = [(c.t.iloc[0], c.t.iloc[gap[0]]), (c.t.iloc[gap[0] + 1], c.t.iloc[-1])]
        g0, _ = CL.segment(f, d, v, *segs[0], 2, kin, tb); g1, _ = CL.segment(f, d, v, *segs[1], 2, kin, tb)
        r = (g1.S / g0.S).dropna(); r = r[(r.index >= 6) & (r.index <= 10)]
        a2.plot(r.index, r.values, color=col, marker="o", ms=8, markeredgecolor="white", markeredgewidth=2, label=lab)
    a2.axhline(1, color=INK2, lw=1); a2.axvspan(7, 8.5, color="#f0efec", zorder=0)
    a2.set_xlabel("주파수 [Hz]"); a2.set_ylabel("감도 비 |S_k| / |S_0| (1보다 작으면 감쇠)"); a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=11)
    a2.set_title("같은 시행 k 0 구간 대비 감도 (회색 = 7–8.5 Hz 공진)", loc="left", pad=34)
    fig.suptitle(f"지연과 위상 — Td 중앙 {med:.0f} ms(8 Hz {med * 2.88:.0f}°)·95% {p95:.0f} ms({p95 * 2.88:.0f}°), 위상이 어긋나면 감쇠가 증폭으로", x=0.01, ha="left", fontsize=15, y=1.04)
    fig.tight_layout(); save(fig, "td_sens")


def fig_lc():
    f = os.path.join(MIT, "20261026_172154_hold.csv"); d = pd.read_csv(f); v = pd.read_csv(f.replace(".csv", ".vib.csv"))
    tm0 = d[d.phase == "mit"].t.values[0]; t = v.t.values - tm0; k = (t >= 24) & (t <= 31)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.6, 1]})
    a1.plot(t[k], v.tau_vib.values[k], color=C2, lw=1.2)
    for x, lab, y in ((27.32, "첫 포화 27.32 s", 0.27), (27.85, "새 감시 ② 발동 27.85 s (재생)", 0.21), (30.67, "실제 중단 30.67 s", 0.27)):
        a1.axvline(x, color=INK, lw=1.2, ls="--"); a1.text(x - 0.05 if x > 30 else x + 0.05, y, lab, fontsize=11, color=INK, ha="right" if x > 30 else "left")
    a1.set_xlabel("MIT 시작 뒤 시간 [s]"); a1.set_ylabel("τ_vib [N·m (eff)]"); a1.set_title("리밋 사이클 172154 (옛 예측기, k 0.2) — 약 25 Hz 지속", loc="left")
    fr = np.linspace(1, 50, 2000); s = 2j * np.pi * fr; wn = 2 * np.pi * 29.6; H = (s * s + wn * wn) / (s * s + (wn / 1.5) * s + wn * wn)
    a2.plot(fr, 20 * np.log10(np.abs(H)), color=C1)
    h8 = (lambda s8: (s8 * s8 + wn * wn) / (s8 * s8 + (wn / 1.5) * s8 + wn * wn))(2j * np.pi * 8)
    a2.scatter([8], [20 * np.log10(abs(h8))], s=80, color=C1, edgecolors="white", linewidths=2, zorder=3)
    a2.annotate(f"8 Hz: {20 * np.log10(abs(h8)):.1f} dB, {np.degrees(np.angle(h8)):+.0f}°", (8, 20 * np.log10(abs(h8))), xytext=(10, -22), textcoords="offset points", fontsize=11)
    a2.axvspan(25, 35, color="#f0efec", zorder=0); a2.set_ylim(-30, 3)
    a2.set_xlabel("주파수 [Hz]"); a2.set_ylabel("노치 이득 [dB]"); a2.set_title("대책: 노치 29.6 Hz, Q 1.5 (회색 = 국소 모드 25–35 Hz)", loc="left")
    fig.suptitle("MIT 순간 튐 → 약 27 Hz 국소 모드 → 옛 예측기 고주파 이득 → 리밋 사이클, 노치·감시로 차단", x=0.01, ha="left", fontsize=15)
    fig.tight_layout(); save(fig, "limit_cycle")


def fig_load():  # 부하 질량 vs a′ 중 J2 명령 토크(하위 1%·클립) — 21-8·21-10
    runs = {0.0: ["20261027_105921", "20261027_110326", "20261027_110726"], 0.2: ["20261029_190551", "20261029_190931", "20261029_191206"],
            0.3: ["20261029_200511", "20261029_200846", "20261029_201255"], 0.4: ["20261029_194245"]}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.6))
    for kg, tags in runs.items():
        p1, cl = [], []
        for t in tags:
            d = pd.read_csv(os.path.join(MIT, f"{t}_sine.csv")); x = d[d.phase == "mit"].tff2.values
            p1.append(np.percentile(x, 1)); cl.append(100 * np.mean(np.abs(x) >= 3.9))
        col = C2 if kg == 0.2 else C1
        a1.scatter([kg] * len(p1), p1, s=110, color=col, edgecolors="white", linewidths=2, zorder=3)
        a2.scatter([kg] * len(cl), cl, s=110, color=col, edgecolors="white", linewidths=2, zorder=3)
        a1.annotate(f"{np.mean(p1):.2f}", (kg, np.mean(p1)), xytext=(10, 0), textcoords="offset points", fontsize=11, color=INK2, va="center")
        a2.annotate(f"{np.mean(cl):.1f}%", (kg, np.mean(cl)), xytext=(10, 0), textcoords="offset points", fontsize=11, color=INK2, va="center")
    a1.axhline(-4.0, color=INK, lw=1.5, ls="--"); a1.text(-0.04, -4.0, "명령 상한 −4.0", va="bottom", ha="left", fontsize=11)
    a2.axhline(5, color=INK, lw=1.5, ls="--"); a2.text(-0.04, 5, "중단 기준 5%", va="bottom", ha="left", fontsize=11)
    for a in (a1, a2):
        a.set_xticks([0, 0.2, 0.3, 0.4]); a.set_xticklabels(["무부하", "0.2 kg\n(손가락 파지)", "0.3 kg\n(플랜지, 주 값)", "0.4 kg\n(중단)"]); a.set_xlim(-0.05, 0.45)
    a1.set_ylabel("J2 명령 토크 하위 1% [N·m (eff)]"); a1.set_title("a′ 0.25 Hz 중 J2 명령(TDE) — 유지 토크보다 1.0–1.3 큼", loc="left")
    a2.set_ylabel("명령 상한 도달 비율 [%]"); a2.set_title("J2 명령 클립", loc="left")
    fig.suptitle("부하 상한 약 0.3 kg(가반 20%) — 0.4 kg에서 J2 명령이 상한에 닿아 클립 13.6%", x=0.01, ha="left", fontsize=15)
    fig.tight_layout(); save(fig, "load_limit")


def fig_seg():  # 감쇠 가능 구간 타임라인 — (가) 실시간 판정 22-1, (나) 오프라인 22-5
    GA = [("163243", 32.8, "--x--", "     "), ("102416 ④-on", 32.8, "-----", "     "), ("101912", 34.2, "--G-G", "  *  "),
          ("103052", 38.6, "GG-GG", "*  * "), ("104447", 40.2, "-GGGG", " *** "), ("105310 ④-on", 40.3, "GGGGG", " * * ")]
    NA = [("151607", 35.6, "GG", "* "), ("143718", 36.1, "GG", "* "), ("145621", 36.0, "GG", "  "),
          ("150714", 40.2, "-G", "  "), ("142840", 40.6, "-G", "  "), ("142200", 42.1, "G-", "  ")]
    kon = {"103052": (1, 2), "105310 ④-on": (1, 2, 3)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [2.4, 1]})
    for ax, rows, cols in ((a1, GA, ["식별", "k 0.2", "k 0.5", "k 1.0", "재식별"]), (a2, NA, ["처프 A (k 0)", "처프 B (k > 0)"])):
        for r, (name, tl, g, s) in enumerate(rows):
            for c in range(len(cols)):
                star = s[c] == "*"; gate = g[c] == "G"
                fc = C1 if star else "#b7d3f6" if gate else "#f0efec"
                ax.add_patch(plt.Rectangle((c + 0.04, r + 0.08), 0.92, 0.84, color=fc, ec="white", lw=2))
                lab = "★" if star else "G" if gate else ""
                if ax is a1 and c in kon.get(name, ()):
                    lab += " k on"
                ax.text(c + 0.5, r + 0.5, lab, ha="center", va="center", fontsize=12, color="white" if star else INK)
        ax.set_xlim(0, len(cols)); ax.set_ylim(len(rows), 0); ax.grid(False)
        ax.set_xticks(np.arange(len(cols)) + 0.5); ax.set_xticklabels(cols)
        ax.set_yticks(np.arange(len(rows)) + 0.5); ax.set_yticklabels([f"{n}  ({t:.1f}°)" for n, t, _, _ in rows])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    a1.set_title("(가) 시행 안 식별 6회 — 18 s 구간, 실시간 판정: 8/24", loc="left")
    a2.set_title("(나) φ_lead 고정 6회 — 오프라인: 2/6", loc="left")
    fig.suptitle("감쇠 가능 구간 — 33° 무리 0/8, 38–40° 무리 7/12, 연속은 104447 한 번(54 s)", x=0.01, ha="left", fontsize=15)
    fig.text(0.01, -0.04, "G = 게이트 조건 충족(H ≥ 0.5·잔차 ≤ 35°·설명력 ≥ 0.7·절반 ψ 차 ≤ 30°), ★ = G이고 다음 구간과 |Δψ| ≤ 30°(감쇠 가능). 마지막 구간은 다음이 없어 ★ 판정 밖. "
             "(가)+(나)를 같은 오프라인 잣대로 다시 세면 11/30(37%).", fontsize=10, color=INK2)
    fig.tight_layout(); save(fig, "seg_timeline")


def fig_xy():  # xy 축별 분해 — ①·②·④-off 각 3회 평균, ① 대비 비(21-11)
    import vib_axes_report as A
    from imu_frame import ImuToBase
    from piper_kin import PiperKin
    tb = ImuToBase(PiperKin())
    g = {"①": ["vib/20261029_110132_slow_j12_sxy", "vib/20261029_204150_slow_j12_sxy", "vib/20261029_204228_slow_j12_sxy"],
         "②": ["mit/20261029_112234_sine", "mit/20261029_201957_sine", "mit/20261029_203040_sine"],
         "④-off": ["mit/20261029_112534_sine", "mit/20261029_202438_sine", "mit/20261029_203303_sine"]}
    K = [("gy_5_10", "피치\n(J2 평면)"), ("gz_5_10", "요\n(J1)"), ("gx_5_10", "롤"), ("a|_5_10", "|acc|"), ("g|_5_10", "|gyro|")]
    m = {c: pd.DataFrame([A.report(os.path.join(ROOT, "logs", f"{p}.csv"), tb) for p in ps])[[k for k, _ in K]].mean() for c, ps in g.items()}
    fig, ax = plt.subplots(figsize=(11, 5.6)); x = np.arange(len(K)); w = 0.36
    for i, (c, col) in enumerate((("②", C1), ("④-off", C2))):
        r = (m[c] / m["①"]).values
        ax.bar(x + (i - 0.5) * w, r, w - 0.04, color=col, label=f"{c} / ①", zorder=2)
        for xx, rr in zip(x + (i - 0.5) * w, r):
            ax.text(xx, rr + 0.02, f"{rr:.2f}", ha="center", fontsize=11, color=INK)
    ax.axhline(1, color=INK2, lw=1); ax.axhline(0.83, color=INK, lw=1.2, ls="--", label="감소 문턱 0.83")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in K]); ax.set_ylabel("① 위치 모드 대비 비 (5–10 Hz rms)"); ax.set_ylim(0, 1.35)
    ax.legend(loc="upper left", frameon=False, ncol=3)
    ax.set_title("xy(J1+J2) — 각 3회 평균: J2 평면(피치)도 거의 안 줄고, |acc|·|gyro| 차이 없음", loc="left")
    fig.tight_layout(); save(fig, "xy_axes")


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4", "5", "6", "7"]
    for w in which:
        {"1": fig_state, "2": fig_d3, "3": fig_td, "4": fig_lc, "5": fig_load, "6": fig_seg, "7": fig_xy}[w]()
