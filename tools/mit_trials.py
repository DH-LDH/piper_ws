#!/usr/bin/env python3
"""계층별 비교 시행 목록(단계 D-1) — 같은 등속 후 정지 왕복을 조건(제어기)마다, 블록마다 순서를 섞어 한 번에 한 시행씩 돌린다. 기본은 dry-run.

조건: pos   위치 모드(vib_probe --stage ramp --round, 내려앉지 않음)
      pd    MIT, 6축 펌웨어 PD + 유지 토크(①′)
      tde   MIT, 추종층 TDE(J2 하이브리드, M̄ 0.01)(②)
      asmc  MIT, TDE + ASMC16(ε 0.08)(④-off)
      asmc3 MIT, asmc를 J1+J2+J3에 동시에
      pos+zvd / pd+zvd / asmc3+zvd 등 "+zv"·"+zvd"를 붙이면 같은 조건에 입력 성형(7.0 Hz, ζ 0.10 고정, 9절 11/19 30번 (가) 묶음)
      MIT 추종층 조건은 모두 --tau-rate(기본 0.3 N·m), J2 토크 상한은 make --j2-tau-limit(목록의 j2_lim 열, 기본 4). MIT 시행은 끝나면 비상 정지로 내려앉으므로 run은 한 번에 한 시행만 하고 끝난다 → 복구 뒤 같은 명령을 다시 실행.
make: 조건 × 블록 수, 블록마다 순서를 섞어 목록 CSV로.  run: 다음 시행 하나를 실행하고 결과·기록 파일을 목록에 적는다. 정상 종료가 아니면 그 시행을 다음에 다시 한다(시도 수를 적음).
재측정: MIT 시행이 끝나면 튐이 안 겹친 정지 창 수를 방향별로 세고, 3개 미만이면 같은 블록 끝에 그 조건을 한 번 더 넣는다(9절 11/17 22:04 8번, retake_of 열에 원래 idx). 재측정의 재측정은 하지 않는다. --no-retake로 끔.
사용: python3 tools/mit_trials.py make docs/data/trials_d1b_v10.csv --speed 10 --conds pos pd tde asmc asmc3 --reps 5
      python3 tools/mit_trials.py run docs/data/trials_d1b_v10.csv --imu [--go]        (--fake: 가짜 팔)
"""
import argparse
import csv
import os
import random
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
POSE_A = [-0.3, 121.1, -73.2, 1.6, 52.0, 13.7]
COLS = ["idx", "block", "cond", "joint", "speed", "dist", "round_reps", "status", "tries", "time", "file", "result", "retake_of", "j2_lim"]
MIT = {"pd": [], "tde": ["--tdc", "--hybrid", "--mbar", "0.01", "--dist-mode", "torque"],
       "asmc": ["--tdc", "--hybrid", "--mbar", "0.01", "--dist-mode", "torque", "--asmc16", "--eps", "0.08"],
       "asmc3": ["--tdc", "--hybrid", "--mbar", "0.01", "--dist-mode", "torque", "--asmc16", "--eps", "0.08", "--tdc-joints", "1", "2", "3"]}


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write(path, rows):
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(rows)
    os.replace(path + ".tmp", path)


def make(a):
    if os.path.exists(a.plan) and not a.force:
        raise SystemExit(f"★ {a.plan} 이미 있음 — 덮어쓰려면 --force")
    rng = random.Random(a.seed); rows = []
    for b in range(a.reps):
        order = a.conds[:]; rng.shuffle(order)
        for c in order:
            rows.append(dict(idx=len(rows) + 1, block=b + 1, cond=c, joint=a.joint, speed=f"{a.speed:g}", dist=f"{a.dist:+g}", round_reps=a.round_reps, status="", tries=0, time="", file="", result="", retake_of="", j2_lim=f"{a.j2_tau_limit:g}"))
    os.makedirs(os.path.dirname(os.path.abspath(a.plan)), exist_ok=True); write(a.plan, rows)
    print(f"목록 {a.plan}: 조건 {len(a.conds)}개 × {a.reps}블록 = {len(rows)}시행 (seed {a.seed}), J2 토크 상한 {a.j2_tau_limit:g} N·m")
    for b in range(a.reps):
        print(f"  블록 {b + 1}: " + " → ".join(r["cond"] for r in rows if r["block"] == b + 1))


def valid_windows(path):  # → {"hold": n, "bhold": n} — vib_stop_report 시행 집계의 n_win(튐 제외 구간과 안 겹친 정지 창 수)
    tmp = path.replace(".csv", ".nwin.tmp")
    subprocess.run(["nice", "-n", "15", sys.executable, os.path.join(TOOLS, "vib_stop_report.py"), path, "--all", "--trial-out", tmp], capture_output=True, text=True)
    try:
        with open(tmp) as f:
            out = {x["stop"]: int(x["n_win"]) for x in csv.DictReader(f)}
        os.remove(tmp)
    except (OSError, KeyError, ValueError):
        return {}
    return out


def command(r, a):
    base, _, shape = r["cond"].partition("+"); sh = ["--shape", shape] if shape else []
    name = os.path.splitext(os.path.basename(a.plan))[0].replace("trials_", ""); note = f"{name}_{int(r['idx']):03d}_{r['cond'].replace('+', '_')}"
    ex = ["--expect", *[f"{x:g}" for x in a.pose], "--expect-tol", f"{a.tol:g}"]
    if base == "pos":
        cmd = [sys.executable, os.path.join(TOOLS, "vib_probe.py"), "--stage", "ramp", "--joint", r["joint"], "--speed", r["speed"], "--dist=" + r["dist"], "--round", "--reps", r["round_reps"], "--note", note] + ex + sh
    else:
        cmd = [sys.executable, os.path.join(TOOLS, "mit_probe.py"), "--stage", "ramp", "--joint", r["joint"], "--speed", r["speed"], "--dist=" + r["dist"], "--reps", r["round_reps"], "--hold", "2",
               "--speed-err", "--dev-limit", "6", "--speed-consec", "10", "--note", note] + ex + sh + MIT[base] + (["--tau-rate", f"{a.tau_rate:g}", "--j2-tau-limit", r.get("j2_lim") or "4"] if MIT[base] else [])
    return cmd + [f for f, on in (("--imu", a.imu), ("--fake", a.fake), ("--go", a.go)) if on]


def run(a):
    rows = read(a.plan); todo = [r for r in rows if r["status"] != "done"]
    if not todo:
        print("끝: 남은 시행 없음"); return 0
    r = todo[0]; cmd = command(r, a); mit = not r["cond"].startswith("pos")
    print(f"── [{r['idx']}/{len(rows)}] 블록 {r['block']} 조건 {r['cond']} ({'MIT — 끝나면 비상 정지로 내려앉음' if mit else '위치 모드 — 내려앉지 않음'})" + ("" if a.go else " [dry-run]"), flush=True)
    for attempt in range(4):  # 시작 전 정지 확인이 J5 서보 움찔로 걸리면 팔은 그대로 서 있음 → 3번까지 다시 시도
        out = subprocess.run(cmd, capture_output=True, text=True); txt = out.stdout + out.stderr
        res = next((ln[4:] for ln in txt.splitlines() if ln.startswith("결과: ")), txt.strip().splitlines()[-1] if txt.strip() else "출력 없음")
        if not (a.go and "위치 모드에서 정지 상태가 아님" in res and attempt < 3):
            break
        print(f"   정지 확인에 걸림 — {attempt + 1}번째 다시 시도", flush=True); time.sleep(2.0)
    path = next((ln[4:] for ln in txt.splitlines() if ln.startswith("기록: ")), "")
    keep = ("등속 후 정지", "성형", "변화율 제한", "루프 멈춤", "링크 정지 기울기", "결과", "★", "추종층", "하이브리드", "팔은")
    print("\n".join("   " + ln for ln in txt.splitlines() if ln.startswith(keep) or "[dry-run] J" in ln))
    if not a.go:
        print(f"끝(dry-run): 남은 시행 {len(todo)}"); return 0
    ok = res.startswith("정상 종료"); r["tries"] = int(r["tries"] or 0) + 1
    r.update(status="done" if ok else "", time=time.strftime("%m-%d %H:%M:%S"), file=os.path.basename(path) if ok else r["file"], result=res if ok else f"[시도 {r['tries']} 실패] {res}")
    if ok and mit and a.retake:
        nw = valid_windows(path)
        if nw:
            r["result"] += f" | 유효 창 hold {nw.get('hold', 0)} / bhold {nw.get('bhold', 0)}"; print(f"   유효 창(튐 안 겹침): 자세 A 도착 {nw.get('hold', 0)} / {r['round_reps']}, 반대쪽 {nw.get('bhold', 0)} / {r['round_reps']}")
            if min(nw.get("hold", 0), nw.get("bhold", 0)) < 3 and not r.get("retake_of"):
                at = max(i for i, x in enumerate(rows) if x["block"] == r["block"]) + 1
                rows.insert(at, dict(r, idx=max(int(x["idx"]) for x in rows) + 1, status="", tries=0, time="", file="", result="", retake_of=r["idx"]))
                print(f"   ★ 유효 창 3개 미만 — 블록 {r['block']} 끝에 {r['cond']} 재측정 1회 추가")
        else:
            print("   ※ 유효 창 수를 세지 못함(분석 출력 확인)")
    write(a.plan, rows); left = sum(x["status"] != "done" for x in rows)
    nxt = next((x for x in rows if x["status"] != "done"), None)
    print(("   기록 " + os.path.basename(path) if ok else "★ 정상 종료가 아님 — 이 시행을 다음에 다시 한다(출력을 확인)") + f" | 남은 시행 {left}"
          + (f" | 다음: 블록 {nxt['block']} {nxt['cond']}" if nxt else "") + (" | 복구 뒤 같은 명령을 다시 실행" if mit or not ok else " | 바로 같은 명령을 다시 실행"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make"); m.add_argument("plan")
    m.add_argument("--joint", type=int, default=2); m.add_argument("--speed", type=float, default=10.0); m.add_argument("--dist", type=float, default=10.0)
    m.add_argument("--conds", nargs="+", choices=[b + s for b in ("pos", "pd", "tde", "asmc", "asmc3") for s in ("", "+zv", "+zvd")], default=["pos", "pd", "tde", "asmc", "asmc3"])
    m.add_argument("--reps", type=int, default=5, help="블록 수(블록마다 조건이 한 번씩)"); m.add_argument("--round-reps", type=int, default=6)
    m.add_argument("--j2-tau-limit", type=float, default=4.0, help="MIT 추종층 조건의 J2 명령 토크 상한[N·m] — 목록에 적어 두고 시행마다 그대로 씀")
    m.add_argument("--seed", type=int, default=1); m.add_argument("--force", action="store_true")
    r = sp.add_parser("run"); r.add_argument("plan")
    r.add_argument("--go", action="store_true"); r.add_argument("--fake", action="store_true"); r.add_argument("--imu", action="store_true")
    r.add_argument("--tau-rate", type=float, default=0.3, help="MIT 추종층 조건의 토크 변화율 제한[N·m/주기]")
    r.add_argument("--no-retake", dest="retake", action="store_false", help="유효 창 3개 미만 시행의 재측정 추가를 끔")
    r.add_argument("--pose", type=float, nargs=6, default=POSE_A); r.add_argument("--tol", type=float, default=1.0)
    a = ap.parse_args()
    sys.exit(make(a) if a.cmd == "make" else run(a))


if __name__ == "__main__":
    main()
