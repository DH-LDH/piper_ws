#!/usr/bin/env python3
"""계층별 비교 시행 목록(단계 D-1) — 같은 등속 후 정지 왕복을 조건(제어기)마다, 블록마다 순서를 섞어 한 번에 한 시행씩 돌린다. 기본은 dry-run.

조건: pos   위치 모드(vib_probe --stage ramp --round, 내려앉지 않음)
      pd    MIT, 6축 펌웨어 PD + 유지 토크(①′)
      tde   MIT, 추종층 TDE(J2 하이브리드, M̄ 0.01)(②)
      asmc  MIT, TDE + ASMC16(ε 0.08)(④-off)
      asmc3 MIT, asmc를 J1+J2+J3에 동시에
      MIT 추종층 조건은 모두 --tau-rate(기본 0.3 N·m). MIT 시행은 끝나면 비상 정지로 내려앉으므로 run은 한 번에 한 시행만 하고 끝난다 → 복구 뒤 같은 명령을 다시 실행.
make: 조건 × 블록 수, 블록마다 순서를 섞어 목록 CSV로.  run: 다음 시행 하나를 실행하고 결과·기록 파일을 목록에 적는다. 정상 종료가 아니면 그 시행을 다음에 다시 한다(시도 수를 적음).
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
COLS = ["idx", "block", "cond", "joint", "speed", "dist", "round_reps", "status", "tries", "time", "file", "result"]
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
            rows.append(dict(idx=len(rows) + 1, block=b + 1, cond=c, joint=a.joint, speed=f"{a.speed:g}", dist=f"{a.dist:+g}", round_reps=a.round_reps, status="", tries=0, time="", file="", result=""))
    os.makedirs(os.path.dirname(os.path.abspath(a.plan)), exist_ok=True); write(a.plan, rows)
    print(f"목록 {a.plan}: 조건 {len(a.conds)}개 × {a.reps}블록 = {len(rows)}시행 (seed {a.seed})")
    for b in range(a.reps):
        print(f"  블록 {b + 1}: " + " → ".join(r["cond"] for r in rows if r["block"] == b + 1))


def command(r, a):
    name = os.path.splitext(os.path.basename(a.plan))[0].replace("trials_", ""); note = f"{name}_{int(r['idx']):03d}_{r['cond']}"
    ex = ["--expect", *[f"{x:g}" for x in a.pose], "--expect-tol", f"{a.tol:g}"]
    if r["cond"] == "pos":
        cmd = [sys.executable, os.path.join(TOOLS, "vib_probe.py"), "--stage", "ramp", "--joint", r["joint"], "--speed", r["speed"], "--dist=" + r["dist"], "--round", "--reps", r["round_reps"], "--note", note] + ex
    else:
        cmd = [sys.executable, os.path.join(TOOLS, "mit_probe.py"), "--stage", "ramp", "--joint", r["joint"], "--speed", r["speed"], "--dist=" + r["dist"], "--reps", r["round_reps"], "--hold", "2",
               "--speed-err", "--dev-limit", "6", "--speed-consec", "10", "--note", note] + ex + MIT[r["cond"]] + (["--tau-rate", f"{a.tau_rate:g}"] if MIT[r["cond"]] else [])
    return cmd + [f for f, on in (("--imu", a.imu), ("--fake", a.fake), ("--go", a.go)) if on]


def run(a):
    rows = read(a.plan); todo = [r for r in rows if r["status"] != "done"]
    if not todo:
        print("끝: 남은 시행 없음"); return 0
    r = todo[0]; cmd = command(r, a); mit = r["cond"] != "pos"
    print(f"── [{r['idx']}/{len(rows)}] 블록 {r['block']} 조건 {r['cond']} ({'MIT — 끝나면 비상 정지로 내려앉음' if mit else '위치 모드 — 내려앉지 않음'})" + ("" if a.go else " [dry-run]"), flush=True)
    for attempt in range(4):  # 시작 전 정지 확인이 J5 서보 움찔로 걸리면 팔은 그대로 서 있음 → 3번까지 다시 시도
        out = subprocess.run(cmd, capture_output=True, text=True); txt = out.stdout + out.stderr
        res = next((ln[4:] for ln in txt.splitlines() if ln.startswith("결과: ")), txt.strip().splitlines()[-1] if txt.strip() else "출력 없음")
        if not (a.go and "위치 모드에서 정지 상태가 아님" in res and attempt < 3):
            break
        print(f"   정지 확인에 걸림 — {attempt + 1}번째 다시 시도", flush=True); time.sleep(2.0)
    path = next((ln[4:] for ln in txt.splitlines() if ln.startswith("기록: ")), "")
    keep = ("등속 후 정지", "변화율 제한", "루프 멈춤", "링크 정지 기울기", "결과", "★", "추종층", "하이브리드", "팔은")
    print("\n".join("   " + ln for ln in txt.splitlines() if ln.startswith(keep) or "[dry-run] J" in ln))
    if not a.go:
        print(f"끝(dry-run): 남은 시행 {len(todo)}"); return 0
    ok = res.startswith("정상 종료"); r["tries"] = int(r["tries"] or 0) + 1
    r.update(status="done" if ok else "", time=time.strftime("%m-%d %H:%M:%S"), file=os.path.basename(path) if ok else r["file"], result=res if ok else f"[시도 {r['tries']} 실패] {res}")
    write(a.plan, rows); left = sum(x["status"] != "done" for x in rows)
    nxt = next((x for x in rows if x["status"] != "done"), None)
    print(("   기록 " + os.path.basename(path) if ok else "★ 정상 종료가 아님 — 이 시행을 다음에 다시 한다(출력을 확인)") + f" | 남은 시행 {left}"
          + (f" | 다음: 블록 {nxt['block']} {nxt['cond']}" if nxt else "") + (" | 복구 뒤 같은 명령을 다시 실행" if mit or not ok else " | 바로 같은 명령을 다시 실행"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make"); m.add_argument("plan")
    m.add_argument("--joint", type=int, default=2); m.add_argument("--speed", type=float, default=10.0); m.add_argument("--dist", type=float, default=10.0)
    m.add_argument("--conds", nargs="+", choices=["pos", "pd", "tde", "asmc", "asmc3"], default=["pos", "pd", "tde", "asmc", "asmc3"])
    m.add_argument("--reps", type=int, default=5, help="블록 수(블록마다 조건이 한 번씩)"); m.add_argument("--round-reps", type=int, default=6)
    m.add_argument("--seed", type=int, default=1); m.add_argument("--force", action="store_true")
    r = sp.add_parser("run"); r.add_argument("plan")
    r.add_argument("--go", action="store_true"); r.add_argument("--fake", action="store_true"); r.add_argument("--imu", action="store_true")
    r.add_argument("--tau-rate", type=float, default=0.3, help="MIT 추종층 조건의 토크 변화율 제한[N·m/주기]")
    r.add_argument("--pose", type=float, nargs=6, default=POSE_A); r.add_argument("--tol", type=float, default=1.0)
    a = ap.parse_args()
    sys.exit(make(a) if a.cmd == "make" else run(a))


if __name__ == "__main__":
    main()
