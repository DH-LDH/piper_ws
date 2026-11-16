#!/usr/bin/env python3
"""시행 목록 실행기(단계 C·D) — 조건을 섞은 순서 목록을 만들고 vib_probe --stage ramp를 차례로 돌린다. 기본은 dry-run.

make: 속도 × 거리 × 방향 조건을 반복 수만큼, 반복(블록)마다 순서를 섞어 목록 CSV로(한 블록 안에 조건이 한 번씩 — 온도·처짐 변화가 조건에 몰리지 않게).
run : 목록에서 아직 안 한 시행을 순서대로 실행하고 시행마다 결과·기록 파일·펌웨어·SDK·시행 직전 기울기를 목록에 적는다. 정상 종료가 아니면 거기서 멈춘다.
      다시 실행하면 남은 시행부터 이어서 한다. 중단된 시행은 status를 지우면 다시 한다.
사용: python3 tools/vib_trials.py make docs/data/trials_c1.csv --joint 2 --speeds 5 10 20 --dists 2.5 5 10 --reps 5
      python3 tools/vib_trials.py run docs/data/trials_c1.csv --imu [--go] [--n 18] [--pause]      (--fake: 가짜 팔)
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
POSE_A = [-0.3, 121.1, -73.2, 1.6, 52.0, 13.7]  # VIB_E1 자세 A — 멈추는 자세
COLS = ["idx", "block", "cond", "joint", "speed", "dist", "load_kg", "pose", "tol", "status", "time", "file", "fw", "sdk", "tilt_deg", "result"]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write(path, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, COLS); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)


def make(a):
    if os.path.exists(a.plan) and not a.force:
        raise SystemExit(f"★ {a.plan} 이미 있음 — 덮어쓰려면 --force")
    conds = [(v, d * s) for v in a.speeds for d in a.dists for s in ([1, -1] if a.dirs == "both" else [1] if a.dirs == "+" else [-1])]
    rng = random.Random(a.seed); rows = []
    for b in range(a.reps):
        order = conds[:]; rng.shuffle(order)
        for v, d in order:
            rows.append(dict(idx=len(rows) + 1, block=b + 1, cond=f"J{a.joint}_v{v:g}_d{d:+g}", joint=a.joint, speed=f"{v:g}", dist=f"{d:+g}", load_kg=f"{a.load:g}",
                             pose=" ".join(f"{x:g}" for x in a.pose), tol=f"{a.tol:g}", status="", time="", file="", fw="", sdk="", tilt_deg="", result=""))
    os.makedirs(os.path.dirname(os.path.abspath(a.plan)), exist_ok=True); write(a.plan, rows)
    print(f"목록 {a.plan}: 조건 {len(conds)}개 × {a.reps}회 = {len(rows)}시행 (seed {a.seed})")
    print("  처음 6개: " + ", ".join(r["cond"] for r in rows[:6]))


def pre_tilt(path):  # 시행 직전(움직이기 전 정지) 기울기
    try:
        sys.path.insert(0, os.path.join(TOOLS, "..", "src", "common_pkg"))
        from imu_frame import static_tilt
        with open(path) as f:
            t = [float(r["t"]) for r in csv.DictReader(f) if r["phase"] in ("settle", "lead")]
        return f"{static_tilt(path, t[0] + 0.3, t[-1]):.2f}"
    except Exception:
        return ""


def run(a):
    rows = read(a.plan); todo = [r for r in rows if r["status"] != "done"]
    stuck = [r["idx"] for r in rows if r["status"] not in ("", "done")]
    if stuck:
        raise SystemExit(f"★ 중단된 시행 {stuck} — 원인을 확인하고 목록의 status를 지운 뒤 다시 실행")
    name = os.path.splitext(os.path.basename(a.plan))[0].replace("trials_", "")
    print(f"{a.plan}: 전체 {len(rows)}, 남은 {len(todo)}, 이번에 {min(a.n, len(todo)) if a.n else len(todo)}시행" + ("" if a.go else " [dry-run]"))
    for k, r in enumerate(todo):
        if a.n and k >= a.n:
            break
        cmd = [sys.executable, os.path.join(TOOLS, "vib_probe.py"), "--stage", "ramp", "--joint", r["joint"], "--speed", r["speed"], "--dist=" + r["dist"],
               "--hold", str(a.hold), "--pre-hold", str(a.pre_hold), "--pre-vmax", str(a.pre_vmax), "--spd", str(a.spd), "--can", a.can,
               "--note", f"{name}_{int(r['idx']):03d}", "--expect", *r["pose"].split(), "--expect-tol", r["tol"]]
        cmd += [f for f, on in (("--imu", a.imu), ("--imu2", a.imu2), ("--fake", a.fake), ("--go", a.go)) if on]
        if a.pause and a.go:
            input(f"[{r['idx']}/{len(rows)}] {r['cond']} — Enter로 시작(Ctrl-C로 끝)")
        print(f"── [{r['idx']}/{len(rows)}] 블록 {r['block']} {r['cond']}", flush=True)
        out = subprocess.run(cmd, capture_output=True, text=True); txt = out.stdout + out.stderr
        res = next((ln[4:] for ln in txt.splitlines() if ln.startswith("결과: ")), txt.strip().splitlines()[-1] if txt.strip() else "출력 없음")
        path = next((ln[4:] for ln in txt.splitlines() if ln.startswith("기록: ")), "")
        if not a.go:
            print("\n".join("   " + ln for ln in txt.splitlines() if "등속 후 정지" in ln or "끝단 이동" in ln or ln.startswith(("결과", "★")))); continue
        meta = {}
        if path and os.path.exists(path.replace(".csv", ".meta.json")):
            meta = json.load(open(path.replace(".csv", ".meta.json")))
        ok = res.startswith("정상 종료")
        r.update(status="done" if ok else "abort", time=time.strftime("%m-%d %H:%M:%S"), file=os.path.basename(path), fw=meta.get("fw", ""), sdk=meta.get("sdk", ""),
                 tilt_deg=pre_tilt(path) if (a.imu and path) else "", result=res)
        write(a.plan, rows)
        print(f"   {res} | 기울기 {r['tilt_deg'] or '-'}° | {r['file']}", flush=True)
        if not ok:
            print("★ 정상 종료가 아님 — 여기서 멈춤. 팔 상태를 확인하고(복구 절차), 목록의 이 시행 status를 지운 뒤 다시 실행"); return 1
        time.sleep(a.gap)
    left = sum(r["status"] != "done" for r in rows)
    print(f"끝: 남은 시행 {left}" + ("" if a.go else " (dry-run이라 기록하지 않음)"))
    return 0


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make"); m.add_argument("plan")
    m.add_argument("--joint", type=int, default=2)
    m.add_argument("--speeds", type=float, nargs="+", default=[5, 10, 20], help="등속[deg/s]")
    m.add_argument("--dists", type=float, nargs="+", default=[2.5, 5, 10], help="이동 거리[deg]")
    m.add_argument("--dirs", choices=["both", "+", "-"], default="both", help="도착 방향(+면 관절각이 커지며 도착 — J2는 내림)")
    m.add_argument("--reps", type=int, default=5)
    m.add_argument("--load", type=float, default=0.0, help="부하[kg] — 기록용")
    m.add_argument("--pose", type=float, nargs=6, default=POSE_A, help="멈추는 자세[deg]")
    m.add_argument("--tol", type=float, default=1.0, help="멈추는 자세 허용 오차[deg]")
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--force", action="store_true")
    r = sp.add_parser("run"); r.add_argument("plan")
    r.add_argument("--go", action="store_true", help="실제로 명령 송신(없으면 시행마다 dry-run)")
    r.add_argument("--fake", action="store_true")
    r.add_argument("--imu", action="store_true"); r.add_argument("--imu2", action="store_true")
    r.add_argument("--n", type=int, default=0, help="이번에 할 시행 수(0 = 남은 것 전부)")
    r.add_argument("--pause", action="store_true", help="시행마다 Enter를 기다림")
    r.add_argument("--gap", type=float, default=2.0, help="시행 사이 쉬는 시간[s]")
    r.add_argument("--hold", type=float, default=3.0); r.add_argument("--pre-hold", type=float, default=3.0); r.add_argument("--pre-vmax", type=float, default=5.0)
    r.add_argument("--spd", type=int, default=10); r.add_argument("--can", default="can_piper")
    a = ap.parse_args()
    sys.exit(make(a) if a.cmd == "make" else run(a))


if __name__ == "__main__":
    main()
