#!/usr/bin/env python3
"""손목 D455 IMU(가속도·자이로) 기록 — link6에 달린 EE 가속도. 종료는 Ctrl-C 또는 SIGTERM.

CSV: t(epoch s, 수신 시각), t_dev(장치 시각 ms), stream(a/g), x, y, z — 가속도 m/s², 자이로 rad/s (카메라 좌표계)
사용: python3 tools/imu_rec.py out.imu.csv [--sec 20]   (vib_probe/mit_probe --imu가 하위 프로세스로 띄움)
"""
import argparse
import csv
import signal
import sys
import threading
import os
import time

RSUSB = os.path.expanduser("~/librealsense_rsusb/py")  # 젯슨 커널에 HID 센서 허브가 없어 pip 휠(V4L)로는 IMU가 안 보임 — RSUSB 소스 빌드
if os.path.isdir(RSUSB):
    sys.path.insert(0, RSUSB)
import pyrealsense2 as rs

ACCEL_HZ = (250, 63)  # D455 지원 값 중 높은 것부터 — 요청값일 뿐, BMI085 장치는 250을 요청해도 실제 200 Hz(간격 4.96 ms, 11/17 확인)라 끝날 때 잰 값을 표시
GYRO_HZ = (400, 200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sec", type=float, default=0.0, help="기록 시간[s], 0이면 종료 신호까지")
    a = ap.parse_args()
    if not rs.context().devices:
        raise SystemExit("★ RealSense 장치 없음 — 손목캠 USB 확인, 카메라 노드가 장치를 잡고 있으면 종료")
    lock = threading.Lock(); rows = []; n = {"a": 0, "g": 0}; last = {"a": None, "g": None}; gaps = {"a": [], "g": []}  # gaps: 장치 시각 간격[ms], 스트림마다 앞 20000개

    def cb(f):  # SDK 스레드 — 버퍼에만 쌓음
        m = f.as_motion_frame()
        if not m:
            return
        d = m.get_motion_data(); k = "a" if m.get_profile().stream_type() == rs.stream.accel else "g"
        td = m.get_timestamp()
        with lock:
            rows.append((time.time(), td, k, d.x, d.y, d.z)); n[k] += 1
            if last[k] is not None and len(gaps[k]) < 20000:
                gaps[k].append(td - last[k])
            last[k] = td

    pipe = None
    for ah in ACCEL_HZ:  # 장치마다 지원 주기가 달라 되는 조합을 찾음
        for gh in GYRO_HZ:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, ah)
            cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, gh)
            try:
                pipe = rs.pipeline(); pipe.start(cfg, cb); break
            except RuntimeError:
                pipe = None
        if pipe:
            break
    if pipe is None:
        raise SystemExit("★ IMU 스트림 시작 실패")
    print(f"IMU 기록 시작: 가속도 {ah} Hz, 자이로 {gh} Hz(요청값) → {a.out}", flush=True)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    f = open(a.out, "w", newline=""); w = csv.writer(f)
    w.writerow(["t", "t_dev", "stream", "x", "y", "z"])
    t_end = time.time() + a.sec if a.sec > 0 else float("inf")
    try:
        while not stop.is_set() and time.time() < t_end:
            time.sleep(0.2)
            with lock:
                batch, rows[:] = rows[:], []
            w.writerows((f"{t:.6f}", f"{td:.3f}", k, f"{x:.5f}", f"{y:.5f}", f"{z:.5f}") for t, td, k, x, y, z in batch)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        with lock:
            w.writerows((f"{t:.6f}", f"{td:.3f}", k, f"{x:.5f}", f"{y:.5f}", f"{z:.5f}") for t, td, k, x, y, z in rows)
        f.close()
        def rate(k):  # 장치 시각 간격 중앙값으로 잰 실제 표본율 — 시작 직후 시각 기준이 바뀌는 표본(1 s 넘는 간격)은 뺌
            g = sorted(x for x in gaps[k] if 0 < x < 1000)
            return f"{g[len(g) // 2]:.2f} ms = {1000 / g[len(g) // 2]:.0f} Hz" if g else "간격 없음"
        print(f"IMU 기록 종료: 가속도 {n['a']}개(실제 간격 중앙 {rate('a')}), 자이로 {n['g']}개({rate('g')})", flush=True)
        os._exit(0)  # librealsense 종료 정리가 가끔 멈춤(10-23) — 파일은 이미 닫음


if __name__ == "__main__":
    sys.exit(main())
