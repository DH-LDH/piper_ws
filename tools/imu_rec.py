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
import time

import pyrealsense2 as rs

ACCEL_HZ = (200, 100, 63)  # D455 지원 값 중 높은 것부터
GYRO_HZ = (400, 200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sec", type=float, default=0.0, help="기록 시간[s], 0이면 종료 신호까지")
    a = ap.parse_args()
    if not rs.context().devices:
        raise SystemExit("★ RealSense 장치 없음 — 손목캠 USB 확인, 카메라 노드가 장치를 잡고 있으면 종료")
    lock = threading.Lock(); rows = []; n = {"a": 0, "g": 0}

    def cb(f):  # SDK 스레드 — 버퍼에만 쌓음
        m = f.as_motion_frame()
        if not m:
            return
        d = m.get_motion_data(); k = "a" if m.get_profile().stream_type() == rs.stream.accel else "g"
        with lock:
            rows.append((time.time(), m.get_timestamp(), k, d.x, d.y, d.z)); n[k] += 1

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
    print(f"IMU 기록 시작: 가속도 {ah} Hz, 자이로 {gh} Hz → {a.out}", flush=True)
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
        print(f"IMU 기록 종료: 가속도 {n['a']}개, 자이로 {n['g']}개", flush=True)


if __name__ == "__main__":
    sys.exit(main())
