"""루프 구간 타이밍 계측 — profile:=true일 때만 생성된다(꺼지면 노드 쪽은 None 분기 하나만 남음).

CSV 한 행 = 한 구간: node, section, t_ros_ns(루프 시작 ROS time), dur_ms, note
"""
import atexit
import csv
import os
import threading
import time
from collections import defaultdict

import numpy as np

LOG_ROOT = os.environ.get("PIPER_LOG_DIR", os.path.expanduser("~/IsaacSim_ranger_piper/logs"))
SUMMARY_SEC = 5.0


def declare_profile(node):  # profile/profile_run 파라미터 선언 → 켜져 있으면 ProfileLog, 아니면 None
    on = bool(node.declare_parameter("profile", False).value)
    run = str(node.declare_parameter("profile_run", "").value) or time.strftime("%Y%m%d_%H%M%S")
    return ProfileLog(node, os.path.join(LOG_ROOT, run)) if on else None


class ProfileLog:  # 노드당 CSV 1개 + 5초 요약. 캡처 스레드에서도 쓰므로 lock으로 보호
    def __init__(self, node, run_dir):
        os.makedirs(run_dir, exist_ok=True)
        self.node = node
        self.name = node.get_name()
        self.path = os.path.join(run_dir, f"{self.name}.csv")
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(["node", "section", "t_ros_ns", "dur_ms", "note"])
        self._acc = defaultdict(list)
        self._lock = threading.Lock()
        node.create_timer(SUMMARY_SEC, self._summary)
        atexit.register(self.close)  # Ctrl-C로 끝나도 마지막 5초분이 남게
        node.get_logger().warn(f"[profile] 타이밍 계측 ON → {self.path}")

    def now_ns(self):
        return self.node.get_clock().now().nanoseconds

    def loop(self, name):
        return Loop(self, name)

    def record(self, section, dur_ms, t_ros_ns, note=""):
        with self._lock:
            if self._f.closed:
                return
            self._w.writerow([self.name, section, t_ros_ns, f"{dur_ms:.3f}", note])
            self._acc[section].append(dur_ms)

    def _summary(self):  # 구간별 n/mean/p95/max 한 줄씩 출력하고 CSV flush
        with self._lock:
            acc, self._acc = self._acc, defaultdict(list)
            self._f.flush()
        if not acc:
            return
        lines = []
        for sec in sorted(acc):
            a = np.asarray(acc[sec])
            lines.append(f"  {sec:<24s} n={a.size:4d} mean={a.mean():7.2f} "
                         f"p95={np.percentile(a, 95):7.2f} max={a.max():7.2f} ms")
        self.node.get_logger().info(f"[profile {SUMMARY_SEC:.0f}s]\n" + "\n".join(lines))

    def close(self):
        with self._lock:
            if not self._f.closed:
                self._f.close()


class Loop:  # 한 루프의 begin → lap(구간)… → end. begin 사이 간격을 <name>.interval로 남긴다
    def __init__(self, log, name):
        self.log = log
        self.name = name
        self.note = ""
        self._prev = None
        self._t0 = self._tl = 0.0
        self._ros = 0

    def begin(self):
        now = time.perf_counter()
        self._ros = self.log.now_ns()
        if self._prev is not None:
            self.log.record(f"{self.name}.interval", (now - self._prev) * 1e3, self._ros)
        self._prev = self._t0 = self._tl = now
        self.note = ""

    def lap(self, section):
        now = time.perf_counter()
        self.log.record(f"{self.name}.{section}", (now - self._tl) * 1e3, self._ros)
        self._tl = now

    def value(self, section, ms, note=""):  # 소요시간이 아닌 측정값(지연 등)을 같은 형식으로 기록
        self.log.record(f"{self.name}.{section}", ms, self._ros, note)

    def end(self):
        self.log.record(f"{self.name}.total", (time.perf_counter() - self._t0) * 1e3,
                        self._ros, self.note)
