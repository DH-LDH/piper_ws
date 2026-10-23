"""손목 D455 IMU 실시간 수신 (D2-0) — SDK 콜백 스레드가 최신 가속도·자이로와 짧은 버퍼를 유지, 제어 루프는 latest()로 읽기만.

젯슨 커널에 HID 센서 허브가 없어 pip 휠(V4L)로는 IMU가 안 보임 → ~/librealsense_rsusb/py(RSUSB 소스 빌드)를 먼저 찾음.
장치 타임스탬프(ms)를 그대로 보관하고, 빈틈(직전 표본과 간격 > 2.5배 중앙 주기)의 횟수·최대 길이를 센다.
PC 수신 시각은 몰려 와서 중앙 23 ms·95% 46 ms 늦다(10-24) → 표본 시각 = 장치 시각 + 오프셋, 오프셋은 최근 창의 (수신 − 장치) 최솟값.
사용: s = ImuStream(); s.start(); a, g, age = s.latest(); ... s.stop(); s.gap_stats()
"""
import collections
import os
import sys
import threading
import time

RSUSB = os.path.expanduser("~/librealsense_rsusb/py")
if os.path.isdir(RSUSB) and RSUSB not in sys.path:
    sys.path.insert(0, RSUSB)
import pyrealsense2 as rs  # noqa: E402

ACCEL_HZ = (63, 250)  # 10-23 실측: 합산 표본 수에 비례해 빠짐 — 250/400 Hz 빈틈 9–10%, 63/200 Hz 1–3%
GYRO_HZ = (200, 400)


class ImuStream:
    def __init__(self, buf_s=2.0, log=None):
        self.lock = threading.Lock()
        self.a = self.g = None              # (host_t, dev_ms, x, y, z)
        self.buf = {"a": collections.deque(maxlen=int(buf_s * 400)), "g": collections.deque(maxlen=int(buf_s * 800))}
        self.n = {"a": 0, "g": 0}
        self.last_dev = {"a": None, "g": None}
        self.gaps = {"a": [], "g": []}      # 빈틈 길이[ms]
        self.nominal = {}
        self.log = log                      # csv.writer면 모든 표본을 기록
        self.offs = collections.deque(maxlen=400)  # (수신 − 장치) 최근 2 s(자이로 200 Hz) — 최솟값이 최소 지연 기준
        self.off = None
        self.pipe = None

    def _cb(self, f):
        m = f.as_motion_frame()
        if not m:
            return
        k = "a" if m.get_profile().stream_type() == rs.stream.accel else "g"
        d = m.get_motion_data(); ht = time.time(); dt = m.get_timestamp()
        with self.lock:
            self.offs.append(ht - dt / 1000.0)
            if self.n[k] % 20 == 0 or self.off is None:  # 창 중앙값에서 1 s 안의 최솟값 — 장치 시각 도메인이 바뀐 첫 표본 같은 튀는 값 제외
                med = sorted(self.offs)[len(self.offs) // 2]
                self.off = min(x for x in self.offs if abs(x - med) < 1.0)
            row = (dt / 1000.0 + self.off, dt, d.x, d.y, d.z)  # 표본 시각(PC 시계로 환산한 장치 시각)
            prev = self.last_dev[k]
            if prev is not None and 0 < dt - prev < 1000 and dt - prev > 2.5 * self.nominal[k]:
                self.gaps[k].append(dt - prev)
            self.last_dev[k] = dt
            self.buf[k].append(row); self.n[k] += 1
            if k == "a":
                self.a = row
            else:
                self.g = row
            if self.log is not None:
                self.log.writerow((f"{ht:.6f}", f"{dt:.3f}", k, f"{d.x:.5f}", f"{d.y:.5f}", f"{d.z:.5f}"))  # 원본은 수신 시각 그대로

    def start(self):
        if not rs.context().devices:
            raise RuntimeError("RealSense 장치 없음 — 손목캠 USB 확인, 카메라 노드가 장치를 잡고 있으면 종료")
        for ah in ACCEL_HZ:  # 장치마다 지원 주기가 달라 되는 조합을 찾음
            for gh in GYRO_HZ:
                cfg = rs.config()
                cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, ah)
                cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, gh)
                self.nominal = {"a": 1000.0 / ah, "g": 1000.0 / gh}
                try:
                    self.pipe = rs.pipeline(); self.pipe.start(cfg, self._cb)
                    self.hz = (ah, gh)
                    return self.hz
                except RuntimeError:
                    self.pipe = None
        raise RuntimeError("IMU 스트림 시작 실패")

    def stop(self):
        if self.pipe is not None:
            self.pipe.stop(); self.pipe = None

    def latest(self):  # → (가속도 xyz, 자이로 xyz, 가장 오래된 쪽 나이[s], 표본 시각 기준) — 아직 없으면 None
        with self.lock:
            a, g = self.a, self.g
        if a is None or g is None:
            return None, None, float("inf")
        now = time.time()
        return a[2:], g[2:], now - min(a[0], g[0])

    def gap_stats(self):  # 스트림별 (표본 수, 빈틈 수, 빈틈 합 ms, 최대 ms)
        with self.lock:
            return {k: (self.n[k], len(v), sum(v), max(v) if v else 0.0) for k, v in self.gaps.items()}
