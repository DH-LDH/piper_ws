#!/usr/bin/env python3
"""3DM-GQ7(Parker LORD) IMU 기록 — MIP 직렬 직접 파싱, ROS 불필요. 종료는 Ctrl-C 또는 SIGTERM.

CSV(imu_rec.py와 같은 형식): t(epoch s, 수신 시각), t_dev(장치 시각 ms), stream(a/g), x, y, z — 가속도 m/s², 자이로 rad/s (센서 좌표계)
사용: python3 tools/gq7_rec.py out.gq7.csv [--sec 20] [--hz 500] [--port /dev/ttyACM0]
"""
import argparse
import csv
import glob
import signal
import struct
import sys
import time

import serial

G0 = 9.80665
SET_IMU = 0x80
F_ACC, F_GYRO, F_TS = 0x04, 0x05, 0xD3  # 스케일 가속도[g], 스케일 자이로[rad/s], GPS 시각(tow s)


def pkt(ds, *fields):
    pay = b"".join(bytes([len(f) + 1, f[0]]) + bytes(f[1:]) for f in fields)
    b = bytes([0x75, 0x65, ds, len(pay)]) + pay
    c1 = c2 = 0
    for x in b:
        c1 = (c1 + x) & 255; c2 = (c2 + c1) & 255
    return b + bytes([c1, c2])


class Mip:
    def __init__(self, port):
        self.s = serial.Serial(port, 115200, timeout=0.05)
        self.buf = b""; self.bad = 0
        time.sleep(0.3)  # 열자마자 쓰면 첫 명령이 유실됨

    def packets(self):
        self.buf += self.s.read(8192)
        while True:
            i = self.buf.find(b"\x75\x65")
            if i < 0:
                self.buf = self.buf[-1:]; return
            if len(self.buf) < i + 4:
                self.buf = self.buf[i:]; return
            n = self.buf[i + 3]
            if len(self.buf) < i + 6 + n:
                self.buf = self.buf[i:]; return
            p = self.buf[i:i + 6 + n]
            c1 = c2 = 0
            for x in p[:-2]:
                c1 = (c1 + x) & 255; c2 = (c2 + c1) & 255
            if (c1, c2) != (p[-2], p[-1]):  # 체크섬 불일치 — 동기 바이트만 버리고 다시 찾음
                self.bad += 1; self.buf = self.buf[i + 2:]; continue
            self.buf = self.buf[i + 6 + n:]
            yield p

    @staticmethod
    def fields(p):
        pay = p[4:-2]; i = 0
        while i + 2 <= len(pay) and pay[i] >= 2:
            yield pay[i + 1], pay[i + 2:i + pay[i]]
            i += pay[i]

    def cmd(self, ds, f, wait=0.5):
        self.s.write(pkt(ds, f)); out = []; ack = None; t0 = time.time()
        while time.time() - t0 < wait and ack is None:
            for p in self.packets():
                if p[2] != ds:
                    continue
                for d, b in self.fields(p):
                    if d == 0xF1 and b[0] == f[0]:
                        ack = b[1]
                    else:
                        out.append((d, b))
        if ack != 0:
            raise SystemExit(f"★ GQ7 명령 0x{ds:02X}/0x{f[0]:02X} 실패 (ack {ack})")
        return out


def find_port():
    c = glob.glob("/dev/serial/by-id/*Microstrain*") + glob.glob("/dev/serial/by-id/*MicroStrain*")
    if not c:
        raise SystemExit("★ GQ7 장치 없음 — USB 확인")
    return c[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--sec", type=float, default=0.0, help="기록 시간[s], 0이면 종료 신호까지")
    ap.add_argument("--hz", type=int, default=500, help="표본율[Hz] — 기준율(1000)의 약수")
    ap.add_argument("--port", default="")
    a = ap.parse_args()
    m = Mip(a.port or find_port())
    m.cmd(0x01, [0x02])  # 대기 상태로
    base = struct.unpack(">H", [b for d, b in m.cmd(0x0C, [0x0E, SET_IMU]) if d == 0x8E][0][1:3])[0]
    if base % a.hz:
        raise SystemExit(f"★ --hz는 {base}의 약수여야 함")
    dec = base // a.hz
    old = [b for d, b in m.cmd(0x0C, [0x0F, 0x02, SET_IMU]) if d == 0x8F][0]  # 끝날 때 되돌릴 기존 형식
    fmt = [0x0F, 0x01, SET_IMU, 3]
    for fd in (F_TS, F_ACC, F_GYRO):
        fmt += [fd, dec >> 8, dec & 255]
    m.cmd(0x0C, fmt)  # 저장하지 않음 — 전원을 껐다 켜면 원래 설정
    m.cmd(0x0C, [0x11, 0x01, SET_IMU, 0x01])
    m.cmd(0x01, [0x06])  # 스트림 재개
    print(f"GQ7 기록 시작: {a.hz} Hz → {a.out}", flush=True)
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    f = open(a.out, "w", newline=""); w = csv.writer(f)
    w.writerow(["t", "t_dev", "stream", "x", "y", "z"])
    n = 0; t_first = t_last = None
    t_end = time.time() + a.sec if a.sec > 0 else float("inf")
    try:
        while not stop[0] and time.time() < t_end:
            for p in m.packets():
                if p[2] != SET_IMU:
                    continue
                t = time.time(); td = None; acc = gyr = None
                for d, b in m.fields(p):
                    if d == F_TS:
                        td = struct.unpack(">d", b[:8])[0] * 1000.0
                    elif d == F_ACC:
                        acc = struct.unpack(">3f", b[:12])
                    elif d == F_GYRO:
                        gyr = struct.unpack(">3f", b[:12])
                if td is None or acc is None or gyr is None:
                    continue
                w.writerow((f"{t:.6f}", f"{td:.3f}", "a", *(f"{v * G0:.6f}" for v in acc)))
                w.writerow((f"{t:.6f}", f"{td:.3f}", "g", *(f"{v:.6f}" for v in gyr)))
                n += 1; t_last = td
                if t_first is None:
                    t_first = td
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
        try:  # 기존 스트림 형식으로 복구
            m.cmd(0x01, [0x02]); m.cmd(0x0C, [0x0F, 0x01] + list(old)); m.cmd(0x01, [0x06])
        except SystemExit as e:
            print(e, flush=True)
        dur = (t_last - t_first) / 1000.0 if n > 1 else 0.0
        rate = (n - 1) / dur if dur > 0 else 0.0
        print(f"GQ7 기록 종료: {n}개, 장치 시각 기준 {dur:.2f} s, {rate:.1f} Hz, 체크섬 오류 {m.bad}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
