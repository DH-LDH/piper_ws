#!/usr/bin/env python3
# =============================================================================
# eih_cam_calib.py — 손목캠 외부파라미터(link6→eih_cam) 캘리브레이션.
#
# 원리: 마커를 고정해두고 팔 자세만 바꾸면, 외부파라미터가 맞을 때 마커의
# body_link 좌표는 자세와 무관하게 같은 값이어야 한다. 그 편차를 최소화하는
# (pitch, roll, xyz, 마커스케일)을 푼다 — 마커의 실제 위치를 몰라도 된다.
#
# 사용법:
#   1) ros2 launch piper_hw_pkg piper_real.launch.py really_enable:=true arm:=false
#   2) python3 eih_cam_calib.py            (다른 터미널)
#   3) set_arm_joint.py로 자세를 바꿔가며(손목 j4/j5/j6를 크게 돌릴 것)
#      마커가 화면에 보이는 상태에서 Enter로 캡처 — 4자세 이상, 6~8자세 권장
#   4) 'q' + Enter로 종료하면 결과가 나온다
# =============================================================================
import sys
import threading

import cv2
import numpy as np
from scipy.optimize import least_squares

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros

from step22_common import _quat_to_R

_LATCH = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

MARKER_ID_DEFAULT = 0
MARKER_SIZE_DEFAULT = 0.034   # [m] 현재 설정값 — 스케일 추정이 이걸 얼마나 고쳐야 하는지 알려준다


def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _R_l6_cam(pitch_rad, roll_rad):
    """piper_eih_camera_node._publish_cam_tcp_tf()와 같은 규약: R = Ry(pitch) @ Rz(roll)."""
    return _Ry(pitch_rad) @ _Rz(roll_rad)


class EihCalib(Node):
    def __init__(self, marker_id, marker_size):
        super().__init__("eih_cam_calib")
        self.marker_id = marker_id
        self.marker_size = marker_size
        h = marker_size / 2.0
        self.obj_pts = np.array([[-h, h, 0.], [h, h, 0.], [h, -h, 0.], [-h, -h, 0.]], np.float32)
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "ArucoDetector"):
            pr = cv2.aruco.DetectorParameters()
            pr.cornerRefinementMethod = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1)
            self.det = cv2.aruco.ArucoDetector(d, pr)
        else:
            self.det = (d, cv2.aruco.DetectorParameters_create())
        self.K = None
        self.dist = np.zeros(5)
        self.latest = None   # (p_cam, rep) — 최신 프레임의 검출 결과
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(CameraInfo, "/vision/eih_camera_info", self._on_info, _LATCH)
        self.create_subscription(Image, "/vision/eih_image", self._on_image, 5)

    def _on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, float).reshape(3, 3)

    def _on_image(self, msg: Image):
        if self.K is None:
            return
        rgba = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
        if hasattr(cv2.aruco, "ArucoDetector"):
            cs, ids, _ = self.det.detectMarkers(gray)
        else:
            cs, ids, _ = cv2.aruco.detectMarkers(gray, self.det[0], parameters=self.det[1])
        ids = ids.flatten().tolist() if ids is not None else []
        if self.marker_id not in ids:
            self.latest = None
            return
        corners = cs[ids.index(self.marker_id)].reshape(4, 2)
        ok, rvecs, tvecs, rep = cv2.solvePnPGeneric(
            self.obj_pts, corners, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok or len(rvecs) == 0:
            self.latest = None
            return
        reps = np.asarray(rep, float).ravel() if len(rep) else np.zeros(len(rvecs))
        best, best_rep, best_tz = None, 0.0, -np.inf
        for kk, (rv, tv) in enumerate(zip(rvecs, tvecs)):
            R, _ = cv2.Rodrigues(rv)
            if R[2, 2] > best_tz:
                best_tz = R[2, 2]
                best = np.asarray(tv, float).ravel()
                best_rep = float(reps[kk]) if kk < len(reps) else 0.0
        self.latest = (best, best_rep)

    def capture(self):
        """현재 프레임의 (카메라좌표 마커위치, body_link→link6 변환)을 한 쌍 잡는다."""
        if self.latest is None:
            return None, "마커 미검출"
        p_cam, rep = self.latest
        try:
            tf = self.tf_buffer.lookup_transform("body_link", "link6", Time())
        except Exception as e:
            return None, f"TF body_link→link6 없음 ({e})"
        q, t = tf.transform.rotation, tf.transform.translation
        R_b_l6 = _quat_to_R((q.w, q.x, q.y, q.z))
        t_b_l6 = np.array([t.x, t.y, t.z], float)
        return (p_cam, R_b_l6, t_b_l6, rep), None


def _body_points(samples, pitch, roll, off, scale):
    R_lc = _R_l6_cam(pitch, roll)
    return np.array([R_b_l6 @ (R_lc @ (p_cam * scale) + off) + t_b_l6
                     for (p_cam, R_b_l6, t_b_l6, _) in samples])


# 파라미터별 탐색 범위 — scale을 열어두면 s→0인 퇴화해(모든 점이 손목 회전중심에
# 모여 편차가 0이 되는 가짜 해)로 빠진다.
_LO = [-np.pi, -np.pi, -0.30, -0.30, -0.30, 0.5]
_HI = [ np.pi,  np.pi,  0.30,  0.30,  0.30, 2.0]


def _fit(samples, x0, free, label):
    """free에 든 파라미터만 풀고 나머지는 x0 고정. 반환: (해, 편차 RMS[mm])."""
    idx = {"pitch": 0, "roll": 1, "x": 2, "y": 3, "z": 4, "scale": 5}
    fi = [idx[k] for k in free]

    def unpack(v):
        x = np.array(x0, float)
        x[fi] = v
        return x

    def resid(v):
        x = unpack(v)
        pts = _body_points(samples, x[0], x[1], x[2:5], x[5])
        return (pts - pts.mean(0)).ravel()   # 자세마다 달라지는 정도 = 오차

    if fi:
        x = unpack(least_squares(resid, np.array(x0, float)[fi], method="trf",
                                 bounds=([_LO[i] for i in fi], [_HI[i] for i in fi])).x)
    else:
        x = np.array(x0, float)   # 자유 파라미터 없음 = 현재 설정 그대로 평가만
    pts = _body_points(samples, x[0], x[1], x[2:5], x[5])
    rms = float(np.sqrt((np.linalg.norm(pts - pts.mean(0), axis=1) ** 2).mean()) * 1000)
    mk = pts.mean(0)
    print(f"  {label:26s} 편차 RMS ={rms:6.1f}mm  "
          f"pitch={np.degrees(x[0]):+6.2f}° roll={np.degrees(x[1]):+7.2f}° "
          f"xyz=({x[2]:+.4f},{x[3]:+.4f},{x[4]:+.4f})m s={x[5]:.3f}  "
          f"마커=({mk[0]:+.3f},{mk[1]:+.3f},{mk[2]:+.3f})m")
    return x, rms


SAMPLE_FILE = "eih_calib_samples.npz"


def _save(samples):
    np.savez(SAMPLE_FILE,
             p_cam=np.array([s[0] for s in samples]),
             R=np.array([s[1] for s in samples]),
             t=np.array([s[2] for s in samples]),
             rep=np.array([s[3] for s in samples]))
    print(f"\n샘플 {len(samples)}개를 {SAMPLE_FILE}에 저장했습니다 "
          f"(재분석: python3 {sys.argv[0]} --load)")


def _load():
    d = np.load(SAMPLE_FILE)
    return list(zip(d["p_cam"], d["R"], d["t"], d["rep"]))


def _report(samples, marker_size):
    if len(samples) < 4:
        print(f"\n자세가 {len(samples)}개뿐이라 풀 수 없습니다 (최소 4개).")
        return
    # 거리 다양성이 없으면 scale과 병진이 서로 구분되지 않는다 — 미리 경고.
    z = np.array([float(np.linalg.norm(s[0])) for s in samples])
    print(f"\n{len(samples)}자세, 마커거리 {z.min()*1000:.0f}~{z.max()*1000:.0f}mm")
    if (z.max() - z.min()) < 0.15:
        print("  ★ 거리 변화가 150mm 미만 — scale(마커 크기)은 이 데이터로 못 가립니다.")
    print("'편차'는 같은 마커가 자세마다 다르게 보이는 정도(0이 정답):\n")

    x0 = [np.radians(40.0), np.radians(-90.0), -0.085, -0.010, 0.026, 1.0]
    _fit(samples, x0, [], "현재 설정 그대로")
    _fit(samples, x0, ["pitch", "roll"], "회전만")
    _fit(samples, x0, ["pitch", "roll", "x", "y", "z"], "회전+병진")
    xs, _ = _fit(samples, x0, ["pitch", "roll", "x", "y", "z", "scale"], "회전+병진+마커스케일")
    if abs(xs[5] - 1.0) > 0.25:
        print("  ★ scale이 1에서 크게 벗어남 — 거리 다양성 부족으로 인한 퇴화해일 수 있으니")
        print("    거리를 크게 바꾼 자세를 몇 개 추가해 다시 풀 것.")
    print("\n편차가 가장 많이 줄어드는 줄이 진짜 원인입니다. '마커=' 값이 해마다 크게 다르면")
    print("아직 자세 다양성이 부족한 것이니 자세를 더 모으세요.")
    print(f"적용: piper_eih_camera_node의 cam_tcp_offset_pitch_deg / _roll_deg / _x / _y / _z")


def main():
    if "--load" in sys.argv:
        samples = _load()
        print(f"{SAMPLE_FILE}에서 {len(samples)}자세 로드")
        _report(samples, MARKER_SIZE_DEFAULT)
        return

    marker_id = int(sys.argv[1]) if len(sys.argv) > 1 else MARKER_ID_DEFAULT
    marker_size = float(sys.argv[2]) if len(sys.argv) > 2 else MARKER_SIZE_DEFAULT

    rclpy.init()
    node = EihCalib(marker_id, marker_size)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    print(f"손목캠 외부파라미터 캘리브레이션 (마커 ID{marker_id}, 크기 {marker_size*1000:.1f}mm)")
    print("마커를 고정한 채 팔 자세를 바꿔가며 Enter로 캡처하세요 — 손목(j4/j5/j6)을 크게 돌릴수록 좋습니다.")
    print("4자세 이상 필요, 6~8자세 권장. 끝내려면 q + Enter.\n")

    samples = []
    try:
        while True:
            cmd = input(f"[{len(samples)}자세 캡처됨] Enter=캡처 / q=종료 > ").strip().lower()
            if cmd == "q":
                break
            s, err = node.capture()
            if s is None:
                print(f"  ★ 캡처 실패: {err}")
                continue
            samples.append(s)
            p_cam, _, _, rep = s
            print(f"  ✓ 캡처 {len(samples)}: cam=({p_cam[0]*1000:+.0f},{p_cam[1]*1000:+.0f},"
                  f"{p_cam[2]*1000:+.0f})mm rep={rep:.2f}px")
    except (KeyboardInterrupt, EOFError):
        pass

    if samples:
        _save(samples)
    _report(samples, marker_size)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
