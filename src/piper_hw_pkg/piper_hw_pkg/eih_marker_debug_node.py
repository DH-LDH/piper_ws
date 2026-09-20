import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import array
import math

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo

# 손목캠 마커 검출 눈으로 확인용 디버그 뷰어 — rqt_image_view로 /vision/eih_debug_image 구독해서 볼 것.
# 실제 파이프라인(vision_node.py)과 완전히 독립적으로 자체 검출해서 그리기만 한다 — 핵심 로직에 영향 없음.

_LATCH = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

# ID별 실제 한 변 길이[m] — step22_common.py 상수와 맞춰둔 것. 모르는 ID는 fallback 사용.
MARKER_SIZE_BY_ID = {0: 0.035, 1: 0.035, 2: 0.035, 3: 0.035,
                      4: 0.020, 5: 0.10, 6: 0.020, 7: 0.04}
FALLBACK_MARKER_SIZE = 0.020

_HAS_ARUCO_DETECTOR = hasattr(cv2.aruco, "ArucoDetector")


def _make_detector():
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if not _HAS_ARUCO_DETECTOR:
        return d, cv2.aruco.DetectorParameters_create()
    return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())


def _detect_markers(gray, det):
    if _HAS_ARUCO_DETECTOR:
        return det.detectMarkers(gray)
    d, pr = det
    return cv2.aruco.detectMarkers(gray, d, parameters=pr)


def _rvec_to_rpy_deg(rvec):
    R, _ = cv2.Rodrigues(rvec)
    pitch = -math.asin(np.clip(R[2, 0], -1.0, 1.0))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _reproj_error(corners, obj_pts, rvec, tvec, K, dist):
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    return float(np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1).mean())


class EihMarkerDebugNode(Node):
    def __init__(self):
        super().__init__("eih_marker_debug_node")
        self.det = _make_detector()
        self.K = None
        self.dist = np.zeros(5)
        self.declare_parameter("marker_size_m", 0.0)  # 0=ID별 기본값 사용, >0이면 모든 ID에 이 크기 강제

        self.create_subscription(CameraInfo, "/vision/eih_camera_info", self._on_info, _LATCH)
        self.create_subscription(Image, "/vision/eih_image", self._on_image, 5)
        self.pub_debug = self.create_publisher(Image, "/vision/eih_debug_image", 5)
        self.get_logger().info("eih_marker_debug_node 초기화 완료 — rqt_image_view로 /vision/eih_debug_image 볼 것")

    def _on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, float).reshape(3, 3)

    def _on_image(self, msg: Image):
        if self.K is None:
            return
        rgba = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = _detect_markers(gray, self.det)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(bgr, corners, ids)
            override = self.get_parameter("marker_size_m").value
            for c, mid in zip(corners, ids.flatten()):
                size = override if override > 0 else MARKER_SIZE_BY_ID.get(int(mid), FALLBACK_MARKER_SIZE)
                half = size / 2.0
                obj_pts = np.array([[-half, half, 0.], [half, half, 0.],
                                     [half, -half, 0.], [-half, -half, 0.]])
                ok, rvec, tvec = cv2.solvePnP(obj_pts, c.reshape(4, 2), self.K, self.dist)
                if not ok:
                    continue
                cv2.drawFrameAxes(bgr, self.K, self.dist, rvec, tvec, size)
                roll, pitch, yaw = _rvec_to_rpy_deg(rvec)
                rep = _reproj_error(c, obj_pts, rvec, tvec, self.K, self.dist)
                x, y, z = tvec.flatten()
                dist_m = float(np.linalg.norm(tvec))  # 렌즈 광학중심 기준 직선거리
                corner_px = c.reshape(4, 2)[0]
                for i, line in enumerate((
                        f"ID{mid} size={size*1000:.0f}mm",
                        f"xyz(cam)=({x:+.3f},{y:+.3f},{z:+.3f})m dist={dist_m*100:.1f}cm",
                        f"rpy=({roll:+.0f},{pitch:+.0f},{yaw:+.0f})deg",
                        f"rep={rep:.2f}px")):
                    cv2.putText(bgr, line, (int(corner_px[0]), int(corner_px[1]) - 40 + i * 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        out = Image()
        out.header = msg.header
        out.height, out.width = bgr.shape[0], bgr.shape[1]
        out.encoding = "bgr8"
        out.step = out.width * 3
        out.data = array.array("B", bgr.tobytes())
        self.pub_debug.publish(out)


def main():
    rclpy.init()
    node = EihMarkerDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
