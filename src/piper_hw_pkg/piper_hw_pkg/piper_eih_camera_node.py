import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import array
import math
import threading

import cv2
import numpy as np
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from rcl_interfaces.msg import SetParametersResult

# 실물 손목캠(eih) 소스: RealSense D455 컬러 스트림 → /vision/eih_image 발행.

_LATCH = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

FRAME_WIDTH, FRAME_HEIGHT, FRAME_FPS = 640, 480, 30
PUBLISH_HZ = 15.0

CAM_TCP_OFFSET_X_DEFAULT = -0.085          # [m] 상하
CAM_TCP_OFFSET_Y_DEFAULT = -0.01           # [m] 좌우
CAM_TCP_OFFSET_Z_DEFAULT = 0.026           # [m] 앞뒤
# ★ pitch는 각도계로 재면 안 된다. 카메라 하우징 기준면과 센서 광축이 일치한다는
# 보장이 없어 12.5° 틀린 적이 있다. 각도 오차는 거리에 비례하는 지향 오차로 나타나서
# 팔이 다가갈수록 추정 위치가 움직인다 — eih_cam_calib.py로 산출한 값을 쓸 것.
CAM_TCP_OFFSET_PITCH_DEG_DEFAULT = 27.5    # [deg] 틸트(link6 y축 회전)
CAM_TCP_OFFSET_ROLL_DEG_DEFAULT = -90.0    # [deg] 광축 롤 — 카메라 이미지 좌우를 로봇 좌우에 맞춤


class PiperEihCameraNode(Node):
    def __init__(self):
        super().__init__("piper_eih_camera_node")

        self._latest_rgba = None
        self._latest_lock = threading.Lock()
        self._stop_capture = threading.Event()
        self._capture_thread = None

        self.declare_parameter("cam_tcp_offset_x", CAM_TCP_OFFSET_X_DEFAULT)
        self.declare_parameter("cam_tcp_offset_y", CAM_TCP_OFFSET_Y_DEFAULT)
        self.declare_parameter("cam_tcp_offset_z", CAM_TCP_OFFSET_Z_DEFAULT)
        self.declare_parameter("cam_tcp_offset_pitch_deg", CAM_TCP_OFFSET_PITCH_DEG_DEFAULT)
        self.declare_parameter("cam_tcp_offset_roll_deg", CAM_TCP_OFFSET_ROLL_DEG_DEFAULT)
        self.tf_static = StaticTransformBroadcaster(self)
        self._publish_cam_tcp_tf()
        self.add_on_set_parameters_callback(self._on_cam_tcp_params_set)

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, FRAME_FPS)
        try:
            profile = self.pipeline.start(config)
        except RuntimeError as e:
            self.get_logger().error(f"RealSense 파이프라인 시작 실패: {e} — /vision/eih_image 발행 안 됨")
            self.pipeline = None
            self.map1 = self.map2 = None
        else:
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K = np.array([[intr.fx, 0.0, intr.ppx],
                          [0.0, intr.fy, intr.ppy],
                          [0.0, 0.0, 1.0]])
            D = np.array(intr.coeffs)
            # vision_node.py는 왜곡계수 없이 K만 쓰므로 여기서 미리 undistort해둔다.
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                K, D, None, K, (FRAME_WIDTH, FRAME_HEIGHT), cv2.CV_16SC2)

            self.pub_info = self.create_publisher(CameraInfo, "/vision/eih_camera_info", _LATCH)
            info = CameraInfo()
            info.width, info.height = FRAME_WIDTH, FRAME_HEIGHT
            info.k = K.flatten().tolist()
            info.d = [0.0] * 5  # 아래에서 undistort된 프레임을 발행하므로 발행 이미지 기준 왜곡 없음
            self.pub_info.publish(info)
            self.get_logger().info(
                f"RealSense 팩토리 캘리브레이션 적용: fx={intr.fx:.1f} fy={intr.fy:.1f} "
                f"ppx={intr.ppx:.1f} ppy={intr.ppy:.1f}")

        self.pub_image = self.create_publisher(Image, "/vision/eih_image", 5)

        if self.pipeline is not None:
            # 캡처는 별도 스레드에서: RealSense I/O가 executor를 막지 않게.
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()

        self.create_timer(1.0 / PUBLISH_HZ, self._on_timer)
        self.get_logger().info("piper_eih_camera_node 초기화 완료 (RealSense D455)")

    def _publish_cam_tcp_tf(self, x=None, y=None, z=None, pitch_deg=None, roll_deg=None):
        x = float(self.get_parameter("cam_tcp_offset_x").value) if x is None else x
        y = float(self.get_parameter("cam_tcp_offset_y").value) if y is None else y
        z = float(self.get_parameter("cam_tcp_offset_z").value) if z is None else z
        if pitch_deg is None:
            pitch_deg = float(self.get_parameter("cam_tcp_offset_pitch_deg").value)
        if roll_deg is None:
            roll_deg = float(self.get_parameter("cam_tcp_offset_roll_deg").value)
        # R = Ry(pitch) @ Rz(roll) — 틸트 후 광축 롤(카메라 이미지 좌우↔로봇 좌우 정렬)
        cp, sp = math.cos(math.radians(pitch_deg) / 2.0), math.sin(math.radians(pitch_deg) / 2.0)
        cr, sr = math.cos(math.radians(roll_deg) / 2.0), math.sin(math.radians(roll_deg) / 2.0)
        w, qx, qy, qz = cp * cr, sp * sr, sp * cr, cp * sr

        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "link6"
        tf.child_frame_id = "eih_cam"
        tf.transform.translation.x = x
        tf.transform.translation.y = y
        tf.transform.translation.z = z
        tf.transform.rotation.x = qx
        tf.transform.rotation.y = qy
        tf.transform.rotation.z = qz
        tf.transform.rotation.w = w
        self.tf_static.sendTransform(tf)
        self.get_logger().info(
            f"link6→eih_cam TF 발행: xyz=({x:.3f},{y:.3f},{z:.3f}) "
            f"pitch={pitch_deg:.1f}deg roll={roll_deg:.1f}deg")

    def _on_cam_tcp_params_set(self, params):
        # get_parameter는 아직 옛값이라 변경분은 params에서 직접 읽는다.
        overrides = {p.name: p.value for p in params}
        self._publish_cam_tcp_tf(
            x=overrides.get("cam_tcp_offset_x"),
            y=overrides.get("cam_tcp_offset_y"),
            z=overrides.get("cam_tcp_offset_z"),
            pitch_deg=overrides.get("cam_tcp_offset_pitch_deg"),
            roll_deg=overrides.get("cam_tcp_offset_roll_deg"))
        return SetParametersResult(successful=True)

    def _capture_loop(self):
        while not self._stop_capture.is_set():
            try:
                frames = self.pipeline.wait_for_frames(1000)
            except RuntimeError:
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            frame_bgr = np.asanyarray(color.get_data())
            frame_bgr = cv2.remap(frame_bgr, self.map1, self.map2, cv2.INTER_LINEAR)
            rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
            with self._latest_lock:
                self._latest_rgba = rgba

    def _on_timer(self):
        with self._latest_lock:
            rgba = self._latest_rgba
        if rgba is None:
            return
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = rgba.shape[0], rgba.shape[1]
        msg.encoding = "rgba8"
        msg.step = msg.width * 4
        msg.data = array.array("B", rgba.tobytes())  # bytes 그대로 넣으면 publish()가 ~200ms로 느려짐
        self.pub_image.publish(msg)

    def destroy_node(self):
        self._stop_capture.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
        if self.pipeline is not None:
            self.pipeline.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = PiperEihCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # launch가 SIGINT를 보내면 rclpy 시그널 핸들러가 이미 컨텍스트를 내려서
        # 여기서 또 부르면 RCLError를 뱉는다(동작엔 영향 없지만 매번 traceback).
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
