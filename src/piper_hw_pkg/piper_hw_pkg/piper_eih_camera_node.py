import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import array
import math
import threading
import time

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

from loop_profiler import declare_profile

# 실물 손목캠(eih) 소스: RealSense D455 컬러 스트림 → /vision/eih_image 발행.

_LATCH = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

FRAME_WIDTH, FRAME_HEIGHT, FRAME_FPS = 640, 480, 30
PUBLISH_HZ = 15.0

CAM_TCP_OFFSET_X_DEFAULT = -0.085          # [m] 상하
CAM_TCP_OFFSET_Y_DEFAULT = -0.01           # [m] 좌우
CAM_TCP_OFFSET_Z_DEFAULT = 0.026           # [m] 앞뒤
CAM_TCP_OFFSET_PITCH_DEG_DEFAULT = 27.5    # [deg] 틸트
CAM_TCP_OFFSET_ROLL_DEG_DEFAULT = -90.0    # [deg] 광축 


class PiperEihCameraNode(Node):  # RealSense D455 → /vision/eih_image + link6→eih_cam TF
    def __init__(self):  # 파라미터·TF·RealSense 기동, 캡처 스레드 시작
        super().__init__("piper_eih_camera_node")

        self._latest_rgba = None
        self._latest_lock = threading.Lock()
        self._stop_capture = threading.Event()
        self._capture_thread = None
        self._prof = declare_profile(self)
        self._prof_cap = self._prof.loop("capture") if self._prof else None
        self._prof_pub = self._prof.loop("pub") if self._prof else None
        self._latest_meta = None  # profile용 (캡처 perf_counter, frame_number)
        self._last_pub_fn = None

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
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                K, D, None, K, (FRAME_WIDTH, FRAME_HEIGHT), cv2.CV_16SC2)

            self.pub_info = self.create_publisher(CameraInfo, "/vision/eih_camera_info", _LATCH)
            info = CameraInfo()
            info.width, info.height = FRAME_WIDTH, FRAME_HEIGHT
            info.k = K.flatten().tolist()
            info.d = [0.0] * 5  
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

    def _publish_cam_tcp_tf(self, x=None, y=None, z=None, pitch_deg=None, roll_deg=None):  # link6→eih_cam 정적 TF 발행 — 손목캠 장착 위치와 틸트를 좌표계에 반영
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

    def _on_cam_tcp_params_set(self, params):  # 오프셋 파라미터가 바뀌면 TF 즉시 재발행(실시간 캘리브용)
        # get_parameter는 아직 옛값이라 변경분은 params에서 직접 읽는다.
        overrides = {p.name: p.value for p in params}
        self._publish_cam_tcp_tf(
            x=overrides.get("cam_tcp_offset_x"),
            y=overrides.get("cam_tcp_offset_y"),
            z=overrides.get("cam_tcp_offset_z"),
            pitch_deg=overrides.get("cam_tcp_offset_pitch_deg"),
            roll_deg=overrides.get("cam_tcp_offset_roll_deg"))
        return SetParametersResult(successful=True)

    def _capture_loop(self):  # 별도 스레드 — 프레임 받아 왜곡보정 후 최신 1장만 보관
        p = self._prof_cap
        while not self._stop_capture.is_set():
            try:
                frames = self.pipeline.wait_for_frames(1000)
            except RuntimeError:
                continue
            color = frames.get_color_frame()
            if not color:
                continue
            if p:
                p.begin()  # interval = 실제 프레임 도착 간격(조도에 따른 fps 저하가 여기 보인다)
                p.note = self._frame_note(color)
            frame_bgr = np.asanyarray(color.get_data())
            frame_bgr = cv2.remap(frame_bgr, self.map1, self.map2, cv2.INTER_LINEAR)
            rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
            with self._latest_lock:
                self._latest_rgba = rgba
                if p:
                    self._latest_meta = (time.perf_counter(), color.get_frame_number())
            if p: p.end()

    @staticmethod
    def _frame_note(color):  # 프레임 번호와 실제 노출[us](지원 시) — 조도별 비교용
        fn = color.get_frame_number()
        md = rs.frame_metadata_value.actual_exposure
        exp = color.get_frame_metadata(md) if color.supports_frame_metadata(md) else -1
        return f"fn={fn};exp_us={exp}"

    def _on_timer(self):  # 15Hz로 최신 프레임을 /vision/eih_image에 발행
        p = self._prof_pub
        if p: p.begin()
        with self._latest_lock:
            rgba = self._latest_rgba
            meta = self._latest_meta
        if rgba is None:
            return
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = rgba.shape[0], rgba.shape[1]
        msg.encoding = "rgba8"
        msg.step = msg.width * 4
        msg.data = array.array("B", rgba.tobytes())  # bytes 그대로 넣으면 publish()가 ~200ms로 느려짐
        self.pub_image.publish(msg)
        if p and meta is not None:
            p.lap("publish")
            t_cap, fn = meta
            p.note = f"fn={fn};dup={int(fn == self._last_pub_fn)}"  # dup=1: 새 프레임 없이 같은 장 재발행
            self._last_pub_fn = fn
            p.value("cap_to_pub", (time.perf_counter() - t_cap) * 1e3, p.note)
            p.end()

    def destroy_node(self):  # 종료 시 캡처 스레드와 RealSense 파이프라인 정리
        self._stop_capture.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
        if self.pipeline is not None:
            self.pipeline.stop()
        super().destroy_node()


def main():  # 노드 기동 진입점
    rclpy.init()
    node = PiperEihCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
