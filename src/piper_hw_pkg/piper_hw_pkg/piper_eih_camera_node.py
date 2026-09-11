import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import Image, CameraInfo

# 실물 손목캠(eih) 소스 — 프레임을 잡아 /vision/eih_image로 그대로 발행한다.
# vision_node.py는 무수정 — _img_to_gray()가 4채널(RGBA) 프레임을 기대하므로(vision_node.py:77)
# 여기서 BGR→RGBA 변환만 맞춰준다.

_LATCH = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

CAMERA_DEVICE = 0     # TODO(실측/장치 확인 필요) — /dev/video0 등, 필요시 ROS 파라미터로 덮어쓰기
FRAME_WIDTH, FRAME_HEIGHT = 640, 480
PUBLISH_HZ = 15.0

# 카메라 내부파라미터 — TODO(캘리브레이션 필요): 체스보드/차트 캘리브레이션 결과로 갱신할 것.
# 지금은 "핀홀 근사 + 화면 중앙 주점" 자리표시자일 뿐이라 실제 solvePnP 정확도를 보장 못 한다.
_FOCAL_PLACEHOLDER_PX = 500.0
K_PLACEHOLDER = [
    _FOCAL_PLACEHOLDER_PX, 0.0, FRAME_WIDTH / 2.0,
    0.0, _FOCAL_PLACEHOLDER_PX, FRAME_HEIGHT / 2.0,
    0.0, 0.0, 1.0,
]


class PiperEihCameraNode(Node):
    def __init__(self):
        super().__init__("piper_eih_camera_node")

        self.declare_parameter("device", CAMERA_DEVICE)
        device = self.get_parameter("device").value
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            self.get_logger().error(f"카메라 장치 열기 실패(device={device}) — /vision/eih_image 발행 안 됨")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        self.pub_image = self.create_publisher(Image, "/vision/eih_image", 5)
        self.pub_info = self.create_publisher(CameraInfo, "/vision/eih_camera_info", _LATCH)

        info = CameraInfo()
        info.width, info.height = FRAME_WIDTH, FRAME_HEIGHT
        info.k = K_PLACEHOLDER
        self.pub_info.publish(info)
        self.get_logger().warn(
            "카메라 내부파라미터가 자리표시자입니다 — 실제 캘리브레이션 전까지 마커 pose 정확도 보장 안 됨")

        self.create_timer(1.0 / PUBLISH_HZ, self._on_timer)
        self.get_logger().info(f"piper_eih_camera_node 초기화 완료 (device={device})")

    def _on_timer(self):
        if not self.cap.isOpened():
            return
        ok, frame_bgr = self.cap.read()
        if not ok:
            return
        rgba = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGBA)
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = rgba.shape[0], rgba.shape[1]
        msg.encoding = "rgba8"
        msg.step = msg.width * 4
        msg.data = rgba.tobytes()
        self.pub_image.publish(msg)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
