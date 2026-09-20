import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster

from step22_common import ARM_BASE_YAW_DEG

# v1(AMR 없이 팔 단독)에서는 amr_node가 아예 안 돈다 — arm_node.py가 그래도 기다리는
# /amr/lock(픽 시퀀스 개시)·/amr/place_lock(place 시퀀스 개시) 신호를 대신 쏴주는 최소
# 스텁. "AMR이 이미 정위치에 도킹했다"는 v1의 전제(팔 고정 베이스)를 그대로 신호로 낸다.
# body_link도 같은 이유로 여기서 정적으로 쏴준다 — 안 하면 vision_node의 body_link→eih_cam
# TF 조회가 매번 실패해서 손목캠 마커 인식이 통째로 죽는다.
# 2026-09-17 실기에서 확인: EndPoseCtrl 기준 X가 sim의 body_link Y(KEEP_DIST 전진방향)에
# 대응함 — sim의 ARM_BASE_YAW_DEG(=90°, 팔이 AMR 몸체 기준 90도 돌아서 장착된 것)를 그대로
# 반영해야 함(identity로 두면 X/Y가 뒤바뀜).

STARTUP_LOCK_DELAY_SEC = 3.0   # 다른 노드들이 다 올라올 시간을 준다


class PiperFakeAmrNode(Node):
    def __init__(self):
        super().__init__("piper_fake_amr_node")
        self.pub_lock = self.create_publisher(Bool, "/amr/lock", 1)
        self.pub_place_lock = self.create_publisher(Bool, "/amr/place_lock", 1)
        self._sent_place_lock = False

        self.create_subscription(String, "/arm/status", self._on_arm_status, 10)
        self._startup_timer = self.create_timer(STARTUP_LOCK_DELAY_SEC, self._send_initial_lock)

        self.tf_static = StaticTransformBroadcaster(self)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "body_link"
        tf.child_frame_id = "world"  # URDF 루트(world)를 body_link 밑에 붙임 — base_link에 부모 이중선언 방지
        h = math.radians(ARM_BASE_YAW_DEG) / 2.0
        tf.transform.rotation.z = math.sin(h)
        tf.transform.rotation.w = math.cos(h)
        self.tf_static.sendTransform(tf)

        self.get_logger().info("piper_fake_amr_node 초기화 완료 — AMR 없이 lock/place_lock/body_link TF 대신 발행")

    def _send_initial_lock(self):
        self.pub_lock.publish(Bool(data=True))
        self.get_logger().info("/amr/lock=True 발행 — 픽 시퀀스 개시 신호(가짜 AMR)")
        self._startup_timer.cancel()  # 1회만

    def _on_arm_status(self, msg: String):
        phase = (msg.data.split(" ")[0].split("=")[-1]
                 if "phase=" in msg.data else msg.data)
        if phase == "place_wait" and not self._sent_place_lock:
            self._sent_place_lock = True
            self.pub_place_lock.publish(Bool(data=True))
            self.get_logger().info("/amr/place_lock=True 발행 — place 시퀀스 개시 신호(가짜 AMR)")


def main():
    rclpy.init()
    node = PiperFakeAmrNode()
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
