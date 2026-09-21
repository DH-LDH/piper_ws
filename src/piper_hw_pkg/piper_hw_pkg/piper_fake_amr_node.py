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


STARTUP_LOCK_DELAY_SEC = 3.0   # 대기


class PiperFakeAmrNode(Node):  # AMR 없는 구성의 스텁 — 개시 신호와 body_link TF만 대신 쏜다
    def __init__(self):  # 토픽 등록 + body_link→world TF 1회 발행(팔 90° 장착 회전 포함)
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
        tf.child_frame_id = "world"  # URDF 루트(world)를 body_link 밑에 붙임
        h = math.radians(ARM_BASE_YAW_DEG) / 2.0  # 팔이 베이스 기준 90° 돌아 장착 — 빼면 X/Y가 뒤바뀐다
        tf.transform.rotation.z = math.sin(h)
        tf.transform.rotation.w = math.cos(h)
        self.tf_static.sendTransform(tf)

        self.get_logger().info("piper_fake_amr_node 초기화 완료 — AMR 없이 lock/place_lock/body_link TF 대신 발행")

    def _send_initial_lock(self):  # 기동 3초 뒤 /amr/lock=True 1회 — 픽 시퀀스 개시 신호
        self.pub_lock.publish(Bool(data=True))
        self.get_logger().info("/amr/lock=True 발행 — 픽 시퀀스 개시 신호(가짜 AMR)")
        self._startup_timer.cancel()  # 1회만

    def _on_arm_status(self, msg: String):  # phase가 place_wait가 되면 place_lock 1회 — place 개시 신호
        phase = (msg.data.split(" ")[0].split("=")[-1]
                 if "phase=" in msg.data else msg.data)
        if phase == "place_wait" and not self._sent_place_lock:
            self._sent_place_lock = True
            self.pub_place_lock.publish(Bool(data=True))
            self.get_logger().info("/amr/place_lock=True 발행 — place 시퀀스 개시 신호(가짜 AMR)")


def main():  # 노드 기동 진입점
    rclpy.init()
    node = PiperFakeAmrNode()
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
