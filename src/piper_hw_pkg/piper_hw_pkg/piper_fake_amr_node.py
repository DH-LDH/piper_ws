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

# 팔이 고정 베이스에 단독으로 설치된 구성이라 이동대차가 없다. 그런데 arm_node는
# 시퀀스 개시 신호(/amr/lock, /amr/place_lock)를 기다리므로, "이미 정위치에 있다"는
# 전제를 그대로 신호로 쏴주는 최소 스텁이 필요하다.
#
# body_link TF도 여기서 정적으로 쏜다 — 없으면 vision_node의 body_link→eih_cam 조회가
# 매번 실패해 손목캠 마커 인식이 통째로 죽는다. 이때 ARM_BASE_YAW_DEG(90°) 회전을
# 반드시 넣어야 한다. 팔이 베이스 기준 90도 돌아 장착돼 있어서, identity로 두면
# EndPoseCtrl의 X가 body_link Y(전진방향)에 대응하는 관계가 깨져 X/Y가 뒤바뀐다.

STARTUP_LOCK_DELAY_SEC = 3.0   # 다른 노드들이 다 올라올 시간을 준다


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
        tf.child_frame_id = "world"  # URDF 루트(world)를 body_link 밑에 붙임 — base_link에 부모 이중선언 방지
        h = math.radians(ARM_BASE_YAW_DEG) / 2.0
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
        # launch가 SIGINT를 보내면 rclpy 시그널 핸들러가 이미 컨텍스트를 내려서
        # 여기서 또 부르면 RCLError를 뱉는다(동작엔 영향 없지만 매번 traceback).
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
