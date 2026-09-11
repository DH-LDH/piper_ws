import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String

# v1(AMR 없이 팔 단독)에서는 amr_node가 아예 안 돈다 — arm_node.py가 그래도 기다리는
# /amr/lock(픽 시퀀스 개시)·/amr/place_lock(place 시퀀스 개시) 신호를 대신 쏴주는 최소
# 스텁. "AMR이 이미 정위치에 도킹했다"는 v1의 전제(팔 고정 베이스)를 그대로 신호로 낸다.

STARTUP_LOCK_DELAY_SEC = 3.0   # 다른 노드들이 다 올라올 시간을 준다


class PiperFakeAmrNode(Node):
    def __init__(self):
        super().__init__("piper_fake_amr_node")
        self.pub_lock = self.create_publisher(Bool, "/amr/lock", 1)
        self.pub_place_lock = self.create_publisher(Bool, "/amr/place_lock", 1)
        self._sent_place_lock = False

        self.create_subscription(String, "/arm/status", self._on_arm_status, 10)
        self._startup_timer = self.create_timer(STARTUP_LOCK_DELAY_SEC, self._send_initial_lock)
        self.get_logger().info("piper_fake_amr_node 초기화 완료 — AMR 없이 lock/place_lock 대신 발행")

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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
