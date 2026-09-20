#!/usr/bin/env python3
# =============================================================================
# step_confirm.py — arm_node의 step_confirm:=true 모드에서 단계별 진행 승인용.
#
# arm_node가 phase를 바꿀 때마다 멈추고 /arm/step_confirm(Bool) 신호를 기다린다
# (콘솔에 "[STEP] '<phase>' 단계 진입 대기 중"이 뜸). 이 스크립트를 띄워두고
# Enter를 칠 때마다 그 신호를 한 번씩 쏴서 다음 단계로 넘어가게 한다.
# 대기 중인 단계는 arm_node가 /arm/step_wait로 알려준다.
#
# 사용법:
#   python3 step_confirm.py
#   (그 다음 매번 Enter — Ctrl+C로 종료)
# =============================================================================
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Bool, String

_LATCH = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class StepConfirm(Node):
    def __init__(self):
        super().__init__("step_confirm")
        self.pub = self.create_publisher(Bool, "/arm/step_confirm", 10)
        self.waiting_phase = None
        self.create_subscription(String, "/arm/step_wait", self._on_step_wait, _LATCH)

    def _on_step_wait(self, msg: String):
        self.waiting_phase = msg.data
        print(f"\n  [대기] '{msg.data}' 단계 — Enter를 치면 진행합니다")


def main():
    rclpy.init()
    node = StepConfirm()
    # input()이 블로킹이라 스핀은 별도 스레드에서 — 안 그러면 대기 단계 알림을 못 받는다.
    spinner = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spinner.start()
    time.sleep(0.5)  # 디스커버리 대기 — 이미 걸려 있는 대기 단계(latch)를 먼저 받는다
    print("Enter를 치면 arm_node가 다음 단계로 진행합니다 (Ctrl+C 종료)")
    try:
        while True:
            input()
            phase = node.waiting_phase or "?(arm_node 대기 알림 수신 전)"
            node.pub.publish(Bool(data=True))
            print(f"  → '{phase}' 단계 진행 신호 전송")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        rclpy.shutdown()      # spin 스레드를 먼저 빠져나오게 한다(먼저 destroy하면 abort)
        spinner.join(timeout=1.0)


if __name__ == "__main__":
    main()
