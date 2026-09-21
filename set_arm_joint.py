#!/usr/bin/env python3
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Float32MultiArray


def main():  # 인자로 받은 관절각을 /arm/joint_hold_target으로 발행 — driver가 MOVE J로 적용
    if len(sys.argv) != 7:
        print(f"사용법: {sys.argv[0]} j1 j2 j3 j4 j5 j6   (도 단위, joint1~6)")
        sys.exit(1)
    q6_deg = [float(v) for v in sys.argv[1:7]]
    q6_rad = np.radians(q6_deg)

    rclpy.init()
    node = Node("set_arm_joint")
    # TRANSIENT_LOCAL: driver가 이 노드보다 늦게 떠도 마지막 값을 받는다
    latch = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(Float32MultiArray, "/arm/joint_hold_target", latch)
    # 구독자(driver)와 매칭될 때까지 기다린다
    deadline = time.time() + 5.0
    while pub.get_subscription_count() == 0 and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if pub.get_subscription_count() == 0:
        print("  ★ /arm/joint_hold_target 구독자가 없습니다 — driver(또는 plant)가 떠 있는지 확인할 것")
        node.destroy_node(); rclpy.shutdown(); sys.exit(2)
    pub.publish(Float32MultiArray(data=[float(v) for v in q6_rad]))
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.1)  # publish 실제 전송 대기
    print(f"  → /arm/joint_hold_target publish ({pub.get_subscription_count()}개 구독자): "
          f"{q6_deg} deg = {q6_rad.round(3).tolist()} rad")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
