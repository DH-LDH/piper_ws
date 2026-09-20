#!/usr/bin/env python3
# =============================================================================
# set_arm_joint.py — 정적 픽 검증용 관절 튜닝 도구.
#
# arm_node/amr_node를 켜지 않은 상태(plant_node만 실행)에서는
# ros.arm_status가 "wait"에 고정되고, plant_node는 매틱
# ros.joint_hold_target(/arm/joint_hold_target 구독값, 기본 SEARCH_Q)을
# 그대로 팔 6관절에 적용한다(plant_node.py:1049-1051). 이 토픽에 원하는
# 관절값을 publish하면 락/vision/IK 파이프라인 없이 팔을 그 자세로 바로
# 보낼 수 있다 — GUI로 손목캠 시야·수직하강 여부를 직접 확인할 때 쓴다.
#
# 사용법(도(deg) 단위 6개, joint1~6 순서):
#   python3 set_arm_joint.py 0 45 -90 0 45 0
# =============================================================================
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Float32MultiArray


def main():
    if len(sys.argv) != 7:
        print(f"사용법: {sys.argv[0]} j1 j2 j3 j4 j5 j6   (도 단위, joint1~6)")
        sys.exit(1)
    q6_deg = [float(v) for v in sys.argv[1:7]]
    q6_rad = np.radians(q6_deg)

    rclpy.init()
    node = Node("set_arm_joint")
    # TRANSIENT_LOCAL: plant_node가 이 노드보다 늦게/먼저 떠도 마지막 값을 받는다
    latch = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(Float32MultiArray, "/arm/joint_hold_target", latch)
    # 구독자(driver/plant)와 매칭될 때까지 기다린다 — 바로 publish하고 종료하면
    # 디스커버리가 늦은 경우 메시지가 통째로 사라져서 팔이 안 움직인다.
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
