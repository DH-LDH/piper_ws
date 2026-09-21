#!/usr/bin/env python3
# =============================================================================
# set_arm_joint.py — 팔을 원하는 관절 자세로 직접 보내는 도구.
#
# arm_node를 안 띄우면 driver의 arm_phase가 "wait"에 고정되고, driver는 매 틱
# /arm/joint_hold_target 구독값을 JointCtrl로 그대로 적용한다. 이 토픽에 관절값을
# publish하면 마커 검출·시퀀스·도달 판정을 전부 건너뛰고 팔을 그 자세로 보낼 수 있다.
# 손목캠 시야 확인, 캘리브레이션 자세 잡기, 새 자세의 간섭 여부 확인에 쓴다.
#
# 쓰는 법:
#   ros2 launch piper_hw_pkg piper_real.launch.py really_enable:=true arm:=false
#   python3 set_arm_joint.py 0 45 -90 0 45 0      # 도(deg) 단위, joint1~6 순서
#
# ★ IK도 간섭 검사도 거치지 않는다. 팔이 그 각도로 그냥 간다.
#   현재 자세는 read_arm_joint.py로 읽을 수 있다.
# =============================================================================
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
    # 구독자(driver)와 매칭될 때까지 기다린다 — 바로 publish하고 종료하면
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
