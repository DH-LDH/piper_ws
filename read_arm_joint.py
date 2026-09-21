#!/usr/bin/env python3
# =============================================================================
# read_arm_joint.py — 지금 팔이 어느 관절각에 있는지 읽는다(도 단위).
#
# piper_driver_node가 /joint_states로 실제 관절각(엔코더값)을 60Hz로 발행하므로
# 그걸 받아서 찍기만 한다. set_arm_joint.py가 "보내는" 도구라면 이건 "읽는" 도구다.
# 손으로 팔을 원하는 자세에 맞춘 뒤 이 값을 place_done_q_deg 등에 그대로 넣으면 된다.
#
# 사용법:
#   python3 read_arm_joint.py          # 한 번 읽고 종료
#   python3 read_arm_joint.py -w       # 0.5초마다 계속(손으로 움직이며 볼 때)
#
# ★ 손으로 움직이려면 팔의 토크가 빠져 있어야 한다. 토크가 걸린 상태에서 억지로
#   밀지 말 것. 그리고 토크를 뺄 때는 반드시 팔을 손으로 받치고 할 것
#   (브레이크 없는 축이 있어 무동력 낙하 위험).
# =============================================================================
import math
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from step22_common import PIPER_JOINT_NAMES


def main():  # /joint_states를 받아 현재 관절각을 도 단위로 찍는다
    watch = "-w" in sys.argv[1:] or "--watch" in sys.argv[1:]

    rclpy.init()
    node = Node("read_arm_joint")
    latest = {"msg": None}
    node.create_subscription(JointState, "/joint_states",
                             lambda m: latest.__setitem__("msg", m), 10)

    deadline = time.time() + 5.0
    while latest["msg"] is None and time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if latest["msg"] is None:
        print("  ★ /joint_states 수신 없음 — piper_driver_node가 떠 있는지 확인할 것")
        node.destroy_node(); rclpy.shutdown(); sys.exit(2)

    def show():  # 최신 수신값 한 줄 출력 — 런치 인자에 그대로 붙여넣을 수 있는 형태
        m = latest["msg"]
        try:
            deg = [math.degrees(m.position[m.name.index(n)]) for n in PIPER_JOINT_NAMES]
        except ValueError:
            print(f"  ★ 팔 관절이 안 보임 (받은 이름: {list(m.name)})")
            return
        # 그대로 복사해서 런치 인자에 붙여넣을 수 있는 형태로 같이 찍는다.
        print(f"  joint1~6 = [{', '.join(f'{v:.1f}' for v in deg)}]  deg")

    if not watch:
        show()
    else:
        print("  (Ctrl-C로 종료)")
        try:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.1)
                show()
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
