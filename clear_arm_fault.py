#!/usr/bin/env python3
# =============================================================================
# clear_arm_fault.py — arm_status의 fault(TARGET_POS_EXCEEDS_LIMIT 등)를 복구한다.
#
# piper_driver_node는 fault를 감지하면 _fault_latched=True로 걸어 잠그고 이후
# 모션 명령을 전부 끊는다(ROS 토픽으로는 복구 불가 — 코드상 그럴 방법이 없음).
# 복구는 CAN으로 EmergencyStop(0x02, "복구")를 직접 보내는 것뿐이라 piper_sdk에
# 바로 붙는다. driver_node와 같은 CAN 버스를 같이 열게 되므로, 이 스크립트는
# piper_real.launch.py를 끈 상태(또는 최소한 driver_node가 죽은 상태)에서 쓴다.
#
# ★ 안전: 복구 명령 직후 짧은 순간 토크가 안 걸려 팔이 처지는 현상이 실기에서
#   재현됨(HANDOFF_JETSON.md 5절, 2026-09-17). 반드시 팔을 손으로 받치고 실행할 것.
#
# 쓰는 법:
#   python3 clear_arm_fault.py
# =============================================================================
import sys
import time

from piper_sdk import C_PiperInterface_V2


def main():
    print("  ★ 팔을 손으로 받쳤는지 확인하세요 — 복구 직후 짧게 처질 수 있습니다.")
    input("  준비됐으면 Enter: ")

    piper = C_PiperInterface_V2("can_piper")
    piper.ConnectPort()
    time.sleep(0.1)

    st_before = piper.GetArmStatus().arm_status.arm_status
    print(f"  복구 전 arm_status = {st_before}")

    piper.EmergencyStop(0x02)  # MotionCtrl_1(emergency_stop=0x02 "복구", ...)
    time.sleep(1.0)  # 상태 갱신에 0.2초로는 부족했음(실측) — 여유 있게 대기

    st_after = piper.GetArmStatus().arm_status.arm_status
    print(f"  복구 후 arm_status = {st_after}")
    if int(st_after) == 0:
        print("  → 정상(0). fault 해소됨.")
    else:
        print("  → 아직 fault 남음. 관절이 실제로 한계를 넘는 자세라면 손으로 안전한 "
              "자세로 옮긴 뒤 다시 시도할 것.")


if __name__ == "__main__":
    main()
