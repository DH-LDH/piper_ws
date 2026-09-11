import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, Int32
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState

from step22_common import (
    SEARCH_Q, PIPER_JOINT_NAMES, TOOL_AXIS_LOCAL, GRIPPER_DOWN,
    HOVER_PITCH_DEG, GRASP_PITCH_DEG, PLACE_PITCH_DEG,
    quat_from_two_vec, _quat_mul, _quat_axis_angle, _quat_to_R,
    unpack_joint_hold_target,
)

try:
    from piper_sdk import C_PiperInterface_V2, ArmMsgFeedbackStatusEnum
except ImportError:
    C_PiperInterface_V2 = None  # 노드 임포트는 되게 하고, 실제 사용 시점에 에러 (빌드/CI에서 미설치 허용)
    ArmMsgFeedbackStatusEnum = None

# ── 사용자 조정 파라미터(ROS 파라미터로도 덮어쓰기 가능) ─────────────────────

CAN_NAME_DEFAULT = "can0"
REALLY_ENABLE_DEFAULT = False   # True로 명시해야 EnableArm/모션 명령을 실제로 보낸다(첫 기동 안전장치)
MOVE_SPD_RATE_DEFAULT = 20      # [%] 초기 실물 테스트 안전을 위해 낮게 — 검증 후 사용자가 올릴 것
TICK_HZ = 60.0                  # arm_node.py/gripper 쪽 스텝 기반 상수(15초=900스텝 등)를 그대로 쓰기 위해 sim과 동일 rate 유지

# plant_node.py:108의 로컬 상수 — Lula가 아니라 실물 펌웨어가 IK를 풀지만, "어느 롤로
# 접근할지"는 파지 기하 자체의 문제라 실물에서도 동일해야 한다.
GRASP_ROLL_DEG = -90.0

# arm_node.py가 이 phase 문자열일 때 관절유지(JointCtrl, SEARCH_Q)를 쓴다 — 나머지는 전부
# /arm/cartesian_target을 EndPoseCtrl로 스트리밍(plant_node.py:1671/1709 화이트리스트와 동일).
JOINT_HOLD_PHASES = ("wait", "place_ready", "place_home")
PLACE_MARKER_PHASES = ("place_hover", "place_detect", "place_descend")
CARTESIAN_PHASES = ("pre", "grasp", "lift", "place_lower",
                     "place_hover", "place_detect", "place_descend",
                     "place_release", "place_retreat")

# GetArmStatus().arm_status가 이 값이면 새 모션 명령을 멈추고 경고만 남긴다(자동복구 안 함).
# 0x01 비상정지, 0x02 IK 무해, 0x03 특이점, 0x04 관절한계초과, 0x07 충돌.
FAULT_CODES = {1, 2, 3, 4, 7}

# ── 조정 파라미터 끝 ─────────────────────────────────────────────────────────


def _build_down_quats():
    """DOWN_QUAT/HOVER_DOWN_QUAT/PLACE_DOWN_QUAT — plant_node.py:1173-1208과 동일 식."""
    def _one(pitch_deg):
        q = quat_from_two_vec(TOOL_AXIS_LOCAL, GRIPPER_DOWN)
        if abs(pitch_deg) > 1e-6:
            q = _quat_mul(_quat_axis_angle(np.array([1.0, 0.0, 0.0]), pitch_deg), q)
            q = q / np.linalg.norm(q)
        if abs(GRASP_ROLL_DEG) > 1e-6:
            q = _quat_mul(q, _quat_axis_angle(TOOL_AXIS_LOCAL, GRASP_ROLL_DEG))
            q = q / np.linalg.norm(q)
        return q
    return _one(GRASP_PITCH_DEG), _one(HOVER_PITCH_DEG), _one(PLACE_PITCH_DEG)


def _quat_to_rpy_deg(q):
    """쿼터니언(w,x,y,z) → (roll,pitch,yaw)[deg], R=Rz(yaw)@Ry(pitch)@Rx(roll) 가정.
    ★검증 필요: piper 펌웨어의 RX,RY,RZ 합성 순서가 이와 같은지 실측으로 확인 전까지는
    추정치 — GetArmEndPoseMsgs() 피드백과 첫 테스트 시 반드시 대조할 것."""
    R = _quat_to_R(q)
    pitch = -math.asin(np.clip(R[2, 0], -1.0, 1.0))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class PiperDriverNode(Node):
    def __init__(self):
        super().__init__("piper_driver_node")

        self.declare_parameter("can_name", CAN_NAME_DEFAULT)
        self.declare_parameter("really_enable", REALLY_ENABLE_DEFAULT)
        self.declare_parameter("move_spd_rate_ctrl", MOVE_SPD_RATE_DEFAULT)
        can_name = self.get_parameter("can_name").value
        self.really_enable = bool(self.get_parameter("really_enable").value)
        self.move_spd = int(self.get_parameter("move_spd_rate_ctrl").value)

        down_q, hover_q, place_q = _build_down_quats()
        quat_by_phase = {ph: (place_q if ph in PLACE_MARKER_PHASES else down_q)
                         for ph in CARTESIAN_PHASES}
        quat_by_phase["hover"] = hover_q
        self._rpy_by_phase = {ph: _quat_to_rpy_deg(q) for ph, q in quat_by_phase.items()}

        self.arm_phase = "wait"
        self.joint_hold_target = np.array(SEARCH_Q, float)
        self.cartesian_target = None
        self._last_move_mode = None  # ModeCtrl 중복호출 방지(CAN 트래픽 절약)
        self._fault_latched = False

        self.piper = None
        if C_PiperInterface_V2 is None:
            self.get_logger().error(
                "piper_sdk가 설치돼 있지 않습니다 — pip3 install piper_sdk python-can 필요. "
                "노드는 뜨지만 CAN 명령은 전부 무시됩니다.")
        else:
            self.piper = C_PiperInterface_V2(can_name)
            self.piper.ConnectPort()
            if self.really_enable:
                self.get_logger().warn(f"really_enable=True — 실제로 팔을 활성화합니다 (speed {self.move_spd}%)")
                while not self.piper.EnablePiper():
                    time.sleep(0.01)
            else:
                self.get_logger().warn(
                    "really_enable=False(기본값) — CAN 연결/피드백만 하고 모션 명령은 보내지 않습니다. "
                    "실제로 움직이려면 파라미터 really_enable:=true 로 재기동할 것.")

        self.pub_ee_pose = self.create_publisher(Point, "/arm/ee_pose_body", 10)
        self.pub_joint_states = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_tick = self.create_publisher(Int32, "/plant/tick", 20)
        self.pub_gripper_feedback = self.create_publisher(Float32MultiArray, "/piper/gripper_feedback", 10)

        self.create_subscription(String, "/arm/status", self._on_arm_status, 10)
        self.create_subscription(Point, "/arm/cartesian_target", self._on_cart_target, 10)
        self.create_subscription(Float32MultiArray, "/arm/joint_hold_target", self._on_joint_hold, 10)
        self.create_subscription(Float32MultiArray, "/piper/gripper_target_cmd", self._on_gripper_target, 10)

        self._tick_n = 0
        self.create_timer(1.0 / TICK_HZ, self._tick)
        self.get_logger().info(f"piper_driver_node 초기화 완료 (can={can_name}, really_enable={self.really_enable})")

    # ── 구독 콜백 ────────────────────────────────────────────────────────────
    def _on_arm_status(self, msg: String):
        self.arm_phase = (msg.data.split(" ")[0].split("=")[-1]
                           if "phase=" in msg.data else msg.data)

    def _on_cart_target(self, msg: Point):
        self.cartesian_target = np.array([msg.x, msg.y, msg.z], float)

    def _on_joint_hold(self, msg: Float32MultiArray):
        self.joint_hold_target = unpack_joint_hold_target(msg.data)

    def _on_gripper_target(self, msg: Float32MultiArray):
        if self.piper is None or not self.really_enable or self._fault_latched:
            return
        angle_mm, effort_nm = float(msg.data[0]), float(msg.data[1])
        self.piper.GripperCtrl(round(angle_mm * 1000), round(effort_nm * 1000), 0x01, 0)

    # ── 60Hz 틱 ─────────────────────────────────────────────────────────────
    def _tick(self):
        self._tick_n += 1
        self.pub_tick.publish(Int32(data=self._tick_n))
        if self.piper is None:
            return

        self._publish_feedback()

        st = self.piper.GetArmStatus().arm_status
        if int(st) in FAULT_CODES:
            if not self._fault_latched:
                self.get_logger().error(f"★ 팔 고장/이상 상태 감지(arm_status={st}) — 새 모션 명령 중단")
            self._fault_latched = True
            return
        self._fault_latched = False

        if not self.really_enable:
            return

        if self.arm_phase in JOINT_HOLD_PHASES:
            self._send_move_mode(0x01)  # MOVE J
            q_deg = np.degrees(self.joint_hold_target)
            self.piper.JointCtrl(*[round(v * 1000) for v in q_deg])
        elif self.arm_phase in CARTESIAN_PHASES and self.cartesian_target is not None:
            self._send_move_mode(0x02)  # MOVE L
            rx, ry, rz = self._rpy_by_phase[self.arm_phase]
            x, y, z = self.cartesian_target
            self.piper.EndPoseCtrl(
                round(x * 1_000_000), round(y * 1_000_000), round(z * 1_000_000),
                round(rx * 1000), round(ry * 1000), round(rz * 1000))
        # 그 외(detect/verify/place_wait 등)는 팔이 마지막 목표를 유지 — 새 명령 없음.

    def _send_move_mode(self, move_mode):
        if self._last_move_mode == move_mode:
            return
        self.piper.MotionCtrl_2(0x01, move_mode, self.move_spd, 0x00)
        self._last_move_mode = move_mode

    def _publish_feedback(self):
        ep = self.piper.GetArmEndPoseMsgs().end_pose
        self.pub_ee_pose.publish(Point(
            x=ep.X_axis / 1_000_000.0, y=ep.Y_axis / 1_000_000.0, z=ep.Z_axis / 1_000_000.0))

        js = self.piper.GetArmJointMsgs().joint_state
        deg = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(PIPER_JOINT_NAMES)
        msg.position = [math.radians(v / 1000.0) for v in deg]
        self.pub_joint_states.publish(msg)

        gs = self.piper.GetArmGripperMsgs().gripper_state
        # 계약: [angle_mm, effort_Nm, foc_status] — piper_gripper_node.py와 맞출 것
        self.pub_gripper_feedback.publish(Float32MultiArray(
            data=[gs.grippers_angle / 1000.0, gs.grippers_effort / 1000.0,
                  float(gs.status_code)]))

    def destroy_node(self):
        # 종료 시 DisableArm을 자동 호출하지 않는다 — 브레이크 미보유 축이 있다면
        # 무동력 낙하 위험(문서상 근거 불충분). 마지막 자세 유지가 기본 동작.
        super().destroy_node()


def main():
    rclpy.init()
    node = PiperDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
