import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import math
import time
from collections import deque
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, Int32
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState

from step22_common import (
    SEARCH_Q, PIPER_JOINT_NAMES, TOOL_AXIS_LOCAL, GRIPPER_DOWN,
    HOVER_PITCH_DEG, GRASP_PITCH_DEG, PLACE_PITCH_DEG, ARM_BASE_YAW_DEG,
    quat_from_two_vec, _quat_mul, _quat_axis_angle, _quat_to_R,
    unpack_joint_hold_target,
)
from loop_profiler import declare_profile

try:
    from piper_sdk import C_PiperInterface_V2, ArmMsgFeedbackStatusEnum
except ImportError:
    C_PiperInterface_V2 = None  # 노드 임포트는 되게 하고, 실제 사용 시점에 에러 (빌드/CI에서 미설치 허용)
    ArmMsgFeedbackStatusEnum = None

# ── 사용자 조정 파라미터(ROS 파라미터로도 덮어쓰기 가능) ─────────────────────

CAN_NAME_DEFAULT = "can_piper"  # udev 고정
REALLY_ENABLE_DEFAULT = False   # 안전장치
ENABLE_ARM_MOTION_DEFAULT = True  # 그리퍼만 테스트할 때
MOVE_SPD_RATE_DEFAULT = 5       # 로봇팔 속도
TICK_HZ = 60.0                  # 900스텝=15초


GRASP_ROLL_DEG = -90.0


JOINT_HOLD_PHASES = ("wait", "place_ready", "place_home", "place_done")
PLACE_MARKER_PHASES = ("place_hover", "place_detect", "place_descend",
                       "place_release", "place_retreat")
CARTESIAN_PHASES = ("hover", "pre", "grasp", "lift", "place_lower",
                     "place_hover", "place_detect", "place_descend",
                     "place_release", "place_retreat")


FAULT_CODES = {1, 2, 3, 4, 7}

MODE_RESEND_TICKS = 30      # 피드백 모드가 다르면 MotionCtrl_2를 0.5초마다 재전송
STALL_TICKS = 30            # 관절 목표 미도달 + 0.5초간 정지면 같은 JointCtrl 재전송(첫 명령 유실 대비)
STALL_ERR_DEG = 1.0
STALL_MOVE_DEG = 0.2

# ── 조정 파라미터 끝 ─────────────────────────────────────────────────────────


def _body_to_native_xy(x_body, y_body):  # body_link XY → EndPoseCtrl 고유좌표. 팔이 90° 돌아 장착돼 회전이 필요
    """arm_node가 body_link 기준으로 계산한 XY를 EndPoseCtrl 고유좌표로 변환.
    팔이 베이스 기준 ARM_BASE_YAW_DEG만큼 돌아 장착돼 있어 이 회전이 필요하다."""
    a = math.radians(-ARM_BASE_YAW_DEG)
    return (x_body * math.cos(a) - y_body * math.sin(a),
            x_body * math.sin(a) + y_body * math.cos(a))


def _native_to_body_xy(x_native, y_native):  # 위의 역변환 — 팔 피드백을 body_link로 되돌린다
    """위 변환의 역 — EndPoseCtrl 피드백(native)을 body_link로 바꿔 /arm/ee_pose_body에 싣는다."""
    a = math.radians(ARM_BASE_YAW_DEG)
    return (x_native * math.cos(a) - y_native * math.sin(a),
            x_native * math.sin(a) + y_native * math.cos(a))


def _build_down_quats(place_pitch_deg=PLACE_PITCH_DEG):  # grasp/hover/place 3단계의 그리퍼 지향 쿼터니언 생성
    """grasp / hover / place 각 단계의 그리퍼 지향 쿼터니언을 만든다.
    셋 다 "아래를 본다"가 기본이고 단계별 pitch와 공통 롤만 다르다."""
    def _one(pitch_deg):  # pitch 하나에 대한 쿼터니언 — 아래보기 + 단계 pitch + 공통 롤
        q = quat_from_two_vec(TOOL_AXIS_LOCAL, GRIPPER_DOWN)
        if abs(pitch_deg) > 1e-6:
            q = _quat_mul(_quat_axis_angle(np.array([1.0, 0.0, 0.0]), pitch_deg), q)
            q = q / np.linalg.norm(q)
        if abs(GRASP_ROLL_DEG) > 1e-6:
            q = _quat_mul(q, _quat_axis_angle(TOOL_AXIS_LOCAL, GRASP_ROLL_DEG))
            q = q / np.linalg.norm(q)
        return q
    return _one(GRASP_PITCH_DEG), _one(HOVER_PITCH_DEG), _one(place_pitch_deg)


def _quat_to_rpy_deg(q):  # 쿼터니언 → (roll,pitch,yaw)[deg]. EndPoseCtrl이 오일러각을 받는다
    """쿼터니언(w,x,y,z) → (roll,pitch,yaw)[deg], R=Rz(yaw)@Ry(pitch)@Rx(roll) 가정.
    해가 180도 대칭으로 둘 나오지만 같은 자세를 가리키므로 어느 쪽이든 무방하다."""
    R = _quat_to_R(q)
    pitch = -math.asin(np.clip(R[2, 0], -1.0, 1.0))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class PiperDriverNode(Node):  # CAN 브릿지 — ROS 목표를 PiPER SDK 명령으로, 팔 피드백을 ROS 토픽으로
    def __init__(self):  # CAN 연결·팔 활성화·토픽 등록. really_enable=false면 명령은 안 보낸다
        super().__init__("piper_driver_node")

        self.declare_parameter("can_name", CAN_NAME_DEFAULT)
        self.declare_parameter("really_enable", REALLY_ENABLE_DEFAULT)
        self.declare_parameter("enable_arm_motion", ENABLE_ARM_MOTION_DEFAULT)
        self.declare_parameter("move_spd_rate_ctrl", MOVE_SPD_RATE_DEFAULT)
        self.declare_parameter("place_pitch_deg", PLACE_PITCH_DEG)
        can_name = self.get_parameter("can_name").value
        self.really_enable = bool(self.get_parameter("really_enable").value)
        self.enable_arm_motion = bool(self.get_parameter("enable_arm_motion").value)
        self.move_spd = int(self.get_parameter("move_spd_rate_ctrl").value)

        down_q, hover_q, place_q = _build_down_quats(
            float(self.get_parameter("place_pitch_deg").value))
        quat_by_phase = {ph: (place_q if ph in PLACE_MARKER_PHASES else down_q)
                         for ph in CARTESIAN_PHASES}
        quat_by_phase["hover"] = hover_q
        self._rpy_by_phase = {ph: _quat_to_rpy_deg(q) for ph, q in quat_by_phase.items()}

        self.arm_phase = "wait"
        self.joint_hold_target = np.array(SEARCH_Q, float)
        self.cartesian_target = None
        self._last_move_mode = None  # ModeCtrl 중복호출 방지(CAN 트래픽 절약)
        self._mode_sent_tick = 0
        self._last_endpose = None
        self._last_jointctrl = None
        self._jointctrl_tick = 0
        self._q_hist = deque(maxlen=STALL_TICKS + 1)  # 최근 관절각[deg] — 정지 판정용
        self._cmd_sent = 0; self._cmd_skipped = 0; self._cmd_resent = 0; self._mode_waits = 0
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
                self.get_logger().warn(
                    f"really_enable=True — 실제로 팔을 활성화합니다 (speed {self.move_spd}%). "
                    f"enable_arm_motion={self.enable_arm_motion} — True면 arm_phase 기본값('wait')만으로도 "
                    f"즉시 SEARCH_Q로 JointCtrl이 나갑니다(arm_node 없어도).")
                while not self.piper.EnablePiper():
                    time.sleep(0.01)
                self.piper.MotionCtrl_2(0x01, 0x01, self.move_spd, 0x00)  # ctrl_mode를 CAN 명령 모드로
                self._last_move_mode = 0x01
                self.piper.GripperTeachingPendantParamConfig(100, 70, 1)  # 그리퍼 최대행정 70mm 설정
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
        self._prof = declare_profile(self)
        self._prof_tick = self._prof.loop("tick") if self._prof else None
        self.create_timer(1.0 / TICK_HZ, self._tick_profiled if self._prof else self._tick)
        self.create_timer(5.0, self._publish_cmd_stats)
        self.get_logger().info(f"piper_driver_node 초기화 완료 (can={can_name}, really_enable={self.really_enable})")

    # ── 구독 콜백 ────────────────────────────────────────────────────────────
    def _on_arm_status(self, msg: String):  # arm_node의 phase 수신 — 바뀌면 명령 캐시를 비운다(중복생략 오작동 방지)
        phase = (msg.data.split(" ")[0].split("=")[-1]
                  if "phase=" in msg.data else msg.data)
        if phase != self.arm_phase:
            self._last_endpose = None
            self._last_jointctrl = None
            self.cartesian_target = None  # 이전 phase 목표가 새 phase 자세로 나가지 않게(10-01 place_hover 사고)
        self.arm_phase = phase

    def _on_cart_target(self, msg: Point):  # 직교 목표점 수신 — CARTESIAN_PHASES에서 MOVE L로 나간다
        self.cartesian_target = np.array([msg.x, msg.y, msg.z], float)

    def _on_joint_hold(self, msg: Float32MultiArray):  # 관절 목표 수신 — JOINT_HOLD_PHASES에서 MOVE J로 나간다
        self.joint_hold_target = unpack_joint_hold_target(msg.data)

    def _on_gripper_target(self, msg: Float32MultiArray):  # 그리퍼 목표 수신 → GripperCtrl. fault 중엔 무시
        if self.piper is None or not self.really_enable or self._fault_latched:
            return
        angle_mm, effort_nm = float(msg.data[0]), float(msg.data[1])
        self.piper.GripperCtrl(round(angle_mm * 1000), round(effort_nm * 1000), 0x01, 0)

    # ── 60Hz 틱 ─────────────────────────────────────────────────────────────
    def _tick_profiled(self):  # profile:=true일 때만 쓰는 _tick 래퍼. note에 phase/틱번호/송신여부
        p = self._prof_tick
        p.begin()
        sent0 = self._cmd_sent
        self._tick()
        p.note = f"{self.arm_phase};tick={self._tick_n};sent={int(self._cmd_sent != sent0)}"
        p.end()

    def _tick(self):  # 60Hz 심장박동 — 피드백 발행 → fault 확인 → phase별 모션 명령
        self._tick_n += 1
        self.pub_tick.publish(Int32(data=self._tick_n))
        if self.piper is None:
            return
        p = self._prof_tick

        self._publish_feedback()
        if p: p.lap("feedback")

        arm_st = self.piper.GetArmStatus().arm_status  # 바깥 arm_status는 래퍼, 진짜 상태값은 한 겹 더 안에 있음
        st = arm_st.arm_status
        if p: p.lap("status")
        if int(st) in FAULT_CODES:
            if not self._fault_latched:
                self.get_logger().error(f"★ 팔 고장/이상 상태 감지(arm_status={st}) — 새 모션 명령 중단")
            self._fault_latched = True
            return
        self._fault_latched = False

        if not self.really_enable or not self.enable_arm_motion:
            return

        if self.arm_phase in JOINT_HOLD_PHASES:
            if not self._ensure_move_mode(0x01, arm_st):  # MOVE J 전환 전 명령은 펌웨어가 버린다(10-01 실측)
                return
            q_deg = np.degrees(self.joint_hold_target)
            cmd = tuple(round(v * 1000) for v in q_deg)
            if cmd != self._last_jointctrl:
                self._last_jointctrl = cmd
                self._jointctrl_tick = self._tick_n
                self._cmd_sent += 1
                self.piper.JointCtrl(*cmd)
            elif self._joint_stalled(q_deg):
                self._jointctrl_tick = self._tick_n
                self._cmd_resent += 1
                self.get_logger().warn(f"관절 목표 미도달·정지 감지 — JointCtrl 재전송 (phase={self.arm_phase})")
                self.piper.JointCtrl(*cmd)
            else:
                self._cmd_skipped += 1
        elif self.arm_phase in CARTESIAN_PHASES and self.cartesian_target is not None:
            if not self._ensure_move_mode(0x02, arm_st):  # MOVE L
                return
            rx, ry, rz_body = self._rpy_by_phase[self.arm_phase]
            rz = rz_body - ARM_BASE_YAW_DEG  # yaw도 XY와 같은 회전보정을 받아야 한다
            x_body, y_body, z = self.cartesian_target
            x, y = _body_to_native_xy(x_body, y_body)
            cmd = (round(x * 1_000_000), round(y * 1_000_000), round(z * 1_000_000),
                   round(rx * 1000), round(ry * 1000), round(rz * 1000))
            if cmd != self._last_endpose:
                self._last_endpose = cmd
                self._cmd_sent += 1
                self.piper.EndPoseCtrl(*cmd)
            else:
                self._cmd_skipped += 1
        if p: p.lap("cmd")

    def _publish_cmd_stats(self):  # 5초마다 전송/중복생략 횟수 로그 — 명령이 실제로 나가는지 확인용
        self.get_logger().info(
            f"모션명령 전송 {self._cmd_sent}회 / 중복생략 {self._cmd_skipped}회 / "
            f"재전송 {self._cmd_resent}회 / 모드대기 {self._mode_waits}틱 (phase={self.arm_phase})")

    def _ensure_move_mode(self, move_mode, arm_st):  # 피드백이 CAN_CTRL+해당 모드면 True. 아니면 MotionCtrl_2 (재)전송 후 False
        if int(arm_st.ctrl_mode) == 0x01 and int(arm_st.mode_feed) == move_mode:
            self._last_move_mode = move_mode
            return True
        self._mode_waits += 1
        if self._last_move_mode != move_mode or self._tick_n - self._mode_sent_tick >= MODE_RESEND_TICKS:
            self.piper.MotionCtrl_2(0x01, move_mode, self.move_spd, 0x00)
            self._last_move_mode = move_mode
            self._mode_sent_tick = self._tick_n
        return False

    def _joint_stalled(self, q_target_deg):  # 목표와 1° 이상 차이 + 0.5초간 거의 안 움직임 + 직전 송신 후 0.5초 경과
        if len(self._q_hist) <= STALL_TICKS or self._tick_n - self._jointctrl_tick < STALL_TICKS:
            return False
        q_now, q_old = self._q_hist[-1], self._q_hist[0]
        return (np.max(np.abs(q_now - q_target_deg)) > STALL_ERR_DEG
                and np.max(np.abs(q_now - q_old)) < STALL_MOVE_DEG)

    def _publish_feedback(self):  # 팔의 엔드포즈·관절각·그리퍼 상태를 ROS 토픽으로 중계
        ep = self.piper.GetArmEndPoseMsgs().end_pose
        x_body, y_body = _native_to_body_xy(ep.X_axis / 1_000_000.0, ep.Y_axis / 1_000_000.0)
        self.pub_ee_pose.publish(Point(
            x=x_body, y=y_body, z=ep.Z_axis / 1_000_000.0))

        js = self.piper.GetArmJointMsgs().joint_state
        deg = [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(PIPER_JOINT_NAMES)
        msg.position = [math.radians(v / 1000.0) for v in deg]
        self._q_hist.append(np.array(deg) / 1000.0)
        hs = self.piper.GetArmHighSpdInfoMsgs()
        motors = [hs.motor_1, hs.motor_2, hs.motor_3, hs.motor_4, hs.motor_5, hs.motor_6]
        msg.velocity = [m.motor_speed / 1000.0 for m in motors]  # [rad/s] 0.001rad/s → SI. 관절축(위치미분 대비 비 0.89, 10-01 실측)
        msg.effort = [m.effort / 1000.0 for m in motors]  # [N·m] SDK가 전류×고정계수로 환산한 추정 토크
        self.pub_joint_states.publish(msg)

        gs = self.piper.GetArmGripperMsgs().gripper_state
        # 계약: [angle_mm, effort_Nm, foc_status] — piper_gripper_node.py와 맞출 것
        self.pub_gripper_feedback.publish(Float32MultiArray(
            data=[gs.grippers_angle / 1000.0, gs.grippers_effort / 1000.0,
                  float(gs.status_code)]))

    def destroy_node(self):  # 종료 시 DisableArm을 부르지 않는다 — 무동력 낙하 방지
        super().destroy_node()


def main():  # 노드 기동 진입점
    rclpy.init()
    node = PiperDriverNode()
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
