import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Bool, String, Int32

from step22_common import pack_grip_state

GRIPPER_STROKE_MAX_M = 0.06  # 그리퍼 오픈 최대 

SMC_ENABLED   = True
SMC_F_TARGET  = 1.5     # [N·m] 목표 토크 
SMC_PHI       = 0.5     # [N·m] 경계층
SMC_ALPHA_0   = 0.0     # 완전히 열린 상태
SMC_K_A       = 0.01    # SMC 게인 
SMC_A_RATE    = 0.01
SMC_A_MIN     = 0.30    # α 하한 
SMC_A_MAX     = 1.00    # α 상한 
GRIP_CLOSE_SCALE = 0.50  # SMC_ENABLED=False일 때 고정 α
GRIPPER_EFFORT_LIMIT_NM = 2.0  # [N·m]  안전 상한

CONTACT_F_MIN       = 0.3   # [N·m] 감지 토크 
GRIP_JUDGE_BY_FORCE = True
GRIP_WARMUP_N       = 2     # 그리퍼 웜업
GRIP_F_CONTACT_N    = 5
GRIP_SETTLE_N       = 20
GRIP_SMC_SETTLE     = True
GRIP_TIMEOUT        = 150

# ── 조정 파라미터 끝 ─────────────────────────────────────────────────────────


class PiperGripperNode(Node):  # 그리퍼 힘제어 — α-SMC로 접촉력을 목표값에 맞춰 닫는다
    def __init__(self):  # 상태 변수 초기화 + 토픽 등록. 기동 3초 뒤 한 번 열어둔다
        super().__init__("piper_gripper_node")

        self.active = False
        self.arm_step = 0
        self.F_con = None       # grippers_effort 최신값(N·m)
        self.foc_status = 0
        self.grip_alpha = float(SMC_ALPHA_0)
        self.grip_contact = False
        self.grip_contact_run = 0
        self.grip_contact_at_step = None
        self.cf_peak = 0.0
        self.n_samples = 0
        self.done_sent = False
        self.holding = False
        self.stroke_mm = None   # 그리퍼 실개구부[mm] — arm_node의 파지 판정 입력

        self.pub_target = self.create_publisher(Float32MultiArray, "/piper/gripper_target_cmd", 10)
        self.pub_done = self.create_publisher(Bool, "/gripper/done", 1)
        # 파지 성공 판정용 상시 발행 
        self.pub_grip_state = self.create_publisher(Float32MultiArray, "/gripper/grip_state", 10)
        self.pub_status = self.create_publisher(String, "/gripper/status", 5)

        self.create_subscription(Float32MultiArray, "/piper/gripper_feedback", self._on_feedback, 10)
        self.create_subscription(Bool, "/gripper/cmd", self._on_gripper_cmd, 1)
        self.create_subscription(Int32, "/plant/tick", self._tick, 20)
        self.create_timer(2.0, self._publish_status)
        # 기동 시 한 번 열어둔다 — 안 하면 이전 파지에서 닫힌 상태로 시작한다.
        self._open_timer = self.create_timer(3.0, self._open_on_startup)
        self.get_logger().info("piper_gripper_node 초기화 완료 — /gripper/cmd 대기")

    def _open_on_startup(self):  # 기동 1회 오픈
        self._open_timer.cancel()
        self.grip_alpha = 0.0
        self._publish_target(0.0)
        self.get_logger().info("기동 시 그리퍼 오픈")

    def _on_feedback(self, msg: Float32MultiArray):  # 개구부·접촉력 수신(부호 제거) → /gripper/grip_state로 재발행
        angle_mm, effort_nm, foc = msg.data
        self.F_con = abs(float(effort_nm))
        self.foc_status = int(foc)
        self.stroke_mm = float(angle_mm)
        self.pub_grip_state.publish(Float32MultiArray(data=pack_grip_state(
            self.stroke_mm, self.F_con, self.grip_contact, self.holding, self.active)))

    def _publish_target(self, alpha):  # α(0=열림,1=닫힘) → 실제 개구부[mm] 명령으로 변환해 발행
        angle_mm = (1.0 - alpha) * GRIPPER_STROKE_MAX_M * 1000.0
        self.pub_target.publish(Float32MultiArray(
            data=[float(angle_mm), float(GRIPPER_EFFORT_LIMIT_NM)]))

    def _on_gripper_cmd(self, msg: Bool):  # True=파지 시작(상태 리셋), False=릴리즈(즉시 오픈)
        if not msg.data:
            self.active = False
            self.holding = False
            self.grip_alpha = 0.0
            self._publish_target(0.0)
            print("  [gripper] 릴리즈 — 그리퍼 오픈")
            return
        if self.active: return
        self.active = True
        self.arm_step = 0
        self.grip_alpha = float(SMC_ALPHA_0 if SMC_ENABLED else GRIP_CLOSE_SCALE)
        self.grip_contact = False
        self.grip_contact_run = 0
        self.grip_contact_at_step = None
        self.cf_peak = 0.0
        self.n_samples = 0
        self.done_sent = False
        self.holding = False
        self._publish_target(self.grip_alpha)

    def _smc_alpha_command(self, F_contact):  # α-SMC 1스텝 — 목표력과의 오차를 경계층으로 포화시켜 α를 갱신
        s_surf = SMC_F_TARGET - float(F_contact)    #얼마나 더 쥘지 목표토크(1.5)-현재 토크
        sat = float(np.clip(s_surf / SMC_PHI, -1.0, 1.0))   #오차 -1~+1로 정규화
        da = float(np.clip(SMC_K_A * sat, -SMC_A_RATE, SMC_A_RATE))
        if self.holding and da < 0.0:
            da = 0.0
        self.grip_alpha = float(np.clip(self.grip_alpha + da, SMC_A_MIN, SMC_A_MAX))    #계속 잡기
        return self.grip_alpha

    def _tick(self, _msg: Int32):  # 60Hz — SMC 갱신 → 접촉 판정 → 안정화/타임아웃이면 done 발행
        if not self.active: return
        self.arm_step += 1

        F_con = self.F_con
        if F_con is not None:
            self.n_samples += 1
            if F_con > self.cf_peak: self.cf_peak = F_con

        if SMC_ENABLED and F_con is not None and self.arm_step > GRIP_WARMUP_N:
            a_new = self._smc_alpha_command(F_con)
            self._publish_target(a_new)

        hit = (GRIP_JUDGE_BY_FORCE and F_con is not None and F_con >= CONTACT_F_MIN)
        past_warmup = self.arm_step > GRIP_WARMUP_N
        if past_warmup and hit:
            self.grip_contact_run += 1
        else:
            self.grip_contact_run = 0
        if self.grip_contact_run >= GRIP_F_CONTACT_N and not self.grip_contact:
            self.grip_contact = True
            self.grip_contact_at_step = self.arm_step
            print(f"    [접촉 감지] (스텝 {self.arm_step}) F_contact={F_con:.2f}N·m")

        smc_settled = True
        if GRIP_SMC_SETTLE and SMC_ENABLED:
            smc_settled = (F_con is not None and abs(SMC_F_TARGET - F_con) < SMC_PHI)
        settle_ok = (self.grip_contact
                     and self.arm_step >= self.grip_contact_at_step + GRIP_SETTLE_N
                     and smc_settled)
        timeout = self.arm_step >= GRIP_TIMEOUT

        if (settle_ok or timeout) and not self.done_sent:
            self.done_sent = True
            if not self.grip_contact:
                print("  [gripper] ★ 접촉 미감지 — 헛집음 가능성")
            _fin = self.F_con
            _stk = self.stroke_mm
            print(f"    ★★ [실접촉토크 요약] 샘플 {self.n_samples}  peak {self.cf_peak:.2f}N·m  "
                  f"최종 {_fin if _fin is None else round(_fin,2)}N·m  "
                  f"개구부 {'-' if _stk is None else f'{_stk:.1f}mm'}  "
                  f"foc_status=0b{self.foc_status:08b}")
            self.pub_done.publish(Bool(data=self.grip_contact))
            if settle_ok:
                self.holding = True
                print("    [gripper] 접촉 유지 모드 진입 — 리프트 중에도 α-SMC 계속 작동")
            else:
                self.active = False

    def _publish_status(self):  # 2초마다 α·개구부·접촉력을 /gripper/status로 발행(진단용)
        self.pub_status.publish(String(
            data=f"alpha={self.grip_alpha:.3f} stroke="
                 f"{self.stroke_mm if self.stroke_mm is not None else -1:.1f}mm F_contact="
                 f"{self.F_con if self.F_con is not None else -1:.2f} "
                 f"contact={self.grip_contact} active={self.active}"))


def main():  # 노드 기동 진입점
    rclpy.init()
    node = PiperGripperNode()
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
