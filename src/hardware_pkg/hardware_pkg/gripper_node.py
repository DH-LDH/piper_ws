import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, Bool, String, Int32
from sensor_msgs.msg import JointState

from step22_common import (
    GRIP_JOINT_L, GRIP_JOINT_R, GRIP_STROKE_MAX, pack_grip_state,
)

# ── 사용자 조정 파라미터 ─────────────────────────────────────────────────────

# α-SMC 힘제어 (α=닫힘 정도, 개구부 = 2×(1-α)×GRIP_STROKE_MAX)
SMC_ENABLED   = True     # False면 GRIP_CLOSE_SCALE 고정 개루프로 닫기만 함
SMC_F_TARGET  = 60.0     # [N] 목표 협착력 (실측 파지력 대역 87~234N 기준, 필요마찰력 대비 36배 여유)
SMC_PHI       = 20.0     # [N] 경계층 — 정착 판정 폭이자 비례제어 대역
SMC_ALPHA_0   = 0.50     # α 초기값 (닫기 목표 스케일)
SMC_K_A       = 0.001    # [α/스텝] SMC 게인 — 그리퍼가 뻣뻣해(dF/dα≈1e4 N/α) 작게 잡아야 함
SMC_A_RATE    = 0.001    # [α/스텝] 1스텝 최대 변화량 (SMC_K_A와 같은 값으로 둬야 실제 상한 역할)
SMC_A_MIN     = 0.50     # α 하한 — 개구부가 물체 크기(OBJ_S) 밑으로는 안 벌어지게 하는 안전장치
SMC_A_MAX     = 1.00     # α 상한 (1.0 = 최대로 조임)
GRIP_CLOSE_SCALE = 0.50  # SMC_ENABLED=False일 때 쓰는 고정 α

# 접촉 판정 / 종료
CONTACT_F_MIN       = 1.0   # [N] 이 힘 이상이면 "닿았다"로 카운트
GRIP_JUDGE_BY_FORCE = True  # 접촉 판정을 실접촉력으로 (False면 타임아웃까지 대기)
GRIP_WARMUP_N       = 2     # 그립 시작 후 이 스텝까지는 판정/SMC 유예
GRIP_F_CONTACT_N    = 5     # 연속 이 스텝 이상 접촉이면 "접촉 감지" 확정
GRIP_SETTLE_N       = 20    # 접촉 감지 후 이 스텝 이상 지나야 정착 완료
GRIP_SMC_SETTLE     = True  # 정착 조건에 |목표-실측|<SMC_PHI 도 요구
GRIP_TIMEOUT        = 150   # [스텝] 끝내 접촉 못 찾으면 포기(헛집음 보고)

# ── 조정 파라미터 끝 ─────────────────────────────────────────────────────────


# PiPER 그리퍼: joint7/joint8 완전 독립 prismatic, joint8=-joint7 미러링.
# joint7=0이 닫힘, joint7=GRIP_STROKE_MAX가 열림(실측 확정) — _publish_targets에서 (1-alpha)로 매핑.
def _grip_base_targets(names):
    return np.array([1.0 if nm == GRIP_JOINT_L else -1.0 for nm in names], float)


class GripperNode(Node):
    def __init__(self):
        super().__init__("gripper_node")

        self.grip_names = None        # /mir/joint_states에서 1회 추출
        self.active = False
        self.arm_step = 0
        self.F_con = None
        self.grip_base_targets = None
        self.grip_alpha = float(SMC_ALPHA_0)
        self.grip_contact = False
        self.grip_contact_run = 0
        self.grip_contact_at_step = None
        self.cf_peak = 0.0
        self.cf_first_step = None
        self.n_samples = 0
        self.done_sent = False  # /gripper/done 중복 발행 방지
        self.holding = False    # 정착 후에도 α-SMC를 계속 돌려 접촉을 유지하는 모드

        self.pub_gripper_cmd = self.create_publisher(JointState, "/mir/gripper_cmd", 10)
        self.pub_done = self.create_publisher(Bool, "/gripper/done", 1)
        # 실물(piper_gripper_node)과 같은 계약 — arm_node의 그리퍼 기반 파지 판정이
        # sim/실물 어느 쪽에서도 같은 코드로 돌게 한다(verify_mode=gripper/both).
        self.pub_grip_state = self.create_publisher(Float32MultiArray, "/gripper/grip_state", 10)

        self.pub_status = self.create_publisher(String, "/gripper/status", 5)

        self.F_pads = None  # pad별 접촉력(비대칭 잼 확인용)
        self.stroke_mm = None  # 실개구부[mm] = joint7 - joint8(미러링이라 2×joint7)

        self.create_subscription(JointState, "/mir/joint_states", self._on_joint_states, 10)
        self.create_subscription(Float32, "/mir/contact_force", self._on_contact_force, 10)
        self.create_subscription(Float32MultiArray, "/mir/contact_force_pads", self._on_contact_force_pads, 10)
        self.create_subscription(Bool, "/gripper/cmd", self._on_gripper_cmd, 1)

        self.create_subscription(Int32, "/plant/tick", self._tick, 20)
        self.create_timer(2.0, self._publish_status)
        self.get_logger().info("gripper_node 초기화 완료 — /gripper/cmd 대기")

    def _on_joint_states(self, msg: JointState):
        if self.grip_names is None:
            names = [n for n in (GRIP_JOINT_L, GRIP_JOINT_R) if n in msg.name]
            if len(names) == 2:
                self.grip_names = names
                self.grip_base_targets = _grip_base_targets(names)
                self.get_logger().info(f"그립 관절 {len(names)}개 확인: {names}")
        if GRIP_JOINT_L in msg.name and GRIP_JOINT_R in msg.name:
            pl = float(msg.position[msg.name.index(GRIP_JOINT_L)])
            pr = float(msg.position[msg.name.index(GRIP_JOINT_R)])
            self.stroke_mm = abs(pl - pr) * 1000.0
            self.pub_grip_state.publish(Float32MultiArray(data=pack_grip_state(
                self.stroke_mm, self.F_con if self.F_con is not None else 0.0,
                self.grip_contact, self.holding, self.active)))

    def _on_contact_force(self, msg: Float32):
        self.F_con = float(msg.data)

    def _on_contact_force_pads(self, msg: Float32MultiArray):
        self.F_pads = [float(v) for v in msg.data]

    def _on_gripper_cmd(self, msg: Bool):
        if not msg.data:
            # 재시도용 릴리즈 — SMC/유지모드 전부 중단하고 오픈
            self.active = False
            self.holding = False
            if self.grip_base_targets is not None:
                self.grip_alpha = 0.0
                self._publish_targets(0.0)
                print("  [gripper] 릴리즈 — 그리퍼 오픈")
            return
        if self.active: return
        if self.grip_base_targets is None:
            print("  ★ gripper_node: 그립 관절 목록 미확인 — /mir/joint_states 대기 필요")
            return
        self.active = True
        self.arm_step = 0
        self.grip_alpha = float(SMC_ALPHA_0 if SMC_ENABLED else GRIP_CLOSE_SCALE)
        self.grip_contact = False
        self.grip_contact_run = 0
        self.grip_contact_at_step = None
        self.cf_peak = 0.0
        self.cf_first_step = None
        self.n_samples = 0
        self.done_sent = False
        self.holding = False
        self._publish_targets(self.grip_alpha)

    def _publish_targets(self, alpha):
        tgt = self.grip_base_targets * (1.0 - alpha) * GRIP_STROKE_MAX
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.grip_names
        msg.position = [float(v) for v in tgt]
        self.pub_gripper_cmd.publish(msg)

    def _smc_alpha_command(self, F_contact):
        s_surf = SMC_F_TARGET - float(F_contact)
        sat = float(np.clip(s_surf / SMC_PHI, -1.0, 1.0))
        da = float(np.clip(SMC_K_A * sat, -SMC_A_RATE, SMC_A_RATE))
        if self.holding and da < 0.0:
            # 정착 후(리프트 포함)엔 "풀기" 방향을 막는다 — 물체 무게로 힘이 튀는 걸
            # "너무 세게 쥐었다"로 오인해 풀면 슬립이 심해진다. 조이는 방향은 계속 허용.
            da = 0.0
        self.grip_alpha = float(np.clip(self.grip_alpha + da, SMC_A_MIN, SMC_A_MAX))
        return self.grip_alpha

    def _tick(self, _msg: Int32):
        if not self.active: return
        self.arm_step += 1

        F_con = self.F_con
        if F_con is not None:
            self.n_samples += 1
            if F_con > self.cf_peak: self.cf_peak = F_con
            if F_con >= CONTACT_F_MIN and self.cf_first_step is None:
                self.cf_first_step = self.arm_step
                pads_s = [f"{v:.1f}" for v in self.F_pads] if self.F_pads else ["?"]
                print(f"    ★ [실접촉력] 최초 접촉 (스텝 {self.arm_step}) F_contact={F_con:.2f}N "
                      f"pad별=[{', '.join(pads_s)}]")

        if SMC_ENABLED and F_con is not None and self.arm_step > GRIP_WARMUP_N:
            a_new = self._smc_alpha_command(F_con)
            self._publish_targets(a_new)

        if GRIP_JUDGE_BY_FORCE and F_con is not None:
            hit = (F_con >= CONTACT_F_MIN)
        else:
            hit = False  # 순변위 폴백 미포팅 — F_con 없으면 타임아웃까지 대기
        past_warmup = self.arm_step > GRIP_WARMUP_N
        if past_warmup and hit:
            self.grip_contact_run += 1
        else:
            self.grip_contact_run = 0
        if self.grip_contact_run >= GRIP_F_CONTACT_N and not self.grip_contact:
            self.grip_contact = True
            self.grip_contact_at_step = self.arm_step
            pads_s = [f"{v:.1f}" for v in self.F_pads] if self.F_pads else ["?"]
            print(f"    [접촉 감지] (스텝 {self.arm_step}) F_contact={F_con:.2f}N "
                  f"pad별=[{', '.join(pads_s)}]")

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
                print(f"  [gripper] ★ 접촉 미감지 — 헛집음 가능성")
            _fin = self.F_con
            pads_s = [f"{v:.1f}" for v in self.F_pads] if self.F_pads else ["?"]
            print(f"    ★★ [실접촉력 요약] 샘플 {self.n_samples}  peak {self.cf_peak:.2f}N  "
                  f"최종 {_fin if _fin is None else round(_fin,2)}N  "
                  f"최종pad별=[{', '.join(pads_s)}]")
            self.pub_done.publish(Bool(data=self.grip_contact))
            if settle_ok:
                # 정착됐다고 α-SMC를 끄지 않는다 — 리프트 중에도 계속 F_con을 보며
                # α를 조정해 접촉을 유지한다(꺼두면 한쪽 pad가 빠지는 문제 확인됨).
                self.holding = True
                print(f"    [gripper] 접촉 유지 모드 진입 — 리프트 중에도 α-SMC 계속 작동")
            else:
                self.active = False  # 접촉 자체를 못 찾은 타임아웃 — 포기하고 정지

    def _publish_status(self):
        self.pub_status.publish(String(
            data=f"alpha={self.grip_alpha:.3f} F_contact="
                 f"{self.F_con if self.F_con is not None else -1:.2f} "
                 f"contact={self.grip_contact} active={self.active}"))


def main():
    rclpy.init()
    node = GripperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
