"""B단계 테스트 하네스 — 비전·arm_node 없이 웨이포인트 궤적을 실행하고 지령/실측을 CSV로 남긴다.

mode: joint_direct(웨이포인트만 IK → 관절공간 스플라인) / joint_ik(직교 스플라인 → 매 틱 IK) /
      movej_once(끝점 관절목표 1회 송신, 펌웨어 MOVE J 보간 — 기준선) /
      movel(기존 방식 기준선: MOVE L 목표 정속 스트리밍) / fk_check(명령 없음, FK vs 펌웨어 EndPose만 기록)
dry_run:=true(기본)면 계획·검사·CSV만 하고 명령은 보내지 않는다.
"""
import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import csv
import math
import os
import time

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String, Int32, Bool
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

import piper_kin as K
from step22_common import pack_joint_hold_target

TICK_HZ = 60.0
DOWN_RPY_NATIVE = (180.0, 0.0, -180.0)  # 그리퍼가 아래를 보는 자세(펌웨어 EndPose 기준)
IK_POS_TOL = 1e-4                       # [m] IK 잔차 한계
IK_ROT_TOL = math.radians(0.1)
START_TOL_DEG = 0.5                     # 시작 자세 도달 판정
START_TIMEOUT_S = 40.0
ARRIVE_TOL_MM = 0.5                     # 궤적 끝 후 EE가 끝점 이 안에 0.5 s 머물면 도착
ARRIVE_TIMEOUT_S = 20.0
POST_ARRIVE_S = 1.0                     # 도착 후 추가 기록(잔류 흔들림)
LOG_ROOT = os.environ.get("PIPER_LOG_DIR", os.path.expanduser("~/IsaacSim_ranger_piper/logs"))


def min_jerk(tau):  # 최소저크 시간 스케일 s(τ), τ∈[0,1]
    tau = np.clip(tau, 0.0, 1.0)
    return 10 * tau**3 - 15 * tau**4 + 6 * tau**5


class TrajTestNode(Node):
    def __init__(self):
        super().__init__("traj_test_node")
        from ament_index_python.packages import get_package_share_directory
        default_wp = os.path.join(get_package_share_directory("control_pkg"), "config", "traj_waypoints.yaml")
        P = lambda n, v: self.declare_parameter(n, v).value
        self.wp_file = P("waypoints_file", default_wp)
        self.set_name = P("set", "line")
        self.mode = P("mode", "joint_direct")
        self.rate_div = int(P("rate_div", 1))            # 1=60Hz, 3=20Hz 송신
        self.v_max = float(P("v_max", 0.02))             # [m/s] 직교 궤적 최고속도
        self.w_max = float(P("w_max", 0.15))             # [rad/s] joint_direct 최고 관절속도
        self.rot_w_max = float(P("rot_w_max", 0.3))      # [rad/s] 자세 회전 최고속도
        self.movel_speed = float(P("movel_speed", 0.02))  # [m/s] movel 목표 전진속도 — 펌웨어 5%(≈21 mm/s)와 맞춰 스트리밍이 끝까지 지속되게
        self.max_step_deg = float(P("max_step_deg", 0.6))  # 틱당 관절 변화 상한(60Hz면 36°/s)
        self.dry_run = bool(P("dry_run", True))
        self.repeat = int(P("repeat", 1))
        self.confirm = bool(P("confirm", True))  # 시작점 도착 후 step_confirm.py(Enter) 승인을 받고 실행
        self._confirmed = False

        self.kin = K.PiperKin()
        self.q_meas = None; self.qd_meas = None; self.eff_meas = None; self.ee_native = None
        self.tick = 0
        self.finished = False
        self.pub_status = self.create_publisher(String, "/arm/status", 10)
        self.pub_joint = self.create_publisher(Float32MultiArray, "/arm/joint_hold_target", 10)
        self.pub_cart = self.create_publisher(Float32MultiArray, "/arm/traj_cart_pose", 10)
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.create_subscription(Float32MultiArray, "/arm/ee_pose_native", self._on_ee, 10)
        self.create_subscription(Bool, "/arm/step_confirm", self._on_confirm, 10)
        self.pub_step_wait = self.create_publisher(
            String, "/arm/step_wait", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        tag = time.strftime("%Y%m%d_%H%M%S") + f"_{self.set_name}_{self.mode}_r{self.rate_div}" + ("_dry" if self.dry_run else "")
        self.log_dir = os.path.join(LOG_ROOT, "traj", tag)
        os.makedirs(self.log_dir, exist_ok=True)

        self.plan = None if self.mode == "fk_check" else self._make_plan()
        self.state = "fk" if self.mode == "fk_check" else ("dry" if self.dry_run else "wait_fb")
        self._open_log()
        self.create_subscription(Int32, "/plant/tick", self._on_tick, 20)
        self.get_logger().info(f"traj_test_node: set={self.set_name} mode={self.mode} rate_div={self.rate_div} "
                               f"dry_run={self.dry_run} → {self.log_dir}")

    # ── 피드백 ─────────────────────────────────────────────────────────────
    def _on_js(self, msg):
        self.q_meas = np.array(msg.position[:6])
        self.qd_meas = np.array(msg.velocity[:6]) if len(msg.velocity) >= 6 else np.full(6, np.nan)
        self.eff_meas = np.array(msg.effort[:6]) if len(msg.effort) >= 6 else np.full(6, np.nan)

    def _on_ee(self, msg):
        self.ee_native = np.array(msg.data, float)

    def _on_confirm(self, msg):
        if msg.data and self.state == "wait_confirm":
            self._confirmed = True

    # ── 계획 ───────────────────────────────────────────────────────────────
    def _waypoints(self):  # YAML → (위치 native[n,3], 회전[n])
        with open(self.wp_file) as f:
            pts = np.array(yaml.safe_load(f)["sets"][self.set_name]["points"], float)
        R_down = Rotation.from_matrix(K.rpy_deg_to_R(DOWN_RPY_NATIVE))
        pos = np.array([K.body_to_native(p[:3]) for p in pts])
        rots = Rotation.concatenate([R_down * Rotation.from_euler("xyz", p[3:], degrees=True) for p in pts])
        return pos, rots

    def _make_plan(self):  # 전 구간 60Hz 샘플: q[N,6], p[N,3], R[N], send[N] (송신 틱 표시)
        pos, rots = self._waypoints()
        q0 = self._ik_chain_start(pos[0], rots[0])
        seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        u = np.r_[0.0, np.cumsum(seg)]
        ang = np.r_[0.0, np.cumsum([(rots[i].inv() * rots[i + 1]).magnitude() for i in range(len(rots) - 1)])]
        dt = 1.0 / TICK_HZ

        if self.mode == "movel":  # arm_node._move_l과 같은 정속 직선 전진(가감속 없음), 자세는 구간별 선형 slerp
            step = self.movel_speed * dt
            ps, Rs = [], []
            for i in range(len(pos) - 1):
                n = max(1, int(math.ceil(seg[i] / step)))
                sl = Slerp([0, 1], Rotation.concatenate([rots[i], rots[i + 1]]))
                for k in range(n):
                    a = (k + 1) / n
                    ps.append(pos[i] + a * (pos[i + 1] - pos[i])); Rs.append(sl([a])[0])
            p_arr = np.array(ps); R_list = Rs
            q_arr = self._ik_track(p_arr, R_list, q0)
        elif self.mode == "joint_ik":  # 직교 스플라인 + 최소저크, 매 샘플 IK
            T = max(1.875 * u[-1] / self.v_max, 1.875 * ang[-1] / self.rot_w_max, 1.0)
            s = min_jerk(np.arange(0, T + dt, dt) / T)
            pspl = CubicSpline(u, pos, bc_type="natural") if len(pos) > 2 else None
            ui = s * u[-1]
            p_arr = pspl(ui) if pspl is not None else pos[0] + np.outer(s, pos[-1] - pos[0])
            R_list = list(Slerp(u, rots)(ui))
            q_arr = self._ik_track(p_arr, R_list, q0)
        else:  # joint_direct / movej_once: 웨이포인트만 IK, 관절공간 스플라인 + 최소저크
            qw = [q0]
            for i in range(1, len(pos)):
                qw.append(self._ik_checked(pos[i], rots[i].as_matrix(), qw[-1], f"wp{i}"))
            qw = np.array(qw)
            uq = np.r_[0.0, np.cumsum(np.max(np.abs(np.diff(qw, axis=0)), axis=1))]
            T = max(1.875 * uq[-1] / self.w_max, 1.0)
            s = min_jerk(np.arange(0, T + dt, dt) / T)
            qspl = CubicSpline(uq, qw, bc_type="natural") if len(qw) > 2 else None
            q_arr = qspl(s * uq[-1]) if qspl is not None else qw[0] + np.outer(s, qw[-1] - qw[0])
            fk = [self.kin.fk(q) for q in q_arr]
            p_arr = np.array([f[0] for f in fk]); R_list = [Rotation.from_matrix(f[1]) for f in fk]

        send = (np.arange(len(q_arr)) % self.rate_div == 0)
        send[-1] = True
        if self.mode == "movej_once":  # 끝점 관절목표를 처음 한 번만 송신 — 보간은 펌웨어 MOVE J(place_ready 방식)
            send[:] = False
            send[0] = True
        plan = dict(q=q_arr, p=p_arr, R=R_list, send=send)
        self._check_plan(plan)
        return plan

    def _ik_chain_start(self, p, R):  # 시작 웨이포인트 IK — 'hover'(pick 시 실측) 관절자세에서 출발
        seed = np.radians([0, 119.4, -80.1, 0, 54.1, 0])
        return self._ik_checked(p, R.as_matrix(), seed, "start")

    def _ik_checked(self, p, R, seed, label):
        q, pe, re, ok = self.kin.ik(p, R, seed)
        if pe > IK_POS_TOL or re > IK_ROT_TOL or not ok:
            raise RuntimeError(f"IK 실패({label}): 위치오차 {pe*1000:.2f} mm, 자세오차 {math.degrees(re):.2f}°, "
                               f"한계내={ok}, q={np.degrees(q).round(1)}")
        return q

    def _ik_track(self, p_arr, R_list, q0):  # 직전 해를 seed로 매 샘플 IK
        q_arr, q = [], q0
        for i, (p, R) in enumerate(zip(p_arr, R_list)):
            q = self._ik_checked(p, R.as_matrix(), q, f"sample{i}")
            q_arr.append(q)
        return np.array(q_arr)

    def _check_plan(self, plan):  # 관절 한계·틱당 변화량·해 연속성 검사 — 하나라도 넘으면 실행 거부
        q = plan["q"]
        dq = np.degrees(np.abs(np.diff(q, axis=0)))
        step_max = dq.max() if len(dq) else 0.0
        margin_lo = np.degrees(q - self.kin.q_min).min(axis=0)
        margin_hi = np.degrees(self.kin.q_max - q).min(axis=0)
        T = len(q) / TICK_HZ
        path = np.sum(np.linalg.norm(np.diff(plan["p"], axis=0), axis=1))
        v = np.linalg.norm(np.diff(plan["p"], axis=0), axis=1) * TICK_HZ
        self.plan_summary = (f"샘플 {len(q)} ({T:.1f} s), 송신 {int(plan['send'].sum())}회, EE 경로 {path*1000:.0f} mm, "
                             f"EE 최고속도 {v.max()*1000:.1f} mm/s\n"
                             f"  틱당 관절 변화 최대 {step_max:.3f}° (한계 {self.max_step_deg}°)\n"
                             f"  관절 한계 여유(하한) {margin_lo.round(1)}°\n  관절 한계 여유(상한) {margin_hi.round(1)}°")
        self.get_logger().info("[계획]\n  " + self.plan_summary)
        if step_max > self.max_step_deg:
            raise RuntimeError(f"틱당 관절 변화 {step_max:.3f}° > {self.max_step_deg}° — 속도를 낮추거나 경로를 바꿀 것")
        if margin_lo.min() < 0 or margin_hi.min() < 0:
            raise RuntimeError("관절 한계 초과")

    # ── 실행 ───────────────────────────────────────────────────────────────
    def _status(self, phase):
        self.pub_status.publish(String(data=f"phase={phase} arm_step={self.tick}"))

    def _on_tick(self, msg):
        self.tick += 1
        st = self.state
        if st == "fk":
            self._log_row("fk", None, None, None, False)
        elif st == "wait_fb":
            if self.q_meas is not None:
                self.state = "goto_start"; self.t_state = time.time()
                self.get_logger().info(f"시작 자세로 MOVE J: {np.degrees(self.plan['q'][0]).round(1)}°")
        elif st == "goto_start":  # 첫 점으로 한 번에 MOVE J, 도달·정지 확인
            q0 = self.plan["q"][0]
            self._status("traj_joint")
            self.pub_joint.publish(Float32MultiArray(data=pack_joint_hold_target(q0)))
            err = np.degrees(np.max(np.abs(self.q_meas - q0)))
            self._log_row("goto", q0, None, None, False)
            if err < START_TOL_DEG and np.nanmax(np.abs(self.qd_meas)) < 0.01:
                self.get_logger().info(f"시작 자세 도달(오차 {err:.2f}°)")
                if self.confirm:
                    self.state = "wait_confirm"; self._confirmed = False
                    self.pub_step_wait.publish(String(data=f"traj {self.set_name}/{self.mode}"))
                    self.get_logger().warn("★ 실행 대기 — 다른 터미널의 step_confirm.py에서 Enter를 누르면 궤적을 실행합니다")
                else:
                    self._start_run()
            elif time.time() - self.t_state > START_TIMEOUT_S:
                self.get_logger().error(f"시작 자세 도달 실패(잔여 {err:.2f}°) — 중단"); self.state = "done"
                self.finished = True
        elif st == "wait_confirm":  # 시작점 유지하며 승인 대기
            self._status("traj_joint")
            self.pub_joint.publish(Float32MultiArray(data=pack_joint_hold_target(self.plan["q"][0])))
            self._log_row("goto", self.plan["q"][0], None, None, False)
            if self._confirmed:
                self._start_run()
        elif st == "run":
            k = self.k
            q, p, R, send = self.plan["q"][k], self.plan["p"][k], self.plan["R"][k], self.plan["send"][k]
            if self.mode == "movel":
                self._status("traj_cart")
                if send:
                    rpy = K.R_to_rpy_deg(R.as_matrix())
                    self.pub_cart.publish(Float32MultiArray(data=[*map(float, K.native_to_body(p)), *map(float, rpy)]))
            elif self.mode == "movej_once":
                self._status("traj_joint")
                q = self.plan["q"][-1]  # 검사 통과한 끝점 — 펌웨어 속도제한(5%)으로 이동하므로 틱당 클램프 생략
                if send:
                    self.pub_joint.publish(Float32MultiArray(data=pack_joint_hold_target(
                        np.clip(q, self.kin.q_min, self.kin.q_max))))
            else:
                self._status("traj_joint")
                if send:
                    self.pub_joint.publish(Float32MultiArray(data=pack_joint_hold_target(self._clamp_step(q))))
            self._log_row("run", q, p, R, bool(send))
            self.k += 1
            if self.k >= len(self.plan["q"]):
                self.state = "settle"; self.t_state = time.time(); self.in_tol = 0
        elif st == "settle":  # 마지막 목표 유지, EE가 끝점에 도착·정지할 때까지 기록
            k = len(self.plan["q"]) - 1
            self._status("traj_cart" if self.mode == "movel" else "traj_joint")
            self._log_row("settle", self.plan["q"][k], self.plan["p"][k], self.plan["R"][k], False)
            dist = (np.linalg.norm(self.ee_native[:3] - self.plan["p"][k]) * 1000
                    if self.ee_native is not None else np.inf)
            self.in_tol = self.in_tol + 1 if dist < ARRIVE_TOL_MM else 0
            if self.in_tol == 30 and not hasattr(self, "t_arrive"):
                self.t_arrive = time.time()
                self.get_logger().info(f"끝점 도착 (궤적 종료 후 {self.t_arrive - self.t_state - 0.5:.2f} s, 잔여 {dist:.2f} mm)")
            timeout = time.time() - self.t_state > ARRIVE_TIMEOUT_S
            if timeout and not hasattr(self, "t_arrive"):
                self.get_logger().warn(f"끝점 미도달 — {ARRIVE_TIMEOUT_S:.0f} s 초과(잔여 {dist:.1f} mm), 기록 종료")
            if timeout or (hasattr(self, "t_arrive") and time.time() - self.t_arrive > POST_ARRIVE_S):
                if hasattr(self, "t_arrive"):
                    del self.t_arrive
                if self.run_n < self.repeat:
                    self.state = "goto_start"; self.t_state = time.time()
                else:
                    self.get_logger().info(f"완료 → {self.log_dir}/traj.csv"); self.state = "done"
                    self._f.flush()
                    self.finished = True

    def _start_run(self):
        self.state = "run"; self.k = 0; self.run_n = getattr(self, "run_n", 0) + 1
        self._q_sent = self.q_meas.copy()  # 클램프 기준을 실측 시작 자세로
        self.get_logger().info(f"궤적 실행 {self.run_n}/{self.repeat}")

    def _clamp_step(self, q):  # 마지막 안전망: 직전 송신 대비 틱당 변화량·관절 한계 클램프
        lim = math.radians(self.max_step_deg) * self.rate_div
        prev = self._q_sent
        q = prev + np.clip(q - prev, -lim, lim)
        q = np.clip(q, self.kin.q_min, self.kin.q_max)
        self._q_sent = q
        return q

    # ── 기록 ───────────────────────────────────────────────────────────────
    def _open_log(self):
        self._f = open(os.path.join(self.log_dir, "traj.csv"), "w", newline="")
        self._w = csv.writer(self._f)
        J = range(1, 7)
        self._w.writerow(["t", "tick", "state", "sent"] + [f"qc{j}" for j in J] + ["pcx", "pcy", "pcz", "rcx", "rcy", "rcz"]
                         + [f"q{j}" for j in J] + [f"qd{j}" for j in J] + [f"eff{j}" for j in J]
                         + ["ex", "ey", "ez", "erx", "ery", "erz"]  # 펌웨어 EndPose(native)
                         + ["fk_pos_err_mm", "fk_rot_err_deg"])     # FK(q 실측) vs 펌웨어 EndPose

    def _log_row(self, state, qc, pc, Rc, sent):
        if self.q_meas is None:
            return
        nan6 = [np.nan] * 6
        rc = K.R_to_rpy_deg(Rc.as_matrix()) if Rc is not None else [np.nan] * 3
        fpe = fre = np.nan
        if self.ee_native is not None:
            p, R = self.kin.fk(self.q_meas)
            fpe = np.linalg.norm(p - self.ee_native[:3]) * 1000
            Rf = K.rpy_deg_to_R(self.ee_native[3:])
            fre = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R.T @ Rf) - 1) / 2))))
        self._w.writerow([f"{time.time():.4f}", self.tick, state, int(sent)]
                         + list(np.degrees(qc) if qc is not None else nan6)
                         + list(pc if pc is not None else [np.nan] * 3) + list(rc)
                         + list(np.degrees(self.q_meas)) + list(self.qd_meas) + list(self.eff_meas)
                         + list(self.ee_native if self.ee_native is not None else nan6) + [fpe, fre])

    def _write_plan_csv(self):
        with open(os.path.join(self.log_dir, "plan.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k", "sent"] + [f"qc{j}" for j in range(1, 7)] + ["pcx", "pcy", "pcz", "rcx", "rcy", "rcz"])
            for k in range(len(self.plan["q"])):
                w.writerow([k, int(self.plan["send"][k])] + list(np.degrees(self.plan["q"][k]))
                           + list(self.plan["p"][k]) + list(K.R_to_rpy_deg(self.plan["R"][k].as_matrix())))
        with open(os.path.join(self.log_dir, "plan_summary.txt"), "w") as f:
            f.write(self.plan_summary + "\n")

    def destroy_node(self):
        if hasattr(self, "_f"):
            self._f.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = None
    try:
        node = TrajTestNode()
        if node.state == "dry":  # 계획·검사만 — 드라이버 없이도 끝난다
            node._write_plan_csv()
            print(f"  [dry_run] 계획 통과 → {node.log_dir}/plan.csv (명령 송신 없음)")
            return
        while rclpy.ok() and not node.finished:  # 완료되면 스스로 종료 → launch 전체가 내려간다
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        if rclpy.ok():  # Ctrl-C 종료 중 rclpy 내부 예외는 무시
            print(f"  ★ traj_test_node 중단: {e}")
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
