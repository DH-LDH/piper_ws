import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import math
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Float32MultiArray, Bool, String, Int32
from sensor_msgs.msg import JointState

from step22_common import (
    KEEP_DIST, HOLD_MAX, WHEEL_R, CONV_VELOCITY,
    STEER_JOINTS, DRIVE_JOINTS, WHEEL_XY, STEER_MAX,
    PLACE_SHELF_LATERAL_DIST, PLACE_DOCK_KEEP_DIST,
    _wrap, unpack_chassis_pose,
)

DT = 1.0 / 60.0

# ── 사용자 조정 파라미터 ─────────────────────────────────────────────────────
# 컨베이어(직선궤도) 추종: bx=진행축(along-belt, v_cmd로 직접 보정) /
# by=간격(across-gap, KEEP_DIST 기준). 조향(w_cmd)은 발산 문제로 비활성화,
# 순수 속도매칭만으로 추종한다(설계 경위는 vault Logs/isaacsim 참고).

# 부트스트랩(정지 상태에서 벨트 직선속도 추정)
BOOT_MIN_PTS = 5                      # 회귀에 필요한 최소 관측점 수
BOOT_MIN_DISP = 0.004                 # [m] 회귀에 필요한 최소 bx 변위
BOOT_MAX_RES = 0.010                  # [m] 직선회귀 잔차 허용 상한
BOOT_MAX_PTS = 60                     # 관측점 버퍼 상한
BOOT_GAP_RESET_SEC = 1.5              # [s] 이만큼 관측이 끊기면 버퍼 리셋
SEARCH_STALL_STEPS = 180              # [스텝] 블라인드 서치가 이 시간 넘게 실패하면 강제 전진 전환
BOOT_V_MIN, BOOT_V_MAX = 0.005, 1.5   # [m/s] 추정 벨트속도 허용 범위

# 추종 제어 게인
K_V_ADAPT = 0.35      # v̂(느린 벨트속도 추정치) 적응 게인
K_V_INST  = 1.20      # v_cmd 즉시보정 게인(bx 비례)
V_RATE = 0.10         # [m/s²] v̂ 변화율 제한
V_INST_CLAMP = 0.30   # [m/s] 즉시보정 항 상한
W_MAX = 1.50          # [rad/s] w_cmd 클립 상한(조향은 비활성이라 안전판으로만 남김)

W_CMD_LPF_ALPHA = 1.0   # w_cmd 저역통과필터 계수 — 0(안바뀜)~1(필터없음)
BY_GUARD = 0.28         # [m] 이보다 가까워지면 감속 금지(정지 방지)
V_MAX = 0.80            # [m/s] 최대 주행속도

AMR_TRACK_BIAS_COMP = 0.0   # [m] 추종 목표 거리 보정치 — Ranger Mini 정상상태오차 재실측 전까지 0
AMR_TRACK_KEEP_DIST = KEEP_DIST + AMR_TRACK_BIAS_COMP

# 락 게이트 (락 = "타겟에 충분히 붙었다"는 판정)
LOCK_BX_TOL = 0.030   # [m] 락 성립 허용오차(진행축)
LOCK_BY_TOL = 0.030   # [m] 락 성립 허용오차(간격)
LOCK_HOLD_N = 90      # [스텝] 이만큼 연속 만족해야 락 확정

# place 도킹(고정 선반) 
PLACE_DOCK_KP = 1.0          # [1/s] bx→v_cmd, ey→vy_cmd 비례 게인
PLACE_DOCK_V_MAX = 0.30      # [m/s] 도킹 중 속도 상한(픽 추종보다 느리게)
PLACE_DOCK_SEARCH_V = 0.25   # [m/s] 선반 미검출 시 전진 서치 속도
PLACE_TURN_W = 0.2           # [rad/s] 도킹 전 제자리 회전 각속도(오픈루프 명령)
PLACE_TURN_TARGET_RAD = math.pi/2.0   # [rad] 오도메트리 적분치가 이만큼 되면 회전 종료(실측 90도 확정)
PLACE_TURN_MAX_STEPS = int(PLACE_TURN_TARGET_RAD / PLACE_TURN_W * 60) * 2  # [스텝] 안전 타임아웃(목표각 비례)
PLACE_TURN_RAMP_STEPS = 90   # [스텝, ~1.5s@60Hz] 회전 시작 사다리꼴 가속 구간
PLACE_CLEAR_VY = 0.15        # [m/s] 회전 전 벨트에서 멀어지는 크랩 속도(-vy=away)
# plant_node가 선반을 재배치할 때 쓰는 PLACE_SHELF_LATERAL_DIST에서 역산 — 단일 소스로 둬야
# 두 값이 어긋나 도킹서치가 선반을 못 찾는 문제가 재발하지 않는다.
PLACE_CLEAR_STEPS = int(PLACE_SHELF_LATERAL_DIST / PLACE_CLEAR_VY * 60)  # [스텝]
PLACE_DOCK_BLIND_STOP = PLACE_DOCK_KEEP_DIST + 0.35   # [m] 이보다 가까운 데서 관측이 끊기면 정지(충돌 방지)
# 도킹 락은 픽보다 느슨하게 — 마커 자세추정 오차가 선반 중심까지 지렛대로 증폭되지만
# 상판이 넓어(550×400mm) 그 정도 오차는 허용된다. by(거리)는 더 안정적이라 더 조인다.
PLACE_LOCK_BX_TOL = 0.10   # [m]
PLACE_LOCK_BY_TOL = 0.05   # [m]
PLACE_LOCK_HOLD_N = 30     # [스텝] 0.5s

STEER_ANGLE_HOLD_EPS = 0.02   # [m/s] 바퀴 목표속도가 이 아래면 조향각을 유지(진동 방지)

# ── 조정 파라미터 끝 ─────────────────────────────────────────────────────────


class AmrNode(Node):
    def __init__(self):
        super().__init__("amr_node")

        self.phase = "search"
        self.k = 0
        self.boot_pts = []
        self.sim_step = 0
        self.boot_k = 0
        self.boot_sim_step = 0
        self.v_hat = 0.0       # 벨트 속도 추정치(느린 적응) — 옛 om_hat 대응
        self.by_hat = 0.0      # 부트스트랩 시점 관측 간격(진단용)
        self.est = None; self.est_sim_step = -10**9; self.est_marker_id = -1
        self.last_ex = 0.0     # 타겟 놓치기 직전 마지막 ex — 블라인드 서치 방향 결정용
        self.prev_ey = 0.0; self.prev_ey_step = None  # ey 변화율(댐핑) 계산용
        self.dey_dt = 0.0  # _on_chassis_pose에서 새 관측마다 갱신
        self.blind_steps = 0   # 연속 블라인드(타겟 미검출) 스텝 수
        self.lock_run = 0; self.lock_k = 0
        self.lock_published = False
        self.ff_steps = 0; self.guard_steps = 0
        self.cam_hit = 0; self.cam_miss = 0
        self.steer_prev = np.zeros(4)   # 4WIS 역기구학 — 이전 조향각(진동방지용 이력)
        self.w_cmd_filt = 0.0           # w_cmd 저역통과필터 상태
        self._last_v_cmd = 0.0          # 진단/오프라인 시뮬레이션 검증용

        self.arm_status = "wait"        # /arm/status 최신값 — place_dock 전환 트리거
        self.place_clear_steps = 0      # place_clear(회전 전 벨트에서 멀어지기) 진행 틱 수
        self.place_turn_steps = 0       # place_turn(도킹 전 180도 회전) 진행 틱 수(타임아웃용)
        self.place_turn_yaw = 0.0       # place_turn 중 휠 오도메트리로 적분한 실측 회전각
        self.w_meas = 0.0               # /mir/joint_states로 계산한 실측 각속도(휠 기반, 폴백용)
        self.w_imu = None               # /mir/imu_yaw_rate — 슬립에 오염되지 않는 실측 각속도
        self.place_last_by = None       # 마지막 유효 관측의 by — 근접 블라인드 정지 판정용
        self.place_lock_run = 0; self.place_lock_k = 0
        self.place_lock_published = False

        self.pub_wheel = self.create_publisher(JointState, "/mir/wheel_cmd", 10)
        self.pub_lock = self.create_publisher(Bool, "/amr/lock", 1)
        self.pub_place_lock = self.create_publisher(Bool, "/amr/place_lock", 1)
        self.pub_telemetry = self.create_publisher(Float32MultiArray, "/amr/telemetry", 10)
        self.pub_status = self.create_publisher(String, "/amr/status", 5)

        self.create_subscription(Float32MultiArray, "/vision/chassis_pose",
                                  self._on_chassis_pose, 10)
        self.create_subscription(Float32MultiArray, "/vision/place_pose",
                                  self._on_place_pose, 10)
        self.create_subscription(String, "/arm/status", self._on_arm_status, 10)
        self.create_subscription(JointState, "/mir/joint_states", self._on_joint_states, 10)
        self.create_subscription(Float32, "/mir/imu_yaw_rate", self._on_imu_yaw_rate, 10)

        self.create_subscription(Int32, "/plant/tick", self._tick, 20)
        self.create_timer(2.0, self._publish_status)
        self.get_logger().info("amr_node 초기화 완료 — [탐색] 마커 통과 대기")

    def _on_arm_status(self, msg: String):
        self.arm_status = msg.data.split(" ")[0].split("=")[-1] if "phase=" in msg.data else msg.data

    def _on_imu_yaw_rate(self, msg: Float32):
        self.w_imu = float(msg.data)

    def _on_joint_states(self, msg: JointState):
        """휠 오도메트리 — place_turn의 실측 회전각 적분용(비전과 무관, 고유수용감각).
        바퀴 i의 접지속도(vx_i,vy_i)=조향각 방향×구동속도에서 w_i=(x_i*vy_i-y_i*vx_i)/r_i²
        (v=w×r 역산)로 각 바퀴의 몸체 각속도 추정치를 뽑아 4바퀴 평균."""
        idx = {n: i for i, n in enumerate(msg.name)}
        try:
            steer = [msg.position[idx[n]] for n in STEER_JOINTS]
            drive = [msg.velocity[idx[n]] for n in DRIVE_JOINTS]
        except (KeyError, IndexError):
            return
        ws = []
        for (x_i, y_i), th, wv in zip(WHEEL_XY, steer, drive):
            r2 = x_i*x_i + y_i*y_i
            if r2 < 1e-6:
                continue
            vlin = wv * WHEEL_R
            vx_i, vy_i = vlin*math.cos(th), vlin*math.sin(th)
            ws.append((x_i*vy_i - y_i*vx_i) / r2)
        if ws:
            self.w_meas = float(np.mean(ws))

    # place 선반(고정) 관측 — place_dock 단계에서만 self.est를 채운다
    def _on_place_pose(self, msg: Float32MultiArray):
        if self.phase != "place_dock": return
        bx, by, phi, valid, marker_id, sim_step, _bz = unpack_chassis_pose(msg.data)
        self.sim_step = sim_step
        if valid:
            self.est = (bx, by, phi)
            self.est_marker_id = marker_id
            self.est_sim_step = sim_step
            self.cam_hit += 1

    # ── 비전 콜백 — 최신 추정치 캐시 (원본 S['est']/S['est_age'] 패턴) ──────
    def _on_chassis_pose(self, msg: Float32MultiArray):
        if self.phase not in ("search", "track"): return
        bx, by, phi, valid, marker_id, sim_step, _bz = unpack_chassis_pose(msg.data)
        self.sim_step = sim_step
        if valid:
            self.est = (bx, by, phi)
            self.est_marker_id = marker_id
            # dey_dt는 새 관측이 실제로 도착한 시점에서만 계산 — _tick()(60Hz)에서
            # 계산하면 미갱신 틱과 갱신 틱의 dt를 똑같이 1틱으로 쳐서 값이 부풀려진다.
            ey_now = by - AMR_TRACK_KEEP_DIST
            if self.prev_ey_step is not None and sim_step > self.prev_ey_step:
                self.dey_dt = (ey_now - self.prev_ey) / ((sim_step - self.prev_ey_step) / 60.0)
            else:
                self.dey_dt = 0.0
            self.prev_ey = ey_now; self.prev_ey_step = sim_step
            self.est_sim_step = sim_step
            self.cam_hit += 1
            if self.phase == "search":
                t_now = sim_step / 60.0
                if self.boot_pts and (t_now - self.boot_pts[-1][0]) > BOOT_GAP_RESET_SEC:
                    self.boot_pts.clear()
                self.boot_pts.append((t_now, bx, by, phi, marker_id))
                if len(self.boot_pts) > BOOT_MAX_PTS:
                    del self.boot_pts[0]
        else:
            self.cam_miss += 1

    def _try_bootstrap(self):
        """정지 상태(v_cmd=w_cmd=0)에서 bx(진행축) 시간이력을 직선회귀 —
        기구학상 d(bx)/dt = CONV_VELOCITY(순수, v_cmd=0이므로)라 기울기가
        바로 벨트속도 추정치가 된다. by는 이 구간 평균으로 초기 간격만 기록
        (진단용, 제어에는 안 씀 — 아래 track phase에서 KEEP_DIST 기준으로 감)."""
        pts = self.boot_pts
        dbg = (self.k % 60 == 0)
        if len(pts) < BOOT_MIN_PTS:
            if dbg: print(f"    [부트디버그] 점 부족 {len(pts)}/{BOOT_MIN_PTS}")
            return False
        A = np.array(pts, float)
        t, bx, by, mid = A[:, 0], A[:, 1], A[:, 2], A[:, 4]
        disp = float(bx.max() - bx.min())
        if disp < BOOT_MIN_DISP:
            if dbg: print(f"    [부트디버그] 변위 부족 {disp*1000:.1f}mm/{BOOT_MIN_DISP*1000:.0f}mm "
                          f"(점 {len(pts)}개)")
            return False
        # Theil-Sen(점간 순차쌍 기울기의 중앙값) — 소표본에서 최소자승보다 강건.
        i, j = np.triu_indices(len(t), k=1)
        dt = t[j] - t[i]
        ok = np.abs(dt) > 1e-6
        if not np.any(ok):
            if dbg: print(f"    [부트디버그] 점 시각 중복(dt=0) — 기울기 계산 불가 (점 {len(pts)}개)")
            return False
        slopes = (bx[j] - bx[i])[ok] / dt[ok]
        v_fit = float(np.median(slopes))
        c0 = float(np.median(bx - v_fit*t))
        resid = bx - (v_fit*t + c0)
        res = float(np.sqrt(np.mean(resid**2)))
        if res > BOOT_MAX_RES:
            if dbg: print(f"    [부트디버그] 직선피팅 잔차 초과 {res*1000:.1f}mm/{BOOT_MAX_RES*1000:.0f}mm "
                          f"(점 {len(pts)}개, 변위 {disp*1000:.0f}mm)")
            return False
        if not (BOOT_V_MIN <= abs(v_fit) <= BOOT_V_MAX):
            if dbg:
                mid_str = "".join(str(int(m)) for m in mid)
                bx_str = ",".join(f"{v*1000:+.1f}" for v in bx)
                print(f"    [부트디버그] v̂ 범위 밖 {v_fit:+.4f} (허용 {BOOT_V_MIN}~{BOOT_V_MAX}) "
                      f"marker_id={mid_str} bx[mm]={bx_str}")
            return False

        self.v_hat = float(v_fit)
        self.by_hat = float(by.mean())
        print("\n" + "="*68)
        print(f"  ★ 벨트속도 추정 (자체스텝 {self.k}, plant스텝 {self.sim_step}, "
              f"점 {len(A)}, 변위 {disp*1000:.0f}mm)")
        print(f"      v̂ {self.v_hat*1000:+.1f}mm/s (참 {CONV_VELOCITY*1000:.0f})   "
              f"관측간격 {self.by_hat*1000:.0f}mm (목표 {AMR_TRACK_KEEP_DIST*1000:.0f})")
        print("="*68 + "\n")
        return True

    def _tick(self, msg: Int32): # plant의 물리 틱 신호로 구동
        self.k = int(msg.data)
        k = self.k

        # 픽 완료 감지 — "lift"/"verify"는 재시도 중에도 거치는 중간 단계라 트리거로 쓰면
        # 안 됨. 더는 재시도가 없는 상태(place_wait=성공, done=포기)만 트리거로 쓴다.
        if self.phase == "track" and self.arm_status in ("place_wait", "done"):
            self.phase = "place_clear"
            self.place_clear_steps = 0
            self.est = None  # 픽 타겟 관측 폐기 — place_pose 콜백이 새로 채움
            print(f"  [플레이스] 회전 전 벨트에서 멀어지는 중 (plant스텝 {self.sim_step})\n")

        if self.phase == "search":
            self._publish_wheel(0.0, 0.0, 0.0)
            if self._try_bootstrap():
                self.phase = "track"; self.boot_k = k
                self.boot_sim_step = self.sim_step
                print(f"  [추종] 직선 추종 시작 (plant스텝 {self.sim_step})\n")
            self._publish_telemetry()
            return

        if self.phase == "place_clear":
            self.place_clear_steps += 1
            if self.place_clear_steps >= PLACE_CLEAR_STEPS:
                self._publish_wheel(0.0, 0.0, 0.0)
                self.phase = "place_turn"; self.place_turn_steps = 0
                self.place_turn_yaw = 0.0
                print(f"  [플레이스] 여유 확보 완료 → 제자리 180도 회전 시작 "
                      f"(plant스텝 {self.sim_step})\n")
            else:
                self._publish_wheel(0.0, -PLACE_CLEAR_VY, 0.0)
            self._publish_telemetry()
            return

        if self.phase == "place_turn":
            # 휠 오도메트리는 제자리 회전 중 네 바퀴가 미끄러져 실제 회전량보다 크게
            # 읽힌다 — IMU(자이로) 신호를 우선 쓰고, 없을 때만 휠 기반으로 폴백한다.
            self.place_turn_yaw += (self.w_imu if self.w_imu is not None else self.w_meas) * DT
            self.place_turn_steps += 1
            i = self.place_turn_steps
            if abs(self.place_turn_yaw) >= PLACE_TURN_TARGET_RAD or i >= PLACE_TURN_MAX_STEPS:
                self._publish_wheel(0.0, 0.0, 0.0)
                self.phase = "place_dock"
                _src = "IMU" if self.w_imu is not None else "휠오도"
                print(f"  [플레이스] 회전 완료({_src} 실측 {math.degrees(self.place_turn_yaw):+.1f}도, "
                      f"{i}스텝) → 도킹 서치 시작 (plant스텝 {self.sim_step})\n")
            else:
                ramp = min(i, PLACE_TURN_RAMP_STEPS) / PLACE_TURN_RAMP_STEPS
                w = PLACE_TURN_W * float(np.clip(ramp, 0.0, 1.0))
                self._publish_wheel(0.0, 0.0, w)
            self._publish_telemetry()
            return

        if self.phase == "place_dock":
            self._tick_place_dock(k)
            return

        # ── [track] 진행축(속도매칭) + 간격(약한 조향보정) ───────────────
        est_age = k - self.est_sim_step   # 실측 물리스텝 기준 지연
        if self.est is None or est_age > HOLD_MAX:
            # 타겟을 놓치면 v̂ 자체를 서치 방향으로 램프시켜(v_cmd만 램프하면 옛 v̂가
            # 재검출 순간 그대로 튀어나와 타겟을 다시 지나친다), 수렴한 상태에서
            # 재검출이 이어지게 한다.
            self.blind_steps += 1
            direction = math.copysign(1.0, self.last_ex) if abs(self.last_ex) > 1e-6 else 0.0
            # 선형 벨트는 한 번 지나치면 다시 안 돌아온다 — 일정 시간 못 찾으면 전진으로 전환
            if self.blind_steps > SEARCH_STALL_STEPS:
                direction = 1.0
            self.v_hat = float(np.clip(self.v_hat + direction*V_RATE*DT, 0.0, BOOT_V_MAX))
            v_cmd = self.v_hat
            w_cmd = 0.0
            ex = ey = 0.0
            self.ff_steps += 1
            if k % 60 == 0:
                print(f"    [서치] plant스텝{self.sim_step} 타겟 미검출 "
                      f"{self.blind_steps/60.0:.1f}s째 — 방향{direction:+.0f} "
                      f"v̂={self.v_hat*1000:+.0f}mm/s")
        else:
            self.blind_steps = 0
            bx, by, phi = self.est
            ey = by - AMR_TRACK_KEEP_DIST   # 간격 오차 — 진단용(조향 제거로 제어엔 안 씀)
            ex = bx                         # 진행축 오차 — 속도로 직접 보정
            self.last_ex = ex               # 다음에 놓칠 경우 서치 방향 판단용

            dv = float(np.clip(K_V_ADAPT*ex, -V_RATE, V_RATE))*DT
            self.v_hat = float(np.clip(self.v_hat+dv, -BOOT_V_MAX, BOOT_V_MAX))
            v_cmd = self.v_hat + float(np.clip(K_V_INST*ex, -V_INST_CLAMP, V_INST_CLAMP))

            # 조향(w_cmd 기반 간격보정)은 고속에서 발산해 포기 — 벨트·타겟 다 직선운동이라
            # 순수 속도매칭만으로 충분하다는 판단(vault 참고). w_cmd는 항상 0.
            w_cmd = 0.0
            if k % 60 == 0:
                print(f"    [진단] plant스텝{self.sim_step} "
                      f"(추종후{(self.sim_step-self.boot_sim_step)/60.0:.1f}s) "
                      f"marker_id={self.est_marker_id} "
                      f"bx={bx*1000:+.0f}mm by={by*1000:+.0f}mm "
                      f"v̂={self.v_hat*1000:+.0f}mm/s ex={ex*1000:+.0f}mm ey={ey*1000:+.0f}mm "
                      f"dey_dt={self.dey_dt*1000:+.0f}mm/s lock_run={self.lock_run} est_age={est_age}")

        if self.est is not None and self.est[1] < BY_GUARD:
            # 간격이 위험할 만큼 좁아지면 최소한 벨트속도는 유지(정지 금지)
            v_cmd = max(v_cmd, self.v_hat)
            self.guard_steps += 1

        v_cmd = float(np.clip(v_cmd, 0.0, V_MAX))
        w_cmd = float(np.clip(w_cmd, -W_MAX, W_MAX))
        self.w_cmd_filt += W_CMD_LPF_ALPHA * (w_cmd - self.w_cmd_filt)
        self._last_v_cmd = v_cmd   # 진단/오프라인 시뮬레이션 검증용(발행값 그대로 노출)
        self._publish_wheel(v_cmd, 0.0, self.w_cmd_filt)

        # ── 락 게이트 ────────────────────────────────────────────────────
        if self.est is not None and est_age <= HOLD_MAX:
            bx, by = self.est[0], self.est[1]
            if abs(bx) < LOCK_BX_TOL and abs(by-AMR_TRACK_KEEP_DIST) < LOCK_BY_TOL:
                self.lock_run += 1
            else:
                self.lock_run = 0
        if self.lock_k == 0 and self.lock_run >= LOCK_HOLD_N:
            self.lock_k = k
            print(f"\n  ★★ 락 성립 (자체스텝 {k}, plant스텝 {self.sim_step}, "
                  f"추종 후 plant기준 {(self.sim_step-self.boot_sim_step)/60.0:.1f}s)\n")

        if self.lock_k and not self.lock_published:
            self.lock_published = True
            self.pub_lock.publish(Bool(data=True))

        self._publish_telemetry()

    def _tick_place_dock(self, k):
        """place 선반(고정)으로 도킹. place_clear 이후 선반은 새 좌측(crab +Y)축 위에
        있으므로(90도 회전 설계) 미검출 시 서치도 전진이 아니라 크랩(vy)이다.
        검출된 뒤엔 bx/by를 vx/vy로 동시 서보."""
        # 락 성립 후엔 서보를 끄고 그 자리에 세운다 — 안 그러면 마커를 놓칠 때마다
        # 미검출 분기로 떨어져 다시 크랩 전진하며 선반을 밀고 들어간다.
        if self.place_lock_k:
            self._publish_wheel(0.0, 0.0, 0.0)
            self._publish_telemetry()
            return

        est_age = k - self.est_sim_step
        if self.est is None or est_age > HOLD_MAX:
            # 선반 코앞에서 관측이 끊기면 블라인드로 더 들어가지 않고 정지(서치는 아직 멀 때만)
            if self.place_last_by is not None and self.place_last_by < PLACE_DOCK_BLIND_STOP:
                v_cmd, vy_cmd = 0.0, 0.0
                if k % 60 == 0:
                    print(f"    [도킹정지] plant스텝{self.sim_step} 근접({self.place_last_by*1000:.0f}mm)"
                          f"에서 관측 끊김 — 충돌 방지로 정지")
            else:
                v_cmd, vy_cmd = 0.0, PLACE_DOCK_SEARCH_V
                if k % 60 == 0:
                    print(f"    [도킹서치] plant스텝{self.sim_step} 선반 미검출 — "
                          f"{PLACE_DOCK_SEARCH_V*1000:+.0f}mm/s 크랩(좌측) 서치")
        else:
            bx, by, phi = self.est
            self.place_last_by = by
            ey = by - PLACE_DOCK_KEEP_DIST
            # bx는 양/음 방향 다 허용해야 함 — 하한을 0으로 자르면 선반이 진행축 뒤쪽에
            # 있을 때 보정이 안 걸려 시야에서 놓친다(픽 추종의 "전진만" 가정은 도킹에 안 맞음).
            v_cmd = float(np.clip(PLACE_DOCK_KP*bx, -PLACE_DOCK_V_MAX, PLACE_DOCK_V_MAX))
            vy_cmd = float(np.clip(PLACE_DOCK_KP*ey, -PLACE_DOCK_V_MAX, PLACE_DOCK_V_MAX))
            if k % 60 == 0:
                print(f"    [도킹] plant스텝{self.sim_step} ID{self.est_marker_id} bx={bx*1000:+.0f}mm "
                      f"by={by*1000:+.0f}mm phi={math.degrees(phi):+.1f}° "
                      f"v_cmd={v_cmd*1000:+.0f}mm/s "
                      f"vy_cmd={vy_cmd*1000:+.0f}mm/s lock_run={self.place_lock_run}")

        v_cmd = float(np.clip(v_cmd, -V_MAX, V_MAX))   # 도킹은 후진 보정도 필요(위 주석 참고)
        vy_cmd = float(np.clip(vy_cmd, -V_MAX, V_MAX))
        self._publish_wheel(v_cmd, vy_cmd, 0.0)

        if self.est is not None and est_age <= HOLD_MAX:
            if (abs(self.est[0]) < PLACE_LOCK_BX_TOL
                    and abs(self.est[1]-PLACE_DOCK_KEEP_DIST) < PLACE_LOCK_BY_TOL):
                self.place_lock_run += 1
            else:
                self.place_lock_run = 0
        if self.place_lock_k == 0 and self.place_lock_run >= PLACE_LOCK_HOLD_N:
            self.place_lock_k = k
            print(f"\n  ★★ 플레이스 도킹 완료 (자체스텝 {k}, plant스텝 {self.sim_step})\n")
        if self.place_lock_k and not self.place_lock_published:
            self.place_lock_published = True
            self.pub_place_lock.publish(Bool(data=True))

        self._publish_telemetry()

    def _publish_wheel(self, vx_cmd, vy_cmd, w_cmd):
        """body-frame (vx_cmd,vy_cmd,w_cmd) → 4WIS 역기구학(스웨이브 방식).
        vy_cmd(w_cmd=0)는 전 바퀴가 평행이 되는 크랩주행 — 실물 스펙
        max_steer_angle_parallel=90°(STEER_MAX)가 이걸 위한 값이다.
        바퀴 i: 접지속도=(vx_cmd-w_cmd*y_i, vy_cmd+w_cmd*x_i) → 조향각+속도.
        90° 넘는 조향변화는 180°반전+속도부호반전으로 최적화."""
        steer = np.zeros(4); speed = np.zeros(4)
        for i, (x_i, y_i) in enumerate(WHEEL_XY):
            vx = vx_cmd - w_cmd*y_i
            vy = vy_cmd + w_cmd*x_i
            spd = math.hypot(vx, vy)
            if spd < STEER_ANGLE_HOLD_EPS:
                steer[i] = self.steer_prev[i]
                speed[i] = 0.0
                continue
            ang = math.atan2(vy, vx)
            # 조향각이 가동범위(±90°) 밖이면 clip 대신 θ±180°(속도 부호 반전, 등가)로
            # 뒤집어 범위 안으로 정규화한다 — clip으로 자르면 방향 성분이 잘려나간다.
            if abs(ang) > math.pi/2.0:
                ang = _wrap(ang + math.pi)
                spd = -spd
            steer[i] = float(np.clip(ang, -STEER_MAX, STEER_MAX))  # 수치오차 방어용
            speed[i] = spd
        self.steer_prev = steer

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = STEER_JOINTS + DRIVE_JOINTS
        msg.position = [float(v) for v in steer]
        msg.velocity = [float(v)/WHEEL_R for v in speed]
        self.pub_wheel.publish(msg)

    def _publish_telemetry(self):
        self.pub_telemetry.publish(Float32MultiArray(data=[
            float(self.v_hat), float(self.by_hat), float(AMR_TRACK_KEEP_DIST),
            float(self.lock_run), float(self.boot_k)]))

    def _publish_status(self):
        self.pub_status.publish(String(
            data=f"phase={self.phase} v_hat={self.v_hat:+.4f} "
                 f"by_hat={self.by_hat:.4f} lock_run={self.lock_run} "
                 f"lock_k={self.lock_k} cam_hit={self.cam_hit} cam_miss={self.cam_miss}"))


def main():
    rclpy.init()
    node = AmrNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
