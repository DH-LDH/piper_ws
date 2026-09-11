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
from geometry_msgs.msg import Point

from step22_common import (
    KEEP_DIST, OBJ_CENTER_BODY, HOLD_MAX, SEARCH_Q, GRASP_APPROACH_DIR,
    HOVER_STANDOFF, HOVER_APPROACH_DIR, EE_GRIP_OFFSET, MK_TOP_OFFSET,
    BODY_LINK_WORLD_Z, PLACE_APPROACH_DIR,
    OBJ_S, BOARD_T, CELL_GAP, CELL_T,
    unpack_chassis_pose, unpack_eih_marker, pack_joint_hold_target,
)

# ── 사용자 조정 파라미터 ─────────────────────────────────────────────────────

# 이동 속도 / 보간
EE_SPEED_HOVER = 0.10    # [m/s] hover 이동 속도
EE_SPEED_PRE   = 0.05    # [m/s] pre-grip 접근 속도
EE_SPEED_GRASP = 0.025   # [m/s] 최종 하강 속도(가장 느림)
EE_SPEED_LIFT  = 0.06    # [m/s] 리프트 속도
MOVE_L_PERIOD  = 3       # [스텝] 이 주기마다 한 번씩 목표점 갱신

# 파지점 기하 (EE_GRIP_OFFSET은 step22_common에서 import — hover 계산과 공유)
APPROACH_DIST     = 0.05 # [m] 파지점 위 pre-grip 대기 높이
GRASP_DEPTH_EXTRA = 0.010  # [m] 추가 내려가기

# 최초 검출(detect) 타당성 게이트 — 기대 위치에서 너무 벗어난 검출은 오검출로 기각
DETECT_MAX_ERR = 0.10  # [m]
PRE_REDETECT     = True
PRE_REDETECT_MAX = 0.06  # [m] 1회 보정 상한(넘으면 오검출로 보고 기각)

# GRASP 하강 중 실시간 추종
GRASP_CORR_ENABLED  = True   # 차체캠 델타보정 — 하강 중 AMR 드리프트를 따라감
GRASP_CORR_JUMP_MAX = 0.08   # [m] 직전 대비 점프가 이보다 크면 기각(마커 면 전환 방어)
GRASP_EIH_TRACK_ENABLED = True   # 손목캠 재검출 — 하강 중에도 윗면 마커를 다시 재서 파지점 갱신
GRASP_EIH_JUMP_MAX      = 0.08   # [m] 최초 detect 파지점 대비 누적 허용 오차(오검출 방어)
GRASP_Z_DROP_MAX        = 0.05   # [m] 최초 검출보다 이보다 더 낮은 z는 "벨트에서 떨어진 물체"로 기각

# place(선반에 내려놓기) — 마커 기반 정밀 배치(place_ready→hover→detect→descend, 아래
# 그대로 보존)는 IK가 이상한 해로 계속 빠져 도킹 후 팔 자세 그대로 수직으로 살짝
# 내리는 단순 경로(PLACE_LOWER_DIST)로 대체했다. 아래 상수는 그 보존된 경로가
# 재활성화될 때를 위해 남겨둔다.
PLACE_DETECT_TIMEOUT   = 900    # [스텝, 15s] 이 안에 상판 마커를 못 보면 도킹시 추정치로 진행
PLACE_REDETECT_MAX     = 0.06   # [m] 1회 보정 상한(PRE_REDETECT_MAX와 동일 근거)
PLACE_Z_DROP_MAX        = 0.05  # [m] 최초 검출보다 낮은 z는 오검출로 기각(GRASP_Z_DROP_MAX 대응)
PLACE_HOME_SETTLE_STEPS = 180   # [스텝, 3s] 초기 자세 복귀 후 정착 대기
PLACE_READY_STEPS = 150         # [스텝, 2.5s] 중립 자세 복귀 대기 (현재 경로에서는 미사용)

PLACE_LOWER_DIST = 0.02         # [m] place_lock 시점 팔 자세에서 그리퍼를 수직으로 내리는 거리

# _move_l()의 "도달"은 시간/거리 비율일 뿐 실제 팔 위치를 보장하지 않으므로,
# ee_pose(실제 FK 위치)로 물리적 도달을 재확인한다(안 하면 하강 중 목표가 계속
# 갱신될 때 "공중에서 헛집음"이 발생함— 실측 확인).
PRE_ARRIVE_TOL = 0.010        # [m]
PRE_ARRIVE_MAX_WAIT = 120     # [스텝, 2s] pre 단계 물리 도달 대기 상한
GRASP_ARRIVE_TOL = 0.004      # [m] 물체(5cm) 대비 도달 판정 허용오차
GRASP_ARRIVE_MAX_WAIT = 90    # [스텝, 1.5s] grasp 단계 물리 도달 대기 상한

LIFT_HEIGHT = 0.10       # [m] 파지 후 들어올릴 높이 — plant_node의 실측 성공판정 임계값(5cm)보다 여유 있게

# 파지 성공/실패 판정 (차체 카메라 기준 — 실물 이식 가능). 리프트 완료 후
# VERIFY_WINDOW_N 동안 지켜보며: 안 보임→성공(들려서 화각 이탈), 보임+낮은 높이→실패,
# 보임+높은 높이→성공(들고 있음).
VERIFY_BASE_N     = 5     # 기준 높이를 만들 때 쓸 그립 직전 관측 수(중앙값 — 단발 튐 방어)
VERIFY_WINDOW_N   = 90    # [스텝] 리프트 완료 후 관찰 시간(1.5초)
VERIFY_LOW_BAND   = 0.02  # [m] 기준높이+이 값 이하면 "낮은 높이"(=바닥에 있다)
VERIFY_LOW_HITS   = 2     # 낮은 높이 관측이 이만큼 쌓이면 실패 확정(단발 오검출 방어)

MAX_GRASP_RETRY = 2      # 헛집음 시 재상승→재진입 최대 횟수(0이면 재시도 안 함)

# ── 조정 파라미터 끝 ─────────────────────────────────────────────────────────


class ArmNode(Node):
    def __init__(self):
        super().__init__("arm_node")

        self.arm_phase = "wait"
        self.arm_step = 0
        self.ml_start = np.zeros(3)
        self.ml_target = np.zeros(3)
        self.ml_grasp = np.zeros(3)
        self.ml_grasp_anchor = np.zeros(3)
        self._move_l_pos = None       # _move_l()의 현재 추종 위치(상태)
        self._move_l_seg_start = None # ml_start 변경 감지용(바뀌면 새 구간으로 리셋)

        self.ee_pose = None          # plant가 보내는 현재 EE 위치(몸체좌표)
        self.chassis_est = None; self.chassis_age = 999  # 차체캠(vision) 캐시
        self.chassis_marker_id = -1  # 디버그: 옆면 마커 ID0~3 중 어느 게 best인지
        self.eih_new = None          # 최신 끝단캠 검출 (매 수신마다 소비)

        self.detect_rej_n = 0        # 최초 검출 타당성 게이트에서 기각된 횟수

        # PRE 재검출 진단
        self.pre_try_n = 0; self.pre_miss_n = 0
        self.pre_miss_run = 0; self.pre_miss_max = 0
        self.pre_corr_n = 0; self.pre_rej_n = 0
        self.pre_corr_max = 0.0; self.pre_corr_last = 0.0

        # GRASP 델타보정
        self.grasp_bx0 = 0.0; self.grasp_by0 = KEEP_DIST
        self.grasp_delta_last = np.zeros(2)
        self.gr_corr_n = 0; self.gr_corr_rej = 0; self.gr_corr_seen = False
        self.gr_corr_max = 0.0
        self.gr_dbg_last_mid = -99  # 디버그: grasp 하강 중 마커ID 전환 추적

        # GRASP 하강 중 손목캠(eih) 검출률 + 실시간 재검출
        self.gr_eih_try = 0; self.gr_eih_miss = 0
        self.gr_eih_miss_run = 0; self.gr_eih_miss_max = 0
        self.gr_eih_tail = []  # 최근 N개 검출여부(bool) — 도달 직전 구간 오클루전 확인용
        GR_EIH_TAIL_N = 30
        self._GR_EIH_TAIL_N = GR_EIH_TAIL_N
        self.gr_eih_corr_n = 0; self.gr_eih_rej_n = 0
        self.gr_eih_corr_max = 0.0; self.gr_eih_corr_last = 0.0

        self.grasp_wait_n = 0  # _move_l 시간상 도달 후 실제(ee_pose) 도달 대기 카운터
        self.pre_wait_n = 0    # pre 단계의 같은 카운터

        self.grip_contact_result = None  # /gripper/done 수신 시 세팅
        self.grasp_attempt = 0  # 재시도 횟수 (0=첫 시도)

        # 차체캠 기반 파지 판정
        self.chassis_bz = None       # 차체캠이 본 물체 중심 높이(몸체좌표)
        self.bz_recent = []          # 최근 높이 관측 버퍼(기준선 산출용)
        self.verify_bz0 = None       # 그립 직전 기준 높이(중앙값)
        self.lift_bz_max = None      # 리프트~verify 중 관측된 최대 높이
        self.verify_hits = 0         # 리프트 완료 후 물체가 관측된 횟수
        self.verify_low_hits = 0     # 그중 "낮은 높이"(=바닥) 관측 수
        self.verify_high_hits = 0    # 그중 "높은 위치"(=들려 있음) 관측 수
        self.verify_bz_last = None   # 마지막 관측 높이(로그용)

        # place(픽 후 선반에 놓기)
        self.place_est = None        # 차체캠이 본 place 선반 (bx,by,phi)
        self.ml_place_center = np.zeros(3)  # place_dock 완료 시점 선반 중심(1회 스냅샷, 대략치)
        self.place_wait_n = 0        # place_descend의 물리도달 대기 카운터(grasp_wait_n과 동일 용도)
        self.place_top_new = None    # 손목캠이 본 상판 마커(ID6) 최신 관측(body_link 상대)
        self.ml_place_anchor = None  # place_detect 최초 검출 놓는점(z-drop/점프 판정 기준)

        self.pub_joint_hold = self.create_publisher(Float32MultiArray, "/arm/joint_hold_target", 10)
        self.pub_cart_target = self.create_publisher(Point, "/arm/cartesian_target", 10)
        self.pub_status = self.create_publisher(String, "/arm/status", 10)
        self.pub_gripper_cmd = self.create_publisher(Bool, "/gripper/cmd", 1)
        self.pub_pick_event = self.create_publisher(Bool, "/arm/pick_event", 1)

        self.create_subscription(Bool, "/amr/lock", self._on_lock, 1)
        self.create_subscription(Bool, "/amr/place_lock", self._on_place_lock, 1)
        self.create_subscription(Float32MultiArray, "/vision/chassis_pose", self._on_chassis_pose, 10)
        self.create_subscription(Float32MultiArray, "/vision/eih_marker_body", self._on_eih, 10)
        self.create_subscription(Float32MultiArray, "/vision/place_pose", self._on_place_pose, 10)
        self.create_subscription(Float32MultiArray, "/vision/place_marker_body", self._on_place_marker, 10)
        self.create_subscription(Point, "/arm/ee_pose_body", self._on_ee_pose, 10)
        self.create_subscription(Bool, "/gripper/done", self._on_gripper_done, 10)

        self.create_subscription(Int32, "/plant/tick", self._tick, 20)
        self.create_timer(2.0, self._publish_status)
        self.get_logger().info("arm_node 초기화 완료 — [wait] 락 대기")

    # ── 구독 콜백 ────────────────────────────────────────────────────────────
    def _on_lock(self, msg: Bool):
        if not msg.data or self.arm_phase != "wait": return
        print("      → 락 수신, 팔 파지 시퀀스 개시\n")
        self.arm_phase = "hover"; self.arm_step = 0
        if self.ee_pose is not None:
            self.ml_start = self.ee_pose.copy()
        # HOVER_STANDOFF만 넣으면 IK 타겟(link6 원점) 기준 거리라 그리퍼 끝단이 물체
        # 속으로 파고든다 — EE_GRIP_OFFSET을 더해 link6 타겟을 그만큼 더 뒤로 뺀다.
        self.ml_target = np.array(OBJ_CENTER_BODY) - HOVER_APPROACH_DIR * (HOVER_STANDOFF + EE_GRIP_OFFSET)

    def _on_place_lock(self, msg: Bool):
        if not msg.data or self.arm_phase != "place_wait": return
        print("      → place_lock 수신, 플레이스 시퀀스 개시\n")
        # 단순화 경로: 도킹이 이미 거리를 맞춰줬으므로 지금 팔 자세 그대로 그리퍼만
        # 수직으로 PLACE_LOWER_DIST만큼 내린다(마커 기반 정밀 배치는 IK가 이상한 해로
        # 계속 빠져 대체 — 그 경로는 아래 _tick()에 보존).
        self.ml_start = (self.ee_pose.copy() if self.ee_pose is not None
                          else np.array(OBJ_CENTER_BODY, float))
        self.ml_target = self.ml_start - np.array([0.0, 0.0, PLACE_LOWER_DIST])
        self.arm_phase = "place_lower"; self.arm_step = 0

    def _on_place_pose(self, msg: Float32MultiArray):
        bx, by, phi, valid, marker_id, sim_step, bz = unpack_chassis_pose(msg.data)
        if valid:
            self.place_est = (bx, by, phi)

    def _on_place_marker(self, msg: Float32MultiArray):
        """손목캠이 본 상판 마커(ID6, body_link 상대) — place_detect/place_descend가
        이걸로 실시간 보정한다(_on_eih의 place 버전, _place_point() 참고)."""
        x, y, z, valid = unpack_eih_marker(msg.data)
        self.place_top_new = (np.array([x, y, z], float) if valid else None)

    def _on_chassis_pose(self, msg: Float32MultiArray):
        bx, by, phi, valid, marker_id, sim_step, bz = unpack_chassis_pose(msg.data)
        if valid:
            self.chassis_est = (bx, by, phi)
            self.chassis_age = 0
            self.chassis_marker_id = marker_id
            self.chassis_bz = bz
            self.bz_recent.append(bz)
            if len(self.bz_recent) > VERIFY_BASE_N:
                del self.bz_recent[0]
            # 리프트 중 최고 높이 — 판정엔 안 쓰고 로그 진단용으로만 남긴다.
            if self.arm_phase == "lift":
                if self.lift_bz_max is None or bz > self.lift_bz_max:
                    self.lift_bz_max = bz

            # 판정 근거: 리프트 완료 후(verify) 물체가 보이는가, 
            if self.arm_phase == "verify":
                self.verify_hits += 1
                self.verify_bz_last = bz
                base = (self.verify_bz0 if self.verify_bz0 is not None
                        else OBJ_CENTER_BODY[2] - BODY_LINK_WORLD_Z)
                if bz <= base + VERIFY_LOW_BAND:
                    self.verify_low_hits += 1
                else:
                    self.verify_high_hits += 1

    def _on_ee_pose(self, msg: Point):
        self.ee_pose = np.array([msg.x, msg.y, msg.z], float)

    def _on_gripper_done(self, msg: Bool):
        self.grip_contact_result = bool(msg.data)

    def _grasp_point(self, mk):
        """윗면 마커 위치(mk, 몸체좌표) → link6 IK 목표(파지점). 깊이 보정(GRASP_DEPTH_EXTRA)은
        월드 -Z로, 접근축(GRASP_APPROACH_DIR) 보정은 그리퍼 길이(EE_GRIP_OFFSET)에만 걸어야
        한다 — 섞어서 한 번에 빼면 내려간 만큼 옆으로도 밀린다. detect/pre/grasp가 공유하는
        단일 함수로 통일(따로 복붙하면 어긋나기 쉬움)."""
        # mk는 body_link 상대높이인데 ml_target/IK(Lula)는 world 기준이라 변환 필요.
        mk_w = mk + np.array([0.0, 0.0, BODY_LINK_WORLD_Z])
        obj_c = mk_w - np.array([0.0, 0.0, GRASP_DEPTH_EXTRA])
        return obj_c - GRASP_APPROACH_DIR * EE_GRIP_OFFSET

    def _place_point(self, mk):
        """상판 마커(ID6) 위치(mk, 몸체좌표) → link6 IK 목표(놓는점). _grasp_point()와
        같은 부호규약(PLACE_APPROACH_DIR, world 기준) — 마커는 상판 표면 바로 위에
        보드/셀 두께만큼 떠서 붙어 있다(plant_node.build_place_shelf 참고). 물체를
        그 표면에 내려놓으려면 중심이 표면보다 OBJ_S/2 위여야 하고, IK 목표(link6)는
        거기서 그리퍼 길이(EE_GRIP_OFFSET)만큼 더 위다 — 세 오프셋을 하나로 묶는다."""
        mk_w = mk + np.array([0.0, 0.0, BODY_LINK_WORLD_Z])
        board_part = BOARD_T + CELL_GAP + CELL_T/2.0
        rise = OBJ_S/2.0 - board_part + EE_GRIP_OFFSET
        return mk_w - PLACE_APPROACH_DIR * rise

    def _verify_by_chassis_cam(self):
        """차체 카메라 관측만으로 파지 성공 여부를 판정한다(None=판정 보류). 실물에도
        있는 차체 카메라만 쓰는 실물 이식 가능한 설계 — god's-eye 좌표는 안 쓴다."""
        base = self.verify_bz0 if self.verify_bz0 is not None else OBJ_CENTER_BODY[2] - BODY_LINK_WORLD_Z
        basis = "그립 전 기준" if self.verify_bz0 is not None else "기본높이(기준선 없음)"

        # 낮은 높이로 충분히 관측됐다 → 물체는 바닥에 있다. 즉시 실패 확정
        # (더 기다릴 이유가 없다 — 못 집었거나 이미 놓쳤거나 둘 중 하나).
        if self.verify_low_hits >= VERIFY_LOW_HITS:
            print(f"    [파지 판정/차체캠] 리프트 후 물체가 낮은 높이에서 관측 "
                  f"{self.verify_low_hits}회 (마지막 {self.verify_bz_last*1000:.1f}mm, "
                  f"기준 {base*1000:.1f}mm+{VERIFY_LOW_BAND*1000:.0f}mm 이하, {basis}) "
                  f"→ 실패(못 집었거나 놓침)")
            return False

        # 관찰 창을 다 채울 때까지는 판정 보류 — 뒤늦게 떨어질 수 있다.
        if self.arm_step < VERIFY_WINDOW_N:
            return None

        mx = (f"{self.lift_bz_max*1000:.1f}mm" if self.lift_bz_max is not None else "-")
        if self.verify_hits == 0:
            print(f"    [파지 판정/차체캠] 리프트 후 {VERIFY_WINDOW_N}스텝 동안 "
                  f"물체 미검출 (리프트 중 최고 {mx}) → 성공(화각 이탈)")
        else:
            print(f"    [파지 판정/차체캠] 리프트 후 관측 {self.verify_hits}회 "
                  f"전부 높은 위치 (마지막 {self.verify_bz_last*1000:.1f}mm, "
                  f"기준 {base*1000:.1f}mm, {basis}) → 성공(들고 있음)")
        return True

    def _on_pick_verdict(self, ok: bool):
        """판정 결과 처리 — 성공이면 종료, 실패면 재시도 또는 포기."""
        # plant가 채점(실측 대조)할 수 있도록 이번 시도 결과를 알린다.
        # 이 토픽은 "제어 신호"가 아니라 "카메라 판정 보고" — plant가 자기 god's-eye
        # 실측과 대조해 판정 정확도를 기록하는 용도.
        self.pub_pick_event.publish(Bool(data=bool(ok)))
        if ok:
            print(f"  [팔] ✓ 파지 검증 성공 — place 대기로 전환")
            self.arm_phase = "place_wait"; self.arm_step = 0
            return
        self.grasp_attempt += 1
        if self.grasp_attempt > MAX_GRASP_RETRY:
            print(f"  [팔] ✗ 파지 검증 실패 — 재시도 {MAX_GRASP_RETRY}회 모두 소진, 포기")
            self.arm_phase = "done"; self.arm_step = 0
            return
        print(f"  [팔] ✗ 파지 검증 실패(헛집음) — 재시도 {self.grasp_attempt}/{MAX_GRASP_RETRY} "
              f"준비: 그리퍼 오픈 → 재상승 → 재진입")
        self.pub_gripper_cmd.publish(Bool(data=False))  # 릴리즈
        self.ml_start = (self.ee_pose.copy() if self.ee_pose is not None
                          else self.ml_target.copy())
        self.ml_target = np.array(OBJ_CENTER_BODY) - HOVER_APPROACH_DIR * (HOVER_STANDOFF + EE_GRIP_OFFSET)
        self.verify_bz0 = None; self.lift_bz_max = None
        self.verify_hits = 0; self.verify_low_hits = 0; self.verify_high_hits = 0
        self.verify_bz_last = None
        self.arm_phase = "hover"; self.arm_step = 0

    def _on_eih(self, msg: Float32MultiArray):
        x, y, z, valid = unpack_eih_marker(msg.data)
        self.eih_new = (np.array([x, y, z], float) if valid else None)

        if self.arm_phase == "detect":
            if self.eih_new is not None:
                mk = self.eih_new
                # OBJ_CENTER_BODY[2]는 world 절대높이, mk는 body_link 상대높이라
                # BODY_LINK_WORLD_Z를 빼서 기준을 맞춘다.
                mk_exp = np.array([0.0, KEEP_DIST,
                                   OBJ_CENTER_BODY[2] - BODY_LINK_WORLD_Z + MK_TOP_OFFSET])
                err = float(np.linalg.norm(mk - mk_exp))
                if err > DETECT_MAX_ERR:
                    # 오검출 — 기각하고 다음 프레임을 기다린다(detect 단계 유지).
                    self.detect_rej_n += 1
                    if self.detect_rej_n % 20 == 1:
                        print(f"    [검출 기각] 기대 위치에서 {err*1000:.0f}mm 벗어남 "
                              f"(한계 {DETECT_MAX_ERR*1000:.0f}mm) — 오검출로 보고 재검출 "
                              f"({self.detect_rej_n}회째)  mk={mk.round(4)}")
                    return
                grasp = self._grasp_point(mk)
                pre = grasp - GRASP_APPROACH_DIR * APPROACH_DIST
                print(f"\n  [팔] 윗면 마커 검출 ✓ (몸체) {mk.round(4)}")
                # 마커는 물체 중심이 아니라 그 위(윗면+보드/셀 두께)에 붙어 있다 —
                # 기대값도 그 오프셋을 반영해야 검출 오차를 제대로 읽을 수 있다.
                print(f"       기대값 (0, {KEEP_DIST}, "
                      f"{OBJ_CENTER_BODY[2] - BODY_LINK_WORLD_Z + MK_TOP_OFFSET:.3f})")
                print(f"       파지점 {grasp.round(4)}")
                self.ml_start = (self.ee_pose.copy() if self.ee_pose is not None
                                  else self.ml_target.copy())
                self.ml_target = pre
                self.ml_grasp = grasp
                self.ml_grasp_anchor = grasp.copy()  # grasp 단계 재검출의 고정 기준점(누적 clamp용)
                self.pre_try_n = 0; self.pre_miss_n = 0
                self.pre_miss_run = 0; self.pre_miss_max = 0
                self.pre_corr_n = 0; self.pre_rej_n = 0
                self.pre_corr_max = 0.0; self.pre_corr_last = 0.0
                self.pre_wait_n = 0
                bx0, by0 = (self.chassis_est[0], self.chassis_est[1]) \
                    if self.chassis_est is not None else (0.0, KEEP_DIST)
                self.grasp_bx0, self.grasp_by0 = bx0, by0
                self.grasp_delta_last = np.zeros(2)
                self.gr_corr_n = 0; self.gr_corr_rej = 0; self.gr_corr_seen = False
                self.gr_corr_max = 0.0
                self.gr_dbg_last_mid = -99  # ★디버그: grasp 진입 전이므로 항상 초기화
                self.arm_phase = "pre"; self.arm_step = 0
            return

        if self.arm_phase == "pre" and PRE_REDETECT:
            self.pre_try_n += 1
            if self.eih_new is None:
                self.pre_miss_n += 1
                self.pre_miss_run += 1
                self.pre_miss_max = max(self.pre_miss_max, self.pre_miss_run)
            else:
                self.pre_miss_run = 0
                g2 = self._grasp_point(self.eih_new)
                d = float(np.linalg.norm(g2 - self.ml_grasp))
                dropped = (self.ml_grasp_anchor[2] - g2[2]) > GRASP_Z_DROP_MAX
                if d <= PRE_REDETECT_MAX and not dropped:
                    # z는 최초 검출(anchor)보다 아래로 내려가지 않게 clamp — 근접
                    # 재검출 노이즈가 누적돼 벨트를 파고드는 것 방지(XY는 그대로 추종).
                    g2[2] = max(g2[2], self.ml_grasp_anchor[2])
                    self.ml_grasp = g2
                    self.ml_target = g2 - GRASP_APPROACH_DIR * APPROACH_DIST
                    self.pre_corr_n += 1
                    self.pre_corr_last = d
                    self.pre_corr_max = max(self.pre_corr_max, d)
                else:
                    self.pre_rej_n += 1
            return

        if self.arm_phase == "grasp":
            # 하강 중 손목캠(eih) 검출률 측정 (§9-2 오클루전 확인용, 계속 유지)
            hit = self.eih_new is not None
            self.gr_eih_try += 1
            if not hit:
                self.gr_eih_miss += 1
                self.gr_eih_miss_run += 1
                self.gr_eih_miss_max = max(self.gr_eih_miss_max, self.gr_eih_miss_run)
            else:
                self.gr_eih_miss_run = 0
            self.gr_eih_tail.append(hit)
            if len(self.gr_eih_tail) > self._GR_EIH_TAIL_N:
                self.gr_eih_tail.pop(0)

            # PRE와 동일한 재검출 로직을 grasp 하강 중에도 적용 — 점프 판정은 최초 detect
            # 파지점(ml_grasp_anchor) 대비 누적 드리프트로 clamp. 물체가 벨트에서 떨어지면
            # z가 크게 낮아지는데, 이를 그대로 쫓으면 바닥을 뚫고 내려가므로 GRASP_Z_DROP_MAX
            # 이상 낮은 z는 "떨어진 물체"로 기각한다.
            if GRASP_EIH_TRACK_ENABLED and hit:
                g2 = self._grasp_point(self.eih_new)
                d = float(np.linalg.norm(g2 - self.ml_grasp_anchor))
                dropped = (self.ml_grasp_anchor[2] - g2[2]) > GRASP_Z_DROP_MAX
                if dropped and self.gr_eih_rej_n % 30 == 0:
                    print(f"    [GRASP 안전기각] z={g2[2]*1000:.0f}mm가 최초검출 "
                          f"{self.ml_grasp_anchor[2]*1000:.0f}mm보다 {GRASP_Z_DROP_MAX*1000:.0f}mm "
                          f"이상 낮음 — 물체가 벨트에서 떨어진 것으로 보고 무시")
                if d <= GRASP_EIH_JUMP_MAX and not dropped:
                    # z clamp: anchor보다 아래로는 안 내려간다(벨트 충돌 방지, PRE와 동일 논리)
                    g2[2] = max(g2[2], self.ml_grasp_anchor[2])
                    self.ml_grasp = g2
                    self.ml_target = g2   # ml_target도 같이 갱신(안 하면 팔이 낡은 목표로 내려감)
                    self.gr_eih_corr_n += 1
                    self.gr_eih_corr_last = d
                    self.gr_eih_corr_max = max(self.gr_eih_corr_max, d)
                    if self.chassis_est is not None:
                        self.grasp_bx0 = self.chassis_est[0]
                        self.grasp_by0 = self.chassis_est[1]
                        self.grasp_delta_last = np.zeros(2)
                else:
                    self.gr_eih_rej_n += 1

    # ── 60Hz 틱 ─────────────────────────────────────────────────────────────
    def _tick(self, _msg: Int32):
        self.chassis_age += 1
        ap = self.arm_phase

        if ap == "wait":
            self.pub_joint_hold.publish(Float32MultiArray(
                data=pack_joint_hold_target(SEARCH_Q)))
            return

        self.arm_step += 1

        if ap == "hover":
            done = self._move_l(EE_SPEED_HOVER)
            if done:
                print(f"  [팔] HOVER 도달 → 윗면 마커 검출")
                self.arm_phase = "detect"; self.arm_step = 0

        elif ap == "detect":
            if self.arm_step > 900:
                print(f"  ★ 팔: 마커 검출 실패로 중단"); self.arm_phase = "fail"
            # 실제 전이는 _on_eih 콜백에서 처리 (검출 즉시 반응)

        elif ap == "pre":
            if self.arm_step % 180 == 0:  # 3초마다 진단(막히면 원인 바로 보이게)
                dist_now = (float(np.linalg.norm(self.ee_pose - self.ml_target))
                            if self.ee_pose is not None else -1.0)
                print(f"    [PRE 진행] step={self.arm_step} dist_to_target={dist_now*1000:.1f}mm "
                      f"보정={self.pre_corr_n}회(최근{self.pre_corr_last*1000:.1f}mm) "
                      f"기각={self.pre_rej_n}회 미검출={self.pre_miss_n}/{self.pre_try_n}")
            done = self._move_l(EE_SPEED_PRE)
            if done:
                # 시간 기준 done만으론 물리 도달을 보장 못 해 실제 ee_pose로 재확인(2026-09-03).
                err_p = ((self.ee_pose - self.ml_target)
                         if self.ee_pose is not None else np.zeros(3))
                dist_p = float(np.linalg.norm(err_p))
                if dist_p > PRE_ARRIVE_TOL and self.pre_wait_n < PRE_ARRIVE_MAX_WAIT:
                    self.pre_wait_n += 1
                    return  # 계속 목표를 향해 대기
                if dist_p > PRE_ARRIVE_TOL:
                    print(f"    ★ [팔] pre 물리도달 실패(잔여 {dist_p*1000:.1f}mm, "
                          f"{PRE_ARRIVE_MAX_WAIT}스텝 대기 초과) — 그냥 진행")
                else:
                    print(f"    [pre 도달오차] {dist_p*1000:.1f}mm")
                if self.pre_try_n:
                    print(f"    [PRE 검출률] {self.pre_try_n-self.pre_miss_n}/{self.pre_try_n} "
                          f"({100.0*(self.pre_try_n-self.pre_miss_n)/self.pre_try_n:.0f}%)  "
                          f"미검출 {self.pre_miss_n}회, 최장연속 {self.pre_miss_max}회")
                if self.pre_corr_n:
                    print(f"    [PRE 재검출] 보정 {self.pre_corr_n}회 "
                          f"(최대 {self.pre_corr_max*1000:.1f}mm, "
                          f"마지막 {self.pre_corr_last*1000:.1f}mm)  기각 {self.pre_rej_n}회")
                print(f"  [팔] pre-grip 도달 → 감속 하강")
                self.ml_start = (self.ee_pose.copy() if self.ee_pose is not None
                                  else self.ml_target.copy())
                self.ml_target = self.ml_grasp.copy()
                self.gr_eih_try = 0; self.gr_eih_miss = 0
                self.gr_eih_miss_run = 0; self.gr_eih_miss_max = 0
                self.gr_eih_tail = []
                self.gr_eih_corr_n = 0; self.gr_eih_rej_n = 0
                self.gr_eih_corr_max = 0.0; self.gr_eih_corr_last = 0.0
                self.grasp_wait_n = 0
                self.arm_phase = "grasp"; self.arm_step = 0

        elif ap == "grasp":
            if GRASP_CORR_ENABLED:
                if self.chassis_est is not None and self.chassis_age <= HOLD_MAX:
                    dbx = self.chassis_est[0] - self.grasp_bx0
                    dby = self.chassis_est[1] - self.grasp_by0
                    prev = self.grasp_delta_last
                    jump = float(np.hypot(dbx-prev[0], dby-prev[1]))
                    mid = self.chassis_marker_id
                    mid_switched = (mid != self.gr_dbg_last_mid)
                    if mid_switched:
                        print(f"    [그립보정디버그] ID{self.gr_dbg_last_mid}→ID{mid} 전환 "
                              f"@step{self.arm_step}  Δ후보=({dbx*1000:+.1f},{dby*1000:+.1f})mm "
                              f"jump={jump*1000:.1f}mm")
                        self.gr_dbg_last_mid = mid
                    if (not self.gr_corr_seen) or jump <= GRASP_CORR_JUMP_MAX:
                        if mid_switched and jump*1000 > 15.0:
                            print(f"      → 수용(전환 직후 점프 {jump*1000:.1f}mm < "
                                  f"JUMP_MAX={GRASP_CORR_JUMP_MAX*1000:.0f}mm, 실이동으로 오인 가능)")
                        self.grasp_delta_last = np.array([dbx, dby])
                        self.gr_corr_seen = True
                        self.gr_corr_n += 1
                        self.gr_corr_max = max(self.gr_corr_max, float(np.hypot(dbx, dby)))
                    else:
                        self.gr_corr_rej += 1
                        if mid_switched:
                            print(f"      → 기각(전환 직후 점프 {jump*1000:.1f}mm > JUMP_MAX, Δ 유지)")
                dx, dy = self.grasp_delta_last
                self.ml_target = self.ml_grasp + np.array([dx, dy, 0.0])

            done = self._move_l(EE_SPEED_GRASP)
            if done:
                # ee_pose(실제 FK 위치)로 물리적 도달을 재확인 — done은 시간/거리
                # 비율일 뿐이라, 하강 중 목표(ml_grasp)가 계속 갱신되면(위 델타보정,
                # 손목캠 재검출) 마지막 갱신으로 남은 거리가 줄어드는 순간 실제로는
                # 안 왔는데 done이 튀어서 그리퍼를 조기에 닫는 문제가 있었음.
                err = (self.ee_pose - self.ml_grasp) if self.ee_pose is not None else np.zeros(3)
                dist = float(np.linalg.norm(err))
                if dist > GRASP_ARRIVE_TOL and self.grasp_wait_n < GRASP_ARRIVE_MAX_WAIT:
                    self.grasp_wait_n += 1
                    return  # 그리퍼 닫지 않고 계속 목표를 향해 대기
                if dist > GRASP_ARRIVE_TOL:
                    print(f"    ★ [팔] 파지점 물리도달 실패(잔여 {dist*1000:.1f}mm, "
                          f"{GRASP_ARRIVE_MAX_WAIT}스텝 대기 초과) — 그냥 진행")
                print(f"    [파지 실제오차] xyz=({err[0]*1000:+.1f},{err[1]*1000:+.1f},"
                      f"{err[2]*1000:+.1f})mm 거리={dist*1000:.1f}mm ee_pose={self.ee_pose.round(4) if self.ee_pose is not None else None} "
                      f"ml_grasp={self.ml_grasp.round(4)}")
                if GRASP_CORR_ENABLED:
                    dxf, dyf = self.grasp_delta_last
                    print(f"    [GRASP 델타보정] 차체카메라 {self.gr_corr_n}회 반영, "
                          f"{self.gr_corr_rej}회 기각(점프)  최대 {self.gr_corr_max*1000:.1f}mm  "
                          f"최종 Δ ({dxf*1000:+.1f}, {dyf*1000:+.1f})mm")
                if self.gr_eih_try:
                    tail_hit = sum(1 for h in self.gr_eih_tail if h)
                    print(f"    [GRASP 손목캠 검출률] {self.gr_eih_try-self.gr_eih_miss}/"
                          f"{self.gr_eih_try} ({100.0*(self.gr_eih_try-self.gr_eih_miss)/self.gr_eih_try:.0f}%)  "
                          f"미검출 {self.gr_eih_miss}회, 최장연속미검출 {self.gr_eih_miss_max}회  "
                          f"도달직전 {len(self.gr_eih_tail)}프레임 중 {tail_hit}회 검출")
                if GRASP_EIH_TRACK_ENABLED:
                    print(f"    [GRASP 손목캠 재검출] 반영 {self.gr_eih_corr_n}회 "
                          f"(최대 {self.gr_eih_corr_max*1000:.1f}mm, "
                          f"마지막 {self.gr_eih_corr_last*1000:.1f}mm)  "
                          f"기각 {self.gr_eih_rej_n}회  최종 파지점 {self.ml_grasp.round(4)}")
                print(f"  [팔] 파지점 도달 → 그리퍼 닫기")
                #   파지 판정 기준선: 그리퍼를 닫기 직전 물체 높이(최근 관측
                #   중앙값 — 단발 오검출로 기준선이 틀어지는 걸 막는다).
                #   리프트 중 이 값 대비 얼마나 올라가는지로 성공을 가른다.
                self.verify_bz0 = (float(np.median(self.bz_recent))
                                    if self.bz_recent else None)
                if self.verify_bz0 is not None:
                    print(f"       [판정기준] 그립 전 물체 높이 {self.verify_bz0*1000:.1f}mm "
                          f"(차체캠 {len(self.bz_recent)}회 중앙값)")
                else:
                    print(f"       ★ [판정기준] 차체캠 미검출 — 절대높이로 판정")
                self.arm_phase = "grip"; self.arm_step = 0
                self.grip_contact_result = None
                self.pub_gripper_cmd.publish(Bool(data=True))

        elif ap == "grip":
            if self.grip_contact_result is not None:
                print(f"  [팔] 그립 완료({'접촉 감지' if self.grip_contact_result else '헛집음 가능성'}) "
                      f"→ 들어올리기")
                self.ml_start = (self.ee_pose.copy() if self.ee_pose is not None
                                  else self.ml_target.copy())
                self.ml_target = self.ml_start - GRASP_APPROACH_DIR * LIFT_HEIGHT
                self.lift_bz_max = None
                self.verify_hits = 0; self.verify_low_hits = 0
                self.verify_high_hits = 0; self.verify_bz_last = None
                self.arm_phase = "lift"; self.arm_step = 0

        elif ap == "lift":
            done = self._move_l(EE_SPEED_LIFT)
            if done:
                #   차체캠으로 직접 판정한다(시뮬 전용 god's-eye
                #   좌표를 안 쓴다 — 실물 이식을 위해). 판정 근거는 리프트가
                #   올라가는 동안 이미 모아둔 관측(lift_bz_max)이다.
                self.arm_phase = "verify"; self.arm_step = 0

        elif ap == "verify":
            verdict = self._verify_by_chassis_cam()
            if verdict is not None:
                self._on_pick_verdict(verdict)

        elif ap == "place_wait":
            if self.arm_step > 5400:  # 90초 안전판(place_clear+turn+dock 정상 완주 ~30초보다 여유) — place_lock이 끝내 안 오면 포기
                print("  ★ 팔: place_lock 대기 타임아웃 — 포기")
                self.arm_phase = "place_done"; self.arm_step = 0
            # 실제 전이는 _on_place_lock 콜백에서 처리 (수신 즉시 반응)

        elif ap == "place_lower":
            # 단순화된 place 경로 — 현재 자세에서 grasp와 같은 속도/orientation(DOWN_QUAT)으로
            # PLACE_LOWER_DIST만큼 수직 하강한다.
            done = self._move_l(EE_SPEED_GRASP)
            if done:
                print(f"  [팔] PLACE 하강 완료 → 그리퍼 열기")
                self.pub_gripper_cmd.publish(Bool(data=False))
                self.arm_phase = "place_release"; self.arm_step = 0

        # ─────────────────────────────────────────────────────────────────
        # 아래 place_ready~place_retreat는 마커 기반 정밀 배치(구버전) — 지금은
        # _on_place_lock()이 place_lower로 직행해 이 블록들에 도달하지 않는다.
        # 나중에 근접 시야/IK 갈래 문제를 해결하면 _on_place_lock()의 진입점만
        # "place_ready"로 되돌려 재활성화할 수 있도록 코드를 그대로 남겨둔다.
        # ─────────────────────────────────────────────────────────────────
        elif ap == "place_ready":
            # 물체를 쥔 채 중립(SEARCH_Q)으로 복귀 — 그리퍼 명령은 안 건드리므로
            # 물체는 계속 잡고 있다. AMR이 90도 돌고 크랩하는 동안 리프트 자세로
            # 버티는 것보다 이쪽이 관성·간섭 면에서도 낫다.
            self.pub_joint_hold.publish(Float32MultiArray(
                data=pack_joint_hold_target(SEARCH_Q)))
            if self.arm_step > PLACE_READY_STEPS:
                print(f"  [팔] 중립 자세 복귀 완료 → PLACE HOVER 시작")
                self.ml_start = (self.ee_pose.copy() if self.ee_pose is not None
                                  else self.ml_place_center.copy())
                self.ml_target = (self.ml_place_center
                                   - PLACE_APPROACH_DIR * (HOVER_STANDOFF + EE_GRIP_OFFSET))
                self.arm_phase = "place_hover"; self.arm_step = 0

        elif ap == "place_hover":
            done = self._move_l(EE_SPEED_HOVER)
            if done:
                print(f"  [팔] PLACE HOVER 도달 → 상판 마커 검출")
                self.ml_place_anchor = None
                self.arm_phase = "place_detect"; self.arm_step = 0

        elif ap == "place_detect":
            # pick의 hover→detect와 동일 패턴 — 손목캠으로 상판 마커(ID6)를 보고 놓는점을 계산
            if self.place_top_new is not None:
                pt = self._place_point(self.place_top_new)
                print(f"\n  [팔] 상판 마커 검출 ✓ (몸체) {self.place_top_new.round(4)}")
                print(f"       놓는점 {pt.round(4)}  (도킹 추정치는 "
                      f"{(self.ml_place_center - PLACE_APPROACH_DIR*EE_GRIP_OFFSET).round(4)})")
                self.ml_place_anchor = pt.copy()
                self.ml_start = (self.ee_pose.copy() if self.ee_pose is not None
                                  else self.ml_target.copy())
                self.ml_target = pt
                self.place_wait_n = 0
                self.arm_phase = "place_descend"; self.arm_step = 0
            elif self.arm_step > PLACE_DETECT_TIMEOUT:
                print(f"  ★ 팔: 상판 마커 미검출({PLACE_DETECT_TIMEOUT}스텝) — "
                      f"도킹 시점 추정치로 하강 진행")
                pt = self.ml_place_center - PLACE_APPROACH_DIR * EE_GRIP_OFFSET
                self.ml_place_anchor = pt.copy()
                self.ml_start = (self.ee_pose.copy() if self.ee_pose is not None
                                  else self.ml_target.copy())
                self.ml_target = pt
                self.place_wait_n = 0
                self.arm_phase = "place_descend"; self.arm_step = 0

        elif ap == "place_descend":
            # pick의 grasp 단계와 동일한 실시간 재검출+보정 — 최초 검출(anchor) 대비
            # 점프가 크면(오검출) 기각하고, anchor보다 낮은 z는 상판을 뚫고 내려가는
            # 걸 막기 위해 clamp한다(GRASP_Z_DROP_MAX와 같은 방향의 안전장치).
            if self.place_top_new is not None and self.ml_place_anchor is not None:
                pt = self._place_point(self.place_top_new)
                d = float(np.linalg.norm(pt - self.ml_place_anchor))
                dropped = (self.ml_place_anchor[2] - pt[2]) > PLACE_Z_DROP_MAX
                if d <= PLACE_REDETECT_MAX and not dropped:
                    pt[2] = max(pt[2], self.ml_place_anchor[2])
                    self.ml_target = pt
            if self.arm_step % 60 == 0 and self.ee_pose is not None:
                gap = float(np.linalg.norm(self.ee_pose - self.ml_target))
                print(f"    [PLACE진단] step{self.arm_step} ee_pose={self.ee_pose.round(4)} "
                      f"target={self.ml_target.round(4)} 잔여={gap*1000:.1f}mm "
                      f"move_l_pos={(self._move_l_pos.round(4) if self._move_l_pos is not None else None)}")
            done = self._move_l(EE_SPEED_GRASP)
            if done:
                err = (self.ee_pose - self.ml_target) if self.ee_pose is not None else np.zeros(3)
                dist = float(np.linalg.norm(err))
                if dist > GRASP_ARRIVE_TOL and self.place_wait_n < GRASP_ARRIVE_MAX_WAIT:
                    self.place_wait_n += 1
                    return
                print(f"  [팔] PLACE 목표 도달(잔여 {dist*1000:.1f}mm) → 그리퍼 열기")
                self.pub_gripper_cmd.publish(Bool(data=False))
                self.arm_phase = "place_release"; self.arm_step = 0

        elif ap == "place_release":
            # 단순화된 경로 — 후퇴(retreat) 없이 정착 후 바로 place_home(SEARCH_Q)으로 간다.
            if self.arm_step > 60:
                print(f"  [팔] 릴리즈 완료 → 초기 자세로 복귀")
                self.arm_phase = "place_home"; self.arm_step = 0

        # place_retreat(마커기반 경로의 후퇴 단계) — place_release가 바로 place_home으로
        # 넘어가는 현재 경로에서는 도달하지 않는다. 재활성화 시 place_release도 같이 되돌릴 것.
        elif ap == "place_retreat":
            done = self._move_l(EE_SPEED_LIFT)
            if done:
                print(f"  [팔] 후퇴 완료 → 초기 자세로 복귀")
                self.arm_phase = "place_home"; self.arm_step = 0

        elif ap == "place_home":
            # 플레이스가 끝나면 팔을 대기자세(SEARCH_Q)로 되돌린다(직접 관절목표, IK 경유 안 함)
            self.pub_joint_hold.publish(Float32MultiArray(
                data=pack_joint_hold_target(SEARCH_Q)))
            if self.arm_step > PLACE_HOME_SETTLE_STEPS:
                print(f"  [팔] ✓ 초기 자세 복귀 완료 — 플레이스 시퀀스 종료")
                self.arm_phase = "place_done"; self.arm_step = 0

        self.pub_status.publish(String(data=f"phase={self.arm_phase} arm_step={self.arm_step}"))

    def _move_l(self, speed):
        """목표(ml_target)로 매틱 speed만큼씩 다가가는 속도제한 추종. (완료 여부) 반환.
        매틱 target 위치를 publish — plant가 IK를 풀어 실제로 적용한다. (구 방식은
        total=norm(target-start)의 경과시간 비율이라, target이 계속 갱신되는
        움직이는 물체를 쫓을 땐 total이 같이 늘어나 영원히 안 끝났음 — 실측 확인.)"""
        if self.arm_step % MOVE_L_PERIOD != 0:
            return False
        if self._move_l_pos is None or not np.array_equal(self._move_l_seg_start, self.ml_start):
            self._move_l_pos = self.ml_start.copy()
        self._move_l_seg_start = self.ml_start.copy()
        target = self.ml_target
        remaining = target - self._move_l_pos
        dist = float(np.linalg.norm(remaining))
        step = speed * (MOVE_L_PERIOD/60.0)
        if dist <= step:
            self._move_l_pos = target.copy()
            done = True
        else:
            self._move_l_pos = self._move_l_pos + remaining/dist*step
            done = False
        pos = self._move_l_pos
        self.pub_cart_target.publish(Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])))
        return done

    def _publish_status(self):
        self.get_logger().info(f"phase={self.arm_phase}")


def main():
    rclpy.init()
    node = ArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
