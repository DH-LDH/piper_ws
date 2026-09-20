import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Float32MultiArray, Bool, String, Int32
from geometry_msgs.msg import Point

from step22_common import (
    KEEP_DIST, OBJ_CENTER_BODY, HOLD_MAX, SEARCH_Q, GRASP_APPROACH_DIR,
    HOVER_STANDOFF, HOVER_APPROACH_DIR, EE_GRIP_OFFSET, MK_TOP_OFFSET,
    BODY_LINK_WORLD_Z, PLACE_CENTER_BODY_GUESS,
    OBJ_S, BOARD_T, CELL_GAP, CELL_T, PLACE_PITCH_DEG,
    unpack_chassis_pose, unpack_eih_marker, unpack_grip_state,
    pack_joint_hold_target,
)

# ── 사용자 조정 파라미터 ─────────────────────────────────────────────────────

# 이동 속도 / 보간
EE_SPEED_HOVER = 0.10    # [m/s] hover 이동 속도
EE_SPEED_PRE   = 0.05    # [m/s] pre-grip 접근 속도
EE_SPEED_GRASP = 0.025   # [m/s] 최종 하강 속도(가장 느림)
EE_SPEED_LIFT  = 0.06    # [m/s] 리프트 속도
MOVE_L_PERIOD  = 3       # [스텝] 이 주기마다 한 번씩 목표점 갱신

# 파지점 기하 (self.ee_grip_offset은 step22_common에서 import — hover 계산과 공유)
APPROACH_DIST     = 0.05 # [m] 파지점 위 pre-grip 대기 높이 (launch: approach_dist)
GRASP_DEPTH_EXTRA = 0.010  # [m] 마커 평면보다 더 내려가는 깊이 (launch: grasp_depth_extra)

# 최초 검출(detect) 타당성 게이트 — 기대 위치에서 너무 벗어난 검출은 오검출로 기각
DETECT_MAX_ERR = 0.25  # [m] 2026-09-17: obj_expected_y_body 추정치가 부정확해서 완화
PRE_REDETECT     = True
PRE_REDETECT_MAX = 0.06  # [m] 1회 보정 상한(넘으면 오검출로 보고 기각)

# GRASP 하강 중 실시간 추종
GRASP_CORR_ENABLED  = True   # 차체캠 델타보정 — 하강 중 AMR 드리프트를 따라감
GRASP_CORR_JUMP_MAX = 0.08   # [m] 직전 대비 점프가 이보다 크면 기각(마커 면 전환 방어)
GRASP_EIH_TRACK_ENABLED = True   # 손목캠 재검출 — 하강 중에도 윗면 마커를 다시 재서 파지점 갱신
GRASP_EIH_JUMP_MAX      = 0.08   # [m] 최초 detect 파지점 대비 누적 허용 오차(오검출 방어)
GRASP_Z_DROP_MAX        = 0.05   # [m] 최초 검출보다 이보다 더 낮은 z는 "벨트에서 떨어진 물체"로 기각
# 최초 검출(anchor)은 hover 높이의 원거리 관측이라 z가 높게 잡히기 쉽다 — 근접 재검출이
# 그보다 낮다고 말하면 이 값까지는 따라 내려간다(0이면 예전처럼 anchor 아래로 안 감).
GRASP_Z_BELOW_ANCHOR_MAX = 0.02  # [m] (launch: grasp_z_below_anchor_max)

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

# place 놓는점 계산 방식(place_point_mode)
#   "shelf_geom"(sim 기본): 선반 상판 두께/보드 두께 등 sim 지오메트리로 계산(_place_point)
#   "marker_inset"(실물):   마커 중심에서 "팔 베이스 쪽으로" PLACE_INSET_M만큼 당긴 지점에
#                            놓는다. 물체를 마커 위에 그대로 올리면 (a)마커를 덮어버려
#                            재검출이 끊기고 (b)쥔 물체가 마커 시야를 가린다. 마커 한 변
#                            (34mm)만큼 당기면 마커 근접모서리 바로 바깥에 놓인다.
# 당기는 방향은 place_inset_mode로 고른다:
#   "marker_y"(실물 기본): 마커 자신의 -Y 방향. 마커를 선반에 비스듬히 붙여도 "마커
#                           기준 앞쪽"이 그대로 따라간다(vision이 yaw를 같이 발행).
#   "body_y":              body -Y(팔 베이스 쪽 정면) 고정.
#   "radial":              body 원점→마커 수평방향의 반대(종전 동작).
PLACE_INSET_M       = 0.035   # [m] 마커 중심→놓는점 거리 (마커 한 변 34mm와 같은 값)
PLACE_RELEASE_GAP   = 0.005   # [m] 릴리즈 시 물체 바닥과 선반면 사이 간격
PLACE_RETREAT_DIST  = 0.05    # [m] 릴리즈 후 수직 후퇴 거리(초기자세 복귀 전 물체를 안 건드리게)

# _move_l()의 "도달"은 시간/거리 비율일 뿐 실제 팔 위치를 보장하지 않으므로,
# ee_pose(실제 FK 위치)로 물리적 도달을 재확인한다(안 하면 하강 중 목표가 계속
# 갱신될 때 "공중에서 헛집음"이 발생함— 실측 확인).
PRE_ARRIVE_TOL = 0.010        # [m]
# [스텝] 물리 도달 대기 상한. 5% 속도에서는 5초로 못 간다 — 2026-09-20 실기에서
# pre/grasp 둘 다 이 상한을 넘겨 "그냥 진행"으로 빠졌고, 그게 덜 문 원인이었다.
# (launch: arrive_max_wait)
PRE_ARRIVE_MAX_WAIT = 300     # [스텝, 5s]
# [m] 도달 판정 허용오차. 이 값은 "여기까지 왔으면 그리퍼를 닫는다"는 뜻이라 곧바로
# 파지 깊이 오차가 된다 — 기본 4mm는 sim의 5cm 큐브(깊이 25mm) 기준이었고, 실물
# 16.36mm 타겟(깊이 8mm)에는 너무 크다(최대 4mm 덜 내려간 채로 닫힐 수 있다).
GRASP_ARRIVE_TOL = 0.004      # [m] 3D 거리 허용오차 (launch: grasp_arrive_tol)
# z는 따로, 더 좁게 본다 — XY는 몇 mm 어긋나도 손가락 사이에 들어오지만 z가 덜 내려가면
# 물체 윗모서리만 물게 된다. (launch: grasp_arrive_z_tol)
GRASP_ARRIVE_Z_TOL = 0.002    # [m]
GRASP_ARRIVE_MAX_WAIT = 300   # [스텝, 5s] grasp 단계 물리 도달 대기 상한(위와 같은 파라미터)

LIFT_HEIGHT = 0.10       # [m] 파지 후 들어올릴 높이 — plant_node의 실측 성공판정 임계값(5cm)보다 여유 있게

# ── 파지 성공/실패 판정 ─────────────────────────────────────────────────────
# verify_mode 파라미터로 고른다.
#   "gripper"(실물 v1 기본): 그리퍼 개구부(스트로크)+접촉으로 판정. 차체캠이 없어도 된다.
#   "chassis"(sim 기본):     아래 차체캠 로직. 실물에 차체캠을 달면 그대로 되살아난다.
#   "both":                  그리퍼가 판정하고 차체캠 결과는 로그로만 남긴다(교차검증).
#
# [차체캠 방식 — 실물에선 비활성(차체캠 미장착)] 리프트 완료 후 VERIFY_WINDOW_N 동안
# 지켜보며: 안 보임→성공(들려서 화각 이탈), 보임+낮은 높이→실패, 보임+높은 높이→성공.
# ★ 실물에서 이 방식을 쓰면 안 되는 이유: 차체캠이 없으면 관측이 0건이라 "미검출=성공"
#   분기로 무조건 빠진다(헛집어도 성공 보고). 차체캠을 달면 verify_mode=both로 먼저
#   교차검증하고, 일치하면 chassis로 돌리거나 둘의 AND를 쓰면 된다.
VERIFY_BASE_N     = 5     # 기준 높이를 만들 때 쓸 그립 직전 관측 수(중앙값 — 단발 튐 방어)
VERIFY_WINDOW_N   = 90    # [스텝] 리프트 완료 후 관찰 시간(1.5초)
VERIFY_LOW_BAND   = 0.02  # [m] 기준높이+이 값 이하면 "낮은 높이"(=바닥에 있다)
VERIFY_LOW_HITS   = 2     # 낮은 높이 관측이 이만큼 쌓이면 실패 확정(단발 오검출 방어)

# [그리퍼 방식 — 실물 기본] 물체를 물고 있으면 손가락이 물체 폭에서 멈춘다. 헛집으면
# 손가락끼리 맞닿아 개구부가 0 근처로 내려간다 — 접촉력만으로는 이 둘을 구분 못 한다
# (빈손이어도 손가락끼리 눌리면 접촉으로 잡힌다). 그래서 스트로크가 주 판정근거다.
GRIP_VERIFY_WINDOW_N   = 60    # [스텝, 1s] 리프트 완료 후 그리퍼 상태 관찰 시간
GRIP_VERIFY_MIN_SAMPLE = 10    # 이만큼은 받아야 판정한다(토픽 유실 방어)
# ★ 보고되는 개구부는 "손가락 패드 사이 실제 간격"이 아니다. 실기 실측(2026-09-20):
# 16.36mm 물체를 정상 파지했는데 25.9mm로 보고됐다 — 패드 두께 등으로 상수만큼 크게
# 나온다(영점 미설정 영향도 있음). 이 오프셋을 빼지 않으면 판정선이 통째로 어긋난다.
# 재기 방법: 아무것도 없이 그리퍼를 닫고 그때 보고되는 개구부가 곧 이 값이다.
GRIP_STROKE_OFFSET_MM  = 0.0   # [mm] 보고값 − 실제 간격 (launch: grip_stroke_offset_mm)
GRIP_STROKE_MIN_RATIO  = 0.50  # 물체 폭 대비 이 비율 밑이면 "사이에 아무것도 없다"
GRIP_STROKE_MARGIN_MM  = 10.0  # [mm] 물체 폭+이 값보다 넓으면 "물체를 안 물었다"(덜 닫힘)
GRIP_SLIP_DROP_MM      = 3.0   # [mm] 그립 직후 대비 이만큼 더 닫히면 리프트 중 놓친 것

MAX_GRASP_RETRY = 2      # 헛집음 시 재상승→재진입 최대 횟수(0이면 재시도 안 함)

# ── 조정 파라미터 끝 ─────────────────────────────────────────────────────────


class ArmNode(Node):
    def __init__(self):
        super().__init__("arm_node")

        self.arm_phase = "wait"
        self.arm_step = 0
        # 단계별 수동 확인 모드 — True면 phase가 바뀔 때마다 멈추고 /arm/step_confirm 대기.
        self.step_confirm = bool(self.declare_parameter("step_confirm", False).value)
        self.step_ready = not self.step_confirm
        self._last_gated_phase = self.arm_phase
        self._gate_pub_phase = self.arm_phase  # 확인 대기 중 driver에 알릴 phase
        # link6 원점 → 그리퍼 손가락 "끝" 거리. sim 기본값(0.135)은 손가락 장착점(joint7)이라
        # 실물 손가락 길이가 빠져 있다 — 실물은 launch에서 실측값으로 덮어쓴다.
        self.ee_grip_offset = float(
            self.declare_parameter("ee_grip_offset", EE_GRIP_OFFSET).value)
        # 파지 기하 3종 — 실기 실측으로 맞추는 값이라 런치 인자로 뺐다.
        self.approach_dist = float(
            self.declare_parameter("approach_dist", APPROACH_DIST).value)
        self.grasp_depth_extra = float(
            self.declare_parameter("grasp_depth_extra", GRASP_DEPTH_EXTRA).value)
        self.grasp_z_below_anchor_max = float(
            self.declare_parameter("grasp_z_below_anchor_max", GRASP_Z_BELOW_ANCHOR_MAX).value)
        self.grasp_arrive_tol = float(
            self.declare_parameter("grasp_arrive_tol", GRASP_ARRIVE_TOL).value)
        self.grasp_arrive_z_tol = float(
            self.declare_parameter("grasp_arrive_z_tol", GRASP_ARRIVE_Z_TOL).value)
        self.arrive_max_wait = int(
            self.declare_parameter("arrive_max_wait", GRASP_ARRIVE_MAX_WAIT).value)
        # 손목캠 재검출 추종 on/off — 캘리브레이션 중엔 꺼서 detect 시점 목표를 고정시킨다.
        self.pre_redetect = bool(
            self.declare_parameter("pre_redetect", PRE_REDETECT).value)
        self.grasp_eih_track = bool(
            self.declare_parameter("grasp_eih_track", GRASP_EIH_TRACK_ENABLED).value)
        self.create_subscription(Bool, "/arm/step_confirm", self._on_step_confirm, 10)
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
        # z clamp(anchor 하한)가 실제로 물린 횟수 — 물리면 "근접 관측은 더 내려가라는데
        # 최초 검출 기준 하한에 막혀 덜 내려간" 상태다. 덜 물리는 증상의 유력 원인.
        self.gr_z_clamp_n = 0; self.gr_z_clamp_max = 0.0

        self.grasp_wait_n = 0  # _move_l 시간상 도달 후 실제(ee_pose) 도달 대기 카운터
        self.pre_wait_n = 0    # pre 단계의 같은 카운터
        self.hover_wait_n = 0  # hover 단계의 같은 카운터
        self.lift_wait_n = 0   # lift 단계의 같은 카운터
        self.retreat_wait_n = 0  # place_retreat 단계의 같은 카운터
        self.place_hover_wait_n = 0  # place_hover 단계의 같은 카운터

        self.grip_contact_result = None  # /gripper/done 수신 시 세팅
        self.grasp_attempt = 0  # 재시도 횟수 (0=첫 시도)

        # 파지 판정 방식 — 실물 런치에서 "gripper"로 오버라이드(차체캠 미장착).
        self.verify_mode = str(self.declare_parameter("verify_mode", "chassis").value)
        if self.verify_mode not in ("gripper", "chassis", "both"):
            print(f"  ★ verify_mode={self.verify_mode} 알 수 없음 — 'gripper'로 진행")
            self.verify_mode = "gripper"
        # 타겟 치수 — 폭은 그리퍼 판정(스트로크), 높이는 place 릴리즈 높이에 쓴다.
        self.obj_width_m = float(self.declare_parameter("obj_width_m", OBJ_S).value)
        self.obj_height_m = float(self.declare_parameter("obj_height_m", OBJ_S).value)
        self.grip_stroke_offset = float(
            self.declare_parameter("grip_stroke_offset_mm", GRIP_STROKE_OFFSET_MM).value)

        # 그리퍼 기반 파지 판정 상태
        self.grip_stroke = None       # 최신 개구부[mm]
        self.grip_effort = 0.0        # 최신 접촉 저항(단위는 노드별 — 로그용)
        self.grip_contact_flag = False
        self.grip_holding = False
        self.stroke_recent = []       # 최근 개구부 버퍼(그립 직후 기준값 산출용)
        self.grip_stroke_at_grip = None  # 그립 완료 시점 개구부(슬립 판정 기준)
        self.grip_stroke_min = None   # 리프트~verify 중 최소 개구부
        self.grip_v_n = 0             # verify 중 받은 grip_state 샘플 수

        # 차체캠 기반 파지 판정
        self.chassis_bz = None       # 차체캠이 본 물체 중심 높이(몸체좌표)
        self.bz_recent = []          # 최근 높이 관측 버퍼(기준선 산출용)
        self.verify_bz0 = None       # 그립 직전 기준 높이(중앙값)
        self.lift_bz_max = None      # 리프트~verify 중 관측된 최대 높이
        self.verify_hits = 0         # 리프트 완료 후 물체가 관측된 횟수
        self.verify_low_hits = 0     # 그중 "낮은 높이"(=바닥) 관측 수
        self.verify_high_hits = 0    # 그중 "높은 위치"(=들려 있음) 관측 수
        self.verify_bz_last = None   # 마지막 관측 높이(로그용)

        # place(픽 후 선반에 놓기) — use_marker_place=True(실물, AMR 없음)면 마커 기반
        # place_ready~place_descend 경로를 쓰고, False(기본값, 기존 sim 경로)면 도킹
        # 좌표만으로 수직 하강하는 place_lower 경로를 그대로 쓴다.
        self.use_marker_place = bool(
            self.declare_parameter("use_marker_place", False).value)
        # sim은 body_link(AMR)가 world보다 이만큼 높아 Lula(world 기준 IK) 계산에 필요했지만,
        # 실물은 body_link=world(팔 베이스)라 0으로 둬야 한다 — 실물 launch에서 0.0으로 오버라이드.
        self.body_link_world_z = float(
            self.declare_parameter("body_link_world_z", BODY_LINK_WORLD_Z).value)
        # hover가 처음 향하는 "물체 대략 위치" 추정값 — sim은 OBJ_CENTER_BODY(월드 절대높이
        # 기반)를 그대로 쓰지만, 실물은 body_link=world라 body_link 기준 실측값을 직접 넣어야
        # 한다(예: 물체 높이-팔 베이스 높이). 기본값은 sim과 수치가 같도록 역산해둠.
        self.obj_expected_x_body = float(
            self.declare_parameter("obj_expected_x_body", OBJ_CENTER_BODY[0]).value)
        self.obj_expected_y_body = float(
            self.declare_parameter("obj_expected_y_body", OBJ_CENTER_BODY[1]).value)
        self.obj_expected_z_body = float(
            self.declare_parameter("obj_expected_z_body", OBJ_CENTER_BODY[2] - BODY_LINK_WORLD_Z).value)
        # place 놓는점 계산 방식 + 실물 전용 파라미터(위 PLACE_* 상수 설명 참고)
        self.place_point_mode = str(
            self.declare_parameter("place_point_mode", "shelf_geom").value)
        self.place_inset = float(self.declare_parameter("place_inset_m", PLACE_INSET_M).value)
        self.place_inset_mode = str(
            self.declare_parameter("place_inset_mode", "radial").value)
        if self.place_inset_mode not in ("marker_y", "body_y", "radial"):
            print(f"  ★ place_inset_mode={self.place_inset_mode} 알 수 없음 — radial로 진행")
            self.place_inset_mode = "radial"
        self.place_top_yaw = None    # 손목캠이 본 place 마커의 body 기준 +Y 방향[rad]
        self.place_release_gap = float(
            self.declare_parameter("place_release_gap", PLACE_RELEASE_GAP).value)
        # 쥔 물체가 마커 시야를 가리므로 최초 검출 후엔 목표를 얼어붙이고 맹목 하강한다.
        self.place_freeze_after_detect = bool(
            self.declare_parameter("place_freeze_after_detect", False).value)
        # place pitch — sim은 선반이 높아 30°라야 IK가 풀렸지만, 실물은 픽과 같은 수직
        # 접근(0°)이 기본이다. 여기서 만든 방향벡터를 place 전 구간이 공유한다
        # (piper_driver_node의 place_pitch_deg와 같은 값을 줘야 자세가 맞는다).
        self.place_pitch_deg = float(
            self.declare_parameter("place_pitch_deg", PLACE_PITCH_DEG).value)
        _ppr = np.radians(self.place_pitch_deg)
        self.place_approach_dir = np.array([0.0, np.sin(_ppr), -np.cos(_ppr)])

        self.place_est = None        # 차체캠이 본 place 선반 (bx,by,phi)
        # marker_inset 모드에서는 "place 마커가 대략 있을 body_link 위치"를 뜻하고,
        # shelf_geom 모드에서는 종전대로 "놓을 물체 중심 위치"를 뜻한다.
        self.ml_place_center = np.array([
            float(self.declare_parameter("place_expected_x_body", PLACE_CENTER_BODY_GUESS[0]).value),
            float(self.declare_parameter("place_expected_y_body", PLACE_CENTER_BODY_GUESS[1]).value),
            float(self.declare_parameter("place_expected_z_body", PLACE_CENTER_BODY_GUESS[2]).value),
        ], float) if self.use_marker_place else np.zeros(3)
        # place_dock(AMR)이 없는 실물 경로에서는 위 자리표시자가 유일한 기준점이라
        # amr_node가 있던 sim과 달리 아무도 갱신해주지 않는다 — 마커 재검출
        # (place_detect/place_descend)이 이 추정치를 실제 위치로 보정한다.
        self.place_wait_n = 0        # place_descend의 물리도달 대기 카운터(grasp_wait_n과 동일 용도)
        self.place_top_new = None    # 손목캠이 본 place 마커 최신 관측(body_link 상대)
        self.ml_place_anchor = None  # place_detect 최초 검출 놓는점(z-drop/점프 판정 기준)

        self.pub_joint_hold = self.create_publisher(Float32MultiArray, "/arm/joint_hold_target", 10)
        self.pub_cart_target = self.create_publisher(Point, "/arm/cartesian_target", 10)
        self.pub_status = self.create_publisher(String, "/arm/status", 10)
        self.pub_gripper_cmd = self.create_publisher(Bool, "/gripper/cmd", 1)
        self.pub_pick_event = self.create_publisher(Bool, "/arm/pick_event", 1)
        # step_confirm.py가 "지금 무슨 단계를 기다리는지" 알 수 있게 — 늦게 떠도 받도록 latch.
        self.pub_step_wait = self.create_publisher(
            String, "/arm/step_wait",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self.create_subscription(Bool, "/amr/lock", self._on_lock, 1)
        self.create_subscription(Bool, "/amr/place_lock", self._on_place_lock, 1)
        self.create_subscription(Float32MultiArray, "/vision/chassis_pose", self._on_chassis_pose, 10)
        self.create_subscription(Float32MultiArray, "/vision/eih_marker_body", self._on_eih, 10)
        self.create_subscription(Float32MultiArray, "/vision/place_pose", self._on_place_pose, 10)
        self.create_subscription(Float32MultiArray, "/vision/place_marker_body", self._on_place_marker, 10)
        self.create_subscription(Point, "/arm/ee_pose_body", self._on_ee_pose, 10)
        self.create_subscription(Bool, "/gripper/done", self._on_gripper_done, 10)
        self.create_subscription(Float32MultiArray, "/gripper/grip_state", self._on_grip_state, 10)

        self.create_subscription(Int32, "/plant/tick", self._tick, 20)
        self.create_timer(2.0, self._publish_status)
        self.get_logger().info(
            f"arm_node 초기화 완료 — [wait] 락 대기  (파지기하: ee_grip_offset="
            f"{self.ee_grip_offset*1000:.0f}mm approach_dist={self.approach_dist*1000:.0f}mm "
            f"grasp_depth_extra={self.grasp_depth_extra*1000:.0f}mm "
            f"z_below_anchor_max={self.grasp_z_below_anchor_max*1000:.0f}mm)")

    # ── 구독 콜백 ────────────────────────────────────────────────────────────
    def _on_lock(self, msg: Bool):
        if not msg.data or self.arm_phase != "wait": return
        print("      → 락 수신, 팔 파지 시퀀스 개시\n")
        self.arm_phase = "hover"; self.arm_step = 0
        if self.ee_pose is not None:
            self.ml_start = self.ee_pose.copy()
        # HOVER_STANDOFF만 넣으면 IK 타겟(link6 원점) 기준 거리라 그리퍼 끝단이 물체
        # 속으로 파고든다 — self.ee_grip_offset을 더해 link6 타겟을 그만큼 더 뒤로 뺀다.
        self.ml_target = self._obj_center_body_rel() - HOVER_APPROACH_DIR * (HOVER_STANDOFF + self.ee_grip_offset)

    def _obj_center_body_rel(self):
        """물체 대략 위치(body_link 기준) — obj_expected_*_body 파라미터로 실물/sim 각각 설정."""
        return np.array([self.obj_expected_x_body, self.obj_expected_y_body,
                          self.obj_expected_z_body])

    def _on_place_lock(self, msg: Bool):
        if not msg.data or self.arm_phase != "place_wait": return
        if self.use_marker_place:
            print("      → place_lock 수신, 마커 기반 플레이스 시퀀스 개시(place_ready)\n")
            self.arm_phase = "place_ready"; self.arm_step = 0
            return
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
        """손목캠이 본 place 마커(body_link 상대) — place_detect/place_descend가
        이걸로 실시간 보정한다(_on_eih의 place 버전, _place_point() 참고)."""
        x, y, z, valid, yaw = unpack_eih_marker(msg.data)
        self.place_top_new = (np.array([x, y, z], float) if valid else None)
        if valid:
            self.place_top_yaw = float(yaw)

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
                        else OBJ_CENTER_BODY[2] - self.body_link_world_z)
                if bz <= base + VERIFY_LOW_BAND:
                    self.verify_low_hits += 1
                else:
                    self.verify_high_hits += 1

    def _on_ee_pose(self, msg: Point):
        self.ee_pose = np.array([msg.x, msg.y, msg.z], float)

    def _on_gripper_done(self, msg: Bool):
        self.grip_contact_result = bool(msg.data)

    def _on_grip_state(self, msg: Float32MultiArray):
        """그리퍼 상시 상태(개구부/접촉) — 그리퍼 기반 파지 판정의 유일한 입력."""
        stroke, effort, contact, holding, _active = unpack_grip_state(msg.data)
        self.grip_stroke = stroke
        self.grip_effort = effort
        self.grip_contact_flag = contact
        self.grip_holding = holding
        self.stroke_recent.append(stroke)
        if len(self.stroke_recent) > VERIFY_BASE_N:
            del self.stroke_recent[0]
        # 리프트~verify 동안의 최소 개구부 — 중간에 한 번이라도 빈손 수준으로 닫히면
        # 그 순간 물체를 놓친 것이다(마지막 값만 보면 다시 벌어진 프레임에 속는다).
        if self.arm_phase in ("lift", "verify"):
            if self.grip_stroke_min is None or stroke < self.grip_stroke_min:
                self.grip_stroke_min = stroke
        if self.arm_phase == "verify":
            self.grip_v_n += 1

    def _grasp_point(self, mk):
        """윗면 마커 위치(mk, 몸체좌표) → link6 IK 목표(파지점). 깊이 보정(grasp_depth_extra)은
        월드 -Z로, 접근축(GRASP_APPROACH_DIR) 보정은 그리퍼 길이(self.ee_grip_offset)에만 걸어야
        한다 — 섞어서 한 번에 빼면 내려간 만큼 옆으로도 밀린다. detect/pre/grasp가 공유하는
        단일 함수로 통일(따로 복붙하면 어긋나기 쉬움)."""
        # mk는 body_link 상대높이인데 ml_target/IK(Lula)는 world 기준이라 변환 필요.
        mk_w = mk + np.array([0.0, 0.0, self.body_link_world_z])
        obj_c = mk_w - np.array([0.0, 0.0, self.grasp_depth_extra])
        return obj_c - GRASP_APPROACH_DIR * self.ee_grip_offset

    def _place_point(self, mk):
        """place 마커 위치(mk, 몸체좌표) → link6 IK 목표(놓는점). place_point_mode로 분기."""
        if self.place_point_mode == "marker_inset":
            return self._place_point_marker_inset(mk)
        return self._place_point_shelf_geom(mk)

    def _place_inset_dir(self, mk_w):
        """마커 중심에서 놓는점까지 "어느 방향으로" 비킬지 — body XY 단위벡터."""
        if self.place_inset_mode == "marker_y":
            if self.place_top_yaw is None:
                # 마커를 아직 못 봤다(사전 추정치로 계산 중) — radial로 폴백한다.
                print("    ★ [place] marker_y 모드인데 마커 yaw 미수신 — radial로 폴백")
            else:
                # 마커 자신의 -Y (+Y축 방향이 place_top_yaw)
                return np.array([-np.cos(self.place_top_yaw),
                                  -np.sin(self.place_top_yaw)])
        elif self.place_inset_mode == "body_y":
            return np.array([0.0, -1.0])
        v = mk_w[:2]
        n = float(np.linalg.norm(v))
        # 베이스 바로 위(n≈0)면 방향이 정의되지 않는다 — 그땐 당기지 않는다.
        return -v / n if n > 1e-6 else np.zeros(2)

    def _place_point_marker_inset(self, mk):
        """[실물] 마커 중심에서 팔 베이스 쪽으로 place_inset만큼 당긴 지점에 놓는다.
        물체를 마커 위에 그대로 올리면 마커가 덮여 재검출이 끊기고, 쥔 물체가 시야도
        가린다. 높이는 "물체 바닥이 선반면보다 place_release_gap 위"가 되게 잡는다 —
        손끝은 물체 윗면보다 grasp_depth_extra만큼 아래를 쥐고 있으므로(_grasp_point),
        손끝~물체 바닥 거리는 (물체높이 - grasp_depth_extra)다."""
        mk_w = mk + np.array([0.0, 0.0, self.body_link_world_z])
        u = self._place_inset_dir(mk_w)
        tip = np.array([mk_w[0] + u[0]*self.place_inset,
                        mk_w[1] + u[1]*self.place_inset,
                        mk_w[2] + self.place_release_gap
                        + (self.obj_height_m - self.grasp_depth_extra)])
        # 깊이는 순수 수직으로, 그리퍼 길이만 접근축으로 — _grasp_point()와 같은 규약.
        return tip - self.place_approach_dir * self.ee_grip_offset

    def _place_point_from_guess(self):
        """마커를 끝내 못 봤을 때 쓰는 사전 추정 놓는점. ml_place_center의 의미가
        모드마다 달라서(marker_inset=마커 위치, shelf_geom=놓을 물체 중심) 갈라둔다."""
        if self.place_point_mode == "marker_inset":
            return self._place_point(self.ml_place_center)
        return self.ml_place_center - self.place_approach_dir * self.ee_grip_offset

    def _place_point_shelf_geom(self, mk):
        """[sim] 상판 마커(ID6) 위치(mk, 몸체좌표) → link6 IK 목표(놓는점). _grasp_point()와
        같은 부호규약(place_approach_dir, world 기준) — 마커는 상판 표면 바로 위에
        보드/셀 두께만큼 떠서 붙어 있다(plant_node.build_place_shelf 참고). 물체를
        그 표면에 내려놓으려면 중심이 표면보다 OBJ_S/2 위여야 하고, IK 목표(link6)는
        거기서 그리퍼 길이(self.ee_grip_offset)만큼 더 위다 — 세 오프셋을 하나로 묶는다."""
        mk_w = mk + np.array([0.0, 0.0, self.body_link_world_z])
        board_part = BOARD_T + CELL_GAP + CELL_T/2.0
        rise = OBJ_S/2.0 - board_part + self.ee_grip_offset
        return mk_w - self.place_approach_dir * rise

    def _verify_by_chassis_cam(self):
        """차체 카메라 관측만으로 파지 성공 여부를 판정한다(None=판정 보류).
        ★ verify_mode="chassis"/"both"일 때만 호출된다. 실물 v1은 차체캠이 없어서
        관측이 0건 → "미검출=성공"으로 빠지므로 쓰면 안 된다(그래서 기본이 gripper).
        차체캠을 달면 verify_mode=both로 이 함수를 그대로 되살려 교차검증할 것."""
        base = self.verify_bz0 if self.verify_bz0 is not None else OBJ_CENTER_BODY[2] - self.body_link_world_z
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

    def _wait_physical_arrive(self, wait_attr, tol, label):
        """_move_l의 시간상 done 이후 실제(ee_pose) 도달을 기다린다. True면 진행해도 됨.
        5% 속도에서는 스트림이 끝난 뒤에도 팔이 한참 더 가므로 이 확인이 없으면
        다음 단계가 "아직 안 간 상태"에서 시작한다(2026-09-20 실기에서 확인)."""
        err = ((self.ee_pose - self.ml_target)
               if self.ee_pose is not None else np.zeros(3))
        dist = float(np.linalg.norm(err))
        n = getattr(self, wait_attr)
        if dist > tol and n < self.arrive_max_wait:
            setattr(self, wait_attr, n + 1)
            return False
        if dist > tol:
            print(f"    ★ [팔] {label} 물리도달 실패(잔여 {dist*1000:.1f}mm, "
                  f"{self.arrive_max_wait}스텝 대기 초과) — 그냥 진행")
        else:
            print(f"    [{label} 도달오차] {dist*1000:.1f}mm")
        return True

    def _verify_by_gripper(self):
        """그리퍼 개구부+접촉만으로 파지 성공 여부를 판정한다(None=판정 보류).
        차체캠이 없는 실물 v1의 기본 경로 — 카메라 시야/오클루전에 의존하지 않는다."""
        # 판정선은 "보고값" 스케일로 옮겨서 비교한다(위 GRIP_STROKE_OFFSET_MM 설명 참고).
        w = self.obj_width_m * 1000.0
        lo = self.grip_stroke_offset + w * GRIP_STROKE_MIN_RATIO
        hi = self.grip_stroke_offset + w + GRIP_STROKE_MARGIN_MM

        if self.grip_stroke is None:
            # 토픽이 아예 안 온다 — 판정 근거가 없으므로 접촉 플래그로 폴백한다.
            if self.arm_step < GRIP_VERIFY_WINDOW_N:
                return None
            fb = bool(self.grip_contact_result)
            print(f"    ★ [파지 판정/그리퍼] /gripper/grip_state 미수신 — "
                  f"접촉 플래그로 폴백 판정({fb}). 그리퍼 노드가 떠 있는지 확인할 것")
            return fb

        # 빈손: 손가락이 물체 폭 아래까지 닫혔다 → 사이에 아무것도 없다(즉시 확정).
        if self.grip_stroke_min is not None and self.grip_stroke_min < lo:
            print(f"    [파지 판정/그리퍼] 최소 개구부 {self.grip_stroke_min:.1f}mm < "
                  f"하한 {lo:.1f}mm (물체폭 {w:.1f}mm×{GRIP_STROKE_MIN_RATIO:.2f} "
                  f"+ 오프셋 {self.grip_stroke_offset:.1f}mm) → 실패(헛집음/놓침)")
            return False

        # 슬립: 그립 완료 시점보다 눈에 띄게 더 닫혔다 → 리프트 중 빠져나갔다.
        if (self.grip_stroke_at_grip is not None and self.grip_stroke_min is not None
                and (self.grip_stroke_at_grip - self.grip_stroke_min) > GRIP_SLIP_DROP_MM):
            print(f"    [파지 판정/그리퍼] 그립 직후 {self.grip_stroke_at_grip:.1f}mm → "
                  f"최소 {self.grip_stroke_min:.1f}mm ("
                  f"{self.grip_stroke_at_grip - self.grip_stroke_min:.1f}mm 더 닫힘, "
                  f"한계 {GRIP_SLIP_DROP_MM:.1f}mm) → 실패(리프트 중 놓침)")
            return False

        if self.arm_step < GRIP_VERIFY_WINDOW_N or self.grip_v_n < GRIP_VERIFY_MIN_SAMPLE:
            return None

        if self.grip_stroke > hi:
            print(f"    [파지 판정/그리퍼] 개구부 {self.grip_stroke:.1f}mm > 상한 {hi:.1f}mm "
                  f"— 물체를 물지 않았거나 덜 닫힘 → 실패")
            return False
        if not self.grip_contact_flag:
            print(f"    [파지 판정/그리퍼] 개구부 {self.grip_stroke:.1f}mm는 정상이나 "
                  f"접촉 미감지(effort={self.grip_effort:.2f}) → 실패")
            return False
        print(f"    [파지 판정/그리퍼] 개구부 {self.grip_stroke:.1f}mm "
              f"(허용 {lo:.1f}~{hi:.1f}mm, 최소 {self.grip_stroke_min:.1f}mm) "
              f"접촉 유지 effort={self.grip_effort:.2f} 샘플 {self.grip_v_n} → 성공(들고 있음)")
        # 판정선 가장자리에 붙으면 다음 런에서 뒤집힌다 — 오프셋 재조정 신호.
        near = min(self.grip_stroke - lo, hi - self.grip_stroke)
        if near < 3.0:
            print(f"    ★ [주의] 판정선까지 {near:.1f}mm밖에 안 남았다 — "
                  f"grip_stroke_offset_mm(현재 {self.grip_stroke_offset:.1f})를 "
                  f"실측으로 맞출 것(빈손으로 닫았을 때의 보고값)")
        return True

    def _verify(self):
        """verify_mode에 따라 판정 주체를 고른다(None=판정 보류)."""
        if self.verify_mode == "chassis":
            return self._verify_by_chassis_cam()
        v = self._verify_by_gripper()
        if v is not None and self.verify_mode == "both":
            # 교차검증 — 판정은 그리퍼가 하고 차체캠은 기록만 남긴다.
            c = self._verify_by_chassis_cam()
            print(f"    [교차검증] 그리퍼={v}  차체캠="
                  f"{'보류' if c is None else c}  일치={'-' if c is None else (c == v)}")
        return v

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
        self.ml_target = self._obj_center_body_rel() - HOVER_APPROACH_DIR * (HOVER_STANDOFF + self.ee_grip_offset)
        self.verify_bz0 = None; self.lift_bz_max = None
        self.verify_hits = 0; self.verify_low_hits = 0; self.verify_high_hits = 0
        self.verify_bz_last = None
        self.grip_stroke_at_grip = None; self.grip_stroke_min = None; self.grip_v_n = 0
        self.hover_wait_n = 0
        self.arm_phase = "hover"; self.arm_step = 0

    def _on_eih(self, msg: Float32MultiArray):
        x, y, z, valid, _yaw = unpack_eih_marker(msg.data)
        self.eih_new = (np.array([x, y, z], float) if valid else None)

        if self.arm_phase == "detect":
            if self.eih_new is not None:
                mk = self.eih_new
                mk_exp = np.array([self.obj_expected_x_body, self.obj_expected_y_body,
                                   self.obj_expected_z_body + MK_TOP_OFFSET])
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
                pre = grasp - GRASP_APPROACH_DIR * self.approach_dist
                print(f"\n  [팔] 윗면 마커 검출 ✓ (몸체) {mk.round(4)}")
                # 마커는 물체 중심이 아니라 그 위(윗면+보드/셀 두께)에 붙어 있다 —
                # 기대값도 그 오프셋을 반영해야 검출 오차를 제대로 읽을 수 있다.
                print(f"       기대값 ({self.obj_expected_x_body}, {self.obj_expected_y_body}, "
                      f"{self.obj_expected_z_body + MK_TOP_OFFSET:.3f})")
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

        if self.arm_phase == "pre" and self.pre_redetect:
            self.pre_try_n += 1
            if self.eih_new is None:
                self.pre_miss_n += 1
                self.pre_miss_run += 1
                self.pre_miss_max = max(self.pre_miss_max, self.pre_miss_run)
            else:
                self.pre_miss_run = 0
                g2 = self._grasp_point(self.eih_new)
                d = float(np.linalg.norm(g2 - self.ml_grasp))          # 직전 대비(로그용)
                d_anc = float(np.linalg.norm(g2 - self.ml_grasp_anchor))  # 최초검출 대비(수용 판정)
                dropped = (self.ml_grasp_anchor[2] - g2[2]) > GRASP_Z_DROP_MAX
                if d_anc <= PRE_REDETECT_MAX and not dropped:
                    # z는 최초 검출(anchor)보다 아래로 내려가지 않게 clamp — 근접
                    # 재검출 노이즈가 누적돼 벨트를 파고드는 것 방지(XY는 그대로 추종).
                    g2[2] = max(g2[2], self.ml_grasp_anchor[2] - self.grasp_z_below_anchor_max)
                    self.ml_grasp = g2
                    self.ml_target = g2 - GRASP_APPROACH_DIR * self.approach_dist
                    self.pre_corr_n += 1
                    self.pre_corr_last = d
                    self.pre_corr_max = max(self.pre_corr_max, d_anc)  # anchor 대비 최대 이동량
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
            if self.grasp_eih_track and hit:
                g2 = self._grasp_point(self.eih_new)
                d = float(np.linalg.norm(g2 - self.ml_grasp_anchor))
                dropped = (self.ml_grasp_anchor[2] - g2[2]) > GRASP_Z_DROP_MAX
                if dropped and self.gr_eih_rej_n % 30 == 0:
                    print(f"    [GRASP 안전기각] z={g2[2]*1000:.0f}mm가 최초검출 "
                          f"{self.ml_grasp_anchor[2]*1000:.0f}mm보다 {GRASP_Z_DROP_MAX*1000:.0f}mm "
                          f"이상 낮음 — 물체가 벨트에서 떨어진 것으로 보고 무시")
                if d <= GRASP_EIH_JUMP_MAX and not dropped:
                    # z clamp: anchor보다 아래로는 안 내려간다(벨트 충돌 방지, PRE와 동일 논리)
                    z_lo = self.ml_grasp_anchor[2] - self.grasp_z_below_anchor_max
                    if g2[2] < z_lo:
                        self.gr_z_clamp_n += 1
                        self.gr_z_clamp_max = max(self.gr_z_clamp_max, z_lo - g2[2])
                    g2[2] = max(g2[2], z_lo)
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

    def _on_step_confirm(self, msg: Bool):
        if msg.data:
            self.step_ready = True

    # ── 60Hz 틱 ─────────────────────────────────────────────────────────────
    def _tick(self, _msg: Int32):
        self.chassis_age += 1
        ap = self.arm_phase

        if self.step_confirm and ap != self._last_gated_phase:
            self._last_gated_phase = ap
            self.step_ready = False
            self.pub_step_wait.publish(String(data=ap))
            print(f"  [STEP] '{ap}' 단계 진입 대기 중 — 진행하려면 step_confirm.py 실행(Enter)")
        if self.step_confirm and not self.step_ready:
            # 대기 중엔 이전 단계를 그대로 유지시킨다 — phase만 먼저 바뀌면 driver가
            # JointCtrl도 EndPoseCtrl도 못 보내는 공백이 생겨 팔이 그 자리에 멈춘다.
            if self._gate_pub_phase == "wait":
                self.pub_joint_hold.publish(Float32MultiArray(
                    data=pack_joint_hold_target(SEARCH_Q)))
            return
        self._gate_pub_phase = ap

        if ap == "wait":
            self.pub_joint_hold.publish(Float32MultiArray(
                data=pack_joint_hold_target(SEARCH_Q)))
            return

        self.arm_step += 1

        if ap == "hover":
            done = self._move_l(EE_SPEED_HOVER)
            if done:
                # hover에도 물리 도달 확인이 필요하다 — 여기서 뒤처진 채로 detect에
                # 들어가면 pre/grasp가 통째로 밀려서 결국 덜 문 채로 닫힌다
                # (2026-09-20 실기: grasp 진입 시점에 이미 134mm 뒤처져 있었음).
                err_h = ((self.ee_pose - self.ml_target)
                         if self.ee_pose is not None else np.zeros(3))
                dist_h = float(np.linalg.norm(err_h))
                if dist_h > PRE_ARRIVE_TOL and self.hover_wait_n < self.arrive_max_wait:
                    self.hover_wait_n += 1
                    if self.hover_wait_n % 180 == 0:
                        print(f"    [HOVER 대기] 잔여 {dist_h*1000:.1f}mm "
                              f"({self.hover_wait_n}/{self.arrive_max_wait}스텝)")
                    return
                if dist_h > PRE_ARRIVE_TOL:
                    print(f"    ★ [팔] hover 물리도달 실패(잔여 {dist_h*1000:.1f}mm, "
                          f"{self.arrive_max_wait}스텝 대기 초과) — 그냥 진행")
                else:
                    print(f"    [hover 도달오차] {dist_h*1000:.1f}mm")
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
                if dist_p > PRE_ARRIVE_TOL and self.pre_wait_n < self.arrive_max_wait:
                    self.pre_wait_n += 1
                    return  # 계속 목표를 향해 대기
                if dist_p > PRE_ARRIVE_TOL:
                    print(f"    ★ [팔] pre 물리도달 실패(잔여 {dist_p*1000:.1f}mm, "
                          f"{self.arrive_max_wait}스텝 대기 초과) — 그냥 진행")
                else:
                    print(f"    [pre 도달오차] {dist_p*1000:.1f}mm")
                if self.pre_try_n:
                    print(f"    [PRE 검출률] {self.pre_try_n-self.pre_miss_n}/{self.pre_try_n} "
                          f"({100.0*(self.pre_try_n-self.pre_miss_n)/self.pre_try_n:.0f}%)  "
                          f"미검출 {self.pre_miss_n}회, 최장연속 {self.pre_miss_max}회")
                if self.pre_corr_n:
                    drift = self.ml_grasp - self.ml_grasp_anchor
                    print(f"    [PRE 재검출] 보정 {self.pre_corr_n}회 "
                          f"(anchor대비 최대 {self.pre_corr_max*1000:.1f}mm, "
                          f"직전대비 마지막 {self.pre_corr_last*1000:.1f}mm)  기각 {self.pre_rej_n}회")
                    print(f"    [PRE 파지점 이동] 최초 {self.ml_grasp_anchor.round(4)} → "
                          f"최종 {self.ml_grasp.round(4)}  Δ=({drift[0]*1000:+.1f},"
                          f"{drift[1]*1000:+.1f},{drift[2]*1000:+.1f})mm")
                print(f"  [팔] pre-grip 도달 → 감속 하강")
                self.ml_start = (self.ee_pose.copy() if self.ee_pose is not None
                                  else self.ml_target.copy())
                self.ml_target = self.ml_grasp.copy()
                self.gr_eih_try = 0; self.gr_eih_miss = 0
                self.gr_eih_miss_run = 0; self.gr_eih_miss_max = 0
                self.gr_eih_tail = []
                self.gr_eih_corr_n = 0; self.gr_eih_rej_n = 0
                self.gr_eih_corr_max = 0.0; self.gr_eih_corr_last = 0.0
                self.gr_z_clamp_n = 0; self.gr_z_clamp_max = 0.0
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
                not_there = (dist > self.grasp_arrive_tol
                             or abs(err[2]) > self.grasp_arrive_z_tol)
                if not_there and self.grasp_wait_n < self.arrive_max_wait:
                    self.grasp_wait_n += 1
                    return  # 그리퍼 닫지 않고 계속 목표를 향해 대기
                if not_there:
                    print(f"    ★ [팔] 파지점 물리도달 실패(잔여 {dist*1000:.1f}mm, "
                          f"z {err[2]*1000:+.1f}mm, {self.arrive_max_wait}스텝 대기 초과) "
                          f"— 그냥 진행")
                print(f"    [파지 실제오차] xyz=({err[0]*1000:+.1f},{err[1]*1000:+.1f},"
                      f"{err[2]*1000:+.1f})mm 거리={dist*1000:.1f}mm ee_pose={self.ee_pose.round(4) if self.ee_pose is not None else None} "
                      f"ml_grasp={self.ml_grasp.round(4)}")
                # "덜 내려갔다"의 원인을 두 갈래로 갈라주는 줄 — z 잔여는 "명령한 z에
                # 팔이 실제로 도달했는가"(제어 오차)지, 기하가 맞는가와는 별개다.
                #   z잔여 ≈ 0 인데 눈으로 덜 내려갔으면 → 기하 편향 → ee_grip_offset을 줄인다
                #   z잔여 > 0 이면 → 아직 못 내려온 것 → 허용오차/대기/속도 문제
                if self.gr_z_clamp_n:
                    print(f"    ★ [z하한 물림] 근접 재검출이 더 낮은 z를 {self.gr_z_clamp_n}회 "
                          f"요구했으나 최초검출(anchor) 하한에 막힘 — 최대 "
                          f"{self.gr_z_clamp_max*1000:.1f}mm 덜 내려감. "
                          f"grasp_z_below_anchor_max(현재 "
                          f"{self.grasp_z_below_anchor_max*1000:.0f}mm)를 키우거나 "
                          f"hover 원거리 검출의 z 편향을 볼 것")
                print(f"    [깊이진단] 명령 z잔여 {err[2]*1000:+.1f}mm "
                      f"(허용 {self.grasp_arrive_z_tol*1000:.1f}mm)  "
                      f"설계 파지깊이 = 마커면 아래 {self.grasp_depth_extra*1000:.1f}mm  "
                      f"눈으로 δmm 덜 물렸으면 ee_grip_offset을 "
                      f"{self.ee_grip_offset:.4f}−δ 로")
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
                if self.grasp_eih_track:
                    print(f"    [GRASP 손목캠 재검출] 반영 {self.gr_eih_corr_n}회 "
                          f"(최대 {self.gr_eih_corr_max*1000:.1f}mm, "
                          f"마지막 {self.gr_eih_corr_last*1000:.1f}mm)  "
                          f"기각 {self.gr_eih_rej_n}회  최종 파지점 {self.ml_grasp.round(4)}")
                print(f"  [팔] 파지점 도달 → grip 단계로")
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

        elif ap == "grip":
            # 닫기 명령을 grip 첫 틱으로 미뤘다 — step_confirm이 켜져 있으면 Enter 전엔 안 닫힌다.
            if self.arm_step == 1:
                print(f"  [팔] 그리퍼 닫기 명령 전송")
                self.pub_gripper_cmd.publish(Bool(data=True))
            if self.grip_contact_result is not None:
                print(f"  [팔] 그립 완료({'접촉 감지' if self.grip_contact_result else '헛집음 가능성'}) "
                      f"→ 들어올리기")
                self.ml_start = (self.ee_pose.copy() if self.ee_pose is not None
                                  else self.ml_target.copy())
                self.ml_target = self.ml_start - GRASP_APPROACH_DIR * LIFT_HEIGHT
                self.lift_bz_max = None
                self.lift_wait_n = 0
                self.verify_hits = 0; self.verify_low_hits = 0
                self.verify_high_hits = 0; self.verify_bz_last = None
                # 그리퍼 판정 기준: 그립 직후 개구부(최근 관측 중앙값 — 단발 튐 방어).
                # 리프트 중 이보다 더 닫히면 물체가 빠져나간 것이다.
                self.grip_stroke_at_grip = (float(np.median(self.stroke_recent))
                                             if self.stroke_recent else None)
                self.grip_stroke_min = None; self.grip_v_n = 0
                if self.grip_stroke_at_grip is not None:
                    print(f"       [판정기준] 그립 직후 개구부 "
                          f"{self.grip_stroke_at_grip:.1f}mm (물체폭 "
                          f"{self.obj_width_m*1000:.1f}mm + 오프셋 "
                          f"{self.grip_stroke_offset:.1f}mm 기대)")
                self.arm_phase = "lift"; self.arm_step = 0

        elif ap == "lift":
            done = self._move_l(EE_SPEED_LIFT)
            if done:
                # 리프트가 실제로 다 올라가기 전에 verify로 넘어가면 물체가 아직
                # 바닥에 닿아 있을 수 있다 — 판정 전에 물리 도달을 확인한다.
                if not self._wait_physical_arrive("lift_wait_n", PRE_ARRIVE_TOL, "lift"):
                    return
                #   차체캠으로 직접 판정한다(시뮬 전용 god's-eye
                #   좌표를 안 쓴다 — 실물 이식을 위해). 판정 근거는 리프트가
                #   올라가는 동안 이미 모아둔 관측(lift_bz_max)이다.
                self.arm_phase = "verify"; self.arm_step = 0

        elif ap == "verify":
            verdict = self._verify()
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
                print(f"  [팔] PLACE 하강 완료 → 릴리즈 단계로")
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
                                   - self.place_approach_dir * (HOVER_STANDOFF + self.ee_grip_offset))
                self.place_hover_wait_n = 0
                self.arm_phase = "place_hover"; self.arm_step = 0

        elif ap == "place_hover":
            done = self._move_l(EE_SPEED_HOVER)
            if done:
                # 여기서 뒤처진 채 검출하면 place 마커를 예상보다 먼 거리에서 재게 된다.
                if not self._wait_physical_arrive("place_hover_wait_n",
                                                   PRE_ARRIVE_TOL, "place_hover"):
                    return
                print(f"  [팔] PLACE HOVER 도달 → 상판 마커 검출")
                self.ml_place_anchor = None
                self.arm_phase = "place_detect"; self.arm_step = 0

        elif ap == "place_detect":
            # pick의 hover→detect와 동일 패턴 — 손목캠으로 상판 마커(ID6)를 보고 놓는점을 계산
            if self.place_top_new is not None:
                pt = self._place_point(self.place_top_new)
                print(f"\n  [팔] 상판 마커 검출 ✓ (몸체) {self.place_top_new.round(4)}")
                if self.place_point_mode == "marker_inset":
                    _mkw = self.place_top_new + np.array([0.0, 0.0, self.body_link_world_z])
                    _u = self._place_inset_dir(_mkw)
                    _yaw_s = ("-" if self.place_top_yaw is None
                              else f"{np.degrees(self.place_top_yaw):+.1f}deg")
                    print(f"       오프셋 {self.place_inset*1000:.0f}mm "
                          f"방향=({_u[0]:+.3f},{_u[1]:+.3f}) "
                          f"모드={self.place_inset_mode} 마커yaw={_yaw_s}")
                    # 마커를 로봇 정면에 반듯이 붙이면 marker_y와 radial이 거의 겹쳐서
                    # 어느 모드가 먹었는지 구분이 안 된다 — 세 후보를 같이 찍는다.
                    _v = _mkw[:2]; _n = float(np.linalg.norm(_v))
                    _cand = {
                        "marker_y": (None if self.place_top_yaw is None else
                                      np.array([-np.cos(self.place_top_yaw),
                                                -np.sin(self.place_top_yaw)])),
                        "body_y": np.array([0.0, -1.0]),
                        "radial": (-_v/_n if _n > 1e-6 else np.zeros(2)),
                    }
                    _parts = []
                    for _k, _d in _cand.items():
                        if _d is None:
                            _parts.append(f"{_k}=미수신"); continue
                        _diff = np.degrees(np.arccos(np.clip(float(_d @ _u), -1.0, 1.0)))
                        _parts.append(f"{_k}=({_d[0]:+.3f},{_d[1]:+.3f}) Δ{_diff:.1f}°")
                    print(f"       방향후보 " + "  ".join(_parts))
                print(f"       놓는점 {pt.round(4)}  (사전 추정치는 "
                      f"{self._place_point_from_guess().round(4)})")
                self.ml_place_anchor = pt.copy()
                self.ml_start = (self.ee_pose.copy() if self.ee_pose is not None
                                  else self.ml_target.copy())
                self.ml_target = pt
                self.place_wait_n = 0
                self.arm_phase = "place_descend"; self.arm_step = 0
            elif self.arm_step > PLACE_DETECT_TIMEOUT:
                print(f"  ★ 팔: place 마커 미검출({PLACE_DETECT_TIMEOUT}스텝) — "
                      f"사전 추정치로 하강 진행")
                pt = self._place_point_from_guess()
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
            # 쥔 물체가 마커를 가리면 근접 재검출이 오히려 해가 된다(부분 오클루전
            # PnP는 코너 일부만 보고 엉뚱한 자세를 낸다) — freeze면 최초 검출점으로
            # 그대로 내려간다.
            if self.place_freeze_after_detect and self.arm_step == 1:
                print(f"    [PLACE] 맹목 하강 모드 — 최초 검출 놓는점 고정 "
                      f"{self.ml_target.round(4)}")
            if (not self.place_freeze_after_detect
                    and self.place_top_new is not None and self.ml_place_anchor is not None):
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
                # 여기서 덜 내려가면 릴리즈 높이가 그만큼 커진다 — grasp와 같은 기준 적용.
                if ((dist > self.grasp_arrive_tol or abs(err[2]) > self.grasp_arrive_z_tol)
                        and self.place_wait_n < self.arrive_max_wait):
                    self.place_wait_n += 1
                    return
                print(f"  [팔] PLACE 목표 도달(잔여 {dist*1000:.1f}mm) → 릴리즈 단계로")
                self.arm_phase = "place_release"; self.arm_step = 0

        elif ap == "place_release":
            # 여는 명령을 place_release 첫 틱으로 미뤘다 — grip 단계와 같은 이유로,
            # step_confirm이 켜져 있으면 Enter 전엔 물체를 놓지 않는다(놓기 직전에
            # 눈으로 확인하고 중단할 수 있어야 한다).
            if self.arm_step == 1:
                print(f"  [팔] 그리퍼 열기 명령 전송")
                self.pub_gripper_cmd.publish(Bool(data=False))
            if self.arm_step > 60:
                # 놓자마자 관절목표(SEARCH_Q)로 튀면 손가락이 방금 놓은 물체를 친다 —
                # marker_inset 경로는 수직으로 먼저 빠진 뒤 복귀한다.
                if self.place_point_mode == "marker_inset":
                    print(f"  [팔] 릴리즈 완료 → 수직 후퇴 {PLACE_RETREAT_DIST*1000:.0f}mm")
                    self.ml_start = (self.ee_pose.copy() if self.ee_pose is not None
                                      else self.ml_target.copy())
                    self.ml_target = self.ml_start + np.array([0.0, 0.0, PLACE_RETREAT_DIST])
                    self.retreat_wait_n = 0
                    self.arm_phase = "place_retreat"; self.arm_step = 0
                else:
                    print(f"  [팔] 릴리즈 완료 → 초기 자세로 복귀")
                    self.arm_phase = "place_home"; self.arm_step = 0

        elif ap == "place_retreat":
            done = self._move_l(EE_SPEED_LIFT)
            if done:
                # 여기서 안 기다리면 실제로 10mm쯤만 뜬 채 place_home(관절 복귀)이
                # 시작돼 손가락이 방금 놓은 물체를 스친다(2026-09-20 실기에서 확인).
                if not self._wait_physical_arrive("retreat_wait_n", PRE_ARRIVE_TOL,
                                                   "place_retreat"):
                    return
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
        # launch가 SIGINT를 보내면 rclpy 시그널 핸들러가 이미 컨텍스트를 내려서
        # 여기서 또 부르면 RCLError를 뱉는다(동작엔 영향 없지만 매번 traceback).
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
