import math
import numpy as np

# =============================================================================
# 기하 상수 — 팔 접근 자세, 마커 치수, 좌표 규약
#
# 여기 값은 여러 노드가 공유하므로 한 곳에서만 정의한다. 실기에서 실제로 쓰는
# 값 중 상당수는 launch 인자로 덮어쓰이며(ee_grip_offset, obj_expected_* 등),
# 그때 이 파일의 값은 "런치 인자를 안 줬을 때의 기본값" 역할만 한다.
# =============================================================================

BODY_LINK_WORLD_Z = 0.277   # [m] base_link의 world Z — world 절대높이 비교 시 이만큼 빼야 함
                            #     실물은 body_link=팔 베이스라 launch에서 0.0으로 덮어쓴다
OBJ_S = 0.05    # [m] 타겟 큐브 한 변 — 실물 타겟은 obj_width_m/obj_height_m 인자로 준다

KEEP_DIST = 0.42     # [m] 차체-타겟 거리. joint5 여유가 좁아 함부로 늘리면 IK 도달 불가

# HOVER_PITCH_DEG / KEEP_DIST는 세트로만 유효하다 — joint5 특이점 회피, link6 하우징
# 간섭 회피, IK 수렴이라는 3개 제약이 얽혀 있어 하나만 바꾸면 나머지가 깨진다.
HOVER_PITCH_DEG = 0.0   # [deg] hover 자세의 팔 접근 pitch (0=완전수직)
_hp = math.radians(HOVER_PITCH_DEG)
HOVER_APPROACH_DIR = np.array([0.0, math.sin(_hp), -math.cos(_hp)])
HOVER_STANDOFF = 0.138  # [m] hover 시 그리퍼 끝단-물체 "윗면" 거리 (= 0.125/cos(HOVER_PITCH_DEG))

# [m] EndPose 명령 기준점 → 그리퍼 손끝 거리. 물리적 손가락 길이가 아니라 펌웨어
# EndPose 기준점과 URDF link6 원점의 불일치까지 합친 값이다 — 그래서 joint7
# 장착점(135.8mm)보다 짧은 값이 나올 수 있다. 실기 확정값은 launch가 준다.
EE_GRIP_OFFSET = 0.135

HOLD_MAX = 12        # [스텝] 카메라 관측이 이보다 오래 끊기면 값을 못 믿는다
ARM_BASE_YAW_DEG = 90.0   # [deg] 팔이 차체 기준 90도 돌아서 장착됨 — EndPoseCtrl XY 변환에 쓴다

TOOL_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])    # PiPER link6 로컬 +Z축
GRIPPER_DOWN    = np.array([0.0, 0.0, -1.0])   # 그리퍼가 향해야 하는 world 방향(아래)

GRASP_PITCH_DEG = 0.0   # [deg] grasp 자세의 팔 접근 pitch — HOVER_PITCH_DEG와 같은 제약
_gp = math.radians(GRASP_PITCH_DEG)
GRASP_APPROACH_DIR = np.array([0.0, math.sin(_gp), -math.cos(_gp)])   # world 기준 접근방향(단위벡터)

# place 전용 pitch. 선반이 픽보다 높으면 픽과 같은 수직접근(0°)으로는 IK가 안 풀려
# 기울여야 하는 경우가 있다. 실기는 launch의 place_pitch_deg로 주며, arm_node가
# 그 값으로 접근방향 벡터를 직접 만든다.
PLACE_PITCH_DEG = 30.0

SEARCH_Q = np.radians([0, 45, -90, 0, 45, 0])   # 대기/중립 자세
PIPER_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

# =============================================================================
# 마커
# =============================================================================
MARKER_SIZE = 0.035    # [m] 타겟 옆면 마커(ID0~3) 한 변 — 차체캠이 본다
MARKER_FACES = [0, 1, 2, 3]
FACE_NORMAL = {0:(1.,0.,0.), 1:(0.,1.,0.), 2:(-1.,0.,0.), 3:(0.,-1.,0.)}

TOP_MARKER_ID   = 4      # 실기 픽 마커 ID는 launch의 eih_pick_marker_id가 준다
TOP_MARKER_SIZE = 0.020  # [m] 실기 크기도 launch의 eih_pick_marker_size_m가 준다

BOARD_T, CELL_T, CELL_GAP = 0.001, 0.0005, 0.0003   # 마커가 인쇄된 보드/셀 두께

# 물체 중심 → 마커 평면까지의 거리. 손목캠은 물체 "중심"이 아니라 이만큼 위의
# 마커 평면을 재므로, 파지점을 물체 중심 기준으로 잡으려면 이 값을 빼야 한다.
MK_TOP_OFFSET = OBJ_S/2.0 + BOARD_T + CELL_GAP + CELL_T/2.0

# place 선반 마커 — 전면(차체캠용)/상판(손목캠용)
PLACE_SIDE_MARKER_ID = 5   # 선반 전면, 차체캠이 봄
PLACE_TOP_MARKER_ID  = 6   # 선반 상판, 손목캠이 봄. 실기 ID는 launch가 준다

SHELF_TOP_D    = 0.40   # [m] 상판 안길이(전면 마커가 이 면에 붙음)
SHELF_PLATE_T  = 0.04   # [m] 상판 두께
MK_SHELF_FRONT_OFFSET = SHELF_TOP_D/2.0 + BOARD_T + CELL_GAP + CELL_T/2.0

SHELF_FRONT_MARKER_SIZE = 0.10   # [m] 원거리 검출용 전면 마커(ID5) — 다리 틈에 맞춘 최대치

# 원거리용 큰 마커(ID5)는 상판 아래, 근접용 작은 마커(ID7)는 상판 위. 각각의
# 위치에서 상판 모서리에 가리지 않고 화면 잘림도 없도록 실측으로 확정한 배치다.
PLACE_NEAR_MARKER_ID   = 7
PLACE_NEAR_MARKER_SIZE = 0.04   # [m] 근접(0.13m)까지 붙어도 화면에 안 잘림
SHELF_FAR_MARKER_X  = 0.0       # 둘 다 가운데 정렬 — 높이로 분리되므로 겹치지 않는다
SHELF_NEAR_MARKER_X = 0.0
_near_board = PLACE_NEAR_MARKER_SIZE*4.0/3.0
_far_board  = SHELF_FRONT_MARKER_SIZE*4.0/3.0
_mk_gap = 0.015                                       # 두 마커 사이 여백(아르코 quiet zone)
SHELF_NEAR_MARKER_Z = SHELF_PLATE_T/2.0 - _near_board/2.0
SHELF_FAR_MARKER_Z  = (SHELF_NEAR_MARKER_Z - _near_board/2.0 - _mk_gap) - _far_board/2.0

# hover가 처음 향하는 "물체 대략 위치"의 기본값. 실기에서는 launch의
# obj_expected_*_body가 항상 덮어쓰므로 이 값 자체는 쓰이지 않는다.
OBJ_CENTER_BODY = (0.0, KEEP_DIST, 0.20 + 0.02/2 + OBJ_S/2.0)

# place 마커가 대략 있을 body_link 상대위치의 기본값. arm_node의 place_hover가
# 여기로 먼저 접근한 뒤 place_detect가 실제 마커 재검출로 정밀 보정한다.
# 실기에서는 launch의 place_expected_*_body가 덮어쓴다.
PLACE_CENTER_BODY_GUESS = (0.20, KEEP_DIST, OBJ_CENTER_BODY[2])

# solvePnP용 코너 모델 — 마커마다 크기가 다르므로 제 것을 써야 한다
# (큰 마커에 작은 모델을 쓰면 거리가 그 비율만큼 통째로 틀어진다).
_HALF = MARKER_SIZE / 2.0
OBJ_PTS = np.array([[-_HALF,_HALF,0.],[_HALF,_HALF,0.],
                    [_HALF,-_HALF,0.],[-_HALF,-_HALF,0.]], float)
_HSF = SHELF_FRONT_MARKER_SIZE / 2.0
SHELF_FRONT_PTS = np.array([[-_HSF,_HSF,0.],[_HSF,_HSF,0.],
                            [_HSF,-_HSF,0.],[-_HSF,-_HSF,0.]], float)
_HSN = PLACE_NEAR_MARKER_SIZE / 2.0
PLACE_NEAR_PTS = np.array([[-_HSN,_HSN,0.],[_HSN,_HSN,0.],
                           [_HSN,-_HSN,0.],[-_HSN,-_HSN,0.]], float)
_HT = TOP_MARKER_SIZE / 2.0
TOP_PTS = np.array([[-_HT,_HT,0.],[_HT,_HT,0.],
                    [_HT,-_HT,0.],[-_HT,-_HT,0.]], np.float32)

# =============================================================================
# §1. 수학 헬퍼 (값 절대 바꾸지 말 것)
# =============================================================================

def _wrap(a):  # 각도를 -π~+π로 되감는다
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def _quat_to_R(q):  # 쿼터니언(w,x,y,z) → 3x3 회전행렬
    w,x,y,z = [float(v) for v in q]
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], float)

def _quat_mul(a, b):  # 쿼터니언 곱 — 회전을 이어 붙인다
    """쿼터니언 곱 (w,x,y,z). a 다음 b를 '몸체 기준'으로 적용."""
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2], float)

def _quat_axis_angle(axis, deg):  # 축과 각도[deg] → 쿼터니언
    """축-각 → 쿼터니언 (w,x,y,z)."""
    a = np.array(axis, float); a = a/np.linalg.norm(a)
    h = math.radians(deg)/2.0
    return np.array([math.cos(h), *(a*math.sin(h))], float)

def quat_from_two_vec(u, v):  # 벡터 u를 v로 돌리는 최소 회전 — 그리퍼를 아래로 향하게 할 때 쓴다
    u = u/np.linalg.norm(u); v = v/np.linalg.norm(v)
    d = float(np.dot(u,v))
    if d < -0.999999:
        ax = np.cross([1,0,0], u)
        if np.linalg.norm(ax) < 1e-6: ax = np.cross([0,1,0], u)
        ax /= np.linalg.norm(ax); return np.array([0.0,*ax])
    ax = np.cross(u,v); q = np.array([1.0+d,*ax]); return q/np.linalg.norm(q)

# =============================================================================
# §2. 마커 기하 — 순수 함수. vision_node가 씬 접근 없이 마커 면의 자세/오프셋을
# 상수만으로 재구성할 수 있게 한다.
# =============================================================================

def _plate_basis(n):  # 면 법선 n으로부터 그 면의 좌표축 3개를 만든다
    up = np.array([0.,0.,1.])
    if abs(float(n[2])) > 0.95: up = np.array([0.,1.,0.])
    y = up - float(np.dot(up,n))*n
    y /= np.linalg.norm(y)
    x = np.cross(y,n)
    return np.array([x,y,n]).T

def compute_marker_faces():  # 물체 옆면 마커 4개의 자세·오프셋 — 차체캠 경로 전용(현재 미사용)
    """타겟 옆면 마커 4개의 {fid: (R_cp, off)} — R_cp는 면 기준 자세,
    off는 물체 중심에서 마커 평면까지의 벡터."""
    h = OBJ_S/2.0
    d_mk = h + BOARD_T + CELL_GAP + CELL_T/2.0
    faces = {}
    for fid in MARKER_FACES:
        n = np.array(FACE_NORMAL[fid], float)
        R_cp = _plate_basis(n)
        off = d_mk*n
        faces[fid] = (R_cp, off)
    return faces

# =============================================================================
# §3. 토픽 페이로드 패킹/언패킹 — Float32MultiArray 스키마를 한 곳에서만
# 정의한다(여러 노드가 각자 순서를 외워서 어긋나는 사고 방지).
# =============================================================================

def pack_chassis_pose(bx, by, phi, valid, marker_id, sim_step=0, bz=0.0):  # 차체캠 결과 → Float32MultiArray 페이로드
    """vision → arm. [bx,by,phi,valid,marker_id,sim_step,bz]
    bz: 물체 중심 높이(몸체좌표, 파지 성공 판정용)."""
    return [float(bx), float(by), float(phi), 1.0 if valid else 0.0,
            float(marker_id), float(sim_step), float(bz)]

def unpack_chassis_pose(data):  # 위의 역 — 받는 쪽에서 푼다
    bx, by, phi, valid, marker_id, sim_step, bz = data
    return (float(bx), float(by), float(phi), bool(valid), int(marker_id),
            int(sim_step), float(bz))

def pack_eih_marker(x, y, z, valid, yaw=0.0):  # 손목캠 마커 위치/yaw → 페이로드
    """vision → arm. [x,y,z,valid,yaw]
    yaw: 마커 자신의 +Y축을 body XY 평면에 투영한 방향[rad]. place에서 "마커 기준
    어느 쪽으로 비켜 놓을지"를 정할 때 쓴다(위치만으로는 마커의 방향을 알 수 없다)."""
    return [float(x), float(y), float(z), 1.0 if valid else 0.0, float(yaw)]

def unpack_eih_marker(data):  # 위의 역 — arm_node가 파지점 계산에 쓴다
    # yaw는 나중에 추가된 필드 — 옛 4개짜리 메시지도 그대로 받는다.
    x, y, z, valid = data[0], data[1], data[2], data[3]
    yaw = float(data[4]) if len(data) > 4 else 0.0
    return float(x), float(y), float(z), bool(valid), yaw

def pack_grip_state(stroke_mm, effort, contact, holding, active):  # 그리퍼 개구부·접촉 상태 → 페이로드
    """gripper → arm. [stroke_mm, effort, contact, holding, active]
    stroke_mm: 현재 그리퍼 개구부[mm]. effort: 접촉 저항 크기[N·m].
    파지 성공 판정(arm_node._verify_by_gripper)이 이 셋을 같이 본다."""
    return [float(stroke_mm), float(effort),
            1.0 if contact else 0.0, 1.0 if holding else 0.0, 1.0 if active else 0.0]

def unpack_grip_state(data):  # 위의 역 — arm_node의 파지 판정 입력
    stroke_mm, effort, contact, holding, active = data
    return (float(stroke_mm), float(effort), bool(contact), bool(holding), bool(active))

def pack_joint_hold_target(q6):  # 관절 목표 6개 → 페이로드
    """arm → driver. 관절 직접 지령 목표 6개."""
    return [float(v) for v in q6]

def unpack_joint_hold_target(data):  # 위의 역 — driver가 MOVE J로 적용
    return np.array(data, float)
