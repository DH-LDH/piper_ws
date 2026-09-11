import math
import numpy as np

# =============================================================================
# [SCENE] 씬 구성 파라미터 (plant_node.py 전용, 다른 노드도 참조 가능)
# =============================================================================
OMEGA_BELT, R_BELT = 0.30, 0.55
DISC_R, DISC_H, DISC_Z = 0.65, 0.02, 0.312   # [m] 턴테이블 반지름/두께/설치높이(추정값)
BODY_LINK_WORLD_Z = 0.277   # [m] base_link의 world Z — world 절대높이 비교 시 이만큼 빼야 함
DISC_M, DISC_KD, DISC_MU = 8.0, 1.0e6, 0.60
OBJ_S, OBJ_M, OBJ_MU = 0.05, 0.20, 0.60   # [m] 타겟 큐브 크기/질량/마찰 — 그리퍼 개구부 대비 여유 있게 축소
PHI_REF = -math.pi / 2.0
WHEELBASE, TRACK, WHEEL_R = 0.494, 0.364, 0.100
DRIVE_MU = 1.0        # 4바퀴 전부 구동륜(캐스터 없음)
STEER_MAX = math.pi / 2.0   # [rad] 조향각 한계 (실물 스펙 max_steer_angle_parallel=90°)
WHEEL_KD, MAX_FORCE, MAX_JVEL = 1.0e4, 1.0e5, 1000.0
STEER_KS, STEER_KD, STEER_MF = 3.0e5, 3.0e3, 5.0e4
STEER_JOINTS = ["fl_steering_joint", "fr_steering_joint",
                "rl_steering_joint", "rr_steering_joint"]
DRIVE_JOINTS = ["fl_wheel_joint", "fr_wheel_joint",
                "rl_wheel_joint", "rr_wheel_joint"]
# 바퀴 위치(몸체 x,y) — steer/drive 리스트와 같은 순서
WHEEL_XY = [( WHEELBASE/2,  TRACK/2), ( WHEELBASE/2, -TRACK/2),
            (-WHEELBASE/2,  TRACK/2), (-WHEELBASE/2, -TRACK/2)]

KEEP_DIST = 0.42     # [m] AMR-타겟 거리(차체캠 화각 확보용). joint5 여유가 좁아 함부로 늘리면 IK 도달 불가(vault 참고)
# HOVER_PITCH_DEG/KEEP_DIST/CONV_Z는 세트로만 유효 — joint5 특이점 회피, link6 하우징 간섭 회피,
# IK 수렴이라는 3개 제약이 얽혀 있다. 바꿀 때는 probe_fk.py로 FK 오차까지 재검증할 것(vault 2026-09-03 참고).
HOVER_PITCH_DEG = 0.0   # [deg] hover 자세의 팔 접근 pitch (0=완전수직)
_hp = math.radians(HOVER_PITCH_DEG)
HOVER_APPROACH_DIR = np.array([0.0, math.sin(_hp), -math.cos(_hp)])
HOVER_STANDOFF = 0.138  # [m] hover 시 그리퍼 끝단-물체 "윗면" 거리 (= 0.125/cos(HOVER_PITCH_DEG))

EE_GRIP_OFFSET = 0.135  # [m] link6(EE_FRAME) 원점 → 그리퍼 끝단 거리. IK 목표 계산 시 이만큼 더해야 함

HOVER_Q_VERIFIED = np.radians([0.0, 130.02, -99.99, 0.0, 60.0, 0.0])  # GUI 수동 튜닝으로 확정한 고정 hover 관절값(deg)

BY_FAIL = 0.20       # [m] 이보다 가까워지면 충돌 위험으로 중단 (plant_node)
HOLD_MAX = 12        # [스텝] 카메라 관측이 이보다 오래 끊기면 값을 못 믿는다

ARM_BASE_LOCAL   = np.array([-0.01858, 0.02232, 0.04462])   # [m] 팔 베이스 위치, Ranger Mini 상판 기준(뷰포트 실측)
ARM_BASE_YAW_DEG = 90.0   # [deg]
EE_FRAME = "link6"

ARM_KS, ARM_KD, ARM_MF = 1.0e6, 1.0e4, 1.0e5
GRIP_KS, GRIP_KD, GRIP_MF = 1.0e5, 1.0e3, 1.0e3

GRIP_JOINT_L, GRIP_JOINT_R = "joint7", "joint8"
GRIP_STROKE_MAX = 0.05   # [m] 그리퍼 최대 개구부(joint7 USD limit 실측값, 그리퍼 폭 100mm)

TOOL_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])   # PiPER link6 로컬 +Z축(실측)
GRIPPER_DOWN    = np.array([0.0, 0.0, -1.0])   # 그리퍼가 향해야 하는 world 방향(아래)
DOWN_FLIP_AXIS_DEG = 0.0   # [deg] 위 두 축이 정반대라 뒤집는 회전축이 하나로 안 정해짐 — world XY 각도로 직접 지정

GRASP_PITCH_DEG = 0.0   # [deg] grasp 자세의 팔 접근 pitch — HOVER_PITCH_DEG와 같은 제약 적용
_gp = math.radians(GRASP_PITCH_DEG)
GRASP_APPROACH_DIR = np.array([0.0, math.sin(_gp), -math.cos(_gp)])   # world 기준 접근방향(단위벡터)

# place(선반에 내려놓기) 전용 pitch — 픽의 pitch=0을 그대로 쓰면 IK가 100% 실패한다(선반이
# 픽보다 world z 40~50mm 높아 기하가 다름). 10~50° 범위에서만 풀리고 30°가 가장 여유
# 있다(vault 2026-09-10 참고). place 전 구간(hover/detect/descend/release/retreat)이 공유.
PLACE_PITCH_DEG = 30.0
_pp = math.radians(PLACE_PITCH_DEG)
PLACE_APPROACH_DIR = np.array([0.0, math.sin(_pp), -math.cos(_pp)])   # world 기준 접근방향(단위벡터)

SEARCH_Q = np.radians([0, 45, -90, 0, 45, 0])
PIPER_JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']

# =============================================================================
# 카메라
# =============================================================================
BASE_LINK = "/World/Ranger/Geometry/base_link"
ARM_MOUNT = BASE_LINK + "/arm_mount"
CAM_RIG   = BASE_LINK + "/rs_mount"

RS_RES = (1280, 800)
RS_NEAR_CLIP = 0.03
CAM_POS_LOCAL = (0.0, 0.29, -0.05)   # [m] 차체캠 로컬 위치 — z는 물체 중심과 눈높이를 맞춘 값(vault 2026-09-04)

CAM_FORWARD_ONLY = True   # 차량 정면(base_link +X)을 바라봄 (턴테이블 방향 사선 시야 대신)
CAM_TILT_DEG = 0   # [deg] 카메라 틸트 — 사용자 지시로 수평 고정(정면 응시)

EIH_RES = (640, 480)
EIH_FOCAL_MM = 1.93
EIH_HAPER_MM = 3.896

EIH_MOUNT_POS       = (-0.0115, 0.0731, 0.05)     # [m] 손목캠 로컬 위치(실측)
EIH_MOUNT_QUAT_WXYZ = (0.08001329, 0.02508435, 0.79368112, -0.60252703)   # 손목캠 로컬 방향(실측)

# =============================================================================
# 마커
# =============================================================================
MARKER_SIZE = 0.035    # [m] 옆면 마커(ID0~3) 한 변 길이 — OBJ_S(타겟 크기)에 맞춘 비율
MARKER_FACES = [0, 1, 2, 3]
FACE_NORMAL = {0:(1.,0.,0.), 1:(0.,1.,0.), 2:(-1.,0.,0.), 3:(0.,-1.,0.)}

TOP_MARKER_ID   = 4
TOP_MARKER_SIZE = 0.020   # [m] 윗면 마커(ID4) 한 변 길이

BOARD_T, CELL_T, CELL_GAP = 0.001, 0.0005, 0.0003

# 물체 중심 → 마커 평면까지의 거리(plant_node.build_marker()의 d_mk/d_top과 같은 식).
# 손목캠은 물체 "중심"이 아니라 이만큼 위의 마커 평면을 재므로, 파지점을 물체 중심
# 기준으로 잡으려면 이 값을 빼야 한다.
MK_TOP_OFFSET = OBJ_S/2.0 + BOARD_T + CELL_GAP + CELL_T/2.0
COL_WHITE, COL_BLACK = (0.9,0.9,0.9), (0.02,0.02,0.02)
MK_ROOT = "/World/target/marker"
DOME_INTENSITY = 400.0

# place 선반(벨트와 분리된 고정 선반) — 전면 마커(차체캠용)/상판 마커(손목캠용)
PLACE_SHELF_PATH = "/World/place_shelf"
PLACE_SIDE_MARKER_ID = 5   # 선반 전면(AMR 접근 방향), 차체캠이 봄
PLACE_TOP_MARKER_ID  = 6   # 선반 상판, 손목캠이 봄

SHELF_TOP_W    = 0.55   # [m] 상판 폭(로컬 X)
SHELF_TOP_D    = 0.40   # [m] 상판 안길이(로컬 Y, 전면 마커가 이 면에 붙음)
SHELF_PLATE_T  = 0.04   # [m] 상판 두께
SHELF_LEG_SIDE = 0.08   # [m] 다리 단면
MK_SHELF_TOP_OFFSET   = SHELF_PLATE_T/2.0 + BOARD_T + CELL_GAP + CELL_T/2.0
MK_SHELF_FRONT_OFFSET = SHELF_TOP_D/2.0   + BOARD_T + CELL_GAP + CELL_T/2.0

SHELF_FRONT_MARKER_SIZE = 0.10   # [m] 다리 사이 전면 마커(ID5) 크기 — 원거리 검출용, 다리 틈에 맞춘 최대치

# AMR의 실제 yaw는 런마다 달라질 수 있어(교정 수단 없음) 아래 world 좌표는 "초기 스폰용
# 잠정값"일 뿐이다 — 실제 선반 위치는 픽 성공 시점에 plant_node가 그 순간 AMR pose 기준
# 상대 오프셋으로 다시 배치한다(헤딩과 무관하게 도킹 기하가 항상 일관되도록, vault 2026-09-08 참고).
PLACE_SHELF_POS = (R_BELT + 1.5, 0.0, 0.2337)   # [m] world (x, y, z) — 초기 스폰용 잠정 위치
PLACE_SHELF_BEHIND_DIST  = 2.5   # [m] 도킹 전진서치가 커버할 거리 — 락 위치에서 "원래 진행방향의 반대"로 이만큼
# amr_node의 place_clear 이동거리(PLACE_CLEAR_VY×STEPS)와 반드시 같아야 함 — 단일 소스로 두려면
# amr_node가 이 값에서 역산하도록 유지할 것.
PLACE_SHELF_LATERAL_DIST = 0.75  # [m] "벨트에서 멀어지는"(place_clear와 같은 축) 방향으로 이만큼

PLACE_POINT_Y = -0.13   # [m] 선반 중심 기준 놓는 지점(로컬 y, 음수=AMR 쪽). 중심에 놓으면 팔 도달거리를 넘어감

# 놓는 순간 물체 중심의 body_link 상대 높이 — 픽처럼 그리퍼 끝단을 물체 "중심"에 맞추기 위함
# (상판 윗면을 목표로 잡으면 물체가 OBJ_S/2만큼 파고든다)
PLACE_OBJ_Z_BODY = (PLACE_SHELF_POS[2] + SHELF_PLATE_T/2.0 + OBJ_S/2.0) - BODY_LINK_WORLD_Z

# 원거리용 큰 마커(ID5)는 상판 아래 다리 사이, 근접용 작은 마커(ID7)는 상판 위 — 각각의 위치에서
# 상판 모서리에 가리지 않고 화면 잘림 없이 보이도록 실측으로 확정한 배치(vault 2026-09-09 참고).
PLACE_NEAR_MARKER_ID   = 7
PLACE_NEAR_MARKER_SIZE = 0.04   # [m] 근접(0.13m)까지 붙어도 화면에 안 잘림
SHELF_FAR_MARKER_X  = 0.0       # 둘 다 가운데 정렬 — 높이로 분리되므로 겹치지 않는다
SHELF_NEAR_MARKER_X = 0.0
_near_board = PLACE_NEAR_MARKER_SIZE*4.0/3.0
_far_board  = SHELF_FRONT_MARKER_SIZE*4.0/3.0
_mk_gap = 0.015                                       # 두 마커 사이 여백(아르코 quiet zone)
SHELF_NEAR_MARKER_Z = SHELF_PLATE_T/2.0 - _near_board/2.0
SHELF_FAR_MARKER_Z  = (SHELF_NEAR_MARKER_Z - _near_board/2.0 - _mk_gap) - _far_board/2.0

PLACE_DOCK_KEEP_DIST = 0.55   # [m] 차체-선반 도킹거리 — KEEP_DIST에 선반 반깊이(SHELF_TOP_D/2)만큼 더 띄움

# =============================================================================
# 실행 상수 (씬 경로 — 실행/실험 파라미터는 plant_node.py 상단으로 옮겼다)
# =============================================================================
# 턴테이블 관련 상수/코드는 참고용으로 남겨둠(build_turntable() 등, plant_node.py에서 호출만 주석처리)
TT_PATH, BASE_PATH = "/World/Turntable", "/World/Turntable/base"
DISC_PATH, DJNT = "/World/Turntable/disc", "joint_disc"
OBJ_PATH, PRIM = "/World/target", "/World/Ranger"

USE_CONVEYOR = True   # False면 턴테이블(build_turntable)로 복귀
CONV_PATH = "/World/Conveyor"
CONV_BELT_PATH = CONV_PATH + "/belt"
# 벨트 치수(길이=이동축 Y, 폭=X, 두께=Z). 높이는 턴테이블 때 실측 확정한 DISC_H 재사용.
CONV_LEN, CONV_WID, CONV_H = 24.0, 0.5, DISC_H
CONV_BELT_SEGMENTS = 20  # 실물 벨트 mesh를 몇 조각으로 나눠 이어붙일지(세그먼트 길이 1.2m 유지)
CONV_Y_START = -6.0  # [m] 벨트 시작(근접)단 world Y 고정 — CONV_LEN을 늘려도 끝단만 늘어난다
OBJ_SPAWN_Y = -4.0  # [m] 물체 초기 위치의 along-belt(Y) 오프셋 — 벨트 시작점 쪽(AMR은 Y=0)
CONV_Z = 0.20  # [m] 벨트 윗면 높이

# 실물 컨베이어 에셋(다리 제거, 벨트 표면만 사용 — 상세 경위는 vault 로그)
CONV_USE_REAL_BELT_ASSET = True
CONV_BELT_ASSET_SUBPATH = "/Isaac/Props/Conveyors/ConveyorBelt_A04.usd"  # A04=직선(A01은 코너였음)
CONV_BELT_ASSET_SRC_PRIM = "/World/Belt"  # 위 usd 파일 내부의 절대 경로(pxr로 실측 확인)

CONV_VELOCITY = 0.300  # [m/s] 벨트 속도 — 플랜트/AMR 양쪽이 그대로 씀

OBJ_CENTER_BODY = (0.0, KEEP_DIST, CONV_Z + CONV_H/2 + OBJ_S/2.0)
CONV_GAP_FROM_AMR = 0.60  # [m] 벨트 몸체를 AMR 스폰 지점에서 얼마나 떨어뜨릴지(차체 충돌 방지)


_HALF = MARKER_SIZE / 2.0
OBJ_PTS = np.array([[-_HALF,_HALF,0.],[_HALF,_HALF,0.],
                    [_HALF,-_HALF,0.],[-_HALF,-_HALF,0.]], float)
# 선반 전면 마커는 픽 타겟보다 훨씬 크므로 solvePnP용 점도 따로 둔다(OBJ_PTS 재사용 시 거리 왜곡).
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
# §1. 수학 헬퍼 (원본 step22.py와 동일 — 값 절대 바꾸지 말 것)
# =============================================================================

def _yaw(q):
    w,x,y,z = [float(v) for v in q]
    return math.atan2(2.0*(w*z+x*y), 1.0-2.0*(y*y+z*z))

def _wrap(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def _quat_to_R(q):
    w,x,y,z = [float(v) for v in q]
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], float)

def _R_to_quat(R):
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0:
        s = math.sqrt(tr+1.0)*2
        w=0.25*s; x=(R[2,1]-R[1,2])/s; y=(R[0,2]-R[2,0])/s; z=(R[1,0]-R[0,1])/s
    else:
        i = int(np.argmax([R[0,0],R[1,1],R[2,2]]))
        if i == 0:
            s = math.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2
            w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
        elif i == 1:
            s = math.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2
            w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
        else:
            s = math.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2
            w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    return np.array([w,x,y,z])

def _R_z(deg):
    a = math.radians(deg); c,s = math.cos(a), math.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]], float)

def _lookat_quat(C, T, up=(0.,0.,1.)):
    C,T,up = np.array(C,float), np.array(T,float), np.array(up,float)
    zc = C-T; zc /= (np.linalg.norm(zc)+1e-12)
    xc = np.cross(up, zc)
    if np.linalg.norm(xc) < 1e-6: xc = np.cross(np.array([0.,1.,0.]), zc)
    xc /= (np.linalg.norm(xc)+1e-12)
    yc = np.cross(zc, xc)
    return _R_to_quat(np.array([xc,yc,zc]).T)

def _quat_mul(a, b):
    """쿼터니언 곱 (w,x,y,z). a 다음 b를 '몸체 기준'으로 적용."""
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2], float)

def _quat_axis_angle(axis, deg):
    """축-각 → 쿼터니언 (w,x,y,z)."""
    a = np.array(axis, float); a = a/np.linalg.norm(a)
    h = math.radians(deg)/2.0
    return np.array([math.cos(h), *(a*math.sin(h))], float)

def quat_from_two_vec(u, v):
    u = u/np.linalg.norm(u); v = v/np.linalg.norm(v)
    d = float(np.dot(u,v))
    if d < -0.999999:
        ax = np.cross([1,0,0], u)
        if np.linalg.norm(ax) < 1e-6: ax = np.cross([0,1,0], u)
        ax /= np.linalg.norm(ax); return np.array([0.0,*ax])
    ax = np.cross(u,v); q = np.array([1.0+d,*ax]); return q/np.linalg.norm(q)

# =============================================================================
# §2. 마커 기하 — 순수 함수 (stage 없이도 build_marker()의 faces[fid]와 동일한 값을
# 계산 — vision_node가 씬 접근 없이 이 값을 그대로 재구성할 수 있다)
# =============================================================================

def _plate_basis(n):
    up = np.array([0.,0.,1.])
    if abs(float(n[2])) > 0.95: up = np.array([0.,1.,0.])
    y = up - float(np.dot(up,n))*n
    y /= np.linalg.norm(y)
    x = np.cross(y,n)
    return np.array([x,y,n]).T

def compute_marker_faces():
    """build_marker()의 faces 딕셔너리와 동일한 값 — {fid: (R_cp, off)}.
    stage 접근 없이 상수만으로 재구성 (vision_node 전용 진입점)."""
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
# 정의한다(5개 노드가 각자 순서를 외워서 어긋나는 사고 방지).
# =============================================================================

def pack_chassis_pose(bx, by, phi, valid, marker_id, sim_step=0, bz=0.0):
    """vision → amr,arm. [bx,by,phi,valid,marker_id,sim_step,bz]
    bz: 물체 중심 높이(몸체좌표, 파지 성공 판정용). sim_step: plant 물리 스텝 번호
    (Ω̂ 회귀추정의 시간축 — 노드 자체 타이머를 쓰면 벽시계/물리시간 어긋남으로 값이 틀어진다)."""
    return [float(bx), float(by), float(phi), 1.0 if valid else 0.0,
            float(marker_id), float(sim_step), float(bz)]

def unpack_chassis_pose(data):
    bx, by, phi, valid, marker_id, sim_step, bz = data
    return (float(bx), float(by), float(phi), bool(valid), int(marker_id),
            int(sim_step), float(bz))

def pack_eih_marker(x, y, z, valid):
    """vision → arm. [x,y,z,valid]"""
    return [float(x), float(y), float(z), 1.0 if valid else 0.0]

def unpack_eih_marker(data):
    x, y, z, valid = data
    return float(x), float(y), float(z), bool(valid)

def pack_joint_hold_target(q6):
    """arm → plant (wait 위상 관절목표, 6개)."""
    return [float(v) for v in q6]

def unpack_joint_hold_target(data):
    return np.array(data, float)
