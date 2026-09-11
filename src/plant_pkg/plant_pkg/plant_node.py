import argparse, os, sys, time, math, csv, datetime
import numpy as np
import cv2

# 파이프 리다이렉트 시 라인버퍼 강제 (run_step22.py와 동일 이유)
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--headless", action="store_true")
ap.add_argument("--max-sec", type=float, default=36000.0)  # 2026-09-02: 900s(15분)는
# 관찰 중 자동종료가 계속 걸려서 10시간으로 늘림 — 그래도 완전 무한은 아님(안전장치 유지)
ap.add_argument("--keep-open", action="store_true")
args = ap.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import omni.kit.async_engine
from pxr import (Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema,
                 UsdLux, Gf, Sdf)
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.utils.stage import get_current_stage, add_reference_to_stage
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim, RigidPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import (
    LulaKinematicsSolver, ArticulationKinematicsSolver)
try:
    from isaacsim.sensors.camera import Camera
except Exception:
    from omni.isaac.sensor import Camera

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Float32, Float32MultiArray, Bool, String, Int32
from sensor_msgs.msg import Image, CameraInfo, JointState
from geometry_msgs.msg import Point, TransformStamped
import tf2_ros

import step22_common as C
from step22_common import (
    OMEGA_BELT, R_BELT, DISC_R, DISC_H, DISC_Z, DISC_M, DISC_KD, DISC_MU,
    OBJ_S, OBJ_M, OBJ_MU, PHI_REF, DRIVE_MU,
    WHEEL_KD, MAX_FORCE, MAX_JVEL, STEER_KS, STEER_KD, STEER_MF,
    STEER_JOINTS, DRIVE_JOINTS, KEEP_DIST, HOVER_STANDOFF,
    HOVER_PITCH_DEG, HOVER_APPROACH_DIR, EE_GRIP_OFFSET,
    ARM_BASE_LOCAL, ARM_BASE_YAW_DEG,
    EE_FRAME, ARM_KS, ARM_KD, GRIP_KS, GRIP_KD,
    GRIP_JOINT_L, GRIP_JOINT_R, GRIP_STROKE_MAX,
    TOOL_AXIS_LOCAL, GRIPPER_DOWN, GRASP_PITCH_DEG,
    SEARCH_Q, PIPER_JOINT_NAMES, BASE_LINK, ARM_MOUNT, CAM_RIG,
    RS_RES, RS_NEAR_CLIP, CAM_POS_LOCAL, CAM_TILT_DEG, CAM_FORWARD_ONLY,
    EIH_RES, EIH_FOCAL_MM, EIH_HAPER_MM, EIH_MOUNT_POS, EIH_MOUNT_QUAT_WXYZ,
    OBJ_CENTER_BODY, MARKER_SIZE, MARKER_FACES, FACE_NORMAL,
    TOP_MARKER_ID, TOP_MARKER_SIZE, BOARD_T, CELL_T, CELL_GAP,
    COL_WHITE, COL_BLACK, MK_ROOT, DOME_INTENSITY,
    PLACE_SHELF_PATH, PLACE_SIDE_MARKER_ID, PLACE_TOP_MARKER_ID, PLACE_SHELF_POS,
    SHELF_TOP_W, SHELF_TOP_D, SHELF_PLATE_T, SHELF_LEG_SIDE,
    MK_SHELF_TOP_OFFSET, MK_SHELF_FRONT_OFFSET,
    SHELF_FRONT_MARKER_SIZE, SHELF_FAR_MARKER_Z, SHELF_NEAR_MARKER_Z,
    PLACE_NEAR_MARKER_ID, PLACE_NEAR_MARKER_SIZE,
    SHELF_FAR_MARKER_X, SHELF_NEAR_MARKER_X, PLACE_POINT_Y,
    PLACE_SHELF_BEHIND_DIST, PLACE_SHELF_LATERAL_DIST, PLACE_PITCH_DEG,
    TT_PATH, BASE_PATH, DISC_PATH, DJNT, OBJ_PATH, PRIM, BY_FAIL,
    USE_CONVEYOR, CONV_PATH, CONV_BELT_PATH, CONV_LEN, CONV_WID, CONV_H, CONV_Y_START,
    CONV_Z, CONV_VELOCITY, CONV_GAP_FROM_AMR, CONV_BELT_SEGMENTS, OBJ_SPAWN_Y,
    CONV_USE_REAL_BELT_ASSET, CONV_BELT_ASSET_SUBPATH, CONV_BELT_ASSET_SRC_PRIM,
    _yaw, _quat_to_R, _R_to_quat, _R_z, _lookat_quat,
    _quat_mul, _quat_axis_angle, quat_from_two_vec, _plate_basis,
    unpack_joint_hold_target,
)

# ── 사용자 조정 파라미터 ─────────────────────────────────────────────────────

# True면 턴테이블을 처음부터 돌리지 않는다(정적 픽 테스트용).
STATIC_TEST = True

# 외란 스케줄 — (물리스텝, 새 턴테이블 각속도[rad/s])
# 이 스텝에 도달하면 속도를 바꾼다. 무외란 베이스라인은 빈 리스트 []로.
# STATIC_TEST=True면 이 스케줄도 무시된다(턴테이블이 애초에 안 돈다).
BELT_SCHEDULE = [
    (600,  0.30),   # +20%, 0.36
    (1000, 0.30),   # -33%, 0.24
]
OM_RECOVER_TOL  = 0.05   # 외란 회복 판정 허용오차(비율)
OM_RECOVER_HOLD = 60     # [스텝] 이만큼 연속 허용오차 안이면 "회복"으로 본다

# 컨베이어(선형 벨트) 속도 외란 스케줄 — BELT_SCHEDULE(턴테이블 전용)과 별개로
# 컨베이어 모드(tt=None)에서만, 스텝이 아니라 이벤트(A=시작~lock, B=lock~grip 진입 직전,
# C=grip~원복) 기준으로 동작한다.
CONV_DISTURB_ENABLED = True  # False면 외란 없이 CONV_VELOCITY 그대로 유지(온/오프 스위치)
CONV_DISTURB_PHASE_A = 0.400  # [m/s]
CONV_DISTURB_PHASE_B = 0.200  # [m/s] — C는 CONV_VELOCITY(원복 baseline) 그대로 사용. A/B 낙차가
                               # 너무 크면 제어루프가 감당 못 해 충돌한다(vault 참고).
CONV_RECOVER_TOL  = 0.15  # 외란 회복 판정 허용오차(비율) — v̂ 추정 노이즈가 있어 om보다 넉넉히
CONV_RECOVER_HOLD = 60    # [스텝]

# 그리퍼 목표자세의 툴축 둘레 롤 — 0/180이면 joint6이 IK로 도달 안 되는 각도(+86.5°)가
# 필요해지는데, -90°로 돌리면 같은 파지를 하면서 필요한 joint6이 -3.4°로 내려와 풀린다
# (+90/±180은 다른 이유로 못 씀 — probe_roll.py 검증, vault 2026-09-03 참고). 부수효과로
# 손목이 90°/270°로 서지 않아 차체캠의 타겟 추종도 덜 가린다.
GRASP_ROLL_DEG = -90.0

# pre/grasp/lift에서 joint6 고정(q6[5]=0)을 풀지 여부 (hover는 영향 없음). DOWN_QUAT은
# 롤까지 지정하는 6구속이라 joint6은 이 기하구조에서 항상 ≈+91°로 결정돼 있다 — 0으로
# 고정하면 손가락 개폐축이 접근 pitch만큼 기울어 물체/벨트를 파고든다(실측 확인).
# 풀면(True) 개폐축이 완전 수평이 된다.
PGL_FREE_JOINT6 = True

# HOVER 접근 — 매틱 Cartesian IK 대신 1회 IK + 관절보간(§HOVER 손목스핀 대응)
HOVER_JOINT_MOVE_SEC = 2.0   # [s] q0→q_hover 관절보간에 걸리는 시간

# hover(pitch=30°,roll=0°)→pre(pitch=60°,roll=180°) 진입 시 자세가 크게 바뀌어
# 손목이 한 틱만에 재배치되던 문제 대응 — 진입 순간만 관절보간으로 부드럽게 넘긴다.
PRE_ENTRY_BLEND_SEC = 1.0   # [s] hover 이탈 직후 이 시간 동안만 관절보간, 이후 평소처럼 매틱 IK 추종

# 실행 / 종료 조건
START_OFFSET_Y = 0.4     # [m] AMR 초기 위치가 물체보다 벨트 진행축(Y)으로 얼마나 뒤에서 출발하는지
SETTLE_STEPS = 90        # [스텝] 시작 직후 물리 안정화 구간(이 동안 wheel_cmd 무시)
MAX_STEPS    = float("inf")  # [스텝] 물리스텝 타임아웃 비활성화 — 런처의 --max-sec 벽시계 상한은 별도로 살아있음

# 센싱 / 로깅 주기
CONTACT_SENSE  = True    # pad↔물체 실접촉력 관측(끄면 gripper SMC가 힘을 못 받는다)
CAM_EVERY      = 6       # [스텝] 차체 카메라 발행 주기
EIH_EVERY      = 6       # [스텝] 손목 카메라 발행 주기
TRACE_EVERY    = 120     # [스텝] 콘솔 추적 출력 주기
POSE_LOG_EVERY = 2       # [스텝] pose_log.csv 기록 주기

# 디코이(마커 없는 가짜 타겟) 배치 — 진짜 타겟 주변 along-belt(Y) 오프셋. 진짜 타겟만
# 골라 집는지 검증하는 용도(vision_node가 마커 없는 물체는 자동으로 무시).
DECOY_OFFSETS_Y = [-1.05, -0.70, -0.35, 0.35, 0.70, 1.05]

# ── 조정 파라미터 끝 ─────────────────────────────────────────────────────────

_HOME = os.path.expanduser("~")
_PROJ = _HOME + "/projects/ranger_piper"
RANGER_USD_PATH = f"{_PROJ}/ranger_mini/usd/ranger_mini_v2/ranger_mini_v2.usda"
PIPER_USD_PATH  = f"{_PROJ}/_src/piper_isaac_sim/USD/piper_v2.usd"
URDF_PATH    = f"{_PROJ}/piper/urdf/piper_description.urdf"
YAML_PATH    = f"{_PROJ}/piper/urdf/piper_robot_description.yaml"
RS_SUBCAM    = "Camera_OmniVision_OV9782_Color"
OUT_DIR      = f"{_PROJ}/step22_nodes"
RESULT_FILE  = OUT_DIR + "/results.csv"
POSE_LOG_FILE = OUT_DIR + "/pose_log.csv"

_HALF = MARKER_SIZE / 2.0
OBJ_PTS = C.OBJ_PTS
TOP_PTS = C.TOP_PTS

# =============================================================================
# §1. 씬 빌드 
# =============================================================================

def _mk_material(stage, path, name, mu):
    PhysicsMaterial(prim_path=path, name=name, static_friction=mu,
                    dynamic_friction=mu, restitution=0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(
        stage.GetPrimAtPath(path)).CreateFrictionCombineModeAttr("min")

def _bind(stage, prim, mat_path):
    mat = UsdShade.Material(stage.GetPrimAtPath(mat_path))
    UsdShade.MaterialBindingAPI(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")

def _mk_visual_mat(stage, path, color, rough=0.85):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path+"/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat

def _bind_visual(prim, mat):
    UsdShade.MaterialBindingAPI(prim).Bind(
        mat, UsdShade.Tokens.strongerThanDescendants)

def _weaken_parent_binding(stage, path):
    api = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(path))
    rel = api.GetDirectBindingRel()
    if rel and rel.GetTargets():
        try:
            UsdShade.MaterialBindingAPI.SetMaterialBindingStrength(
                rel, UsdShade.Tokens.weakerThanDescendants)
        except Exception: pass

def _quad(stage, path, cx,cy,cz, sx,sy,sz, color, mat=None, collide=False):
    g = UsdGeom.Cube.Define(stage, path)
    g.CreateSizeAttr(1.0)
    g.CreateExtentAttr([(-0.5,-0.5,-0.5),(0.5,0.5,0.5)])
    g.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    x = UsdGeom.Xformable(g)
    x.AddTranslateOp().Set(Gf.Vec3d(cx,cy,cz))
    x.AddScaleOp().Set(Gf.Vec3f(sx,sy,sz))
    if mat is not None: _bind_visual(g.GetPrim(), mat)
    # 기본은 순수 시각용(마커 board/cell처럼 충돌체가 있으면 안 되는 장식판) — collide=True를
    # 넘긴 호출부만 정적 충돌체(고정 구조물용, RigidBody 없이 CollisionAPI만)를 얻는다.
    if collide:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    return g

def build_turntable(stage):
    _mk_material(stage, "/World/PM/disc", "disc_mat", DISC_MU)
    _mk_material(stage, "/World/PM/obj", "obj_mat", OBJ_MU)
    root = UsdGeom.Xform.Define(stage, TT_PATH)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    base = UsdGeom.Xform.Define(stage, BASE_PATH)
    UsdGeom.Xformable(base).AddTranslateOp().Set(Gf.Vec3d(0.,0.,DISC_Z))
    bprim = base.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(bprim)
    UsdPhysics.MassAPI.Apply(bprim).CreateMassAttr(1.0)
    bg = UsdGeom.Cube.Define(stage, BASE_PATH+"/geom")
    bg.CreateSizeAttr(1.0)
    bg.CreateExtentAttr([(-0.5,-0.5,-0.5),(0.5,0.5,0.5)])
    UsdGeom.Xformable(bg).AddScaleOp().Set(Gf.Vec3f(0.02,0.02,0.02))
    UsdPhysics.CollisionAPI.Apply(bg.GetPrim())
    fj = UsdPhysics.FixedJoint.Define(stage, TT_PATH+"/root_joint")
    fj.CreateBody1Rel().SetTargets([BASE_PATH])
    fj.CreateLocalPos0Attr(Gf.Vec3f(0.,0.,DISC_Z))
    fj.CreateLocalPos1Attr(Gf.Vec3f(0.,0.,0.))
    disc = UsdGeom.Cylinder.Define(stage, DISC_PATH)
    disc.CreateAxisAttr("Z"); disc.CreateRadiusAttr(DISC_R)
    disc.CreateHeightAttr(DISC_H)
    disc.CreateExtentAttr([(-DISC_R,-DISC_R,-DISC_H/2),(DISC_R,DISC_R,DISC_H/2)])
    disc.CreateDisplayColorAttr([Gf.Vec3f(0.30,0.30,0.32)])
    UsdGeom.Xformable(disc).AddTranslateOp().Set(Gf.Vec3d(0.,0.,DISC_Z))
    dprim = disc.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(dprim); UsdPhysics.CollisionAPI.Apply(dprim)
    UsdPhysics.MassAPI.Apply(dprim).CreateMassAttr(DISC_M)
    _bind(stage, dprim, "/World/PM/disc")
    j = UsdPhysics.RevoluteJoint.Define(stage, TT_PATH+"/"+DJNT)
    j.CreateBody0Rel().SetTargets([BASE_PATH])
    j.CreateBody1Rel().SetTargets([DISC_PATH])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.,0.,0.))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.,0.,0.))
    drv = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drv.CreateTypeAttr("force"); drv.CreateStiffnessAttr(0.0)
    drv.CreateDampingAttr(DISC_KD); drv.CreateMaxForceAttr(1.0e7)
    DynamicCuboid(prim_path=OBJ_PATH, name="target",
                  position=np.array([R_BELT, 0.0, DISC_Z+DISC_H/2+OBJ_S/2.0]),
                  scale=np.array([OBJ_S,OBJ_S,OBJ_S]),
                  mass=OBJ_M, color=np.array([0.5,0.5,0.55]))
    _bind(stage, stage.GetPrimAtPath(OBJ_PATH), "/World/PM/obj")

def _build_conveyor_belt_cube(stage, belt_x):
    """벨트의 물리(RigidBody/Collision/SurfaceVelocity) 바디 — 항상 이 투명 Cube가 담당한다.
    실물 에셋 사용 시 `_build_conveyor_belt_visual()`이 이 위에 겉모습만 입히고 이 Cube는
    invisible로 감춘다. CONV_USE_REAL_BELT_ASSET=False거나 로드 실패 시 이 Cube가 그대로 보인다."""
    belt = UsdGeom.Cube.Define(stage, CONV_BELT_PATH)
    belt.CreateSizeAttr(1.0)
    belt.CreateExtentAttr([(-0.5,-0.5,-0.5),(0.5,0.5,0.5)])
    UsdGeom.Xformable(belt).AddTranslateOp().Set(
        Gf.Vec3d(belt_x, CONV_Y_START + CONV_LEN/2.0, CONV_Z))
    UsdGeom.Xformable(belt).AddScaleOp().Set(Gf.Vec3f(CONV_WID, CONV_LEN, CONV_H))
    belt.CreateDisplayColorAttr([Gf.Vec3f(0.20,0.20,0.22)])
    return belt.GetPrim()

def _get_assets_root_path():
    try:
        from isaacsim.storage.native import get_assets_root_path
        return get_assets_root_path()
    except Exception:
        try:
            from omni.isaac.nucleus import get_assets_root_path
            return get_assets_root_path()
        except Exception:
            return None

def _build_conveyor_belt_visual(stage, belt_x):
    """실물 컨베이어 에셋의 벨트 표면 mesh를 "시각 전용" 오버레이로 CONV_BELT_PATH 위에
    겹쳐 얹는다. 물리는 여전히 `_build_conveyor_belt_cube()`의 투명 Cube가 담당한다 —
    실물 에셋의 "Belt" 서브프림은 물리 콜리전이 안 잡히는 걸 헤드리스로 검증 확인했고
    (진짜 충돌체는 다리/프레임 쪽), 다리까지 쓰면 목표 치수 대비 너무 커서 다리 없이
    벨트 mesh만 목표 박스(CONV_WID×CONV_LEN×CONV_H)에 맞춰 비균일 스케일한다.
    CONV_BELT_SEGMENTS개로 나눠 이어붙인다(하나로 늘리면 텍스처가 늘어져 보임).
    성공하면 True, 실패하면 False를 반환해 호출부가 Cube를 그대로 보이게 둔다."""
    root = _get_assets_root_path()
    if not root:
        print("  [plant] ★ get_assets_root_path() 실패 — 실물 벨트 시각 오버레이 생략(Cube 그대로 보임)")
        return False
    asset_url = root + CONV_BELT_ASSET_SUBPATH
    seg_len = CONV_LEN / CONV_BELT_SEGMENTS
    created_paths = []
    try:
        # "/World/Belt"만 참조하면 머티리얼 바인딩이 컴포지션 시 드롭된다 —
        # "/World/Looks"를 따로 참조해 살리고 세그먼트마다 명시적으로 재바인딩한다.
        looks_prim = stage.DefinePrim("/World/Looks", "Scope")
        looks_prim.GetReferences().AddReference(asset_url, "/World/Looks")
        mat_prim = stage.GetPrimAtPath("/World/Looks/M_ConveyorBelt_A01_Belt")
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        for i in range(CONV_BELT_SEGMENTS):
            visual_path = f"{CONV_PATH}/belt_visual_{i}"
            created_paths.append(visual_path)
            vprim = stage.DefinePrim(visual_path, "Xform")
            vprim.GetReferences().AddReference(asset_url, CONV_BELT_ASSET_SRC_PRIM)
            if vprim.IsInstanceable():
                vprim.SetInstanceable(False)
            # mesh 자식 이름이 asset마다 달라 하드코딩 대신 Belt 서브트리 전체를 탐색
            mesh_prims = [p for p in Usd.PrimRange(vprim) if p.IsA(UsdGeom.Mesh)]
            if mesh_prims and mat_prim.IsValid():
                for mp in mesh_prims:
                    UsdShade.MaterialBindingAPI(mp).Bind(UsdShade.Material(mat_prim))
            elif i == 0:
                print(f"  [plant] ★ 벨트 mesh 재바인딩 대상 못 찾음(mesh={len(mesh_prims)}개, mat={mat_prim.IsValid()}) — 회색으로 보일 수 있음")
            local_box = bbox_cache.ComputeUntransformedBound(vprim).ComputeAlignedBox()
            lo, hi = local_box.GetMin(), local_box.GetMax()
            if lo[0] > hi[0]:
                raise RuntimeError(f"세그먼트 {i} 빈 bbox — 에셋 지오메트리 미해석")
            center = (lo + hi) * 0.5
            native_size = hi - lo
            # 원본 mesh 치수가 1.0이 아니므로(실측 ~2m) 목표/원본 비율로 스케일을 구해야
            # 실제로 CONV_WID×seg_len×CONV_H 박스가 된다.
            scale = Gf.Vec3d(CONV_WID/native_size[0], seg_len/native_size[1], CONV_H/native_size[2])
            seg_y = CONV_Y_START + (i + 0.5) * seg_len
            xf = UsdGeom.Xformable(vprim)
            # 원본 Belt 서브프림이 이미 갖고 있는 배치용 xformOp을 지우고 우리 변환만 새로 얹는다
            # (안 지우면 이중 변환됨). 원본 authored scale이 double이라 precision을 맞춰야 함.
            xf.ClearXformOpOrder()
            prec = UsdGeom.XformOp.PrecisionDouble
            xf.AddTranslateOp(prec).Set(Gf.Vec3d(belt_x, seg_y, CONV_Z))
            xf.AddScaleOp(prec).Set(scale)
            xf.AddTranslateOp(prec, opSuffix="recenter").Set(Gf.Vec3d(-center[0], -center[1], -center[2]))
        return True
    except Exception as e:
        print(f"  [plant] ★ 실물 벨트 시각 오버레이 로드 실패({e!r}) — Cube 그대로 보임")
        for p in created_paths:
            if stage.GetPrimAtPath(p):
                stage.RemovePrim(p)
        return False

def build_conveyor(stage):
    """턴테이블(build_turntable) 대체 — 벨트 자체는 안 움직이고 PhysxSurfaceVelocityAPI
    (마찰 표면속도)로 위에 놓인 물체를 민다(Isaac Sim 공식 컨베이어 방식과 동일 원리).
    타겟 물체는 턴테이블 때와 동일한 world 좌표(R_BELT, 0, CONV_Z)에 둔다. 벨트 몸체는
    R_BELT가 아니라 CONV_GAP_FROM_AMR(AMR 기준 거리)로 따로 배치해 AMR 차체와의 충돌을
    피한다. CONV_USE_REAL_BELT_ASSET=True면 실물 컨베이어 에셋을 이 Cube 위에 시각
    전용으로 덧씌운다(물리는 항상 이 Cube가 담당, `_build_conveyor_belt_visual()` 참고)."""
    _mk_material(stage, "/World/PM/belt", "belt_mat", DISC_MU)
    _mk_material(stage, "/World/PM/obj", "obj_mat", OBJ_MU)

    UsdGeom.Xform.Define(stage, CONV_PATH)
    belt_x = (R_BELT + KEEP_DIST) - CONV_GAP_FROM_AMR   # AMR 스폰(R_START)에서 CONV_GAP_FROM_AMR만큼
    bprim = _build_conveyor_belt_cube(stage, belt_x)
    if CONV_USE_REAL_BELT_ASSET:
        if _build_conveyor_belt_visual(stage, belt_x):
            UsdGeom.Imageable(bprim).MakeInvisible()
    rb = UsdPhysics.RigidBodyAPI.Apply(bprim)
    # kinematic 안 켜면 벨트 자체가 중력으로 자유낙하해 물체가 뚫고 떨어진다(실측 확인) —
    # 표면속도로 물체만 밀고 벨트 자신은 고정돼야 한다.
    rb.CreateKinematicEnabledAttr().Set(True)
    UsdPhysics.CollisionAPI.Apply(bprim)
    UsdPhysics.MassAPI.Apply(bprim).CreateMassAttr(50.0)
    # CONV_VELOCITY=0(정적 테스트)일 때는 표면속도 API를 켜지 않는다 — 속도 0이어도
    # Enabled=True면 PhysX가 "컨베이어 접촉"으로 처리해 정지마찰이 깨지고 물체가
    # 서서히 미끄러진다(실측 확인, 자세한 경위는 vault 참고).
    if abs(CONV_VELOCITY) > 1e-9:
        # attribute만 만들어두고 Enabled=False로 시작 — 4개 노드 접속 확인 후
        # on_physics()의 settle→run 전환 시점에 True로 뒤집는다(아래).
        sv = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(bprim)
        sv.CreateSurfaceVelocityEnabledAttr().Set(False)
        # LocalSpace=True면 이 큐브의 비균일 스케일(Y=CONV_LEN)까지 곱혀 속도가 왜곡된다
        sv.CreateSurfaceVelocityLocalSpaceAttr().Set(False)
        sv.CreateSurfaceVelocityAttr().Set(Gf.Vec3f(0.0, CONV_VELOCITY, 0.0))
    else:
        print("  [plant] CONV_VELOCITY=0 — 표면속도 API 비활성(정지마찰 정상화)")
    _bind(stage, bprim, "/World/PM/belt")

    DynamicCuboid(prim_path=OBJ_PATH, name="target",
                  position=np.array([R_BELT, OBJ_SPAWN_Y, CONV_Z+CONV_H/2+OBJ_S/2.0]),
                  scale=np.array([OBJ_S,OBJ_S,OBJ_S]),
                  mass=OBJ_M, color=np.array([0.5,0.5,0.55]))
    _bind(stage, stage.GetPrimAtPath(OBJ_PATH), "/World/PM/obj")
    # 벨트 마찰에 밀려 살짝씩 회전하는 아티팩트 방지 — 회전 자유도를 전부 잠가 순수 평행이동만 허용
    PhysxSchema.PhysxRigidBodyAPI.Apply(
        stage.GetPrimAtPath(OBJ_PATH)).CreateLockedRotAxisAttr((1 << 0) | (1 << 1) | (1 << 2))

def _bits(mid, size_px=60):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    img = cv2.aruco.generateImageMarker(d, mid, size_px)
    b = np.zeros((6,6), dtype=bool)
    step = size_px // 6
    for r in range(6):
        for c in range(6):
            b[r,c] = img[r*step+step//2, c*step+step//2] > 127
    return b

def _local_scale(stage, path):
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
    s = np.array([1.,1.,1.])
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeScale:
            v = op.Get(); s = np.array([float(v[0]),float(v[1]),float(v[2])])
    return s

def build_marker(stage):
    """옆면 4개(ID 0~3, AMR용) + 윗면 1개(ID 4, 팔용)."""
    sx,sy,sz = _local_scale(stage, OBJ_PATH)
    h = OBJ_S/2.0
    _weaken_parent_binding(stage, OBJ_PATH)
    m_w = _mk_visual_mat(stage, "/World/Looks/mk_white", COL_WHITE)
    m_b = _mk_visual_mat(stage, "/World/Looks/mk_black", COL_BLACK)
    UsdGeom.Xform.Define(stage, MK_ROOT)

    cell = MARKER_SIZE/6.0
    board = MARKER_SIZE + 2.0*cell
    d_mk = h + BOARD_T + CELL_GAP + CELL_T/2.0
    faces = {}
    for fid in MARKER_FACES:
        n = np.array(FACE_NORMAL[fid], float)
        R_cp = _plate_basis(n)
        off = d_mk*n
        root = MK_ROOT + f"/face{fid}"
        xf = UsdGeom.Xformable(UsdGeom.Xform.Define(stage, root))
        xf.AddTranslateOp().Set(Gf.Vec3d(*(off/np.array([sx,sy,sz]))))
        qw,qx,qy,qz = _R_to_quat(R_cp)
        xf.AddOrientOp().Set(Gf.Quatf(qw, Gf.Vec3f(qx,qy,qz)))
        xf.AddScaleOp().Set(Gf.Vec3f(1./sx, 1./sy, 1./sz))
        bits = _bits(fid)
        _quad(stage, root+"/board", 0.,0., -(CELL_GAP+CELL_T/2.0+BOARD_T/2.0),
              board, board, BOARD_T, COL_WHITE, mat=m_w)
        for r in range(6):
            for c in range(6):
                if bits[r,c]: continue
                _quad(stage, f"{root}/cell_{r}_{c}",
                      (c-2.5)*cell, (2.5-r)*cell, 0.,
                      cell, cell, CELL_T, COL_BLACK, mat=m_b)
        faces[fid] = (R_cp, off)

    tcell = TOP_MARKER_SIZE/6.0
    tboard = TOP_MARKER_SIZE + 2.0*tcell
    d_top = h + BOARD_T + CELL_GAP + CELL_T/2.0
    n_top = np.array([0.,0.,1.])
    R_top = _plate_basis(n_top)
    off_top = d_top*n_top
    troot = MK_ROOT + "/face_top"
    txf = UsdGeom.Xformable(UsdGeom.Xform.Define(stage, troot))
    txf.AddTranslateOp().Set(Gf.Vec3d(*(off_top/np.array([sx,sy,sz]))))
    qw,qx,qy,qz = _R_to_quat(R_top)
    txf.AddOrientOp().Set(Gf.Quatf(qw, Gf.Vec3f(qx,qy,qz)))
    txf.AddScaleOp().Set(Gf.Vec3f(1./sx, 1./sy, 1./sz))
    tbits = _bits(TOP_MARKER_ID)
    _quad(stage, troot+"/board", 0.,0., -(CELL_GAP+CELL_T/2.0+BOARD_T/2.0),
          tboard, tboard, BOARD_T, COL_WHITE, mat=m_w)
    for r in range(6):
        for c in range(6):
            if tbits[r,c]: continue
            _quad(stage, f"{troot}/cell_{r}_{c}",
                  (c-2.5)*tcell, (2.5-r)*tcell, 0.,
                  tcell, tcell, CELL_T, COL_BLACK, mat=m_b)

    print(f"  마커: 옆면 ID{MARKER_FACES} {MARKER_SIZE*1000:.0f}mm + "
          f"윗면 ID{TOP_MARKER_ID} {TOP_MARKER_SIZE*1000:.0f}mm")
    return {"faces": faces, "top_R": R_top, "top_off": off_top}

def build_decoys(stage):
    """마커 없는 가짜 타겟들을 진짜 타겟 주변에 배치. vision_node가 옆면은
    self.faces(ID0~3)에 없는 마커를 건너뛰고 윗면은 TOP_MARKER_ID(4)만 찾으므로
    (vision_node.py:150,218) 마커가 없는 디코이는 별도 처리 없이 자동으로
    무시된다 — 진짜 타겟만 골라 집는지 검증하는 용도."""
    _mk_material(stage, "/World/PM/decoy", "decoy_mat", OBJ_MU)
    for i, dy in enumerate(DECOY_OFFSETS_Y):
        path = f"/World/decoy_{i}"
        DynamicCuboid(prim_path=path, name=f"decoy_{i}",
                      position=np.array([R_BELT, OBJ_SPAWN_Y+dy, CONV_Z+CONV_H/2+OBJ_S/2.0]),
                      scale=np.array([OBJ_S, OBJ_S, OBJ_S]),
                      mass=OBJ_M, color=np.array([0.5, 0.5, 0.55]))
        _bind(stage, stage.GetPrimAtPath(path), "/World/PM/decoy")
        PhysxSchema.PhysxRigidBodyAPI.Apply(
            stage.GetPrimAtPath(path)).CreateLockedRotAxisAttr((1 << 0) | (1 << 1) | (1 << 2))
    print(f"  ★ 디코이 {len(DECOY_OFFSETS_Y)}개 배치(마커 없음, Y오프셋 {DECOY_OFFSETS_Y})")

def build_place_shelf(stage):
    """벨트와 분리된 고정 place 선반 — 다리+상판이 있는 실제 선반 형태. 상판 앞면
    (차체캠용, ID5)/윗면(손목캠용, ID6) 마커가 붙는다. root translate의 z
    (PLACE_SHELF_POS[2])는 "상판 중심 높이"를 뜻하고, 다리는 거기서 바닥까지 내려간다."""
    m_w = _mk_visual_mat(stage, "/World/Looks/place_white", COL_WHITE)
    m_b = _mk_visual_mat(stage, "/World/Looks/place_black", COL_BLACK)
    root = UsdGeom.Xform.Define(stage, PLACE_SHELF_PATH)
    _shelf_xf = UsdGeom.Xformable(root)
    _shelf_xf.AddTranslateOp().Set(Gf.Vec3d(*PLACE_SHELF_POS))
    _shelf_xf.AddRotateZOp().Set(0.0)  # 픽 성공 시 plant_node가 이 두 op을 다시 Set()해 재배치

    # kinematic RigidBody를 걸어야 물체가 상판을 뚫고 바닥까지 떨어지지 않는다(벨트 Cube와
    # 같은 원리 — 정지 상태를 유지하되 충돌은 받아야 하는 물체). kinematic이라 중력엔
    # 반응 안 하고, Xform을 직접 Set()해 움직이는 것만 반영된다(락 시점 재배치와 호환).
    rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rb.CreateKinematicEnabledAttr().Set(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(50.0)

    # 상판
    _quad(stage, PLACE_SHELF_PATH+"/plate", 0., 0., 0.,
          SHELF_TOP_W, SHELF_TOP_D, SHELF_PLATE_T, (0.35, 0.35, 0.40), collide=True)
    # 다리 — 상판 바닥(로컬 z=-PLATE_T/2)에서 바닥(월드 z=0, 로컬 z=-root_z)까지
    root_z = PLACE_SHELF_POS[2]
    leg_top = -SHELF_PLATE_T/2.0
    leg_bot = -root_z
    leg_h = leg_top - leg_bot
    leg_cz = (leg_top + leg_bot)/2.0
    leg_dx = SHELF_TOP_W/2.0 - SHELF_LEG_SIDE/2.0
    leg_dy = SHELF_TOP_D/2.0 - SHELF_LEG_SIDE/2.0
    for i, (lx, ly) in enumerate([(leg_dx, leg_dy), (leg_dx, -leg_dy),
                                   (-leg_dx, leg_dy), (-leg_dx, -leg_dy)]):
        _quad(stage, f"{PLACE_SHELF_PATH}/leg_{i}", lx, ly, leg_cz,
              SHELF_LEG_SIDE, SHELF_LEG_SIDE, leg_h, (0.30, 0.30, 0.33), collide=True)

    def _face(fid, size, normal, off_dist, root_path, extra=(0., 0., 0.)):
        n = np.array(normal, float)
        R_cp = _plate_basis(n)
        off = off_dist*n + np.array(extra, float)
        xf = UsdGeom.Xformable(UsdGeom.Xform.Define(stage, root_path))
        xf.AddTranslateOp().Set(Gf.Vec3d(*off))
        qw, qx, qy, qz = _R_to_quat(R_cp)
        xf.AddOrientOp().Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
        cell = size/6.0
        board = size + 2.0*cell
        bits = _bits(fid)
        _quad(stage, root_path+"/board", 0., 0., -(CELL_GAP+CELL_T/2.0+BOARD_T/2.0),
              board, board, BOARD_T, COL_WHITE, mat=m_w)
        for r in range(6):
            for c in range(6):
                if bits[r, c]: continue
                _quad(stage, f"{root_path}/cell_{r}_{c}",
                      (c-2.5)*cell, (2.5-r)*cell, 0., cell, cell, CELL_T, COL_BLACK, mat=m_b)

    # 전면(-Y, AMR 접근 방향을 바라봄) — 차체캠용. 상판(+Z) — 손목캠용. 오프셋은 각 면의
    # 실제 반두께만큼(플레이트가 정육면체가 아니라 면마다 다름). 마커 크기(SHELF_FRONT_MARKER_SIZE)는
    # 도킹거리에서 ArUco 최소 인식 픽셀을 넘도록 픽 타겟용보다 훨씬 크게 잡혀 있다.
    _face(PLACE_SIDE_MARKER_ID, SHELF_FRONT_MARKER_SIZE, (0., -1., 0.), MK_SHELF_FRONT_OFFSET,
          PLACE_SHELF_PATH+"/face_front",
          extra=(SHELF_FAR_MARKER_X, 0., SHELF_FAR_MARKER_Z))
    # 근접용(ID7) — 큰 마커는 가까워지면 화면을 넘쳐 잘리므로, 차체캠 눈높이에
    # 작은 마커를 나란히 두고 근접 구간에서 이쪽으로 넘긴다.
    _face(PLACE_NEAR_MARKER_ID, PLACE_NEAR_MARKER_SIZE, (0., -1., 0.), MK_SHELF_FRONT_OFFSET,
          PLACE_SHELF_PATH+"/face_near",
          extra=(SHELF_NEAR_MARKER_X, 0., SHELF_NEAR_MARKER_Z))
    # 상판 마커(ID6)는 "여기에 놓는다"는 표시 — 팔 목표와 같은 PLACE_POINT_Y를 쓴다.
    _face(PLACE_TOP_MARKER_ID, TOP_MARKER_SIZE, (0., 0., 1.), MK_SHELF_TOP_OFFSET,
          PLACE_SHELF_PATH+"/face_top", extra=(0., PLACE_POINT_Y, 0.))

    print(f"  ★ place 선반 배치(초기): pos={PLACE_SHELF_POS}  전면ID{PLACE_SIDE_MARKER_ID}/상판ID{PLACE_TOP_MARKER_ID}"
          f"  (픽 성공 시 락 위치 기준으로 재배치됨)")

def build_lighting(stage):
    _quad(stage, "/World/dark_floor", 0.,0.,0.001, 40.,40.,0.002, (0.03,0.03,0.03))
    dome = UsdLux.DomeLight.Define(stage, "/World/dome")
    dome.CreateIntensityAttr(DOME_INTENSITY)
    p = dome.GetPrim()
    for attr in ("visibleInPrimaryRay", "primvars:visibleInPrimaryRay"):
        try:
            a = p.GetAttribute(attr)
            if not a: a = p.CreateAttribute(attr, Sdf.ValueTypeNames.Bool)
            a.Set(False)
        except Exception: pass

def _find_joint(stage, name):
    for p in stage.Traverse():
        if p.GetName() == name: return p
    return None

def setup_ranger_drives(stage):
    """Ranger Mini 2.0 4WIS 구동 셋업 — 조향 4(position)+구동 4(velocity).
    캐스터 없음(4바퀴 전부 구동륜). URDF를 --no-fix-base로 임포트해서
    root_joint 자체가 없음 — 별도 비활성화 불필요."""
    _mk_material(stage, "/World/PM/drive", "drive_mat", DRIVE_MU)
    dm = UsdShade.Material(stage.GetPrimAtPath("/World/PM/drive"))
    for jn in STEER_JOINTS:
        jp = _find_joint(stage, jn)
        drv = UsdPhysics.DriveAPI.Apply(jp, "angular")
        drv.CreateTypeAttr().Set("force")
        drv.CreateStiffnessAttr().Set(STEER_KS)
        drv.CreateDampingAttr().Set(STEER_KD)
        drv.CreateMaxForceAttr().Set(STEER_MF)
    for jn in DRIVE_JOINTS:
        jp = _find_joint(stage, jn)
        drv = UsdPhysics.DriveAPI.Apply(jp, "angular")
        drv.CreateTypeAttr().Set("force")
        drv.CreateStiffnessAttr().Set(0.0)
        drv.CreateDampingAttr().Set(WHEEL_KD)
        drv.CreateMaxForceAttr().Set(MAX_FORCE)
        PhysxSchema.PhysxJointAPI.Apply(jp).CreateMaxJointVelocityAttr().Set(MAX_JVEL)
    for p in stage.Traverse():
        path = str(p.GetPath())
        if not (path.startswith(PRIM) and p.HasAPI(UsdPhysics.CollisionAPI)):
            continue
        if "wheel_link" in path.lower():
            UsdShade.MaterialBindingAPI(p).Bind(
                dm, UsdShade.Tokens.weakerThanDescendants, "physics")

def mount_arm(stage):
    """PiPER를 Ranger Mini 상판에 마운트. 그리퍼(joint7/8) USD에 내장돼
    있어 별도 마운트/드라이브 셋업 불필요."""
    UsdGeom.Xform.Define(stage, ARM_MOUNT)
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(ARM_MOUNT))
    xf.AddTranslateOp().Set(Gf.Vec3d(*ARM_BASE_LOCAL.tolist()))
    xf.AddRotateZOp().Set(ARM_BASE_YAW_DEG)

    # piper_v2.usd는 defaultPrim이 없어 add_reference_to_stage()를 못 쓴다 — 로봇 본체가
    # 실제로 들어있는 "/piper_camera" 서브트리를 직접 지정해 참조한다.
    asset_path = ARM_MOUNT + "/piper"
    prim = stage.DefinePrim(asset_path, "Xform")
    prim.GetReferences().AddReference(assetPath=PIPER_USD_PATH, primPath="/piper_camera")

    for pr in Usd.PrimRange(stage.GetPrimAtPath(asset_path)):
        if pr.GetName() == "root_joint": pr.SetActive(False); break

    arm_base = None
    for pr in Usd.PrimRange(stage.GetPrimAtPath(asset_path)):
        if pr.HasAPI(UsdPhysics.ArticulationRootAPI):
            pr.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if pr.HasAPI(UsdPhysics.RigidBodyAPI) and arm_base is None:
            if "base" in pr.GetName().lower(): arm_base = pr
    if arm_base is None:
        print("  ★★ PiPER base 자동탐지 실패"); return None

    fj = UsdPhysics.FixedJoint.Define(stage, ARM_MOUNT+"/mount_joint")
    fj.CreateBody0Rel().SetTargets([BASE_LINK])
    fj.CreateBody1Rel().SetTargets([str(arm_base.GetPath())])
    fj.CreateLocalPos0Attr(Gf.Vec3f(*ARM_BASE_LOCAL.tolist()))
    fj.CreateLocalPos1Attr(Gf.Vec3f(0.,0.,0.))
    cy = math.cos(math.radians(ARM_BASE_YAW_DEG/2))
    sy = math.sin(math.radians(ARM_BASE_YAW_DEG/2))
    fj.CreateLocalRot0Attr(Gf.Quatf(cy, 0., 0., sy))
    return asset_path

def find_ee_link(stage, arm_path):
    cands = []
    for pr in Usd.PrimRange(stage.GetPrimAtPath(arm_path)):
        if pr.HasAPI(UsdPhysics.RigidBodyAPI):
            nm = pr.GetName().lower()
            dep = str(pr.GetPath()).count('/')
            pri = 2 if any(k in nm for k in ('flange','tool','j6','link6')) else 0
            cands.append((pri, dep, pr))
    if not cands: return None
    cands.sort(key=lambda x:(x[0],x[1]), reverse=True)
    return cands[0][2]

def _rs_asset_path():
    try:
        from isaacsim.storage.native import get_assets_root_path
        root = get_assets_root_path()
    except Exception:
        try:
            from omni.isaac.nucleus import get_assets_root_path
            root = get_assets_root_path()
        except Exception: return None
    if not root: return None
    return [root+"/Isaac/Sensors/RealSense/D455/rsd455.usd",
            root+"/Isaac/Sensors/intel/RealSense/rsd455.usd",
            root+"/Isaac/Sensors/Intel/RealSense/rsd455.usd"]

def _strip_physics(stage, root_path):
    for pr in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if pr.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(pr).CreateRigidBodyEnabledAttr(False)
        if pr.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(pr).CreateCollisionEnabledAttr(False)
        if pr.HasAPI(UsdPhysics.ArticulationRootAPI):
            pr.RemoveAPI(UsdPhysics.ArticulationRootAPI)

def build_chassis_camera(stage, aim_body):
    cands = _rs_asset_path()
    if not cands: return None
    UsdGeom.Xform.Define(stage, CAM_RIG)
    asset_path = CAM_RIG + "/asset"
    used = None
    for c in cands:
        try:
            add_reference_to_stage(usd_path=c, prim_path=asset_path)
            if stage.GetPrimAtPath(asset_path).GetChildren():
                used = c; break
        except Exception: continue
    if used is None: return None

    subcam = None
    for pr in Usd.PrimRange(stage.GetPrimAtPath(asset_path)):
        if pr.IsA(UsdGeom.Camera) and pr.GetName() == RS_SUBCAM:
            subcam = pr; break
    if subcam is None: return None
    _strip_physics(stage, asset_path)

    cache = UsdGeom.XformCache()
    M_cam = cache.GetLocalToWorldTransform(subcam)
    M_rig = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(CAM_RIG))
    T_cam_rig = M_cam * M_rig.GetInverse()
    # CAM_FORWARD_ONLY: aim_body를 사선으로 조준하는 대신 차량 정면(+Y, 뷰포트 기준 좌측)을 바라보게 함
    look_target = (np.array(CAM_POS_LOCAL) + np.array([0.0, 1.0, 0.0])) if CAM_FORWARD_ONLY else aim_body
    w,x,y,z = _lookat_quat(CAM_POS_LOCAL, look_target)
    if abs(CAM_TILT_DEG) > 1e-9:
        w,x,y,z = _quat_mul(np.array([w,x,y,z]), _quat_axis_angle([1.,0.,0.], CAM_TILT_DEG))
    D = Gf.Matrix4d()
    D.SetRotate(Gf.Quatd(w, Gf.Vec3d(x,y,z)))
    D.SetTranslateOnly(Gf.Vec3d(*CAM_POS_LOCAL))
    M_rig_local = T_cam_rig.GetInverse() * D
    UsdGeom.Xformable(stage.GetPrimAtPath(CAM_RIG)).AddTransformOp().Set(M_rig_local)

    ucam = UsdGeom.Camera(subcam)
    cr = ucam.GetClippingRangeAttr().Get()
    ucam.GetClippingRangeAttr().Set(
        Gf.Vec2f(RS_NEAR_CLIP, float(cr[1]) if cr else 100.0))
    print(f"  차체 카메라: {subcam.GetPath()}")
    return str(subcam.GetPath())

def build_eih_camera(stage, ee_prim):
    d455_root = str(ee_prim.GetPath()) + "/d455"
    cands = _rs_asset_path()
    if not cands: return None
    used = None
    for c in cands:
        try:
            add_reference_to_stage(usd_path=c, prim_path=d455_root)
            if stage.GetPrimAtPath(d455_root).GetChildren():
                used = c; break
        except Exception: continue
    if used is None: return None


    rsd_prim = stage.GetPrimAtPath(d455_root + "/RSD455")
    print(f"    [RSD455-CHECK] prim={rsd_prim.GetPath()} valid={rsd_prim.IsValid()}")
    if rsd_prim.IsValid():
        rsd_xf = UsdGeom.Xformable(rsd_prim)
        # 2026-09-02 뷰포트 실측: RGB(color) 렌즈가 그리퍼 중점에 오도록 위치+회전 재조정.
        rsd_tx, rsd_ty, rsd_tz = -0.02232, 0.00308, 0.00219
        rx_deg, ry_deg, rz_deg = 171.801, -7.575, -80.844

        tr_op, rot_op = None, None
        for op in rsd_xf.GetOrderedXformOps():
            t = op.GetOpType()
            print(f"    [RSD455-CHECK] 기존 xformOp: {op.GetOpName()} type={t}")
            if t == UsdGeom.XformOp.TypeTranslate and tr_op is None:
                tr_op = op
            elif t in (UsdGeom.XformOp.TypeRotateXYZ, UsdGeom.XformOp.TypeOrient) and rot_op is None:
                rot_op = op

        if tr_op is not None:
            tr_op.Set(Gf.Vec3d(rsd_tx, rsd_ty, rsd_tz))
        else:
            tr_op = rsd_xf.AddTranslateOp()
            tr_op.Set(Gf.Vec3d(rsd_tx, rsd_ty, rsd_tz))

        if rot_op is not None and rot_op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            rot_op.Set(Gf.Vec3f(rx_deg, ry_deg, rz_deg))
        elif rot_op is not None and rot_op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            # USD rotateXYZ와 동일 순서(X→Y→Z)로 GfRotation 합성 후 쿼터니언 추출.
            # 기존 op 정밀도(double/float)에 맞춰 타입을 맞춰야 SetValueImpl이 안 튕긴다.
            rot = (Gf.Rotation(Gf.Vec3d(0, 0, 1), rz_deg)
                   * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry_deg)
                   * Gf.Rotation(Gf.Vec3d(1, 0, 0), rx_deg))
            q = rot.GetQuat()
            im = q.GetImaginary()
            if rot_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                rot_op.Set(Gf.Quatf(q.GetReal(), Gf.Vec3f(im[0], im[1], im[2])))
            else:
                rot_op.Set(Gf.Quatd(q.GetReal(), Gf.Vec3d(im[0], im[1], im[2])))
        else:
            rot_op = rsd_xf.AddRotateXYZOp()
            rot_op.Set(Gf.Vec3f(rx_deg, ry_deg, rz_deg))

        # translate -> rotate 순서를 명시적으로 고정(M=T*R) — 기존 op 순서를 그대로 두면 뒤바뀔 수 있음
        rsd_xf.SetXformOpOrder([tr_op, rot_op])
        print(f"    [RSD455-CHECK] 보정 적용 완료")

    xf = UsdGeom.Xformable(stage.GetPrimAtPath(d455_root))
    xf.AddTranslateOp().Set(Gf.Vec3d(*EIH_MOUNT_POS))
    qw, qx, qy, qz = EIH_MOUNT_QUAT_WXYZ
    print(f"    [MOUNT-CHECK] 적용되는 EIH_MOUNT_QUAT_WXYZ=({qw},{qx},{qy},{qz}) from {C.__file__}")
    xf.AddOrientOp().Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)))
    _strip_physics(stage, d455_root)

    cp = None
    for pr in Usd.PrimRange(stage.GetPrimAtPath(d455_root)):
        if pr.IsA(UsdGeom.Camera):
            if pr.GetName() == RS_SUBCAM: cp = str(pr.GetPath()); break
            if cp is None: cp = str(pr.GetPath())
    if cp is None:
        cp = d455_root + "/sensor_cam"
        UsdGeom.Camera.Define(stage, cp)
    cam = UsdGeom.Camera(stage.GetPrimAtPath(cp))
    cam.GetFocalLengthAttr().Set(EIH_FOCAL_MM)
    cam.GetHorizontalApertureAttr().Set(EIH_HAPER_MM)
    cam.GetVerticalApertureAttr().Set(EIH_HAPER_MM*EIH_RES[1]/EIH_RES[0])
    # D455 기본 clippingRange가 GRASP 근접 거리보다 클 수 있어 근평면을 낮춰둔다
    cr = cam.GetClippingRangeAttr().Get()
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, float(cr[1]) if cr else 100.0))
    print(f"  끝단 카메라: {cp}")
    return cp

def _setup_contact_sensing(stage, arm_path):
    """PiPER 손가락 링크(link7/link8)를 직접 찾아 접촉센싱에 사용."""
    if not CONTACT_SENSE or arm_path is None:
        return [], []
    pads = []
    for pr in Usd.PrimRange(stage.GetPrimAtPath(arm_path)):
        nm = pr.GetName().lower()
        if pr.HasAPI(UsdPhysics.RigidBodyAPI) and nm in ("link7", "link8"):
            pads.append(str(pr.GetPath()))
    print(f"  ★ 접촉센싱: pad 후보 리지드바디 {len(pads)}개")
    for q in pads:
        print(f"      {q}")
    if not pads:
        print("  ★ 접촉센싱: pad 리지드바디 없음 — 비활성화")
        return [], []

    views, ok_paths = [], []
    for n, q in enumerate(pads):
        v = None
        for kwargs in (
            dict(prim_paths_expr=q, name=f"pad_cv_{n}",
                 contact_filter_prim_paths_expr=[OBJ_PATH],
                 max_contact_count=16, prepare_contact_sensors=True,
                 track_contact_forces=True),
            dict(prim_paths_expr=q, name=f"pad_cv_{n}",
                 contact_filter_prim_paths_expr=[OBJ_PATH],
                 max_contact_count=16),
        ):
            try:
                v = RigidPrim(**kwargs); break
            except TypeError:
                continue
            except Exception as e:
                print(f"     ✗ {q.split('/')[-1]}: {type(e).__name__}: {str(e)[:110]}")
                v = None; break
        if v is not None:
            views.append(v); ok_paths.append(q)
            print(f"     ✓ 뷰 생성 {q.split('/')[-1]}")
    print(f"  ★ 접촉센싱: 뷰 {len(views)}/{len(pads)}개 ↔ {OBJ_PATH}")
    return views, ok_paths

def _setup_body_contact_sensing(stage, arm_path):
    """진단 전용: 손가락(link7/8)이 아닌 팔 링크가 물체를 미는지 관측 — 그리퍼 SMC(F_con)엔 안 섞임."""
    if not CONTACT_SENSE or arm_path is None:
        return [], []
    names = ("link1", "link2", "link3", "link4", "link5", "link6", "gripper_base")
    views, paths = [], []
    for pr in Usd.PrimRange(stage.GetPrimAtPath(arm_path)):
        if not (pr.HasAPI(UsdPhysics.RigidBodyAPI) and pr.GetName().lower() in names):
            continue
        q = str(pr.GetPath())
        try:
            v = RigidPrim(prim_paths_expr=q, name=f"body_cv_{len(views)}",
                          contact_filter_prim_paths_expr=[OBJ_PATH],
                          max_contact_count=16, prepare_contact_sensors=True,
                          track_contact_forces=True)
        except Exception:
            continue
        views.append(v); paths.append(pr.GetName())
    print(f"  ★ 진단 접촉센싱(팔 본체): {len(views)}개 ↔ {OBJ_PATH}  {paths}")
    return views, paths


def _setup_belt_contact_sensing(stage, arm_path):
    """진단 전용: 손가락(link6/7/8)이 벨트를 박는지 물리 접촉으로 직접 관측."""
    if not CONTACT_SENSE or arm_path is None:
        return [], []
    views, names = [], []
    for pr in Usd.PrimRange(stage.GetPrimAtPath(arm_path)):
        if not (pr.HasAPI(UsdPhysics.RigidBodyAPI)
                and pr.GetName().lower() in ("link6", "link7", "link8")):
            continue
        try:
            v = RigidPrim(prim_paths_expr=str(pr.GetPath()), name=f"belt_cv_{len(views)}",
                          contact_filter_prim_paths_expr=[CONV_BELT_PATH],
                          max_contact_count=16, prepare_contact_sensors=True,
                          track_contact_forces=True)
        except Exception:
            continue
        views.append(v); names.append(pr.GetName())
    print(f"  ★ 진단 접촉센싱(벨트): {len(views)}개 ↔ {CONV_BELT_PATH}  {names}")
    return views, names


def _grip_contact_force(pad_view, dt=1.0/60.0):
    views = pad_view or []
    if not views:
        return None, None
    per_pad = []
    for v in views:
        try:
            M = v.get_contact_force_matrix(dt=dt)
            if M is None:
                per_pad.append(0.0); continue
            M = np.asarray(M, float).reshape(-1, 3)
            per_pad.append(float(np.linalg.norm(M, axis=-1).sum()))
        except Exception:
            return None, None
    arr = np.array(per_pad, float)
    return float(arr.sum()), arr

def _gfquat_wxyz(q):
    im = q.GetImaginary()
    return (q.GetReal(), im[0], im[1], im[2])

def _prim_transform(xc, child_prim, parent_prim):
    """child가 parent 좌표계에서 어떤 자세인지 (R,t). eih_detect_body()의
    변환식과 동일 — 차체캠(정적, 1회)과 끝단캠(동적, 매틱) 둘 다 이걸로 구한다."""
    M_c = xc.GetLocalToWorldTransform(child_prim)
    M_p = xc.GetLocalToWorldTransform(parent_prim)
    T = M_c * M_p.GetInverse()
    R = _quat_to_R(_gfquat_wxyz(T.ExtractRotationQuat()))
    t = np.array(T.ExtractTranslation())
    return R, t

def _prim_transform_live_parent(xc, child_prim, parent_pos, parent_quat_wxyz):
    """_prim_transform()과 동일하지만 parent pose를 물리 API의 라이브 값(pos, quat_wxyz)으로
    받는다 — PRIM(articulation root)의 USD xformOp은 spawn 시점에 고정돼 매틱 갱신이 안 된다."""
    M_c = xc.GetLocalToWorldTransform(child_prim)
    w, x, y, z = parent_quat_wxyz
    M_p = Gf.Matrix4d()
    M_p.SetRotate(Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z))))
    M_p.SetTranslateOnly(Gf.Vec3d(*[float(v) for v in parent_pos]))
    T = M_c * M_p.GetInverse()
    R = _quat_to_R(_gfquat_wxyz(T.ExtractRotationQuat()))
    t = np.array(T.ExtractTranslation())
    return R, t


# =============================================================================
# §2. ROS2 노드
# =============================================================================

_LATCH = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class PlantNode(Node):
    def __init__(self):
        super().__init__("plant_node")

        # publishers
        self.pub_chassis_img = self.create_publisher(Image, "/vision/chassis_image", 5)
        self.pub_eih_img = self.create_publisher(Image, "/vision/eih_image", 5)
        self.pub_chassis_info = self.create_publisher(CameraInfo, "/vision/chassis_camera_info", _LATCH)
        self.pub_eih_info = self.create_publisher(CameraInfo, "/vision/eih_camera_info", _LATCH)
        self.pub_joint_states = self.create_publisher(JointState, "/mir/joint_states", 10)
        self.pub_contact_force = self.create_publisher(Float32, "/mir/contact_force", 10)
        # IMU 상당(자이로 z) — 실물 Ranger Mini의 IMU에 해당. 휠 오도메트리는 제자리
        # 회전에서 네 바퀴가 지면을 비비며 미끄러져 실제 차체 회전보다 크게 읽힌다.
        self.pub_imu_yaw_rate = self.create_publisher(Float32, "/mir/imu_yaw_rate", 10)
        self.pub_contact_pads = self.create_publisher(Float32MultiArray, "/mir/contact_force_pads", 10)
        self.pub_ee_pose = self.create_publisher(Point, "/arm/ee_pose_body", 10)
        # 물리 틱 신호 — amr/arm/gripper는 자체 벽시계 타이머 대신 이걸로 구동돼야 한다
        # (헤드리스 물리는 실시간보다 빠르게 돌 수 있어 벽시계 기준 시간축은 어긋난다).
        self.pub_tick = self.create_publisher(Int32, "/plant/tick", 20)

        # subscribers (최신값만 캐시)
        # wheel_cmd: position=조향각 4개, velocity=구동속도 4개 (amr_node._publish_wheel 참고)
        self.steer_cmd = np.zeros(4)
        self.drive_cmd = np.zeros(4)
        self.create_subscription(JointState, "/mir/wheel_cmd", self._on_wheel_cmd, 10)

        self.arm_status = "wait"
        self.create_subscription(String, "/arm/status", self._on_arm_status, 10)
        self.joint_hold_target = np.array(SEARCH_Q, float)  # 토픽 수신 전 잠깐 쓰이는 기본값(arm_node의 wait와 동일)
        self.create_subscription(Float32MultiArray, "/arm/joint_hold_target", self._on_joint_hold, 10)
        self.cartesian_target = None
        self.create_subscription(Point, "/arm/cartesian_target", self._on_cart_target, 10)
        # arm_node가 차체캠으로 내린 판정 보고 — 제어엔 안 쓰고 채점에만 쓴다
        self.pick_event = None
        self.create_subscription(Bool, "/arm/pick_event", self._on_pick_event, 10)

        # gripper_node가 실제 grip 명령을 보내기 전까지 쓰이는 기본값 — 오픈 상태로 시작
        # (joint7=0이 닫힘, joint7=GRIP_STROKE_MAX가 열림 — gripper_node.py 참고)
        self.gripper_cmd_pos = ([GRIP_JOINT_L, GRIP_JOINT_R], [GRIP_STROKE_MAX, -GRIP_STROKE_MAX])
        self.create_subscription(JointState, "/mir/gripper_cmd", self._on_gripper_cmd, 10)

        self.om_hat = 0.0
        self.create_subscription(Float32MultiArray, "/amr/telemetry", self._on_amr_telemetry, 10)

        self.amr_locked = False
        self.create_subscription(Bool, "/amr/lock", self._on_amr_lock, 1)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        self.get_logger().info("plant_node 초기화 완료")

    def _on_wheel_cmd(self, msg: JointState):
        if len(msg.position) >= 4:
            self.steer_cmd = np.array(msg.position[:4], float)
        if len(msg.velocity) >= 4:
            self.drive_cmd = np.array(msg.velocity[:4], float)

    def _on_arm_status(self, msg: String):
        self.arm_status = msg.data.split(" ")[0].split("=")[-1] if "phase=" in msg.data else msg.data

    def _on_joint_hold(self, msg: Float32MultiArray):
        self.joint_hold_target = unpack_joint_hold_target(msg.data)

    def _on_cart_target(self, msg: Point):
        self.cartesian_target = np.array([msg.x, msg.y, msg.z], float)

    def _on_pick_event(self, msg: Bool):
        self.pick_event = bool(msg.data)

    def _on_gripper_cmd(self, msg: JointState):
        self.gripper_cmd_pos = (list(msg.name), np.array(msg.position, float))

    def _on_amr_telemetry(self, msg: Float32MultiArray):
        if len(msg.data) >= 1:
            self.om_hat = float(msg.data[0])

    def _on_amr_lock(self, msg: Bool):
        self.amr_locked = bool(msg.data)


def _img_msg(node, arr, frame_id, stamp):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = arr.shape[0], arr.shape[1]
    msg.encoding = "rgba8"
    msg.step = msg.width * 4
    msg.data = np.ascontiguousarray(arr[:, :, :4]).tobytes()
    return msg


def _caminfo_msg(K, res, frame_id):
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width, msg.height = int(res[0]), int(res[1])
    msg.k = [float(v) for v in np.asarray(K, float).ravel()]
    return msg


def _tf_msg(node, R, t, parent, child, stamp):
    m = TransformStamped()
    m.header.stamp = stamp
    m.header.frame_id = parent
    m.child_frame_id = child
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = \
        float(t[0]), float(t[1]), float(t[2])
    qw, qx, qy, qz = _R_to_quat(R)
    m.transform.rotation.w, m.transform.rotation.x = float(qw), float(qx)
    m.transform.rotation.y, m.transform.rotation.z = float(qy), float(qz)
    return m


# =============================================================================
# §3. main
# =============================================================================

async def main():
    world = World(stage_units_in_meters=1.0)
    await world.initialize_simulation_context_async()
    world.scene.add_default_ground_plane()
    stage = get_current_stage()

    print("="*68)
    build_conveyor(stage) if USE_CONVEYOR else build_turntable(stage)
    mk = build_marker(stage)
    if USE_CONVEYOR:
        build_decoys(stage)
        build_place_shelf(stage)
    build_lighting(stage)
    add_reference_to_stage(usd_path=RANGER_USD_PATH, prim_path=PRIM)
    # setup_ranger_drives(stage)는 여기서 호출하지 않는다 — Ranger USD는 참조 직후엔
    # 아직 관절이 합성 안 된 상태라, 못 찾은 조향 조인트를 조용히 건너뛰어 바퀴가 월드에
    # 고정된 채 구동만 힘을 받는 발산이 났다(실측 확인). amr.initialize() 이후에 호출한다.

    arm_path = mount_arm(stage)
    if arm_path is None: return

    # R_START(across-gap)는 KEEP_DIST/IK 캘리브레이션이 걸려있어 안 건드리고 along-belt(Y)만 계산.
    # STATIC_TEST=False면 START_OFFSET_Y만큼 뒤에서 출발해 bootstrap→catch-up을 탄다.
    R_START = R_BELT + KEEP_DIST
    drift = 0.0 if STATIC_TEST else CONV_VELOCITY*(SETTLE_STEPS/60.0)
    y_a = drift if STATIC_TEST else drift - START_OFFSET_Y
    h = 90.0   # 궤도 위상 개념 없음 — 벨트 진행축(Y)에 항상 body+X를 맞춰 고정 시작
    xapi = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(PRIM))
    xapi.SetTranslate(Gf.Vec3d(R_START, y_a, 0.0))
    xapi.SetRotate(Gf.Vec3f(0.0, 0.0, h))

    _cp, _sp = math.cos(PHI_REF), math.sin(PHI_REF)
    _o = mk["faces"][MARKER_FACES[0]][1]
    aim_body = np.array(OBJ_CENTER_BODY,float) + np.array(
        [_cp*_o[0]-_sp*_o[1], _sp*_o[0]+_cp*_o[1], _o[2]])
    chassis_cam_path = build_chassis_camera(stage, aim_body)

    ee_prim = find_ee_link(stage, arm_path)
    eih_path = build_eih_camera(stage, ee_prim)

    # 컨베이어는 관절 없는 RigidBody라 SingleArticulation 대상이 아님 — tt=None.
    tt = None if USE_CONVEYOR else SingleArticulation(prim_path=TT_PATH, name="turntable")
    amr = SingleArticulation(prim_path=PRIM, name="ranger")
    pad_view, pad_paths = _setup_contact_sensing(stage, arm_path)
    body_view, body_names = _setup_body_contact_sensing(stage, arm_path)
    belt_view, belt_names = _setup_belt_contact_sensing(stage, arm_path)
    if tt is not None: world.scene.add(tt)
    world.scene.add(amr)
    await world.reset_async()
    if tt is not None: tt.initialize()
    amr.initialize()
    setup_ranger_drives(stage)
    # 팔/그리퍼용 별도 드라이브 셋업 없음 — PiPER 공식 USD가 전 관절의 drive
    # stiffness/damping을 이미 갖고 있다(실측 확인). 4WIS만 setup_ranger_drives()에서 처리.

    if tt is not None:
        tt.get_articulation_controller().set_gains(
            kps=np.array([0.0]), kds=np.array([DISC_KD]))
        tt.get_articulation_controller().apply_action(
            ArticulationAction(joint_velocities=np.array([0.0 if STATIC_TEST else OMEGA_BELT])))

    dof_names = list(amr.dof_names)
    steer_idx = np.array([dof_names.index(n) for n in STEER_JOINTS], dtype=int)
    drive_idx = np.array([dof_names.index(n) for n in DRIVE_JOINTS], dtype=int)
    arm_idx = np.array([dof_names.index(n) for n in PIPER_JOINT_NAMES], dtype=int)
    grip_idx = np.array([dof_names.index(GRIP_JOINT_L), dof_names.index(GRIP_JOINT_R)],
                        dtype=int)
    print(f"\n  DOF {amr.num_dof}")
    print(f"    조향 {steer_idx.tolist()}  구동 {drive_idx.tolist()}")
    print(f"    팔   {arm_idx.tolist()}")
    print(f"    그립 {grip_idx.tolist()}")

    kps = np.zeros(amr.num_dof); kds = np.zeros(amr.num_dof)
    for i in steer_idx: kps[i], kds[i] = STEER_KS, STEER_KD
    for i in drive_idx: kds[i] = WHEEL_KD
    for i in arm_idx:  kps[i], kds[i] = ARM_KS, ARM_KD
    for i in grip_idx: kps[i], kds[i] = GRIP_KS, GRIP_KD
    amr.get_articulation_controller().set_gains(kps=kps, kds=kds)

    os.makedirs(OUT_DIR, exist_ok=True)
    cam = eih_cam = None
    try:
        cam = Camera(prim_path=chassis_cam_path, resolution=RS_RES)
        cam.initialize()
    except Exception as e:
        print("  차체 카메라 실패:", str(e)[:60])
    try:
        eih_cam = Camera(prim_path=eih_path, resolution=EIH_RES, name="eih")
        eih_cam.initialize()
    except Exception as e:
        print("  끝단 카메라 실패:", str(e)[:60])

    def _K_of(path, res, fallback_f, fallback_ha):
        u = UsdGeom.Camera(stage.GetPrimAtPath(path))
        f = float(u.GetFocalLengthAttr().Get() or fallback_f)
        ha = float(u.GetHorizontalApertureAttr().Get() or fallback_ha)
        va = float(u.GetVerticalApertureAttr().Get() or ha*res[1]/res[0])
        return np.array([[f/ha*res[0], 0., res[0]/2.],
                         [0., f/va*res[1], res[1]/2.], [0.,0.,1.]], float)

    K = _K_of(chassis_cam_path, RS_RES, 1.93, 3.896)
    if cam is not None:
        try:
            Kw = np.array(cam.get_intrinsics_matrix(), float)
            if Kw.shape == (3,3) and Kw[0,0] > 1.0: K = Kw
        except Exception: pass
    eih_K = _K_of(eih_path, EIH_RES, EIH_FOCAL_MM, EIH_HAPER_MM)
    if eih_cam is not None:
        try:
            Kw = np.array(eih_cam.get_intrinsics_matrix(), float)
            if Kw.shape == (3,3) and Kw[0,0] > 1.0: eih_K = Kw
        except Exception: pass

    lula = LulaKinematicsSolver(robot_description_path=YAML_PATH, urdf_path=URDF_PATH)
    art_ik = ArticulationKinematicsSolver(amr, lula, EE_FRAME)
    # Lula는 articulation root(PRIM) 기준인데 ARM_BASE_LOCAL은 BASE_LINK 기준이라
    # BASE_LINK가 root 대비 떠있는 만큼 보정해서 전달한다.
    _xc_bl = UsdGeom.XformCache()
    _, _t_bl = _prim_transform(_xc_bl, stage.GetPrimAtPath(BASE_LINK),
                                stage.GetPrimAtPath(PRIM))
    _arm_base_lula = ARM_BASE_LOCAL + _t_bl
    print(f"  [진단] base_link가 articulation root 대비 {_t_bl.round(4)} 오프셋 "
          f"— Lula base를 {ARM_BASE_LOCAL.round(3)} → {_arm_base_lula.round(3)}로 보정")
    lula.set_robot_base_pose(_arm_base_lula, _R_to_quat(_R_z(ARM_BASE_YAW_DEG)))
    DOWN_QUAT = quat_from_two_vec(TOOL_AXIS_LOCAL, GRIPPER_DOWN)
    if abs(GRASP_PITCH_DEG) > 1e-6:
        DOWN_QUAT = _quat_mul(_quat_axis_angle(np.array([1.0, 0.0, 0.0]), GRASP_PITCH_DEG), DOWN_QUAT)
        DOWN_QUAT = DOWN_QUAT / np.linalg.norm(DOWN_QUAT)
    if abs(GRASP_ROLL_DEG) > 1e-6:
        DOWN_QUAT = _quat_mul(DOWN_QUAT,
                              _quat_axis_angle(TOOL_AXIS_LOCAL, GRASP_ROLL_DEG))
        DOWN_QUAT = DOWN_QUAT / np.linalg.norm(DOWN_QUAT)
        print(f"  ★ GRASP_ROLL {GRASP_ROLL_DEG:+.0f}° 적용 (툴축 둘레)")
    print(f"  Lula base(몸체 고정) {ARM_BASE_LOCAL.round(3)} yaw{ARM_BASE_YAW_DEG:.0f}°")
    print(f"  DOWN_QUAT {DOWN_QUAT.round(3)}  EE={EE_FRAME}")

    HOVER_DOWN_QUAT = quat_from_two_vec(TOOL_AXIS_LOCAL, GRIPPER_DOWN)
    HOVER_DOWN_QUAT = _quat_mul(_quat_axis_angle(np.array([1.0, 0.0, 0.0]), HOVER_PITCH_DEG), HOVER_DOWN_QUAT)
    HOVER_DOWN_QUAT = HOVER_DOWN_QUAT / np.linalg.norm(HOVER_DOWN_QUAT)
    # hover도 DOWN_QUAT과 같은 롤을 적용해야 한다 — 안 그러면 hover→pre 전환 시 손목이
    # 90° 스윕하며 물체 바로 위에서 손가락 판이 물체를 쓸어버린다(실측 확인).
    if abs(GRASP_ROLL_DEG) > 1e-6:
        HOVER_DOWN_QUAT = _quat_mul(HOVER_DOWN_QUAT,
                                    _quat_axis_angle(TOOL_AXIS_LOCAL, GRASP_ROLL_DEG))
        HOVER_DOWN_QUAT = HOVER_DOWN_QUAT / np.linalg.norm(HOVER_DOWN_QUAT)
    # EE_FRAME(link6)은 그리퍼 끝단보다 물체에서 EE_GRIP_OFFSET만큼 더 멀어야
    # 실제 그리퍼 끝단이 HOVER_STANDOFF만큼 떨어진다.
    HOVER_TARGET = np.array(OBJ_CENTER_BODY) - HOVER_APPROACH_DIR * (HOVER_STANDOFF + EE_GRIP_OFFSET)
    print(f"  HOVER_DOWN_QUAT {HOVER_DOWN_QUAT.round(3)} (pitch={HOVER_PITCH_DEG:.0f}°)  HOVER_TARGET {HOVER_TARGET.round(3)}")

    # place(선반에 내려놓기) 전용 orientation — 픽의 pitch=0을 재사용하면 IK가 100% 실패한다
    # (기하 자체의 도달불가, 시드 문제 아님 — vault 참고). place_hover~place_retreat 전 구간이 공유.
    PLACE_DOWN_QUAT = quat_from_two_vec(TOOL_AXIS_LOCAL, GRIPPER_DOWN)
    PLACE_DOWN_QUAT = _quat_mul(_quat_axis_angle(np.array([1.0, 0.0, 0.0]), PLACE_PITCH_DEG), PLACE_DOWN_QUAT)
    PLACE_DOWN_QUAT = PLACE_DOWN_QUAT / np.linalg.norm(PLACE_DOWN_QUAT)
    if abs(GRASP_ROLL_DEG) > 1e-6:
        PLACE_DOWN_QUAT = _quat_mul(PLACE_DOWN_QUAT,
                                     _quat_axis_angle(TOOL_AXIS_LOCAL, GRASP_ROLL_DEG))
        PLACE_DOWN_QUAT = PLACE_DOWN_QUAT / np.linalg.norm(PLACE_DOWN_QUAT)
    print(f"  PLACE_DOWN_QUAT {PLACE_DOWN_QUAT.round(3)} (pitch={PLACE_PITCH_DEG:.0f}°)")

    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE) as _f:
            _rows = [r.strip().split(",") for r in _f if not r.startswith("timestamp")]
        _n = len(_rows); _ok = sum(1 for r in _rows if r and r[2]=="True")
        print(f"\n  [누적 성공률] {_ok}/{_n}회 ({100*_ok/max(_n,1):.0f}%)  ← {RESULT_FILE}")
    print("="*68)

    # ── ROS2 ────────────────────────────────────────────────────────────────
    rclpy.init()
    ros = PlantNode()

    # Script Editor에서 World.instance()._pick_probe로 접근(관절은 ros.joint_hold_target 갱신).
    world._pick_probe = {"ros": ros, "amr": amr, "arm_idx": arm_idx,
                          "art_ik": art_ik, "lula": lula, "tt": tt,
                          "DOWN_QUAT": DOWN_QUAT}

    # 차체캠 정적 tf — 마운트 고정이라 1회만

    xc = UsdGeom.XformCache()
    stamp0 = ros.get_clock().now().to_msg()
    R_bc, t_bc = _prim_transform(xc, stage.GetPrimAtPath(chassis_cam_path),
                                  stage.GetPrimAtPath(PRIM))
    ros.tf_static_broadcaster.sendTransform(
        _tf_msg(ros, R_bc, t_bc, "body_link", "chassis_cam", stamp0))
    ros.pub_chassis_info.publish(_caminfo_msg(K, RS_RES, "chassis_cam"))
    ros.pub_eih_info.publish(_caminfo_msg(eih_K, EIH_RES, "eih_cam"))
    ros._caminfo_payload = (_caminfo_msg(K, RS_RES, "chassis_cam"),
                            _caminfo_msg(eih_K, EIH_RES, "eih_cam"),
                            R_bc, t_bc)
    print(f"  [tf] 차체캠 body_link←chassis_cam 정적 브로드캐스트 t={t_bc.round(4)}")

    # ── 상태 (원본 S 딕셔너리에 대응하는 것들만 유지) ────────────────────────
    st = {
        "k": 0, "phase": "settle",
        "belt_om": OMEGA_BELT, "belt_i": 0, "belt_events": [],
        "dist_active": False, "dist_k0": 0, "dist_run": 0,
        "dist_max_bx": 0.0, "dist_max_by": 0.0, "dist_lost_pll": False,
        "conv_v": CONV_DISTURB_PHASE_A, "conv_phase": "A", "conv_arm_prev": None,
        "conv_dist_active": False, "conv_dist_k0": 0, "conv_dist_run": 0,
        "conv_dist_max_bx": 0.0, "conv_dist_max_by": 0.0,
        "marks": [], "plog": None, "plog_file": None, "plog_rows": [],
        "gripper_default_pos": np.zeros(len(grip_idx)),
        "finished": False, "attempt_n": 0,  #재시도 지원
        "hover_dbg_n": 0, "hover_dbg_prev_status": None,  # ★임시 디버그(특이점 조사)
        "hover_q_target": None, "hover_q0": None, "hover_t0": None, "hover_ik_fail_n": 0,
        "pgl_ik_fail_n": 0, "pgl_prev_status": None,  # pre/grasp/lift IK 실패 진단
        "pgl_seed": None,      # pre/grasp/lift IK 고정 warm start (아래 참고)
        "pgl_roll_dbg": 0,     # joint6 고정이 만드는 손가락 개폐축 오차 진단
        "grip_pause_done": False,  #grip 진입 순간 스크린샷용 임시 디버그
        "hover_pause_done": False,  # hover 도달(hover→detect 전이) 순간 스크린샷용
        "prev_arm_status": None,
        "reorient_q0": None, "reorient_qtarget": None, "reorient_t0": None,
    }

    hdr = ["step","t_sec","amr_x","amr_y","amr_yaw","wL","wR","v_amr","w_amr",
           "disc_w","obj_x","obj_y","obj_z","g_bx","g_by","om_hat","phase","arm_status"]
    try:
        pf = open(POSE_LOG_FILE, "w", newline="", encoding="utf-8")
        pw = csv.writer(pf); pw.writerow(hdr)
        st["plog_file"] = pf; st["plog"] = pw
        print(f"  시계열 로그: {POSE_LOG_FILE}")
    except Exception as e:
        print(f"  ★ 시계열 로그 열기 실패: {str(e)[:60]}")

    obj_view = SingleXFormPrim(prim_path=OBJ_PATH, name="obj_view")
    world._pick_probe["obj_view"] = obj_view
    eih_prim = stage.GetPrimAtPath(eih_path)

    def _finish_run(reason):
        try:
            if st["plog_file"]:
                st["plog_file"].flush(); st["plog_file"].close()
        except Exception:
            pass
        st["plog"] = None; st["plog_file"] = None
        print(f"\n  [plant] 실행 종료: {reason}\n")
        world.pause()

    def on_physics(dt):
        if amr is None or not amr.handles_initialized: return
        if tt is not None and not tt.handles_initialized: return  # 컨베이어(tt=None)는 통과
        if not world.is_playing(): return

        rclpy.spin_once(ros, timeout_sec=0)

        if st["phase"] == "settle":
            st["k"] += 1
            amr.apply_action(ArticulationAction(
                joint_velocities=np.zeros(4), joint_indices=drive_idx))
            amr.apply_action(ArticulationAction(
                joint_positions=SEARCH_Q,
                joint_indices=arm_idx))
            vision_ready = ros.pub_chassis_img.get_subscription_count() > 0
            # "/plant/tick"은 amr/arm/gripper 3개 노드 전부가 구독 — 구독자 수로
            # 셋 다 접속했는지 확인 후에만 벨트를 켠다(안 그러면 다른 노드 접속
            # 전에 벨트가 먼저 물체를 밀어버림).
            control_ready = ros.pub_tick.get_subscription_count() >= 3
            all_ready = vision_ready and control_ready
            _fallback_steps = 36000   # 폴백 타임아웃(10분) — 노드 하나 깜빡해도 무한대기 방지
            if st["k"] >= SETTLE_STEPS and (all_ready or st["k"] > _fallback_steps):
                st["phase"] = "run"
                if not vision_ready:
                    print("\n  ★ [plant] vision_node 미접속 상태로 진행합니다")
                if not control_ready:
                    print(f"\n  ★ [plant] amr/arm/gripper 중 일부 미접속 상태로 진행합니다"
                          f"(/plant/tick 구독자 {ros.pub_tick.get_subscription_count()}/3)")
                if abs(CONV_VELOCITY) > 1e-9:
                    _belt_prim = stage.GetPrimAtPath(CONV_BELT_PATH)
                    _v0 = CONV_DISTURB_PHASE_A if (CONV_DISTURB_ENABLED and tt is None) else CONV_VELOCITY
                    PhysxSchema.PhysxSurfaceVelocityAPI(_belt_prim).GetSurfaceVelocityAttr().Set(
                        Gf.Vec3f(0.0, _v0, 0.0))
                    PhysxSchema.PhysxSurfaceVelocityAPI(_belt_prim).GetSurfaceVelocityEnabledAttr().Set(True)
                    st["conv_v"] = _v0
                    print(f"  ★ [plant] 컨베이어 벨트 구동 시작 (v={_v0:.3f}m/s, "
                          f"vision={vision_ready} tick구독자={ros.pub_tick.get_subscription_count()}/3)")
                _pa_settled, _qa_settled = amr.get_world_pose()
                # settle 시점 AMR 자세 보관 — 이후 팔이 뻗을 때 AMR이 가라앉거나 기우는지 비교용
                # (Lula base 회전은 여기서 _R_z(90°)로 1회 고정, 실제 roll/pitch는 반영 안 함)
                st["pa_settled"] = np.array(_pa_settled, float)
                st["qa_settled"] = np.array(_qa_settled, float)
                _ab_prim_path = None
                for _pr in Usd.PrimRange(stage.GetPrimAtPath(arm_path)):
                    if _pr.HasAPI(UsdPhysics.RigidBodyAPI) and "base" in _pr.GetName().lower():
                        _ab_prim_path = str(_pr.GetPath()); break
                _base_z = None
                if _ab_prim_path is not None:
                    try:
                        _abp, _ = SingleXFormPrim(prim_path=_ab_prim_path,
                                                  name="lula_base_probe").get_world_pose()
                        _base_z = float(_abp[2])
                    except Exception as _e:
                        print(f"  ★ arm_base 실측 실패({type(_e).__name__}) — 구식 계산으로 폴백")
                if _base_z is None:
                    _base_z = float(ARM_BASE_LOCAL[2]) + float(_pa_settled[2])
                _arm_base_lula_fixed = np.array([ARM_BASE_LOCAL[0], ARM_BASE_LOCAL[1], _base_z])
                st["arm_base_lula_fixed"] = _arm_base_lula_fixed  # 틱 간 유지되도록 st에 저장
                lula.set_robot_base_pose(_arm_base_lula_fixed, _R_to_quat(_R_z(ARM_BASE_YAW_DEG)))
                print(f"  ★ [plant] Lula base z 보정: AMR 루트 world z={float(_pa_settled[2]):.4f}m 반영 "
                      f"— base {_arm_base_lula.round(4)} → {_arm_base_lula_fixed.round(4)}")
                # [BASE검증] 위 합성값(정착 전/후 값을 더한 추정치)을 실측과 대조 —
                # 틀리면 IK/FK가 전부 그만큼 틀려 그리퍼가 벨트를 박는 원인이 된다.
                try:
                    _bl_v = SingleXFormPrim(prim_path=BASE_LINK, name="dbg_bl")
                    _bl_p, _ = _bl_v.get_world_pose()
                    _ab_path = None
                    for _pr in Usd.PrimRange(stage.GetPrimAtPath(arm_path)):
                        if _pr.HasAPI(UsdPhysics.RigidBodyAPI) and "base" in _pr.GetName().lower():
                            _ab_path = str(_pr.GetPath()); break
                    _ab_p = None
                    if _ab_path:
                        _ab_p, _ = SingleXFormPrim(prim_path=_ab_path, name="dbg_ab").get_world_pose()
                    print("  ================ [BASE검증] Lula base vs 실측 ================")
                    print(f"    가정: root z {float(_pa_settled[2]):.4f} + t_bl z {float(_t_bl[2]):.4f}"
                          f" = BASE_LINK z {float(_pa_settled[2])+float(_t_bl[2]):.4f}")
                    print(f"    실측 BASE_LINK world = {np.asarray(_bl_p, float).round(4)}")
                    if _ab_p is not None:
                        print(f"    실측 arm_base world  = {np.asarray(_ab_p, float).round(4)}"
                              f"   ← Lula base가 여기여야 함")
                        _d = np.asarray(_ab_p, float) - _arm_base_lula_fixed
                        print(f"    Lula에 준 base       = {_arm_base_lula_fixed.round(4)}")
                        print(f"    ★ 오차 = {(_d*1000).round(1)} mm  (|Δz|={abs(_d[2])*1000:.1f}mm)")
                    print("  ==============================================================")
                except Exception as _e:
                    print(f"    [BASE검증] 실패: {type(_e).__name__}: {_e}")
                print("\n  [plant] settle 완료 — 각 노드가 넘겨받습니다\n")
            elif st["k"] == SETTLE_STEPS:
                print("  [plant] settle 완료 — vision/amr/arm/gripper 노드 접속 대기 중"
                      "(★ 이 동안은 벨트가 안 움직입니다)…")
            return

        st["k"] += 1
        k = st["k"]
        stamp = ros.get_clock().now().to_msg()
        ros.pub_tick.publish(Int32(data=k))

        pa, qa = amr.get_world_pose()
        xa, ya = float(pa[0]), float(pa[1])
        psi = _yaw(qa)
        # Lula base orientation을 실측 yaw(psi)로 매틱 갱신 — AMR이 계속 움직이는
        # 지금은 settle 시점 고정값(ARM_BASE_YAW_DEG)만으로는 어긋난다.
        lula.set_robot_base_pose(st["arm_base_lula_fixed"], _R_to_quat(_R_z(math.degrees(psi))))
        po, _ = obj_view.get_world_pose()
        dx, dy = float(po[0])-xa, float(po[1])-ya
        cs, sn = math.cos(psi), math.sin(psi)
        g_bx = cs*dx + sn*dy
        g_by = -sn*dx + cs*dy

        # ── 외란 주입 (실험 하네스 — 4개 제어노드는 이 값을 모른다) ─────────
        # 턴테이블 회전속도 외란 전용 로직 — 컨베이어 모드(tt=None)는 아래 별도 블록이 담당.
        while (tt is not None and not STATIC_TEST and st["belt_i"] < len(BELT_SCHEDULE)
               and k >= BELT_SCHEDULE[st["belt_i"]][0]):
            _bk, _bw = BELT_SCHEDULE[st["belt_i"]]
            _old = st["belt_om"]
            tt.get_articulation_controller().apply_action(
                ArticulationAction(joint_velocities=np.array([_bw])))
            st["belt_om"] = _bw
            st["belt_i"] += 1
            st["dist_active"] = True; st["dist_k0"] = k; st["dist_run"] = 0
            st["dist_max_bx"] = 0.0; st["dist_max_by"] = 0.0
            print(f"\n  {'~'*66}")
            print(f"  ★ 외란 주입 (스텝 {k}, t={k/60.0:.2f}s): "
                  f"턴테이블 ω {_old:+.3f} → {_bw:+.3f} rad/s "
                  f"({(_bw-_old)/abs(_old)*100:+.0f}%)")
            print(f"  {'~'*66}\n")

        if st["dist_active"]:
            st["dist_max_bx"] = max(st["dist_max_bx"], abs(g_bx))
            st["dist_max_by"] = max(st["dist_max_by"], abs(g_by - KEEP_DIST))
            _rel = abs(ros.om_hat - st["belt_om"]) / max(abs(st["belt_om"]), 1e-6)
            if _rel < OM_RECOVER_TOL: st["dist_run"] += 1
            else: st["dist_run"] = 0
            if st["dist_run"] >= OM_RECOVER_HOLD:
                _rt = (k - st["dist_k0"]) / 60.0
                print(f"\n  ✓ 외란 회복 (스텝 {k}): Ω̂ {ros.om_hat:+.4f} → "
                      f"목표 {st['belt_om']:+.4f} 이내 {OM_RECOVER_TOL*100:.0f}%")
                print(f"      회복 소요 {_rt:.2f}s  |  이탈 최대 bx "
                      f"{st['dist_max_bx']*1000:.1f}mm, by {st['dist_max_by']*1000:.1f}mm\n")
                st["dist_active"] = False

        # ── 컨베이어 속도 외란 (컨베이어 모드 전용, tt=None일 때만) ─────────
        # 구간(이벤트 기준): A(시작~lock) → B(lock~grasp 종료/grip 진입 직전) → C(grip~, 원복)
        if CONV_DISTURB_ENABLED and tt is None and abs(CONV_VELOCITY) > 1e-9:
            _next_phase = _cv = None
            if st["conv_phase"] == "A" and ros.amr_locked:
                _next_phase, _cv = "B", CONV_DISTURB_PHASE_B
            elif (st["conv_phase"] == "B" and st["conv_arm_prev"] == "grasp"
                  and ros.arm_status == "grip"):
                _next_phase, _cv = "C", CONV_VELOCITY
            if _next_phase is not None:
                _cold = st["conv_v"]
                _belt_prim = stage.GetPrimAtPath(CONV_BELT_PATH)
                PhysxSchema.PhysxSurfaceVelocityAPI(_belt_prim).GetSurfaceVelocityAttr().Set(
                    Gf.Vec3f(0.0, _cv, 0.0))
                st["conv_v"] = _cv
                st["conv_phase"] = _next_phase
                st["conv_dist_active"] = True; st["conv_dist_k0"] = k; st["conv_dist_run"] = 0
                st["conv_dist_max_bx"] = 0.0; st["conv_dist_max_by"] = 0.0
                print(f"\n  {'~'*66}")
                print(f"  ★ 컨베이어 외란 구간 {_next_phase} 진입 (스텝 {k}, t={k/60.0:.2f}s): "
                      f"v {_cold*1000:+.1f} → {_cv*1000:+.1f} mm/s")
                print(f"  {'~'*66}\n")
            st["conv_arm_prev"] = ros.arm_status

        if st["conv_dist_active"]:
            st["conv_dist_max_bx"] = max(st["conv_dist_max_bx"], abs(g_bx))
            st["conv_dist_max_by"] = max(st["conv_dist_max_by"], abs(g_by - KEEP_DIST))
            _crel = abs(ros.om_hat - st["conv_v"]) / max(abs(st["conv_v"]), 1e-6)
            if _crel < CONV_RECOVER_TOL: st["conv_dist_run"] += 1
            else: st["conv_dist_run"] = 0
            if st["conv_dist_run"] >= CONV_RECOVER_HOLD:
                _crt = (k - st["conv_dist_k0"]) / 60.0
                print(f"\n  ✓ 컨베이어 외란 회복 (스텝 {k}): v̂ {ros.om_hat*1000:+.1f}mm/s → "
                      f"목표 {st['conv_v']*1000:+.1f}mm/s 이내 {CONV_RECOVER_TOL*100:.0f}%")
                print(f"      회복 소요 {_crt:.2f}s  |  이탈 최대 bx "
                      f"{st['conv_dist_max_bx']*1000:.1f}mm, by {st['conv_dist_max_by']*1000:.1f}mm\n")
                st["conv_dist_active"] = False

        # ── camera_info / 정적 tf 재발행 (초반 20초, 1초 주기) — 늦게 뜬 vision_node가 즉시 받도록 ──
        if k <= 1200 and k % 60 == 0:
            ci_c, ci_e, R_s, t_s = ros._caminfo_payload
            ros.pub_chassis_info.publish(ci_c)
            ros.pub_eih_info.publish(ci_e)
            ros.tf_static_broadcaster.sendTransform(
                _tf_msg(ros, R_s, t_s, "body_link", "chassis_cam", stamp))

        # ── 카메라 publish (10Hz) ───────────────────────────────────────────
        if k % CAM_EVERY == 0 and cam is not None:
            try:
                img = cam.get_rgba()
                if img is not None and getattr(img, "size", 0) > 0:
                    ros.pub_chassis_img.publish(
                        _img_msg(ros, np.asarray(img), f"chassis_cam:{k}", stamp))
            except Exception: pass
        if k % EIH_EVERY == 0 and eih_cam is not None:
            try:
                img = eih_cam.get_rgba()
                if img is not None and getattr(img, "size", 0) > 0:
                    ros.pub_eih_img.publish(
                        _img_msg(ros, np.asarray(img), f"eih_cam:{k}", stamp))
            except Exception: pass
            # 끝단캠 tf는 팔이 움직이므로 매틱 브로드캐스트
            # parent를 라이브 pa + yaw(psi)로 재구성해서 정지된 PRIM 쿼리를 우회
            # (qa 전체를 쓰면 Z가 어긋남 — 이 좌표계는 순수 yaw만 가정하고 짜여 있음).
            xc2 = UsdGeom.XformCache()
            R_e, t_e = _prim_transform_live_parent(
                xc2, eih_prim, pa, _R_to_quat(_R_z(math.degrees(psi))))
            ros.tf_broadcaster.sendTransform(
                _tf_msg(ros, R_e, t_e, "body_link", "eih_cam", stamp))

        # ── joint_states / ee_pose (매틱) ──────────────────────────────────
        jp = amr.get_joint_positions(); jv = amr.get_joint_velocities()
        js = JointState()
        js.header.stamp = stamp
        js.name = dof_names
        js.position = [float(v) for v in jp]
        js.velocity = [float(v) for v in jv]
        ros.pub_joint_states.publish(js)

        # 몸체 yaw 각속도(IMU 자이로 상당) — 실제 자세 변화에서 뽑으므로 바퀴 슬립에
        # 오염되지 않는다. amr_node의 place_turn이 이걸 적분해 정확히 90도를 채운다.
        _qa_imu = amr.get_world_pose()[1]
        _yaw_imu = _yaw(_qa_imu)
        _prev_yaw = st.get("imu_prev_yaw")
        if _prev_yaw is not None:
            _dyaw = math.atan2(math.sin(_yaw_imu - _prev_yaw), math.cos(_yaw_imu - _prev_yaw))
            ros.pub_imu_yaw_rate.publish(Float32(data=float(_dyaw * 60.0)))
        st["imu_prev_yaw"] = _yaw_imu

        ee_p, ee_R = art_ik.compute_end_effector_pose()
        ros.pub_ee_pose.publish(Point(x=float(ee_p[0]), y=float(ee_p[1]), z=float(ee_p[2])))

        # [프레임진단] grip 진입 순간 1회 — Lula가 "link6"이라 부르는 프레임이 실제 link6과
        # 일치하는지, 실제 USD 프림 위치를 몸체좌표로 찍어 확인한다.
        if ros.arm_status == "grip" and not st.get("frame_dbg_done"):
            st["frame_dbg_done"] = True
            try:
                _xcf = UsdGeom.XformCache()
                _bl = stage.GetPrimAtPath(BASE_LINK)
                _names = {}
                for _pr in Usd.PrimRange(stage.GetPrimAtPath(arm_path)):
                    _n = _pr.GetName()
                    if _n in ("link6", "link7", "link8", "RSD455") and _n not in _names:
                        _names[_n] = _pr
                print("\n  ================ [프레임진단] grip 순간 (몸체좌표) ================")
                print(f"    물체 중심(실측)        = {np.array([g_bx, g_by, float(po[2])]).round(4)}")
                print(f"    Lula가 보고한 link6    = {np.asarray(ee_p, float).round(4)}  ← IK/FK가 쓰는 프레임")
                for _n in ("link6", "RSD455", "link7", "link8"):
                    if _n in _names:
                        _, _t = _prim_transform(_xcf, _names[_n], _bl)
                        print(f"    실제 USD {_n:8s}      = {np.asarray(_t, float).round(4)}")
                print("  ==================================================================\n")
            except Exception as _e:
                print(f"    [프레임진단] 실패: {type(_e).__name__}: {_e}")

        if (ros.arm_status in ("pre", "grasp") and ros.cartesian_target is not None
                and k % 60 == 0):
            _d = np.asarray(ee_p, float) - np.asarray(ros.cartesian_target, float)
            print(f"    [도달진단 {ros.arm_status}] target={np.round(ros.cartesian_target,4)} "
                  f"ee_fk={np.round(np.asarray(ee_p,float),4)} |Δ|={np.linalg.norm(_d)*1000:.1f}mm "
                  f"q(deg)={np.degrees(amr.get_joint_positions()[arm_idx]).round(1)}")

        # ── 접촉력 (60Hz) ───────────────────────────────────────────────────
        F_con, F_pads = _grip_contact_force(pad_view, dt)
        if F_con is not None:
            ros.pub_contact_force.publish(Float32(data=float(F_con)))
            ros.pub_contact_pads.publish(Float32MultiArray(data=[float(v) for v in F_pads]))

        # [벨트충돌] 손가락/손목이 벨트를 박는지 — 처음 잡히는 순간 1회 보고(진단 전용)
        if belt_view and not st.get("belt_hit_done"):
            _, F_belt = _grip_contact_force(belt_view, dt)
            if F_belt is not None and float(np.max(F_belt)) > 0.5:
                _i = int(np.argmax(F_belt))
                print(f"    ★★★ [벨트충돌] **{belt_names[_i]}**이 벨트를 박고 있다 — "
                      f"{F_belt[_i]:.1f}N @step{k} (팔 단계={ros.arm_status})  "
                      f"전체=[{', '.join(f'{n}:{v:.1f}' for n, v in zip(belt_names, F_belt))}]")
                st["belt_hit_done"] = True

        # [높이추적] 손가락 최하점의 실제 world z를 벨트/물체 높이와 나란히 진단 출력
        if ros.arm_status in ("pre", "grasp", "grip") and k % 30 == 0:
            try:
                _xch = UsdGeom.XformCache()
                _l6 = None
                for _pr in Usd.PrimRange(stage.GetPrimAtPath(arm_path)):
                    if _pr.GetName() == "link6":
                        _l6 = _pr; break
                if _l6 is not None:
                    _M = _xch.GetLocalToWorldTransform(_l6)
                    _R6 = _quat_to_R(_gfquat_wxyz(_M.ExtractRotationQuat()))
                    _t6 = np.array(_M.ExtractTranslation(), float)
                    # 손가락 충돌메시(link6 로컬) 최하점 후보들을 world로 옮겨 최소 z
                    _cand = [np.array([sx, sy, sz]) for sx in (-0.0762, 0.0762)
                             for sy in (-0.029, 0.027) for sz in (0.0585, 0.135)]
                    _zmin = min(float((_t6 + _R6 @ c)[2]) for c in _cand)
                    _belt_top = CONV_Z + CONV_H/2.0
                    print(f"    [높이추적 {ros.arm_status}] 손가락 최하점 z={_zmin:.4f}  "
                          f"물체윗면={float(po[2])+OBJ_S/2:.4f}  벨트윗면={_belt_top:.4f}  "
                          f"→ 벨트까지 {(_zmin-_belt_top)*1000:+.0f}mm")
                # [AMR자세] 팔이 뻗으면 AMR이 가라앉거나 기울어 Lula base(settle 시 1회 고정)와
                # 어긋나는지 확인 — 어긋나면 FK가 실제보다 높게 나와 벨트를 박는 원인이 된다.
                _pa_now, _qa_now = amr.get_world_pose()
                _R_amr = _quat_to_R(np.array(_qa_now, float))
                _up = _R_amr @ np.array([0.0, 0.0, 1.0])
                _tilt = math.degrees(math.acos(float(np.clip(_up[2], -1.0, 1.0))))
                _dz = float(_pa_now[2]) - float(st["pa_settled"][2]) \
                    if st.get("pa_settled") is not None else float("nan")
                print(f"    [AMR자세 {ros.arm_status}] world z={float(_pa_now[2]):.4f} "
                      f"(settle 대비 {_dz*1000:+.1f}mm)  기울기={_tilt:.2f}°  "
                      f"yaw={math.degrees(_yaw(_qa_now)):.2f}° (Lula 가정 {ARM_BASE_YAW_DEG:.0f}°)")
            except Exception:
                pass

        if body_view and not st.get("body_hit_done"):
            _, F_body = _grip_contact_force(body_view, dt)
            if F_body is not None and float(np.max(F_body)) > 0.5:
                _i = int(np.argmax(F_body))
                print(f"    ★★★ [간섭 감지] 손가락이 아닌 **{body_names[_i]}**이 물체를 "
                      f"밀고 있다 — {F_body[_i]:.1f}N @step{k} (팔 단계={ros.arm_status})  "
                      f"전체=[{', '.join(f'{n}:{v:.1f}' for n, v in zip(body_names, F_body))}]")
                st["body_hit_done"] = True

        # ── 명령 적용 ────────────────────────────────────────────────────────
        amr.apply_action(ArticulationAction(
            joint_positions=np.array(ros.steer_cmd, float), joint_indices=steer_idx))
        amr.apply_action(ArticulationAction(
            joint_velocities=np.array(ros.drive_cmd, float), joint_indices=drive_idx))

        if ros.arm_status != "hover":
            st["hover_q_target"] = None  # hover 이탈 시 캐시 무효화 — 재진입(재시도) 시 새로 산출

        # hover(pitch30/roll0)→pre(pitch60/roll180) 진입 순간 감지 — 자세가 한번에
        # 바뀌어 손목이 크게 재배치되는 걸 매틱 IK로 그대로 서보하면 홱 도는 것처럼
        # 보인다(사용자 관찰). 진입 첫 PRE_ENTRY_BLEND_SEC초만 관절보간으로 넘긴다.
        if (st["prev_arm_status"] != "pre" and ros.arm_status == "pre"
                and ros.cartesian_target is not None):
            # hover→detect→pre로 이어져 "hover" 직후가 아니라 "detect" 직후에 pre가
            # 시작된다 — prev_arm_status=="hover" 조건이 한 번도 안 걸리던 버그 수정.
            act0, ok0 = art_ik.compute_inverse_kinematics(
                target_position=ros.cartesian_target, target_orientation=DOWN_QUAT)
            if ok0 and act0.joint_positions is not None and not np.any(np.isnan(act0.joint_positions)):
                st["reorient_q0"] = amr.get_joint_positions()[arm_idx].copy()
                st["reorient_qtarget"] = np.array(act0.joint_positions, float)
                # PGL_FREE_JOINT6=False면 여기서도 joint6=0으로 고정 — 아래 pre/grasp/lift IK
                # 분기와 짝을 맞춰야 진입 직후 손목이 다시 홱 도는 걸 피할 수 있다.
                if not PGL_FREE_JOINT6:
                    st["reorient_qtarget"][5] = 0.0
                st["reorient_t0"] = k
                print(f"    [PRE 진입] 손목 재배치 관절보간 시작 "
                      f"q0(deg)={np.degrees(st['reorient_q0']).round(1)} "
                      f"q_target(deg)={np.degrees(st['reorient_qtarget']).round(1)}")

        # place 선반 위치 튜닝 진단용 — 픽 완료 직후 AMR이 실제로 서 있는 world 위치를 1회만 찍는다
        if st["prev_arm_status"] != "lift" and ros.arm_status == "lift":
            _pa_lift, _ = amr.get_world_pose()
            print(f"    [place진단] lift 진입 시점 AMR world pose="
                  f"{np.round(np.array(_pa_lift, float), 3)}")
            # 물체의 LockedRotAxis(벨트 위 회전 아티팩트 방지용)를 여기서 해제 — 잠긴 채로
            # place_turn(제자리 회전) 중 그리퍼와 함께 돌지 못해 접촉 기하가 어긋나며 빠졌었다.
            # 잡힌 뒤에는 벨트 마찰로 돌 일이 없어 풀어도 안전하다.
            PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(OBJ_PATH)).CreateLockedRotAxisAttr(0)

            # place 선반을 "이번 락 pose 기준 상대offset"으로 재배치 — AMR 트래킹 중 실제 yaw를
            # 교정할 수단이 없어 런마다 편차가 크므로, 정적 world 좌표 대신 락 시점 위치+yaw에서
            # 상대 오프셋(뒤로 BEHIND_DIST, 벨트에서 멀어지는 쪽으로 LATERAL_DIST)으로 매번
            # 재배치(teleport)한다 — 헤딩이 얼마든 이후 도킹 기하가 항상 일관된다.
            psi_lift = _yaw(qa)
            cs2, sn2 = math.cos(psi_lift), math.sin(psi_lift)
            off_x = -PLACE_SHELF_BEHIND_DIST*cs2 + PLACE_SHELF_LATERAL_DIST*sn2
            off_y = -PLACE_SHELF_BEHIND_DIST*sn2 - PLACE_SHELF_LATERAL_DIST*cs2
            shelf_x, shelf_y = float(_pa_lift[0])+off_x, float(_pa_lift[1])+off_y
            shelf_yaw_deg = math.degrees(psi_lift + math.pi/2.0)
            _shelf_prim = stage.GetPrimAtPath(PLACE_SHELF_PATH)
            _shelf_ops = UsdGeom.Xformable(_shelf_prim).GetOrderedXformOps()
            _shelf_ops[0].Set(Gf.Vec3d(shelf_x, shelf_y, PLACE_SHELF_POS[2]))
            _shelf_ops[1].Set(shelf_yaw_deg)
            print(f"    [place진단] 선반 재배치: 락 yaw={math.degrees(psi_lift):.1f}° → "
                  f"선반 pos=({shelf_x:.3f}, {shelf_y:.3f}, {PLACE_SHELF_POS[2]:.3f}) "
                  f"yaw={shelf_yaw_deg:.1f}°")

        st["prev_arm_status"] = ros.arm_status

        if ros.arm_status in ("wait", "place_ready", "place_home"):
            amr.apply_action(ArticulationAction(
                joint_positions=ros.joint_hold_target, joint_indices=arm_idx))
        elif ros.arm_status == "hover":
            # HOVER는 매틱 Cartesian IK 대신 1회 IK로 관절목표를 구해 관절보간으로 이동한다 —
            # 매틱 재계산 시 SEARCH_Q(중립)→hover 사이 손목특이점 부근에서 해가 튀어 link6이
            # 도는 문제 대응. 실패하면 hover_q_target을 None으로 남겨 다음 틱에 재시도한다.
            if st["hover_q_target"] is None:
                st["hover_q0"] = amr.get_joint_positions()[arm_idx].copy()
                act_h, ok_h = art_ik.compute_inverse_kinematics(
                    target_position=HOVER_TARGET, target_orientation=HOVER_DOWN_QUAT)
                if ok_h and act_h.joint_positions is not None and not np.any(np.isnan(act_h.joint_positions)):
                    st["hover_q_target"] = np.array(act_h.joint_positions, float)
                    # HOVER_DOWN_QUAT이 롤까지 구속하므로 joint6은 결정된 값(≈+91°)이지
                    # 여유자유도가 아니다 — 0으로 박으면 PRE 진입 시 손목이 물체 위에서
                    # 91° 되돌아가며 손가락 판이 물체를 쓸고 지나간다(실측 확인).
                    if not PGL_FREE_JOINT6:
                        st["hover_q_target"][5] = 0.0
                    st["hover_ik_fail_n"] = 0
                    print(f"    [HOVER] IK 새로 풀어서 관절목표 계산 "
                          f"q0(deg)={np.degrees(st['hover_q0']).round(1)} "
                          f"q_target(deg)={np.degrees(st['hover_q_target']).round(1)}")
                    st["hover_t0"] = k
                else:
                    st["hover_ik_fail_n"] = st.get("hover_ik_fail_n", 0) + 1
                    if st["hover_ik_fail_n"] % 60 == 1:
                        print(f"    [HOVER] ★ IK 실패({st['hover_ik_fail_n']}회째, 다음 틱 재시도) "
                              f"q0(deg)={np.degrees(st['hover_q0']).round(1)}")
            if st["hover_q_target"] is not None:
                t = min((k - st["hover_t0"]) / (HOVER_JOINT_MOVE_SEC * 60.0), 1.0)
                q = st["hover_q0"] + t * (st["hover_q_target"] - st["hover_q0"])
                amr.apply_action(ArticulationAction(
                    joint_positions=q, joint_indices=arm_idx))
        elif (st["reorient_qtarget"] is not None
              and (k - st["reorient_t0"]) < PRE_ENTRY_BLEND_SEC * 60.0):
            t = min((k - st["reorient_t0"]) / (PRE_ENTRY_BLEND_SEC * 60.0), 1.0)
            q = st["reorient_q0"] + t * (st["reorient_qtarget"] - st["reorient_q0"])
            amr.apply_action(ArticulationAction(joint_positions=q, joint_indices=arm_idx))
        elif ros.arm_status in ("pre", "grasp", "lift", "place_lower",
                                 "place_hover", "place_detect", "place_descend",
                                 "place_release", "place_retreat") \
                and ros.cartesian_target is not None:
            # 이 화이트리스트에 상태가 빠지면 IK가 전혀 안 풀려 팔이 그 자리에 얼어붙는데
            # arm_node는 (_move_l이 시간기준이라) 모르고 다음 단계로 진행해버린다 — place_*
            # 상태 추가 시 반드시 여기도 같이 넣을 것. orientation은 DOWN_QUAT으로 고정
            # 요청한다(위치만 주면 손목이 180° 뒤집힌 해로도 수렴하는 모호성이 있었다).
            # art_ik.compute_inverse_kinematics()는 매 틱 현재 관절값을 시드로 쓰는데, 그
            # 출력이 다음 틱 입력이 되는 피드백 루프라 joint6을 0으로 덮어쓰는 것과 맞물려
            # 관절이 서서히 드리프트했다(hover는 진입 시 1회만 풀어서 이 문제가 없었음).
            # 그래서 lula를 직접 불러 시드를 우리가 관리한다: 단계 진입 시점엔 관절값으로
            # 시드를 붙박고, 이후엔 매 틱 "직전 IK 해"로 시드를 갱신한다(순수 계산 루프라
            # 물리 피드백이 안 섞임 — 완전 고정 시드는 반대로 먼 목표에 수렴 못 하는 문제가 있었다).
            _subset = art_ik.get_joints_subset()
            if st["pgl_prev_status"] != ros.arm_status:
                st["pgl_ik_fail_n"] = 0
                # place_hover만 예외 — place_wait 동안 팔이 lift/verify 자세(전혀 다른 방향)에
                # 얼어붙어 있어 그대로 warm-start하면 이상한 해로 수렴한다(실측 확인). 픽의
                # hover처럼 SEARCH_Q(대기자세)에서 새로 풀게 한다.
                st["pgl_seed"] = (np.array(SEARCH_Q, float) if ros.arm_status == "place_hover"
                                  else _subset.get_joint_positions())
                st["pgl_roll_dbg"] = 0
            st["pgl_prev_status"] = ros.arm_status
            if st["pgl_seed"] is None:
                st["pgl_seed"] = _subset.get_joint_positions()
            # 마커기반 place_hover/detect/descend만 PLACE_DOWN_QUAT(pitch=30)을 쓴다. 나머지
            # (place_lower/release/retreat)는 단순화 경로가 DOWN_QUAT(pitch=0) 자세에서 곧장
            # 넘어오므로 DOWN_QUAT을 써야 한다 — 안 맞추면 IK가 팔을 이상하게 꺾는다(실측 확인).
            _ik_quat = (PLACE_DOWN_QUAT
                        if ros.arm_status in ("place_hover", "place_detect", "place_descend")
                        else DOWN_QUAT)
            _jp, ok = lula.compute_inverse_kinematics(
                EE_FRAME, ros.cartesian_target, _ik_quat, st["pgl_seed"])
            if not ok:
                # 고정 시드가 해에서 너무 멀어 수렴 못 하는 경우의 폴백 — 현재 관절로 재시도.
                # 실패했다고 명령을 아예 안 보내면 팔이 얼어붙는데 arm_node는 그걸 모르고
                # 시간 기준으로 다음 단계로 넘어가버리므로, 드리프트보다 이쪽이 낫다.
                _jp, ok = lula.compute_inverse_kinematics(
                    EE_FRAME, ros.cartesian_target, _ik_quat,
                    _subset.get_joint_positions())
                if ok:
                    st["pgl_seed"] = np.array(_jp, float)  # 수렴한 해로 시드 재설정
            if ok and _jp is not None and not np.any(np.isnan(_jp)):
                st["pgl_seed"] = np.array(_jp, float)
            act = _subset.make_articulation_action(_jp, None) if _jp is not None else None
            used_ori = True
            ik_valid = (ok and act is not None and act.joint_positions is not None
                        and not np.any(np.isnan(act.joint_positions)))
            if not ik_valid:
                # IK가 조용히 실패하면 관절명령을 안 보내 팔이 그 자리에 멈추는데, arm_node는
                # 그걸 모르고 시간기준으로 진행해버린다 — 60틱마다 경고 출력해 드러낸다.
                st["pgl_ik_fail_n"] += 1
                if st["pgl_ik_fail_n"] % 60 == 1:
                    cur_q = amr.get_joint_positions()[arm_idx]
                    print(f"    [{ros.arm_status.upper()}] ★ IK 실패({st['pgl_ik_fail_n']}회째, "
                          f"관절 유지) target={ros.cartesian_target.round(3)} "
                          f"cur_q(deg)={np.degrees(cur_q).round(1)}")
            if ik_valid:
                q6 = np.array(act.joint_positions, float)
                # PGL_FREE_JOINT6=False면 joint6을 0으로 고정한다 — DOWN_QUAT은 롤까지
                # 포함한 6구속이라 joint6은 결정된 값이지 여유자유도가 아니다. 0으로
                # 덮어쓰면 link6 위치는 그대로지만 그리퍼가 툴축 둘레로 굴러가 손가락
                # 개폐축이 의도와 어긋난다 — 아래 [ROLL진단]이 그 오차를 찍어준다.
                _q6_ik = float(q6[5])
                if not PGL_FREE_JOINT6:
                    q6[5] = 0.0
                if not np.any(np.isnan(q6)):
                    # 단계 진입 직후 30틱만 IK해 궤적을 출력(특이점 조사용 디버그)
                    if st["hover_dbg_prev_status"] != ros.arm_status:
                        st["hover_dbg_n"] = 0
                    st["hover_dbg_prev_status"] = ros.arm_status
                    if st["hover_dbg_n"] < 30:
                        cur_q = amr.get_joint_positions()[arm_idx]
                        print(f"    [IK디버그 {ros.arm_status} #{st['hover_dbg_n']:02d}] "
                              f"target={ros.cartesian_target.round(3)} ori={'DOWN' if used_ori else 'None'} "
                              f"cur_q(deg)={np.degrees(cur_q).round(1)} "
                              f"sol_q(deg)={np.degrees(q6).round(1)} "
                              f"dq(deg)={np.degrees(q6-cur_q).round(1)}")
                        st["hover_dbg_n"] += 1
                    # [ROLL진단] joint6 고정이 손가락 개폐축을 실제로 얼마나 돌려놓는지 — 오차가
                    # 크면 고칠 곳은 q6[5]가 아니라 DOWN_QUAT의 롤(GRASP_ROLL_DEG)이다.
                    if st["pgl_roll_dbg"] < 3:
                        if PGL_FREE_JOINT6:
                            print(f"    [ROLL진단 {ros.arm_status}] joint6="
                                  f"{math.degrees(_q6_ik):+.1f}° 그대로 적용(고정 해제) "
                                  f"∴ 손가락 개폐축 = 의도대로 수평")
                        else:
                            print(f"    [ROLL진단 {ros.arm_status}] IK가 원한 joint6="
                                  f"{math.degrees(_q6_ik):+.1f}° → 0°로 고정  ∴ 손가락 개폐축이 "
                                  f"접근 pitch({GRASP_PITCH_DEG:.0f}°)만큼 아래로 기울어 닫힘")
                        st["pgl_roll_dbg"] += 1
                    amr.apply_action(ArticulationAction(
                        joint_positions=q6, joint_indices=arm_idx))
        # detect/grip/done/fail: 팔 목표 유지(아무 것도 안 보냄)

        if ros.gripper_cmd_pos is not None:
            names, pos = ros.gripper_cmd_pos
            idx = np.array([dof_names.index(n) for n in names if n in dof_names], dtype=int)
            if len(idx) == len(pos):
                amr.apply_action(ArticulationAction(
                    joint_positions=pos, joint_indices=idx))

        # ── 시계열 로그 ──────────────────────────────────────────────────────
        if POSE_LOG_EVERY and k % POSE_LOG_EVERY == 0 and st["plog"] is not None:
            wL = float(jv[drive_idx[0]]); wR = float(jv[drive_idx[1]])  # fl,fr 참고용
            lin_w = np.asarray(amr.get_linear_velocity(), float)
            ang_w = np.asarray(amr.get_angular_velocity(), float)
            v_amr = float(math.cos(psi)*lin_w[0] + math.sin(psi)*lin_w[1])
            w_amr = float(ang_w[2])
            try: disc_w = float(tt.get_joint_velocities()[0])
            except Exception: disc_w = 0.0
            st["plog"].writerow([k, f"{k/60.0:.3f}", f"{xa:.4f}", f"{ya:.4f}", f"{psi:.4f}",
                                  f"{wL:.4f}", f"{wR:.4f}", f"{v_amr:.4f}", f"{w_amr:.4f}",
                                  f"{disc_w:.4f}", f"{float(po[0]):.4f}", f"{float(po[1]):.4f}",
                                  f"{float(po[2]):.4f}", f"{g_bx:.4f}", f"{g_by:.4f}",
                                  f"{ros.om_hat:.4f}", st["phase"], ros.arm_status])

        if TRACE_EVERY and k % TRACE_EVERY == 0:
            print(f"  [추적 {k:>4}] god bx {g_bx*1000:>+5.0f} by {g_by*1000:>5.0f}  "
                  f"Ω̂ {ros.om_hat:+.4f}  팔[{ros.arm_status}]")

        # ── 파지 결과 채점 (arm_node의 /arm/pick_event 트리거) ──────────────
        # god's-eye 좌표로 채점만 함 — 제어엔 안 씀(실물엔 없는 값).
        if ros.pick_event is not None:
            cam_ok = bool(ros.pick_event)          # arm_node의 차체캠 판정
            z_now = float(po[2])
            z_ref = CONV_Z + CONV_H/2 + OBJ_S/2.0 if USE_CONVEYOR else DISC_Z + DISC_H/2 + OBJ_S/2.0
            truth_ok = (z_now - z_ref) > 0.05      # 실측 정답(시뮬 전용)
            st["attempt_n"] += 1
            agree = (cam_ok == truth_ok)
            print(f"\n  {'='*60}")
            print(f"  파지 결과 (시도 {st['attempt_n']}): {'✓ 성공' if truth_ok else '✗ 실패(헛집음)'}  [실측]")
            print(f"    [실측 높이] {z_now*1000:.0f}mm "
                  f"({'+' if z_now-z_ref>=0 else ''}{(z_now-z_ref)*1000:.0f}mm, 기준 {z_ref*1000:.0f}mm)")
            print(f"    [차체캠 판정] {'성공' if cam_ok else '실패'} → "
                  f"{'✓ 실측과 일치' if agree else '★ 실측과 불일치(오판)'}")
            print(f"  {'='*60}\n")

            os.makedirs(OUT_DIR, exist_ok=True)
            _write_header = not os.path.exists(RESULT_FILE)
            with open(RESULT_FILE, "a", newline="") as f:
                w = csv.writer(f)
                if _write_header:
                    w.writerow(["timestamp","pick_ok","cam_verdict","verdict_agree",
                                "om_hat_err_pct","attempt_n"])
                _o_err = (ros.om_hat-OMEGA_BELT)/OMEGA_BELT*100
                w.writerow([datetime.datetime.now().strftime("%H:%M:%S"),
                            truth_ok, cam_ok, agree, f"{_o_err:+.2f}", st["attempt_n"]])
            with open(RESULT_FILE) as f:
                rows = [r.strip().split(",") for r in f if not r.startswith("timestamp")]
            n = len(rows); ok2 = sum(1 for r in rows if r and r[1]=="True")
            print(f"  [누적 성공률] {ok2}/{n}회 ({100*ok2/max(n,1):.0f}%)  (시도 단위 집계, 재시도 포함)")

            ros.pick_event = None

        # ── 시퀀스 최종 종료 감지 (현재 비활성화 — done/fail이어도 world는 계속 돈다) ──
        # arm_node가 성공(done) 또는 포기(done)/검출실패(fail)로 멈췄을 때 정지시키려면:
        # if not st["finished"] and ros.arm_status in ("done", "fail"):
        #     st["finished"] = True
        #     _finish_run("파지 시퀀스 완료" if ros.arm_status == "done"
        #                 else "마커 검출 실패로 중단")

        # ── 파탄/타임아웃 감지 ───────────────────────────────────────────────
        if g_by < BY_FAIL:
            _finish_run(f"충돌 위험 by {g_by:+.3f}")
        elif k >= MAX_STEPS:
            _finish_run("MAX_STEPS 타임아웃")

    world.add_physics_callback("plant", on_physics)
    if BELT_SCHEDULE:
        print("  ★ 외란 스케줄:", BELT_SCHEDULE)
    await world.play_async()
    print("\n  [plant] 시작 — vision/amr/arm/gripper 노드를 각자 터미널에서 켜세요\n")

    return ros


ros_node_holder = {}

async def _main_wrapper():
    ros_node_holder["ros"] = await main()

omni.kit.async_engine.run_coroutine(_main_wrapper())

t0 = time.time()
started = False
while simulation_app.is_running():
    simulation_app.update()
    el = time.time() - t0
    ros = ros_node_holder.get("ros")
    if ros is not None:
        w = World.instance()
        try:
            playing = w.is_playing() if w else False
        except Exception:
            playing = False
        if playing:
            started = True
        elif started:
            print(f"\n  [런처] 시뮬레이션 정지 감지 — 종료 (경과 {el:.1f}s)")
            break
    if el > args.max_sec:
        print(f"\n  [런처] 벽시계 상한 {args.max_sec:.0f}s 초과 — 강제 종료")
        break

if args.keep_open and not args.headless:
    print("\n  [런처] --keep-open: 창을 유지합니다. 닫으면 종료됩니다.")
    while simulation_app.is_running():
        simulation_app.update()
        ros = ros_node_holder.get("ros")
        if ros is not None:
            rclpy.spin_once(ros, timeout_sec=0)
else:
    for _ in range(30):
        simulation_app.update()

try:
    ros = ros_node_holder.get("ros")
    if ros is not None:
        ros.destroy_node()
    rclpy.shutdown()
except Exception:
    pass
simulation_app.close()
print("  [plant] 종료 완료")
