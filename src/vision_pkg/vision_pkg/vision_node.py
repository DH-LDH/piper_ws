import sys
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import array
import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Float32MultiArray, String
from sensor_msgs.msg import Image, CameraInfo
import tf2_ros

from step22_common import (
    TOP_PTS, TOP_MARKER_ID, TOP_MARKER_SIZE, PLACE_TOP_MARKER_ID,
    _quat_to_R, pack_eih_marker,
)

#
#   손목 카메라(eih) → 픽/플레이스 마커 → /vision/eih_marker_body, /vision/place_marker_body
#   차체 카메라      → 물체 옆면·선반 전면 마커 → /vision/chassis_pose, /vision/place_pose

# ── 파라미터 ─────────────────────────────────────────────────────

# [차체캠] 마커 포즈 채택 기준 
# UP_MIN     = 0.80   # 마커  하한 
# REPROJ_MAX = 3.0    #  재투영 오차 상한
# PLACE_REPROJ_MAX = 15.0   #  place 도킹용
# PLACE_PHI_ALPHA = 0.08    # 도킹 중 안정화 

EIH_PRINT_EVERY = 30   # 손목캠 픽 마커 진단 출력 주기(약 2초)
EIH_REPROJ_MAX_PX  = 30.0   #  절대 상한 
EIH_REPROJ_MAX_REL = 0.20   #  rep / 마커 한 변 픽셀길이 — 스케일 불변 게이트(관측용 느슨값)

# ArUco 코너 정밀화
CORNER_REFINE = "subpix"   # "subpix"(정밀) "none"(빠름)
CORNER_REFINE_WIN  = 5
CORNER_REFINE_ITER = 50
CORNER_REFINE_ACC  = 0.01


_LATCH = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


_HAS_ARUCO_DETECTOR = hasattr(cv2.aruco, "ArucoDetector")


def _make_detector(corner_refine=CORNER_REFINE):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if not _HAS_ARUCO_DETECTOR:
        return d, cv2.aruco.DetectorParameters_create()  # OpenCV<4.7 구API
    pr = cv2.aruco.DetectorParameters()
    mode = {"subpix": getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", 1),
            "none": getattr(cv2.aruco, "CORNER_REFINE_NONE", 0)}
    try:
        pr.cornerRefinementMethod = mode.get(corner_refine, mode["none"])
        pr.cornerRefinementWinSize = CORNER_REFINE_WIN
        pr.cornerRefinementMaxIterations = CORNER_REFINE_ITER
        pr.cornerRefinementMinAccuracy = CORNER_REFINE_ACC
    except Exception:
        pass
    return cv2.aruco.ArucoDetector(d, pr)


def _detect_markers(gray, det):
    if _HAS_ARUCO_DETECTOR:
        return det.detectMarkers(gray)
    d, pr = det
    return cv2.aruco.detectMarkers(gray, d, parameters=pr)


def _img_to_gray(msg: Image):
    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
    rgb = arr[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _tf_to_Rt(tf):
    q = tf.transform.rotation
    t = tf.transform.translation
    R = _quat_to_R((q.w, q.x, q.y, q.z))
    return R, np.array([t.x, t.y, t.z], float)


class VisionNode(Node):
    def __init__(self):
        super().__init__("vision_node")
        self.det = _make_detector(self.declare_parameter("corner_refine", CORNER_REFINE).value)
        self.dist = np.zeros(5)
        self.eih_K = None
        # [차체캠] 되살릴 때 아래 주석을 풀 것
        # self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # self.faces = compute_marker_faces()
        # _n_front = np.array([0., -1., 0.])
        # _basis = _plate_basis(_n_front)
        # self.place_face = (_basis, MK_SHELF_FRONT_OFFSET*_n_front
        #                    + np.array([SHELF_FAR_MARKER_X, 0., SHELF_FAR_MARKER_Z]))
        # self.near_face = (_basis, MK_SHELF_FRONT_OFFSET*_n_front
        #                   + np.array([SHELF_NEAR_MARKER_X, 0., SHELF_NEAR_MARKER_Z]))
        # self.K = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.declare_parameter("eih_pick_marker_id", TOP_MARKER_ID)
        self.declare_parameter("eih_pick_marker_size_m", TOP_MARKER_SIZE)
        self.pick_marker_id = int(self.get_parameter("eih_pick_marker_id").value)
        self.pick_marker_size = float(self.get_parameter("eih_pick_marker_size_m").value)
        _h = self.pick_marker_size / 2.0
        self.pick_top_pts = np.array([[-_h, _h, 0.], [_h, _h, 0.],
                                       [_h, -_h, 0.], [-_h, -_h, 0.]], np.float32)
        # place 지그 상판 마커 — 실물은 픽 마커와 같은 34mm를 쓰는데, 예전엔 TOP_PTS(20mm)로
        # 풀어서 거리가 20/34배로 축소돼 나왔다(놓는점이 통째로 틀어짐).
        self.place_marker_id = int(
            self.declare_parameter("eih_place_marker_id", PLACE_TOP_MARKER_ID).value)
        self.place_marker_size = float(
            self.declare_parameter("eih_place_marker_size_m", TOP_MARKER_SIZE).value)
        _hp = self.place_marker_size / 2.0
        self.place_top_pts = np.array([[-_hp, _hp, 0.], [_hp, _hp, 0.],
                                        [_hp, -_hp, 0.], [-_hp, -_hp, 0.]], np.float32)
        self.eih_reproj_max_px = float(
            self.declare_parameter("eih_reproj_max_px", EIH_REPROJ_MAX_PX).value)
        self.eih_reproj_max_rel = float(
            self.declare_parameter("eih_reproj_max_rel", EIH_REPROJ_MAX_REL).value)

        # 실제 파이프라인이 쓰는 검출/포즈를 그대로 그려서 발행 — 별도 디버그 노드와 달리
        # 여기서 보이는 값이 곧 /vision/eih_marker_body로 나가는 값이다.
        self.eih_debug_view = bool(self.declare_parameter("eih_debug_view", False).value)
        self.pub_eih_debug = (self.create_publisher(Image, "/vision/eih_debug_image", 5)
                               if self.eih_debug_view else None)

        self.eih_hit = 0   # 손목캠 픽 마커 검출 수 — 주기적 진단 출력 간격 계산용
        self.eih_rej_rep = 0     # 재투영 게이트에서 기각된 픽 마커 프레임 수
        self.eih_rep_max = 0.0   # 통과한 프레임의 최대 rep(px) — 게이트 재조정 근거
        # [차체캠] 되살릴 때 아래 주석을 풀 것
        # self.cam_hit = 0; self.cam_miss = 0
        # self.rej_up = 0; self.rej_rep = 0
        # self._place_rej = 0
        # self._place_phi_ema = None
        # self.first_hit_printed = False

        self.pub_eih = self.create_publisher(Float32MultiArray, "/vision/eih_marker_body", 10)
        self.pub_place_marker = self.create_publisher(Float32MultiArray, "/vision/place_marker_body", 10)
        self.pub_status = self.create_publisher(String, "/vision/status", 5)
        # [차체캠]
        # self.pub_chassis = self.create_publisher(Float32MultiArray, "/vision/chassis_pose", 10)
        # self.pub_place = self.create_publisher(Float32MultiArray, "/vision/place_pose", 10)

        self.create_subscription(CameraInfo, "/vision/eih_camera_info", self._on_eih_info, _LATCH)
        self.create_subscription(Image, "/vision/eih_image", self._on_eih_image, 5)
        # [차체캠]
        # self.create_subscription(CameraInfo, "/vision/chassis_camera_info", self._on_chassis_info, _LATCH)
        # self.create_subscription(Image, "/vision/chassis_image", self._on_chassis_image, 5)

        self.create_timer(2.0, self._publish_status)
        self.get_logger().info("vision_node 초기화 완료 — camera_info 대기 중")


    def _on_eih_info(self, msg: CameraInfo):
        self.eih_K = np.array(msg.k, float).reshape(3, 3)
        self.get_logger().info(f"끝단 카메라 내부파라미터 수신 K={self.eih_K[0,0]:.1f}")

    def _publish_status(self):
        self.pub_status.publish(String(
            data=f"eih_hit={self.eih_hit} rej_rep={self.eih_rej_rep} "
                 f"rep_max={self.eih_rep_max:.2f}px"))

    def _solve_top_facing(self, corners, pts=TOP_PTS):
        """마커 코너 → 위를 보는 면 기준 카메라좌표 위치. 반환: (t, rvec, rep_err) 또는 None
        (rvec/rep는 디버그 뷰 전용 — 포즈 계산 자체는 t만 쓴다)."""
        try:
            ok, rvecs, tvecs, rep = cv2.solvePnPGeneric(
                pts, corners, self.eih_K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except Exception:
            ok, rvecs, tvecs, rep = 0, [], [], []
        if not ok or len(rvecs) == 0:
            return None
        reps = np.asarray(rep, float).ravel() if len(rep) else np.zeros(len(rvecs))
        best, best_rv, best_rep, best_tz = None, None, 0.0, -np.inf
        for kk, (rv, tv) in enumerate(zip(rvecs, tvecs)):
            R, _ = cv2.Rodrigues(rv)
            if R[2, 2] > best_tz:
                best_tz = R[2, 2]; best = np.asarray(tv, float).ravel()
                best_rv = rv; best_rep = float(reps[kk]) if kk < len(reps) else 0.0
        return best, best_rv, best_rep

    def _marker_body_yaw(self, rvec, R_bc):
        """마커 자신의 +Y축을 body XY 평면에 투영한 방향[rad].
        eih_cam이 OpenCV 관례(x-right, y-down, z-fwd)로 정의돼 있어 solvePnP 결과를
        그대로 body로 옮기면 된다 — R_body = R_bc @ R_cam."""
        R_cm, _ = cv2.Rodrigues(rvec)
        R_bm = R_bc @ R_cm
        ay = R_bm[:, 1]
        return math.atan2(float(ay[1]), float(ay[0]))

    def _eih_reproj_ok(self, corners, rep, tag):
        """손목캠 재투영 게이트 — (통과여부, 마커 한 변 픽셀길이, 상대오차).
        px 절대값과 마커 크기 대비 비율을 둘 다 봐야 원거리/근접이 같은 기준이 된다."""
        c = np.asarray(corners, float).reshape(4, 2)
        side_px = float(np.mean([np.linalg.norm(c[(i+1) % 4] - c[i]) for i in range(4)]))
        rel = rep / side_px if side_px > 1e-6 else 9.99
        ok = (rep <= self.eih_reproj_max_px) and (rel <= self.eih_reproj_max_rel)
        if not ok:
            self.eih_rej_rep += 1
            if self.eih_rej_rep % 30 == 1:
                print(f"    [eih 게이트 기각] {tag} rep={rep:.2f}px "
                      f"(상한 {self.eih_reproj_max_px:.1f}px) "
                      f"rel={rel*100:.1f}% (상한 {self.eih_reproj_max_rel*100:.1f}%) "
                      f"마커변={side_px:.0f}px — {self.eih_rej_rep}회째")
        return ok, side_px, rel

    # ── 손목캠 디버그 뷰(eih_debug_view=true일 때만) ────────────────────────
    def _dbg_frame(self, msg: Image):
        """디버그 뷰가 꺼져 있으면 None — 켜져 있을 때만 프레임을 BGR로 복사한다."""
        if self.pub_eih_debug is None:
            return None
        rgba = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    def _dbg_publish(self, dbg, msg: Image, lines=(), color=(0, 255, 0)):
        if dbg is None:
            return
        for i, line in enumerate(lines):
            cv2.putText(dbg, line, (8, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
        out = Image()
        out.header = msg.header
        out.height, out.width = dbg.shape[0], dbg.shape[1]
        out.encoding = "bgr8"
        out.step = out.width * 3
        out.data = array.array("B", dbg.tobytes())
        self.pub_eih_debug.publish(out)

    # ── 끝단 카메라: 윗면 마커(ID0) → 몸체좌표 위치 ─────────────────────────
    def _on_eih_image(self, msg: Image):
        if self.eih_K is None: return
        dbg = self._dbg_frame(msg)
        try:
            R_bc, t_bc = _tf_to_Rt(self.tf_buffer.lookup_transform(
                "body_link", "eih_cam", Time()))
        except Exception:
            self._dbg_publish(dbg, msg, ["NO TF body_link->eih_cam"], (0, 0, 255))
            return
        gray = _img_to_gray(msg)

        cs, ids, _ = _detect_markers(gray, self.det)
        id_list = ids.flatten().tolist() if ids is not None else []
        if dbg is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(dbg, cs, ids)

        # place 지그 상판 마커 — 별도 토픽으로 항상 같이 처리
        _sol, _pc = None, None
        if self.place_marker_id in id_list:
            _pc = cs[id_list.index(self.place_marker_id)].reshape(4, 2)
            _sol = self._solve_top_facing(_pc, self.place_top_pts)
        if _sol is not None and not self._eih_reproj_ok(
                _pc, _sol[2], f"place ID{self.place_marker_id}")[0]:
            _sol = None
        if _sol is not None:
            _p_body = R_bc @ _sol[0] + t_bc
            _yaw = self._marker_body_yaw(_sol[1], R_bc)
            self.pub_place_marker.publish(Float32MultiArray(
                data=pack_eih_marker(_p_body[0], _p_body[1], _p_body[2], True, _yaw)))
        else:
            self.pub_place_marker.publish(Float32MultiArray(data=pack_eih_marker(0, 0, 0, False)))

        hdr = f"pick ID{self.pick_marker_id} size={self.pick_marker_size*1000:.0f}mm"
        if self.pick_marker_id not in id_list:
            self.pub_eih.publish(Float32MultiArray(data=pack_eih_marker(0, 0, 0, False)))
            self._dbg_publish(dbg, msg, [hdr, f"NOT DETECTED (seen={id_list})"], (0, 0, 255))
            return
        idx = id_list.index(self.pick_marker_id)
        corners = cs[idx].reshape(4, 2)

        sol = self._solve_top_facing(corners, self.pick_top_pts)
        if sol is None:
            self.pub_eih.publish(Float32MultiArray(data=pack_eih_marker(0, 0, 0, False)))
            self._dbg_publish(dbg, msg, [hdr, "solvePnP FAILED"], (0, 0, 255))
            return
        best, rvec, rep = sol

        gate_ok, side_px, rel = self._eih_reproj_ok(corners, rep, f"pick ID{self.pick_marker_id}")
        if not gate_ok:
            self.pub_eih.publish(Float32MultiArray(data=pack_eih_marker(0, 0, 0, False)))
            self._dbg_publish(dbg, msg, [
                hdr, f"REJECT rep={rep:.2f}px rel={rel*100:.1f}% side={side_px:.0f}px"], (0, 0, 255))
            return
        self.eih_rep_max = max(self.eih_rep_max, rep)

        p_body = R_bc @ best + t_bc
        self.pub_eih.publish(Float32MultiArray(
            data=pack_eih_marker(p_body[0], p_body[1], p_body[2], True,
                                  self._marker_body_yaw(rvec, R_bc))))

        # 2초마다 진단 — 캘리브레이션 때 화면을 안 봐도 런치 로그에 그대로 남는다.
        self.eih_hit += 1
        if self.eih_hit % EIH_PRINT_EVERY == 1:
            print(f"    [진단-eih] ID{self.pick_marker_id} size={self.pick_marker_size*1000:.1f}mm  "
                  f"cam=({best[0]*1000:+.0f},{best[1]*1000:+.0f},{best[2]*1000:+.0f})mm "
                  f"d={float(np.linalg.norm(best))*1000:.1f}mm rep={rep:.2f}px"
                  f"(rel {rel*100:.1f}%, 기각누적 {self.eih_rej_rep})  "
                  f"body=({p_body[0]*1000:+.1f},{p_body[1]*1000:+.1f},{p_body[2]*1000:+.1f})mm")

        if dbg is not None:
            cv2.drawFrameAxes(dbg, self.eih_K, self.dist, rvec,
                              best.reshape(3, 1), self.pick_marker_size)
            self._dbg_publish(dbg, msg, [
                hdr,
                f"cam=({best[0]:+.3f},{best[1]:+.3f},{best[2]:+.3f})m "
                f"d={float(np.linalg.norm(best))*100:.1f}cm rep={rep:.2f}px",
                f"body=({p_body[0]:+.3f},{p_body[1]:+.3f},{p_body[2]:+.3f})m"])


def main():
    rclpy.init()
    node = VisionNode()
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



# =============================================================================
# [보관] 차체 카메라 경로 — 현재 미장착이라 통째로 비활성
#
# 차체(고정) 카메라로 물체 옆면 마커(ID0~3)와 선반 전면 마커(ID5/ID7)를 보고
# /vision/chassis_pose, /vision/place_pose를 내던 경로. 지금 구성에는 차체 카메라가
# 없어 한 줄도 실행되지 않으므로, 손목캠 코드를 읽는 데 방해되지 않도록 파일 끝으로
# 모아뒀다.
#
# 되살리는 법
#   1. 아래 네 블록의 주석을 푼다 (import / _Rz_rad / _on_chassis_info / 본체)
#   2. 블록들을 원래 자리로 되돌린다 — 메서드는 VisionNode 클래스 안이어야 한다
#   3. 파일 상단 [차체캠] 표시 상수(UP_MIN, REPROJ_MAX 등)와
#      __init__의 [차체캠] 표시 블록(상태변수/발행자/구독자)도 함께 푼다
#   4. _publish_status를 차체캠 통계판으로 되돌릴지 정한다(현재는 손목캠판)
#
# 그러면 arm_node에서 verify_mode:=both로 손목캠 기반 파지 판정과 교차검증할 수 있다.
# =============================================================================

# [차체캠] 되살릴 때 아래도 같이 import할 것
# from step22_common import (
#     OBJ_PTS, PLACE_SIDE_MARKER_ID, MK_SHELF_FRONT_OFFSET, SHELF_FRONT_PTS,
#     SHELF_FAR_MARKER_Z, SHELF_NEAR_MARKER_Z, PLACE_NEAR_MARKER_ID, PLACE_NEAR_PTS,
#     SHELF_FAR_MARKER_X, SHELF_NEAR_MARKER_X,
#     _wrap, _plate_basis, compute_marker_faces, pack_chassis_pose,
# )

# [차체캠]
# def _Rz_rad(phi):
#     c, s = math.cos(phi), math.sin(phi)
#     return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], float)

    # [차체캠]
    # def _on_chassis_info(self, msg: CameraInfo):
    #     self.K = np.array(msg.k, float).reshape(3, 3)
    #     self.get_logger().info(f"차체 카메라 내부파라미터 수신 K={self.K[0,0]:.1f}")

    # [차체캠] 본체 — _publish_status(차체판) / _solve_face_center / _on_chassis_image
    # def _publish_status(self):   ← 차체캠 통계판. 아래 손목캠판으로 대체했다.
    #     self.pub_status.publish(String(
    #         data=f"hit={self.cam_hit} miss={self.cam_miss} "
    #              f"rej_up={self.rej_up} rej_rep={self.rej_rep}"))
    #
    # def _solve_face_center(self, corners, R_bc, t_bc, R_cp_f, off_f, obj_pts=OBJ_PTS,
    #                         return_raw=False):
    #     """마커 코너 → 그 마커가 붙은 물체(또는 선반) 중심의 body_link 위치/헤딩.
    #     obj_pts는 그 마커의 실제 크기로 만든 코너 모델(크기 다른 마커는 반드시 제 것을 넘길 것).
    #     return_raw=True면 (R_b_obj, p_body)도 반환(호출부가 phi 안정화 후 오프셋을 재계산할 때 씀).
    #     반환: (bx,by,bz,phi,up_z,rep_err[,R_b_obj,p_body]) 또는 None(solvePnP 실패)."""
    #     Sm = np.diag([1.0, -1.0, -1.0])
    #     try:
    #         ok, rvecs, tvecs, rep = cv2.solvePnPGeneric(
    #             obj_pts, corners, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    #     except Exception:
    #         ok, rvecs, tvecs, rep = 0, [], [], []
    #     if not ok or len(rvecs) == 0:
    #         return None
    #     reps = np.asarray(rep, float).ravel() if len(rep) else np.zeros(len(rvecs))
    #     best = None
    #     for kk in range(len(rvecs)):
    #         R_bm = R_bc @ (Sm @ cv2.Rodrigues(rvecs[kk])[0])
    #         up = float(R_bm[2, 1])
    #         r = float(reps[kk]) if kk < len(reps) else 0.0
    #         if best is None or up > best[0]: best = (up, kk, R_bm, r)
    #     up_z, kk, R_b_mk, rep_err = best
    #     p_cam = Sm @ np.asarray(tvecs[kk], float).ravel()
    #     p_body = R_bc @ p_cam + t_bc
    #     R_b_obj = R_b_mk @ R_cp_f.T
    #     p_obj = p_body - R_b_obj @ off_f
    #     bx, by, bz = float(p_obj[0]), float(p_obj[1]), float(p_obj[2])
    #     phi = _wrap(math.atan2(R_b_obj[1, 0], R_b_obj[0, 0]))
    #     if return_raw:
    #         return bx, by, bz, phi, up_z, rep_err, R_b_obj, p_body
    #     return bx, by, bz, phi, up_z, rep_err
    #
    # # ── 차체 카메라: 옆면 마커(ID 0~3) → (bx,by,phi) ────────────────────────
    # # [현재 미사용 — 차체 카메라 미장착] camera_info가 안 들어와 아래 콜백은 첫 줄에서
    # # 반환된다. 카메라를 달면 그대로 동작하며, arm_node의 verify_mode=both로 손목캠
    # # 기반 파지 판정과 교차검증할 수 있다.
    # def _on_chassis_image(self, msg: Image):
    #     if self.K is None: return
    #     # frame_id에 "chassis_cam:<스텝번호>"가 실려 오면 그대로 페이로드에 옮긴다
    #     sim_step = 0
    #     if ":" in msg.header.frame_id:
    #         try: sim_step = int(msg.header.frame_id.split(":")[1])
    #         except Exception: sim_step = 0
    #     try:
    #         R_bc, t_bc = _tf_to_Rt(self.tf_buffer.lookup_transform(
    #             "body_link", "chassis_cam", Time()))
    #     except Exception:
    #         return
    #     gray = _img_to_gray(msg)
    #
    #     corners, use_id, place_corners, near_corners = None, None, None, None
    #     for g in (gray, self.clahe.apply(gray)):
    #         cs, ids, _ = _detect_markers(g, self.det)
    #         if ids is not None and len(ids) > 0:
    #             best_a = 0.0
    #             for kk, i in enumerate(ids.ravel()):
    #                 i = int(i)
    #                 c = cs[kk].reshape(-1, 2)
    #                 if i == PLACE_SIDE_MARKER_ID and place_corners is None:
    #                     place_corners = c
    #                 if i == PLACE_NEAR_MARKER_ID and near_corners is None:
    #                     near_corners = c
    #                 if i not in self.faces: continue  # 윗면(ID4)/place 마커 제외
    #                 a = float(cv2.contourArea(c.astype(np.float32)))
    #                 if a > best_a: best_a, corners, use_id = a, c, i
    #         if corners is not None and (place_corners is not None
    #                                     or near_corners is not None): break
    #
    #     res_p = self._solve_face_center(place_corners, R_bc, t_bc, *self.place_face,
    #                                      obj_pts=SHELF_FRONT_PTS, return_raw=True) \
    #         if place_corners is not None else None
    #     res_n = self._solve_face_center(near_corners, R_bc, t_bc, *self.near_face,
    #                                      obj_pts=PLACE_NEAR_PTS, return_raw=True) \
    #         if near_corners is not None else None
    #
    #     def _gate_ok(r):
    #         return r is not None and r[4] >= UP_MIN and r[5] <= PLACE_REPROJ_MAX
    #
    #     # 근접 마커(ID7)가 잡히면 그쪽 우선 — 작아서 원거리에선 애초에 검출이 안 되고,
    #     # 검출됐다는 건 이미 충분히 가까워 화면에서 더 크고 정확하다는 뜻이다.
    #     if _gate_ok(res_n):
    #         use_res, use_mid, use_off = res_n, PLACE_NEAR_MARKER_ID, self.near_face[1]
    #     elif _gate_ok(res_p):
    #         use_res, use_mid, use_off = res_p, PLACE_SIDE_MARKER_ID, self.place_face[1]
    #     else:
    #         use_res, use_mid, use_off = None, -1, None
    #
    #     if use_res is not None:
    #         pbx, pby, pbz, pphi, _, _, R_b_obj, p_body = use_res
    #         # phi(선반 방향)를 EMA로 안정화한 뒤 그 값으로 오프셋 회전을 다시 계산한다 —
    #         # 원 phi는 근접 정면 각도에서 평면마커 PnP 자세 모호성으로 프레임마다 크게 튄다.
    #         if self._place_phi_ema is None:
    #             self._place_phi_ema = pphi
    #         else:
    #             self._place_phi_ema = _wrap(
    #                 self._place_phi_ema + PLACE_PHI_ALPHA * _wrap(pphi - self._place_phi_ema))
    #         R_stable = _Rz_rad(self._place_phi_ema)
    #         p_obj_stable = p_body - R_stable @ use_off
    #         pbx, pby, pbz = (float(p_obj_stable[0]), float(p_obj_stable[1]),
    #                           float(p_obj_stable[2]))
    #         self.pub_place.publish(Float32MultiArray(
    #             data=pack_chassis_pose(pbx, pby, self._place_phi_ema, True, use_mid,
    #                                     sim_step, pbz)))
    #     else:
    #         # 진단용 — ID5/ID7이 아예 안 잡힌 건지, 잡혔는데 게이트에서 기각된 건지 구분
    #         if place_corners is not None or near_corners is not None:
    #             self._place_rej += 1
    #             if self._place_rej % 30 == 1:
    #                 for tag, cor, r in (("ID5(원거리)", place_corners, res_p),
    #                                     ("ID7(근접)", near_corners, res_n)):
    #                     if cor is None:
    #                         continue
    #                     if r is None:
    #                         print(f"    [place진단-vision] {tag} 코너는 잡힘 — solvePnP 실패")
    #                     else:
    #                         print(f"    [place진단-vision] {tag} 게이트 기각 "
    #                               f"up_z={r[4]:.3f}(≥{UP_MIN}) rep={r[5]:.2f}px(≤{PLACE_REPROJ_MAX})")
    #         self.pub_place.publish(Float32MultiArray(data=pack_chassis_pose(0, 0, 0, False, -1, sim_step)))
    #
    #     if corners is None:
    #         self.cam_miss += 1
    #         self.pub_chassis.publish(Float32MultiArray(data=pack_chassis_pose(0, 0, 0, False, -1, sim_step)))
    #         return
    #
    #     res = self._solve_face_center(corners, R_bc, t_bc, *self.faces[use_id])
    #     if res is None:
    #         self.cam_miss += 1
    #         self.pub_chassis.publish(Float32MultiArray(data=pack_chassis_pose(0, 0, 0, False, -1, sim_step)))
    #         return
    #     bx, by, bz, phi, up_z, rep_err = res
    #     if up_z < UP_MIN:
    #         self.rej_up += 1; self.cam_miss += 1
    #         self.pub_chassis.publish(Float32MultiArray(data=pack_chassis_pose(0, 0, 0, False, -1, sim_step)))
    #         return
    #     if rep_err > REPROJ_MAX:
    #         self.rej_rep += 1; self.cam_miss += 1
    #         self.pub_chassis.publish(Float32MultiArray(data=pack_chassis_pose(0, 0, 0, False, -1, sim_step)))
    #         return
    #
    #     self.cam_hit += 1
    #     if not self.first_hit_printed:
    #         self.first_hit_printed = True
    #         print(f"  [vision] 첫 검출 ID{use_id}  bx={bx*1000:.1f}mm by={by*1000:.1f}mm "
    #               f"bz={bz*1000:.1f}mm")
    #     if self.cam_hit % 60 == 0:
    #         print(f"    [진단-vision] ID{use_id} bx={bx*1000:+.0f}mm by={by*1000:+.0f}mm "
    #               f"bz={bz*1000:+.0f}mm up_z={up_z:.3f} rep={rep_err:.3f}")
    #     self.pub_chassis.publish(Float32MultiArray(
    #         data=pack_chassis_pose(bx, by, phi, True, use_id, sim_step, bz)))
    #
