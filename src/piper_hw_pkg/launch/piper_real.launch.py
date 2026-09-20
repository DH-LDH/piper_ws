from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # 손목캠 TCP 오프셋은 xacro가 아니라 piper_eih_camera_node의 cam_tcp_offset_* 파라미터가 담당 —
    # 여기선 팔 자체의 링크 체인(link1..link6, robot_state_publisher가 joint_states로 갱신)만 필요.
    xacro_path = get_package_share_directory("piper_description") + "/urdf/piper_description.xacro"
    robot_description = ParameterValue(Command(["xacro ", xacro_path]), value_type=str)

    really_enable = LaunchConfiguration("really_enable")
    can_name = LaunchConfiguration("can_name")
    move_spd_rate_ctrl = LaunchConfiguration("move_spd_rate_ctrl")
    eih_pick_marker_id = LaunchConfiguration("eih_pick_marker_id")
    eih_pick_marker_size_m = LaunchConfiguration("eih_pick_marker_size_m")
    obj_expected_x_body = LaunchConfiguration("obj_expected_x_body")
    obj_expected_y_body = LaunchConfiguration("obj_expected_y_body")
    obj_expected_z_body = LaunchConfiguration("obj_expected_z_body")
    step_confirm = LaunchConfiguration("step_confirm")
    corner_refine = LaunchConfiguration("corner_refine")
    debug_view = LaunchConfiguration("debug_view")
    arm = LaunchConfiguration("arm")
    debug_view_topic = LaunchConfiguration("debug_view_topic")

    return LaunchDescription([
        DeclareLaunchArgument("really_enable", default_value="false",
                              description="true여야 실제로 EnableArm/모션 명령을 보낸다"),
        DeclareLaunchArgument("can_name", default_value="can_piper"),  # udev로 고정한 이름 — can0/can1은 부팅마다 순서 바뀜
        DeclareLaunchArgument("move_spd_rate_ctrl", default_value="5"),  # [%]
        DeclareLaunchArgument("eih_pick_marker_id", default_value="0"),  # 실물 픽 타겟 마커 ID
        DeclareLaunchArgument("eih_pick_marker_size_m", default_value="0.034"),
        # 선반 place 마커 — 픽과 같은 34mm. 크기를 틀리면 solvePnP 거리가 그 비율로
        # 통째로 틀어진다(예전엔 sim 기본 20mm로 풀려서 0.59배로 가깝게 나왔음).
        DeclareLaunchArgument("eih_place_marker_id", default_value="3"),
        DeclareLaunchArgument("eih_place_marker_size_m", default_value="0.034"),
        # 손목캠 재투영 게이트 — 근접 관측이라 px 절대값보다 "마커 크기 대비 비율"이 기준.
        # 기본값은 거의 안 거르는 관측용 값이다(detect엔 타임아웃이 없어 잘못 조이면
        # 픽이 멈춘다). [진단-eih]의 rel(%)을 한 번 보고 그 2배쯤으로 내릴 것.
        DeclareLaunchArgument("eih_reproj_max_px", default_value="30.0"),
        DeclareLaunchArgument("eih_reproj_max_rel", default_value="0.20"),
        # 손목캠 외부파라미터(link6→eih_cam) — eih_cam_calib.py가 뽑아주는 값.
        DeclareLaunchArgument("cam_tcp_offset_x", default_value="-0.085"),
        DeclareLaunchArgument("cam_tcp_offset_y", default_value="-0.010"),
        DeclareLaunchArgument("cam_tcp_offset_z", default_value="0.026"),
        DeclareLaunchArgument("cam_tcp_offset_pitch_deg", default_value="27.5"),
        DeclareLaunchArgument("cam_tcp_offset_roll_deg", default_value="-90.0"),
        # hover가 처음 향할 "물체 대략 위치"(팔 베이스 기준) — 마커를 화각에 넣는 것이
        # 목적이라 정밀할 필요는 없다. z는 (물체 높이 − 팔 베이스 높이)로 잡는다.
        DeclareLaunchArgument("obj_expected_x_body", default_value="0.0"),
        DeclareLaunchArgument("obj_expected_y_body", default_value="0.40"),
        DeclareLaunchArgument("obj_expected_z_body", default_value="-0.13"),
        # "EndPoseCtrl 명령 기준점 → 손끝" 거리. 물리적 손가락 길이가 아니라 펌웨어 EndPose
        # 기준점과 URDF link6 원점의 불일치까지 합친 값이라, joint7 장착점(135.8mm)보다
        # 짧은 게 정상이다. 물체 높이를 바꿔가며 재도 같은 값이 나오는 상수다.
        DeclareLaunchArgument("ee_grip_offset", default_value="0.121"),
        # 파지 깊이/접근 기하 — 물체가 안 잡히면 여기부터 만진다(arm_node.py 상수와 같은 기본값).
        DeclareLaunchArgument("approach_dist", default_value="0.05"),        # [m] pre 대기 높이
        # [m] 마커면(물체 윗면)보다 더 내려가는 깊이 = 물체 높이/2가 기준. 오차 보정용이
        # 아니라 "얼마나 깊이 물지" 설계값이므로 물체가 바뀌면 같이 바꿀 것.
        DeclareLaunchArgument("grasp_depth_extra", default_value="0.008"),   # 현재 타겟 16.36mm 큐브
        # 근접 재검출이 "더 낮다"고 할 때 최초 검출(hover 원거리 관측)보다 아래로 허용하는 한계.
        DeclareLaunchArgument("grasp_z_below_anchor_max", default_value="0.02"),
        # 재검출 수용 한계 — 축별로 나눴다. XY 오검출은 헛집고 재시도하면 그만이지만
        # z가 아래로 틀리면 손가락이 테이블을 찍으므로 z만 좁게 둔다. 파지 직전
        # (렌즈~마커 120mm)에는 화각 여유가 85mm라 XY는 이 값보다 화각이 먼저 막는다.
        DeclareLaunchArgument("redetect_max_xy", default_value="0.12"),
        DeclareLaunchArgument("grasp_jump_max_xy", default_value="0.15"),
        DeclareLaunchArgument("eih_z_up_max", default_value="0.03"),
        # 도달 판정 허용오차 = 곧 파지 깊이 오차. 기본 4mm는 sim의 5cm 큐브 기준이라
        # 16.36mm 타겟(깊이 8mm)에는 과하다 — z는 따로 1.5mm로 좁게 본다.
        DeclareLaunchArgument("grasp_arrive_tol", default_value="0.004"),
        DeclareLaunchArgument("grasp_arrive_z_tol", default_value="0.0015"),
        # 물리 도달 대기 상한[스텝, 60Hz]. 5% 속도면 100mm 이동에만 4초 이상 걸린다 —
        # 900스텝(15초)으로 넉넉히. 여기서 포기하면 덜 문 채로 그리퍼가 닫힌다.
        DeclareLaunchArgument("arrive_max_wait", default_value="900"),
        # 관절 복귀(SEARCH_Q)는 MOVE J라 도달 확인 수단이 없어 시간으로만 기다린다.
        # 5% 속도의 큰 이동이라 기본값 2.5~3초로는 모자란다 — step_confirm=true일 땐
        # 사람이 Enter 누를 때까지의 시간이 가려줬을 뿐이다. 무인 실행 대비로 10초.
        DeclareLaunchArgument("place_ready_steps", default_value="600"),
        DeclareLaunchArgument("place_home_settle_steps", default_value="600"),
        # 관절 도달 허용오차[deg]. 0 = 로그만(판정은 시간 기준, 동작 변화 없음).
        # 첫 런에서 [place_home 관절] 도달오차 실측을 보고 그 2~3배로 켤 것.
        DeclareLaunchArgument("place_joint_arrive_tol_deg", default_value="0.0"),
        # 플레이스 종료 후 자세[deg, joint1~6]. 기본은 SEARCH_Q(=대기자세).
        # IK를 안 거치는 직접 관절지령이라 도달/간섭 검사가 없다 — 바꿀 땐
        # set_arm_joint.py로 그 자세를 먼저 확인하고 넣을 것.
        DeclareLaunchArgument("place_home_q_deg", default_value="[0.0, 45.0, -90.0, 0.0, 45.0, 0.0]"),
        # place_home(SEARCH_Q) 도달 확인 뒤 마지막으로 갈 자세[deg].
        # read_arm_joint.py로 팔을 손으로 잡아 실측한 값.
        DeclareLaunchArgument("place_done_q_deg",
                              default_value="[0.0, -1.5, -5.0, -8.0, 28.5, 5.0]"),
        # SEARCH_Q에서 관절 80° 이상 이동하는 자세라 5% 속도에선 place_home보다 오래 걸린다.
        DeclareLaunchArgument("place_done_settle_steps", default_value="1200"),
        # 파지 판정 — 차체 카메라가 없으면 chassis 모드는 "미검출=성공"으로 빠진다.
        # 카메라를 달면 both로 바꿔 두 판정을 교차검증할 것.
        DeclareLaunchArgument("verify_mode", default_value="gripper"),
        # 타겟 치수 — 폭은 그리퍼 개구부 판정, 높이는 place 릴리즈 높이 계산에 쓴다.
        DeclareLaunchArgument("obj_width_m", default_value="0.01636"),
        DeclareLaunchArgument("obj_height_m", default_value="0.01636"),
        # 보고되는 개구부는 실제 간격보다 상수만큼 크다(패드 두께 + 영점 미설정).
        # 16.36mm 물체를 정상 파지했을 때 25.9mm로 보고된 데서 역산한 값이다.
        # TODO(재확인): 아무것도 없이 그리퍼를 닫고 보고되는 값으로 다시 맞출 것.
        DeclareLaunchArgument("grip_stroke_offset_mm", default_value="9.5"),
        # place — 마커 중심에서 팔 베이스 쪽으로 한 변(34mm) 당긴 곳에 놓는다(마커를
        # 덮지 않게). 릴리즈는 물체 바닥이 선반면에서 5mm 뜬 높이에서.
        DeclareLaunchArgument("place_inset_m", default_value="0.035"),
        # 당기는 방향: marker_y=마커 자신의 -Y(마커를 비스듬히 붙여도 따라감),
        # body_y=팔 베이스 정면(-Y) 고정, radial=원점→마커 반대방향(종전).
        DeclareLaunchArgument("place_inset_mode", default_value="marker_y"),
        DeclareLaunchArgument("place_release_gap", default_value="0.005"),
        DeclareLaunchArgument("place_freeze_after_detect", default_value="true"),
        # 선반이 높으면 기울여야 IK가 풀리지만, 현재 배치는 픽과 같은 수직 접근이 기본.
        DeclareLaunchArgument("place_pitch_deg", default_value="0.0"),
        # place 마커가 대략 있을 body_link 위치. 선반이 픽 타겟과 같은 선반이라
        # 기본값은 픽 hover 위치(obj_expected_*)를 그대로 물려받는다 — 어차피 여기로
        # 먼저 가서 마커를 찾고 place_detect가 실제 위치로 보정하는 출발점일 뿐이다.
        # 선반을 옮기거나 마커가 화각에 안 들어오면 이 인자만 따로 덮어쓰면 된다.
        DeclareLaunchArgument("place_expected_x_body",
                              default_value=LaunchConfiguration("obj_expected_x_body")),
        DeclareLaunchArgument("place_expected_y_body",
                              default_value=LaunchConfiguration("obj_expected_y_body")),
        DeclareLaunchArgument("place_expected_z_body",
                              default_value=LaunchConfiguration("obj_expected_z_body")),
        # 캘리브레이션 중엔 false로 — detect 시점 파지점을 고정해서 팔 거동을 결정적으로 만든다.
        DeclareLaunchArgument("pre_redetect", default_value="true"),
        DeclareLaunchArgument("grasp_eih_track", default_value="true"),
        DeclareLaunchArgument("step_confirm", default_value="false",
                              description="true면 phase 전환마다 멈추고 step_confirm.py의 Enter 대기"),
        DeclareLaunchArgument("corner_refine", default_value="subpix"),  # "subpix" 또는 "none"
        # 손목캠 검출 확인용 창 — vision_node가 실제로 쓰는 검출/포즈를 그대로 그린다
        # (별도 eih_marker_debug_node는 자체 검출이라 파이프라인 값과 다를 수 있음).
        # false면 arm_node를 안 띄운다 — driver가 phase를 'wait'로 유지해서
        # set_arm_joint.py로 관절을 직접 지령할 수 있다(캘리브레이션용).
        DeclareLaunchArgument("arm", default_value="true"),
        DeclareLaunchArgument("debug_view", default_value="true",
                              description="true면 rqt_image_view를 같이 띄운다"),
        # 원본 프레임만 보고 싶으면 /vision/eih_image(카메라 노드가 그대로 발행하는 것)로 바꿀 것.
        DeclareLaunchArgument("debug_view_topic", default_value="/vision/eih_debug_image"),

        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}]),

        Node(package="piper_hw_pkg", executable="piper_driver_node",
             parameters=[{"really_enable": really_enable, "can_name": can_name,
                          "move_spd_rate_ctrl": move_spd_rate_ctrl,
                          "place_pitch_deg": LaunchConfiguration("place_pitch_deg")}],
             output="screen"),
        Node(package="piper_hw_pkg", executable="piper_gripper_node", output="screen"),
        Node(package="piper_hw_pkg", executable="piper_eih_camera_node",
             parameters=[{"cam_tcp_offset_x": LaunchConfiguration("cam_tcp_offset_x"),
                          "cam_tcp_offset_y": LaunchConfiguration("cam_tcp_offset_y"),
                          "cam_tcp_offset_z": LaunchConfiguration("cam_tcp_offset_z"),
                          "cam_tcp_offset_pitch_deg": LaunchConfiguration("cam_tcp_offset_pitch_deg"),
                          "cam_tcp_offset_roll_deg": LaunchConfiguration("cam_tcp_offset_roll_deg")}],
             output="screen"),
        Node(package="piper_hw_pkg", executable="piper_fake_amr_node", output="screen"),

        Node(package="vision_pkg", executable="vision_node",
             parameters=[{"eih_pick_marker_id": eih_pick_marker_id,
                          "eih_pick_marker_size_m": eih_pick_marker_size_m,
                          "eih_place_marker_id": LaunchConfiguration("eih_place_marker_id"),
                          "eih_place_marker_size_m": LaunchConfiguration("eih_place_marker_size_m"),
                          "eih_reproj_max_px": LaunchConfiguration("eih_reproj_max_px"),
                          "eih_reproj_max_rel": LaunchConfiguration("eih_reproj_max_rel"),
                          "corner_refine": corner_refine,
                          "eih_debug_view": debug_view}],
             output="screen"),
        Node(package="rqt_image_view", executable="rqt_image_view",
             arguments=[debug_view_topic],
             condition=IfCondition(debug_view), output="screen"),
        Node(package="control_pkg", executable="arm_node",
             condition=IfCondition(arm),
             parameters=[{"body_link_world_z": 0.0,
                          "obj_expected_x_body": obj_expected_x_body,
                          "obj_expected_y_body": obj_expected_y_body,
                          "obj_expected_z_body": obj_expected_z_body,
                          "ee_grip_offset": LaunchConfiguration("ee_grip_offset"),
                          "approach_dist": LaunchConfiguration("approach_dist"),
                          "grasp_depth_extra": LaunchConfiguration("grasp_depth_extra"),
                          "grasp_z_below_anchor_max": LaunchConfiguration("grasp_z_below_anchor_max"),
                          "redetect_max_xy": LaunchConfiguration("redetect_max_xy"),
                          "grasp_jump_max_xy": LaunchConfiguration("grasp_jump_max_xy"),
                          "eih_z_up_max": LaunchConfiguration("eih_z_up_max"),
                          "grasp_arrive_tol": LaunchConfiguration("grasp_arrive_tol"),
                          "grasp_arrive_z_tol": LaunchConfiguration("grasp_arrive_z_tol"),
                          "arrive_max_wait": LaunchConfiguration("arrive_max_wait"),
                          "place_ready_steps": LaunchConfiguration("place_ready_steps"),
                          "place_home_settle_steps": LaunchConfiguration("place_home_settle_steps"),
                          "place_joint_arrive_tol_deg": LaunchConfiguration("place_joint_arrive_tol_deg"),
                          "place_home_q_deg": ParameterValue(
                              LaunchConfiguration("place_home_q_deg"), value_type=None),
                          "place_done_q_deg": ParameterValue(
                              LaunchConfiguration("place_done_q_deg"), value_type=None),
                          "place_done_settle_steps": LaunchConfiguration("place_done_settle_steps"),
                          "pre_redetect": LaunchConfiguration("pre_redetect"),
                          "grasp_eih_track": LaunchConfiguration("grasp_eih_track"),
                          "verify_mode": LaunchConfiguration("verify_mode"),
                          "obj_width_m": LaunchConfiguration("obj_width_m"),
                          "obj_height_m": LaunchConfiguration("obj_height_m"),
                          "grip_stroke_offset_mm": LaunchConfiguration("grip_stroke_offset_mm"),
                          "place_inset_m": LaunchConfiguration("place_inset_m"),
                          "place_inset_mode": LaunchConfiguration("place_inset_mode"),
                          "place_release_gap": LaunchConfiguration("place_release_gap"),
                          "place_freeze_after_detect": LaunchConfiguration("place_freeze_after_detect"),
                          "place_pitch_deg": LaunchConfiguration("place_pitch_deg"),
                          "place_expected_x_body": LaunchConfiguration("place_expected_x_body"),
                          "place_expected_y_body": LaunchConfiguration("place_expected_y_body"),
                          "place_expected_z_body": LaunchConfiguration("place_expected_z_body"),
                          "step_confirm": step_confirm}],
             output="screen"),
    ])
