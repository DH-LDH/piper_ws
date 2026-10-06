import time

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # B단계 궤적 시험: 드라이버 + robot_state_publisher + traj_test_node만. 비전·arm_node·그리퍼는 띄우지 않는다
    xacro_path = get_package_share_directory("piper_description") + "/urdf/piper_description.xacro"
    robot_description = ParameterValue(Command(["xacro ", xacro_path]), value_type=str)
    L = LaunchConfiguration
    prof = {"profile": L("profile"),
            "profile_run": [time.strftime("%Y%m%d_%H%M%S_"), "traj_", L("set"), "_", L("mode")]}

    traj = Node(package="control_pkg", executable="traj_test_node", output="screen",
                parameters=[{"set": L("set"), "mode": L("mode"), "rate_div": L("rate_div"),
                             "v_max": L("v_max"), "w_max": L("w_max"), "rot_w_max": L("rot_w_max"),
                             "movel_speed": L("movel_speed"), "max_step_deg": L("max_step_deg"),
                             "repeat": L("repeat"), "dry_run": L("dry_run"), "confirm": L("confirm")}])

    return LaunchDescription([
        DeclareLaunchArgument("really_enable", default_value="false"),
        DeclareLaunchArgument("can_name", default_value="can_piper"),
        DeclareLaunchArgument("move_spd_rate_ctrl", default_value="5"),
        DeclareLaunchArgument("profile", default_value="false"),
        DeclareLaunchArgument("set", default_value="line", description="line / arc / s_curve / axis_y"),
        DeclareLaunchArgument("mode", default_value="joint_direct",
                              description="joint_direct / joint_ik / movel / movej_once / fk_check"),
        DeclareLaunchArgument("rate_div", default_value="1", description="1=60Hz, 3=20Hz 송신"),
        DeclareLaunchArgument("v_max", default_value="0.02"),
        DeclareLaunchArgument("w_max", default_value="0.15"),
        DeclareLaunchArgument("rot_w_max", default_value="0.3"),
        DeclareLaunchArgument("movel_speed", default_value="0.02"),
        DeclareLaunchArgument("max_step_deg", default_value="0.6"),
        DeclareLaunchArgument("repeat", default_value="1"),
        DeclareLaunchArgument("dry_run", default_value="true", description="false여야 실제 명령을 보낸다"),
        DeclareLaunchArgument("confirm", default_value="true",
                              description="true면 시작점 도착 후 step_confirm.py Enter를 기다렸다가 실행"),

        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}]),
        Node(package="piper_hw_pkg", executable="piper_driver_node", output="screen",
             parameters=[{"really_enable": L("really_enable"), "can_name": L("can_name"),
                          "move_spd_rate_ctrl": L("move_spd_rate_ctrl"),
                          "initial_phase": "idle", **prof}]),  # 하네스 지시 전엔 명령 없음(SEARCH_Q 섞임 방지)
        traj,
        RegisterEventHandler(OnProcessExit(target_action=traj, on_exit=[EmitEvent(event=Shutdown())])),  # 완료 시 전체 종료
    ])
