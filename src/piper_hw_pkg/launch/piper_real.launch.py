from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    xacro_path = get_package_share_directory("piper_hw_pkg") + "/urdf/eih_cam_mount.xacro"
    robot_description = ParameterValue(Command(["xacro ", xacro_path]), value_type=str)

    really_enable = LaunchConfiguration("really_enable")
    use_marker_place = LaunchConfiguration("use_marker_place")
    can_name = LaunchConfiguration("can_name")

    return LaunchDescription([
        DeclareLaunchArgument("really_enable", default_value="false",
                              description="true여야 실제로 EnableArm/모션 명령을 보낸다"),
        DeclareLaunchArgument("use_marker_place", default_value="true",
                              description="두 번째 아르코 마커로 place 위치 인식(v1 기본)"),
        DeclareLaunchArgument("can_name", default_value="can0"),

        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}]),

        Node(package="piper_hw_pkg", executable="piper_driver_node",
             parameters=[{"really_enable": really_enable, "can_name": can_name}],
             output="screen"),
        Node(package="piper_hw_pkg", executable="piper_gripper_node", output="screen"),
        Node(package="piper_hw_pkg", executable="piper_eih_camera_node", output="screen"),
        Node(package="piper_hw_pkg", executable="piper_fake_amr_node", output="screen"),

        Node(package="vision_pkg", executable="vision_node", output="screen"),
        Node(package="control_pkg", executable="arm_node",
             parameters=[{"use_marker_place": use_marker_place}],
             output="screen"),
    ])
