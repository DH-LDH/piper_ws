from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    xacro_path = get_package_share_directory("piper_description") + "/urdf/piper_description.xacro"
    robot_description = ParameterValue(Command(["xacro ", xacro_path]), value_type=str)

    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}]),
    ])
