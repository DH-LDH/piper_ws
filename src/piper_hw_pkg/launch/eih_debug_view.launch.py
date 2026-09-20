from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package="piper_hw_pkg", executable="piper_eih_camera_node", output="screen"),
        Node(package="piper_hw_pkg", executable="eih_marker_debug_node",
             parameters=[{"marker_size_m": 0.034}], output="screen"),
    ])
