import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import TextSubstitution
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    ld = LaunchDescription()

    # bagfile: arg that can be set from the command line or a default will be used
    maxsize = DeclareLaunchArgument(
        "maxsize", default_value=TextSubstitution(text='0')
    )

    ld.add_action(maxsize)

    # rosbag process
    rosbag_process = ExecuteProcess(
        output = "screen",
        cmd=['ros2', 'bag', 'record',
            '-b', LaunchConfiguration('maxsize'),
            '/camera/left/image_raw',
            '/camera/left/camera_info',
            '/camera/right/image_raw'
            '/camera/right/camera_info']
    )

    ld.add_action(rosbag_process)

    return ld
