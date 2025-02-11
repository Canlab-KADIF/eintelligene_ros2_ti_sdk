import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import TextSubstitution
from launch.substitutions import LaunchConfiguration
from launch.actions import ExecuteProcess
from launch.actions import LogInfo

def generate_launch_description():

    # default ROSBAG file
    bagfile_default = os.path.join(
        os.environ['WORK_DIR'], 'data', 'ros_bag', 'camera_radar_2024_03_18-18_58_23'
    )

    # ratefactor: arg that can be set from the command line or a default will be used
    ratefactor = DeclareLaunchArgument(
        name="ratefactor",
        default_value=TextSubstitution(text="1.0")
    )

    # bagfile: arg that can be set from the command line or a default will be used
    bagfile = DeclareLaunchArgument(
        name="bagfile",
        default_value=TextSubstitution(text=bagfile_default)
    )

    # rosbag play
    rosbag_process = ExecuteProcess(
        output = "screen",
        cmd=['ros2', 'bag', 'play',
            '-r', LaunchConfiguration('ratefactor'),
            '-l', LaunchConfiguration('bagfile'),
            '--read-ahead-queue-size', '5000',
            '--clock'
        ],
        on_exit=[
            LogInfo(msg="rosbag2 exited")
        ]
    )

    ld = LaunchDescription([
        ratefactor,
        bagfile,
        rosbag_process,
    ])

    return ld
