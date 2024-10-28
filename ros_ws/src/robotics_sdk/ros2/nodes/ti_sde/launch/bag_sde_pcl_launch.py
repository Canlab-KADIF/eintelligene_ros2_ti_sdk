import os
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import TextSubstitution
from launch.substitutions import LaunchConfiguration

def get_launch_file(pkg, file_name):
    pkg_dir = get_package_share_directory(pkg)
    return os.path.join(pkg_dir, 'launch', file_name)

def generate_launch_description():
    # Flag for exporting the performance data to a file: 0 - disable, 1 - enable
    exportPerfStats_arg = DeclareLaunchArgument(
        "exportPerfStats", default_value=TextSubstitution(text="0")
    )

    # Include SDE launch file
    sde_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(get_launch_file('ti_sde', 'sde_launch.py')),
        launch_arguments={
            "sde_algo_type": "1",
            "enable_pc": "1",
            "exportPerfStats": LaunchConfiguration('exportPerfStats'),
        }.items()
    )

    # Include rosbag launch file
    bag_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(get_launch_file('ti_sde', 'rosbag_launch.py'))
    )

    return LaunchDescription([
        exportPerfStats_arg,
        sde_launch,
        bag_launch,
    ])