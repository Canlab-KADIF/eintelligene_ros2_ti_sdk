import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription()

    viz_dir = get_package_share_directory('ti_viz_nodes')
    rviz_dir = os.path.join(viz_dir, 'rviz')

    width_arg = DeclareLaunchArgument(
        name='width',
        default_value='1280',
        description='Width of the input image'
    )

    height_arg = DeclareLaunchArgument(
        name='height',
        default_value='720',
        description='Height of the input image'
    )

    # color conversion for input image_raw for visualization
    params = [{
        "width":            LaunchConfiguration('width'),
        "height":           LaunchConfiguration('height'),
        "input_yuv_topic":  "camera/image_raw",
        "output_rgb_topic": "camera/image_raw_rgb",
        "yuv_format":       "YUV420"
    }]
    yuv2rbg_node1 = Node(
        package = "ti_viz_nodes",
        executable = "viz_color_conv_yuv2rgb",
        name = "viz_color_conv_yuv2rgb_input",
        output = "screen",
        parameters = params
    )

    # color conversion for image_rect_nv12 for visualization
    params = [{
        "width":            LaunchConfiguration('width'),
        "height":           LaunchConfiguration('height'),
        "input_yuv_topic":  "camera/image_rect_nv12",
        "output_rgb_topic": "camera/image_rect_rgb",
        "yuv_format":       "YUV420",
        "yuv420_luma_only": False
    }]
    yuv2rbg_node2 = Node(
        package = "ti_viz_nodes",
        executable = "viz_color_conv_yuv2rgb",
        name = "viz_color_conv_yuv2rgb_node_rect",
        output = "screen",
        parameters = params
    )

    # color conversion for image_rect_nv12 for visualization
    approx_time_sync = LaunchConfiguration('approx_time_sync', default=False)

    params = [{
        "rectified_image_topic":   "camera/image_rect_rgb",
        "vision_cnn_tensor_topic": "vision_cnn/tensor",
        "vision_cnn_image_topic":  "vision_cnn/out_image",
        "approx_time_sync":        approx_time_sync
    }]
    viz_human_pose_node = Node(
        package = "ti_viz_nodes",
        executable = "viz_humanpose",
        name = "viz_humanpose",
        output = "screen",
        parameters = params
    )

    # rviz node
    rviz2_node = Node(
        package = "rviz2",
        executable = "rviz2",
        name = "rviz2",
        output = "screen",
        arguments=["-d", os.path.join(rviz_dir, 'humanpose_cnn.rviz')]
    )

    # Create the launch description with launch and node information
    ld.add_action(width_arg)
    ld.add_action(height_arg)
    ld.add_action(yuv2rbg_node1)
    ld.add_action(yuv2rbg_node2)
    ld.add_action(viz_human_pose_node)
    ld.add_action(rviz2_node)

    return ld
