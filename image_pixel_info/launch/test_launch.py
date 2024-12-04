from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_pixel_info',
            executable='camera_publisher',
            name='mjpg_camera_publisher',
            output='screen',
        ),
        Node(
            package='image_pixel_info',
            executable='LMS_image',
            name='LMS_image',
            output='screen',
        ),
        Node(
            package='image_pixel_info',
            executable='judge_luminance',
            name='judge_luminance',
            output='screen',
        ),
    ])

