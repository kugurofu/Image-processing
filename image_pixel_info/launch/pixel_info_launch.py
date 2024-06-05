from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_pixel_info',
            executable='pixel_info_node',
            name='pixel_info_node',
            output='screen',
        ),
    ])

