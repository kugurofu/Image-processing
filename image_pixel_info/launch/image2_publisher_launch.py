from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():    
    return LaunchDescription([
        Node(
            package='image_pixel_info',
            executable='image2_publisher_node',
            name='image2_publisher_node',
            output='screen',
        ),
    ])
