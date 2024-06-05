from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():    
    return LaunchDescription([
        Node(
            package='image_color_discrimination',
            executable='color_discrimination_node',
            name='color_discrimination_node',
            output='screen',
        ),
    ])
