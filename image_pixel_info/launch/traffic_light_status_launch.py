from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        '''
        Node(
            package='image_pixel_info',
            executable='mjpg_camera_publisher',
            name='mjpg_camera_publisher',
            output='screen',
        ),
        '''
        Node(
            package='image_pixel_info',
            executable='yolov8',
            name='yolov8',
            output='screen',
        ),
        Node(
            package='image_pixel_info',
            executable='traffic_light_controller',
            name='traffic_light_controller',
            output='screen',
        ),
    ])

