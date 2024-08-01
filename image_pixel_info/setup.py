from setuptools import find_packages, setup

package_name = 'image_pixel_info'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/pixel_info_launch.py','launch/image_publisher_launch.py','launch/image2_publisher_launch.py','launch/traffic_light_status_launch.py']),
        ('share/' + package_name + '/images', ['images/test1.jpeg','images/test2.jpg','images/test3.jpg','images/test4.jpg','images/202406131905_1.jpg','images/202406131905_2.jpg','images/202406141149_2.jpg','images/202406141149_3.jpg','images/202406141157_1.jpg','images/202406212220_1.jpg','images/202406212220_2.jpg']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='Package to get pixel info from an image topic in ROS2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'pixel_info_node = image_pixel_info.pixel_info_node:main',
        	'image_publisher_node = image_pixel_info.image_publisher_node:main',
        	'image2_publisher_node = image_pixel_info.image2_publisher_node:main',
        	'mjpg_camera_publisher = image_pixel_info.mjpg_camera_publisher:main',
        	'image_subscriber = image_pixel_info.image_subscriber_node:main',
        	'fisheye_undistort = image_pixel_info.fisheye_undistort:main',
        	'caribration = image_pixel_info.caribration:main',
        	'yolo = image_pixel_info.yolo:main',
        	'yolov8 = image_pixel_info.yolov8:main',
        	'traffic_light_controller = image_pixel_info.traffic_light_controller:main',
        	'waypoint_controller = image_pixel_info.waypoint_controller:main',
        ],
    },
)
