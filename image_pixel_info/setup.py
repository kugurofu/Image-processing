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
        ('share/' + package_name + '/images', ['images/test6.jpeg','images/test7.jpeg','images/test5.jpeg','images/test2.jpg','images/test3.jpg','images/test4.jpg','images/202406131905_1.jpg','images/202406131905_2.jpg','images/202406141149_2.jpg','images/202406141149_3.jpg','images/202406141157_1.jpg','images/202406212220_1.jpg','images/202406212220_2.jpg','images/images.jpeg','images/1.jpeg','images/2.jpeg','images/3.jpeg','images/4.jpeg','images/5.jpeg','images/6.jpeg','images/7.jpeg','images/8.jpeg','images/9.jpeg','images/10.jpeg','images/11.jpeg','images/12.jpeg','images/13.jpeg','images/14.jpeg','images/15.jpeg','images/16.jpeg','images/17.jpeg','images/18.jpeg','images/19.jpeg','images/20.jpeg','images/21.jpeg','images/22.jpeg','images/23.jpeg','images/24.jpeg','images/25.jpeg','images/26.jpeg','images/27.jpeg','images/28.jpeg','images/29.jpeg','images/30.jpeg','images/31.webp','images/32.webp','images/33.jpeg','images/34.jpeg','images/35.jpeg','images/36.jpeg','images/37.jpeg','images/38.jpeg','images/39.jpeg','images/40.jpeg',]),
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
        	'camera_publisher = image_pixel_info.camera_publisher:main',
        	'image_subscriber = image_pixel_info.image_subscriber_node:main',
        	'fisheye_undistort = image_pixel_info.fisheye_undistort:main',
        	'caribration = image_pixel_info.caribration:main',
        	'yolo = image_pixel_info.yolo:main',
        	'yolov8 = image_pixel_info.yolov8:main',
        	'traffic_light_controller = image_pixel_info.traffic_light_controller:main',
        	'waypoint_controller = image_pixel_info.waypoint_controller:main',
        	'retinex = image_pixel_info.retinex:main',
        	'retinex_image = image_pixel_info.retinex_image:main',
        	'luminancefactor_image = image_pixel_info.luminancefactor_image:main',
        	'LMS_image = image_pixel_info.LMS_image:main',
        	'judge_luminance = image_pixel_info.judge_luminance:main',
        ],
    },
)
