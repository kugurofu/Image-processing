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
        ('share/' + package_name + '/launch', ['launch/pixel_info_launch.py','launch/image_publisher_launch.py']),
        ('share/' + package_name + '/images', ['images/test1.jpeg']),
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
        ],
    },
)
