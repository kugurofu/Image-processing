from setuptools import find_packages, setup

package_name = 'image_color_discrimination'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/color_discrimination_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='tenten31569@icloud.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        	'color_discrimination_node = image_color_discrimination.color_discrimination_node:main',
        	'color_discrimination2_node = image_color_discrimination.color_discrimination2_node:main',
        	'grayscale_node = image_color_discrimination.grayscale_node:main',
        	'grayscale2_node = image_color_discrimination.grayscale2_node:main',
        	'rgb_judge_node = image_color_discrimination.rgb_judge_node:main',
        	'gray_judge_node = image_color_discrimination.gray_judge_node:main'
        ],
    },
)
