import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import subprocess
import yaml

class WaypointMonitor(Node):
    def __init__(self):
        super().__init__('waypoint_monitor')
        self.subscription = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10)
        self.waypoints = self.load_waypoints('/home/ubuntu/ros2_ws/src/kbkn_maps/waypoints/gazebo/orange_hosei/slam_toolbox/waypoints.yaml')
        self.current_waypoint_index = 0
        self.waypoints_to_launch = [0, 2]  # 例: 1番目、3番目のウェイポイントで起動
        self.process = None  # 起動したプロセスを保持する変数
        self.process_stop_index = None  # プロセスを停止するウェイポイントのインデックス

    def load_waypoints(self, filepath):
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
        waypoints = [(point['point']['x'], point['point']['y']) for point in data['waypoints']]
        return waypoints

    def odom_callback(self, msg):
        current_position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.current_waypoint_index < len(self.waypoints):
            target_position = self.waypoints[self.current_waypoint_index]
            if self.is_at_waypoint(current_position, target_position):
                self.get_logger().info(f'Waypoint {self.current_waypoint_index} reached!')
                
                # 現在のインデックスが停止するウェイポイントに一致する場合、プロセスを停止
                if self.process is not None and self.current_waypoint_index == self.process_stop_index:
                    self.terminate_process()

                # 現在のインデックスが指定されたウェイポイントに一致する場合、プロセスを起動
                if self.current_waypoint_index in self.waypoints_to_launch:
                    self.launch_image_publisher()
                    # プロセスを1つ後のウェイポイントで停止するように設定
                    self.process_stop_index = self.current_waypoint_index + 1

                self.current_waypoint_index += 1

    def is_at_waypoint(self, current_position, target_position):
        return (abs(current_position[0] - target_position[0]) < 0.1 and
                abs(current_position[1] - target_position[1]) < 0.1)

    def launch_image_publisher(self):
        try:
            self.process = subprocess.Popen(
                ['ros2', 'launch', 'image_pixel_info', 'traffic_light_status_launch.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.get_logger().info('Successfully launched traffic_light_status_launch.py')
        except subprocess.SubprocessError as e:
            self.get_logger().error(f'Failed to launch traffic_light_status_launch.py: {e}')

    def terminate_process(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)  # プロセスが停止するまで待機
                self.get_logger().info('Successfully terminated traffic_light_status_launch.py')
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.get_logger().info('Force killed traffic_light_status_launch.py')
            self.process = None
            self.process_stop_index = None  # プロセス停止インデックスをリセット

def main(args=None):
    rclpy.init(args=args)
    waypoint_monitor = WaypointMonitor()
    rclpy.spin(waypoint_monitor)
    waypoint_monitor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

