import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition

class TrafficLightController(Node):

    def __init__(self):
        super().__init__('traffic_light_controller')
        
        self.subscription = self.create_subscription(
            String,
            '/traffic_light_status',
            self.traffic_light_callback,
            10)
        
        self.client = self.create_client(ChangeState, '/waypoint_follower/change_state')
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.timer = self.create_timer(1.0, self.check_service)
        
        self.stop_timer = None
        self.stop_command_duration = 1  # Duration to send stop commands in seconds

        self.previous_status = None  # Keep track of the previous status

        self.get_logger().info('TrafficLightController node has been initialized')

    def check_service(self):
        if not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Waiting for service /waypoint_follower/change_state...')
        else:
            self.get_logger().info('Service /waypoint_follower/change_state is available.')
            self.timer.cancel()  # Stop the timer once the service is available

    def traffic_light_callback(self, msg):
        if msg.data in ['red', 'blue']:
            self.deactivate_navigation()
        if self.previous_status == 'red' and msg.data == 'blue':
            self.activate_navigation()
            rclpy.shutdown()  # Shutdown the node after reactivation kesebakeizokutekiniriyoukanou?
        self.previous_status = msg.data

    def deactivate_navigation(self):
        # Send a stop command
        stop_twist = Twist()
        self.cmd_vel_pub.publish(stop_twist)
        
        # Call the service to deactivate navigation
        req = ChangeState.Request()
        req.transition.id = Transition.TRANSITION_DEACTIVATE
        self.call_change_state_service(req, "deactivated")

        # Start the timer to send stop commands
        self.stop_timer = self.create_timer(0.1, self.send_stop_command)
        self.stop_command_end_time = self.get_clock().now() + rclpy.duration.Duration(seconds=self.stop_command_duration)

    def send_stop_command(self):
        if self.get_clock().now() >= self.stop_command_end_time:
            if self.stop_timer is not None:
                self.stop_timer.cancel()
                self.stop_timer = None
        else:
            stop_twist = Twist()
            self.cmd_vel_pub.publish(stop_twist)

    def activate_navigation(self):
        # Stop the stop command timer if it is running
        if self.stop_timer is not None:
            self.stop_timer.cancel()
            self.stop_timer = None

        # Call the service to activate navigation
        req = ChangeState.Request()
        req.transition.id = Transition.TRANSITION_ACTIVATE
        self.call_change_state_service(req, "activated")

    def call_change_state_service(self, req, action):
        self.future = self.client.call_async(req)
        self.future.add_done_callback(lambda future: self.response_callback(future, action))

    def response_callback(self, future, action):
        try:
            response = future.result()
            self.get_logger().info(f'Navigation {action}: {response.success}')
        except Exception as e:
            self.get_logger().error(f'Service call failed: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    traffic_light_controller = TrafficLightController()
    rclpy.spin(traffic_light_controller)
    traffic_light_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

