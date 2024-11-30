import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ImageExtractor(Node):
    def __init__(self):
        super().__init__('image_extractor')
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.listener_callback,
            10)
        self.bridge = CvBridge()
        
    def listener_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Extract the specified regions
        region1 = cv_image[176:306, 267:404]
        region2 = cv_image[400:530, 267:404]
        
        # Display the extracted regions
        cv2.imshow("Region 1", region1)
        cv2.imshow("Region 2", region2)
        cv2.waitKey(1)
        
        # Visualize the RGB values in 3D mesh for both regions
        self.visualize_rgb_mesh(region1, region2)
        
        # Print the average RGB values
        self.print_average_rgb(region1, "Region 1")
        self.print_average_rgb(region2, "Region 2")

    def visualize_rgb_mesh(self, region1, region2):
        # Create meshgrid for pixel coordinates
        x1 = np.arange(region1.shape[1])
        y1 = np.arange(region1.shape[0])
        xx1, yy1 = np.meshgrid(x1, y1)

        x2 = np.arange(region2.shape[1])
        y2 = np.arange(region2.shape[0])
        xx2, yy2 = np.meshgrid(x2, y2)
        
        # Extract RGB channels for region1
        r1 = region1[:, :, 2]
        g1 = region1[:, :, 1]
        b1 = region1[:, :, 0]

        # Extract RGB channels for region2
        r2 = region2[:, :, 2]
        g2 = region2[:, :, 1]
        b2 = region2[:, :, 0]

        # Create 3D plot
        fig = plt.figure(figsize=(20, 10))

        # Plot Red channel
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.plot_surface(xx1, yy1, r1, facecolors=plt.cm.Reds(r1 / 255.0), rstride=1, cstride=1, linewidth=0, vmin = 0, vmax=255)
        ax1.set_title('Red Channel for Region 1')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Intensity')
        ax1.set_zlim(0,255)

        ax2 = fig.add_subplot(234, projection='3d')
        ax2.plot_surface(xx2, yy2, r2, facecolors=plt.cm.Reds(r2 / 255.0), rstride=1, cstride=1, linewidth=0, vmin = 0, vmax=255)
        ax2.set_title('Red Channel for Region 2')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Intensity')
        ax2.set_zlim(0,255)

        # Plot Green channel
        ax3 = fig.add_subplot(232, projection='3d')
        ax3.plot_surface(xx1, yy1, g1, facecolors=plt.cm.Greens(g1 / 255.0), rstride=1, cstride=1, linewidth=0, vmin = 0, vmax=255)
        ax3.set_title('Green Channel for Region 1')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Intensity')
        ax3.set_zlim(0,255)

        ax4 = fig.add_subplot(235, projection='3d')
        ax4.plot_surface(xx2, yy2, g2, facecolors=plt.cm.Greens(g2 / 255.0), rstride=1, cstride=1, linewidth=0, vmin = 0, vmax=255)
        ax4.set_title('Green Channel for Region 2')
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_zlabel('Intensity')
        ax4.set_zlim(0,255)

        # Plot Blue channel
        ax5 = fig.add_subplot(233, projection='3d')
        ax5.plot_surface(xx1, yy1, b1, facecolors=plt.cm.Blues(b1 / 255.0), rstride=1, cstride=1, linewidth=0, vmin = 0, vmax=255)
        ax5.set_title('Blue Channel for Region 1')
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.set_zlabel('Intensity')
        ax5.set_zlim(0,255)

        ax6 = fig.add_subplot(236, projection='3d')
        ax6.plot_surface(xx2, yy2, b2, facecolors=plt.cm.Blues(b2 / 255.0), rstride=1, cstride=1, linewidth=0, vmin = 0, vmax=255)
        ax6.set_title('Blue Channel for Region 2')
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Intensity')
        ax6.set_zlim(0,255)

        plt.show()
        
    def print_average_rgb(self, region, title):
        # Calculate the average RGB values
        mean_b = np.mean(region[:, :, 0])
        mean_g = np.mean(region[:, :, 1])
        mean_r = np.mean(region[:, :, 2])
        
        print(f'{title} - Average RGB:')
        print(f'Red: {mean_r:.2f}')
        print(f'Green: {mean_g:.2f}')
        print(f'Blue: {mean_b:.2f}')

def main(args=None):
    rclpy.init(args=args)
    image_extractor = ImageExtractor()
    rclpy.spin(image_extractor)
    image_extractor.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

