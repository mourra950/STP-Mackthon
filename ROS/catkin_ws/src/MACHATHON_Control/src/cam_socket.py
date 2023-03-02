#!/usr/bin/env python3
"""
This node is used to get the camera image over wifi and publish it as a ros topic
"""
import socket
from typing import List, Optional
from std_msgs.msg import Float32

import cv2
import rospy
import numpy as np
from PIL import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
# from test import run_car

class CameraSocket:
    """
    Class to get image from a socket server on the esp

    Parameters
    ----------
    ip_address: str
        IP address of the socket server
    port: int
        Port to connect to the socket server
    """

    def __init__(self, ip_address: str, port: int):
        self.ip_address = ip_address
        self.port = port
        self.buffer_size = 2048
        self.server_buffer_size = 1400

    def get_image(self) :
        """
        Get image from the socket server

        Returns
        -------
        Optional[np.ndarray]
            The image as a number array (None if no image was received)
        """
        client_socket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
        client_socket.sendto(str.encode("0"), (self.ip_address, self.port))
        client_socket.settimeout(0.2)
        try:
            image = self.get_image_unsafe(client_socket)
        except Exception:
            image = None

        client_socket.close()
        return image

    def get_image_unsafe(self, client_socket: socket.socket) :
        """
        Get image from the socket server
        Note: this can throw an exception

        Returns
        -------
        Optional[np.ndarray]
            The image as a number array (None if no image was received)
        """
        n_bytes = client_socket.recvfrom(self.buffer_size)[0]
        n_frames = int(str(n_bytes)[2:-1])

        all_data: List[bytes] = []
        while True:
            msg_from_server = client_socket.recvfrom(self.buffer_size)[0]
            if (
                len(msg_from_server) == self.server_buffer_size
                and msg_from_server[0] == 255
                and msg_from_server[1] == 216
                and msg_from_server[2] == 255
            ):
                all_data = []
            all_data.append(msg_from_server)
            if (
                len(msg_from_server) < self.server_buffer_size
                and msg_from_server[-1] == 217
                and msg_from_server[-2] == 255
            ):
                data = b"".join(all_data)
                image = None
                if n_frames == len(data):
                    np_img = np.frombuffer(data, dtype=np.uint8)
                    image = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
                return image


def main():
    """
    Main function to launch the ros node
    """
    rospy.init_node("camera_socket", anonymous=True)
    ip_address = rospy.get_param("/camera/IP")
    port = rospy.get_param("/camera/port")
    camera_socket = CameraSocket(ip_address, port)

    image_pub = rospy.Publisher(
        rospy.get_param("/camera/image_topic"), Image, queue_size=1
    )
    # throttle_pub = rospy.Publisher("/throttle", Float32, queue_size=1)
    # steering_pub = rospy.Publisher("/steering", Float32, queue_size=1)
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        image = camera_socket.get_image()
        # speed= car_get_speed()
        # print(image)
        # print('ahmed')
        
        if image is None:

            continue
        # cv2.imshow('omar',image)
        # cv2.waitKey(2)
        # img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # img=cv2.rotate(image,rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)
        # cv2.imshow('colored',img)
        # img[0:50]=0
        # lower_bound = np.array([240, 240, 240])	 
        # upper_bound = np.array([255, 255, 255])
        # find the colors within the boundaries 
        # img = cv2.inRange(img, lower_bound,upper_bound)
        
        # cv2.waitKey(1)
        # cv2.imshow('thresh',img)
        # cv2.imwrite('./test.png',img)
        # steering,throttle=run_car(img)    
        # throttle_pub.publish(throttle)
        # steering_pub.publish(steering)
        image_msg = CvBridge().cv2_to_imgmsg(image, "bgr8")
        image_pub.publish(image_msg)
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
