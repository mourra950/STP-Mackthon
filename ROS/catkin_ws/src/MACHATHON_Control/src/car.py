#!/usr/bin/env python3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# from PIL import Image
from std_msgs.msg import Float32
import cv2
import rospy
import numpy as np


class Car:
    def __init__(self):
        rospy.init_node("car_controller", anonymous=True)
        
        
        self.previous = 0
        self.count = -20

        # self.throttle_topic = rospy.get_param("/throttle", Float32, queue_size=10)
        self.image = None
        self.speed = 0
        self.rate = rospy.Rate(10)
        self.throttle_pub = rospy.Publisher("/throttle", Float32, queue_size=10)
        self.steering_pub = rospy.Publisher("/steering", Float32, queue_size=10)
        self.speed_sub = rospy.Subscriber("/speed", Float32, self.update_car_speed)
        self.image_sub = rospy.Subscriber("/camera_image", Image, self.update_image)
    
        

    def perspect_transform(self,img, src, dst):

        M = cv2.getPerspectiveTransform(src, dst)
        # keep same size as input image
        warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

        return warped



    def car_coords(self,binary_img):
        ypos, xpos = binary_img.nonzero()
        x_pixel = -(ypos - binary_img.shape[0]).astype(np.float64)
        y_pixel = -(xpos - binary_img.shape[1]/2).astype(np.float64)
        return x_pixel, y_pixel


    def to_polar_coords(self,x_pixel, y_pixel):
        # dist = np.sqrt(x_pixel**2 + y_pixel**2)
        angles = np.arctan2(y_pixel, x_pixel)
        return  angles


    def thresholding(self,img, thresh=(100, 100, 100)):
        thresholded = np.zeros_like(img[:, :])
        indecies = (img[:, :, 0] > thresh[0]) & (
            img[:, :, 1] > thresh[1]) & (img[:, :, 2] > thresh[2])
        thresholded[indecies] = 255
        return thresholded

        
    def get_steering_throttle(self):
        global count, previous, dst_size,pid
        dst_size = 50
        count=0
        # img=cv2.resize(img,(640,480))
        try:
            if self.image==None:
                return 0,0
        except:
            print('ahmed')
        
        # print(type(self.image))
        img = CvBridge().imgmsg_to_cv2(self.image, "bgr8")
        
        img=cv2.rotate(img,rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('j',img)
        

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # kernel = np.ones((5,5),np.float32)/25
        # img = cv2.filter2D(img,-1,kernel)
        # img=cv2.equalizeHist(img)
        
        bottom_offset = 0
        
        image = img
        
        cv2.waitKey(1)
        # fps_counter.step()


        throttle = 0

        source = np.float32([[7, 303],
                            [235, 303],
                            [191, 226],
                            [54, 226],
                            ])
        destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                                [image.shape[1]/2 + dst_size,
                                    image.shape[0] - bottom_offset],
                                [image.shape[1]/2 + dst_size, image.shape[0] -
                                    2*dst_size - bottom_offset],
                                [image.shape[1]/2 - dst_size, image.shape[0] -
                                    2*dst_size - bottom_offset],
                                ])

        warped = self.perspect_transform(image, source, destination)
        cv2.imshow('thresh',warped)
        
        ret,thresh = cv2.threshold(warped,180,255,cv2.THRESH_BINARY)
        cv2.imshow('thresh',thresh)
        
        xpix, ypix = self.car_coords(thresh)
        angles = self.to_polar_coords(xpix, ypix)
        # bw[440:480]=0
        throttle = 0
        steering=0
        if angles.any():

            steering = np.mean(angles)
            steeringdegrees=np.mean(angles*180/np.pi)
            # throttle=16-np.clip(abs(steeringdegrees)/2,0,20-count)

            if abs(steeringdegrees)<6:
                count-=2
                count=np.clip(count,0,20)
            
            if throttle<11 :
                count+=5
                count=np.clip(count,0,10) 
            if throttle>6:
                throttle+=6
        else:
            steering=-0.2
        steering=np.clip(3*steering*180/np.pi,-25,35)
        if abs(steering)>16:
            steering=np.clip(3*steering,-30,30)
        return steering,0
        
    def run(self):
        while not rospy.is_shutdown():
            steering, throttle = self.get_steering_throttle()
            self.throttle_pub.publish(throttle)
            self.steering_pub.publish(steering)
            self.rate.sleep()
        rospy.spin()

    def update_car_speed(self,speed):
        self.speed = speed
    
    def update_image(self,img):
        self.image = img

if __name__ == "__main__":
    try:
        car = Car()
        car.run()
    except KeyboardInterrupt:
        pass
        