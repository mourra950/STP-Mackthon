from PIL import Image
from std_msgs.msg import Float32
import cv2
import rospy
import numpy as np


class Car:
    def __init__(self) -> None:
        rospy.init_node("car_controller", anonymous=True)
        # self.throttle_topic = rospy.get_param("/throttle", Float32, queue_size=10)
        self.image = None
        self.speed = 0
        self.rate = rospy.Rate(10)
        self.throttle_pub = rospy.Publisher("/throttle", Float32, queue_size=10)
        self.steering_pub = rospy.Publisher("/steering", Float32, queue_size=10)
        self.speed_sub = rospy.Subscriber("/speed", Float32, self.update_car_speed)
        self.image_sub = rospy.Subscriber("/camera_topic", Image, self.update_image)
    
    def get_steering_throttle():
        global count, previous, dst_size,pid
        cv2.imshow('img',self.image)
        
        # img=cv2.resize(img,(640,480))
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # kernel = np.ones((5,5),np.float32)/25
        # img = cv2.filter2D(img,-1,kernel)
        # img=cv2.equalizeHist(img)
        
        bottom_offset = 20
        
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

        warped = perspect_transform(image, source, destination)
        # warped[0:100]=0
        # rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        
        ret,thresh = cv2.threshold(warped,170,255,cv2.THRESH_BINARY)
        cv2.imshow('thresh',thresh)

        # print('ahmed')
        # cv2.imshow('bw',bw)
        kernel = np.ones((2, 3), np.uint8)
        bw = cv2.erode(bw, kernel, iterations=4)
        

        xpix, ypix = rover_coords(thresh)
        angles = to_polar_coords(xpix, ypix)
        # bw[440:480]=0
        throttle = 12
        steering=0
        if angles.any():
            steering = np.mean(angles)
            # print(steering)
            steeringdegrees=np.mean(angles*180/np.pi)
            
            # print(steering)
            throttle=16-np.clip(abs(steeringdegrees)/2,0,20-count)
            if abs(steeringdegrees)<6:
                count-=2
                count=np.clip(count,0,20)
            
            if throttle<11 :
                count+=5
                count=np.clip(count,0,10) 
            if throttle>6:
                throttle+=6
                # count=np.clip(count,0,10)
            # print(steeringdegrees)
            # pid.setpoint=steering
            # steering=pid(previous)
            previous=steering
        else:
            steering=-0.2
        steering=np.clip(3*steering*180/np.pi,-20,20)
        if abs(steering)>13:
            steering=np.clip(2*steering,-20,20)
        return steering,100
        
    def run():
        while not rospy.is_shutdown():
            steering, throttle = get_steering_throtlle()
            self.throttle_pub.publish(throttle)
            self.steering_pub.publish(steering)
            self.rate.sleep()
        rospy.spin()

    def update_car_speed(speed):
        self.speed = speed
    
    def update_image(img):
        self.image = img

if __name__ == "__main__":
    try:
        car = Car()
        car.run()
    except KeyboardInterrupt:
        pass
        