import time

# pylint: disable=import-error
import cv2
import keyboard
import numpy as np
from simple_pid import PID

dst_size = 50
previous = 0
pid = PID(0.50, 0.07, 0.01, setpoint=0)
count = -20


def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    return warped



def rover_coords(binary_img):
    ypos, xpos = binary_img.nonzero()
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float64)
    y_pixel = -(xpos - binary_img.shape[1]/2).astype(np.float64)
    return x_pixel, y_pixel


def to_polar_coords(x_pixel, y_pixel):
    # dist = np.sqrt(x_pixel**2 + y_pixel**2)
    angles = np.arctan2(y_pixel, x_pixel)
    return  angles


def thresholding(img, thresh=(120, 120, 120)):
    thresholded = np.zeros_like(img[:, :])
    indecies = (img[:, :, 0] > thresh[0]) & (
        img[:, :, 1] > thresh[1]) & (img[:, :, 2] > thresh[2])
    thresholded[indecies] = 255
    return thresholded

# to get throttle


def run_car(img):
    global count, previous, dst_size,pid
    cv2.imshow('img',img)
    
    # img=cv2.resize(img,(640,480))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    

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