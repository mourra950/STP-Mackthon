import time

# pylint: disable=import-error
import cv2
import keyboard
import numpy as np

from machathon_judge import Simulator, Judge

from simple_pid import PID

pid = PID(1, 0.1, 0.01)
count=0.0
previous=0
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
        
    return warped
class FPSCounter:
    def __init__(self):
        self.frames = []

    def step(self):
        self.frames.append(time.monotonic())

    def get_fps(self):
        n_seconds = 5

        count = 0
        cur_time = time.monotonic()
        for f in self.frames:
            if cur_time - f < n_seconds:  # Count frames in the past n_seconds
                count += 1

        return count / n_seconds

def rover_coords(binary_img):
    ypos, xpos = binary_img.nonzero()
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float64)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float64)
    return x_pixel, y_pixel

def to_polar_coords(x_pixel, y_pixel):
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

def thresholding(img, thresh=(220, 220, 220)):
    thresholded = np.zeros_like(img[:,:])
    indecies = (img[:,:,0] > thresh[0]) & (img[:,:,1] > thresh[1]) & (img[:,:,2] > thresh[2])
    thresholded[indecies] = 255
    return thresholded

###to get throttle
# def get_throttle(steering_angle) -> float:
#     global count,previous
#     # temp=previous
#     # if abs(temp-steering_angle)<0.035:
#     #     previous=steering_angle
#     #     steering_angle=temp
    
#     if abs(steering_angle) < 0.03:
#         count=0
        
#         return 4,1.7
#     elif  abs(steering_angle) < 0.07 :
        
        
#         return 1,1
#     else:
#         if count>0.15:
#             count+=0.0145
#         else:
#             count+=0.0165
        
#         return 0.185+count,0.9

def get_throttle(steering_angle) -> float:
    global count,previous
    temp=previous
    
    if abs(temp-steering_angle)<0.06:
        previous=steering_angle
        steering_angle=0
           
    # print(steering_angle)
    # if abs(steering_angle) < 0.03:
    #     count=0.00
    #     return 4,2.1
    # elif abs(steering_angle) < 0.06:
    #     count-=0.02
    #     if count<0:
    #         count=0
    # if count>0.15:
    #     count+=0.015
    # else:
    #     count+=0.018
    
    previous=(1-abs(steering_angle))*1+0.003 * ( 1 / ((3*(steering_angle**2))+0.001))+0.0001
    
    return previous , 0.85



def run_car(simulator: Simulator) -> None:
    """
    Function to control the car using keyboard
    Parameters
    ----------
    simulator : Simulator
        The simulator object to control the car
        The only functions that should be used are:
        - get_image()
        - set_car_steering()
        - set_car_velocity()
        - get_state()
    """
    fps_counter.step()
    dst_size= 37
    bottom_offset= 0
    source = np.float32([[222, 460],
                         [377, 468],
                         [376, 421],
                         [252, 416],
                         ])
    
    
    # Get the image and show it
    img = simulator.get_image()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    fps = fps_counter.get_fps()
    image=img
    destination = np.float32([[image.shape[1]/2 - 1.5*dst_size, image.shape[0] ],
                [image.shape[1]/2 + 1.5*dst_size, image.shape[0] ],
                [image.shape[1]/2 + 1.5*dst_size, image.shape[0] - 2*dst_size ], 
                [image.shape[1]/2 - 1.5*dst_size, image.shape[0] - 2*dst_size ],
                ])
    warped = perspect_transform(image, source, destination)
    # cv2.imshow("image", warped)
   
    cv2.waitKey(1)
   
    # Adham's edit 
    throttle = 0.17
    steering = 0
    steer_fact = 1.3
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    bw = thresholding(rgb)
    kernel = np.ones((5,5),np.uint8)
    bw = cv2.erode(bw,kernel,iterations = 6)
    # kernel = np.ones((35,3),np.uint8)
    # bw = cv2.erode(bw,kernel,iterations = 6)
    # bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    # bw = cv2.erode(bw,kernel,iterations = 6)
    # cv2.imwrite('./warped.png',bw)
    # bw[0:200]=0
    
    
    ##
    #mask
    ##
    # blank=np.ones_like(bw)
    # maskrectangle1=cv2.rectangle(blank.copy(),(320-210,0),(320+210,480),(255,255,255),-1)
    ################
    ##
    #apply mask
    #
    # bw=np.bitwise_and(bw,maskrectangle1)
    ######
    
    
    xpix, ypix = rover_coords(bw[:,:,0])
    dists, angles = to_polar_coords(xpix, ypix)
    steering = np.mean(angles)*steer_fact
    
    # # Control the car using keyboard
    # steering = 0
    # if keyboard.is_pressed("a"):
    #     steering = 1
    # elif keyboard.is_pressed("d"):
    #     steering = -1

    # throttle = 0
    # if keyboard.is_pressed("w"):
    #     throttle = 1
    # elif keyboard.is_pressed("s"):
    #     throttle = 0
    cv2.imshow("image", warped)
    throttle,fact= get_throttle(steering * simulator.max_steer_angle / 1.8)
    simulator.set_car_steering(steering * simulator.max_steer_angle /fact)
    simulator.set_car_velocity(throttle * 25)


if __name__ == "__main__":
    # Initialize any variables needed
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    fps_counter = FPSCounter()

    # You should modify the value of the parameters to the judge constructor
    # according to your team's info
    judge = Judge(team_code="qipVJIqVL", zip_file_path="./solution.zip")

    # Pass the function that contains your main solution to the judge
    judge.set_run_hook(run_car)

    # Start the judge and simulation
    judge.run(send_score=False, verbose=True)