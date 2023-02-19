import time

# pylint: disable=import-error
import cv2
import keyboard
import numpy as np

from machathon_judge import Simulator, Judge

from simple_pid import PID
previous=0
pid = PID(1, 0.1, 0.01)
count=0
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

def thresholding(img, thresh=(200, 200, 200)):
    thresholded = np.zeros_like(img[:,:])
    indecies = (img[:,:,0] > thresh[0]) & (img[:,:,1] > thresh[1]) & (img[:,:,2] > thresh[2])
    thresholded[indecies] = 255
    return thresholded

###to get throttle
def get_throttle(steering_angle) -> float:
    global count
    
    if abs(steering_angle) < 0.07:
        count=0
        
        return 4,1.7
    elif abs(steering_angle) > 1.5:
        
        return 0.065,0.6
    else:
        if count>0.15:
            count+=0.013
        else:
            count+=0.02
        
        return 0.19+count,0.7




def run_car(simulator: Simulator) -> None:
    global count,previous
    img = simulator.get_image()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    image=img
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
    cv2.waitKey(1)
    # fps_counter.step()
    dst_size= 35
    bottom_offset= -90
    c=-0
    source = np.float32([[222, 460-c],
                            [377, 468-c],
                            [376, 421-c],
                            [252, 416-c],
                            ])
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                    [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                    [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                    [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                    ])
    # dst_size= 40
    # source = np.float32([[222, 460],
    #                      [377, 468],
    #                      [376, 421],
    #                      [252, 416],
    #                      ])
    
    
    # # Get the image and show it
    
    # destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] ],
    #             [image.shape[1]/2 + dst_size, image.shape[0] ],
    #             [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size ], 
    #             [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size ],
    #             ])
    warped = perspect_transform(image, source, destination)
    # cv2.imshow("image", warped)
    # lower_bound = np.array([30, 30, 130])
    # upper_bound = np.array([70, 70, 180])
# find the colors within the boundaries
    # red = cv2.inRange(warped, lower_bound, upper_bound)
    # cv2.waitKey(1)
    # kernel = np.array(( 
    #                    [0, 0, 0,0],
    #     [1, 1, 1,1],
    #     [0, 0, 0,0]))
    # red2 = cv2.morphologyEx(red, cv2.MORPH_HITMISS, kernel, iterations=20)
    
    # kernel = np.ones((2, 20), np.uint8)
    # red2 = cv2.dilate(red2, kernel, iterations=10)
    # red2 = cv2.cvtColor(red2, cv2.COLOR_BGR2RGB)
    # red2 = thresholding(red2, (1, 1, 1))
    # red2 = cv2.bitwise_not(red2)
    # cv2.imshow('image', red2)
    
    
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    bw = thresholding(rgb)
    
    # kernel = np.ones((7,7),np.float32)/(7*7)
    # bw = cv2.filter2D(bw,-1,kernel)
    # kernel = np.ones((35,45),np.uint8)
    # bw=cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((5,5),np.uint8)
    bw = cv2.erode(bw,kernel,iterations = 7)
    bw[0:150]=0
    kernel = np.ones((7,7),np.float32)
    bw = cv2.filter2D(bw,-1,kernel)
    bw = cv2.filter2D(bw,-1,kernel)
    cv2.imshow('image', bw)
    # bw=np.bitwise_and(bw,red2)
    
    
    
    xpix, ypix = rover_coords(bw[:,:,0])
    dists, angles = to_polar_coords(xpix, ypix)
    throttle=14
    steering=previous* np.pi/180
    if angles.any():
        steering = np.mean(angles)*1.2
        previous=steering
    
    # steering1 = np.mean(angles* 180/np.pi)
    
#     if abs(previous-steering)>7:
#         steering/=2
#     elif steering<4:
#         steering/=4
# # print(steering1)
#     if abs(steering1)<9:
#         if count>10:
#             count=7
#         count-=0.5
#         throttle=16-count
    
#     elif abs(steering1)>16:
#         throttle=1+count
        
#         count+=6
            
    
    # throttle =2* 1/(4 * (steering**2)+0.0001)
    # throttle,fact= get_throttle(steering )
    throttle=10
    simulator.set_car_steering(steering)
    simulator.set_car_velocity(throttle)


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