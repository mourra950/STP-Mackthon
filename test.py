import time

# pylint: disable=import-error
import cv2
import keyboard
import numpy as np

from machathon_judge import Simulator, Judge

from simple_pid import PID
dst_size = 30
previous = 0
pid = PID(0.50, 0.07, 0.01, setpoint=0)
count = 0


def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

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
    y_pixel = -(xpos - binary_img.shape[1]/2).astype(np.float64)
    return x_pixel, y_pixel


def to_polar_coords(x_pixel, y_pixel):
    # dist = np.sqrt(x_pixel**2 + y_pixel**2)
    angles = np.arctan2(y_pixel, x_pixel)
    return  angles


def thresholding(img, thresh=(200, 200, 200)):
    thresholded = np.zeros_like(img[:, :])
    indecies = (img[:, :, 0] > thresh[0]) & (
        img[:, :, 1] > thresh[1]) & (img[:, :, 2] > thresh[2])
    thresholded[indecies] = 255
    return thresholded

# to get throttle


def get_throttle(steering_angle) -> float:
    global count

    if abs(steering_angle) < 0.07:
        count = 0

        return 4, 1.7
    elif abs(steering_angle) > 1.5:

        return 0.065, 0.6
    else:
        if count > 0.15:
            count += 0.013
        else:
            count += 0.02

        return 0.19+count, 0.7


def run_car(simulator: Simulator) -> None:
    global count, previous, dst_size,pid
    img = simulator.get_image()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    bottom_offset = -100
    
    image = img
    
    cv2.waitKey(1)
    # fps_counter.step()

    

    throttle = 0

    source = np.float32([[222, 460],
                         [377, 468],
                         [376, 421],
                         [252, 416],
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

    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    bw = thresholding(rgb)

    
    kernel = np.ones((4, 6), np.uint8)
    bw = cv2.erode(bw, kernel, iterations=8)
    
    cv2.imshow('image', bw)

    xpix, ypix = rover_coords(bw[:, :, 0])
    angles = to_polar_coords(xpix, ypix)
    # bw[440:480]=0
    throttle = 10
    steering=0
    if angles.any():
        steering = np.mean(angles)
        
        steeringdegrees=np.mean(angles*180/np.pi)
        
        
        throttle=16-np.clip(abs(steeringdegrees)/2,0,20-count)
        if abs(steeringdegrees)<6:
            count-=2
            count=np.clip(count,0,20)
        
        if throttle<11 :
            count+=1
            count=np.clip(count,0,13) 
            # count=np.clip(count,0,10)
        # print(steeringdegrees)
        pid.setpoint=steering
        steering=pid(previous)
        previous=steering
    else:
        steering=-0.2
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
