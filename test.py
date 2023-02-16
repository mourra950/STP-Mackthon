"""
Example code using the judge module
"""
import time

# pylint: disable=import-error
import cv2
import keyboard
import numpy as np

from machathon_judge import Simulator, Judge
# pid library and initialization
from simple_pid import PID
# pid1 = PID(1, 0.1, 0.01)
pid2 = PID(0.1, 0.1, 0.01)

c=False
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
    if abs(steering_angle) < 0.065:
        return 2
    else:
        return 0.3




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
    while True:
        fps_counter.step()

        # Get the image and show it
        img = simulator.get_image()
        
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fps = fps_counter.get_fps()
        
    # find the colors within the boundaries 
        # img2 = cv2.inRange(img, lower_bound,upper_bound)
        # draw fps on image
        # cv2.putText(
        #     img,
        #     f"FPS: {fps:.2f}",
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 255, 0),
        #     2,
        #     cv2.LINE_AA,
        # )
    
        cv2.waitKey(1)

        # Adham's edit 
        throttle = 0.17
        steering = 0
        steer_fact = 1.3
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bw = thresholding(rgb)
        bw[0:250]=0
        bw[420:480]=0
        blank=np.ones_like(bw)
        maskrectangle1=cv2.rectangle(blank.copy(),(320-200,0),(320+200,480),(255,255,255),-1)
        # maskrectangle2=cv2.rectangle(blank.copy(),(540,0),(640,480),(255,255,255),-1)
        
        kernel = np.ones((5,5),np.uint8)
        bw = cv2.erode(bw,kernel,iterations = 6)
        bw=np.bitwise_and(bw,maskrectangle1)
        # bw=np.bitwise_xor(bw,maskrectangle2)
        # img2 = cv2.erode(img2,kernel,iterations = 2)
        xpix, ypix = rover_coords(bw[:,:,0])
        dists, angles = to_polar_coords(xpix, ypix)
        if angles is not None:
            steering = np.mean(angles)
            
            if steering<0.1:
                pid2.setpoint = 0
                steering = pid2(steering)
                    
        
        # send the optimized pid steering
        # pid2.setpoint = np.mean(angles)
        # steering = pid2(steering)

    
        cv2.imshow("image", bw)
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
        #     throttle = -1
        
        throttle = get_throttle(steering * simulator.max_steer_angle / 1.8)
        simulator.set_car_steering(steering * simulator.max_steer_angle / 1.8)
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