"""
Example code using the judge module
"""
import time
import numpy as np
# pylint: disable=import-error
import cv2
import keyboard

from machathon_judge import Simulator, Judge


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

def color_thresh(img, above_thresh=(55,55,155),below_thresh=(65,65,165)):
    # Create an array of zeros same xy size as img, but single channel
    
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh_result = (img[:,:,0] > above_thresh[0]) \
                & (img[:,:,1] > above_thresh[1]) \
                & (img[:,:,2] > above_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    img[above_thresh_result] = 1

  
    below_thresh_result = (img[:,:,0] > below_thresh[0]) \
                & (img[:,:,1] > below_thresh[1]) \
                & (img[:,:,2] > below_thresh[2])
    # Index the array of zeros with the boolean array and set to 0
    img[below_thresh_result] = 0   

    
    # Return the binary image
    return img

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

    # Get the image and show it
    img = simulator.get_image()
      
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    lower_bound = np.array([30, 30, 130])	 
    upper_bound = np.array([70, 70, 180])
# find the colors within the boundaries 
    img = cv2.inRange(img, lower_bound,upper_bound)
    
    img[0:300]=0
    # print(img)
    # img=color_thresh(img)
    # print(img[0:2])  
    
    
    #perception
    
    
    
    
    
    
    fps = fps_counter.get_fps()

    # draw fps on image
    cv2.putText(
        img,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("image", img)
    cv2.waitKey(1)

    # Control the car using keyboard
    steering = 0
    #steering code
    if keyboard.is_pressed("a"):
        steering = 1
    elif keyboard.is_pressed("d"):
        steering = -1

    throttle = 0
    #banzeene control
    if keyboard.is_pressed("w"):
        throttle = 1
    elif keyboard.is_pressed("s"):
        throttle = -1

    simulator.set_car_steering(steering * simulator.max_steer_angle / 1.7)
    simulator.set_car_velocity(throttle * 25)


if __name__ == "__main__":
    # Initialize any variables needed
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    fps_counter = FPSCounter()

    # You should modify the value of the parameters to the judge constructor
    # according to your team's info
    judge = Judge(team_code="your_new_team_code", zip_file_path="your_solution.zip")

    # Pass the function that contains your main solution to the judge
    judge.set_run_hook(run_car)

    # Start the judge and simulation
    judge.run(send_score=False, verbose=True)
