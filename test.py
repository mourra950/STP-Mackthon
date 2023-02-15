"""
Example code using the judge module
"""
import time
import scipy
# pylint: disable=import-error
import cv2
import keyboard
import numpy as np

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

def rover_coords(binary_img):
    ypos, xpos = binary_img.nonzero()
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float64)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float64)
    return x_pixel, y_pixel

def to_polar_coords(x_pixel, y_pixel):
    # dist = np.sqrt(x_pixel**2 + y_pixel**2)
    angles = np.arctan2(y_pixel, x_pixel)
    return  angles #dist

def thresholding(img, thresh=(200, 200, 200)):
    thresholded = np.zeros_like(img[:,:])
    indecies = (img[:,:,0] > thresh[0]) & (img[:,:,1] > thresh[1]) & (img[:,:,2] > thresh[2])
    thresholded[indecies] = 255
    return thresholded

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
    fps = fps_counter.get_fps()
    # thresh
    lower_bound = np.array([30, 30, 130])	 
    upper_bound = np.array([70, 70, 180])
    img2 = cv2.inRange(img, lower_bound,upper_bound)
    
    # draw fps on image
    white = np.zeros_like(img2)
    circle = cv2.circle(white.copy(),(320,660),370,(255,255,255),-1)
    # circle = thresholding(circle)
    # Adham's edit 
    throttle = 0.1
    steering = 0
    steer_fact = 1.3
    # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bw = thresholding(img)
    # bw = cv2.bitwise_and(bw,circle)
    # kernel = np.ones((4,4),np.uint8)
    # img2=cv2.erode(img2,kernel,iterations = 1)
    
    img2 = cv2.bitwise_and(img2,circle)
    xpix, ypix = rover_coords(bw[:,:,0])
    angles = to_polar_coords(xpix, ypix)
    steering = np.mean(angles)*steer_fact
    img2=cv2.Canny(img2,10,200)
    
    lines = cv2.HoughLines(img2,1,np.pi/180,85)
    L=np.array([])
    if(lines is not None):
        for line in lines:
            rho,theta = line[0]
            print(rho)
            theta=((theta*180)/np.pi)-90
            # print(theta)
            L=np.append(L,theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(img2,(x1,y1),(x2,y2),(255,255,255),2)
        steering=np.mean(L)
    else:
        steering=0
        # print(len(img2))
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
    
    simulator.set_car_steering(steering * simulator.max_steer_angle / 1.7)
    simulator.set_car_velocity(throttle * 25)


if __name__ == "__main__":
    # Initialize any variables needed
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    fps_counter = FPSCounter()

    # You should modify the value of the parameters to the judge constructor
    # according to your team's info
    judge = Judge(team_code="9287485", zip_file_path="./STP-Mackathon-9287485.zip")

    # Pass the function that contains your main solution to the judge
    judge.set_run_hook(run_car)

    # Start the judge and simulation
    judge.run(send_score=False, verbose=True)
