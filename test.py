import time

# pylint: disable=import-error
import cv2
import keyboard
import numpy as np

from machathon_judge import Simulator, Judge

from simple_pid import PID

pid = PID(1, 0.1, 0.01)
count = 0
count1 = 0
c = 0
previous = 0


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
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


def thresholding(img, thresh=(200, 200, 200)):
    thresholded = np.zeros_like(img[:, :])
    indecies = (img[:, :, 0] > thresh[0]) & (
        img[:, :, 1] > thresh[1]) & (img[:, :, 2] > thresh[2])
    thresholded[indecies] = 255
    return thresholded

# to get throttle


def get_throttle(steering_angle) -> float:
    global count
    if abs(steering_angle) < 0.070:
        count = 0.05

        return 4, 2
    elif abs(steering_angle) > 0.14:
        # print(abs(round(steering_angle,4)))
        count = 0
        return 0.15, 0.65
    else:
        
        count += 0.020
        

        return 0.16+count, 0.80


def run_car(simulator: Simulator) -> None:
    global c
    c += 0.5
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
    dst_size = 25
    source = np.float32([[220, 450],
                         [380, 450],
                         [377, 416],
                         [252, 416],
                         ])

    # Get the image and show it
    img = simulator.get_image()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    fps = fps_counter.get_fps()
    image = img
    destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0]],
                              [image.shape[1]/2 + dst_size, image.shape[0]],
                              [image.shape[1]/2 + dst_size,
                                  image.shape[0] - 2*dst_size],
                              [image.shape[1]/2 - dst_size,
                                  image.shape[0] - 2*dst_size],
                              ])
    warped = perspect_transform(image, source, destination)
    # cv2.imshow("image", warped)

    cv2.waitKey(1)

    # Adham's edit
    throttle = 0.17
    steering = 0
    steer_fact = 1.3
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    lower_bound = np.array([30, 30, 130])
    upper_bound = np.array([70, 70, 180])
# find the colors within the boundaries
    red = cv2.inRange(warped, lower_bound, upper_bound)
    # kernel = np.ones((5, 5), np.uint8)
    # red1 = cv2.dilate(red.copy(), kernel, iterations=6)
    # red1 = cv2.cvtColor(red1, cv2.COLOR_BGR2RGB)
    # red1 = thresholding(red1, (1, 1, 1))

    kernel = np.array((
        [0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0]
    ))
    red2 = cv2.morphologyEx(red, cv2.MORPH_HITMISS, kernel, iterations=6)
    cv2.imshow('image', red2)
    kernel = np.ones((2, 20), np.uint8)
    # if red2.any():
    red2 = cv2.dilate(red2, kernel, iterations=10)
    red2 = cv2.cvtColor(red2, cv2.COLOR_BGR2RGB)
    red2 = thresholding(red2, (1, 1, 1))
    red2 = cv2.bitwise_not(red2)
    # red2=cv2.erode(red,kernel,iterations = 1)

    bw = thresholding(rgb)
    kernel = np.ones((5, 5), np.uint8)
    # bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    bw = cv2.erode(bw, kernel, iterations=5)
    # red=np.bitwise_not(red)
    bw=np.bitwise_and(bw,red2)

    # kernel = np.ones((35,3),np.uint8)
    # bw = cv2.erode(bw,kernel,iterations = 6)
    # bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    # bw = cv2.erode(bw,kernel,iterations = 6)
    # cv2.imwrite('./warped.png',bw)
    # bw[0:200]=0

    ##
    # mask
    ##
    # blank=np.ones_like(bw)
    # maskrectangle1=cv2.rectangle(blank.copy(),(320-210,0),(320+210,480),(255,255,255),-1)
    ################
    ##
    # apply mask
    #
    # bw=np.bitwise_and(bw,maskrectangle1)
    ######

    xpix, ypix = rover_coords(bw[:, :, 0])
    dists, angles = to_polar_coords(xpix, ypix)
    # print(len(angles))
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
    # print(steering)
    throttle, fact = get_throttle(
        steering * simulator.max_steer_angle / 1.7)
    # +(np.sign(steering)*0.05)
    simulator.set_car_steering((steering * simulator.max_steer_angle / fact))
    simulator.set_car_velocity(throttle * 30.5)


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
