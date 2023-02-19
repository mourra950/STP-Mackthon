import os
import math
import matplotlib.pyplot as plt
import time

# pylint: disable=import-error
import cv2
import keyboard
import numpy as np

from machathon_judge import Simulator, Judge

from simple_pid import PID

pid = PID(1, 0.1, 0.01)
count = 0
c = 0
previous = 0


# Global parameters

# Gaussian smoothing
kernel_size = 3

# Canny Edge Detector
low_threshold = 50
high_threshold = 150

# Region-of-interest vertices
# We want a trapezoid shape, with bottom edge at the bottom of the image
# width of bottom edge of trapezoid, expressed as percentage of image width
trap_bottom_width = 0.85
trap_top_width = 0.07  # ditto for top edge of trapezoid
trap_height = 0.4  # height of the trapezoid expressed as percentage of image height

# Hough Transform
rho = 2  # distance resolution in pixels of the Hough grid
theta = 1 * np.pi/180  # angular resolution in radians of the Hough grid
threshold = 15	 # minimum number of votes (intersections in Hough grid cell)
min_line_length = 10  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments


# Helper functions
def grayscale(img):
	"""Applies the Grayscale transform
	This will return an image with only one color channel
	but NOTE: to see the returned image as grayscale
	you should call plt.imshow(gray, cmap='gray')"""
	return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
	"""Applies the Canny transform"""
	return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
	"""Applies a Gaussian Noise kernel"""
	return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
	"""
	Applies an image mask.

	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	# defining a blank mask to start with
	mask = np.zeros_like(img)

	# defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255

	# filling pixels inside the polygon defined by "vertices" with the fill color
	cv2.fillPoly(mask, vertices, ignore_mask_color)

	# returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image


def draw_lines(img, lines, color=[255, 255, 255], thickness=10):
	"""
	NOTE: this is the function you might want to use as a starting point once you want to
	average/extrapolate the line segments you detect to map out the full
	extent of the lane (going from the result shown in raw-lines-example.mp4
	to that shown in P1_example.mp4).

	Think about things like separating line segments by their
	slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
	line vs. the right line.  Then, you can average the position of each of
	the lines and extrapolate to the top and bottom of the lane.

	This function draws `lines` with `color` and `thickness`.
	Lines are drawn on the image inplace (mutates the image).
	If you want to make the lines semi-transparent, think about combining
	this function with the weighted_img() function below
	"""
	# In case of error, don't draw the line(s)
	if lines is None:
		return
	if len(lines) == 0:
		return
	draw_right = True
	draw_left = True

	# Find slopes of all lines
	# But only care about lines where abs(slope) > slope_threshold
	slope_threshold = 0.5
	slopes = []
	new_lines = []
	for line in lines:
		x1, y1, x2, y2 = line[0]  # line = [[x1, y1, x2, y2]]

		# Calculate slope
		if x2 - x1 == 0.:  # corner case, avoiding division by 0
			slope = 999.  # practically infinite slope
		else:
			slope = (y2 - y1) / (x2 - x1)

		# Filter lines based on slope
		if abs(slope) > slope_threshold:
			slopes.append(slope)
			new_lines.append(line)

	lines = new_lines

	# Split lines into right_lines and left_lines, representing the right and left lane lines
	# Right/left lane lines must have positive/negative slope, and be on the right/left half of the image
	right_lines = []
	left_lines = []
	for i, line in enumerate(lines):
		x1, y1, x2, y2 = line[0]
		img_x_center = img.shape[1] / 2  # x coordinate of center of image
		if slopes[i] > 0 and x1 > img_x_center and x2 > img_x_center:
			right_lines.append(line)
		elif slopes[i] < 0 and x1 < img_x_center and x2 < img_x_center:
			left_lines.append(line)

	# Run linear regression to find best fit line for right and left lane lines
	# Right lane lines
	right_lines_x = []
	right_lines_y = []

	for line in right_lines:
		x1, y1, x2, y2 = line[0]

		right_lines_x.append(x1)
		right_lines_x.append(x2)

		right_lines_y.append(y1)
		right_lines_y.append(y2)

	if len(right_lines_x) > 0:
		right_m, right_b = np.polyfit(right_lines_x, right_lines_y, 1)  # y = m*x + b
	else:
		right_m, right_b = 1, 1
		draw_right = False

	# Left lane lines
	left_lines_x = []
	left_lines_y = []

	for line in left_lines:
		x1, y1, x2, y2 = line[0]

		left_lines_x.append(x1)
		left_lines_x.append(x2)

		left_lines_y.append(y1)
		left_lines_y.append(y2)

	if len(left_lines_x) > 0:
		left_m, left_b = np.polyfit(left_lines_x, left_lines_y, 1)  # y = m*x + b
	else:
		left_m, left_b = 1, 1
		draw_left = False

	# Find 2 end points for right and left lines, used for drawing the line
	# y = m*x + b --> x = (y - b)/m
	y1 = img.shape[0]
	y2 = img.shape[0] * (1 - trap_height)

	right_x1 = (y1 - right_b) / right_m
	right_x2 = (y2 - right_b) / right_m

	left_x1 = (y1 - left_b) / left_m
	left_x2 = (y2 - left_b) / left_m

	# Convert calculated end points from float to int
	y1 = int(y1)
	y2 = int(y2)
	right_x1 = int(right_x1)
	right_x2 = int(right_x2)
	left_x1 = int(left_x1)
	left_x2 = int(left_x2)

	# Draw the right and left lines on image
	if draw_right:
		cv2.line(img, (right_x1, y1), (right_x2, y2), color, thickness)
	if draw_left:
		cv2.line(img, (left_x1, y1), (left_x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
	"""
	`img` should be the output of a Canny transform.

	Returns an image with hough lines drawn.
	"""
	lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array(
	    []), minLineLength=min_line_len, maxLineGap=max_line_gap)
	line_img = np.zeros((*img.shape, 3), dtype=np.uint8)  # 3-channel RGB image
	draw_lines(line_img, lines)
	return line_img

# Python 3 has support for cool math symbols.


def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
	"""
	`img` is the output of the hough_lines(), An image with lines drawn on it.
	Should be a blank image (all black) with lines drawn on it.

	`initial_img` should be the image before any processing.

	The result image is computed as follows:

	initial_img * α + img * β + λ
	NOTE: initial_img and img must be the same shape!
	"""
	return cv2.addWeighted(initial_img, α, img, β, λ)


def filter_colors(image):
	"""
	Filter the image to include only yellow and white pixels
	"""
	# Filter white pixels
	white_threshold = 200  # 130
	lower_white = np.array([white_threshold, white_threshold, white_threshold])
	upper_white = np.array([255, 255, 255])
	white_mask = cv2.inRange(image, lower_white, upper_white)
	white_image = cv2.bitwise_and(image, image, mask=white_mask)

	# Filter yellow pixels
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	lower_yellow = np.array([90, 100, 100])
	upper_yellow = np.array([110, 255, 255])
	yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
	yellow_image = cv2.bitwise_and(image, image, mask=yellow_mask)

	# Combine the two above images
	image2 = cv2.addWeighted(white_image, 1., yellow_image, 1., 0.)
	return image2


def annotate_image_array(image_in):
	# Only keep white and yellow pixels in the image, all other pixels become black
    image = filter_colors(image_in)
    
	# Read in and grayscale the image
    gray = grayscale(image)
    
	# Apply Gaussian smoothing
    blur_gray = gaussian_blur(gray, kernel_size)
    

	# Apply Canny Edge Detector
    edges = canny(blur_gray, low_threshold, high_threshold)
    
    


	# Create masked edges using trapezoid-shaped region-of-interest
    imshape = image.shape
    vertices = np.array([[
		((imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0]),
		((imshape[1] * (1 - trap_top_width)) // 2,
		 imshape[0] - imshape[0] * trap_height),
		(imshape[1] - (imshape[1] * (1 - trap_top_width)) //
		 2, imshape[0] - imshape[0] * trap_height),
		(imshape[1] - (imshape[1] * (1 - trap_bottom_width)) // 2, imshape[0])]], dtype=np.int32)
    
    masked_edges = region_of_interest(edges, vertices)
    cv2.imshow("test",masked_edges)
	# Run Hough on edge detected image
    line_image = hough_lines(masked_edges, rho, theta,
	                         threshold, min_line_length, max_line_gap)
    
    # initial_image = image_in.astype('uint8')
    # annotated_image = weighted_img(line_image, initial_image)

    return line_image

def annotate_image(input_file):
    """ Given input_file image, save annotated image to output_file """
    annotated_image = annotate_image_array(input_file)
    # plt.imsave(output_file, annotated_image)
    return annotated_image






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
    if(steering_angle == 0):
        return 3, 1
    # if abs(steering_angle) < 0.025:
    #     return 2,2
    # else:
    # print((1/(steering_angle**2)) * 0.0000001)
    return 0.003 * (1/(3 * (steering_angle**2))), 1


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
    dst_size = 24
    source = np.float32([[220, 450],
                         [380, 450],
                         [377, 416],
                         [252, 416],
                         ])

    # Get the image and show it
    img = simulator.get_image()
    img2 = annotate_image(img)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    fps = fps_counter.get_fps()
    # image = img
    # destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0]],
    #                           [image.shape[1]/2 + dst_size, image.shape[0]],
    #                           [image.shape[1]/2 + dst_size,
    #                               image.shape[0] - 2*dst_size],
    #                           [image.shape[1]/2 - dst_size,
    #                               image.shape[0] - 2*dst_size],
    #                           ])
    # warped = perspect_transform(image, source, destination)
    # cv2.imshow("image", warped)

    cv2.waitKey(1)


    throttle = 0.17
    steering = 0
    steer_fact = 1.3
  
    cv2.imshow("image", img)
    
    if(steering is None):
        steering = 0.0

    
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
        steering * simulator.max_steer_angle / 1.8)
    # +(np.sign(steering)*0.05)
    simulator.set_car_steering((steering * simulator.max_steer_angle / fact))
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
