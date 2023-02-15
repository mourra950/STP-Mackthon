white = cv2.bitwise_not(np.zeros_like(img))
    circle = cv2.circle(white,center=(480,320),radius=240)
    circle = thresholding(circle)