

    # contours, hierarchy = cv2.findContours(warped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

    # contour_area = []

    # for c in contours:
    #     contour_area.append((cv2.contourArea(c), c))

    # contour_area = sorted(contour_area, key=lambda x:x[0], reverse=True)
    # image2 = np.zeros((height, width, 3), dtype = "uint8")

    # # draw them

    # coords1 = np.vstack([contour_area[0][1], contour_area[1][1]])

    # cv2.fillPoly(image2, [coords1], (255, 255, 255))

    # gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY) 

    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # coords = np.column_stack(np.where(thresh > 0))

    # c2=np.stack((coords[:,1], coords[:,0]), axis=-1)
    # bb=np.zeros_like(image)

    # cv2.fillPoly(bb, [c2], (255, 255, 255))