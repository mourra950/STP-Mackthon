 try:
        left_fit, right_fit = fit_from_lines(left_fit, right_fit, img_w)
        mov_avg_left = np.append(mov_avg_left,np.array([left_fit]), axis=0)
        mov_avg_right = np.append(mov_avg_right,np.array([right_fit]), axis=0)
    except Exception:
        print("sliding window")
        left_fit, right_fit = sliding_windown(img_w)
        mov_avg_left = np.array([left_fit])
        mov_avg_right = np.array([right_fit])
        
    left_fit = np.array([np.mean(mov_avg_left[::-1][:,0][0:MOV_AVG_LENGTH]),
                        np.mean(mov_avg_left[::-1][:,1][0:MOV_AVG_LENGTH]),
                        np.mean(mov_avg_left[::-1][:,2][0:MOV_AVG_LENGTH])])
    right_fit = np.array([np.mean(mov_avg_right[::-1][:,0][0:MOV_AVG_LENGTH]),
                         np.mean(mov_avg_right[::-1][:,1][0:MOV_AVG_LENGTH]),
                         np.mean(mov_avg_right[::-1][:,2][0:MOV_AVG_LENGTH])])
    if mov_avg_left.shape[0] > 1000:
        mov_avg_left = mov_avg_left[0:MOV_AVG_LENGTH]
    if mov_avg_right.shape[0] > 1000:
        mov_avg_right = mov_avg_right[0:MOV_AVG_LENGTH]
        
    final,degrees = draw_lines(image, img_w, left_fit, right_fit, perspective=[src,dst])
    
    cv2.imshow('final', final)
    
    #----------------------------------Steering_GUI-------------------------------------
    # if degrees > 35:
    #     degrees=35
    # elif degrees < -35:
    #     degrees=-35   
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    degrees=round(smoothed_angle)