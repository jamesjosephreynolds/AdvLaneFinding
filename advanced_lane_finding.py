''' Project 4
    Advanced Lane Finding
    Main Pipeline '''

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import calibrate_camera
import warp_and_unwarp
import color_and_gradient
import find_lane_lines
import lane_find_files
import paint_lane

def find_lanes(undist, left_poly_old = [], right_poly_old = []):
    # Take in an undistorted image and the previous lane polynomials

    # Apply Sobel gradient threshold and HSL S-threshold
    sobel_thresh, s_thresh = color_and_gradient.params()
    threshed, _ = color_and_gradient.do(undist, sobel_thresh, s_thresh)

    ## Warp the perspective for straight line region
    poly1, poly2 = warp_and_unwarp.params()
    warped = warp_and_unwarp.warp(threshed, poly1, poly2)

    ## Find the lane lines (utilizing previous location if possible)
    N, W, P = find_lane_lines.params()
    lanes, left_poly, right_poly = find_lane_lines.do(warped, N, W, P)

    # Paint the lane lines onto the original image
    painted = paint_lane.do(undist, left_poly, right_poly, poly1, poly2)

    return dst, left_poly, right_poly

## Calibrate the camera
fname_array = lane_find_files.cal_images()
mtx, dist = calibrate_camera.do(fname_array)
print('Calibration matrix M = '+str(mtx))
print('Distortion coefficents = '+str(dist))

## Test image
src = mpimg.imread('test_images/straight_lines2.jpg', format = 'jpg')
undist = cv2.undistort(src, mtx, dist)




# Paint the lane on the image


plt.subplot(321),plt.imshow(src),plt.title('Input')
plt.subplot(322),plt.imshow(undist),plt.title('Undistorted')
plt.subplot(323),plt.imshow(threshed, cmap = 'gray'),plt.title('Thresholded')
plt.subplot(324),plt.imshow(stack, cmap = 'gray'),plt.title('Composite')
plt.subplot(325),plt.imshow(lanes),plt.title('Composite')
plt.subplot(326),plt.imshow(painted),plt.title('Painted')
plt.show()
