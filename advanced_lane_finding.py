''' Project 4
    Advanced Lane Finding
    Main Pipeline '''

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import calibrate_camera
import undist_and_warp
import color_and_gradient
import find_lane_lines

## Calibrate the camera
mtx, dist = calibrate_camera.do()

## Test image
src = mpimg.imread('test_images/straight_lines2.jpg', format = 'jpg')

## Apply color and gradient thresholds
sobel_thresh = [20, 100]
s_thresh = [120, 255]
src1 = color_and_gradient.do(src, sobel_thresh, s_thresh)

## Undistort the image, and perspective warp
cols = 1280
rows = 720
horz = 450
x1 = 575
x2 = cols-x1
poly = np.float32([[x1,horz],[0,rows],[cols,rows],[x2,horz]])

dst = undist_and_warp.do(src, mtx, dist, poly)
dst1 = undist_and_warp.do(src1, mtx, dist, poly)


plt.subplot(221),plt.imshow(src),plt.title('Input')
plt.subplot(222),plt.imshow(src1 ,cmap = 'gray'),plt.title('Thresh')
plt.subplot(223),plt.imshow(dst),plt.title('WarpUndistort')
plt.subplot(224),plt.imshow(dst1, cmap = 'gray'),plt.title('Composite')
plt.show()

## Find the lane lines
find_lane_lines.do(dst1)
