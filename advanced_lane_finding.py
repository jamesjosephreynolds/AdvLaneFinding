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
fname_array = ['camera_cal/calibration1.jpg',
                 'camera_cal/calibration2.jpg',
                 'camera_cal/calibration3.jpg',
                 'camera_cal/calibration4.jpg',
                 'camera_cal/calibration5.jpg',
                 'camera_cal/calibration6.jpg',
                 'camera_cal/calibration7.jpg',
                 'camera_cal/calibration8.jpg',
                 'camera_cal/calibration9.jpg',
                 'camera_cal/calibration10.jpg',
                 'camera_cal/calibration11.jpg',
                 'camera_cal/calibration12.jpg',
                 'camera_cal/calibration13.jpg',
                 'camera_cal/calibration14.jpg',
                 'camera_cal/calibration15.jpg',
                 'camera_cal/calibration16.jpg',
                 'camera_cal/calibration17.jpg',
                 'camera_cal/calibration18.jpg',
                 'camera_cal/calibration19.jpg',
                 'camera_cal/calibration20.jpg']
mtx, dist = calibrate_camera.do(fname_array)
print('Calibration matrix M = '+str(mtx))
print('Distortion coefficents = '+str(dist))

## Test image
src = mpimg.imread('test_images/straight_lines2.jpg', format = 'jpg')
 
## Apply color and gradient thresholds
sobel_thresh = [20, 100]
s_thresh = [120, 255]
src1, stack = color_and_gradient.do(src, sobel_thresh, s_thresh)

'''plt.subplot(211),plt.imshow(src),plt.title('Input')
plt.subplot(212),plt.imshow(stack),plt.title('Stack')
plt.show()'''


## Undistort the image, and perspective warp
# image info
cols = np.float32(1280)
rows = np.float32(720)
horz = np.float32(450) # horizon y-coordinate
center = np.float32(cols/2) # horizontal center x-coordinate
tr_width = np.float32(200) # width of the trapezoid upper leg
s = np.float32(0.3) # slope of the trapezoid right leg (dy/dx)
    
p1 = [center-tr_width/2, horz] # upper left vertex
p4 = [center+tr_width/2, horz] # upper right vertex
p2 = [p1[0]-(rows-horz)/s, rows] # lower left vertex
p3 = [p4[0]+(rows-horz)/s, rows] # lower right vertex
poly = np.float32([p1,p2,p3,p4])

_, dst = undist_and_warp.do(src, mtx, dist, poly)
_, dst1 = undist_and_warp.do(src1, mtx, dist, poly)


plt.subplot(221),plt.imshow(src),plt.title('Input')
plt.subplot(222),plt.imshow(src1 ,cmap = 'gray'),plt.title('Thresh')
plt.subplot(223),plt.imshow(dst),plt.title('WarpUndistort')
plt.subplot(224),plt.imshow(dst1, cmap = 'gray'),plt.title('Composite')
plt.show()
'''
## Find the lane lines
find_lane_lines.do(dst1)'''
