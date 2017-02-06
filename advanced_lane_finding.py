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

## Calibrate the camera
mtx, dist = calibrate_camera.do()
print('transformation matrix: '+str(mtx))
print('distortion coeffs: '+str(dist))

## Undistort the image, and perspective warp
cols = 1280
rows = 720
horz = 450
x1 = 575
x2 = cols-x1
poly = np.float32([[x1,horz],[0,rows],[cols,rows],[x2,horz]])
src = mpimg.imread('test_images/straight_lines1.jpg', format = 'jpg')
dst = undist_and_warp.do(src, mtx, dist, poly)
plt.subplot(121),plt.imshow(src),plt.title('Input')
plt.subplot(122),plt.imshow(dst),plt.title('Undistort')
plt.show()

## Apply color thresholding

