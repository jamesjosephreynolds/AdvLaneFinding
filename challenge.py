''' Threshold challenge, lesson 4.24'''
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('signs_vehicles_xygrad.png')

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient is 'x':
        sobel = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    abs_sobel = np.abs(sobel)
    scaled_sobel = abs_sobel*255/np.max(abs_sobel)
    grad_binary = np.zeros_like(scaled_sobel, dtype = np.uint8)
    grad_binary[(scaled_sobel > thresh[0])&(scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    mag_sobel = np.sqrt(np.square(sobelx)+np.square(sobely))
    scaled_sobel = mag_sobel*255/np.max(mag_sobel)
    mag_binary = np.zeros_like(scaled_sobel, dtype = np.uint8)
    mag_binary[(scaled_sobel > thresh[0])&(scaled_sobel <= thresh[1])] = 1
    return mag_binary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=sobel_kernel)
    sobely = cv2.Sobel(img_gray,cv2.CV_64F,0,1,ksize=sobel_kernel)
    dir_sobel = np.arctan2(np.abs(sobely),np.abs(sobelx))
    dir_binary = np.zeros_like(dir_sobel)
    dir_binary[(dir_sobel > thresh[0])&(dir_sobel <= thresh[1])] = 1
    return dir_binary

# Choose a Sobel kernel size
ksize = 7 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(50, 255))
grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(50, 255))
mag_binary = mag_thresh(img, sobel_kernel=ksize, thresh=(50, 255))
dir_binary = dir_threshold(img, sobel_kernel=ksize, thresh=(np.pi/4, np.pi/2))

# Combinations of techniques
combined = np.zeros_like(dir_binary)
combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

# Visualize
fig = plt.figure()
ax1 = fig.add_subplot(3,2,1)
ax1.imshow(img)
ax1.set_xlabel('original')
ax2 = fig.add_subplot(3,2,2)
ax2.imshow(gradx, cmap='gray')
ax2.set_xlabel('gradient in x')
ax3 = fig.add_subplot(3,2,3)
ax3.imshow(grady, cmap='gray')
ax3.set_xlabel('gradient in y')
ax4 = fig.add_subplot(3,2,4)
ax4.imshow(mag_binary, cmap='gray')
ax4.set_xlabel('magnitude of gradient')
ax5 = fig.add_subplot(3,2,5)
ax5.imshow(dir_binary, cmap='gray')
ax5.set_xlabel('direction of gradient')
ax6 = fig.add_subplot(3,2,6)
ax6.imshow(combined, cmap='gray')
ax6.set_xlabel('combination')

fig.set_tight_layout(True)
plt.show()
