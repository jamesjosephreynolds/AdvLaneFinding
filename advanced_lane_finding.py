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
from moviepy.editor import VideoFileClip

def paint_lane(src, left_polynomial, right_polynomial):

    dst = np.zeros_like(src).astype(np.uint8)
    ploty = np.linspace(0, 719, num=720)
    left_fit = left_polynomial[0]*ploty**2 + left_polynomial[1]*ploty + left_polynomial[2]
    right_fit = right_polynomial[0]*ploty**2 + right_polynomial[1]*ploty + right_polynomial[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(dst, np.int_([pts]), (0,255, 0))

    return dst

def warp(src, M):

    rows, cols = src.shape[0], src.shape[1]
    dst = cv2.warpPerspective(src, M, (cols, rows))

    return dst

def unwarp(src, Minv):
    
    rows, cols = src.shape[0], src.shape[1]
    dst = cv2.warpPerspective(src, Minv, (cols, rows))

    return dst

def threshold(src, sobel_thresh, s_thresh):

    # Sobel value
    sobel_binary = SobelX(src, sobel_thresh)

    # S value
    color_binary = HlsGrad(src, s_thresh)
            
    stack= np.dstack((np.zeros_like(color_binary), 255*sobel_binary, 255*color_binary))
    dst = np.zeros_like(color_binary)
    dst[(sobel_binary == 1) | (color_binary == 1)] = 1

    return dst

def SobelX(src, sobel_thresh):
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel > sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1

    return sobel_binary

def HlsGrad(src, s_thresh):
    hls = cv2.cvtColor(src, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    color_binary = np.zeros_like(s)
    color_binary[(s > s_thresh[0]) & (s <= s_thresh[1])] = 1

    return color_binary

def find_lanes(image):
    # Take in an image and undistort

    # Calibrate the camera (ideally outside of loop)
    mtx, dist = find_lanes.mtx, find_lanes.dist
    undist = cv2.undistort(src, mtx, dist)

    # Apply Sobel gradient threshold and HSL S-threshold

    image = threshold(undist, find_lanes.sobel, find_lanes.s)

    ## Warp the perspective for straight line region

    image = warp(image, find_lanes.M)

    ## Find the lane lines (utilizing previous location if possible)

    lanes, left_polynomial, right_polynomial = find_lane_lines.do(image,
                                                                  find_lanes.N, find_lanes.W, find_lanes.P)

    # Paint the lane lines onto the undistorted image
    shadow = paint_lane(lanes, left_polynomial, right_polynomial)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwarped = unwarp(shadow, find_lanes.Minv)

    # Combine the result with the original image
    dst = cv2.addWeighted(undist, 1, unwarped, 0.3, 0)

    return dst

## Calibrate the camera
fname_array = lane_find_files.cal_images()
mtx, dist = calibrate_camera.do(fname_array)
print('Calibration matrix M = '+str(mtx))
print('Distortion coefficents = '+str(dist))
find_lanes.mtx = mtx
find_lanes.dist = dist
polygon1, polygon2 = warp_and_unwarp.params()
find_lanes.M = cv2.getPerspectiveTransform(polygon1, polygon2)
find_lanes.Minv = cv2.getPerspectiveTransform(polygon2, polygon1)
find_lanes.N, find_lanes.W, find_lanes.P = find_lane_lines.params()
find_lanes.sobel, find_lanes.s = color_and_gradient.params()


## Set test images
fname_array = lane_find_files.test_images()

for fidx in range(len(fname_array)):
        src = mpimg.imread(fname_array[fidx], format = 'jpg')

        # paint lane lines
        undist = cv2.undistort(src, mtx, dist)
        painted = find_lanes(src)

        plt.subplot(2,1,1),plt.imshow(undist),plt.title('Undistorted')
        plt.subplot(2,1,2),plt.imshow(painted),plt.title('Output')
        filestr = 'output_images/input_output'+str(fidx)+'.png'
        plt.savefig(filestr, format='png')


## Test videos
video_output = 'project_video_out.mp4'
clip1 = VideoFileClip("project_video.mp4")
video_clip = clip1.fl_image(find_lanes)
video_clip.write_videofile(video_output, audio=False)

video_output = 'challenge_video_out.mp4'
clip1 = VideoFileClip("challenge_video.mp4")
video_clip = clip1.fl_image(find_lanes)
video_clip.write_videofile(video_output, audio=False)

video_output = 'harder_challenge_video_out.mp4'
clip1 = VideoFileClip("harder_challenge_video.mp4")
video_clip = clip1.fl_image(find_lanes)
video_clip.write_videofile(video_output, audio=False)
