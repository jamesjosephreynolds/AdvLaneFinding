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

def find_lanes(image):
    # Take in an image and undistort

    # Calibrate the camera (ideally outside of loop)
    mtx, dist = find_lanes.mtx, find_lanes.dist
    undist = cv2.undistort(src, mtx, dist)

    # Apply Sobel gradient threshold and HSL S-threshold
    sobel_thresh, s_thresh = color_and_gradient.params()
    image, _ = color_and_gradient.do(undist, sobel_thresh, s_thresh)

    ## Warp the perspective for straight line region

    image = warp_and_unwarp.warp(image, find_lanes.polygon1, find_lanes.polygon2)

    ## Find the lane lines (utilizing previous location if possible)

    lanes, left_polynomial, right_polynomial = find_lane_lines.do(image,
                                                                  find_lanes.N, find_lanes.W, find_lanes.P)

    # Paint the lane lines onto the undistorted image
    dst = paint_lane.do(image, left_polynomial, right_polynomial,
                        find_lanes.polygon1, find_lanes.polygon2)

    return dst

## Calibrate the camera
fname_array = lane_find_files.cal_images()
mtx, dist = calibrate_camera.do(fname_array)
print('Calibration matrix M = '+str(mtx))
print('Distortion coefficents = '+str(dist))
find_lanes.mtx = mtx
find_lanes.dist = dist
find_lanes.polygon1, find_lanes.polygon2 = warp_and_unwarp.params()
find_lanes.N, find_lanes.W, find_lanes.P = find_lane_lines.params()

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
