import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def do(src, N = 10):
    # Divide the images in N horizontal slices and find the lane
    # lines in each slice

    rows, cols = src.shape[0], src.shape[1]

    # Basic geometry of the figure, slice up into sections
    slice_height = np.uint32(np.trunc(rows / N))
    slice_width = np.uint32(np.trunc(cols / 2))
    mid_height = np.uint32(np.trunc(rows / 2))

    # Sum up along the vertical direction, find a starting point to search
    hist = np.sum(src[mid_height:,:], axis=0)

    left_pts = []
    right_pts = []
    left_start = np.argmax(hist[:slice_width])
    right_start = np.argmax(hist[slice_width:])

    left_prev, right_prev = left_start, right_start
    for idx in range(N):
        top = rows - np.uint32((idx+1)*slice_height)
        bottom = rows - np.uint32((idx)*slice_height)
        slice_left = src[top:bottom, :slice_width]
        slice_right = src[top:bottom, slice_width:]
        print(slice_left.shape)
        print(slice_right.shape)


        ### UPDATE left_prev, right_prev!!!

    out_img = np.dstack((src, src, src))*255

    


    return #left_pts, rightp_ts, leftpoly, rightpoly
