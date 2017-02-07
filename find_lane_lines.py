import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def do(src, N = 8, width = 100):
    # Divide the images in N horizontal slices and find the lane
    # lines in each slice

    rows, cols = src.shape[0], src.shape[1]

    # Basic geometry of the figure, slice up into sections
    slice_height = np.uint32(np.trunc(rows / N))
    slice_width = np.uint32(np.trunc(cols / 2))
    mid_height = np.uint32(np.trunc(rows / 2))

    # Find the lane pixels
    nonzero_row = np.array(src.nonzero()[0])
    nonzero_col = np.array(src.nonzero()[1])

    # Sum up along the vertical direction, find a starting point to search
    hist = np.sum(src[mid_height:,:], axis=0)

    left_pts = []
    right_pts = []
    left_start = np.argmax(hist[:slice_width])
    right_start = slice_width+np.argmax(hist[slice_width:])
    print(right_start)
    print(left_start)
    out_img = np.dstack((src, src, src))*255

    left_prev, right_prev = left_start, right_start
    for idx in range(N):
        top = rows - np.uint32((idx+1)*slice_height)
        bottom = rows - np.uint32((idx)*slice_height)
        left_left = left_start-np.uint32(width/2)
        left_right = left_start+np.uint32(width/2)
        right_left = right_start-np.uint32(width/2)
        right_right = right_start+np.uint32(width/2)
        mid = np.uint32(cols/2)
        slice_left = src[top:bottom, :slice_width]
        slice_right = src[top:bottom, slice_width:]
        cv2.rectangle(out_img,(left_left,bottom),(left_right,top),(0,255,0), 2)
        cv2.rectangle(out_img,(right_left,bottom),(right_right,top),(0,127,127), 2)
        plt.imshow(out_img)

    plt.show()


        ### UPDATE left_prev, right_prev!!!
   


    return #left_pts, rightp_ts, leftpoly, rightpoly
