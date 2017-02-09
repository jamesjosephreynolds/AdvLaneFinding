import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import calibrate_camera
import lane_find_files
import warp_and_unwarp
import find_lane_lines
import color_and_gradient

def do(src, left_poly, right_poly, poly1, poly2):

    empty = np.zeros_like(src).astype(np.uint8)
    ploty = np.linspace(0, 719, num=720)
    left_fit = left_poly[0]*ploty**2 + left_poly[1]*ploty + left_poly[2]
    right_fit = right_poly[0]*ploty**2 + right_poly[1]*ploty + right_poly[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(empty, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    warped = warp_and_unwarp.unwarp(empty, poly1, poly2)
    # Combine the result with the original image
    dst = cv2.addWeighted(src, 1, warped, 0.3, 0)

    return dst

if __name__ == '__main__':

    fname_array = lane_find_files.cal_images()
    mtx, dist = calibrate_camera.do(fname_array)

    sobel_thresh, s_thresh = color_and_gradient.params()

    N, W, P = find_lane_lines.params()
    fname_array = lane_find_files.test_images()
    

    for fidx in range(len(fname_array)):
        src = mpimg.imread(fname_array[fidx], format = 'jpg')

        # undistort
        undist = cv2.undistort(src, mtx, dist)

        # threshold and stack images
        threshed, stack = color_and_gradient.do(undist, sobel_thresh, s_thresh)
        poly1, poly2 = warp_and_unwarp.params()
        warped = warp_and_unwarp.warp(threshed, poly1, poly2)
        lines, left_poly, right_poly = find_lane_lines.do(warped, N, W, P)

        painted = do(undist, left_poly, right_poly, poly1, poly2)
        plt.imshow(painted)
        plt.show()

else:
    # run on an argument
    pass
