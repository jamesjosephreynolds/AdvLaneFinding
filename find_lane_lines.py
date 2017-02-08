import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import calibrate_camera
import warp_and_unwarp
import color_and_gradient
import lane_find_files

def params():

    # Number of horizontal slices through which to search
    num_slices = 8

    # Width of each slice (right and left are same width)
    slice_width = 200

    # Minimum number of pixels needed to recenter slice
    min_pixels = 50

    return num_slices, slice_width, min_pixels

def do(src, N, width, min_pix):
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

    left_indices_all = []
    right_indices_all = []
    left_center = np.argmax(hist[:slice_width]) # horizontal center of the left search rectangle
    right_center = slice_width+np.argmax(hist[slice_width:]) # horizontal center of the right search rectangle
    dst = np.dstack((src, src, src))*255

    for idx in range(N):
        ## rectangle borders
        
        # top and bottom border of both left and right rectangles
        top = rows - np.uint32((idx+1)*slice_height)
        bottom = rows - np.uint32((idx)*slice_height)

        # left and right border of left rectangle
        left_left = left_center-np.uint32(width/2)
        left_right = left_center+np.uint32(width/2)

        # left and right border of right rectangle
        right_left = right_center-np.uint32(width/2)
        right_right = right_center+np.uint32(width/2)

        # draw rectangles
        cv2.rectangle(dst, (left_left,bottom),(left_right,top),(255,165,0), 2) # orange
        cv2.rectangle(dst, (right_left,bottom),(right_right,top),(0,165,1255), 2)

        # search rectangles for nonzero points
        left_indices = ((nonzero_row >= top) & (nonzero_row < bottom)
                          & (nonzero_col >= left_left) & (nonzero_col < left_right)).nonzero()[0]
        right_indices = ((nonzero_row >= top) & (nonzero_row < bottom)
                          & (nonzero_col >= right_left) & (nonzero_col < right_right)).nonzero()[0]

        if len(left_indices) >= min_pix:
            left_center = np.uint32(np.mean(nonzero_col[left_indices]))

        if len(right_indices) >= min_pix:
            right_center = np.uint32(np.mean(nonzero_col[right_indices]))

        left_indices_all.append(left_indices)
        right_indices_all.append(right_indices)

        plt.imshow(dst)

    left_indices_all = np.concatenate(left_indices_all)
    right_indices_all = np.concatenate(right_indices_all)

    left_poly = np.polyfit(nonzero_row[left_indices_all],nonzero_col[left_indices_all],2)
    right_poly = np.polyfit(nonzero_row[right_indices_all],nonzero_col[right_indices_all],2)

    dst[nonzero_row[left_indices_all], nonzero_col[left_indices_all]] = [255, 0, 0]
    dst[nonzero_row[right_indices_all], nonzero_col[right_indices_all]] = [0, 0, 255]

    return dst, left_poly, right_poly

if __name__ == '__main__':
    # Run the function on test images
    print('Running locally on test images, printing and saving allowed')
    fname_array = lane_find_files.cal_images()
    mtx, dist = calibrate_camera.do(fname_array)

    # image info
    fname_array = lane_find_files.test_images()
    sobel_thresh, s_thresh = color_and_gradient.params()
    poly1, poly2 = warp_and_unwarp.params()
    N, W, P = params()


    for fidx in range(len(fname_array)):
        src = mpimg.imread(fname_array[fidx], format = 'jpg')

        #undistort
        undist = cv2.undistort(src, mtx, dist)
        
        # apply color and gradient threshold
        threshed, stack = color_and_gradient.do(undist, sobel_thresh, s_thresh)
        
        # warp the image
        warped = warp_and_unwarp.warp(threshed, poly1, poly2)

        # find the lane lines

        dst3, _, _ = do(warped, N, W, P)
        
        plt.subplot(231),plt.imshow(src),plt.title('Input')
        plt.subplot(232),plt.imshow(undist, cmap = 'gray'),plt.title('Undistort')
        plt.subplot(233),plt.imshow(stack),plt.title('Stack Thresh')
        plt.subplot(234),plt.imshow(threshed, cmap = 'gray'),plt.title('Thresholded')
        plt.subplot(235),plt.imshow(warped, cmap = 'gray'),plt.title('Warped')
        plt.subplot(236),plt.imshow(dst3),plt.title('Lane Lines')
        
        filestr = 'output_images/find_lane_lines'+str(fidx)+'.png'
        plt.savefig(filestr,format='png')
        print('Created '+filestr)

    

else:
    # Run the function on an image argument
    pass
