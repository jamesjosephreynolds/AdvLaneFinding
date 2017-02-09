''' Project 4
    Advanced Lane Finding
    Main Pipeline '''

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import calibrate_camera
import lane_find_files
from moviepy.editor import VideoFileClip

def cal_images():
    # Images for camera calibration
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

    return fname_array

def test_images():
    # Images for testing overall pipeline
    fname_array = ['test_images/straight_lines1.jpg',
                 'test_images/straight_lines2.jpg',
                 'test_images/test1.jpg',
                 'test_images/test2.jpg',
                 'test_images/test3.jpg',
                 'test_images/test4.jpg',
                 'test_images/test5.jpg',
                 'test_images/test6.jpg']

    return fname_array

def calibrate_camera(fname_array):
    '''fname_array is a list of strings specifying image filenames'''
    # load calibration images
    img = mpimg.imread(fname_array[0], format = 'jpg')
    rows, cols = img.shape[0], img.shape[1]
    img_array = np.zeros((len(fname_array),rows,cols,3),dtype = np.uint8)


    # load each image file into an array
    for fidx in range(len(fname_array)):
            
        img = mpimg.imread(fname_array[fidx], format = 'jpg')
        img_array[fidx] = img[0:rows,0:cols] #one image is incorrectly sized, crop

    # find corners
    objpoints = []
    imgpoints = []

    for fidx in range(len(fname_array)):

        gray = cv2.cvtColor(img_array[fidx], cv2.COLOR_RGB2GRAY)

        # some calibration images are missing corners, try various
        # dimensions to maximize calibration image utilization

        # 9x6 corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        if ret is True:
            objp = np.zeros((9*6,3), np.float32)
            objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
            imgpoints.append(corners)
            objpoints.append(objp)

        else:
            
            # 9x5 corners
            ret, corners = cv2.findChessboardCorners(gray, (9,5), None)
            if ret is True:
                objp = np.zeros((9*5,3), np.float32)
                objp[:,:2] = np.mgrid[0:9,0:5].T.reshape(-1,2)
                imgpoints.append(corners)
                objpoints.append(objp)

            else:

                # 8x6 corners
                ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
                if ret is True:
                    objp = np.zeros((8*6,3), np.float32)
                    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
                    imgpoints.append(corners)
                    objpoints.append(objp)

                else:

                    # 8x5 corners
                    ret, corners = cv2.findChessboardCorners(gray, (8,5), None)
                    if ret is True:
                        objp = np.zeros((8*5,3), np.float32)
                        objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1,2)
                        imgpoints.append(corners)
                        objpoints.append(objp)

                    else:

                        # 9x4 corners
                        ret, corners = cv2.findChessboardCorners(gray, (9,4), None)
                        if ret is True:
                            objp = np.zeros((9*4,3), np.float32)
                            objp[:,:2] = np.mgrid[0:9,0:4].T.reshape(-1,2)
                            imgpoints.append(corners)
                            objpoints.append(objp)

                        else:

                            # 7x6 corners
                            ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
                            if ret is True:
                                objp = np.zeros((7*6,3), np.float32)
                                objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
                                imgpoints.append(corners)
                                objpoints.append(objp)

                            else:

                                # 7x5 corners
                                ret, corners = cv2.findChessboardCorners(gray, (7,5), None)
                                if ret is True:
                                    objp = np.zeros((7*5,3), np.float32)
                                    objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)
                                    imgpoints.append(corners)
                                    objpoints.append(objp)
                                    
                                else:
                                    # corners not found
                                    pass

    # use OpenCV to get calibrate data
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    return mtx, dist

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

def warp(src):

    rows, cols = src.shape[0], src.shape[1]
    dst = cv2.warpPerspective(src, warp.M, (cols, rows))

    return dst

def unwarp(src):
    
    rows, cols = src.shape[0], src.shape[1]
    dst = cv2.warpPerspective(src, unwarp.Minv, (cols, rows))

    return dst

def warp_and_unwarp_params():
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

    # warp polygon
    poly1 = np.float32([p1,p2,p3,p4])

    # result polygon (image border)
    poly2 = np.float32([[0,0],[0,rows],[cols,rows],[cols,0]])

    return poly1, poly2

def threshold(src):

    # Sobel value
    sobel_binary = SobelX(src)

    # S value
    color_binary = HlsGrad(src)
            
    stack= np.dstack((np.zeros_like(color_binary), 255*sobel_binary, 255*color_binary))
    dst = np.zeros_like(color_binary)
    dst[(sobel_binary == 1) | (color_binary == 1)] = 1

    return dst

def SobelX(src):
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel > SobelX.thresh[0]) & (scaled_sobel <= SobelX.thresh[1])] = 1

    return sobel_binary

def HlsGrad(src):
    hls = cv2.cvtColor(src, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    color_binary = np.zeros_like(s)
    color_binary[(s > HlsGrad.s_thresh[0]) & (s <= HlsGrad.s_thresh[1])] = 1

    return color_binary

def find_lines(src):
    # Divide the images in N horizontal slices and find the lane
    # lines in each slice

    rows, cols = src.shape[0], src.shape[1]

    # Basic geometry of the figure, slice up into sections
    slice_height = np.uint32(np.trunc(rows / find_lines.num))
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

    for idx in range(find_lines.num):
        ## rectangle borders
        
        # top and bottom border of both left and right rectangles
        top = rows - np.uint32((idx+1)*slice_height)
        bottom = rows - np.uint32((idx)*slice_height)

        # left and right border of left rectangle
        left_left = left_center-np.uint32(find_lines.width/2)
        left_right = left_center+np.uint32(find_lines.width/2)

        # left and right border of right rectangle
        right_left = right_center-np.uint32(find_lines.width/2)
        right_right = right_center+np.uint32(find_lines.width/2)

        # draw rectangles
        cv2.rectangle(dst, (left_left,bottom),(left_right,top),(255,165,0), 2) # orange
        cv2.rectangle(dst, (right_left,bottom),(right_right,top),(0,165,1255), 2)

        # search rectangles for nonzero points
        left_indices = ((nonzero_row >= top) & (nonzero_row < bottom)
                          & (nonzero_col >= left_left) & (nonzero_col < left_right)).nonzero()[0]
        right_indices = ((nonzero_row >= top) & (nonzero_row < bottom)
                          & (nonzero_col >= right_left) & (nonzero_col < right_right)).nonzero()[0]

        if len(left_indices) >= find_lines.min:
            left_center = np.uint32(np.mean(nonzero_col[left_indices]))

        if len(right_indices) >= find_lines.min:
            right_center = np.uint32(np.mean(nonzero_col[right_indices]))

        left_indices_all.append(left_indices)
        right_indices_all.append(right_indices)

    left_indices_all = np.concatenate(left_indices_all)
    right_indices_all = np.concatenate(right_indices_all)

    left_poly = np.polyfit(nonzero_row[left_indices_all],nonzero_col[left_indices_all],2)
    right_poly = np.polyfit(nonzero_row[right_indices_all],nonzero_col[right_indices_all],2)

    dst[nonzero_row[left_indices_all], nonzero_col[left_indices_all]] = [255, 0, 0]
    dst[nonzero_row[right_indices_all], nonzero_col[right_indices_all]] = [0, 0, 255]

    return dst, left_poly, right_poly

def find_lanes(image):
    # Take in an image and undistort

    # Calibrate the camera (ideally outside of loop)
    undist = cv2.undistort(image, find_lanes.mtx, find_lanes.dist)

    # Apply Sobel gradient threshold and HSL S-threshold

    image = threshold(undist)

    ## Warp the perspective for straight line region

    image = warp(image)

    ## Find the lane lines (utilizing previous location if possible)

    lanes, left_polynomial, right_polynomial = find_lines(image)

    # Paint the lane lines onto the undistorted image
    shadow = paint_lane(lanes, left_polynomial, right_polynomial)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    unwarped = unwarp(shadow)

    # Combine the result with the original image
    dst = cv2.addWeighted(undist, 1, unwarped, 0.3, 0)

    return dst

## Calibrate the camera
fname_array = cal_images()
mtx, dist = calibrate_camera(fname_array)
print(mtx), print(dist)
find_lanes.mtx = mtx
find_lanes.dist = dist
polygon1, polygon2 = warp_and_unwarp_params()
print(polygon1), print(polygon2)
warp.M = cv2.getPerspectiveTransform(polygon1, polygon2)
unwarp.Minv = cv2.getPerspectiveTransform(polygon2, polygon1)
print(warp.M), print(unwarp.Minv)
find_lines.num, find_lines.width, find_lines.min = 8, 200, 50
SobelX.thresh, HlsGrad.s_thresh = [20, 100], [170, 255]

## Set test images
fname_array = test_images()

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
print('Starting video 1')
video_output = 'project_video_out.mp4'
clip1 = VideoFileClip("project_video.mp4")
video_clip = clip1.fl_image(find_lanes)
video_clip.write_videofile(video_output, audio=False)
'''
video_output = 'challenge_video_out.mp4'
clip1 = VideoFileClip("challenge_video.mp4")
video_clip = clip1.fl_image(find_lanes)
video_clip.write_videofile(video_output, audio=False)

video_output = 'harder_challenge_video_out.mp4'
clip1 = VideoFileClip("harder_challenge_video.mp4")
video_clip = clip1.fl_image(find_lanes)
video_clip.write_videofile(video_output, audio=False)'''
