''' Project 4
    Advanced Lane Finding
    Main Pipeline '''

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
import time

class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #x location of the first window center form the previous run
        self.center = []

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
    # Given an image and a polynomial best fit for each edge
    # superimpose the lane over the image

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
    # given an undistorted image, create a bird's eye transformation
    # of the area in front of the car

    rows, cols = src.shape[0], src.shape[1]
    dst = cv2.warpPerspective(src, warp.M, (cols, rows))

    return dst

def unwarp(src):
    # given a bird's eye view of the area in front of the car,
    # create a perspective transformation (natural view)
    
    rows, cols = src.shape[0], src.shape[1]
    dst = cv2.warpPerspective(src, unwarp.Minv, (cols, rows))

    return dst

def warp_and_unwarp_params():
    # Constant parameters for warp and unwarp

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
    # Apply both Sobel (X) and HSL (S) thresholds
    # on an image, take union of the two results

    # Sobel value
    sobel_binary = SobelX(src)

    # S value
    s_binary = HlsGrad(src)

    # yellow and white values
    color_binary = ColorFilt(src)
            
    stack= np.dstack((255*color_binary, 255*sobel_binary, 255*s_binary))

    return stack

def SobelX(src):
    # Apply a Sobel gradient to an image
    # keep only the pixels that lie within the thresholds
    
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel > SobelX.thresh[0]) & (scaled_sobel <= SobelX.thresh[1])] = 1

    return sobel_binary

def HlsGrad(src):
    # Apply an HLS color transformation on an image
    # keep only the pixels that lie within the thresholds in the S plane
    
    hls = cv2.cvtColor(src, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    color_binary = np.zeros_like(s)
    color_binary[(s > HlsGrad.s_thresh[0]) & (s <= HlsGrad.s_thresh[1])] = 1

    return color_binary

def ColorFilt(src):
    # Keep only the pixels that lie within the thresholds near yellow and white
    
    color_binary = np.zeros_like(src[:,:,2])
    color_binary[(((src[:,:,0] > ColorFilt.yellow[0][0]) & (src[:,:,0] < ColorFilt.yellow[0][1]))
                 &((src[:,:,1] > ColorFilt.yellow[1][0]) & (src[:,:,1] < ColorFilt.yellow[1][1]))
                 &((src[:,:,2] > ColorFilt.yellow[2][0]) & (src[:,:,2] < ColorFilt.yellow[2][1])))
                 |(((src[:,:,0] > ColorFilt.white[0][0]) & (src[:,:,0] < ColorFilt.white[0][1]))
                 &((src[:,:,1] > ColorFilt.white[1][0]) & (src[:,:,1] < ColorFilt.white[1][1]))
                 &((src[:,:,2] > ColorFilt.white[2][0]) & (src[:,:,2] < ColorFilt.white[2][1])))] = 1

    return color_binary

def find_lines(src):
    # Divide the images in N horizontal slices and find the lane
    # lines in each slice

    src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

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
    if LeftLine.center == []:
        left_center = np.argmax(hist[:slice_width]) # horizontal center of the left search rectangle
    else:
        window_left = LeftLine.center-np.uint32(find_lines.width/2)
        window_right = LeftLine.center+np.uint32(find_lines.width/2)
        left_center = window_left+np.argmax(hist[window_left:window_right])
    LeftLine.center = np.uint32(left_center)

    if RightLine.center == []:
        right_center = slice_width+np.argmax(hist[slice_width:]) # horizontal center of the right search rectangle
    else:
        window_left = RightLine.center-np.uint32(find_lines.width/2)
        window_right = RightLine.center+np.uint32(find_lines.width/2)
        right_center = window_left+np.argmax(hist[window_left:window_right])
    RightLine.center = np.uint32(right_center)
    
    
    dst = np.dstack((src, src, src))*255

    for idx in range(find_lines.num):
        ## rectangle borders
        
        # top and bottom border of both left and right rectangles
        top = rows - np.uint32((idx+1)*slice_height)
        bottom = rows - np.uint32((idx)*slice_height)

        # left and right border of left rectangle
        if left_center <= (find_lines.width/2):
            left_left = 0
        else:
            left_left = left_center-np.uint32(find_lines.width/2)
            
        left_right = left_center+np.uint32(find_lines.width/2)

        # left and right border of right rectangle
        right_left = right_center-np.uint32(find_lines.width/2)
        if right_center >= cols-np.uint32(find_lines.width/2):
            right_right = cols
        else:
            right_right = right_center+np.uint32(find_lines.width/2)

        # draw rectangles
        cv2.rectangle(dst, (left_left,bottom),(left_right,top),(255,165,0), 2) # orange
        cv2.rectangle(dst, (right_left,bottom),(right_right,top),(0,165,1255), 2)

        # search rectangles for nonzero points
        left_indices = ((nonzero_row >= top) & (nonzero_row < bottom)
                          & (nonzero_col >= left_left) & (nonzero_col < left_right)).nonzero()[0]
        right_indices = ((nonzero_row >= top) & (nonzero_row < bottom)
                          & (nonzero_col >= right_left) & (nonzero_col < right_right)).nonzero()[0]

        # Update search reach region for the next rectangle
        # if many pixels suggest to do so
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

def composite_image(img1, img2, img3, img4):
    # show 4 images on the same plot for tuning analysis
    dst = np.zeros_like(img1, dtype = np.uint8)
    dst[0:360,0:640] = cv2.resize(img1, (640,360))
    dst[0:360,640:1281] = cv2.resize(img2, (640,360))
    dst[360:721,0:640] = cv2.resize(img3, (640,360))
    dst[360:721,640:1281] = cv2.resize(img4, (640,360))

    return dst

def find_lanes(image):
    # Main function
    t0 = time.time()

    # Undistort the image
    undist = cv2.undistort(image, find_lanes.mtx, find_lanes.dist)

    # Apply Sobel gradient threshold and HSL S-threshold
    stack = threshold(undist)

    # Warp the perspective to bird's eye
    image = warp(stack)

    # Find the lane lines (Note: need to implement previous result improvement and filtering!)
    lanes, left_polynomial, right_polynomial = find_lines(image)

    # Paint the lane lines onto the blank bird's eye image
    shadow = paint_lane(lanes, left_polynomial, right_polynomial)

    # Warp the bird's eye lane to original image perspective using inverse perspective matrix (Minv)
    unwarped = unwarp(shadow)

    # Overlay the lane estimate onto the original image
    dst = cv2.addWeighted(undist, 1, unwarped, 0.3, 0)

    if find_lanes.composite is True:
        # combine images
        dst = composite_image(undist, stack, lanes, dst)

    return dst

## Choose return type:
# True = 4 image composite
# False = Output only
find_lanes.composite = True

## Calibrate the camera ##
fname_array = cal_images()
mtx, dist = calibrate_camera(fname_array)
find_lanes.mtx = mtx
find_lanes.dist = dist

## Create global variables
LeftLine = Line()
RightLine = Line()

## Load parameters ##
polygon1, polygon2 = warp_and_unwarp_params()
# perspective transform matrices
warp.M = cv2.getPerspectiveTransform(polygon1, polygon2)
unwarp.Minv = cv2.getPerspectiveTransform(polygon2, polygon1)
# lane finder params: number of slices, width of search region, number of pixels
find_lines.num, find_lines.width, find_lines.min = 8, 160, 50
# Sobel and HLS thresholds
SobelX.thresh, HlsGrad.s_thresh = [30, 80], [90, 255]
# Yellow and white thresholds
ColorFilt.yellow, ColorFilt.white = [[215, 255], [160, 255], [0, 160]],[[225, 255], [225, 255], [225, 255]]

## Show camera calibration test images
for fidx in range(len(fname_array)):
    print('Saving camera calibration chessboard samples')
    # Load image
    src = mpimg.imread(fname_array[fidx], format = 'jpg')

    # Two images (original, and undistorted)
    undist = cv2.undistort(src, mtx, dist)

    # Save images
    plt.subplot(2,1,1),plt.imshow(src),plt.title('Original')
    plt.subplot(2,1,2),plt.imshow(undist),plt.title('Undistorted')
    filestr = 'output_images/undistorted_chessboard'+str(fidx)+'.png'
    plt.savefig(filestr, format='png')

fname_array = test_images()

## Show camera calibration test images
for fidx in range(len(fname_array)):
    print('Saving camera calibration road samples')
    # Load image
    src = mpimg.imread(fname_array[fidx], format = 'jpg')

    # Two images (original, and undistorted)
    undist = cv2.undistort(src, mtx, dist)

    # Save images
    plt.subplot(1,2,1),plt.imshow(src),plt.title('Original')
    plt.subplot(1,2,2),plt.imshow(undist),plt.title('Undistorted')
    filestr = 'output_images/undistorted_road'+str(fidx)+'.png'
    plt.savefig(filestr, format='png')

## Show thresholded test images
for fidx in range(len(fname_array)):
    print('Saving thresholded road samples')
    # Load image
    src = mpimg.imread(fname_array[fidx], format = 'jpg')

    # Two images (original, and undistorted)
    undist = cv2.undistort(src, mtx, dist)
    stacked = threshold(undist)

    # Save images
    plt.subplot(1,2,1),plt.imshow(src),plt.title('Original')
    plt.subplot(1,2,2),plt.imshow(stacked),plt.title('Thresholded')
    filestr = 'output_images/thresholded_road'+str(fidx)+'.png'
    plt.savefig(filestr, format='png')

for fidx in range(len(fname_array)):
    # Load image
    src = mpimg.imread(fname_array[fidx], format = 'jpg')

    # Two images (undistorted original, and undistorted painted)
    undist = cv2.undistort(src, mtx, dist)
    painted = find_lanes(src)

    # Compare undistorted image to fully processed (with lane) image
    plt.subplot(1,2,1),plt.imshow(undist),plt.title('Undistorted')
    plt.subplot(1,2,2),plt.imshow(painted),plt.title('Output')
    filestr = 'output_images/final'+str(fidx)+'.png'
    plt.savefig(filestr, format='png')

## Test videos
'''find_lanes.composite = True
print('Starting video 1')
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
video_clip.write_videofile(video_output, audio=False)'''
