import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import calibrate_camera
import lane_find_files

def params():
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
 
def warp(src, poly1, poly2, suppress = True):
    if suppress is False:
        print('Warping image')

    rows, cols = src.shape[0], src.shape[1]
    M = cv2.getPerspectiveTransform(poly1, poly2)
    dst = cv2.warpPerspective(src, M, (cols, rows))

    return dst

def unwarp(src, poly1, poly2, suppress = True):
    if suppress is False:
        print('Unwarping image')

    rows, cols = src.shape[0], src.shape[1]
    M = cv2.getPerspectiveTransform(poly2, poly1)
    dst = cv2.warpPerspective(src, M, (cols, rows))

    return dst

if __name__ == '__main__':
    # Run the undistort and warp function on test images
    print('Running locally on test images, printing and saving allowed')
    fname_array = lane_find_files.cal_images()
    mtx, dist = calibrate_camera.do(fname_array)

    
    fname_array = lane_find_files.test_images()
    poly1, poly2 = params()
    
    for fidx in range(len(fname_array)):
        src = mpimg.imread(fname_array[fidx], format = 'jpg')
        rows, cols = src.shape[0], src.shape[1]
        # undistort the image
        undist = cv2.undistort(src, mtx, dist)
        warped = warp(undist, poly1, poly2)
        plt.subplot(311),plt.imshow(src),plt.title('Input')
        plt.subplot(312),plt.imshow(undist),plt.title('Undistort')
        plt.subplot(313),plt.imshow(warped),plt.title('Output')
        filestr = 'output_images/undist_and_warp'+str(fidx)+'.png'
        plt.show()
        plt.savefig(filestr,format='png')
        print('Created '+filestr)
    
else:
    # Run the undistort and warp function on image argument
    pass
