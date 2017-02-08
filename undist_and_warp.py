import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import calibrate_camera
 
def do(src, mtx, dist, poly, suppress = True):
    if suppress is False:
        print('Undistorting images')

    # undistort the original image
    dst1 = cv2.undistort(src, mtx, dist, None, mtx)

    # perspective transform
    rows, cols = src.shape[0], src.shape[1]
    out_poly = np.float32([[0,0],[0,rows],[cols,rows],[cols,0]])
    M = cv2.getPerspectiveTransform(poly,out_poly)
    dst2 = cv2.warpPerspective(dst1,M,(cols, rows))

    return dst1, dst2

if __name__ == '__main__':
    # Run the undistort and warp function on test images
    print('Running locally on test images, printing and saving allowed')
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
    mtx, dist = calibrate_camera.do(fname_array)

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
    poly = np.float32([p1,p2,p3,p4])

    # load test images
    fname_array = ['test_images/straight_lines1.jpg',
                 'test_images/straight_lines2.jpg',
                 'test_images/test1.jpg',
                 'test_images/test2.jpg',
                 'test_images/test3.jpg',
                 'test_images/test4.jpg',
                 'test_images/test5.jpg',
                 'test_images/test6.jpg']
    for fidx in range(len(fname_array)):
        src = mpimg.imread(fname_array[fidx], format = 'jpg')
        rows, cols = src.shape[0], src.shape[1]
        # undistort the image
        undst, dst = do(src, mtx, dist, poly, suppress = False)
        plt.subplot(131),plt.imshow(src),plt.title('Input')
        plt.subplot(132),plt.imshow(undst),plt.title('Undistort')
        plt.subplot(133),plt.imshow(dst),plt.title('Output')
        filestr = 'output_images/undist_and_warp'+str(fidx)+'.png'
        plt.savefig(filestr,format='png')
        print('Created '+filestr)
    
else:
    # Run the undistort and warp function on image argument
    pass
