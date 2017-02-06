import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import calibrate_camera

def do_wrapper():
    mtx, dist = calibrate_camera.do()
    cols = 1280
    rows = 720
    horz = 450
    x1 = 575
    x2 = cols-x1
    poly = np.float32([[x1,horz],[0,rows],[cols,rows],[x2,horz]])
    src = []
    dst = do(src, mtx, dist, poly, suppress = False)

def do(src, mtx, dist, poly, suppress = True):
    if suppress is False:
        print('Undistorting images')

        # load test images
        fname_array = ['test_images/straight_lines1.jpg',
                 'test_images/straight_lines2.jpg',
                 'test_images/test1.jpg',
                 'test_images/test2.jpg',
                 'test_images/test3.jpg',
                 'test_images/test4.jpg',
                 'test_images/test5.jpg',
                 'test_images/test6.jpg']

    if suppress is False:
        for fidx in range(len(fname_array)):
            src = mpimg.imread(fname_array[fidx], format = 'jpg')
            rows, cols = src.shape[0], src.shape[1]
            # undistort the image
            dst1 = cv2.undistort(src, mtx, dist, None, mtx)
            out_poly = np.float32([[0,0],[0,rows],[cols,rows],[cols,0]])
            M = cv2.getPerspectiveTransform(poly,out_poly)
            dst2 = cv2.warpPerspective(dst1,M,(cols, rows))
            plt.subplot(131),plt.imshow(src),plt.title('Input')
            plt.subplot(132),plt.imshow(dst1),plt.title('Undistort')
            plt.subplot(133),plt.imshow(dst2),plt.title('Output')
            filestr = 'output_images/undist_and_warp'+str(fidx)+'.png'
            plt.savefig(filestr,format='png')
            print('Created '+filestr)

    else:
        rows, cols = src.shape[0], src.shape[1]
        # undistort the image
        dst1 = cv2.undistort(src, mtx, dist, None, mtx)
        out_poly = np.float32([[0,0],[0,rows],[cols,rows],[cols,0]])
        M = cv2.getPerspectiveTransform(poly,out_poly)
        dst2 = cv2.warpPerspective(dst1,M,(cols, rows))

    return dst2

if __name__ == '__main__':
    # Run the undistort and warp function on test images
    print('Running locally on test images, printing and saving allowed')
    do_wrapper()
    
else:
    # Run the undistort and warp function on image argument
    pass
