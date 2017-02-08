import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import lane_find_files

def do(fname_array, suppress = True):
    '''fname_array is a list of strings specifying image filenames'''
    
    if suppress is False:
        print('Calibrating camera')
    
    # load calibration images
    img = mpimg.imread(fname_array[0], format = 'jpg')
    rows, cols = img.shape[0], img.shape[1]
    img_array = np.zeros((len(fname_array),rows,cols,3),dtype = np.uint8)
    if suppress is False:
        fig1 = plt.figure()

    # load each image file into an array
    for fidx in range(len(fname_array)):
        if suppress is False:
            print('Getting image '+fname_array[fidx])
            
        img = mpimg.imread(fname_array[fidx], format = 'jpg')
        img_array[fidx] = img[0:rows,0:cols] #one image is incorrectly sized, crop

        if suppress is False:
            axN = fig1.add_subplot(5,4,fidx+1)
            axN.imshow(img_array[fidx])
            
    # visualize all calibration images
    if suppress is False:
        fig1.set_tight_layout(True)
        plt.savefig('output_images/camera_cal_pre.png',format='png')
        print('Created output_images/camera_cal_pre.png')

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
            if suppress is False:
                print('9x6 corners found in image '+str(fidx))
        else:
            
            # 9x5 corners
            ret, corners = cv2.findChessboardCorners(gray, (9,5), None)
            if ret is True:
                objp = np.zeros((9*5,3), np.float32)
                objp[:,:2] = np.mgrid[0:9,0:5].T.reshape(-1,2)
                imgpoints.append(corners)
                objpoints.append(objp)
                if suppress is False:
                    print('9x5 corners found in image '+str(fidx))
            else:

                # 8x6 corners
                ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
                if ret is True:
                    objp = np.zeros((8*6,3), np.float32)
                    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
                    imgpoints.append(corners)
                    objpoints.append(objp)
                    if suppress is False:
                        print('8x6 corners found in image '+str(fidx))
                else:

                    # 8x5 corners
                    ret, corners = cv2.findChessboardCorners(gray, (8,5), None)
                    if ret is True:
                        objp = np.zeros((8*5,3), np.float32)
                        objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1,2)
                        imgpoints.append(corners)
                        objpoints.append(objp)
                        if suppress is False:
                            print('8x5 corners found in image '+str(fidx))
                    else:

                        # 9x4 corners
                        ret, corners = cv2.findChessboardCorners(gray, (9,4), None)
                        if ret is True:
                            objp = np.zeros((9*4,3), np.float32)
                            objp[:,:2] = np.mgrid[0:9,0:4].T.reshape(-1,2)
                            imgpoints.append(corners)
                            objpoints.append(objp)
                            if suppress is False:
                                print('9x4 corners found in image '+str(fidx))
                        else:

                            # 7x6 corners
                            ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
                            if ret is True:
                                objp = np.zeros((7*6,3), np.float32)
                                objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
                                imgpoints.append(corners)
                                objpoints.append(objp)
                                if suppress is False:
                                    print('7x6 corners found in image '+str(fidx))
                            else:

                                # 7x5 corners
                                ret, corners = cv2.findChessboardCorners(gray, (7,5), None)
                                if ret is True:
                                    objp = np.zeros((7*5,3), np.float32)
                                    objp[:,:2] = np.mgrid[0:7,0:5].T.reshape(-1,2)
                                    imgpoints.append(corners)
                                    objpoints.append(objp)
                                    if suppress is False:
                                        print('7x5 corners found in image '+str(fidx))
                                else:

                                    # no valid combination
                                    if suppress is False:
                                        print('Corners not found in image '+str(fidx))

    # use OpenCV to get calibrate data
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    # visualize undistort
    if suppress is False:
        fig2 = plt.figure()
        for fidx in range(0,4):
            # sub-sample images for visualizing undistort results
            loc_fidx = fidx*5
            gray = cv2.cvtColor(img_array[fidx], cv2.COLOR_RGB2GRAY)
            dst = cv2.undistort(gray, mtx, dist, None, mtx)
            axis3N1 = fig2.add_subplot(4,2,((fidx+1)*2-1))
            axis3N1.imshow(gray, cmap = 'gray')
            axis3N2 = fig2.add_subplot(4,2,((fidx+1)*2))
            axis3N2.imshow(dst, cmap = 'gray')
        
        fig2.set_tight_layout(True)
        plt.savefig('output_images/camera_cal_post.png',format='png')
        print('Created output_images/camera_cal_post.png')

    return mtx, dist

if __name__ == '__main__':
    # Printing debug info and saving files for README
    # in addition to returning calibration matrix and distortion coeffs
    print('Running locally, printing and saving allowed')

    # Run calibration function
    fname_array = lane_find_files.cal_images()
    mtx, dist = do(fname_array, suppress = False)

    # Print info
    print('Calibration matrix M = '+str(mtx))
    print('Distortion coefficents = '+str(dist))

else:
    # Suppress all command line debug statements and file saves
    pass

