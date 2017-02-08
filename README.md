# Project 4 - Advanced Lane Finding #

## Camera Calibration ##
Camera calibration is a straightforward application of lessons in this course.  There was one additional method that I applied to properly calibrate the camera with the provided calibration images.  Namely, I found that not all of the images had the same number of corners, i.e. some of the chessboards were cropped.  So, I implemented a search function, as a set of nested `if` statements, in order to find the correct number of corners for each images.  A sample of that code, from [calibrate_camera.py](calibrate_camera.py), is shown below.

```python
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
```

Even with this search, I was unable to get calibration points out of file 4.  By visual inspection, it's not clear why one of these combinations is not sufficient.

I selected 4 images from the calibration to test the calibration data, by performing a `cv2.undistort` and comparing the before and after images side-by-side.  The first image pair is the easiest to judge by the naked eye, and say the image has been reasonably corrected.

