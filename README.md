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

There are two pairs of images demonstrating the camera calibration below.

![Whoops, there should be a picture here!](output_images/undistorted_chessboard0.png)
![Whoops, there should be a picture here!](output_images/undistorted_road0.png)

It is very clear in the chessboard image, above, that the slightly curved lines have been straightened.  It's not as apparent on the image of the road, but if one examines the left and right edges closely, the undistorted images shows less of the periphery, and the road sign on the right-hand side is a more natural shape.  There are additional images in this repository, saved as output_images/undistorted\*.png showing the rest of the undistort results.

## Thresholding ##
For thresholding I combined three techniques: Sobel gradient threshold in the horizontal (x) direction, HLS threshold in the S-plane, and RGB thresholds around yellow and white colors.  Each of the three results is represented by a different color in the composite images below.  Red pixels are returned from the yellow or white RGB thresholding, green pixels are returned from the Sobel gradient thresholding, and blue pixels are from the HLS thresholding.  The relevant code is broken into four functions: one for each threshold, and a combining function:

```python
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
    ```
    All of the test images are in the [output_images](output_images) folder, I show the one below that I find the most interesting.  There are contributions from each of the three filtering methods apparent.

![Whoops, there should be a picture here!](output_images/thresholded_road5.png)

## Perspective Warping ##

For perspective transformation, I assumed that a constant, x-symmetric trapezoid, mapped into a rectangle, would be my mapping.  I used one of the test images that I judged, by eye, to have very straight lines, and iteratively moved the vertices of the trapezoid, until the resulting warped image showed straight lines.  This image is shown below.

![Whoops, there should be a picture here!](output_images/warped_road0.png)
