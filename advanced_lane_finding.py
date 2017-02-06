''' Project 4
    Advanced Lane Finding
    Main Pipeline '''

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

## Calibrate the camera
import calibrate_camera
mtx, dist = calibrate_camera.do()
print('transformation matrix: '+str(mtx))
print('distortion coeffs: '+str(dist))
