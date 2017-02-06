import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import calibrate_camera

def do_wrapper():
    sobel_thresh = [20, 100]
    s_thresh = [170, 255]
    src = []
    dst = do(src, sobel_thresh, s_thresh, suppress = False)

def do(src, sobel_thresh, s_thresh, suppress = True):
    if suppress is False:
        print('Apply Sobel and color thresholds')

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
            
            # Sobel value
            sobel_binary = SobelX(src, sobel_thresh)

            # S value
            color_binary = HlsGrad(src, s_thresh)
            
            stack= np.dstack((np.zeros_like(color_binary), 255*sobel_binary, 255*color_binary))
            dst = np.zeros_like(color_binary)
            dst[(sobel_binary == 1) | (color_binary == 1)] = 1
            plt.subplot(131),plt.imshow(src),plt.title('Input')
            plt.subplot(132),plt.imshow(stack),plt.title('Stacked')
            plt.subplot(133),plt.imshow(dst,cmap = 'gray'),plt.title('Composite')
            filestr = 'output_images/color_and_gradient'+str(fidx)+'.png'
            plt.savefig(filestr,format='png')
            print('Created '+filestr)

    else:
        # Sobel value
        sobel_binary = SobelX(src, sobel_thresh)
        # S value
        color_binary = HlsGrad(src, s_thresh)
          
        dst = np.zeros_like(color_binary)
        dst[(sobel_binary == 1) | (color_binary == 1)] = 1


    return dst

def SobelX(src, sobel_thresh):
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sobel_binary = np.zeros_like(scaled_sobel)
    sobel_binary[(scaled_sobel > sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1

    return sobel_binary

def HlsGrad(src, s_thresh):
    hls = cv2.cvtColor(src, cv2.COLOR_RGB2HLS)
    s = hls[:,:,2]
    color_binary = np.zeros_like(s)
    color_binary[(s > s_thresh[0]) & (s <= s_thresh[1])] = 1

    return color_binary

if __name__ == '__main__':
    # Run the color and sobel threshold function on test images
    print('Running locally on test images, printing and saving allowed')
    do_wrapper()
    
else:
    # Run the color and sobel threshold function on image argument
    pass
