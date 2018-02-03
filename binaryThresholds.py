import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.filters import maximum_filter


class params():
    # Desired range ratios to strongest edge detected

    sobelPosMin = 0.1

    sobelNegMin = -0.3

    # Runtime Flags
    hls_YellowLow_thresholds = [10, 30, 0.2]
    hls_YellowHigh_thresholds = [30, 255, 1]

    hls_WhiteLow_thresholds = [0, 0, 0.8]
    hls_WhiteHigh_thresholds = [255, 15, 1]
    DEBUG_MODE = False


def colorFilter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    width = hsv.shape[1]
    height = hsv.shape[0]


    # hsv[:,:,2] = maximum_filter(img[:, :, 2], size=(9, 9), mode='nearest')

    maxLum = np.max(hsv[:, :, 2])

    localMaxima = maximum_filter(img[:, :, 2], size=(100, 200), mode='nearest')

    select_yellow = (hsv[:, :, 0] < params.hls_YellowHigh_thresholds[0]) & \
                    (hsv[:, :, 0] > params.hls_YellowLow_thresholds[0]) & \
                    (hsv[:, :, 1] < params.hls_YellowHigh_thresholds[1]) & \
                    (hsv[:, :, 1] > params.hls_YellowLow_thresholds[1]) & \
                    (hsv[:, :, 2] < params.hls_YellowHigh_thresholds[2] * maxLum) & \
                    (hsv[:, :, 2] > params.hls_YellowLow_thresholds[2] * maxLum)

    select_white = (hsv[:, :, 1] < params.hls_WhiteHigh_thresholds[1]) & \
                   (hsv[:, :, 1] > params.hls_WhiteLow_thresholds[1]) & \
                   (hsv[:, :, 2] < params.hls_WhiteHigh_thresholds[2] * maxLum) & \
                   (hsv[:, :, 2] > params.hls_WhiteLow_thresholds[2] * maxLum)

    select_WY = select_white | select_yellow
    result = np.zeros_like(hsv[:, :, 0])
    result[select_WY] = 1

    if params.DEBUG_MODE:
        # fig = plt.figure()
        ax1 = plt.subplot(131)
        plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        ax1.set_title("Hue")
        ax2 = plt.subplot(132)
        plt.imshow(result, cmap='gray')
        ax2.set_title("Vibrance")
        ax2 = plt.subplot(133)
        plt.imshow(localMaxima, cmap='gray')
        ax2.set_title("Vibrance")
        plt.show()

    return result

    # Now we have grey image. Hist EQ to ensure contrast/lighting issues
    # dont mess up our thresholds
    # img = cv2.equalizeHist(img)


def gradientFilter(img):
    """
    Sobel Edge detection in X-direction with binary thresholding.
    Returns a dictionary containing 2 images:
        >Positive Edges (Dark to Bright)
        >Negative Edges (Bright to Dark)

    :param img: Input Image
    :return: Dictionary of binary thresholded sobel images ("+" and "-"
    """
    if len(img.shape) > 2:
        # Assume we've been passed raw image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get X-Direction sobel filtered image
    sobel = cv2.Sobel(
        img,
        cv2.CV_64F,
        1, 0
    )
    white = np.max(np.absolute(sobel))  # Strongest Edge
    sobel = 128*sobel / white  # Quantise to range

    # Binary Threshold to select pixels within desired range
    posSobel = np.zeros_like(sobel)
    negSobel = np.zeros_like(sobel)

    grey = np.mean(sobel[(sobel>0)])
    print(grey)
    posSobel[(sobel >= grey * params.sobelPosMin)] = 1
    negSobel[(sobel <= grey * params.sobelNegMin)] = 1

    negSobel = maximum_filter(negSobel,size=9)

    posSobel[(negSobel == 0)] = 0

    if params.DEBUG_MODE:
        fig = plt.figure()
        ax1 = plt.subplot(221)
        plt.imshow(img, cmap='gray')
        ax1.set_title("Image (Histogram Equalised)")
        ax2 = plt.subplot(222)
        plt.imshow(sobel, cmap='gray')
        ax2.set_title("Sobel in X-direction")
        ax3 = plt.subplot(223)
        plt.imshow(posSobel, cmap='gray')
        ax3.set_title("Positive Sobel-X")
        ax4 = plt.subplot(224)
        plt.imshow(negSobel, cmap='gray')
        ax4.set_title("Negative Sobel-X")

        plt.show()
    sobelImages = {
        "+": posSobel,
        "-": negSobel,
        "0": sobel,

    }
    return posSobel


# --------------------------------------------------------TESTS
if __name__ == "__main__":
    params.DEBUG_MODE = True

    # img = cv2.imread("/home/geragi01/Pictures/Screenshot from 2018-01-30 22-42-03.png")
    img = cv2.imread("/home/geragi01/Pictures/Screenshot from 2018-02-02 20-00-57.png")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gradientFilter(img)
    # colorFilter(img)
