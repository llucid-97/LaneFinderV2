from binaryThresholds import gradientFilter, colorFilter
from camera_undistort import undistort, getCamMatrix
from perspective_transform import pWarp
import params as GlobalParams
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


class windowClass():
    win_y_low = None
    win_y_high = None
    win_xleft_low = None
    win_xleft_high = None
    win_xright_low = None
    win_xright_high = None


class params():
    camParams = getCamMatrix()
    # Files
    repoRoot = GlobalParams.repoRoot
    imgDir = 'test_images'

    # Crop DIMS
    xClip = 1
    yClip = 0.5
    DEBUG_MODE = False
    fig = plt.figure()


class lines():
    global_step = None
    nwindows = 9
    previousWindows = [windowClass()] * nwindows
    last_left_fit = None
    last_right_fit = None


def interpreteFrame(img):
    # 0: Undistort
    undistorted = undistort(img, params.camParams)
    if params.DEBUG_MODE:
        undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)

    # Perspective Transform
    hawkmoon = pWarp(undistorted)
    plt.ion()

    ax1 = plt.subplot(131)
    ax1.imshow(hawkmoon)
    # plt.pause(0.0)

    # 3: Color Filter
    flt = colorFilter(hawkmoon)
    ax2 = plt.subplot(132)
    ax2.imshow(flt)

    plt.pause(0.05)

    # 4: Sliding Window
    debugImage, laneGlow, left_curverad, right_curverad = getLanes(flt)
    ax3 = plt.subplot(133)
    ax3.imshow(laneGlow)
    plt.pause(0.05)
    #
    perspectiveHighlight= pWarp(laneGlow,True)
    output = cv2.addWeighted(undistorted,1,perspectiveHighlight,0.4,0)

    text = "Radius of Curvature: {} m".format(int(left_curverad))
    cv2.putText(output,text,(100,450),cv2.FONT_HERSHEY_DUPLEX,
                1,(255,0,0),1)
    # "Vehicle is {:.2f} m left of center".format(-position)


    # if params.DEBUG_MODE:
    #     fig = plt.figure()
    #     plt.subplot(131)
    #     plt.imshow(hawkmoon)
    #     plt.subplot(132)
    #     plt.imshow(flat_wack)
    #     plt.subplot(133)
    #     plt.imshow(flt)
    #     plt.show()

    return output


def getLanes(binary_Image):
    # Naming Parameters
    xDim, yDim = binary_Image.shape[1::-1]
    debugImage = np.dstack((binary_Image, binary_Image, binary_Image))

    if lines.global_step is None:
        # Sliding Window
        histogram = np.sum(binary_Image[int(yDim / 2):, :], axis=0)
        midpoint = np.int(xDim / 2)

        # Get the index for the max column in left and right halves
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = np.int(yDim / lines.nwindows)

    nonzero = binary_Image.nonzero()
    nonzero_x = np.array(nonzero[1])
    nonzero_y = np.array(nonzero[0])

    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100
    minpix = 0

    left_lane_inds = []
    right_lane_inds = []

    for window, window_n in enumerate(lines.previousWindows):
        print(window_n)
        win_y_low = binary_Image.shape[0] - (window + 1) * window_height
        win_y_high = binary_Image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw window
        cv2.rectangle(debugImage,
                      (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)

        cv2.rectangle(debugImage,
                      (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high),
                      (0, 255, 255), 2)
        # Select Non-zero indices inside windows
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                          (nonzero_x >= win_xleft_low) & (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                           (nonzero_x >= win_xright_low) & (nonzero_x < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzero_x[left_lane_inds]
    lefty = nonzero_y[left_lane_inds]
    rightx = nonzero_x[right_lane_inds]
    righty = nonzero_y[right_lane_inds]

    ym_per_pix = 30 / yDim  # meters per pixel in y dimension
    xm_per_pix = 3.7 / xDim  # meters per pixel in x dimension

    # Fit a second order polynomial to each (to pixels for display)
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    except:
        left_fit = lines.last_left_fit
        right_fit = lines.last_right_fit

    # Distance Accurate Curve fits
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * yDim * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * yDim * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    debugImage[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 125, 0]
    debugImage[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 125, 255]

    # Draw lines onto image
    outImage = np.zeros_like(debugImage)
    ploty = np.linspace(0, yDim - 1, yDim)
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    left_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_points = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    allPoitns = np.int_(np.hstack((left_points, right_points)))

    cv2.fillPoly(outImage, allPoitns,(175,0,150))

    if params.DEBUG_MODE:
        # Generate x and y values for plotting

        plt.imshow(debugImage)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, x)
        # plt.ylim(y, 0)
        plt.show()
    print(left_curverad, 'm', '\n', right_curverad, 'm')

    lines.last_left_fit = left_fit
    lines.last_right_fit = right_fit
    return debugImage,outImage,left_curverad,right_curverad


# -------------------------------------------------TESTS BELOW
if __name__ == "__main__":
    # DEBUG/TEST
    params.DEBUG_MODE = True
    imgPath = './test_images/test1.jpg'
    image = cv2.imread(imgPath)

    interpreteFrame(image)
