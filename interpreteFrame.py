from binaryThresholds import gradientFilter, colorFilter
from camera_undistort import undistort, getCamMatrix
from perspective_transform import pWarp
import params as GlobalParams
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


class windowClass():
    y_low = None
    y_high = None
    xleft_low = None
    xleft_high = None
    xright_low = None
    xright_high = None

    leftx_current = None
    rightx_current = None

    leftx_previous = 100
    rightx_previous = 300

    valid = False
    margin = 30


class params():
    camParams = getCamMatrix()
    # Files
    repoRoot = GlobalParams.repoRoot
    imgDir = 'test_images'

    # Crop DIMS
    yBirdEyeCrop = 300
    DEBUG_MODE = False
    fig = plt.figure()


class linePair():
    global_step = 0  # What frame we are on

    # Sliding Window Parameters
    nwindows = 9
    window_height = None
    previousWindows = []

    # Current Line Params
    left_fit = None
    right_fit = None

    center = None
    curvature = None


global lines
lines = linePair()


def interpreteFrame(img):
    # 0: Undistort
    yDim, xDim, _ = np.shape(img)
    undistorted = undistort(img, params.camParams)
    if params.DEBUG_MODE:
        undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)

    # Perspective Transform
    hawkmoon = pWarp(undistorted)

    plt.ion()

    # Mask
    hawkmoon[:params.yBirdEyeCrop, :, :] = [0, 0, 0]
    avg = np.median(hawkmoon)

    # 3: Color Filter
    flt = colorFilter(hawkmoon)

    # 4: Sobel
    sobel = gradientFilter(hawkmoon)

    # 5: Get Lanes
    debugImage, laneGlow, left_curverad, right_curverad = getLanes(flt)

    # 6: Reverse Perspective Warp
    laneGlow = pWarp(laneGlow, True)

    # Line Center
    from perspective_transform import params as ptParams
    laneGlowPixels = np.nonzero(laneGlow[int(ptParams.hood_top * yDim) - 5, :, 2])

    newCenter = np.mean(laneGlowPixels)
    if lines.center is None:
        lines.center = newCenter

    # Temporal Smoothing (Leaky Integrator)
    gamma = 0.2
    lines.center = ((1 - gamma) * lines.center) + \
                   (gamma * newCenter)
    print("Center", lines.center)
    output = cv2.addWeighted(undistorted, 1, laneGlow, 0.4, 0)

    text = "Radius of Curvature: {} m".format(int(lines.curvature))

    cv2.putText(output, text, (100, 450), cv2.FONT_HERSHEY_DUPLEX,
                1, (255, 0, 0), 1)
    text = "Distance to Center: {0:.2f} m".format(((yDim/2) - lines.center)*(3.7/xDim),)
    cv2.putText(output, text, (100, 420), cv2.FONT_HERSHEY_DUPLEX,
                1, (255, 125, 0), 1)

    ax1 = plt.subplot(151)
    ax1.imshow(hawkmoon)
    ax2 = plt.subplot(152)
    ax2.imshow(flt)
    ax3 = plt.subplot(153)
    ax3.imshow(laneGlow)
    ax4 = plt.subplot(154)
    ax4.imshow(debugImage)
    ax5 = plt.subplot(155)
    ax5.imshow(output)
    plt.pause(0.01)

    return output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def getLanes(binary_Image):
    # Naming Parameters
    xDim, yDim = binary_Image.shape[1::-1]
    debugImage = np.dstack((binary_Image, binary_Image, binary_Image))

    if lines.global_step == 0:
        # Sliding Window
        histogram = np.sum(binary_Image[int(yDim / 2):, :], axis=0)
        midpoint = np.int(xDim / 2)

        # Get the index for the max column in left and right halves
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        lines.window_height = np.int((yDim - params.yBirdEyeCrop) / lines.nwindows)

        leftx_current = leftx_base
        rightx_current = rightx_base

        for i in range(lines.nwindows):
            x = windowClass()
            lines.previousWindows.append(x)

    else:
        print(" ")

    # Easy selector for all pixels that met criteria
    nonzero = binary_Image.nonzero()
    nonzero_x = np.array(nonzero[1])
    nonzero_y = np.array(nonzero[0])

    minpix = 0

    left_lane_inds = []
    right_lane_inds = []

    overlap = False
    numValid = 0

    for i, window in enumerate(lines.previousWindows):
        if lines.global_step == 0:
            window.leftx_current = leftx_current
            window.rightx_current = rightx_current
        margin = window.margin
        window.y_low = binary_Image.shape[0] - (i + 1) * lines.window_height
        window.y_high = binary_Image.shape[0] - i * lines.window_height
        window.xleft_low = window.leftx_current - margin
        window.xleft_high = window.leftx_current + margin
        window.xright_low = window.rightx_current - margin
        window.xright_high = window.rightx_current + margin

        if window.xleft_high > window.xright_low:
            overlap = True

        # Select Non-zero indices inside windows
        good_left_inds = ((nonzero_y >= window.y_low) & (nonzero_y < window.y_high) &
                          (nonzero_x >= window.xleft_low) & (nonzero_x < window.xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= window.y_low) & (nonzero_y < window.y_high) &
                           (nonzero_x >= window.xright_low) & (nonzero_x < window.xright_high)).nonzero()[0]

        if overlap:

            if not (np.size(good_left_inds) == 0) and not (np.size(good_right_inds) == 0):
                tmp_leftx = np.int(np.mean(nonzero_x[good_left_inds]))
                tmp_rightx = np.int(np.mean(nonzero_x[good_right_inds]))

                if abs(tmp_leftx - window.leftx_previous) < abs(tmp_rightx - window.rightx_previous):
                    window.xright_low = window.rightx_previous - margin
                    window.xright_high = window.rightx_previous + margin
                else:
                    window.xleft_low -= window.leftx_previous - margin
                    window.xleft_high -= window.leftx_previous + margin
            else:
                window.xleft_low = window.leftx_previous - margin
                window.xleft_high = window.leftx_previous + margin
                window.xright_low = window.rightx_previous - margin
                window.xright_high = window.rightx_previous + margin

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            window.leftx_previous = window.leftx_current
            window.leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
            window.valid = True
            numValid += 1
            window.margin = 30
        else:
            window.margin = 60
            for j in reversed(range(i)):
                if (lines.previousWindows[j]).valid:
                    window.leftx_current += lines.previousWindows[j].leftx_current
                    window.leftx_current /= 2
                    window.leftx_current = int(window.leftx_current)
                    break

        if len(good_right_inds) > minpix:
            window.rightx_previous = window.rightx_current
            window.rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

        # Draw window
        cv2.rectangle(debugImage,
                      (window.xleft_low, window.y_low),
                      (window.xleft_high, window.y_high),
                      (0, 255, 0), 2)

        cv2.rectangle(debugImage,
                      (window.xright_low, window.y_low),
                      (window.xright_high, window.y_high),
                      (0, 255, 255), 2)

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    confidence_left = np.size(left_lane_inds) / 3000
    confidence_left = sigmoid(confidence_left)
    confidence_right = np.size(right_lane_inds) / 500
    confidence_right = sigmoid(confidence_right)
    print(confidence_right, " Right\n", confidence_left, " Left")
    # Extract left and right line pixel positions
    leftx = nonzero_x[left_lane_inds]
    lefty = nonzero_y[left_lane_inds]
    rightx = nonzero_x[right_lane_inds]
    righty = nonzero_y[right_lane_inds]

    ym_per_pix = 30 / yDim  # meters per pixel in y dimension
    xm_per_pix = 3.7 / xDim  # meters per pixel in x dimension

    # Fit a second order polynomial to each (to pixels for display)
    if numValid > 4:
        tmp_left_fit = np.polyfit(lefty, leftx, 2)
        tmp_right_fit = np.polyfit(righty, rightx, 2)

        if lines.global_step == 0:
            lines.left_fit = tmp_left_fit
            lines.right_fit = tmp_right_fit
    else:
        params.darkMode = True
        tmp_left_fit = lines.left_fit
        tmp_right_fit = lines.right_fit

    # Temporal Filter fit params (Leaky Integrator)
    gamma = 0.2

    # Get Curve Radius
    # 1: Distance Accurate Curve fits
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # 2: Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * yDim * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * yDim * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # 3: Merge and Temporal Filter
    newCurv = np.mean([left_curverad, right_curverad])

    if lines.curvature is None:
        lines.curvature = newCurv
    else:
        lines.curvature = (0.95 * lines.curvature) + \
                  ((1 - 0.95) * newCurv)

    # Temporal Smoothing Lines (leaky Integrator)
    lines.left_fit = ((1 - gamma * confidence_left) * lines.left_fit) + \
                     ((gamma * confidence_left) * tmp_left_fit)
    lines.right_fit = ((1 - gamma * confidence_right) * lines.right_fit) + \
                      ((confidence_right * gamma) * tmp_right_fit)

    debugImage[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 125, 0]
    debugImage[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 125, 255]

    # Draw lines onto image
    outImage = np.zeros_like(debugImage)
    ploty = np.linspace(params.yBirdEyeCrop, yDim - 1, yDim)
    left_fitx = tmp_left_fit[0] * ploty ** 2 + tmp_left_fit[1] * ploty + tmp_left_fit[2]
    right_fitx = tmp_right_fit[0] * ploty ** 2 + tmp_right_fit[1] * ploty + tmp_right_fit[2]

    left_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_points = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    allPoitns = np.int_(np.hstack((left_points, right_points)))

    cv2.fillPoly(outImage, allPoitns, (175, 0, 150))

    if params.DEBUG_MODE:
        # Generate x and y values for plotting

        plt.imshow(debugImage)
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, x)
        # plt.ylim(y, 0)
        plt.show()
    print(left_curverad, 'm', '\n', right_curverad, 'm')

    # lines.left_fit = tmp_left_fit
    # lines.right_fit = tmp_right_fit
    lines.global_step += 1
    return debugImage, outImage, left_curverad, right_curverad


# -------------------------------------------------TESTS BELOW
if __name__ == "__main__":
    # DEBUG/TEST
    params.DEBUG_MODE = True
    imgPath = './test_images/test1.jpg'
    image = cv2.imread(imgPath)

    interpreteFrame(image)
