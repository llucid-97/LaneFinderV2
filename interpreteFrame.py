from binaryThresholds import gradientFilter, colorFilter
from camera_undistort import undistort, getCamMatrix
from perspective_transform import pWarp
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


class WindowPair:
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

    valid_left = False
    valid_right = False

    margin_left = 30
    margin_right = 30


class Params:
    camParams = getCamMatrix()
    # Files
    repoRoot = os.path.dirname(os.path.realpath(__file__))
    imgDir = 'test_images'

    # Crop DIMS
    yBirdEyeCrop = 0
    DEBUG_MODE = False
    PLOT_AT_RUNTIME = True
    fig = plt.figure()


class linePair:
    global_step = 0  # What frame we are on

    # Sliding Window Parameters
    nwindows = 20
    window_height = None
    previousWindows = []

    # Current Line Params
    left_fit = np.array([0, 0, 0])
    right_fit = np.array([0, 0, 0])

    center = None
    curvature = None


lines = linePair()
lane_xDim, lane_yDim = 0, 0


def interpreteFrame(img):
    global lane_xDim
    # 0: Undistort
    yDim, xDim, _ = np.shape(img)
    undistorted = undistort(img, Params.camParams)
    if Params.DEBUG_MODE:
        undistorted = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)

    # 1: Perspective Transform
    birdEye = pWarp(undistorted)

    # 2: Mask
    birdEye[:Params.yBirdEyeCrop, :, :] = [0, 0, 0]

    # 3: Color Filter
    yellow, white = colorFilter(birdEye)

    # 4: Gradient Filter (Not Used, just here to show it was created)
    # sobel = gradientFilter(birdEye)

    # 5: Get Lanes
    debugImage, laneGlow, left_curverad, right_curverad = getLanes(yellow, white)

    # 6: Reverse Perspective Warp
    laneGlow = pWarp(laneGlow, True)

    # 7: Get Line Center
    # Get mean pixel index of bottom line of lane glow:
    from perspective_transform import params as ptParams
    laneGlowPixels = np.nonzero(laneGlow[int(ptParams.hood_top * yDim) - 5, :, 2])
    newCenter = np.mean(laneGlowPixels)

    # Assign or smoothen
    if lines.center is None:
        lines.center = newCenter
    else:
        gamma = 0.2
        lines.center = ((1 - gamma) * lines.center) + \
                       (gamma * newCenter)

    output = cv2.addWeighted(undistorted, 1, laneGlow, 0.4, 0)

    text = "Radius of Curvature: {} m".format(int(lines.curvature))
    cv2.putText(output, text, (100, 450), cv2.FONT_HERSHEY_DUPLEX,
                1, (255, 0, 0), 1)

    text = "Distance to Center: {0:.2f} m".format(
        ((xDim / 2) - lines.center)  # pixel distance of (Image center - line center)
        * (3.7 / lane_xDim)  # meters of lane / pixels of lane
    )
    cv2.putText(output, text, (100, 420), cv2.FONT_HERSHEY_DUPLEX,
                1, (255, 125, 0), 1)
    if Params.PLOT_AT_RUNTIME:
        plt.ion()
        ax1 = plt.subplot(131)
        ax1.imshow(birdEye)
        # ax2 = plt.subplot(152)
        # ax2.imshow(flt)
        ax4 = plt.subplot(132)
        ax4.imshow(debugImage)
        ax5 = plt.subplot(133)
        ax5.imshow(output)
        plt.pause(0.01)
        ax1.cla()
        ax4.cla()
        ax5.cla()

    return output


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def getLanes(yellow_binary, white_binary):
    # Naming Parameters
    global lane_xDim, lane_yDim
    debugImage = np.dstack((white_binary, yellow_binary + yellow_binary, white_binary)) * 127

    if lines.global_step == 0:
        lane_xDim, lane_yDim = yellow_binary.shape[1::-1]

        # Get Histogram peaks
        histogram = np.sum(yellow_binary[int(lane_yDim / 2):, :], axis=0)
        yellow_x_base = np.argmax(histogram)

        histogram = np.sum(white_binary[int(lane_yDim / 2):, :], axis=0)
        white_x_base = np.argmax(histogram)

        lines.window_height = np.int((lane_yDim - Params.yBirdEyeCrop
                                      ) / lines.nwindows)

        yellow_current = yellow_x_base
        white_current = white_x_base

        for i in range(lines.nwindows):
            # Create a persistent window list to allow temporal processing of windows
            x = WindowPair()
            lines.previousWindows.append(x)

    # Easy selector for all pixels that met criteria
    y_nonzero = yellow_binary.nonzero()
    w_nonzero = white_binary.nonzero()

    left_lane_inds = []
    right_lane_inds = []

    overlap = False
    numValidL = 0
    numValidR = 0

    minpix = 10

    for i, window in enumerate(lines.previousWindows):
        # window = WindowPair(window)  # Just for the IDE to know and make linking Easy. %TODO: Delet dis
        # Draw window around known center
        if lines.global_step == 0:
            window.leftx_current = yellow_current
            window.rightx_current = white_current
        lmargin = window.margin_left
        rmargin = window.margin_right
        # Top/Bottom
        window.y_low = yellow_binary.shape[0] - (i + 1) * lines.window_height
        window.y_high = yellow_binary.shape[0] - i * lines.window_height
        # Left Window
        window.xleft_low = window.leftx_current - lmargin
        window.xleft_high = window.leftx_current + lmargin
        # Right Window
        window.xright_low = window.rightx_current - rmargin
        window.xright_high = window.rightx_current + rmargin

        # Select Non-zero indices inside windows
        good_left_inds = ((y_nonzero[0] >= window.y_low) & (y_nonzero[0] < window.y_high) &
                          (y_nonzero[1] >= window.xleft_low) & (y_nonzero[1] < window.xleft_high)).nonzero()[0]
        good_right_inds = ((w_nonzero[0] >= window.y_low) & (w_nonzero[0] < window.y_high) &
                           (w_nonzero[1] >= window.xright_low) & (w_nonzero[1] < window.xright_high)).nonzero()[0]

        # Check for overlaps and spearate them
        window.leftx_previous = int(window.leftx_previous)
        window.rightx_previous = int(window.rightx_previous)
        if window.xleft_high > window.xright_low:
            if not (np.size(good_left_inds) == 0) and not (np.size(good_right_inds) == 0):
                tmp_leftx = np.int(np.mean(y_nonzero[1][good_left_inds]))
                tmp_rightx = np.int(np.mean(w_nonzero[1][good_right_inds]))

                if abs(tmp_leftx - window.leftx_previous) < abs(tmp_rightx - window.rightx_previous):
                    window.xright_low = int(window.rightx_previous) - rmargin
                    window.xright_high = int(window.rightx_previous) + rmargin
                else:
                    window.xleft_low -= int(window.leftx_previous) - lmargin
                    window.xleft_high -= int(window.leftx_previous) + lmargin
            else:
                window.xleft_low = window.leftx_previous - lmargin
                window.xleft_high = window.leftx_previous + lmargin
                window.xright_low = window.rightx_previous - rmargin
                window.xright_high = window.rightx_previous + rmargin

        # Mark all pixels inside window for use
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Update window Centers for next frame
        if len(good_left_inds) > minpix:
            # Take average
            window.leftx_previous = window.leftx_current
            window.leftx_current = np.int(np.mean(y_nonzero[1][good_left_inds]))
            window.valid_left = True
            numValidL += 1
            window.margin_left = 20
        else:
            # Expand search
            window.margin_left = 40
            # if lines.global_step != 0:
            #     window.yellow_current = window.white_current + int(lines.left_fit[0] * window.y_low ** 2 +
            #                                                        lines.left_fit[1] * window.y_low +
            #                                                        lines.left_fit[2])
            #     window.white_current /= 2
            # else:
            for j in reversed(range(i)):
                if (lines.previousWindows[j]).valid_left:
                    window.leftx_current = int((4 * window.leftx_current + lines.previousWindows[j].leftx_current) / 5)
                    break

        if len(good_right_inds) > minpix:
            window.rightx_previous = window.rightx_current
            window.rightx_current = np.int(np.mean(w_nonzero[1][good_right_inds]))
            window.valid_right = True
            numValidR += 1
            window.margin_right = 20
        else:
            # Expand search, move window toward last known valid windows
            window.margin_right = 40
            # if lines.global_step != 0:
            #     window.white_current = window.white_current + int(lines.right_fit[0] * window.y_low ** 2 +
            #                                                         lines.right_fit[1] * window.y_low +
            #                                                         lines.right_fit[2])
            #     window.white_current = int(window.white_current /2)
            # else:
            for j in reversed(range(i)):
                if (lines.previousWindows[j]).valid_right:
                    window.rightx_current = (4 * window.rightx_current + lines.previousWindows[j].rightx_current) / 5
                    window.rightx_current = int(window.rightx_current)
                    break

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

    # Extract left and right line pixel positions
    leftx = y_nonzero[1][left_lane_inds]
    lefty = y_nonzero[0][left_lane_inds]
    rightx = w_nonzero[1][right_lane_inds]
    righty = w_nonzero[0][right_lane_inds]

    ym_per_pix = 30 / lane_yDim  # meters per pixel in y dimension
    xm_per_pix = 3.7 / lane_xDim  # meters per pixel in x dimension

    # Fit a second order polynomial to each (to pixels for display)
    if (numValidL > 4):
        Params.darkMode = False
        tmp_left_fit = np.polyfit(lefty, leftx, 2)
        if lines.global_step == 0:
            lines.left_fit = tmp_left_fit
    else:
        Params.darkMode = True
        tmp_left_fit = lines.left_fit

    if (numValidR > 2):
        tmp_right_fit = np.polyfit(righty, rightx, 2)
        if lines.global_step == 0:
            lines.right_fit = tmp_right_fit
    else:
        # Too few valid Windows. Retain previous
        Params.darkMode = True
        tmp_right_fit = lines.right_fit

    # Get Curve Radius
    # 1: Distance Accurate Curve fits
    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    # Temporal Filter lines (Leaky Integrator)
    gamma = 0.2
    lines.left_fit = ((1 - gamma) * lines.left_fit) + \
                     ((gamma) * tmp_left_fit)
    lines.right_fit = ((1 - gamma) * lines.right_fit) + \
                      ((gamma) * tmp_right_fit)

    # 2: Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * lane_yDim * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * lane_yDim * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # 3: Merge and Temporal Filter
    newCurv = np.mean([left_curverad, right_curverad])

    if lines.curvature is None:
        lines.curvature = newCurv
    else:
        # Temporal filter curvature (leaky Integrator)
        lines.curvature = (0.95 * lines.curvature) + \
                          ((1 - 0.95) * newCurv)
    # debugImage
    debugImage[y_nonzero[0][left_lane_inds], y_nonzero[1][left_lane_inds]] = [255, 175, 0]
    debugImage[w_nonzero[0][right_lane_inds], w_nonzero[1][right_lane_inds]] = [255, 255, 255]

    # Draw lines onto image
    outImage = np.zeros_like(debugImage)

    ploty = np.linspace(Params.yBirdEyeCrop,
                        lane_yDim - 1, lane_yDim)

    left_fitx = np.array(lines.left_fit[0] * ploty ** 2 + lines.left_fit[1] * ploty + lines.left_fit[2]).astype(int)
    right_fitx = np.array(lines.right_fit[0] * ploty ** 2 + lines.right_fit[1] * ploty + lines.right_fit[2]).astype(int)

    left_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_points = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    allPoitns = np.int_(np.hstack((left_points, right_points)))

    cv2.fillPoly(outImage, allPoitns, (175, 0, 150))

    if Params.DEBUG_MODE:
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
    Params.DEBUG_MODE = True
    imgPath = './test_images/test1.jpg'
    image = cv2.imread(imgPath)

    interpreteFrame(image)
