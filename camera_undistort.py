"""
Cameras have error in mapping 3D objects to a 2D place mostly due to lenses
This script calculates a calibration matrix and distortion coefficients for a camera

It uses a known object (chessboard) and maps real images to expected positions
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import params as GlobalParams
class params():
    # Runtime Flags
    DEBUG_MODE = False

    # Chess Board Images
    # Number of inner corners on example in X and Y
    numX = 9
    numY = 6

    # Files
    imgDir = "camera_cal"  # Camera Calibration images directory
    repoRoot = GlobalParams.repoRoot

def getCamMatrix():
    imagePoints = []  # 2D points where the corners appear in image
    objectPoint = []  # 3D coordinates of where the chessboard is in space

    # Create an ideal meshgrid with the X and Y coordinates of evenly spaced
    # squares to represent chessboard corners.
    # Leave Z-coordinate = 0
    defaultObjectPoints = np.zeros((params.numX * params.numY, 3), np.float32)
    defaultObjectPoints[:, :2] = np.mgrid[0:params.numX, 0:params.numY].T.reshape(-1, 2)

    skip = False
    ramImages = []
    for root, _, filename in os.walk(os.path.join(params.repoRoot, params.imgDir)):
        for f in filename:
            # Loop through all images and fill the arrays
            f = os.path.join(root, f)
            img = cv2.imread(f)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ramImages.append(img)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(
                gray,
                (params.numX, params.numY),
                None
            )
            if ret == True:
                shape = img.shape[:2][::-1]
                if params.DEBUG_MODE and not skip:
                    cv2.drawChessboardCorners(
                        img,
                        (params.numX, params.numY),
                        corners, ret
                    )
                    plt.imshow(img)
                    plt.show()
                    userIn = input("skip? [Y/n]")
                    if any(y in userIn for y in ["Y", "y"]):
                        skip = True
                        print("Skipping")
                imagePoints.append(corners)
                objectPoint.append(defaultObjectPoints)

    skip = False
    # Calibrate Camera using data
    ret, camMatrix, distCoeffs, CamRotationVector, CamTranslationVector = cv2.calibrateCamera(
        objectPoint, imagePoints, shape, None, None)
    if ret:
        print("Successfully Obtained Calibration matrix!")
    else:
        print("camCalibration.py is broken.\n\t Probably fed images wrong, retard ¬_¬")
    if params.DEBUG_MODE and not skip:

        # dst = cv2.undistort(img, mtx, dist, None, mtx)
        for img in ramImages:
            undistorted = cv2.undistort(
                img,
                camMatrix,
                distCoeffs,
                None,
                camMatrix
            )
            fig = plt.figure()
            plt.subplot(121)
            plt.imshow(img)
            plt.subplot(122)
            plt.imshow(undistorted)
            # plt.subplot_tool()
            plt.show()
    cameraCorrection = {
        "camMatrix": camMatrix,
        "distCoeffs": distCoeffs,
    }
    return cameraCorrection


def undistort(img, camParamDict):
    undistorted = cv2.undistort(
        img,
        camParamDict["camMatrix"],
        camParamDict["distCoeffs"],
        None,
        camParamDict["camMatrix"],
    )
    return undistorted


