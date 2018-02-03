## Writeup Template

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/undistorted_Test1.png "Road Transformed"
[image3]: ./examples/sobel.png "Binary Example"
[image4]: ./examples/perspectiveRoad.png "Warp Example"
[image5]: ./examples/runTime%20Debug.png "Fit Visual"
[image6]: ./examples/Screenshot%20from%202018-02-03%2021-04-41.png "Output"
[image7]: ./examples/chessBoard.png "Undistorted"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

 Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

 **1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.**  

        You're reading it!

### Camera Calibration

 **1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.**

The code for this step is contained in `camCalibration.py`

I looped through all the example chess board images and fed them into OpenCV's chess board corner detector to get corner coordinates.

![alt text][image7]

I then created "true" coordinates for the corners.
This was done using a numpy meshgrid to create coordinates for evenly spaced steps

With the mesh grid as (X,Y) and Zeros as Z, I used these as `object_points`: 3D true coordinates of the chessboard.

I then used the output `object_points` and the detected corners to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I run the previous steps as a function to get the distortion coefficients and camera matrix from the chess boards.

I then use the test image of the road as the feed image to `cv2.undistort`
![alt text][image2]

While subtle, it is easier to spot the differences if you look around the image edges 
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Color and gradient thresholds are implemented in `binaryThresholds.py`

These search the entire image for pixels that meet certain criteria.
I calibrated the search criteria to match yellow pixels and white pixels and generate a binary image where their locations are marked with 1, and all that don't meet this are marked with zero
 
Gradient thresholds were checked on a sobel-filtered image, but these were not used because in the test video, they added no useful data and only made false positives more prominent.


![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform is done in function *pWarp()* in `perspective_transform.py`

It takes a hard-coded perspective line (matching a sample image), mirrors the line, and creates a parallelogram.

The parallelogram:

* contains the lane we're monitoring
* is symmetric and centered on the frame
* matches a "straight" line in perspective view
* points are specified as ratios of the image frame dimensions so it should match different sizes

A second set of points is made, this time representing a rectangle.
The dimensions here are arbitrary.

Both are specified from the top-left, then clockwise.


This resulted in the following source and destination points:

 
| Source        | Destination   | 
|:-------------:|:-------------:| 
| 550, 443      | 0, 0        | 
| 730, 443      | 400, 0      |
| 1280, 659     | 400, 720      |
| 0, 659        | 0, 720        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane Pixels were identified in function `getLanes()` in `interpreteFrame.py`

A pseudo sliding window search is performed on the binary image.

The windows are initialised to a position obtained from local maxima of a histogram of columns

They are persistent between frames, and "slide" to the mean position of all positive binary pixels within them

The location of all pixels inside windows are extracted, and the numpy polyfit function fits a second order polynomial to them.
  

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines *267* through *285* in function `getLanes()` in `interpreteFrame.py`

It follows the linear approximation method shown in lesson 35 of the CarND Submodule.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The shadows were the biggest challenge.
I completely lost the lanes in them and relied on memory only.

There was no threshold which cleanly separated them from the "false" edges on the road (shadows/different colored tar)

A potential fix would be to:
 * Start with the high thresholds as I have here to get an initial projection in frame N
 * Use the line projections from frame N to create a mask slightly larger than the lines in frame N+1
 * Search for the lines in frame N+1 with lower threshold parameters
 
 This reduces the search space, and allows more aggressive search strategies like:
 * Split the gradient into positive and negative x-directioned Sobel filtered images
 * Filter to match areas where positive edge (dark to light)has a nearby negative edge (light to dark) in x-direction (maybe: max_filter negative sobel image with a horizontally-long filter)
 * Blur with a vertically-tall box Filter so these points connect in y-direction
 * Binary threshold again
 
 This (or some other form of aggressive low-threshold filtering) coupled with the masking should fix the line loss
 
 
 Also, the values of "white" and "yellow" are not constant, so why should our parameters be?
 
 They should be tied to a variable like average luminosity of the image, and measured for like 5 widely varying luminosities, then have a curve fit to them, and make all runtime values functions of that.
 
 Of course, this would have to deal with brightness locally, so instead of a global average, it should map a point-to-point sample of a heavily gaussian blurred image of luminosity channel to the point we want to threshold, and use that as the input luminosity.
   
 If that helps, maybe we could throw in other parameters like white-balance