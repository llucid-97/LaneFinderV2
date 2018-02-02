import numpy as np
import cv2
import matplotlib.pyplot as plt


class params():
    # Parallelogram Crop
    horizon = 0.615
    top_left = 0.429
    hood_top = 0.916
    bottom_left = 0

    dst_width = 400
    dst_height = 720
    DEBUG_MODE = False


def pWarp(img,reverse=False,original_Dims=(1280,720)):
    """
    Perspective transform to get bird's eye view
    :img: Input Image
    :return:
    """
    # img = np.array(img)
    x, y = img.shape[1::-1]
    if reverse:
        x,y = original_Dims

    # It's important that the parallelogram is symmetric and centered
    # Because it's warping with respect to paralellogram
    source_points = np.float32(  # top left and clockwise
        [[x * params.top_left, y * params.horizon],
         [x - x * params.top_left, y * params.horizon],
         [x, y * params.hood_top],
         [0, y * params.hood_top]]
    )
    destination_points = np.float32(  # top left and clockwise
        [[0, 0],
         [params.dst_width, 0],
         [params.dst_width, params.dst_height],
         [0, params.dst_height]]
    )
    print(source_points)
    print(destination_points)

    if reverse:
        M = cv2.getPerspectiveTransform(destination_points, source_points)
        out_x = x
        out_y = y
    else:
        M = cv2.getPerspectiveTransform(source_points, destination_points)
        out_x = params.dst_width
        out_y = params.dst_height

    warped = cv2.warpPerspective(
        img,
        M,
        # (x, y),
        (out_x,out_y),
        flags=cv2.INTER_NEAREST)
    if params.DEBUG_MODE:
        fig = plt.figure()
        ax1 = plt.subplot(121)
        ax1.set_title("Original")
        plt.imshow(img)
        ax2 = plt.subplot(122)
        ax2.set_title("Bird's Eye Perspective Transform")
        plt.imshow(warped)
        plt.show()
    return warped

# --------------------------------------------------------TEST
if __name__ == "__main__":
    img = cv2.imread("test_images/test2.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pWarp(img)
    # x,y = (np.array(img).shape)[1::-1]
    # print(params.top_left/x)
