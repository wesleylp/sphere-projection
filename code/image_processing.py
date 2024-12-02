import cv2
import numpy as np


def preprocessing(img):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)

    img_preprocessed = img_blur

    return img_preprocessed


def detect_circles(img, *args, **kwargs):
    circles = cv2.HoughCircles(img, *args, **kwargs)
    return circles


def plot_circles_on_image(img, circles):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

            # draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    return img
