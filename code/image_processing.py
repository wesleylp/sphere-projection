import cv2
import numpy as np
from constants import BASE, RAIL


def preprocessing(img):
    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = cv2.equalizeHist(img_gray)

    # Apply Gaussian blur to the image
    img_blur = cv2.GaussianBlur(hist, (17, 17), 0)

    img_preprocessed = img_blur

    return img_preprocessed


def plot_limits(img, limits, color=(0, 255, 0), thickness=2):
    cv2.line(img, limits[0], limits[3], color, thickness)
    cv2.line(img, limits[1], limits[2], color, thickness)

    return img


def detect_circles(img, *args, **kwargs):
    circles = cv2.HoughCircles(img, *args, **kwargs)
    return circles


def filter_circles(circles):
    # Implement postprocessing to filter out false positives since we only
    # want to detect one circle.
    circles_filtered = []
    if circles is not None:
        for i in circles[0, :]:
            if i[0] >= RAIL[0][0] and i[0] <= RAIL[1][0] and i[2]:
                circles_filtered.append(i)
    return np.array([circles_filtered])


def plot_circles_on_image(img, circles):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

            # draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    return img
