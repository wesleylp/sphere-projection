import cv2

import time
from image_processing import (
    preprocessing,
    detect_circles,
    filter_circles,
    plot_circles_on_image,
)

from constants import MINRADIUS, MAXRADIUS, PROGRAM_LABEL

cam = cv2.VideoCapture(2)  # change to 0 for the default camera


def main():
    if not cam.isOpened():
        print("Error: Could not open the camera.")
        return

    file = open(f"logging_trilho_acelerado{time.time()}.txt", "w")
    file.write("timestamp\tx\ty\tr\n")

    # loop de calibração
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error while tryng to capture frame from webcam.")
            break

        img = preprocessing(frame)

        circles = detect_circles(
            img,
            method=cv2.HOUGH_GRADIENT,
            dp=0.9,  # Inverse ratio of the accumulator resolution to the image
            # resolution
            minDist=100,  # Minimum distance between the centers of the detected
            # circles
            param1=50,  # Higher threshold for the Canny edge detector
            param2=30,  # Accumulator threshold for the circle centers at the
            # detection stage
            minRadius=MINRADIUS,  # Minimum circle radius
            maxRadius=MAXRADIUS,  # Maximum circle radius. If <= 0, uses the
            # maximum image dimension.
        )

        # postprocessing to filter out false positives since we only
        # want to detect one circle.
        circles = filter_circles(circles)

        if circles is not None and circles.shape[1] == 1:
            x, y, r = circles[0, 0, :]
            file.write(f"{time.time()}\t{x}\t{y}\t{r}\n")
            frame = plot_circles_on_image(frame, circles)
        else:
            # If no circles are detected, we can skip the logging
            file.write(f"{time.time()}\t{0}\t{0}\t{0}\n")
            continue

        key = cv2.waitKey(1)

        if key == ord("q"):
            file.close()
            break

        if key == ord("p"):
            cv2.waitKey(0)

        cv2.putText(
            frame,
            "Press 'p' for pause/unpause...",
            (10, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "Press 'q' for quit...",
            (10, 470),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        cv2.imshow(PROGRAM_LABEL, frame)

    # Release the capture and writer objects
    cam.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    print("Done")
