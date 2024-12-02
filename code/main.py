import cv2
import numpy as np
from image_processing import (
    preprocessing,
    detect_circles,
    filter_circles,
    plot_circles_on_image,
    plot_limits,
)
from estimate import (
    compute_circle_area,
    depth_estimate_from_area,
    depth_estimate_from_radius,
)
from constants import MINRADIUS, MAXRADIUS, BASE, RAIL, PROGRAM_LABEL

cam = cv2.VideoCapture(0)

if __name__ == "__main__":

    depth_ref = None

    while True:

        ret, frame = cam.read()

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

        # TODO:
        # Implement postprocessing to filter out false positives since we only
        # want to detect one circle.
        circles = filter_circles(circles)

        frame = plot_circles_on_image(frame, circles)

        # Plot the limits of the base and rail
        frame = plot_limits(frame, BASE, color=(0, 0, 255), thickness=1)
        frame = plot_limits(frame, RAIL, color=(0, 0, 255), thickness=1)

        if depth_ref is None:
            cv2.imshow(PROGRAM_LABEL, frame)
            depth_ref = float(
                input(
                    "Enter the distance from the camera to the reference point in cm: "
                )
            )
            radius_ref = circles[0][0][2]
            area_ref = compute_circle_area(circles[0][0][2])

        depth_estimate_rad = depth_estimate_from_radius(
            circles[0][0][2], radius_ref, depth_ref
        )

        area_circle = compute_circle_area(circles[0][0][2])
        depth_estimate_a = depth_estimate_from_area(area_circle, area_ref, depth_ref)

        cv2.putText(
            frame,
            f"Radius estimate : {circles[0][0][2]:.2f} pixels",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Area estimate : {area_circle:.2f} pixels",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Depth estimate from radius: {depth_estimate_rad:.2f} cm",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Depth estimate from area: {depth_estimate_a:.2f} cm",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow(PROGRAM_LABEL, frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord("q"):
            break

    # Release the capture and writer objects
    cam.release()
    # out.release()
    cv2.destroyAllWindows()

    print("Done")
