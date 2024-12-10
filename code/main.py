import cv2
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


def main():
    if not cam.isOpened():
        print("Error: Could not open the camera.")
        return

    depth_ref = None

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
            frame = plot_circles_on_image(frame, circles)
        else:
            continue

        # Plot the limits of the base and rail
        frame = plot_limits(frame, BASE, color=(0, 0, 255), thickness=1)
        frame = plot_limits(frame, RAIL, color=(0, 0, 255), thickness=1)

        key = cv2.waitKey(1)

        if key == ord("q"):
            break
        if key == ord("c"):
            # cv2.waitKey(0)
            depth_ref = float(
                input(
                    "Enter the distance from the camera to the reference point in cm: "
                )
            )
            if circles is not None and len(circles) > 0:
                radius_ref = circles[0][0][2]
                area_ref = compute_circle_area(radius_ref)
                print(
                    f"Reference radius: {radius_ref:.2f}, Reference area: {area_ref:.2f}"
                )
        if key == ord("p"):
            cv2.waitKey(0)

        if depth_ref is not None:

            depth_estimate_rad = depth_estimate_from_radius(
                circles[0][0][2], radius_ref, depth_ref
            )

            area_circle = compute_circle_area(circles[0][0][2])
            depth_estimate_a = depth_estimate_from_area(
                area_circle,
                area_ref,
                depth_ref,
            )

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

        if depth_ref is None:
            cv2.putText(
                frame,
                "Waiting for calibration...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            "Press 'c' for calibration...",
            (10, 410),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
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
