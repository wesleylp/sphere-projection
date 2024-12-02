import cv2
from image_processing import preprocessing, detect_circles, plot_circles_on_image

cam = cv2.VideoCapture(0)

if __name__ == "__main__":

    while True:

        ret, frame = cam.read()

        img = preprocessing(frame)

        circles = detect_circles(
            img,
            method=cv2.HOUGH_GRADIENT,
            dp=0.9,  # Inverse ratio of the accumulator resolution to the image resolution
            minDist=100,  # Minimum distance between the centers of the detected circles
            param1=50,  # Higher threshold for the Canny edge detector
            param2=30,  # Accumulator threshold for the circle centers at the detection stage
            minRadius=5,  # Minimum circle radius
            maxRadius=0,  # Maximum circle radius. If <= 0, uses the maximum image dimension.
        )

        # TODO:
        # Implement postprocessing to filter out false positives since we only
        # want to detect one circle.
        # circles = filter_circles(circles)qq

        frame = plot_circles_on_image(frame, circles)

        cv2.imshow("Camera", frame)

        # Press 'c' to calibrate the camera
        if cv2.waitKey(1) == ord("c"):
            cv2.putText(
                frame,
                "Calibrating camera...",
                (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord("q"):
            break

    # Release the capture and writer objects
    cam.release()
    # out.release()
    cv2.destroyAllWindows()

    print("Done")
