import cv2 as cv
import numpy as np


def main() -> None:
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening video stream")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blurred, 60, 150)

        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Draw contours on a black canvas so only contour lines are shown.
        contour_view = cv.cvtColor(np.zeros_like(gray), cv.COLOR_GRAY2BGR)
        cv.drawContours(contour_view, contours, -1, (0, 255, 0), 2)

        cv.imshow("Contours Only", contour_view)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
