#def print_hi(name):
#    # Use a breakpoint in the code line below to debug your script.
#    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.
#
#
## Press the green button in the gutter to run the script.
#if __name__ == '__main__':
#    print_hi('PyCharm')


import numpy as np
import cv2 as cv
import random
from ultralytics import YOLO
#from google.colab.patches import cv2_imshow

def getColors(cls_num):
    """Generate unique colors for each class ID"""
    random.seed(cls_num)
    return tuple(random.randint(0, 255) for _ in range(3))

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error opening video stream")
    exit()

yolo = YOLO("yolov8s.pt")

while True:
    # Capture frame by frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    results = yolo.track(frame, stream=True)

    for result in results:
        class_names = result.names
        for box in result.boxes:
            if box.conf[0] > 0.4:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls = int(box.cls[0])
                class_name = class_names[cls]

                conf = float(box.conf[0])

                colour = getColors(cls)

                cv.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                cv.putText(frame, f"{class_name} {conf:.2f}",
                            (x1, max(y1 - 10, 20)), cv.FONT_HERSHEY_SIMPLEX,
                            0.6, colour, 2)

    # Display the resulting frame
    cv.imshow('Video Capture', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


