import mediapipe
import cv2 
from cvzone.HandTrackingModule import HandDetector 
import numpy as np
import math
import time

# initializes a video capture object (cap) to capture video from the default camera (camera index 0).
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

# represents the folder where captured images will be saved.
folder = "Data/A"
counter = 0

# which continuously captures frames from the camera and processes them.
while True:

    success, img = cap.read()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize


    # displays the original captured frame with hand landmarks drawn on it.
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # wait for a key press (with a delay of 1 millisecond) and check if the pressed key is 's'.
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg' ,imgWhite)
        print(counter)

    # If the 'q' key is pressed, it breaks out of the loop and exits the application.
    elif key == ord('q'):
        break

