import mediapipe
import cv2  # OpenCV library for computer vision tasks
# A custom hand tracking module from the cvzone library, which is used to detect and
# track hands in images or video frames.
from cvzone.HandTrackingModule import HandDetector # Draw hand landmarks and bounding boxes around the detected hands
import numpy as np
import math
import time

# initializes a video capture object (cap) to capture video from the default camera (camera index 0).
cap = cv2.VideoCapture(0)
# initializes a hand detector object (detector) using the HandDetector class from the cvzone library.
# It's configured to detect a maximum of 1 hand.
detector = HandDetector(maxHands=1)

# It's set to 20 and define a margin around the detected hand.
offset = 20
# It's set to 300 and defines the size of the output image where the hand will be placed.
imgSize = 300

# represents the folder where captured images will be saved.
folder = "newData/E"
# It's initialized to 0 and will be used to keep track of the number of captured images.
counter = 0

# which continuously captures frames from the camera and processes them.
while True:
    # captures a frame from the camera using the cap.read() method and
    # assigns it to the variables success (a boolean indicating whether the capture
    # was successful) and img (the captured image).
    success, img = cap.read()

    # uses the findHands method of the detector object to detect hands in the captured image.
    # It returns a list of detected hands (if any) and also updates the img variable with the drawn hand landmarks.
    hands, img = detector.findHands(img)

    # If a hand is detected, this code retrieves the bounding box (bbox)
    # of the detected hand and assigns its coordinates (x, y, w, h) to separate variables.
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

        # performs image processing on the detected hand and places it in a new image with a white background.
        # Depending on the aspect ratio of the hand's bounding box,
        # it either resizes the width or the height of the hand to fit it into the imgSize square.
        # display two windows showing the cropped hand (imgCrop) and the processed image with a
        # white background (imgWhite).
        #cv2.imshow("ImageCrop", imgCrop)
        #cv2.imshow("ImageWhite", imgWhite)

    # displays the original captured frame with hand landmarks drawn on it.
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    # wait for a key press (with a delay of 1 millisecond) and check if the pressed key is 's'.
    if key == ord("s"):
        # If 's' is pressed, this code increments the counter, saves the processed image with
        # a filename that includes the current timestamp in
        # the specified folder, and prints the current count of captured images.
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg' ,imgWhite)
        print(counter)

    # If the 'q' key is pressed, it breaks out of the loop and exits the application.
    elif key == ord('q'):
        break

