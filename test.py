# Import necessary libraries
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize the webcam capture.
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 15)  # Set the frames per second

# Create an instance of the HandDetector class with a maximum of 1 hand to be detected in each frame.
detector = HandDetector(maxHands=1)

# Create an instance of the Classifier class and load a pre-trained model.
classifier = Classifier("newModel/keras_model.h5", "newModel/labels.txt")

# Define the text to be displayed and its properties
quit_message = "When done with your conversation Press 'q' to close the window"
message_position = (20, 40)
text_position = (30, 100)  # Adjusted position for displaying text_to_display
font_scale = 0.5
font_color = (0, 0, 0)
font_thickness = 1
font = cv2.FONT_HERSHEY_SIMPLEX

# Define the labels for the classifier
labels = ['A', 'B', 'C', 'D', 'E']

# Create a full screen window
cv2.namedWindow("Visual Translator", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Visual Translator", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Initialize the text_to_display variable
text_to_display = ''

# Main loop for processing frames from the webcam
while True:
    # Read a frame from the webcam
    success, img = cap.read()

    # Continue if no frame is captured
    if not success or img is None:
        continue

    # Detect hands in the frame
    hands, _ = detector.findHands(img, draw=False)  # Disable drawing the hand landmarks on the webcam

    for hand in hands:
        x, y, w, h = hand['bbox']  # Extract the bounding box coordinates of the hand
        imgCrop = img[y - 20:y + h + 20, x - 20:x + w + 20]  # Crop the region of interest around the hand
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        # Resize the cropped region for classification
        if aspectRatio > 1:
            k = 100 / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, 100))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((100 - wCal) / 2)
            imgWhite = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = 100 / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (100, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((100 - hCal) / 2)
            imgWhite = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # Display the predicted label on the frame
        cv2.putText(img, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

        # Update text_to_display based on key inputs
        key = cv2.waitKey(1)
        if key & 0xFF == ord('a'):
            text_to_display += labels[index]
        elif key & 0xFF == ord('r') and len(text_to_display) > 0:
            text_to_display = text_to_display[:-1]
        elif key & 0xFF == ord('d'):
            text_to_display = ''

    # Display the processed image and text
    cv2.putText(img, text_to_display, text_position, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # Updated text position
    cv2.putText(img, quit_message, message_position, font, font_scale, font_color, font_thickness)
    cv2.imshow("Visual Translator", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
