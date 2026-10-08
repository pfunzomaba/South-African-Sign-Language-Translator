
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

letter = input("Enter the letter you want to collect (A-Z): ").upper()

# Check that the input is a single letter
if len(letter) != 1 or not letter.isalpha():
    print("Please enter one letter from A to Z.")
    exit()

# Create the folder automatically
folder = f"Data/{letter}"
os.makedirs(folder, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize hand detector
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
counter = 0

print(f"\nCollecting images for sign: {letter}")
print("Press 'S' to save an image.")
print("Press 'Q' to quit.")

while True:

    success, img = cap.read()

    if not success or img is None:
        continue

    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white 300x300 image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand area
        imgCrop = img[y - offset:y + h + offset,
                      x - offset:x + w + offset]

        # Make sure the crop is valid
        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        # Resize while keeping the original aspect ratio
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)

            imgResize = cv2.resize(imgCrop, (wCal, imgSize))

            wGap = math.ceil((imgSize - wCal) / 2)

            imgWhite[:, wGap:wGap + wCal] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)

            imgResize = cv2.resize(imgCrop, (imgSize, hCal))

            hGap = math.ceil((imgSize - hCal) / 2)

            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Display the cropped/processed image
        cv2.imshow("Processed Hand", imgWhite)

    # Display the webcam image
    cv2.putText(
        img,
        f"Sign: {letter} | Images: {counter}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        img,
        "Press S = Save | Q = Quit",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Image Collection", img)

    key = cv2.waitKey(1) & 0xFF

    # Save image
    if key == ord("s") and hands:
        counter += 1

        filename = os.path.join(
            folder,
            f"Image_{int(time.time() * 1000)}.jpg"
        )

        cv2.imwrite(filename, imgWhite)

        print(f"Saved image {counter}: {filename}")

    # Quit
    elif key == ord("q"):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()

print(f"\nFinished collecting images for {letter}.")
print(f"Total images collected: {counter}")
print(f"Images saved in: {folder}")

