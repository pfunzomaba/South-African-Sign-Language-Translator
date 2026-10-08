# Dataset

The images were collected using the `datacollecton.py` script, which captures hand gestures through a webcam and saves the processed images into folders according to the selected sign.

## Data Collection

The data collection process includes:

1. Selecting the letter to collect.
2. Capturing hand gestures using a webcam.
3. Detecting the hand using CVZone.
4. Cropping and resizing the detected hand.
5. Saving the processed images for use during model development.

The complete dataset is not included in this repository to keep the repository size manageable.

To collect your own images, run:

```bash
python datacollecton.py
```

The collected images will be saved in the corresponding `Data` folder.
