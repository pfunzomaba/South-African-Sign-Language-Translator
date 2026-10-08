# South African Sign Language Translator 🤟

A computer vision-based Sign Language Translator developed using Python, OpenCV, CVZone and a trained machine learning classification model.

## 📌 About the Project

The South African Sign Language Translator is a computer vision project that uses a webcam to detect hand gestures and recognise signs.

The system processes the detected hand and uses a trained classification model to predict the corresponding letter. The recognised letters can then be displayed as text.

This project was developed collaboratively as part of an **AI and Data Science Bootcamp For Women In South Africa**.

## ✨ Features

* Real-time webcam hand detection
* Hand gesture recognition
* Sign classification using a trained model
* Real-time display of recognised signs
* Ability to build text from recognised letters
* Ability to remove the last letter
* Ability to clear the displayed text
* Full-screen visual translator interface

## 🛠️ Technologies Used

* **Python**
* **OpenCV** – image processing and webcam capture
* **CVZone** – hand detection and classification
* **MediaPipe** – hand tracking
* **NumPy** – numerical and image processing
* **TensorFlow/Keras** – machine learning model

## 🧠 How It Works

The system follows these main steps:

1. The webcam captures a live video stream.
2. The system detects a hand in the video.
3. The detected hand is cropped from the frame.
4. The image is resized for the classification model.
5. The trained model predicts the recognised sign.
6. The predicted letter is displayed on the screen.
7. Letters can be added together to create text.

## 📂 Project Structure

```text
South-African-Sign-Language-Translator/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── datacollecton.py
├── train.py
├── test.py
│
├── Model/
│   ├── keras_model.h5
│   └── labels.txt
│
└── Data/
    └── README.md
```

### `datacollecton.py`

Used to collect hand-sign images through the webcam. The images are processed and saved according to the selected sign.

### `train.py`

Loads the trained classification model and uses it to recognise hand gestures captured through the webcam.

### `test.py`

Runs the visual translator and allows recognised letters to be added to the displayed text.

## 🎮 Controls

| Key | Function                            |
| --- | ----------------------------------- |
| `A` | Add the currently recognised letter |
| `R` | Remove the last letter              |
| `D` | Clear the displayed text            |
| `Q` | Quit the application                |

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/pfunzomaba/South-African-Sign-Language-Translator.git
```

### 2. Open the project folder

```bash
cd South-African-Sign-Language-Translator
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

To collect sign language images:

```bash
python datacollecton.py
```

To run the visual translator:

```bash
python test.py
```

A working webcam is required.

## 📷 Demo

Add screenshots or a short demonstration video/GIF here to show the translator detecting hand signs and displaying the recognised letters.

## 👥 Team Project

This project was developed collaboratively as part of an **AI and Data Science Bootcamp**.

The project provided practical experience in computer vision, machine learning, Python programming, data collection, testing and teamwork.

### My Contribution

This repository represents my work and learning as part of the project team.

**Areas of contribution:**

* Python programming
* Computer vision implementation
* Data collection
* Model integration
* Testing and debugging

## 🎯 Skills Demonstrated

Through this project, I gained practical experience with:

* Python programming
* Computer vision
* Hand tracking
* Machine learning model integration
* Image processing
* Webcam-based applications
* Data collection
* Debugging
* Team collaboration
* Git and GitHub

## 🔮 Future Improvements

Possible future improvements include:

* Improving sign recognition accuracy
* Supporting more signs
* Improving recognition under different lighting conditions
* Supporting complete words and sentences
* Adding text-to-speech functionality
* Developing a graphical user interface
* Creating a mobile version

## 👩‍💻 Author

**Pfunzo maba**
This repository showcases my work and contribution to a collaborative AI and Data Science Bootcamp project.

GitHub: [@pfunzomaba](https://github.com/pfunzomaba)

## 📄 License

This project was developed for educational and portfolio purposes.
