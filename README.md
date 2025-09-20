# Emotion-Based Sticker Overlay

An interactive real-time application that detects human emotions using a CNN model trained on the FER-2013 dataset and overlays fun stickers (PNG with transparency) on user faces via webcam.

---

## Features

- Real-time face landmark detection using **MediaPipe Face Mesh**  
- Emotion classification into **7 categories**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral  
- Sticker overlay on cheeks according to detected emotion  
- Dynamic sticker scaling according to face size  
- Fade effect if half the face is out of frame  
- Multiple faces support (up to 3 faces)  
- Optional video recording of the overlay output  

---

## Project Structure
```bash
Emotion-Based-Sticker-Overlay/
│
├── stickers/ # Folder containing PNG stickers
│ ├── angry.png
│ ├── disgust.png
│ ├── fear.png
│ ├── happy.png
│ ├── sad.png
│ ├── surprise.png
│ └── neutral.png
│
├── emotion_model.h5 # Pretrained FER-2013 model
├── main.py (or test.py) # Main script to run webcam + stickers
├── demo_output.avi # (Optional) demo video output
├── requirements.txt # Dependencies
└── README.md # Project documentation
```

---

## Installation

1. Clone the repository  
   ```bash===
   git clone https://github.com/sribharath910/Emotion-Based-Sticker-Overlay.git
   cd Emotion-Based-Sticker-Overlay
   ```
2. Install dependencies
   ```bash===
   pip install -r requirements.txt
   ```
3. Ensure you have all stickers in the /stickers folder and emotion_model.h5 in project root.


## Usage

python main.py

The webcam window will open showing sticker overlays based on your emotion.

Press q to exit.

If enabled, the demo video will be saved as demo_output.avi

## Dependencies

Python 3.8+
OpenCV
MediaPipe
TensorFlow / Keras
NumPy

