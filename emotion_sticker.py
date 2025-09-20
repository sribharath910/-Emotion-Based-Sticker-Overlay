import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ---------------------------
# Load Emotion Detection Model
# ---------------------------
emotion_model = load_model("emotion_model.h5")  # Pretrained on FER-2013
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ---------------------------
# Load Stickers (PNG with transparency)
# ---------------------------
sticker_paths = {
    "Happy": r"stickers/happy.png",
    "Sad": r"stickers/sad.png",
    "Surprise": r"stickers/surprise.png",
    "Neutral": r"stickers/neutral.png",
    "Angry": r"stickers/angry.png",
    "Disgust": r"stickers/disgust.png",
    "Fear": r"stickers/fear.png",
}
stickers = {k: cv2.imread(v, cv2.IMREAD_UNCHANGED) for k, v in sticker_paths.items()}

# ---------------------------
# Initialize Face Mesh
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# ---------------------------
# Helper: Overlay Sticker
# ---------------------------
def overlay_sticker(frame, sticker, x, y, size=50, alpha=1.0):
    if sticker is None:
        return frame

    sticker_resized = cv2.resize(sticker, (size, size), interpolation=cv2.INTER_AREA)

    if sticker_resized.shape[2] < 4:
        return frame  # needs alpha channel

    b, g, r, a = cv2.split(sticker_resized)
    sticker_rgb = cv2.merge((b, g, r))
    mask = cv2.merge((a, a, a))

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_inv = cv2.bitwise_not(mask)

    h, w = size, size
    if y < 0 or x < 0 or y + h > frame.shape[0] or x + w > frame.shape[1]:
        return frame

    roi = frame[y:y+h, x:x+w]
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    fg = cv2.bitwise_and(sticker_rgb, sticker_rgb, mask=mask)

    blended = cv2.add(bg, fg)

    # Apply fade effect (alpha blending)
    frame[y:y+h, x:x+w] = cv2.addWeighted(roi, 1 - alpha, blended, alpha, 0)
    return frame

# ---------------------------
# Helper: Detect Emotion
# ---------------------------
def detect_emotion(face_roi):
    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (48, 48))
    gray_face = gray_face.astype("float") / 255.0
    gray_face = img_to_array(gray_face)
    gray_face = np.expand_dims(gray_face, axis=0)

    preds = emotion_model.predict(gray_face, verbose=0)[0]
    return emotion_labels[np.argmax(preds)]

# ---------------------------
# Webcam Loop
# ---------------------------
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    h, w, _ = frame.shape
    emotion = "Neutral"  # fallback

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Cheek landmarks
            left_cheek = face_landmarks.landmark[234]
            right_cheek = face_landmarks.landmark[454]

            left_x, left_y = int(left_cheek.x * w), int(left_cheek.y * h)
            right_x, right_y = int(right_cheek.x * w), int(right_cheek.y * h)

            # Face bounding box for emotion detection
            x_min = int(min(l.x for l in face_landmarks.landmark) * w)
            y_min = int(min(l.y for l in face_landmarks.landmark) * h)
            x_max = int(max(l.x for l in face_landmarks.landmark) * w)
            y_max = int(max(l.y for l in face_landmarks.landmark) * h)

            # Crop face
            face_roi = frame[max(0, y_min):y_max, max(0, x_min):x_max]
            if face_roi.size > 0:
                emotion = detect_emotion(face_roi)

            # Dynamic scaling: sticker size based on face width
            face_width = x_max - x_min
            sticker_size = max(40, face_width // 5)

            # Fade if only half the face is visible
            fade_alpha = 1.0
            if x_min <= 0 or x_max >= w:
                fade_alpha = 0.5  # half face detected

            # Overlay sticker based on emotion
            sticker = stickers.get(emotion, stickers["Neutral"])
            frame = overlay_sticker(frame, sticker, left_x - sticker_size//2, left_y - sticker_size//2, sticker_size, fade_alpha)
            frame = overlay_sticker(frame, sticker, right_x - sticker_size//2, right_y - sticker_size//2, sticker_size, fade_alpha)

    # Show emotion label
    cv2.putText(frame, f"Emotion: {emotion}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Emotion-Based Sticker Overlay", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
