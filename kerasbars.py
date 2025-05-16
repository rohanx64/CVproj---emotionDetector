import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('keras_final.h5')

# Emotion labels from FER-2013
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract and preprocess face
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float32') / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        # Predict emotion probabilities
        prediction = model.predict(face, verbose=0)[0]  # shape: (7,)
        predicted_idx = np.argmax(prediction)
        emotion = emotion_labels[predicted_idx]

        # Draw face bounding box and top prediction
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), font, 0.9, (36, 255, 12), 2)

        # Draw probability bars
        bar_x = x + w + 10
        bar_y = y
        bar_height = 15

        for i, (label, prob) in enumerate(zip(emotion_labels, prediction)):
            pct = int(prob * 100)
            bar_length = int(prob * 100)
            cv2.rectangle(frame, (bar_x, bar_y + i*bar_height),
                          (bar_x + bar_length, bar_y + (i+1)*bar_height - 2),
                          (0, 255, 0) if i == predicted_idx else (200, 200, 200), -1)
            cv2.putText(frame, f"{label[:7]} {pct}%", (bar_x + 105, bar_y + (i+1)*bar_height - 5),
                        font, 0.4, (0, 0, 0), 1)

    cv2.imshow('Real-Time Emotion Detection (Keras)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
