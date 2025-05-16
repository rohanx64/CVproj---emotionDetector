import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from model import EmotionCNN  # make sure EmotionCNN is defined in model.py or inline

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=7).to(device)
model.load_state_dict(torch.load("pytorch_final.pt", map_location=device))
model.eval()

# Emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Transform for face image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# OpenCV Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        try:
            face_tensor = transform(face).unsqueeze(0).to(device)  # shape [1, 1, 48, 48]
            with torch.no_grad():
                output = model(face_tensor)
                pred = torch.argmax(F.softmax(output, dim=1))
                emotion = emotion_labels[pred.item()]

            # Draw results
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        except:
            continue

    cv2.imshow('Real-Time Emotion Detection (PyTorch)', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()