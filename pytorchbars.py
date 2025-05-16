import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from model import EmotionCNN

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionCNN(num_classes=7).to(device)
model.load_state_dict(torch.load("pytorch_final.pt", map_location=device))
model.eval()

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        try:
            face_tensor = transform(face).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(face_tensor)
                probabilities = F.softmax(output, dim=1).cpu().numpy().flatten()
                predicted_idx = np.argmax(probabilities)
                emotion = emotion_labels[predicted_idx]

            # Draw face box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, f"{emotion}", (x, y-10), font, 0.9, (0, 255, 0), 2)

            # Draw percentage bars
            bar_x = x + w + 10
            bar_y = y
            bar_height = 15

            for i, (label, prob) in enumerate(zip(emotion_labels, probabilities)):
                pct = int(prob * 100)
                bar_length = int(prob * 100)
                cv2.rectangle(frame, (bar_x, bar_y + i*bar_height),
                              (bar_x + bar_length, bar_y + (i+1)*bar_height - 2),
                              (0, 255, 0) if i == predicted_idx else (200, 200, 200), -1)
                cv2.putText(frame, f"{label[:7]} {pct}%", (bar_x + 105, bar_y + (i+1)*bar_height - 5),
                            font, 0.4, (0, 0, 0), 1)

        except Exception as e:
            print("Error:", e)
            continue

    cv2.imshow('Real-Time Emotion Detection (with percentages)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
