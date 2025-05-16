
# Emotion Detection with CNNs – PyTorch and Keras

This project implements real-time emotion detection from facial expressions using two deep learning approaches: a custom Residual CNN in **PyTorch**, and a standard CNN in **Keras**.

---

## 📁 Files Overview

### 🔵 PyTorch Version
- `model.py` – Contains the `EmotionCNN` class using residual blocks
- `torchTraining-resnet.ipynb` – Main training notebook using FER-2013
- `pytorch_final.pt` – Trained PyTorch model
- `pytorchbox.py` – Real-time webcam detection (PyTorch version)
- `pytorchbars.py` – Same as above, with full emotion probability bars

### 🟠 Keras Version
- `kerasTraining-1.ipynb` – Initial CNN model (basic, unaugmented)
- `kerasTraining-augmented.ipynb` – Same model with aggressive augmentation (rotation, brightness, shear, etc.)
- `kerasTraining-newCNN.ipynb` – Tuned CNN with better augmentation and regularization
- `keras_1.h5`, `keras_2.h5`, `keras_final.h5` – Saved Keras models from various training stages
- `kerasbox.py` – Webcam-based emotion detection using final Keras model
- `kerasbars.py` – Adds probability percentage bars per emotion

---

## ⚙️ Model Comparison

### ✅ PyTorch Residual CNN
- **Pros:**
  - Higher accuracy and faster convergence
  - Uses residual connections and batch norm
  - More decisive predictions
- **Cons:**
  - Less nuanced confidence (very high softmax peaks)

### ✅ Keras Standard CNN
- **Pros:**
  - Softer predictions, more useful for multi-emotion visualization
  - Easier to prototype
- **Cons:**
  - Shallow model = lower accuracy
  - Aggressive augmentation (tested in `kerasTraining-augmented.ipynb`) did **not improve performance**, and sometimes hurt convergence

---

## 🔍 Dataset
- **FER-2013** facial emotion dataset (48x48 grayscale images)
- Augmentation included rotation, flip, zoom, shear, and brightness (selectively applied)
- https://www.kaggle.com/datasets/msambare/fer2013

---

## 🧠 Takeaways
- Light augmentation helps; heavy augmentation (e.g., shear + brightness) was **not beneficial**
- Residual CNNs are better for low-resolution facial data
- Keras is useful for visualization-oriented tasks, while PyTorch is better suited for scalable model development

---

## ▶️ Run the Project
To use the webcam detection scripts:
```bash
python kerasbox.py     # or pytorchbox.py
```
To see probability bars:
```bash
python kerasbars.py    # or pytorchbars.py
```

---

## 📦 Dependencies
Install via:
```bash
pip install -r requirements.txt

