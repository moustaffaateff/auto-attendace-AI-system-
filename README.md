# auto-attendace-AI-system-
a system that uses camera to produce a class attendace report using computer vision techniques 
Below is a **complete, professional README.md** written **after understanding your code and pipeline**.
You can **copy–paste this directly into GitHub**.

---

# Face Recognition–Based Attendance System 🎓📷

An **AI-powered face recognition attendance system** that automatically detects, recognizes, and verifies student attendance using **Computer Vision and Deep Learning**.
The system captures images at different time intervals, detects faces using **MTCNN**, recognizes identities using a **CNN-based model**, and confirms attendance by cross-checking multiple captures to reduce false positives.

---

## 🔍 Project Overview

This project automates the traditional attendance process by replacing manual checks with a **camera-based face recognition pipeline**.
It is designed to work in real classroom environments and focuses on **accuracy, reliability, and scalability**.

The system:

* Captures (15) images from a camera at fixed above the writing board
* Detects faces in captured frames
* Recognizes student identities using a trained deep learning model
* Confirms attendance only if a student appears consistently across multiple time windows

---

## 🧠 System Workflow

1. **Frame Capture**

   * Captures frames from the webcam at regular time intervals
   * Saves frames to disk for offline processing

2. **Face Detection**

   * Uses **MTCNN** to detect faces in each frame
   * Draws bounding boxes and crops detected faces
   * Stores cropped faces for recognition

3. **Face Recognition**

   * Uses a trained **CNN model** to classify detected faces
   * Applies confidence thresholding to reduce misclassification
   * Maps predictions to registered student identities

4. **Attendance Verification**

   * Runs recognition twice (e.g., start of lecture & after 15 minutes)
   * Final attendance is calculated using the **intersection of both results**
   * Ensures only consistently present students are marked as attended

---

## 🧠 Model Architecture

* **Face Detection:**

  * MTCNN (Multi-task Cascaded Convolutional Networks)

* **Face Recognition:**

  * CNN-based classifier
  * Trained using synthetic data of faces  
  * Fine-tuned using **transfer learning**(real faces of the class students) for improved accuracy

* **Input Size:** 125 × 125 RGB images

* **Loss Function:** Categorical Cross-Entropy

* **Optimizer:** Adam

---

## 🛠 Technologies Used

* **Programming Language:** Python
* **Computer Vision:** OpenCV, MTCNN
* **Deep Learning:** TensorFlow / Keras
* **Data Handling:** NumPy
* **Visualization:** Matplotlib
* **Environment:** Jupyter Notebook / Python Scripts

---

## 📁 Project Structure

```
face-recognition-attendance-system/
│
├── capture_frames.py              # Webcam frame capture logic
├── face_detection.py              # Face detection & cropping (MTCNN)
├── face_recognition.py            # Identity prediction using CNN
├── attendance_pipeline.py         # Full attendance workflow
│
├── train_recognition_model.py     # CNN training from scratch
├── fine_tuning_transfer_learning.py # Transfer learning & fine-tuning
├── rename_faces_by_prediction.py  # Auto-label detected faces
│
├── models/
│   └── recognition_model.h5       # Trained face recognition model
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

* synthetic faces from microsoft dataset + Custom student face dataset
* Images organized by student identity (folder-per-class format)
* Data augmentation applied to improve generalization
* Separate training and testing directories

---

## ✅ Key Features

* Automated attendance without manual intervention
* High accuracy using deep learning
* Two-stage verification to reduce false attendance
* Modular and extensible codebase
* Suitable for real-world classroom deployment

---

## 📈 Results

* Achieved **high recognition accuracy (~84%)** after transfer learning (Base model was trained on local laptop)
* Successfully recognized registered students while rejecting unknown faces

---

## 👨‍💻 Author

**Moustafa Atef Farouk**
Computer Engineer | AI & Computer Vision Engineer
📧 Email: [moustafaateff@gmail.com](mailto:moustafaateff@gmail.com)

---

## ⭐ Acknowledgments

* OpenCV & MTCNN open-source community
* TensorFlow & Keras
* Deep Learning research community

---
