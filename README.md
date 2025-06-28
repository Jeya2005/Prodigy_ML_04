# ✋ Task 4 - Hand Gesture Recognition using Computer Vision

🚀 Project Title: Hand Gesture Recognition  
📁 Internship: Machine Learning Internship @ Prodigy Infotech  
👨‍💻 Task Number: 4

---

## 📌 Objective

The aim of this project is to create a system that can recognize hand gestures using a webcam feed in real time. This is done using computer vision techniques such as contour detection and machine learning-based classification.

---

## 🧠 What I Learned

- How to preprocess and clean image data
- Use of OpenCV to capture and process video frames
- Feature extraction using image contours and convexity defects
- Training a classification model (SVM or CNN)
- Performing real-time predictions using a webcam

---

## 🧪 Technologies Used

- Python  
- OpenCV  
- NumPy  
- Scikit-learn (for SVM) / TensorFlow & Keras (for CNN)  
- Matplotlib

---

## 🖼️ Dataset

- Leap Motion Hand Gesture Dataset
- 10 different gestures performed by 10 subjects (total of 10x10 gesture folders)
- Near-infrared images captured using the Leap Motion sensor

### 📂 Dataset Structure:
/00/
├── 01_palm/
├── 02_l/
├── ...
/01/
├── 01_palm/
├── ...

---

## ⚙️ How It Works

1. Load and preprocess grayscale gesture images
2. Resize and flatten images
3. Train an SVM or CNN model to classify gestures
4. Save the trained model and label encoder
5. Run webcam live stream and predict gestures in real-time using OpenCV

---

## 🧾 Installation

Install all required dependencies:

```bash
pip install opencv-python numpy matplotlib scikit-learn tensorflow
