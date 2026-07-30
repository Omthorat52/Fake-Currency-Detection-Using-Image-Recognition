# Fake Currency Detection Using Image Recognition

## 📌 Overview

Fake currency detection is an important application of computer vision that helps identify counterfeit notes. This project uses **image recognition and feature matching techniques** to analyze currency images and determine whether the given note matches a genuine reference image.

The system captures a currency image using a webcam, processes it using **OpenCV**, extracts important visual features using **ORB (Oriented FAST and Rotated BRIEF)**, and compares those features with a genuine currency image to detect authenticity.

---

## 🚀 Features

- Real-time currency image capture using a webcam
- Image preprocessing using grayscale conversion
- Noise reduction using Gaussian Blur
- Edge detection using the Canny algorithm
- Contour detection for currency analysis
- ORB feature extraction for image recognition
- Feature matching using Brute Force Matcher
- Real-time detection result
- Visualization of detected edges and contours

---

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Computer Vision Library:** OpenCV
- **Numerical Processing:** NumPy
- **Techniques Used:**
  - Image Processing
  - Edge Detection
  - Contour Detection
  - Feature Extraction
  - Feature Matching

---

## ⚙️ How It Works

The working process of this project is divided into the following steps:

### 1. Image Capture
The system captures an image of the currency note using a webcam.

### 2. Image Preprocessing
- Converts the captured image into grayscale.
- Applies Gaussian Blur to reduce noise.

### 3. Edge and Contour Detection
- Uses the Canny edge detection algorithm to identify important edges.
- Detects contours to analyze the shape and boundaries of the currency.

### 4. Feature Extraction
- ORB algorithm extracts key features from:
  - Captured currency image
  - Genuine reference currency image

### 5. Feature Matching
- Brute Force Matcher compares extracted features between both images.
- The number of successful matches determines whether the currency is genuine or suspicious.

### 6. Result Generation

- If enough feature matches are found:
