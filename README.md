# Fake Currency Detection Using OpenCV

## Overview
This project is a computer vision-based fake currency detection system that uses image processing and feature matching techniques to identify whether a currency note is genuine or potentially fake.

The system captures an image of a currency note using a camera, extracts important visual features, detects edges and contours, and compares the captured image with a known genuine currency image using ORB (Oriented FAST and Rotated BRIEF) feature matching.

## Features
- Real-time image capture using a webcam
- Edge detection using Canny algorithm
- Contour detection for object analysis
- Feature extraction using ORB
- Feature matching using Brute Force Matcher
- Compares captured currency with a reference genuine currency image
- Displays detected edges and contours for visualization

## Technologies Used
- Python
- OpenCV
- NumPy
- Computer Vision
- Image Processing

## How It Works
1. Capture an image of the currency note using the camera.
2. Convert the image into grayscale format.
3. Apply Gaussian Blur to reduce noise.
4. Detect edges using the Canny edge detector.
5. Extract features from the captured image and reference image using ORB.
6. Match features using the Brute Force Matcher.
7. Analyze the number of successful matches:
   - More matches → Genuine currency detected
   - Fewer matches → Fake currency or insufficient matching features

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Omthorat52/Fake-Currency-Detection-Using-Image-Recognition.git
