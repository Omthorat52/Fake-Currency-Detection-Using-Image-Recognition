import cv2
import numpy as np

# Function to perform edge detection and find contours
def detect_edges_and_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur
    edges = cv2.Canny(blurred, 50, 150)  # Detect edges using Canny edge detector
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, edges

# Function to match features (e.g., watermarks or security threads)
def match_features(image, known_feature_image):
    # Convert both images to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_known_feature = cv2.cvtColor(known_feature_image, cv2.COLOR_BGR2GRAY)
    
    # Use ORB feature detector to find keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray_image, None)
    kp2, des2 = orb.detectAndCompute(gray_known_feature, None)
    
    # Use BFMatcher to match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance (good matches will have smaller distance)
    matches = sorted(matches, key = lambda x:x.distance)
    
    return matches

# Function to detect fake currency based on feature matching
def detect_fake_currency(image_path, known_feature_image_path):
    # Load the input image and the known feature (real currency feature)
    image = cv2.imread(image_path)
    known_feature = cv2.imread(known_feature_image_path)
    
    # Step 1: Detect edges and contours to check for basic features
    contours, edges = detect_edges_and_contours(image)
    
    # Step 2: Match features with known real currency feature (e.g., watermark or thread)
    matches = match_features(image, known_feature)
    
    if len(matches) > 10:  # If we find enough matching features, it's likely real
        print("Real currency detected")
    else:
        print("Fake currency detected")
    
    # Show the edge-detected image and contours for visual inspection
    cv2.imshow("Edges", edges)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)  # Draw contours
    cv2.imshow("Contours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
image_path = "2000 org.jpg"  # Path to the test currency image
known_feature_image_path = "10 fk.jpg"  # Path to a feature image (e.g., watermark)

detect_fake_currency(image_path, known_feature_image_path)
