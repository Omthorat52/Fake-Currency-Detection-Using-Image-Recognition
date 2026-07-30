import cv2
import numpy as np

# Detect edges and contours
def detect_edges_and_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, edges

# Match features using ORB
def match_features(image, known_feature_image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_known_feature = cv2.cvtColor(known_feature_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray_image, None)
    kp2, des2 = orb.detectAndCompute(gray_known_feature, None)

    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        print("Not enough features found in one or both images.")
        return []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

# Capture image from camera
def capture_from_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return None

    print("Press SPACE to capture an image or ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Live Camera - Press SPACE to Capture", frame)
        key = cv2.waitKey(1)

        if key == 27:  # ESC key
            print("Exiting without capture.")
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key == 32:  # SPACE key
            print("Image captured.")
            cap.release()
            cv2.destroyAllWindows()
            return frame

# Main function
def detect_fake_currency_from_camera(known_feature_image_path):
    known_feature = cv2.imread(known_feature_image_path)
    if known_feature is None:
        print(f"Error: Could not load known feature image from '{known_feature_image_path}'")
        return

    captured_image = capture_from_camera()
    if captured_image is None:
        print("No image was captured.")
        return

    contours, edges = detect_edges_and_contours(captured_image)
    matches = match_features(captured_image, known_feature)

    # Detection result
    if len(matches) > 10:
        print("Real currency detected.")
    else:
        print("Fake currency detected or not enough matches.")

    # Visual results
    cv2.imshow("Edges", edges)
    cv2.drawContours(captured_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Contours", captured_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
known_feature_image_path = "20 orig.jpg"  # <-- Replace with your known feature image path
detect_fake_currency_from_camera(known_feature_image_path)
