# Import required libraries
import os
import cv2  # OpenCV for video capture and image processing
import numpy as np
import mediapipe as mp  # Google's ML framework for pose/face/hand tracking
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests  # For downloading model files
import urllib.request
import sys
import platform  # For OS-specific camera permission messages

# Print working directory to help with debugging file paths
print("Current working directory:", os.getcwd())

def draw_landmarks_on_image(image, face_detection_result, hand_detection_result):
    """Draw facial and hand landmarks on a black background image"""
    # Create a black canvas with same dimensions as input image
    black_image = np.zeros(image.shape, dtype=np.uint8)

    # Draw facial landmarks if any faces are detected
    if face_detection_result.face_landmarks:
        for face_landmarks in face_detection_result.face_landmarks:
            # Convert landmarks to MediaPipe's protocol buffer format
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])
            # Draw face mesh tessellation (triangles)
            solutions.drawing_utils.draw_landmarks(
                image=black_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
            # Draw face contours (outline)
            solutions.drawing_utils.draw_landmarks(
                image=black_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
            # Draw iris landmarks
            solutions.drawing_utils.draw_landmarks(
                image=black_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
    
    # Draw hand landmarks if any hands are detected
    if hand_detection_result.hand_landmarks:
        for hand_landmarks in hand_detection_result.hand_landmarks:
            # Convert hand landmarks to protocol buffer format
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            # Draw hand landmarks and connections
            solutions.drawing_utils.draw_landmarks(
                black_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())

    return black_image

def download_face_landmarker_model(model_path):
    """Download the face landmarker model if it doesn't exist"""
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    if not os.path.exists(model_path):
        print(f"Downloading face landmarker model to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        response = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("Download completed.")
    else:
        print("Face landmarker model already exists.")

# Set up face detection model
face_model_path = os.path.join('models', 'face_landmarker.task')
download_face_landmarker_model(face_model_path)

# Initialize face detector with model
base_options = python.BaseOptions(model_asset_path=face_model_path)
face_options = vision.FaceLandmarkerOptions(base_options=base_options)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

# Download and set up hand detection model
hand_landmarker_model_path = 'models/hand_landmarker.task'
if not os.path.exists(hand_landmarker_model_path):
    print(f"Downloading hand landmarker model to {hand_landmarker_model_path}...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        hand_landmarker_model_path
    )
    print("Download completed.")

# Initialize hand detector with model
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=hand_landmarker_model_path),
    num_hands=2)  # Track up to 2 hands

hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# Initialize video capture from default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully and provide OS-specific guidance if it failed
if not cap.isOpened():
    print("Error: Could not open camera. Please check your camera permissions.")
    if platform.system() == "Darwin":  # macOS
        print("On macOS, go to System Preferences > Security & Privacy > Privacy > Camera")
        print("Ensure that your terminal or IDE has permission to access the camera.")
    elif platform.system() == "Windows":
        print("On Windows, go to Settings > Privacy > Camera")
        print("Ensure that 'Allow apps to access your camera' is turned on.")
    elif platform.system() == "Linux":
        print("On Linux, ensure that your user has the necessary permissions to access the camera.")
    sys.exit(1)

# Create window for displaying the output
cv2.namedWindow('Face Mesh and Hands', cv2.WINDOW_NORMAL)

# Main processing loop
while True:
    try:
        # Capture frame from camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Check if another application is using the camera.")
            break

        # Convert frame to RGB (MediaPipe requires RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect face and hand landmarks
        face_detection_result = face_detector.detect(mp_image)
        hand_detection_result = hand_detector.detect(mp_image)

        # Draw landmarks on black background
        annotated_image = draw_landmarks_on_image(rgb_frame, face_detection_result, hand_detection_result)
        # Resize image for better visibility
        scale_factor = 1.5
        width = int(annotated_image.shape[1] * scale_factor)
        height = int(annotated_image.shape[0] * scale_factor)
        resized_image = cv2.resize(annotated_image, (width, height), interpolation=cv2.INTER_LINEAR)

        # Display the result
        cv2.imshow('Face Mesh and Hands', resized_image)

        # Check if window was closed
        if cv2.getWindowProperty('Face Mesh and Hands', cv2.WND_PROP_VISIBLE) < 1:
            break

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
