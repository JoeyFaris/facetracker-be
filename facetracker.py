import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests
import urllib.request
import sys
import platform

print("Current working directory:", os.getcwd())

def draw_landmarks_on_image(image, face_detection_result, hand_detection_result):
    black_image = np.zeros(image.shape, dtype=np.uint8)

    if face_detection_result.face_landmarks:
        for face_landmarks in face_detection_result.face_landmarks:
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=black_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=black_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=black_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    if hand_detection_result.hand_landmarks:
        for hand_landmarks in hand_detection_result.hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                black_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())

    return black_image

def download_face_landmarker_model(model_path):
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

face_model_path = os.path.join('models', 'face_landmarker.task')
download_face_landmarker_model(face_model_path)

base_options = python.BaseOptions(model_asset_path=face_model_path)
face_options = vision.FaceLandmarkerOptions(base_options=base_options)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

hand_landmarker_model_path = 'models/hand_landmarker.task'
if not os.path.exists(hand_landmarker_model_path):
    print(f"Downloading hand landmarker model to {hand_landmarker_model_path}...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        hand_landmarker_model_path
    )
    print("Download completed.")

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=hand_landmarker_model_path),
    num_hands=2)

hand_detector = vision.HandLandmarker.create_from_options(hand_options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera. Please check your camera permissions.")
    if platform.system() == "Darwin":
        print("On macOS, go to System Preferences > Security & Privacy > Privacy > Camera")
        print("Ensure that your terminal or IDE has permission to access the camera.")
    elif platform.system() == "Windows":
        print("On Windows, go to Settings > Privacy > Camera")
        print("Ensure that 'Allow apps to access your camera' is turned on.")
    elif platform.system() == "Linux":
        print("On Linux, ensure that your user has the necessary permissions to access the camera.")
    sys.exit(1)

cv2.namedWindow('Face Mesh and Hands', cv2.WINDOW_NORMAL)
# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()
fps = 0

# Function to calculate and display FPS
def update_fps():
    global frame_count, start_time, fps
    frame_count += 1
    current_time = time.time()
    elapsed_time = current_time - start_time
    if elapsed_time > 1:  # Update FPS every second
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = current_time
    return fps

# Function to display FPS on the frame
def display_fps(image, fps):
    cv2.putText(image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return image


while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame. Check if another application is using the camera.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        face_detection_result = face_detector.detect(mp_image)
        hand_detection_result = hand_detector.detect(mp_image)

        annotated_image = draw_landmarks_on_image(rgb_frame, face_detection_result, hand_detection_result)
        scale_factor = 1.5
        width = int(annotated_image.shape[1] * scale_factor)
        height = int(annotated_image.shape[0] * scale_factor)
        resized_image = cv2.resize(annotated_image, (width, height), interpolation=cv2.INTER_LINEAR)

        cv2.imshow('Face Mesh and Hands', resized_image)


        if cv2.getWindowProperty('Face Mesh and Hands', cv2.WND_PROP_VISIBLE) < 1:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        break

cap.release()
cv2.destroyAllWindows()
