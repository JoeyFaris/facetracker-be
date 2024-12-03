# Face and Hand Tracker

A real-time facial and hand landmark detection application using MediaPipe and OpenCV. This application captures video from your webcam and displays facial mesh, contours, iris tracking, and hand landmarks on a black background.

## Features

- Real-time face mesh detection and tracking
- Facial contour visualization
- Iris tracking
- Hand landmark detection (up to 2 hands)
- Live webcam feed processing
- OS-specific camera permission guidance

## Prerequisites

- Python 3.7+
- Webcam access

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/facetracker-be.git
   cd facetracker-be
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Required Libraries

- OpenCV (cv2)
- MediaPipe
- NumPy
- Requests

## Usage

1. Run the application:
   ```bash
   python backend/facetracker.py
   ```

2. Controls:
   - Press 'q' to quit the application
   - Close the window to exit

## Troubleshooting Camera Access

### macOS
- Go to System Preferences > Security & Privacy > Privacy > Camera
- Ensure your terminal or IDE has camera access permission

### Windows
- Go to Settings > Privacy > Camera
- Enable "Allow apps to access your camera"

### Linux
- Ensure your user has the necessary permissions to access the camera

## Technical Details

The application uses:
- MediaPipe's Face Landmarker for facial feature detection
- MediaPipe's Hand Landmarker for hand tracking
- Models are automatically downloaded on first run
- Landmarks are rendered on a black background for clear visualization

## Model Information

The application uses two ML models:
- Face Landmarker model (`face_landmarker.task`)
- Hand Landmarker model (`hand_landmarker.task`)

Models are automatically downloaded to the `models/` directory on first run.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/) 
