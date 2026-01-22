# Quick Start Guide

Get up and running with Motion Tracker in 5 minutes!

## Installation

### 1. Clone the Repository

```bash
cd /path/to/your/projects
git clone https://github.com/MindDock/motion-tracker.git
cd motion-tracker
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

MediaPipe 0.10+ includes native support for both Apple Silicon and Intel processors, so the same package works on all platforms.

## Run Your First Demo

### Webcam Demo (Real-time Pose Detection)

```bash
python demos/webcam_demo.py
```

This will:
- Open your webcam
- Detect your pose in real-time
- Show joint angles
- Display FPS counter

**Controls:**
- Press `q` to quit
- Press `s` to save screenshot
- Press `r` to reset statistics

### Posture Correction Demo

```bash
python demos/posture_correction_demo.py
```

This monitors your sitting/standing posture and alerts you when it deviates from good form.

**Controls:**
- Press `c` to calibrate with your current good posture
- Press `q` to quit

### AI Fitness Trainer Demo

```bash
python demos/fitness_trainer_demo.py
```

Tracks exercise reps and provides form feedback for:
- Squats (default)
- Push-ups
- Bicep curls
- Shoulder presses

**Controls:**
- Press `1-4` to switch exercises
- Press `r` to reset rep counter
- Press `q` to quit

## Basic Usage in Code

```python
from src.backends.mediapipe_backend import MediaPipeBackend
from src.core.angle_calculator import AngleCalculator
import cv2

# Initialize
estimator = MediaPipeBackend()
estimator.initialize()
calculator = AngleCalculator()

# Open camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect pose
    pose_result = estimator.process_frame(frame)

    if pose_result and pose_result.is_valid():
        # Calculate elbow angle
        angle = calculator.calculate_joint_angle(pose_result, 'left_elbow')
        print(f"Left elbow angle: {angle:.1f}°")

    # Display
    cv2.imshow('Pose', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
estimator.release()
```

## Troubleshooting

### Camera Not Found

If you get "Could not open camera 0", try:

```bash
python demos/webcam_demo.py --camera 1
```

Different camera IDs (0, 1, 2...) correspond to different cameras on your system.

### MediaPipe Import Error

If you see `ImportError: No module named 'mediapipe'`:

```bash
pip install mediapipe>=0.10.0
```

Note: MediaPipe 0.10+ supports both Apple Silicon and Intel natively.

### Low FPS

If performance is slow:

1. Reduce camera resolution:
   ```bash
   python demos/webcam_demo.py --width 640 --height 480
   ```

2. Use lighter model:
   ```python
   estimator = MediaPipeBackend(model_complexity=0)  # 0 is fastest
   ```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [docs/API.md](docs/API.md) for API reference
- See [examples/](examples/) for more code examples
- Join our [Discord community](#) for support

## Getting Help

- GitHub Issues: https://github.com/MindDock/motion-tracker/issues
- Documentation: https://motion-tracker.readthedocs.io
- Email: your.email@example.com

Happy tracking! 🏃‍♂️💪🤸‍♀️
