# Installation Guide

## Prerequisites

- Python 3.10 or higher
- macOS 12.0+ (for Apple Silicon optimization)
- Webcam or video input device

## Quick Installation

### 1. Clone Repository

```bash
git clone https://github.com/MindDock/motion-tracker.git
cd motion-tracker
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- MediaPipe 0.10+ (with Apple Silicon support)
- OpenCV
- NumPy
- And other required packages

## Download MediaPipe Models

MediaPipe 0.10+ requires downloading model files. The system will attempt to download automatically on first run, but you can also download manually.

### Option 1: Automatic Download (Recommended)

The model will download automatically when you first run any demo:

```bash
python demos/webcam_demo.py
```

### Option 2: Manual Download

If automatic download fails due to network issues:

1. Create models directory:
   ```bash
   mkdir -p models
   ```

2. Download the model file you need:

   **Lite Model** (fastest, ~12MB):
   ```bash
   curl -L -o models/pose_landmarker_lite.task \
     "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
   ```

   **Full Model** (balanced, ~25MB, recommended):
   ```bash
   curl -L -o models/pose_landmarker_full.task \
     "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
   ```

   **Heavy Model** (most accurate, ~30MB):
   ```bash
   curl -L -o models/pose_landmarker_heavy.task \
     "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
   ```

3. Use the model by specifying complexity when initializing:
   ```python
   # 0 = lite, 1 = full, 2 = heavy
   backend = MediaPipeBackend(model_complexity=1)
   ```

## Platform-Specific Notes

### Apple Silicon (M1/M2/M3/M4)

MediaPipe 0.10+ includes native ARM64 support. No special installation needed.

### Intel Macs

Same installation process. MediaPipe will automatically use the x86_64 version.

### Other Platforms

MediaPipe supports Linux and Windows. See [MediaPipe documentation](https://ai.google.dev/edge/mediapipe/solutions/setup_python) for platform-specific notes.

## Verify Installation

Test that everything is installed correctly:

```bash
python -c "import mediapipe; import cv2; import numpy; print('✓ All packages installed')"
```

## Troubleshooting

### MediaPipe Import Error

```
ImportError: No module named 'mediapipe'
```

Solution:
```bash
pip install mediapipe>=0.10.0
```

### Model Download Fails

If you see "Failed to download model", use manual download (see above).

### Camera Not Found

```
Error: Could not open camera 0
```

Solutions:
1. Try different camera IDs:
   ```bash
   python demos/webcam_demo.py --camera 1
   ```

2. Check camera permissions in System Settings > Privacy & Security > Camera

3. List available cameras:
   ```bash
   python -c "import cv2; [print(f'Camera {i}') for i in range(5) if cv2.VideoCapture(i).isOpened()]"
   ```

### NumPy Version Conflicts

If you encounter NumPy compatibility issues:

```bash
pip install "numpy>=1.24.0,<2.0.0"
```

Note: MediaPipe 0.10.31 supports NumPy 2.x, but you can use 1.x if needed.

### Low Performance

If FPS is low:

1. Use lite model:
   ```python
   backend = MediaPipeBackend(model_complexity=0)
   ```

2. Reduce resolution:
   ```bash
   python demos/webcam_demo.py --width 640 --height 480
   ```

3. Disable angle display:
   ```bash
   python demos/webcam_demo.py --no-angles
   ```

## Development Installation

For development with testing and documentation tools:

```bash
pip install -e ".[dev]"
```

This installs additional packages:
- pytest (testing)
- black (code formatting)
- flake8 (linting)
- mypy (type checking)

## Uninstallation

To remove the project:

```bash
# Deactivate virtual environment
deactivate

# Remove directory
cd ..
rm -rf motion-tracker
```

## Getting Help

- GitHub Issues: https://github.com/MindDock/motion-tracker/issues
- Documentation: See `docs/` directory
- MediaPipe Docs: https://ai.google.dev/edge/mediapipe

## Next Steps

After installation:
- Read [QUICKSTART.md](../QUICKSTART.md) for usage examples
- Check [README.md](../README.md) for full documentation
- Try the demos in `demos/` directory
