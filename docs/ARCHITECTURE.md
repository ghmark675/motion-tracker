# Architecture Overview

This document describes the architecture and design decisions of the Motion Tracker system.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│  (Posture Correction, Fitness Trainer, Dance Coach, etc.)   │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    Visualization Layer                       │
│          (Skeleton Renderer, AR Overlays, UI)                │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      Analysis Layer                          │
│    (Motion Analyzer, Angle Calculator, Pattern Recognition)  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                     Core Abstraction                         │
│            (PoseEstimator Interface, Data Models)            │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      Backend Layer                           │
│     (MediaPipe, Apple Vision, YOLO, Custom Implementations)  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                      Hardware Layer                          │
│            (Camera, GPU, Neural Engine, CPU)                 │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Core Layer (`src/core/`)

#### PoseEstimator (Abstract Base Class)

The `PoseEstimator` class defines the interface that all backend implementations must follow:

```python
class PoseEstimator(ABC):
    @abstractmethod
    def initialize() -> bool

    @abstractmethod
    def process_frame(frame: np.ndarray) -> Optional[PoseResult]

    @abstractmethod
    def release()

    @abstractmethod
    def get_keypoint_names() -> List[str]
```

**Design Decisions:**
- Abstract interface allows easy swapping of backends
- Context manager support (`__enter__`, `__exit__`)
- Consistent data models across all backends

#### Data Models

**Keypoint:**
- Normalized coordinates (0-1) for resolution independence
- Optional world coordinates in meters for 3D analysis
- Visibility and presence scores for confidence filtering

**PoseResult:**
- Collection of keypoints with metadata
- Timestamp for temporal analysis
- Overall confidence score
- Helper methods for keypoint lookup

#### AngleCalculator

Calculates joint angles from keypoints using vector mathematics:

```
angle = arccos((ba · bc) / (|ba| * |bc|))
```

**Features:**
- Predefined joint definitions for common angles
- Support for custom angle calculations
- 2D and 3D coordinate support
- Robust to missing keypoints

#### MotionAnalyzer

Temporal analysis of pose sequences:

**Features:**
- Circular buffer for efficient history storage
- Temporal smoothing (moving average, exponential)
- Rep counting with state machine
- Statistical analysis (min, max, mean, std)
- Posture rule evaluation

### 2. Backend Layer (`src/backends/`)

#### MediaPipeBackend

Wrapper around Google's MediaPipe Pose solution:

**Specifications:**
- 33 keypoints (full body + face + hands)
- 3D world landmarks in meters
- Native Apple Silicon support (MediaPipe 0.10+)
- Real-time performance (30+ FPS on CPU)

**Keypoint Coverage:**
- Face: nose, eyes, ears, mouth
- Upper body: shoulders, elbows, wrists, hands
- Lower body: hips, knees, ankles, feet

#### Future Backends

**Apple Vision Backend** (planned):
- Native macOS/iOS integration
- Neural Engine acceleration
- ARKit compatibility
- 19 keypoints

**YOLO11 Backend** (planned):
- Multi-person detection
- 17 keypoints
- GPU acceleration
- Real-time (100+ FPS)

### 3. Visualization Layer (`src/visualization/`)

#### SkeletonRenderer

Responsible for all visual output:

**Features:**
- Customizable colors and styles
- Keypoint and connection rendering
- Angle annotations with color coding
- Statistical panels
- Semi-transparent overlays

**Design:**
- Stateless rendering (pure functions)
- Separation of concerns (rendering vs. analysis)
- Configurable visibility thresholds

### 4. Application Layer (`src/applications/`)

Domain-specific implementations built on top of core components:

**Posture Correction:**
- Rule-based evaluation
- Calibration system
- Real-time feedback

**Fitness Trainer:**
- Exercise-specific state machines
- Rep counting algorithms
- Form validation

**Dance Coach (planned):**
- Motion sequence comparison
- Temporal alignment (DTW)
- Scoring system

## Design Patterns

### 1. Strategy Pattern

Backend implementations follow the strategy pattern:
- `PoseEstimator` is the interface
- `MediaPipeBackend`, `VisionBackend`, etc. are concrete strategies
- Applications can switch backends without code changes

### 2. Template Method

`MotionAnalyzer` uses template method for extensibility:
- Base smoothing algorithms provided
- Subclasses can override specific behavior
- Customizable evaluation rules

### 3. Builder Pattern

Configuration objects use builder pattern:
```python
estimator = MediaPipeBackend(
    model_complexity=1,
    min_detection_confidence=0.5,
    smooth_landmarks=True
)
```

## Data Flow

### Typical Processing Pipeline

```
1. Camera Capture
   └─> BGR frame (H x W x 3)

2. Backend Processing
   └─> PoseResult with keypoints

3. Analysis
   ├─> AngleCalculator: joint angles
   ├─> MotionAnalyzer: temporal smoothing
   └─> Application logic: domain-specific analysis

4. Visualization
   ├─> Skeleton rendering
   ├─> Angle annotations
   └─> Statistical overlays

5. Display
   └─> Annotated frame output
```

### Performance Optimizations

1. **Lazy Computation:**
   - Angles calculated only when needed
   - World coordinates optional

2. **Temporal Smoothing:**
   - Reduces jitter in real-time tracking
   - Configurable window size

3. **Visibility Filtering:**
   - Skip rendering for low-confidence keypoints
   - Improves visual clarity

4. **Circular Buffers:**
   - Efficient memory usage for history
   - O(1) insertion and removal

## Coordinate Systems

### Image Coordinates (2D)

- Normalized: (0, 0) = top-left, (1, 1) = bottom-right
- Pixel coordinates computed as: `(x * width, y * height)`

### World Coordinates (3D)

- Origin: midpoint between hips
- Units: meters
- Z-axis: depth (negative = closer to camera)
- Y-axis: vertical (positive = up)
- X-axis: horizontal (positive = right)

## Error Handling

### Graceful Degradation

1. **Missing Keypoints:**
   - Return `None` instead of crashing
   - Skip calculations for invalid data
   - Provide visual feedback to user

2. **Backend Failures:**
   - Check initialization success
   - Validate results before processing
   - Fallback to alternative backends

3. **Camera Issues:**
   - Multiple camera ID attempts
   - Clear error messages
   - Retry mechanisms

## Testing Strategy

### Unit Tests

- Core algorithms (angle calculation, smoothing)
- Data model validation
- Utility functions

### Integration Tests

- Backend initialization
- End-to-end processing pipeline
- Multi-backend comparison

### Performance Tests

- FPS benchmarks
- Latency measurements
- Memory profiling

## Future Extensions

### Planned Features

1. **Multi-Person Tracking:**
   - Person ID assignment
   - Tracking across frames
   - Occlusion handling

2. **3D Reconstruction:**
   - Multi-camera calibration
   - Triangulation
   - Depth estimation

3. **AR Integration:**
   - ARKit overlay support
   - Virtual training guides
   - Real-time form comparison

4. **ML Enhancements:**
   - Custom pose classification
   - Action recognition
   - Anomaly detection

### Extensibility Points

1. **Custom Backends:**
   - Implement `PoseEstimator` interface
   - Add backend-specific optimizations

2. **Custom Applications:**
   - Inherit from base classes
   - Implement domain logic
   - Reuse core components

3. **Custom Renderers:**
   - Extend `SkeletonRenderer`
   - Add specialized visualizations
   - AR/VR output support

## Performance Targets

| Platform | Backend | Resolution | Target FPS |
|----------|---------|------------|------------|
| M4 Mac | MediaPipe | 720p | 35-40 |
| M4 Mac | Vision | 1080p | 60 |
| M4 Mac | YOLO11 | 720p | 120+ |
| M1 Mac | MediaPipe | 720p | 30-35 |
| Intel Mac | MediaPipe | 720p | 25-30 |

## References

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [Apple Vision Framework](https://developer.apple.com/documentation/vision)
- [BlazePose Paper](https://arxiv.org/abs/2006.10204)
- [YOLO11 Documentation](https://docs.ultralytics.com/)
