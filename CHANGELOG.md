# Changelog

All notable changes to this project will be documented in this file.

## [0.1.3] - 2026-01-22

### Fixed
- **Skeleton rendering improvements**:
  - Added neck connections (nose and ears to shoulders)
  - Enhanced angle display with larger font and highlighting
  - Fixed angle text encoding (° -> deg)
  - Added colored circles around joints showing angles
  - Now displays 8 major joint angles clearly

### Changed
- Angle display now filters to major joints only (elbows, shoulders, knees, hips)
- Increased font size for angle text (0.5 -> 0.6)
- Added visual highlighting circle for joints with angles

## [0.1.2] - 2026-01-22

### Added
- **Dance Coach Demo** (`dance_coach_demo.py`):
  - Record reference dance sequences (3-10 seconds)
  - Real-time movement comparison with DTW (Dynamic Time Warping)
  - Joint-by-joint feedback with color-coded status
  - Overall performance scoring (0-100)
  - Save/load reference sequences to file
  - 8 key joints comparison (elbows, shoulders, knees, hips)
- **DanceSequence class**: Store and manage pose sequences
- **DTWMatcher class**: Dynamic Time Warping algorithm for sequence alignment

### Technical Details
- Implements DTW algorithm for temporal alignment
- Normalizes scores based on sequence length and angle differences
- Real-time feedback with <15deg = good, 15-30deg = ok, >30deg = needs work
- Pickle-based serialization for saving reference sequences

## [0.1.1] - 2026-01-22

### Added
- **Enhanced posture metrics**:
  - Head tilt angle (side-to-side)
  - Neck forward/backward angle
  - Body lean angle
  - Shoulder tilt angle
  - Hip tilt angle
  - Spine curvature
- **Expanded joint angle display**:
  - Now shows 8 major joints (elbows, shoulders, knees, hips)
  - Dual panel display: posture metrics (left) + joint angles (right)
- **New AngleCalculator methods**:
  - `calculate_head_tilt()` - detects head tilting
  - `calculate_neck_angle()` - measures neck posture
  - `calculate_body_lean()` - tracks body lean forward/backward
  - `calculate_shoulder_tilt()` - detects shoulder imbalance
  - `calculate_hip_tilt()` - detects hip imbalance
  - `calculate_spine_curve()` - measures spine curvature
  - `calculate_posture_metrics()` - comprehensive posture analysis

### Changed
- Replaced Unicode symbols (°, ✓, ✗) with ASCII equivalents (deg, [OK], [!])
  - Fixes display issues on terminals without Unicode support
- Updated webcam demo to show comprehensive posture analysis
- Improved stats panel layout with separate posture and joint angle sections

### Fixed
- Fixed display encoding issues with special characters
- MediaPipe 0.10+ compatibility (migrated to Tasks API)
- Model download functionality

## [0.1.0] - 2026-01-22

### Added
- Initial release
- Core pose estimation framework
- MediaPipe backend with 33 keypoints
- Angle calculation module
- Motion analysis module
- Skeleton rendering
- Three demo applications:
  - Real-time webcam pose detection
  - Posture correction monitoring
  - AI fitness trainer (4 exercises)
- Complete documentation
- MIT License
