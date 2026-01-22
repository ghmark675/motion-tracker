#!/usr/bin/env python3
"""Posture correction demo for sitting/standing posture monitoring.

This demo monitors your posture in real-time and provides visual feedback
when your posture deviates from ideal alignment.

Usage:
    python demos/posture_correction_demo.py [--camera CAMERA_ID]

Controls:
    - Press 'q' to quit
    - Press 'c' to calibrate good posture
    - Press 's' to save screenshot
"""

import sys
import argparse
from pathlib import Path
import time
import cv2
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backends.mediapipe_backend import MediaPipeBackend
from src.core.angle_calculator import AngleCalculator
from src.core.motion_analyzer import MotionAnalyzer
from src.visualization.skeleton_renderer import SkeletonRenderer


class PostureMonitor:
    """Monitor and evaluate posture quality."""

    def __init__(self):
        """Initialize posture monitor."""
        self.calibrated_angles = None
        self.angle_tolerance = 15.0  # degrees

        # Define posture rules
        self.posture_rules = {
            'neck_forward': {
                'description': 'Head forward position',
                'check': self._check_neck_forward,
            },
            'shoulders_level': {
                'description': 'Shoulder alignment',
                'check': self._check_shoulders_level,
            },
            'back_straight': {
                'description': 'Back straightness',
                'check': self._check_back_straight,
            },
        }

    def calibrate(self, pose_result, angle_calculator):
        """Calibrate with current good posture.

        Args:
            pose_result: Current pose
            angle_calculator: Angle calculator instance
        """
        self.calibrated_angles = angle_calculator.calculate_all_angles(pose_result)
        return True

    def evaluate(self, pose_result, angle_calculator):
        """Evaluate current posture.

        Args:
            pose_result: Current pose
            angle_calculator: Angle calculator instance

        Returns:
            Dictionary of evaluation results
        """
        results = {}

        for rule_name, rule_spec in self.posture_rules.items():
            results[rule_name] = rule_spec['check'](pose_result, angle_calculator)

        return results

    def _check_neck_forward(self, pose_result, angle_calculator):
        """Check if neck is too far forward."""
        # Get nose and shoulder positions
        nose = pose_result.get_keypoint('nose')
        left_shoulder = pose_result.get_keypoint('left_shoulder')
        right_shoulder = pose_result.get_keypoint('right_shoulder')

        if not all([nose, left_shoulder, right_shoulder]):
            return {'status': 'unknown', 'message': 'Cannot detect'}

        # Calculate shoulder midpoint
        shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2

        # Check if nose is too far forward
        forward_threshold = 0.1  # normalized coordinates
        forward_distance = nose.x - shoulder_mid_x

        if abs(forward_distance) > forward_threshold:
            return {
                'status': 'bad',
                'message': f'Head too far {"forward" if forward_distance > 0 else "back"}'
            }

        return {'status': 'good', 'message': 'Good head position'}

    def _check_shoulders_level(self, pose_result, angle_calculator):
        """Check if shoulders are level."""
        left_shoulder = pose_result.get_keypoint('left_shoulder')
        right_shoulder = pose_result.get_keypoint('right_shoulder')

        if not all([left_shoulder, right_shoulder]):
            return {'status': 'unknown', 'message': 'Cannot detect'}

        # Calculate shoulder tilt
        dy = abs(left_shoulder.y - right_shoulder.y)
        tilt_threshold = 0.05  # normalized coordinates

        if dy > tilt_threshold:
            return {
                'status': 'bad',
                'message': f'Shoulders uneven (tilt: {dy*100:.1f}%)'
            }

        return {'status': 'good', 'message': 'Shoulders level'}

    def _check_back_straight(self, pose_result, angle_calculator):
        """Check if back is straight."""
        # Get hip and shoulder angles
        left_hip_angle = angle_calculator.calculate_joint_angle(pose_result, 'left_hip')

        if left_hip_angle is None:
            return {'status': 'unknown', 'message': 'Cannot detect'}

        # Ideal sitting hip angle is around 90-110 degrees
        if 85 <= left_hip_angle <= 115:
            return {'status': 'good', 'message': 'Good sitting posture'}
        elif left_hip_angle < 85:
            return {'status': 'bad', 'message': 'Leaning too far forward'}
        else:
            return {'status': 'bad', 'message': 'Leaning back'}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Posture correction demo')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    return parser.parse_args()


def main():
    """Main demo function."""
    args = parse_args()

    print("=" * 60)
    print("Motion Tracker - Posture Correction Demo")
    print("=" * 60)
    print("\nSit in your normal posture and press 'c' to calibrate")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 'c' to calibrate good posture")
    print("  - Press 's' to save screenshot")
    print("=" * 60)

    # Initialize components
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    estimator = MediaPipeBackend(model_complexity=1)
    estimator.initialize()

    angle_calculator = AngleCalculator(use_3d=True)
    posture_monitor = PostureMonitor()
    renderer = SkeletonRenderer()

    is_calibrated = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Process pose
            pose_result = estimator.process_frame(frame)

            if pose_result and pose_result.is_valid():
                # Calculate angles
                angles = angle_calculator.calculate_all_angles(pose_result)

                # Render skeleton
                frame = renderer.render(frame, pose_result)

                # Evaluate posture
                if is_calibrated:
                    evaluation = posture_monitor.evaluate(pose_result, angle_calculator)

                    # Display evaluation results
                    y_offset = 50
                    overall_status = 'good'

                    for rule_name, result in evaluation.items():
                        status = result['status']
                        message = result['message']

                        if status == 'bad':
                            overall_status = 'bad'
                            color = (0, 0, 255)  # Red
                        elif status == 'good':
                            color = (0, 255, 0)  # Green
                        else:
                            color = (128, 128, 128)  # Gray

                        cv2.putText(
                            frame,
                            f"{rule_name}: {message}",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )
                        y_offset += 30

                    # Overall feedback
                    if overall_status == 'good':
                        feedback = "POSTURE: GOOD [OK]"
                        color = (0, 255, 0)
                    else:
                        feedback = "POSTURE: NEEDS CORRECTION [!]"
                        color = (0, 0, 255)

                    cv2.putText(
                        frame,
                        feedback,
                        (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        color,
                        3,
                    )
                else:
                    # Show calibration prompt
                    cv2.putText(
                        frame,
                        "Press 'c' to calibrate with good posture",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                    )

            cv2.imshow('Posture Correction Demo', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c') and pose_result:
                is_calibrated = posture_monitor.calibrate(pose_result, angle_calculator)
                print("[OK] Posture calibrated")
            elif key == ord('s'):
                cv2.imwrite(f"posture_{int(time.time())}.png", frame)
                print("Screenshot saved")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        estimator.release()

    return 0


if __name__ == '__main__':
    sys.exit(main())
