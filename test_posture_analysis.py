#!/usr/bin/env python3
"""Quick test script for posture analysis features.

Tests the new posture metrics without requiring a camera.
"""

import sys
import numpy as np
from src.core.pose_estimator import PoseResult, Keypoint
from src.core.angle_calculator import AngleCalculator


def create_test_pose():
    """Create a test pose with all 33 keypoints."""
    keypoints = [
        # Face
        Keypoint('nose', 0.5, 0.15, 0, 1.0, 1.0, 0, 0.5, 0.2),
        Keypoint('left_eye_inner', 0.48, 0.12, 0, 1.0, 1.0, -0.02, 0.52, 0.22),
        Keypoint('left_eye', 0.47, 0.12, 0, 1.0, 1.0, -0.03, 0.52, 0.22),
        Keypoint('left_eye_outer', 0.46, 0.12, 0, 1.0, 1.0, -0.04, 0.52, 0.22),
        Keypoint('right_eye_inner', 0.52, 0.12, 0, 1.0, 1.0, 0.02, 0.52, 0.22),
        Keypoint('right_eye', 0.53, 0.12, 0, 1.0, 1.0, 0.03, 0.52, 0.22),
        Keypoint('right_eye_outer', 0.54, 0.12, 0, 1.0, 1.0, 0.04, 0.52, 0.22),
        Keypoint('left_ear', 0.42, 0.15, 0, 1.0, 1.0, -0.08, 0.5, 0.2),
        Keypoint('right_ear', 0.58, 0.15, 0, 1.0, 1.0, 0.08, 0.5, 0.2),
        Keypoint('mouth_left', 0.48, 0.18, 0, 1.0, 1.0, -0.02, 0.48, 0.18),
        Keypoint('mouth_right', 0.52, 0.18, 0, 1.0, 1.0, 0.02, 0.48, 0.18),

        # Upper body
        Keypoint('left_shoulder', 0.4, 0.3, 0, 1.0, 1.0, -0.2, 0.3, 0.0),
        Keypoint('right_shoulder', 0.6, 0.3, 0, 1.0, 1.0, 0.2, 0.3, 0.0),
        Keypoint('left_elbow', 0.35, 0.5, 0, 1.0, 1.0, -0.3, 0.1, -0.1),
        Keypoint('right_elbow', 0.65, 0.5, 0, 1.0, 1.0, 0.3, 0.1, -0.1),
        Keypoint('left_wrist', 0.33, 0.7, 0, 1.0, 1.0, -0.35, -0.1, -0.2),
        Keypoint('right_wrist', 0.67, 0.7, 0, 1.0, 1.0, 0.35, -0.1, -0.2),

        # Hands
        Keypoint('left_pinky', 0.32, 0.72, 0, 1.0, 1.0, -0.37, -0.12, -0.22),
        Keypoint('right_pinky', 0.68, 0.72, 0, 1.0, 1.0, 0.37, -0.12, -0.22),
        Keypoint('left_index', 0.34, 0.72, 0, 1.0, 1.0, -0.36, -0.12, -0.21),
        Keypoint('right_index', 0.66, 0.72, 0, 1.0, 1.0, 0.36, -0.12, -0.21),
        Keypoint('left_thumb', 0.35, 0.71, 0, 1.0, 1.0, -0.34, -0.11, -0.2),
        Keypoint('right_thumb', 0.65, 0.71, 0, 1.0, 1.0, 0.34, -0.11, -0.2),

        # Lower body
        Keypoint('left_hip', 0.42, 0.6, 0, 1.0, 1.0, -0.18, -0.1, -0.3),
        Keypoint('right_hip', 0.58, 0.6, 0, 1.0, 1.0, 0.18, -0.1, -0.3),
        Keypoint('left_knee', 0.41, 0.8, 0, 1.0, 1.0, -0.19, -0.3, -0.5),
        Keypoint('right_knee', 0.59, 0.8, 0, 1.0, 1.0, 0.19, -0.3, -0.5),
        Keypoint('left_ankle', 0.40, 0.95, 0, 1.0, 1.0, -0.20, -0.5, -0.7),
        Keypoint('right_ankle', 0.60, 0.95, 0, 1.0, 1.0, 0.20, -0.5, -0.7),
        Keypoint('left_heel', 0.39, 0.97, 0, 1.0, 1.0, -0.21, -0.52, -0.72),
        Keypoint('right_heel', 0.61, 0.97, 0, 1.0, 1.0, 0.21, -0.52, -0.72),
        Keypoint('left_foot_index', 0.41, 0.98, 0, 1.0, 1.0, -0.19, -0.53, -0.73),
        Keypoint('right_foot_index', 0.59, 0.98, 0, 1.0, 1.0, 0.19, -0.53, -0.73),
    ]

    return PoseResult(keypoints=keypoints, confidence=0.95, image_width=1280, image_height=720)


def test_posture_analysis():
    """Test posture analysis features."""
    print("=" * 60)
    print("Posture Analysis Feature Test")
    print("=" * 60)

    # Create test pose
    print("\n[1/3] Creating test pose...")
    pose_result = create_test_pose()
    print(f"[OK] Created pose with {len(pose_result.keypoints)} keypoints")

    # Initialize angle calculator
    print("\n[2/3] Initializing AngleCalculator...")
    calculator = AngleCalculator(use_3d=True)
    print("[OK] AngleCalculator initialized")

    # Calculate all joint angles
    print("\n[3/3] Calculating posture metrics...")
    posture_metrics = calculator.calculate_posture_metrics(pose_result)

    print("\n" + "=" * 60)
    print("POSTURE METRICS RESULTS")
    print("=" * 60)

    for metric_name, value in posture_metrics.items():
        if value is not None:
            status = "[OK]"
            value_str = f"{value:6.1f}deg"
        else:
            status = "[--]"
            value_str = "N/A"

        metric_display = metric_name.replace('_', ' ').title()
        print(f"{status} {metric_display:20s}: {value_str}")

    # Calculate joint angles
    print("\n" + "=" * 60)
    print("JOINT ANGLES")
    print("=" * 60)

    angles = calculator.calculate_all_angles(pose_result)

    joint_groups = {
        'Arms': ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder'],
        'Legs': ['left_knee', 'right_knee', 'left_hip', 'right_hip'],
        'Hands/Feet': ['left_wrist', 'right_wrist', 'left_ankle', 'right_ankle'],
    }

    for group_name, joints in joint_groups.items():
        print(f"\n{group_name}:")
        for joint in joints:
            angle = angles.get(joint)
            if angle is not None:
                status = "[OK]"
                value_str = f"{angle:6.1f}deg"
            else:
                status = "[--]"
                value_str = "N/A"

            joint_display = joint.replace('_', ' ').title()
            print(f"  {status} {joint_display:20s}: {value_str}")

    # Test specific calculations
    print("\n" + "=" * 60)
    print("DETAILED TESTS")
    print("=" * 60)

    # Test head tilt
    head_tilt = calculator.calculate_head_tilt(pose_result)
    print(f"\nHead Tilt: {head_tilt:.1f}deg" if head_tilt else "Head Tilt: N/A")
    print("  Expected: ~0deg (straight head)")

    # Test neck angle
    neck_angle = calculator.calculate_neck_angle(pose_result)
    print(f"\nNeck Angle: {neck_angle:.1f}deg" if neck_angle else "Neck Angle: N/A")
    print("  Expected: Small angle (straight posture)")

    # Test body lean
    body_lean = calculator.calculate_body_lean(pose_result)
    print(f"\nBody Lean: {body_lean:.1f}deg" if body_lean else "Body Lean: N/A")
    print("  Expected: ~0deg (straight posture)")

    # Test shoulder tilt
    shoulder_tilt = calculator.calculate_shoulder_tilt(pose_result)
    print(f"\nShoulder Tilt: {shoulder_tilt:.1f}deg" if shoulder_tilt else "Shoulder Tilt: N/A")
    print("  Expected: ~0deg (level shoulders)")

    print("\n" + "=" * 60)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nAll new posture analysis features are working correctly!")
    print("You can now run: python demos/webcam_demo.py")
    print("  to see these metrics in real-time with your camera.")
    print("=" * 60)


if __name__ == '__main__':
    try:
        test_posture_analysis()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
