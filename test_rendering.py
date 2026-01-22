#!/usr/bin/env python3
"""Test rendering of skeleton and angles."""

import sys
import numpy as np
import cv2
from src.core.pose_estimator import PoseResult, Keypoint
from src.core.angle_calculator import AngleCalculator
from src.visualization.skeleton_renderer import SkeletonRenderer


def create_test_pose():
    """Create a complete test pose."""
    keypoints = [
        # Face
        Keypoint('nose', 0.5, 0.15, 0, 1.0, 1.0, 0, 0.5, 0.2),
        Keypoint('left_eye', 0.47, 0.12, 0, 1.0, 1.0),
        Keypoint('right_eye', 0.53, 0.12, 0, 1.0, 1.0),
        Keypoint('left_ear', 0.42, 0.15, 0, 1.0, 1.0),
        Keypoint('right_ear', 0.58, 0.15, 0, 1.0, 1.0),
        Keypoint('mouth_left', 0.48, 0.18, 0, 1.0, 1.0),
        Keypoint('mouth_right', 0.52, 0.18, 0, 1.0, 1.0),
        Keypoint('left_eye_inner', 0.48, 0.12, 0, 1.0, 1.0),
        Keypoint('left_eye_outer', 0.46, 0.12, 0, 1.0, 1.0),
        Keypoint('right_eye_inner', 0.52, 0.12, 0, 1.0, 1.0),
        Keypoint('right_eye_outer', 0.54, 0.12, 0, 1.0, 1.0),

        # Upper body
        Keypoint('left_shoulder', 0.4, 0.3, 0, 1.0, 1.0, -0.2, 0.3, 0.0),
        Keypoint('right_shoulder', 0.6, 0.3, 0, 1.0, 1.0, 0.2, 0.3, 0.0),
        Keypoint('left_elbow', 0.35, 0.5, 0, 1.0, 1.0, -0.3, 0.1, -0.1),
        Keypoint('right_elbow', 0.65, 0.5, 0, 1.0, 1.0, 0.3, 0.1, -0.1),
        Keypoint('left_wrist', 0.33, 0.7, 0, 1.0, 1.0, -0.35, -0.1, -0.2),
        Keypoint('right_wrist', 0.67, 0.7, 0, 1.0, 1.0, 0.35, -0.1, -0.2),

        # Hands
        Keypoint('left_pinky', 0.32, 0.72, 0, 1.0, 1.0),
        Keypoint('right_pinky', 0.68, 0.72, 0, 1.0, 1.0),
        Keypoint('left_index', 0.34, 0.72, 0, 1.0, 1.0),
        Keypoint('right_index', 0.66, 0.72, 0, 1.0, 1.0),
        Keypoint('left_thumb', 0.35, 0.71, 0, 1.0, 1.0),
        Keypoint('right_thumb', 0.65, 0.71, 0, 1.0, 1.0),

        # Lower body
        Keypoint('left_hip', 0.42, 0.6, 0, 1.0, 1.0, -0.18, -0.1, -0.3),
        Keypoint('right_hip', 0.58, 0.6, 0, 1.0, 1.0, 0.18, -0.1, -0.3),
        Keypoint('left_knee', 0.41, 0.8, 0, 1.0, 1.0, -0.19, -0.3, -0.5),
        Keypoint('right_knee', 0.59, 0.8, 0, 1.0, 1.0, 0.19, -0.3, -0.5),
        Keypoint('left_ankle', 0.40, 0.95, 0, 1.0, 1.0, -0.20, -0.5, -0.7),
        Keypoint('right_ankle', 0.60, 0.95, 0, 1.0, 1.0, 0.20, -0.5, -0.7),
        Keypoint('left_heel', 0.39, 0.97, 0, 1.0, 1.0),
        Keypoint('right_heel', 0.61, 0.97, 0, 1.0, 1.0),
        Keypoint('left_foot_index', 0.41, 0.98, 0, 1.0, 1.0),
        Keypoint('right_foot_index', 0.59, 0.98, 0, 1.0, 1.0),
    ]

    return PoseResult(keypoints=keypoints, confidence=0.95, image_width=640, image_height=480)


def test_rendering():
    """Test rendering with visualization."""
    print("=" * 60)
    print("Testing Skeleton Rendering")
    print("=" * 60)

    # Create test pose
    print("\n[1/4] Creating test pose...")
    pose_result = create_test_pose()
    print(f"[OK] Created pose with {len(pose_result.keypoints)} keypoints")

    # Calculate angles
    print("\n[2/4] Calculating angles...")
    calculator = AngleCalculator(use_3d=True)
    angles = calculator.calculate_all_angles(pose_result)

    angle_count = sum(1 for a in angles.values() if a is not None)
    print(f"[OK] Calculated {angle_count} joint angles")

    # Print some angles
    major_joints = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']
    for joint in major_joints:
        if angles.get(joint):
            print(f"  {joint}: {angles[joint]:.1f}deg")

    # Initialize renderer
    print("\n[3/4] Initializing renderer...")
    renderer = SkeletonRenderer(
        show_keypoints=True,
        show_connections=True,
        show_labels=False,
        line_thickness=2,
        keypoint_radius=4,
    )
    print("[OK] Renderer initialized")
    print(f"  Connections: {len(renderer.CONNECTIONS)}")
    print(f"  Show keypoints: {renderer.show_keypoints}")
    print(f"  Show connections: {renderer.show_connections}")

    # Render
    print("\n[4/4] Rendering...")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = renderer.render(frame, pose_result, angles)
    print("[OK] Rendered skeleton")

    # Count non-zero pixels to verify something was drawn
    non_zero = np.count_nonzero(rendered)
    print(f"  Non-zero pixels: {non_zero}")

    # Display
    print("\n" + "=" * 60)
    print("Displaying rendered skeleton")
    print("=" * 60)
    print("A window should appear showing:")
    print("  - Yellow skeleton connections (including neck!)")
    print("  - Green keypoint circles")
    print("  - Angle values next to major joints")
    print("\nPress any key in the window to close it.")
    print("=" * 60)

    cv2.imshow('Skeleton Rendering Test', rendered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[OK] Test completed!")
    print("\nIf you saw the skeleton with:")
    print("  1. Yellow lines connecting body parts (including neck)")
    print("  2. Green circles at joints")
    print("  3. Angle numbers (e.g., '168deg') next to elbows/knees")
    print("\nThen the rendering is working correctly!")


if __name__ == '__main__':
    try:
        test_rendering()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
