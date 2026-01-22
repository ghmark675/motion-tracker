#!/usr/bin/env python3
"""Real-time webcam pose estimation demo.

This demo captures video from your webcam and performs real-time pose estimation
with joint angle calculations and visualization.

Usage:
    python demos/webcam_demo.py [--camera CAMERA_ID] [--backend BACKEND]

Controls:
    - Press 'q' to quit
    - Press 's' to save screenshot
    - Press 'r' to reset statistics
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Real-time pose estimation webcam demo')
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='mediapipe',
        choices=['mediapipe'],
        help='Pose estimation backend (default: mediapipe)'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=1280,
        help='Camera frame width (default: 1280)'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=720,
        help='Camera frame height (default: 720)'
    )
    parser.add_argument(
        '--show-fps',
        action='store_true',
        help='Show FPS counter'
    )
    parser.add_argument(
        '--no-angles',
        action='store_true',
        help='Disable angle display'
    )
    return parser.parse_args()


def main():
    """Main demo function."""
    args = parse_args()

    print("=" * 60)
    print("Motion Tracker - Real-time Webcam Demo")
    print("=" * 60)
    print(f"Camera ID: {args.camera}")
    print(f"Backend: {args.backend}")
    print(f"Resolution: {args.width}x{args.height}")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("  - Press 'r' to reset statistics")
    print("=" * 60)

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        return 1

    # Initialize pose estimator
    print("\nInitializing pose estimator...")
    if args.backend == 'mediapipe':
        estimator = MediaPipeBackend(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    else:
        print(f"Unknown backend: {args.backend}")
        return 1

    if not estimator.initialize():
        print("Error: Failed to initialize pose estimator")
        return 1

    print(f"[OK] Initialized {estimator.backend_name} backend")
    print(f"[OK] Detecting {estimator.num_keypoints} keypoints")

    # Initialize angle calculator and motion analyzer
    angle_calculator = AngleCalculator(use_3d=True)
    motion_analyzer = MotionAnalyzer(buffer_size=30, smoothing_window=5)

    # Initialize renderer
    renderer = SkeletonRenderer(
        show_keypoints=True,
        show_connections=True,
        show_labels=False,
    )

    # FPS calculation
    fps_history = []
    frame_count = 0
    start_time = time.time()
    screenshot_count = 0

    print("\nStarting real-time detection...\n")

    try:
        while True:
            loop_start = time.time()

            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Process pose
            pose_result = estimator.process_frame(frame)

            if pose_result and pose_result.is_valid():
                # Update motion analyzer
                motion_analyzer.update(pose_result)

                # Calculate angles
                angles = None
                if not args.no_angles:
                    angles = angle_calculator.calculate_all_angles(pose_result)

                # Render skeleton and angles
                frame = renderer.render(frame, pose_result, angles)

                # Calculate posture metrics
                posture_metrics = angle_calculator.calculate_posture_metrics(pose_result)

                # Prepare statistics - show in two panels
                # Left panel: Posture metrics
                posture_stats = {}
                if posture_metrics.get('head_tilt') is not None:
                    posture_stats['Head Tilt'] = f"{posture_metrics['head_tilt']:.1f}deg"
                if posture_metrics.get('neck_angle') is not None:
                    posture_stats['Neck Angle'] = f"{posture_metrics['neck_angle']:.1f}deg"
                if posture_metrics.get('body_lean') is not None:
                    posture_stats['Body Lean'] = f"{posture_metrics['body_lean']:.1f}deg"
                if posture_metrics.get('shoulder_tilt') is not None:
                    posture_stats['Shoulder Tilt'] = f"{posture_metrics['shoulder_tilt']:.1f}deg"
                if posture_metrics.get('spine_curve') is not None:
                    posture_stats['Spine Curve'] = f"{posture_metrics['spine_curve']:.1f}deg"

                # Right panel: Joint angles
                joint_stats = {
                    'Confidence': f"{pose_result.confidence * 100:.0f}%",
                }

                # Add key angles to stats
                if angles and not args.no_angles:
                    # Show major joints
                    key_angles = [
                        'left_elbow', 'right_elbow',
                        'left_shoulder', 'right_shoulder',
                        'left_knee', 'right_knee',
                        'left_hip', 'right_hip',
                    ]
                    for joint in key_angles:
                        if angles.get(joint) is not None:
                            smoothed = motion_analyzer.get_smoothed_angle(joint)
                            if smoothed:
                                # Format joint name nicely
                                joint_name = joint.replace('_', ' ').title()
                                # Abbreviate for space
                                joint_name = joint_name.replace('Left', 'L').replace('Right', 'R')
                                joint_stats[joint_name] = f"{smoothed:.0f}deg"

                # Add FPS if enabled
                if args.show_fps:
                    current_fps = 1.0 / (time.time() - loop_start + 1e-6)
                    fps_history.append(current_fps)
                    if len(fps_history) > 30:
                        fps_history.pop(0)
                    avg_fps = np.mean(fps_history)
                    joint_stats['FPS'] = f"{avg_fps:.1f}"

                # Draw stats panels
                if posture_stats:
                    frame = renderer.draw_stats_panel(frame, posture_stats, position='top_left')
                frame = renderer.draw_stats_panel(frame, joint_stats, position='top_right')

            else:
                # No pose detected
                cv2.putText(
                    frame,
                    "No pose detected",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )

            # Display frame
            cv2.imshow('Motion Tracker - Webcam Demo', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save screenshot
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count:03d}.png"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r'):
                # Reset statistics
                motion_analyzer.clear_history()
                fps_history.clear()
                print("Statistics reset")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Cleanup
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        print("\n" + "=" * 60)
        print("Session Summary")
        print("=" * 60)
        print(f"Total frames: {frame_count}")
        print(f"Duration: {elapsed_time:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print("=" * 60)

        cap.release()
        cv2.destroyAllWindows()
        estimator.release()

    return 0


if __name__ == '__main__':
    sys.exit(main())
