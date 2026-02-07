#!/usr/bin/env python3
"""Real-time pose estimation demo (Webcam or Local Video) with Recording."""

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
    parser = argparse.ArgumentParser(description='Pose estimation demo (Webcam or Video)')
    
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input video file. If not provided, webcam will be used.'
    )
    
    # Output argument for saving the result
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save the output video (e.g., output.mp4). If not set, video is not saved.'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0). Used only if --input is not set.'
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
        help='Camera frame width (default: 1280). Ignored for video files.'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=720,
        help='Camera frame height (default: 720). Ignored for video files.'
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
    print("Motion Tracker - Demo")
    print("=" * 60)

    is_video_file = args.input is not None
    
    if is_video_file:
        print(f"Input Mode: Local Video File")
        print(f"File Path: {args.input}")
        cap = cv2.VideoCapture(args.input)
    else:
        print(f"Input Mode: Webcam")
        print(f"Camera ID: {args.camera}")
        print(f"Target Resolution: {args.width}x{args.height}")
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    print(f"Backend: {args.backend}")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save screenshot")
    print("  - Press 'r' to reset statistics")
    print("=" * 60)

    if not cap.isOpened():
        source_name = args.input if is_video_file else f"Camera {args.camera}"
        print(f"Error: Could not open {source_name}")
        return 1

    # ---------------------------------------------------------
    # Initialize video recording (VideoWriter)
    # ---------------------------------------------------------
    video_writer = None
    if args.output:
        # Get source video/webcam dimensions and FPS
        # Note: Even if flipped, dimensions usually remain the same, so .get() is sufficient
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)

        # If FPS cannot be read (some cameras return 0 or nan), set a default of 30
        if not original_fps or original_fps <= 0 or np.isnan(original_fps):
            original_fps = 30.0

        print(f"Initializing Video Writer: {args.output}")
        print(f"  - Resolution: {original_width}x{original_height}")
        print(f"  - FPS: {original_fps}")

        # 'mp4v' is a generic MP4 codec. If it fails, try 'avc1' or 'XVID' (with .avi)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        video_writer = cv2.VideoWriter(
            args.output, 
            fourcc, 
            original_fps, 
            (original_width, original_height)
        )

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
    
    angle_calculator = AngleCalculator(use_3d=True)
    motion_analyzer = MotionAnalyzer(buffer_size=30, smoothing_window=5)

    renderer = SkeletonRenderer(
        show_keypoints=True,
        show_connections=True,
        show_labels=False,
    )

    fps_history = []
    frame_count = 0
    start_time = time.time()
    screenshot_count = 0

    print("\nStarting detection...\n")

    try:
        while True:
            loop_start = time.time()

            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                if is_video_file:
                    print("\nEnd of video file reached.")
                else:
                    print("Error: Failed to capture frame")
                break

            if not is_video_file:
                frame = cv2.flip(frame, 1)

            # Process pose
            pose_result = estimator.process_frame(frame)

            if pose_result and pose_result.is_valid():
                motion_analyzer.update(pose_result)

                angles = None
                if not args.no_angles:
                    angles = angle_calculator.calculate_all_angles(pose_result)

                # Render skeleton and angles
                frame = renderer.render(frame, pose_result, angles)

                posture_metrics = angle_calculator.calculate_posture_metrics(pose_result)

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

                if angles and not args.no_angles:
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
                                joint_name = joint.replace('_', ' ').title()
                                joint_name = joint_name.replace('Left', 'L').replace('Right', 'R')
                                joint_stats[joint_name] = f"{smoothed:.0f}deg"

                if args.show_fps:
                    current_fps = 1.0 / (time.time() - loop_start + 1e-6)
                    fps_history.append(current_fps)
                    if len(fps_history) > 30:
                        fps_history.pop(0)
                    avg_fps = np.mean(fps_history)
                    joint_stats['FPS'] = f"{avg_fps:.1f}"

                if posture_stats:
                    frame = renderer.draw_stats_panel(frame, posture_stats, position='top_left')
                frame = renderer.draw_stats_panel(frame, joint_stats, position='top_right')

            else:
                cv2.putText(
                    frame, "No pose detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                )

            # ---------------------------------------------------------
            # If recording is enabled, write the current rendered frame
            # ---------------------------------------------------------
            if video_writer is not None:
                video_writer.write(frame)

            # Display frame
            window_title = 'Motion Tracker'
            cv2.imshow(window_title, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count:03d}.png"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r'):
                motion_analyzer.clear_history()
                fps_history.clear()
                print("Statistics reset")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        print("\n" + "=" * 60)
        print("Session Summary")
        print("=" * 60)
        print(f"Total frames: {frame_count}")
        print(f"Duration: {elapsed_time:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        
        # Release VideoWriter
        if video_writer is not None:
            video_writer.release()
            print(f"Output saved to: {args.output}")

        print("=" * 60)

        cap.release()
        cv2.destroyAllWindows()
        estimator.release()

    return 0


if __name__ == '__main__':
    sys.exit(main())