#!/usr/bin/env python3
"""AI Fitness Trainer demo for exercise form analysis.

This demo analyzes exercise form in real-time, counts repetitions,
and provides feedback on technique.

Supported exercises:
- Squats
- Push-ups
- Bicep curls
- Shoulder press

Usage:
    python demos/fitness_trainer_demo.py [--camera CAMERA_ID] [--exercise EXERCISE]

Controls:
    - Press 'q' to quit
    - Press 'r' to reset rep counter
    - Press 's' to save screenshot
    - Press '1-4' to switch exercises
"""

import sys
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from enum import Enum

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backends.mediapipe_backend import MediaPipeBackend
from src.core.angle_calculator import AngleCalculator
from src.core.motion_analyzer import MotionAnalyzer
from src.visualization.skeleton_renderer import SkeletonRenderer


class Exercise(Enum):
    """Supported exercises."""
    SQUAT = 1
    PUSHUP = 2
    BICEP_CURL = 3
    SHOULDER_PRESS = 4


class ExerciseTracker:
    """Track and analyze exercise performance."""

    def __init__(self, exercise: Exercise):
        """Initialize exercise tracker.

        Args:
            exercise: Exercise type to track
        """
        self.exercise = exercise
        self.rep_count = 0
        self.state = 'idle'
        self.last_state_change = 0
        self.min_frames_between_reps = 15

        # Exercise-specific configurations
        self.configs = {
            Exercise.SQUAT: {
                'joint': 'left_knee',
                'threshold_low': 90,
                'threshold_high': 160,
                'form_checks': self._check_squat_form,
            },
            Exercise.PUSHUP: {
                'joint': 'left_elbow',
                'threshold_low': 70,
                'threshold_high': 160,
                'form_checks': self._check_pushup_form,
            },
            Exercise.BICEP_CURL: {
                'joint': 'left_elbow',
                'threshold_low': 40,
                'threshold_high': 160,
                'form_checks': self._check_curl_form,
            },
            Exercise.SHOULDER_PRESS: {
                'joint': 'left_elbow',
                'threshold_low': 80,
                'threshold_high': 170,
                'form_checks': self._check_press_form,
            },
        }

    def update(self, pose_result, angle_calculator, motion_analyzer):
        """Update tracker with new pose.

        Args:
            pose_result: Current pose
            angle_calculator: Angle calculator
            motion_analyzer: Motion analyzer

        Returns:
            Dictionary with rep count and feedback
        """
        config = self.configs[self.exercise]
        joint = config['joint']

        # Get smoothed angle
        angle = motion_analyzer.get_smoothed_angle(joint)

        if angle is None:
            return {
                'reps': self.rep_count,
                'state': 'unknown',
                'feedback': 'Position yourself in frame',
            }

        # State machine for rep counting
        threshold_low = config['threshold_low']
        threshold_high = config['threshold_high']

        if self.state == 'idle':
            if angle <= threshold_low:
                self.state = 'down'
                self.last_state_change = 0

        elif self.state == 'down':
            if angle >= threshold_high:
                if self.last_state_change >= self.min_frames_between_reps:
                    self.state = 'up'
                    self.last_state_change = 0
                else:
                    self.state = 'idle'
            else:
                self.last_state_change += 1

        elif self.state == 'up':
            if angle <= threshold_low:
                if self.last_state_change >= self.min_frames_between_reps:
                    self.rep_count += 1
                    self.state = 'down'
                    self.last_state_change = 0
                else:
                    self.state = 'idle'
            else:
                self.last_state_change += 1

        # Check form
        form_feedback = config['form_checks'](pose_result, angle_calculator)

        return {
            'reps': self.rep_count,
            'state': self.state,
            'angle': angle,
            'feedback': form_feedback,
        }

    def reset(self):
        """Reset rep counter."""
        self.rep_count = 0
        self.state = 'idle'
        self.last_state_change = 0

    def _check_squat_form(self, pose_result, angle_calculator):
        """Check squat form quality."""
        feedback = []

        # Check knee angle
        left_knee = angle_calculator.calculate_joint_angle(pose_result, 'left_knee')
        right_knee = angle_calculator.calculate_joint_angle(pose_result, 'right_knee')

        if left_knee and right_knee:
            # Knees should be similar angle
            if abs(left_knee - right_knee) > 15:
                feedback.append("Uneven knee bend")

        # Check back angle (should stay relatively straight)
        left_hip = angle_calculator.calculate_joint_angle(pose_result, 'left_hip')
        if left_hip and left_hip < 70:
            feedback.append("Keep back straight")

        return " | ".join(feedback) if feedback else "Good form!"

    def _check_pushup_form(self, pose_result, angle_calculator):
        """Check push-up form quality."""
        feedback = []

        # Check if body is straight (plank position)
        left_hip = angle_calculator.calculate_joint_angle(pose_result, 'left_hip')
        if left_hip and (left_hip < 160 or left_hip > 200):
            feedback.append("Keep body straight")

        # Check elbow symmetry
        left_elbow = angle_calculator.calculate_joint_angle(pose_result, 'left_elbow')
        right_elbow = angle_calculator.calculate_joint_angle(pose_result, 'right_elbow')

        if left_elbow and right_elbow:
            if abs(left_elbow - right_elbow) > 20:
                feedback.append("Even arm bend")

        return " | ".join(feedback) if feedback else "Good form!"

    def _check_curl_form(self, pose_result, angle_calculator):
        """Check bicep curl form quality."""
        feedback = []

        # Check if elbow stays in place (shoulder angle shouldn't change much)
        left_shoulder = angle_calculator.calculate_joint_angle(pose_result, 'left_shoulder')

        if left_shoulder:
            # Elbow should stay relatively stable
            if left_shoulder < 30 or left_shoulder > 80:
                feedback.append("Keep elbow stable")

        return " | ".join(feedback) if feedback else "Good form!"

    def _check_press_form(self, pose_result, angle_calculator):
        """Check shoulder press form quality."""
        feedback = []

        # Check back straightness
        left_hip = angle_calculator.calculate_joint_angle(pose_result, 'left_hip')
        if left_hip and left_hip < 160:
            feedback.append("Stand up straight")

        return " | ".join(feedback) if feedback else "Good form!"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI Fitness Trainer demo')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument(
        '--exercise',
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help='Exercise: 1=Squat, 2=Push-up, 3=Bicep Curl, 4=Shoulder Press'
    )
    return parser.parse_args()


def main():
    """Main demo function."""
    args = parse_args()

    exercise = Exercise(args.exercise)

    print("=" * 60)
    print("Motion Tracker - AI Fitness Trainer Demo")
    print("=" * 60)
    print(f"Current Exercise: {exercise.name}")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset rep counter")
    print("  - Press 's' to save screenshot")
    print("  - Press '1-4' to switch exercises")
    print("=" * 60)

    # Initialize components
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    estimator = MediaPipeBackend(model_complexity=1)
    estimator.initialize()

    angle_calculator = AngleCalculator(use_3d=True)
    motion_analyzer = MotionAnalyzer(buffer_size=30, smoothing_window=5)
    renderer = SkeletonRenderer()

    tracker = ExerciseTracker(exercise)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Process pose
            pose_result = estimator.process_frame(frame)

            if pose_result and pose_result.is_valid():
                # Update motion analyzer
                motion_analyzer.update(pose_result)

                # Calculate angles
                angles = angle_calculator.calculate_all_angles(pose_result)

                # Render skeleton
                frame = renderer.render(frame, pose_result, angles)

                # Update tracker
                result = tracker.update(pose_result, angle_calculator, motion_analyzer)

                # Display exercise info
                exercise_name = tracker.exercise.name.replace('_', ' ')
                cv2.putText(
                    frame,
                    f"Exercise: {exercise_name}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                )

                # Display rep count
                cv2.putText(
                    frame,
                    f"Reps: {result['reps']}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    3,
                )

                # Display state
                state_colors = {
                    'idle': (128, 128, 128),
                    'down': (0, 165, 255),
                    'up': (0, 255, 0),
                }
                state_color = state_colors.get(result['state'], (255, 255, 255))

                cv2.putText(
                    frame,
                    f"State: {result['state'].upper()}",
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    state_color,
                    2,
                )

                # Display angle
                if 'angle' in result and result['angle']:
                    cv2.putText(
                        frame,
                        f"Angle: {result['angle']:.0f}deg",
                        (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                # Display feedback
                feedback = result.get('feedback', '')
                feedback_color = (0, 255, 0) if feedback == "Good form!" else (0, 165, 255)

                cv2.putText(
                    frame,
                    f"Form: {feedback}",
                    (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    feedback_color,
                    2,
                )

            cv2.imshow('AI Fitness Trainer', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                tracker.reset()
                motion_analyzer.clear_history()
                print("Rep counter reset")
            elif key == ord('s'):
                cv2.imwrite(f"fitness_{int(time.time())}.png", frame)
                print("Screenshot saved")
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                exercise_num = int(chr(key))
                tracker = ExerciseTracker(Exercise(exercise_num))
                motion_analyzer.clear_history()
                print(f"Switched to: {Exercise(exercise_num).name}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        estimator.release()

    return 0


if __name__ == '__main__':
    sys.exit(main())
