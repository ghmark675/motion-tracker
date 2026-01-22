#!/usr/bin/env python3
"""Dance Coach Demo - Record and compare dance movements.

This demo allows you to:
1. Record a reference dance sequence
2. Perform the dance and get real-time comparison and scoring
3. Uses DTW (Dynamic Time Warping) for temporal alignment

Usage:
    python demos/dance_coach_demo.py [--camera CAMERA_ID]

Controls:
    - Press 'r' to start/stop recording reference
    - Press 'p' to start/stop practice (comparison mode)
    - Press 'c' to clear reference
    - Press 's' to save/load reference
    - Press 'q' to quit
"""

import sys
import argparse
from pathlib import Path
import time
import pickle
import cv2
import numpy as np
from collections import deque
from typing import List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backends.mediapipe_backend import MediaPipeBackend
from src.core.angle_calculator import AngleCalculator
from src.core.pose_estimator import PoseResult
from src.visualization.skeleton_renderer import SkeletonRenderer


class DanceSequence:
    """Store a sequence of poses for comparison."""

    def __init__(self, name: str = "Dance"):
        """Initialize dance sequence.

        Args:
            name: Name of the dance sequence
        """
        self.name = name
        self.poses: List[PoseResult] = []
        self.angles_history: List[dict] = []
        self.timestamps: List[float] = []

    def add_frame(self, pose: PoseResult, angles: dict, timestamp: float):
        """Add a frame to the sequence.

        Args:
            pose: Pose result
            angles: Calculated angles
            timestamp: Timestamp in seconds
        """
        self.poses.append(pose)
        self.angles_history.append(angles)
        self.timestamps.append(timestamp)

    def get_angle_sequence(self, joint_name: str) -> List[float]:
        """Get angle sequence for a specific joint.

        Args:
            joint_name: Joint name

        Returns:
            List of angles over time
        """
        return [
            angles.get(joint_name, 0) or 0
            for angles in self.angles_history
        ]

    def get_all_angle_sequences(self) -> dict:
        """Get all angle sequences.

        Returns:
            Dictionary mapping joint names to angle sequences
        """
        if not self.angles_history:
            return {}

        all_joints = set()
        for angles in self.angles_history:
            all_joints.update(angles.keys())

        return {
            joint: self.get_angle_sequence(joint)
            for joint in all_joints
        }

    def save(self, filepath: str):
        """Save sequence to file.

        Args:
            filepath: Path to save file
        """
        data = {
            'name': self.name,
            'angles_history': self.angles_history,
            'timestamps': self.timestamps,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str):
        """Load sequence from file.

        Args:
            filepath: Path to load file

        Returns:
            DanceSequence instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        sequence = cls(data['name'])
        sequence.angles_history = data['angles_history']
        sequence.timestamps = data['timestamps']
        return sequence

    def __len__(self):
        """Get sequence length."""
        return len(self.poses)


class DTWMatcher:
    """Dynamic Time Warping for sequence matching."""

    @staticmethod
    def dtw_distance(seq1: List[float], seq2: List[float]) -> float:
        """Calculate DTW distance between two sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            DTW distance (lower is better)
        """
        n, m = len(seq1), len(seq2)

        if n == 0 or m == 0:
            return float('inf')

        # Create cost matrix
        dtw_matrix = np.full((n + 1, m + 1), float('inf'))
        dtw_matrix[0, 0] = 0

        # Fill matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(seq1[i - 1] - seq2[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # insertion
                    dtw_matrix[i, j - 1],      # deletion
                    dtw_matrix[i - 1, j - 1]   # match
                )

        return dtw_matrix[n, m]

    @staticmethod
    def normalize_score(distance: float, seq_length: int) -> float:
        """Normalize DTW distance to 0-100 score.

        Args:
            distance: DTW distance
            seq_length: Average sequence length

        Returns:
            Score from 0-100 (higher is better)
        """
        if seq_length == 0:
            return 0

        # Normalize by sequence length
        normalized = distance / seq_length

        # Convert to score (assuming max difference of 180 degrees)
        # Lower distance = higher score
        score = max(0, 100 - (normalized / 180 * 100))

        return score


class DanceCoach:
    """Dance coach for recording and comparing movements."""

    def __init__(self):
        """Initialize dance coach."""
        self.reference: Optional[DanceSequence] = None
        self.current: Optional[DanceSequence] = None
        self.angle_calculator = AngleCalculator(use_3d=True)
        self.matcher = DTWMatcher()

        # Key joints for comparison
        self.key_joints = [
            'left_elbow', 'right_elbow',
            'left_shoulder', 'right_shoulder',
            'left_knee', 'right_knee',
            'left_hip', 'right_hip',
        ]

    def start_recording_reference(self):
        """Start recording reference sequence."""
        self.reference = DanceSequence("Reference")

    def add_reference_frame(self, pose: PoseResult, timestamp: float):
        """Add frame to reference sequence.

        Args:
            pose: Pose result
            timestamp: Timestamp
        """
        if self.reference is not None:
            angles = self.angle_calculator.calculate_all_angles(pose)
            self.reference.add_frame(pose, angles, timestamp)

    def stop_recording_reference(self):
        """Stop recording reference sequence."""
        if self.reference and len(self.reference) > 0:
            print(f"[OK] Reference recorded: {len(self.reference)} frames")
        else:
            self.reference = None
            print("[!] Reference recording cancelled (no frames)")

    def start_practice(self):
        """Start practice sequence."""
        self.current = DanceSequence("Practice")

    def add_practice_frame(self, pose: PoseResult, timestamp: float):
        """Add frame to practice sequence.

        Args:
            pose: Pose result
            timestamp: Timestamp
        """
        if self.current is not None:
            angles = self.angle_calculator.calculate_all_angles(pose)
            self.current.add_frame(pose, angles, timestamp)

    def stop_practice(self):
        """Stop practice sequence."""
        self.current = None

    def compare_sequences(self) -> dict:
        """Compare current practice with reference.

        Returns:
            Dictionary with comparison results
        """
        if not self.reference or not self.current:
            return {'error': 'Missing reference or current sequence'}

        if len(self.reference) < 10 or len(self.current) < 10:
            return {'error': 'Sequences too short (need at least 10 frames)'}

        results = {}
        ref_angles = self.reference.get_all_angle_sequences()
        curr_angles = self.current.get_all_angle_sequences()

        # Compare each joint
        joint_scores = {}
        for joint in self.key_joints:
            if joint in ref_angles and joint in curr_angles:
                ref_seq = ref_angles[joint]
                curr_seq = curr_angles[joint]

                distance = self.matcher.dtw_distance(ref_seq, curr_seq)
                avg_len = (len(ref_seq) + len(curr_seq)) / 2
                score = self.matcher.normalize_score(distance, avg_len)

                joint_scores[joint] = {
                    'distance': distance,
                    'score': score,
                }

        # Calculate overall score
        if joint_scores:
            overall_score = np.mean([s['score'] for s in joint_scores.values()])
        else:
            overall_score = 0

        results['joint_scores'] = joint_scores
        results['overall_score'] = overall_score

        return results

    def get_real_time_feedback(self, pose: PoseResult) -> dict:
        """Get real-time feedback by comparing with reference.

        Args:
            pose: Current pose

        Returns:
            Feedback dictionary
        """
        if not self.reference or len(self.reference) == 0:
            return {'error': 'No reference available'}

        # Get current angles
        current_angles = self.angle_calculator.calculate_all_angles(pose)

        # Find closest reference frame (simplified - use latest)
        ref_idx = min(len(self.reference) - 1, len(self.current or []))
        ref_angles = self.reference.angles_history[ref_idx]

        # Compare key joints
        feedback = {}
        for joint in self.key_joints:
            curr_angle = current_angles.get(joint)
            ref_angle = ref_angles.get(joint)

            if curr_angle is not None and ref_angle is not None:
                diff = abs(curr_angle - ref_angle)
                status = 'good' if diff < 15 else ('ok' if diff < 30 else 'bad')
                feedback[joint] = {
                    'current': curr_angle,
                    'reference': ref_angle,
                    'difference': diff,
                    'status': status,
                }

        return feedback


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Dance Coach demo')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    return parser.parse_args()


def main():
    """Main demo function."""
    args = parse_args()

    print("=" * 60)
    print("Motion Tracker - Dance Coach Demo")
    print("=" * 60)
    print("\nControls:")
    print("  'r' - Start/Stop recording reference")
    print("  'p' - Start/Stop practice (comparison)")
    print("  'c' - Clear reference")
    print("  's' - Save reference to file")
    print("  'l' - Load reference from file")
    print("  'q' - Quit")
    print("=" * 60)

    # Initialize components
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    estimator = MediaPipeBackend(model_complexity=1)
    if not estimator.initialize():
        print("[ERROR] Failed to initialize pose estimator")
        return 1

    renderer = SkeletonRenderer()
    coach = DanceCoach()

    # State
    recording_reference = False
    practicing = False
    start_time = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Process pose
            pose_result = estimator.process_frame(frame)

            if pose_result and pose_result.is_valid():
                # Render skeleton
                frame = renderer.render(frame, pose_result)

                # Handle recording/practicing
                current_time = time.time()

                if recording_reference:
                    if start_time is None:
                        start_time = current_time
                    timestamp = current_time - start_time
                    coach.add_reference_frame(pose_result, timestamp)

                    # Show recording indicator
                    cv2.putText(
                        frame,
                        f"RECORDING REFERENCE: {len(coach.reference)} frames",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                elif practicing:
                    if start_time is None:
                        start_time = current_time
                    timestamp = current_time - start_time
                    coach.add_practice_frame(pose_result, timestamp)

                    # Get real-time feedback
                    feedback = coach.get_real_time_feedback(pose_result)

                    # Show practice indicator
                    cv2.putText(
                        frame,
                        f"PRACTICE MODE: {len(coach.current)} frames",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    # Show feedback
                    if 'error' not in feedback:
                        y_offset = 80
                        for joint, info in feedback.items():
                            if info['status'] == 'good':
                                color = (0, 255, 0)
                                marker = "[OK]"
                            elif info['status'] == 'ok':
                                color = (0, 165, 255)
                                marker = "[~]"
                            else:
                                color = (0, 0, 255)
                                marker = "[!]"

                            joint_name = joint.replace('_', ' ').title()
                            text = f"{marker} {joint_name}: {info['difference']:.0f}deg off"

                            cv2.putText(
                                frame,
                                text,
                                (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                1,
                            )
                            y_offset += 25

                else:
                    # Show status
                    if coach.reference:
                        status = f"Reference ready: {len(coach.reference)} frames"
                        color = (0, 255, 0)
                    else:
                        status = "Press 'r' to record reference"
                        color = (0, 165, 255)

                    cv2.putText(
                        frame,
                        status,
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )

            # Display
            cv2.imshow('Dance Coach', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('r'):
                if recording_reference:
                    # Stop recording
                    coach.stop_recording_reference()
                    recording_reference = False
                    start_time = None
                else:
                    # Start recording
                    coach.start_recording_reference()
                    recording_reference = True
                    start_time = None
                    print("[OK] Recording reference...")

            elif key == ord('p'):
                if not coach.reference:
                    print("[!] Please record reference first (press 'r')")
                elif practicing:
                    # Stop practicing and show results
                    coach.stop_practice()
                    practicing = False
                    start_time = None

                    # Compare sequences
                    results = coach.compare_sequences()

                    if 'error' in results:
                        print(f"[!] {results['error']}")
                    else:
                        print("\n" + "=" * 40)
                        print("DANCE COMPARISON RESULTS")
                        print("=" * 40)
                        print(f"Overall Score: {results['overall_score']:.1f}/100")
                        print("\nJoint Scores:")
                        for joint, scores in results['joint_scores'].items():
                            joint_name = joint.replace('_', ' ').title()
                            print(f"  {joint_name:20s}: {scores['score']:5.1f}/100")
                        print("=" * 40)
                else:
                    # Start practicing
                    coach.start_practice()
                    practicing = True
                    start_time = None
                    print("[OK] Practice mode started...")

            elif key == ord('c'):
                coach.reference = None
                print("[OK] Reference cleared")

            elif key == ord('s'):
                if coach.reference:
                    filepath = "dance_reference.pkl"
                    coach.reference.save(filepath)
                    print(f"[OK] Reference saved to {filepath}")
                else:
                    print("[!] No reference to save")

            elif key == ord('l'):
                try:
                    filepath = "dance_reference.pkl"
                    coach.reference = DanceSequence.load(filepath)
                    print(f"[OK] Reference loaded from {filepath}")
                except Exception as e:
                    print(f"[!] Failed to load reference: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        estimator.release()

    return 0


if __name__ == '__main__':
    sys.exit(main())
