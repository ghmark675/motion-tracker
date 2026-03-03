#!/usr/bin/env python3
"""Golf Coach Demo - Compare golf swing movements using two local video files.

This demo allows you to:
1. Load a reference golf swing from a template video
2. Compare a golfer's swing from a scored video
3. Uses DTW (Dynamic Time Warping) for temporal alignment
4. Provides golf-specific joint analysis

Usage:
    python demos/golf_coach_demo.py --template-video TEMPLATE_VIDEO --scored-video SCORED_VIDEO

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


class GolfSequence:
    """Store a sequence of poses for golf swing comparison."""

    def __init__(self, name: str = "Golf Swing"):
        """Initialize golf sequence.

        Args:
            name: Name of the golf sequence
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
            GolfSequence instance
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


class GolfCoach:
    """Golf coach for comparing golf swing movements."""

    def __init__(self):
        """Initialize golf coach."""
        self.angle_calculator = AngleCalculator(use_3d=True)
        self.matcher = DTWMatcher()

        # Key joints for golf comparison (wrist angles, spine curve, etc.)
        self.key_joints = [
            'left_wrist', 'right_wrist',  # Important for grip and swing
            'left_elbow', 'right_elbow',  # Arm positioning
            'left_shoulder', 'right_shoulder',  # Shoulder rotation
            'spine_curve',  # Body rotation and posture
        ]

    def process_video_to_sequence(self, video_path: str) -> Optional[GolfSequence]:
        """Process a video file and extract pose sequence.

        Args:
            video_path: Path to the video file

        Returns:
            GolfSequence instance or None if failed
        """
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video: {video_path}")
            return None

        sequence = GolfSequence(Path(video_path).stem)
        
        # Initialize pose estimator
        estimator = MediaPipeBackend(model_complexity=1)
        if not estimator.initialize():
            print("[ERROR] Failed to initialize pose estimator")
            return None

        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process pose
                pose_result = estimator.process_frame(frame)

                if pose_result and pose_result.is_valid():
                    # Calculate all angles including custom ones
                    angles = self.angle_calculator.calculate_all_angles(pose_result)
                    
                    # Add custom golf-specific angles
                    angles['spine_curve'] = self.angle_calculator.calculate_spine_curve(pose_result)
                    
                    timestamp = time.time() - start_time
                    sequence.add_frame(pose_result, angles, timestamp)

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"  Processed {frame_count} frames...")

        finally:
            cap.release()
            estimator.release()

        print(f"[OK] Processed {frame_count} frames, extracted {len(sequence)} poses")
        return sequence

    def compare_sequences(self, reference: GolfSequence, scored: GolfSequence) -> dict:
        """Compare two golf swing sequences.

        Args:
            reference: Reference golf sequence
            scored: Scored golf sequence

        Returns:
            Dictionary with comparison results
        """
        if len(reference) < 10 or len(scored) < 10:
            return {'error': 'Sequences too short (need at least 10 frames)'}

        results = {}
        ref_angles = reference.get_all_angle_sequences()
        scored_angles = scored.get_all_angle_sequences()

        # Compare each joint
        joint_scores = {}
        for joint in self.key_joints:
            if joint in ref_angles and joint in scored_angles:
                ref_seq = ref_angles[joint]
                scored_seq = scored_angles[joint]

                distance = self.matcher.dtw_distance(ref_seq, scored_seq)
                avg_len = (len(ref_seq) + len(scored_seq)) / 2
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Golf Coach demo')
    parser.add_argument('--template-video', type=str, required=True, 
                        help='Path to the template standard golf video')
    parser.add_argument('--scored-video', type=str, required=True, 
                        help='Path to the golf video to be scored')
    return parser.parse_args()


def main():
    """Main demo function."""
    args = parse_args()

    print("=" * 60)
    print("Motion Tracker - Golf Coach Demo")
    print("=" * 60)
    print(f"Template Video: {args.template_video}")
    print(f"Scored Video: {args.scored_video}")
    print("=" * 60)

    # Initialize components
    coach = GolfCoach()

    # Process template video
    print("\n[1/2] Processing template video...")
    reference_sequence = coach.process_video_to_sequence(args.template_video)
    if not reference_sequence:
        print("[ERROR] Failed to process template video")
        return 1

    # Process scored video
    print("\n[2/2] Processing scored video...")
    scored_sequence = coach.process_video_to_sequence(args.scored_video)
    if not scored_sequence:
        print("[ERROR] Failed to process scored video")
        return 1

    # Compare sequences
    print("\nComparing golf swings...")
    results = coach.compare_sequences(reference_sequence, scored_sequence)

    if 'error' in results:
        print(f"[ERROR] {results['error']}")
        return 1

    # Display results
    print("\n" + "=" * 50)
    print("GOLF SWING COMPARISON RESULTS")
    print("=" * 50)
    print(f"Overall Score: {results['overall_score']:.1f}/100")
    print("\nJoint Scores:")
    
    # Sort joints by score for better presentation
    sorted_joints = sorted(
        results['joint_scores'].items(), 
        key=lambda x: x[1]['score'], 
        reverse=True
    )
    
    for joint, scores in sorted_joints:
        joint_name = joint.replace('_', ' ').title()
        print(f"  {joint_name:20s}: {scores['score']:5.1f}/100")
    
    print("=" * 50)
    
    # Provide feedback based on overall score
    overall_score = results['overall_score']
    if overall_score >= 90:
        feedback = "Excellent form! Very close to the template."
    elif overall_score >= 75:
        feedback = "Good form with minor adjustments needed."
    elif overall_score >= 60:
        feedback = "Fair form, but significant improvements possible."
    else:
        feedback = "Needs substantial improvement to match the template."
    
    print(f"\nFeedback: {feedback}")
    print("=" * 50)

    return 0


if __name__ == '__main__':
    sys.exit(main())