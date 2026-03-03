#!/usr/bin/env python3
"""Test script for golf coach demo."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from golf_coach_demo import GolfCoach, GolfSequence


def test_golf_coach():
    """Test golf coach functionality."""
    print("Testing GolfCoach class...")
    
    # Initialize golf coach
    coach = GolfCoach()
    print("✓ GolfCoach initialized successfully")
    
    # Check key joints
    expected_joints = [
        'left_wrist', 'right_wrist',
        'left_elbow', 'right_elbow',
        'left_shoulder', 'right_shoulder',
        'spine_curve'
    ]
    
    for joint in expected_joints:
        if joint in coach.key_joints:
            print(f"✓ Key joint '{joint}' found")
        else:
            print(f"✗ Key joint '{joint}' missing")
            return False
    
    print("✓ All golf-specific joints are present")
    
    # Test DTW matcher
    from golf_coach_demo import DTWMatcher
    seq1 = [1, 2, 3, 4, 5]
    seq2 = [1, 2, 3, 4, 5]
    distance = DTWMatcher.dtw_distance(seq1, seq2)
    print(f"✓ DTW distance calculation works: {distance}")
    
    # Test sequence class
    sequence = GolfSequence("Test")
    print(f"✓ GolfSequence created: {sequence.name}")
    
    print("\nAll tests passed!")
    return True


if __name__ == '__main__':
    test_golf_coach()