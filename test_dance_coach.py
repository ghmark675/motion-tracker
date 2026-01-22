#!/usr/bin/env python3
"""Test script for Dance Coach functionality.

Tests DTW algorithm and sequence comparison without requiring a camera.
"""

import sys
import numpy as np
from demos.dance_coach_demo import DanceSequence, DTWMatcher


def test_dtw_algorithm():
    """Test DTW algorithm with known sequences."""
    print("=" * 60)
    print("Testing DTW Algorithm")
    print("=" * 60)

    matcher = DTWMatcher()

    # Test 1: Identical sequences
    print("\n[Test 1] Identical sequences")
    seq1 = [45, 90, 135, 90, 45]
    seq2 = [45, 90, 135, 90, 45]
    distance = matcher.dtw_distance(seq1, seq2)
    score = matcher.normalize_score(distance, len(seq1))
    print(f"  Sequence 1: {seq1}")
    print(f"  Sequence 2: {seq2}")
    print(f"  DTW Distance: {distance:.2f}")
    print(f"  Score: {score:.1f}/100")
    print(f"  Expected: 100/100 (identical)")
    assert score > 95, "Identical sequences should score near 100"
    print("  [OK] PASSED")

    # Test 2: Similar but time-shifted
    print("\n[Test 2] Time-shifted sequences")
    seq1 = [0, 45, 90, 135, 180]
    seq2 = [0, 0, 45, 90, 135, 180, 180]  # Same pattern, stretched
    distance = matcher.dtw_distance(seq1, seq2)
    score = matcher.normalize_score(distance, (len(seq1) + len(seq2)) / 2)
    print(f"  Sequence 1: {seq1}")
    print(f"  Sequence 2: {seq2}")
    print(f"  DTW Distance: {distance:.2f}")
    print(f"  Score: {score:.1f}/100")
    print(f"  Expected: >80/100 (same pattern, different speed)")
    assert score > 70, "Time-shifted sequences should still score well"
    print("  [OK] PASSED")

    # Test 3: Completely different
    print("\n[Test 3] Completely different sequences")
    seq1 = [0, 0, 0, 0, 0]
    seq2 = [180, 180, 180, 180, 180]
    distance = matcher.dtw_distance(seq1, seq2)
    score = matcher.normalize_score(distance, len(seq1))
    print(f"  Sequence 1: {seq1}")
    print(f"  Sequence 2: {seq2}")
    print(f"  DTW Distance: {distance:.2f}")
    print(f"  Score: {score:.1f}/100")
    print(f"  Expected: <20/100 (completely different)")
    assert score < 30, "Completely different sequences should score low"
    print("  [OK] PASSED")

    # Test 4: Slightly off
    print("\n[Test 4] Slightly off sequences")
    seq1 = [45, 90, 135, 90, 45]
    seq2 = [50, 95, 140, 95, 50]  # 5 degrees off each
    distance = matcher.dtw_distance(seq1, seq2)
    score = matcher.normalize_score(distance, len(seq1))
    print(f"  Sequence 1: {seq1}")
    print(f"  Sequence 2: {seq2}")
    print(f"  DTW Distance: {distance:.2f}")
    print(f"  Score: {score:.1f}/100")
    print(f"  Expected: >90/100 (small differences)")
    assert score > 85, "Slightly off sequences should score high"
    print("  [OK] PASSED")

    print("\n" + "=" * 60)
    print("DTW Algorithm Tests: ALL PASSED")
    print("=" * 60)


def test_dance_sequence():
    """Test DanceSequence functionality."""
    print("\n" + "=" * 60)
    print("Testing DanceSequence Class")
    print("=" * 60)

    # Create a sequence
    print("\n[Test 1] Creating sequence")
    sequence = DanceSequence("Test Dance")
    print(f"  Name: {sequence.name}")
    print(f"  Length: {len(sequence)}")
    assert len(sequence) == 0, "New sequence should be empty"
    print("  [OK] PASSED")

    # Add frames
    print("\n[Test 2] Adding frames")
    for i in range(10):
        angles = {
            'left_elbow': 45 + i * 5,
            'right_elbow': 45 + i * 5,
            'left_knee': 90 + i * 3,
            'right_knee': 90 + i * 3,
        }
        sequence.angles_history.append(angles)
        sequence.timestamps.append(i * 0.1)

    print(f"  Added {len(sequence.angles_history)} frames")
    assert len(sequence.angles_history) == 10, "Should have 10 frames"
    print("  [OK] PASSED")

    # Get angle sequence
    print("\n[Test 3] Extracting angle sequence")
    left_elbow_seq = sequence.get_angle_sequence('left_elbow')
    print(f"  Left elbow angles: {left_elbow_seq}")
    expected = [45 + i * 5 for i in range(10)]
    assert left_elbow_seq == expected, "Angle sequence should match expected"
    print("  [OK] PASSED")

    # Get all sequences
    print("\n[Test 4] Getting all angle sequences")
    all_seqs = sequence.get_all_angle_sequences()
    print(f"  Joints tracked: {list(all_seqs.keys())}")
    assert 'left_elbow' in all_seqs, "Should contain left_elbow"
    assert 'right_knee' in all_seqs, "Should contain right_knee"
    print("  [OK] PASSED")

    # Save and load
    print("\n[Test 5] Save and load")
    filepath = "/tmp/test_dance.pkl"
    sequence.save(filepath)
    print(f"  Saved to: {filepath}")

    loaded = DanceSequence.load(filepath)
    print(f"  Loaded sequence: {loaded.name}")
    assert loaded.name == sequence.name, "Names should match"
    assert len(loaded.angles_history) == len(sequence.angles_history), "Lengths should match"
    print("  [OK] PASSED")

    print("\n" + "=" * 60)
    print("DanceSequence Tests: ALL PASSED")
    print("=" * 60)


def test_integration():
    """Test integration of components."""
    print("\n" + "=" * 60)
    print("Testing Integration")
    print("=" * 60)

    matcher = DTWMatcher()

    # Create reference sequence (a simple movement pattern)
    print("\n[Test 1] Creating reference and practice sequences")
    reference = DanceSequence("Reference")
    practice = DanceSequence("Practice")

    # Reference: smooth sine wave pattern
    for i in range(20):
        t = i / 20 * 2 * np.pi
        angles = {
            'left_elbow': 90 + 45 * np.sin(t),
            'right_elbow': 90 + 45 * np.sin(t),
            'left_knee': 120 + 30 * np.sin(t * 2),
            'right_knee': 120 + 30 * np.sin(t * 2),
        }
        reference.angles_history.append(angles)

    # Practice: similar pattern but slightly different
    for i in range(20):
        t = i / 20 * 2 * np.pi
        angles = {
            'left_elbow': 90 + 45 * np.sin(t) + np.random.normal(0, 3),  # Add noise
            'right_elbow': 90 + 45 * np.sin(t) + np.random.normal(0, 3),
            'left_knee': 120 + 30 * np.sin(t * 2) + np.random.normal(0, 2),
            'right_knee': 120 + 30 * np.sin(t * 2) + np.random.normal(0, 2),
        }
        practice.angles_history.append(angles)

    print(f"  Reference frames: {len(reference.angles_history)}")
    print(f"  Practice frames: {len(practice.angles_history)}")
    print("  [OK] Sequences created")

    # Compare sequences
    print("\n[Test 2] Comparing sequences")
    ref_seqs = reference.get_all_angle_sequences()
    prac_seqs = practice.get_all_angle_sequences()

    scores = {}
    for joint in ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']:
        distance = matcher.dtw_distance(ref_seqs[joint], prac_seqs[joint])
        score = matcher.normalize_score(distance, len(ref_seqs[joint]))
        scores[joint] = score
        print(f"    {joint}: {score:.1f}/100")

    overall = np.mean(list(scores.values()))
    print(f"  Overall score: {overall:.1f}/100")
    print(f"  Expected: >70/100 (similar with noise)")

    assert overall > 60, "Overall score should be reasonable for similar sequences"
    print("  [OK] PASSED")

    print("\n" + "=" * 60)
    print("Integration Tests: ALL PASSED")
    print("=" * 60)


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "DANCE COACH TEST SUITE" + " " * 21 + "║")
    print("╚" + "=" * 58 + "╝")

    try:
        test_dtw_algorithm()
        test_dance_sequence()
        test_integration()

        print("\n")
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 10 + "ALL TESTS PASSED SUCCESSFULLY!" + " " * 17 + "║")
        print("╚" + "=" * 58 + "╝")
        print("\nDance Coach is ready to use!")
        print("Run: python demos/dance_coach_demo.py")
        print("")

        return 0

    except AssertionError as e:
        print(f"\n[FAILED] Test assertion failed: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
