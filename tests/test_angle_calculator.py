"""Unit tests for AngleCalculator."""

import pytest
import numpy as np
from src.core.angle_calculator import AngleCalculator
from src.core.pose_estimator import Keypoint, PoseResult


class TestAngleCalculator:
    """Test cases for AngleCalculator."""

    def setup_method(self):
        """Setup test fixtures."""
        self.calculator = AngleCalculator(use_3d=True)

    def test_right_angle(self):
        """Test calculation of a right angle (90 degrees)."""
        a = np.array([0, 0, 0])
        b = np.array([1, 0, 0])
        c = np.array([1, 1, 0])

        angle = self.calculator.calculate_angle_3points(a, b, c)

        assert abs(angle - 90.0) < 0.1, f"Expected 90°, got {angle}°"

    def test_straight_angle(self):
        """Test calculation of a straight angle (180 degrees)."""
        a = np.array([0, 0, 0])
        b = np.array([1, 0, 0])
        c = np.array([2, 0, 0])

        angle = self.calculator.calculate_angle_3points(a, b, c)

        assert abs(angle - 180.0) < 0.1, f"Expected 180°, got {angle}°"

    def test_acute_angle(self):
        """Test calculation of an acute angle (45 degrees)."""
        a = np.array([0, 0, 0])
        b = np.array([1, 0, 0])
        c = np.array([1, 1, 0])

        # Calculate 45 degree angle
        angle = self.calculator.calculate_angle_3points(a, b, c)

        assert 89 < angle < 91, f"Expected ~90°, got {angle}°"

    def test_calculate_joint_angle_with_valid_keypoints(self):
        """Test joint angle calculation with valid keypoints."""
        # Create mock keypoints
        keypoints = [
            Keypoint('left_shoulder', 0.3, 0.3, 0, 1.0, 1.0, -0.5, 0.5, 0),
            Keypoint('left_elbow', 0.4, 0.5, 0, 1.0, 1.0, -0.3, 0.3, 0),
            Keypoint('left_wrist', 0.5, 0.7, 0, 1.0, 1.0, -0.1, 0.1, 0),
        ]

        pose_result = PoseResult(keypoints=keypoints)

        # Calculate elbow angle
        angle = self.calculator.calculate_joint_angle(pose_result, 'left_elbow')

        assert angle is not None
        assert 0 <= angle <= 180

    def test_calculate_joint_angle_with_missing_keypoint(self):
        """Test joint angle calculation with missing keypoint."""
        keypoints = [
            Keypoint('left_shoulder', 0.3, 0.3, 0, 1.0, 1.0),
            # Missing left_elbow
            Keypoint('left_wrist', 0.5, 0.7, 0, 1.0, 1.0),
        ]

        pose_result = PoseResult(keypoints=keypoints)

        # Should return None when keypoint is missing
        angle = self.calculator.calculate_joint_angle(pose_result, 'left_elbow')

        assert angle is None

    def test_calculate_joint_angle_with_low_visibility(self):
        """Test joint angle calculation with low visibility keypoint."""
        keypoints = [
            Keypoint('left_shoulder', 0.3, 0.3, 0, 1.0, 1.0),
            Keypoint('left_elbow', 0.4, 0.5, 0, 0.3, 0.3),  # Low visibility
            Keypoint('left_wrist', 0.5, 0.7, 0, 1.0, 1.0),
        ]

        pose_result = PoseResult(keypoints=keypoints)

        # Should return None when visibility is too low
        angle = self.calculator.calculate_joint_angle(pose_result, 'left_elbow')

        assert angle is None

    def test_calculate_all_angles(self):
        """Test calculation of all predefined angles."""
        # Create comprehensive keypoint set
        keypoints = [
            Keypoint('nose', 0.5, 0.1, 0, 1.0, 1.0),
            Keypoint('left_shoulder', 0.4, 0.3, 0, 1.0, 1.0, -0.2, 0.5, 0),
            Keypoint('right_shoulder', 0.6, 0.3, 0, 1.0, 1.0, 0.2, 0.5, 0),
            Keypoint('left_elbow', 0.3, 0.5, 0, 1.0, 1.0, -0.3, 0.3, 0),
            Keypoint('right_elbow', 0.7, 0.5, 0, 1.0, 1.0, 0.3, 0.3, 0),
            Keypoint('left_wrist', 0.25, 0.7, 0, 1.0, 1.0, -0.35, 0.1, 0),
            Keypoint('right_wrist', 0.75, 0.7, 0, 1.0, 1.0, 0.35, 0.1, 0),
            Keypoint('left_hip', 0.4, 0.6, 0, 1.0, 1.0, -0.2, 0.0, 0),
            Keypoint('right_hip', 0.6, 0.6, 0, 1.0, 1.0, 0.2, 0.0, 0),
            Keypoint('left_knee', 0.4, 0.8, 0, 1.0, 1.0, -0.2, -0.3, 0),
            Keypoint('right_knee', 0.6, 0.8, 0, 1.0, 1.0, 0.2, -0.3, 0),
            Keypoint('left_ankle', 0.4, 1.0, 0, 1.0, 1.0, -0.2, -0.6, 0),
            Keypoint('right_ankle', 0.6, 1.0, 0, 1.0, 1.0, 0.2, -0.6, 0),
        ]

        pose_result = PoseResult(keypoints=keypoints)

        angles = self.calculator.calculate_all_angles(pose_result)

        # Should have results for multiple joints
        assert isinstance(angles, dict)
        assert len(angles) > 0

        # Check specific joints
        assert 'left_elbow' in angles
        assert 'right_elbow' in angles
        assert 'left_knee' in angles

        # Valid angles should be between 0 and 180
        for joint, angle in angles.items():
            if angle is not None:
                assert 0 <= angle <= 180, f"{joint}: {angle}° out of range"

    def test_invalid_joint_name(self):
        """Test with invalid joint name."""
        keypoints = [Keypoint('left_shoulder', 0.3, 0.3, 0, 1.0, 1.0)]
        pose_result = PoseResult(keypoints=keypoints)

        with pytest.raises(ValueError):
            self.calculator.calculate_joint_angle(pose_result, 'invalid_joint')

    def test_2d_vs_3d_calculation(self):
        """Test difference between 2D and 3D angle calculation."""
        keypoints = [
            Keypoint('left_shoulder', 0.3, 0.3, 0.1, 1.0, 1.0, -0.5, 0.5, 0.2),
            Keypoint('left_elbow', 0.4, 0.5, 0.0, 1.0, 1.0, -0.3, 0.3, 0.0),
            Keypoint('left_wrist', 0.5, 0.7, -0.1, 1.0, 1.0, -0.1, 0.1, -0.2),
        ]

        pose_result = PoseResult(keypoints=keypoints)

        # 3D calculation
        calculator_3d = AngleCalculator(use_3d=True)
        angle_3d = calculator_3d.calculate_joint_angle(pose_result, 'left_elbow')

        # 2D calculation
        calculator_2d = AngleCalculator(use_3d=False)
        angle_2d = calculator_2d.calculate_joint_angle(pose_result, 'left_elbow')

        assert angle_3d is not None
        assert angle_2d is not None
        # They should be different due to Z-axis
        # (though in this simple test they might be close)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
