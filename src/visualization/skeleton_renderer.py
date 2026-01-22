"""Skeleton and pose visualization renderer."""

from typing import Optional, Tuple, Dict, List
import numpy as np
import cv2
from ..core.pose_estimator import PoseResult


class SkeletonRenderer:
    """Render skeleton and annotations on frames."""

    # Color scheme
    COLORS = {
        'keypoint': (0, 255, 0),       # Green
        'connection': (0, 255, 255),   # Yellow
        'angle_good': (0, 255, 0),     # Green
        'angle_warning': (0, 165, 255),  # Orange
        'angle_bad': (0, 0, 255),      # Red
        'text': (255, 255, 255),       # White
        'background': (0, 0, 0),       # Black
    }

    # Skeleton connections
    CONNECTIONS = [
        # Head and neck - IMPORTANT!
        ('nose', 'left_shoulder'),
        ('nose', 'right_shoulder'),
        ('left_ear', 'left_shoulder'),
        ('right_ear', 'right_shoulder'),

        # Torso
        ('left_shoulder', 'right_shoulder'),
        ('left_hip', 'right_hip'),
        ('left_shoulder', 'left_hip'),
        ('right_shoulder', 'right_hip'),

        # Left arm
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('left_wrist', 'left_thumb'),
        ('left_wrist', 'left_index'),
        ('left_wrist', 'left_pinky'),

        # Right arm
        ('right_shoulder', 'right_elbow'),
        ('right_elbow', 'right_wrist'),
        ('right_wrist', 'right_thumb'),
        ('right_wrist', 'right_index'),
        ('right_wrist', 'right_pinky'),

        # Left leg
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('left_ankle', 'left_heel'),
        ('left_ankle', 'left_foot_index'),

        # Right leg
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
        ('right_ankle', 'right_heel'),
        ('right_ankle', 'right_foot_index'),

        # Face details
        ('nose', 'left_eye'),
        ('nose', 'right_eye'),
        ('left_eye', 'left_ear'),
        ('right_eye', 'right_ear'),
        ('mouth_left', 'mouth_right'),
    ]

    def __init__(
        self,
        show_keypoints: bool = True,
        show_connections: bool = True,
        show_labels: bool = False,
        line_thickness: int = 2,
        keypoint_radius: int = 4,
        min_visibility: float = 0.5,
    ):
        """Initialize skeleton renderer.

        Args:
            show_keypoints: Draw keypoint circles
            show_connections: Draw skeleton lines
            show_labels: Draw keypoint labels
            line_thickness: Thickness of connection lines
            keypoint_radius: Radius of keypoint circles
            min_visibility: Minimum visibility to draw keypoint
        """
        self.show_keypoints = show_keypoints
        self.show_connections = show_connections
        self.show_labels = show_labels
        self.line_thickness = line_thickness
        self.keypoint_radius = keypoint_radius
        self.min_visibility = min_visibility

    def render(
        self,
        frame: np.ndarray,
        pose_result: PoseResult,
        angles: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Render skeleton on frame.

        Args:
            frame: Input image
            pose_result: Pose detection result
            angles: Optional dictionary of joint angles

        Returns:
            Annotated image
        """
        if pose_result is None or not pose_result.keypoints:
            return frame

        annotated = frame.copy()
        height, width = frame.shape[:2]

        # Draw connections first (so keypoints are on top)
        if self.show_connections:
            annotated = self._draw_connections(annotated, pose_result, width, height)

        # Draw keypoints
        if self.show_keypoints:
            annotated = self._draw_keypoints(annotated, pose_result, width, height)

        # Draw angles if provided
        if angles is not None:
            annotated = self._draw_angles(annotated, pose_result, angles, width, height)

        return annotated

    def _draw_keypoints(
        self,
        frame: np.ndarray,
        pose_result: PoseResult,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Draw keypoints on frame."""
        for keypoint in pose_result.keypoints:
            if keypoint.visibility < self.min_visibility:
                continue

            x, y = keypoint.to_image_coords(width, height)

            # Draw circle
            cv2.circle(
                frame,
                (x, y),
                self.keypoint_radius,
                self.COLORS['keypoint'],
                -1,
            )

            # Draw label if enabled
            if self.show_labels:
                cv2.putText(
                    frame,
                    keypoint.name.replace('_', ' '),
                    (x + 10, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    self.COLORS['text'],
                    1,
                )

        return frame

    def _draw_connections(
        self,
        frame: np.ndarray,
        pose_result: PoseResult,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Draw skeleton connections."""
        for start_name, end_name in self.CONNECTIONS:
            start_kp = pose_result.get_keypoint(start_name)
            end_kp = pose_result.get_keypoint(end_name)

            if (start_kp is None or end_kp is None or
                start_kp.visibility < self.min_visibility or
                end_kp.visibility < self.min_visibility):
                continue

            start_pos = start_kp.to_image_coords(width, height)
            end_pos = end_kp.to_image_coords(width, height)

            cv2.line(
                frame,
                start_pos,
                end_pos,
                self.COLORS['connection'],
                self.line_thickness,
            )

        return frame

    def _draw_angles(
        self,
        frame: np.ndarray,
        pose_result: PoseResult,
        angles: Dict[str, float],
        width: int,
        height: int,
    ) -> np.ndarray:
        """Draw joint angles on frame."""
        # Only draw angles for major joints to avoid clutter
        major_joints = [
            'left_elbow', 'right_elbow',
            'left_shoulder', 'right_shoulder',
            'left_knee', 'right_knee',
            'left_hip', 'right_hip',
        ]

        for joint_name in major_joints:
            angle = angles.get(joint_name)
            if angle is None:
                continue

            # Get joint keypoint position
            keypoint = pose_result.get_keypoint(joint_name)
            if keypoint is None or keypoint.visibility < self.min_visibility:
                continue

            x, y = keypoint.to_image_coords(width, height)

            # Determine color based on angle range
            color = self._get_angle_color(joint_name, angle)

            # Draw angle text with larger font
            text = f"{angle:.0f}deg"
            self._draw_text_with_background(
                frame,
                text,
                (x + 15, y + 5),
                color,
                font_scale=0.6,
                thickness=2,
            )

            # Draw a small circle at the joint to highlight it
            cv2.circle(frame, (x, y), 8, color, 2)

        return frame

    def _get_angle_color(self, joint_name: str, angle: float) -> Tuple[int, int, int]:
        """Determine color based on joint angle quality."""
        # Example thresholds (can be customized per joint)
        if 'elbow' in joint_name or 'knee' in joint_name:
            # Full extension should be ~170-180°
            if 160 <= angle <= 180:
                return self.COLORS['angle_good']
            elif 140 <= angle < 160:
                return self.COLORS['angle_warning']
            else:
                return self.COLORS['angle_bad']

        return self.COLORS['text']

    @staticmethod
    def _draw_text_with_background(
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color: Tuple[int, int, int],
        font_scale: float = 0.5,
        thickness: int = 1,
    ):
        """Draw text with background rectangle."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        x, y = position

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (x - 2, y - text_height - 2),
            (x + text_width + 2, y + baseline + 2),
            (0, 0, 0),
            -1,
        )

        # Draw text
        cv2.putText(
            frame,
            text,
            position,
            font,
            font_scale,
            color,
            thickness,
        )

    def draw_stats_panel(
        self,
        frame: np.ndarray,
        stats: Dict[str, any],
        position: str = 'top_left',
    ) -> np.ndarray:
        """Draw statistics panel on frame.

        Args:
            frame: Input image
            stats: Dictionary of statistics to display
            position: Panel position ('top_left', 'top_right', etc.)

        Returns:
            Annotated image
        """
        height, width = frame.shape[:2]

        # Prepare text lines
        lines = []
        for key, value in stats.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.1f}")
            else:
                lines.append(f"{key}: {value}")

        # Calculate panel size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        line_height = 25
        padding = 10

        panel_height = len(lines) * line_height + 2 * padding
        max_width = max(
            cv2.getTextSize(line, font, font_scale, thickness)[0][0]
            for line in lines
        ) if lines else 100
        panel_width = max_width + 2 * padding

        # Determine panel position
        if position == 'top_left':
            x, y = 10, 10
        elif position == 'top_right':
            x, y = width - panel_width - 10, 10
        elif position == 'bottom_left':
            x, y = 10, height - panel_height - 10
        else:  # bottom_right
            x, y = width - panel_width - 10, height - panel_height - 10

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, y),
            (x + panel_width, y + panel_height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw text lines
        text_y = y + padding + 20
        for line in lines:
            cv2.putText(
                frame,
                line,
                (x + padding, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )
            text_y += line_height

        return frame
