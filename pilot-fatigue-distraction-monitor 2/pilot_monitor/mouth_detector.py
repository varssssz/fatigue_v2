"""Mouth-opening (yawn-related) feature extraction.

This module is new to this project; the base ``driver-state-detection``
project it builds on has no mouth/yawn handling at all. The approach mirrors
the EAR construction already used for the eyes (a ratio of vertical lip
separation to horizontal mouth width, averaged over several vertical
measurements) applied to MediaPipe's inner-mouth landmarks, so it shares the
same normalization behaviour and failure modes as the existing EAR code
rather than introducing a new one.
"""

import numpy as np
from numpy import linalg as LA


class MouthDetector:
    """Compute a Mouth Aspect Ratio (MAR) analogous to the eyes' EAR."""

    # Inner-mouth landmarks from MediaPipe's 478-point face mesh.
    # Corners: 78 (left), 308 (right). Three vertical pairs sampled across
    # the mouth width, mirroring the two-pair averaging used for EAR.
    MOUTH_CORNER_LEFT = 78
    MOUTH_CORNER_RIGHT = 308
    VERTICAL_PAIRS = [(82, 87), (13, 14), (312, 317)]

    @staticmethod
    def _finite_ratio(numerator, denominator):
        if denominator <= np.finfo(float).eps:
            return None
        return float(numerator / denominator)

    def get_MAR(self, landmarks, frame_size):
        """Return the Mouth Aspect Ratio, or ``None`` if the mouth width collapses.

        Parameters
        ----------
        landmarks : np.ndarray
            Normalized (x, y, z) MediaPipe landmarks, as returned by
            ``pilot_monitor.utils.get_landmarks``.
        frame_size : tuple[int, int]
            ``(width, height)`` of the frame, used to convert normalized
            landmark coordinates into pixel space so the ratio is not
            distorted by non-square frames.
        """
        left_corner = landmarks[self.MOUTH_CORNER_LEFT, :2] * frame_size
        right_corner = landmarks[self.MOUTH_CORNER_RIGHT, :2] * frame_size
        mouth_width = LA.norm(right_corner - left_corner)

        vertical_distances = []
        for top_idx, bottom_idx in self.VERTICAL_PAIRS:
            top = landmarks[top_idx, :2] * frame_size
            bottom = landmarks[bottom_idx, :2] * frame_size
            vertical_distances.append(LA.norm(top - bottom))

        mean_vertical = float(np.mean(vertical_distances))
        return self._finite_ratio(mean_vertical, mouth_width)
