"""Observation-quality gate: is the pilot actually observable right now?

New to this project. The base project only ever distinguishes "face
detected" from "face not detected." That collapses two very different
situations into one: the pilot behaving unusually, and the camera/lighting/
tracking simply failing to observe the pilot properly (poor lighting,
motion blur, an extreme head angle, partial occlusion, landmark jitter).
Feeding either of those into a fatigue/distraction decision produces a
confident-looking but meaningless alert.

This module turns "can we trust what we just measured?" into a first-class,
explicit output (``GOOD`` / ``DEGRADED`` / ``INVALID``) computed *before*
any behavioural feature is normalized, event-detected, or scored, so that a
poor observation can gate the rest of the pipeline instead of silently
posing as a real behavioural sample.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Quality(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"


@dataclass
class QualityResult:
    quality: Quality
    reasons: list = field(default_factory=list)
    out_of_frame_fraction: float = 0.0
    landmark_jitter: float = None
    face_area_fraction: float = None


class ObservationQualityAssessor:
    """Classify each frame's observability before any feature is trusted.

    Parameters
    ----------
    out_of_frame_degraded / out_of_frame_invalid : float
        Fraction of landmarks that had to be clipped to the frame bounds
        (i.e. the model predicted them off-screen) before quality drops to
        DEGRADED / INVALID. A pilot glancing at the edge of frame naturally
        pushes some landmarks toward the boundary; a large fraction means
        the face itself is partly outside the frame.
    extreme_angle_degraded / extreme_angle_invalid : float
        Absolute degrees on any of roll/pitch/yaw beyond which the metric
        head-pose solve becomes unreliable (self-occlusion, landmark
        foreshortening) rather than "the pilot turned their head a lot."
    jitter_degraded : float
        Mean per-landmark frame-to-frame displacement (normalized image
        units) above which tracking is likely unstable (motion blur,
        re-detection flicker) rather than smooth real motion.
    min_face_area_fraction : float
        Bounding-box area (fraction of frame) below which the face is too
        small/far to trust fine-grained EAR/gaze measurements.
    """

    def __init__(
        self,
        out_of_frame_degraded=0.02,
        out_of_frame_invalid=0.10,
        extreme_angle_degraded=45.0,
        extreme_angle_invalid=75.0,
        jitter_degraded=0.05,
        min_face_area_fraction=0.01,
    ):
        self.out_of_frame_degraded = out_of_frame_degraded
        self.out_of_frame_invalid = out_of_frame_invalid
        self.extreme_angle_degraded = extreme_angle_degraded
        self.extreme_angle_invalid = extreme_angle_invalid
        self.jitter_degraded = jitter_degraded
        self.min_face_area_fraction = min_face_area_fraction
        self._prev_landmarks_xy = None

    def reset_jitter_reference(self):
        """Call when tracking restarts (e.g. after a face-missing gap)."""
        self._prev_landmarks_xy = None

    def assess(self, raw_landmarks_xy, roll, pitch, yaw):
        """Classify one frame's observability.

        Parameters
        ----------
        raw_landmarks_xy : np.ndarray or None
            Unclipped (x, y) landmark coordinates in normalized [0, 1]
            image space, i.e. *before* ``pilot_monitor.utils.get_landmarks``
            clips them to bounds. ``None`` means no face was detected.
        roll, pitch, yaw : float or None
            Head-pose angles in degrees, or ``None`` if pose estimation did
            not run (e.g. because no face was detected).
        """
        if raw_landmarks_xy is None:
            self.reset_jitter_reference()
            return QualityResult(Quality.INVALID, reasons=["no_face_detected"])

        reasons = []
        quality = Quality.GOOD

        out_of_frame_fraction = float(
            np.mean(
                (raw_landmarks_xy[:, 0] < 0.0)
                | (raw_landmarks_xy[:, 0] > 1.0)
                | (raw_landmarks_xy[:, 1] < 0.0)
                | (raw_landmarks_xy[:, 1] > 1.0)
            )
        )
        if out_of_frame_fraction >= self.out_of_frame_invalid:
            quality = Quality.INVALID
            reasons.append("face_mostly_out_of_frame")
        elif out_of_frame_fraction >= self.out_of_frame_degraded:
            quality = max(quality, Quality.DEGRADED, key=_severity)
            reasons.append("face_partially_out_of_frame")

        dx = raw_landmarks_xy[:, 0].max() - raw_landmarks_xy[:, 0].min()
        dy = raw_landmarks_xy[:, 1].max() - raw_landmarks_xy[:, 1].min()
        face_area_fraction = float(max(dx, 0.0) * max(dy, 0.0))
        if face_area_fraction < self.min_face_area_fraction:
            quality = max(quality, Quality.DEGRADED, key=_severity)
            reasons.append("face_too_small")

        angles = [a for a in (roll, pitch, yaw) if a is not None]
        if angles:
            max_abs_angle = max(abs(a) for a in angles)
            if max_abs_angle >= self.extreme_angle_invalid:
                quality = Quality.INVALID
                reasons.append("extreme_head_angle")
            elif max_abs_angle >= self.extreme_angle_degraded:
                quality = max(quality, Quality.DEGRADED, key=_severity)
                reasons.append("large_head_angle")
        else:
            quality = max(quality, Quality.DEGRADED, key=_severity)
            reasons.append("pose_unavailable")

        jitter = None
        if self._prev_landmarks_xy is not None and quality != Quality.INVALID:
            jitter = float(
                np.mean(
                    np.linalg.norm(raw_landmarks_xy - self._prev_landmarks_xy, axis=1)
                )
            )
            if jitter >= self.jitter_degraded:
                quality = max(quality, Quality.DEGRADED, key=_severity)
                reasons.append("landmark_jitter")
        self._prev_landmarks_xy = raw_landmarks_xy.copy()

        return QualityResult(
            quality=quality,
            reasons=reasons,
            out_of_frame_fraction=out_of_frame_fraction,
            landmark_jitter=jitter,
            face_area_fraction=face_area_fraction,
        )


_SEVERITY_ORDER = {Quality.GOOD: 0, Quality.DEGRADED: 1, Quality.INVALID: 2}


def _severity(quality):
    return _SEVERITY_ORDER[quality]
