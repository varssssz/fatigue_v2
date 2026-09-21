"""Shared per-frame pipeline, used identically by webcam capture, video
replay, and calibration recording. New to this project -- the base project
only had one hard-coded webcam loop in ``main.py``; duplicating that loop
for replay/calibration would violate the master prompt's explicit
instruction not to duplicate the detection pipeline for replay mode.
"""

from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

from pilot_monitor.calibration import FEATURE_NAMES
from pilot_monitor.events import EventEngine, WindowedEventStats
from pilot_monitor.eye_detector import EyeDetector
from pilot_monitor.mouth_detector import MouthDetector
from pilot_monitor.normalization import normalize
from pilot_monitor.pose_estimation import HeadPoseEstimator
from pilot_monitor.quality import ObservationQualityAssessor, Quality
from pilot_monitor.risk import RiskScorer
from pilot_monitor.alerts import AlertStateMachine
from pilot_monitor.utils import get_landmarks


@dataclass
class FrameFeatures:
    quality: Quality
    quality_reasons: list = field(default_factory=list)
    ear: float = None
    gaze_dx: float = None
    gaze_dy: float = None
    mar: float = None
    roll: float = None
    pitch: float = None
    yaw: float = None
    face_detected: bool = False


def make_detector(model_path, min_detection_confidence=0.5, min_face_presence_confidence=0.5, min_tracking_confidence=0.5, max_faces=1):
    """Build a MediaPipe FaceLandmarker in VIDEO mode (shared by every entry point)."""
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=max_faces,
        min_face_detection_confidence=min_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


class FeatureExtractor:
    """Landmarks -> raw behavioural features + observation quality.

    This is the pipeline stage shared by every mode (calibration, live
    monitoring, replay): it knows nothing about personalization, events, or
    risk -- only "what did the camera actually show this frame."
    """

    def __init__(self, head_pose_kwargs=None, quality_kwargs=None):
        self.eye_detector = EyeDetector()
        self.mouth_detector = MouthDetector()
        self.head_pose = HeadPoseEstimator(**(head_pose_kwargs or {}))
        self.quality_assessor = ObservationQualityAssessor(**(quality_kwargs or {}))

    def process(self, frame, mp_result, frame_size):
        if not mp_result.face_landmarks:
            quality_result = self.quality_assessor.assess(None, None, None, None)
            return (
                FrameFeatures(quality=quality_result.quality, quality_reasons=quality_result.reasons),
                frame,
            )

        landmarks, raw_landmarks = get_landmarks(mp_result.face_landmarks, return_raw=True)
        ear = self.eye_detector.get_EAR(landmarks, frame_size)
        gaze_dx, gaze_dy = self.eye_detector.get_gaze_vector(landmarks, frame_size)
        mar = self.mouth_detector.get_MAR(landmarks, frame_size)
        posed_frame, roll, pitch, yaw = self.head_pose.get_pose(frame, landmarks, frame_size)
        if posed_frame is not None:
            frame = posed_frame

        quality_result = self.quality_assessor.assess(raw_landmarks[:, :2], roll, pitch, yaw)

        features = FrameFeatures(
            quality=quality_result.quality,
            quality_reasons=quality_result.reasons,
            ear=ear,
            gaze_dx=gaze_dx,
            gaze_dy=gaze_dy,
            mar=mar,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            face_detected=True,
        )
        return features, frame


class MonitoringPipeline:
    """Full personalized pipeline: features -> normalization -> events ->
    (optional GRU anomaly) -> risk -> alert state machine.

    Used for live monitoring and replay once a pilot has a calibration
    profile. Calibration recording uses ``FeatureExtractor`` directly
    (there is no profile to normalize against yet).
    """

    def __init__(self, profile, gru_bundle=None, window_s=20.0, event_kwargs=None, alert_kwargs=None):
        self.profile = profile
        self.gru_bundle = gru_bundle  # (TrainedTemporalModel, GRUAutoencoder) or None
        self.feature_extractor = FeatureExtractor()
        self.event_engine = EventEngine(**(event_kwargs or {}))
        self.windowed_stats = WindowedEventStats(window_s=window_s)
        self.risk_scorer = RiskScorer()
        self.alert_state_machine = AlertStateMachine(**(alert_kwargs or {}))
        self._gru_buffer = []

    def process_frame(self, frame, mp_result, frame_size, t_now):
        features, frame = self.feature_extractor.process(frame, mp_result, frame_size)
        is_invalid = features.quality == Quality.INVALID

        raw = {
            "ear": features.ear,
            "gaze_dx": features.gaze_dx,
            "gaze_dy": features.gaze_dy,
            "mar": features.mar,
            "roll": features.roll,
            "pitch": features.pitch,
            "yaw": features.yaw,
        }
        normalized = normalize(self.profile, raw)

        for event in self.event_engine.update(t_now, normalized.zscores, is_invalid):
            self.windowed_stats.add(event)
        active_durations = self.event_engine.active_snapshots(t_now)
        window_stats = self.windowed_stats.compute(t_now, active_durations)

        gru_anomaly = None
        if self.gru_bundle is not None and not is_invalid:
            from pilot_monitor.temporal_model import zscores_to_vector

            self._gru_buffer.append(zscores_to_vector(normalized.zscores))
            meta, model = self.gru_bundle
            if len(self._gru_buffer) >= meta.window_len:
                window = np.stack(self._gru_buffer[-meta.window_len :])
                gru_anomaly = meta.anomaly_zscore(model, window)

        risk_result = self.risk_scorer.score(window_stats, gru_anomaly)
        fatigue_update, distraction_update = self.alert_state_machine.update(
            t_now, risk_result, is_invalid
        )

        return frame, features, normalized, risk_result, fatigue_update, distraction_update, gru_anomaly


def frame_timestamp_ms(previous_timestamp_ms, elapsed_since_stream_start_s):
    """Monotonic millisecond timestamp MediaPipe VIDEO mode requires."""
    return max(previous_timestamp_ms + 1, int(elapsed_since_stream_start_s * 1000))
