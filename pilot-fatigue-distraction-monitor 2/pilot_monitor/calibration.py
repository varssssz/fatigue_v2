"""Per-pilot statistical baseline (calibration profile).

New to this project. The base project has exactly one threshold set, shared
by everyone. This module answers "what does normal look like for *this*
pilot" using robust statistics (median and Median Absolute Deviation, plus
percentiles) rather than mean/standard deviation, because a short
calibration recording will contain a handful of natural blinks and glances
that are legitimate normal behaviour but are exactly the kind of outliers
that would drag a mean/std estimate around; median/MAD is far less sensitive
to that (see the design document, Part 8, for why mean/std was rejected).

A profile is built once per pilot from a short (60-120s) guided recording of
normal, alert behaviour, and is later used by ``normalization.py`` to turn
each new observation into a robust z-score relative to that pilot.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# MAD -> std-equivalent scale factor for a Gaussian, used only to make the
# robust z-score comparable in magnitude to a familiar z-score; it does not
# assume the underlying data is Gaussian.
MAD_TO_STD = 1.4826

FEATURE_NAMES = (
    "ear",
    "gaze_dx",
    "gaze_dy",
    "mar",
    "roll",
    "pitch",
    "yaw",
)


@dataclass
class FeatureBaseline:
    """Robust baseline statistics for a single feature."""

    median: float
    mad: float
    p05: float
    p95: float
    n_samples: int

    def scale(self):
        """MAD, floored so a near-constant feature doesn't explode z-scores."""
        return max(self.mad * MAD_TO_STD, 1e-6)


@dataclass
class PersonalProfile:
    """A pilot's calibrated baseline across all tracked features."""

    pilot_id: str
    baselines: dict = field(default_factory=dict)  # name -> FeatureBaseline
    calibration_duration_s: float = 0.0
    calibration_frame_count: int = 0

    @classmethod
    def fit(cls, pilot_id, feature_samples, calibration_duration_s=0.0):
        """Build a profile from calibration samples.

        Parameters
        ----------
        pilot_id : str
        feature_samples : dict[str, list[float]]
            Per-feature lists of values observed during calibration, already
            filtered to GOOD/DEGRADED-quality frames only (see
            ``pilot_monitor.quality``). ``None`` values must be excluded by
            the caller before this is called.
        """
        baselines = {}
        frame_count = 0
        for name in FEATURE_NAMES:
            values = np.asarray(feature_samples.get(name, []), dtype=float)
            frame_count = max(frame_count, len(values))
            if len(values) < 5:
                # Not enough calibration data for this feature: fall back to
                # a wide-open baseline that normalizes to ~0 deviation until
                # enough real data arrives, rather than raising.
                baselines[name] = FeatureBaseline(0.0, 1.0, -1.0, 1.0, len(values))
                continue
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            p05, p95 = (float(v) for v in np.percentile(values, [5, 95]))
            baselines[name] = FeatureBaseline(median, mad, p05, p95, len(values))
        return cls(
            pilot_id=pilot_id,
            baselines=baselines,
            calibration_duration_s=calibration_duration_s,
            calibration_frame_count=frame_count,
        )

    def robust_zscore(self, feature_name, value):
        """Return ``(value - median) / robust_scale`` for one feature."""
        if value is None or feature_name not in self.baselines:
            return None
        baseline = self.baselines[feature_name]
        return (value - baseline.median) / baseline.scale()

    def to_dict(self):
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data):
        baselines = {
            name: FeatureBaseline(**fields)
            for name, fields in data["baselines"].items()
        }
        return cls(
            pilot_id=data["pilot_id"],
            baselines=baselines,
            calibration_duration_s=data.get("calibration_duration_s", 0.0),
            calibration_frame_count=data.get("calibration_frame_count", 0),
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def default_profile_path(profiles_dir, pilot_id):
    return Path(profiles_dir) / f"{pilot_id}.json"


def collect_calibration_samples(video_source, model_path, duration_s, max_faces=1):
    """Record ``duration_s`` seconds from ``video_source`` and return
    ``(feature_samples, timestamped_features)`` for
    ``PersonalProfile.fit`` and GRU training respectively.

    Only GOOD-quality frames are kept: a calibration baseline built from
    degraded/invalid observations would itself be unreliable (design
    document, Part 8's placement of the quality gate before personalization).
    """
    import time

    import cv2

    from pilot_monitor.pipeline import FeatureExtractor, frame_timestamp_ms, make_detector
    from pilot_monitor.quality import Quality

    feature_samples = {name: [] for name in FEATURE_NAMES}
    timestamped_features = []

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        cap.release()
        raise SystemExit(f"Cannot open video source {video_source}")

    extractor = FeatureExtractor()
    try:
        with make_detector(model_path, max_faces=max_faces) as detector:
            start = time.perf_counter()
            previous_timestamp_ms = -1
            print(f"Calibrating for {duration_s:.0f}s -- look and behave naturally...")
            while True:
                now = time.perf_counter()
                elapsed = now - start
                if elapsed >= duration_s:
                    break
                received, frame = cap.read()
                if not received:
                    break
                if isinstance(video_source, int) and video_source == 0:
                    frame = cv2.flip(frame, 1)

                frame_size = (frame.shape[1], frame.shape[0])
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                import mediapipe as mp

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = frame_timestamp_ms(previous_timestamp_ms, elapsed)
                previous_timestamp_ms = timestamp_ms
                result = detector.detect_for_video(mp_image, timestamp_ms)

                features, _ = extractor.process(frame, result, frame_size)
                if features.quality != Quality.GOOD:
                    continue

                row = {
                    "ear": features.ear,
                    "gaze_dx": features.gaze_dx,
                    "gaze_dy": features.gaze_dy,
                    "mar": features.mar,
                    "roll": features.roll,
                    "pitch": features.pitch,
                    "yaw": features.yaw,
                }
                for name, value in row.items():
                    if value is not None:
                        feature_samples[name].append(value)
                timestamped_features.append((elapsed, row))
    finally:
        cap.release()

    return feature_samples, timestamped_features


def main(argv=None):
    """CLI: record a calibration session, fit a personal profile, and train
    that pilot's GRU temporal anomaly model on the same session."""
    import argparse

    from pilot_monitor.normalization import normalize

    parser = argparse.ArgumentParser(description="Calibrate a pilot's personal baseline")
    parser.add_argument("--pilot-id", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--profiles-dir", type=Path, default=Path("profiles"))
    parser.add_argument("--model-path", type=Path, default=Path("models/face_landmarker.task"))
    parser.add_argument("--window-len", type=int, default=30)
    parser.add_argument("--no-gru", dest="train_gru", action="store_false", default=True)
    args = parser.parse_args(argv)

    if not args.model_path.is_file():
        raise SystemExit(
            f"Face Landmarker model not found at {args.model_path}. Run "
            "`pilot-monitor-download-model` first."
        )

    video_source = str(args.video) if args.video else args.camera
    feature_samples, timestamped_features = collect_calibration_samples(
        video_source, args.model_path, args.duration
    )

    frame_count = max((len(v) for v in feature_samples.values()), default=0)
    if frame_count < 30:
        raise SystemExit(
            f"Only {frame_count} good-quality calibration frames were captured; "
            "need at least 30. Try a longer or better-lit recording."
        )

    profile = PersonalProfile.fit(args.pilot_id, feature_samples, args.duration)
    profile_path = default_profile_path(args.profiles_dir, args.pilot_id)
    profile.save(profile_path)
    print(f"Saved personal baseline for '{args.pilot_id}' to {profile_path} "
          f"({frame_count} good-quality frames).")

    if args.train_gru:
        try:
            import numpy as np

            from pilot_monitor.temporal_model import (
                TORCH_AVAILABLE,
                train_pilot_model,
                zscores_to_vector,
            )

            if not TORCH_AVAILABLE:
                print("PyTorch not installed: skipping GRU training (`uv sync` installs it).")
            else:
                sequence = np.stack(
                    [
                        zscores_to_vector(normalize(profile, row).zscores)
                        for _, row in timestamped_features
                    ]
                )
                train_pilot_model(
                    args.pilot_id, sequence, args.profiles_dir, window_len=args.window_len
                )
                print(f"Trained GRU temporal model for '{args.pilot_id}'.")
        except Exception as error:
            print(f"GRU training skipped/failed: {error}")


if __name__ == "__main__":
    main()
