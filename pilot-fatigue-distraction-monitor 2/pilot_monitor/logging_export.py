"""Structured research logging: every field the design document's Part 22
lists, one row per processed frame, as CSV or JSON-lines. New to this
project -- the base project has no persistence at all, only a live overlay.
"""

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class FrameRecord:
    timestamp: float
    frame_index: int
    pilot_id: str
    quality: str
    quality_reasons: list = field(default_factory=list)
    ear: float = None
    gaze_dx: float = None
    gaze_dy: float = None
    mar: float = None
    roll: float = None
    pitch: float = None
    yaw: float = None
    ear_z: float = None
    gaze_dx_z: float = None
    gaze_dy_z: float = None
    mar_z: float = None
    roll_z: float = None
    pitch_z: float = None
    yaw_z: float = None
    gru_anomaly: float = None
    fatigue_risk: float = None
    distraction_risk: float = None
    fatigue_state: str = None
    distraction_state: str = None
    fatigue_reasons: list = field(default_factory=list)
    distraction_reasons: list = field(default_factory=list)
    context_flight_phase: str = None


class ResearchLogger:
    """Append ``FrameRecord`` rows and flush them as CSV or JSON-lines."""

    def __init__(self):
        self._rows = []

    def log(self, record):
        self._rows.append(record)

    def __len__(self):
        return len(self._rows)

    def to_csv(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self._rows:
            return
        fieldnames = list(asdict(self._rows[0]).keys())
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._rows:
                data = asdict(row)
                for key in ("quality_reasons", "fatigue_reasons", "distraction_reasons"):
                    data[key] = ";".join(data[key])
                writer.writerow(data)

    def to_jsonl(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for row in self._rows:
                file.write(json.dumps(asdict(row)) + "\n")

    def rows(self):
        return list(self._rows)
