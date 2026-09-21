varssssz

## What is new in this project

Everything listed below did not exist in the base project and was designed
and implemented specifically for this research project:

- Personal (per-pilot) statistical baseline and calibration (`pilot_monitor/calibration.py`)
- Robust normalization against that baseline (`pilot_monitor/normalization.py`)
- Directional gaze (signed horizontal/vertical) and per-axis head-deviation features
- Mouth/yawn feature extraction (`pilot_monitor/mouth_detector.py`) — absent from the base project entirely
- Observation-quality classification: GOOD / DEGRADED / INVALID (`pilot_monitor/quality.py`)
- Temporal event engine: explicit closure, blink, gaze-deviation, and head-deviation
  events with start/duration/recovery (`pilot_monitor/events.py`)
- A GRU-based temporal anomaly model trained per pilot on that pilot's own
  calibration data (`pilot_monitor/temporal_model.py`)
- Separated fatigue/distraction risk pathways with explainable alert reasons
  and a persistence/hysteresis/cooldown/recovery alert state machine
  (`pilot_monitor/risk.py`, `pilot_monitor/alerts.py`)
- Video-replay mode sharing one pipeline with webcam capture (`pilot_monitor/pipeline.py`)
- Structured research logging (`pilot_monitor/logging_export.py`)
- The full evaluation/ablation framework and the 4-subject showcase (`evaluation/`)

## Why this file exists

The base repository is small and its provenance is easy to check (it is a
public GitHub project with its own commit history, README, and license file).
Anyone reviewing this submission — an instructor, a plagiarism-detection tool,
or a future collaborator — should be able to see exactly which lines are
reused-with-attribution and which are original contributions, rather than
discovering the overlap themselves. That is the actual defense against a
plagiarism finding: nothing here is presented as original that isn't.
