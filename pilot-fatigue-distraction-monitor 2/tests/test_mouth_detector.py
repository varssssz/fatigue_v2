import numpy as np
import pytest

from pilot_monitor.mouth_detector import MouthDetector


def make_mouth_landmarks(frame_size, mouth_height=10, mouth_width=80):
    landmarks = np.zeros((478, 3), dtype=float)
    detector = MouthDetector()
    center_x, center_y = 320, 240
    landmarks[detector.MOUTH_CORNER_LEFT, :2] = (
        (center_x - mouth_width / 2) / frame_size[0],
        center_y / frame_size[1],
    )
    landmarks[detector.MOUTH_CORNER_RIGHT, :2] = (
        (center_x + mouth_width / 2) / frame_size[0],
        center_y / frame_size[1],
    )
    for top_idx, bottom_idx in detector.VERTICAL_PAIRS:
        landmarks[top_idx, :2] = (center_x / frame_size[0], (center_y - mouth_height / 2) / frame_size[1])
        landmarks[bottom_idx, :2] = (center_x / frame_size[0], (center_y + mouth_height / 2) / frame_size[1])
    return landmarks


def test_mar_increases_when_mouth_opens():
    detector = MouthDetector()
    frame_size = (640, 480)
    closed = make_mouth_landmarks(frame_size, mouth_height=5)
    open_mouth = make_mouth_landmarks(frame_size, mouth_height=40)

    assert detector.get_MAR(open_mouth, frame_size) > detector.get_MAR(closed, frame_size)


def test_mar_invariant_to_frame_size():
    detector = MouthDetector()
    small = make_mouth_landmarks((640, 480), mouth_height=20)
    large = make_mouth_landmarks((1280, 960), mouth_height=20)

    assert detector.get_MAR(small, (640, 480)) == pytest.approx(
        detector.get_MAR(large, (1280, 960))
    )


def test_degenerate_mouth_width_returns_none():
    detector = MouthDetector()
    landmarks = np.zeros((478, 3), dtype=float)
    assert detector.get_MAR(landmarks, (640, 480)) is None
