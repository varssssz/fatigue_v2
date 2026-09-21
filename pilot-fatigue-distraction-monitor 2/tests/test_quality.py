import numpy as np

from pilot_monitor.quality import ObservationQualityAssessor, Quality


def landmarks_grid(n=478, x_range=(0.3, 0.7), y_range=(0.3, 0.7)):
    xs = np.linspace(*x_range, n)
    ys = np.linspace(*y_range, n)
    return np.stack([xs, ys], axis=1)


def test_no_face_is_invalid():
    assessor = ObservationQualityAssessor()
    result = assessor.assess(None, None, None, None)
    assert result.quality == Quality.INVALID
    assert "no_face_detected" in result.reasons


def test_normal_frame_is_good():
    assessor = ObservationQualityAssessor()
    result = assessor.assess(landmarks_grid(), roll=2.0, pitch=-1.0, yaw=3.0)
    assert result.quality == Quality.GOOD


def test_extreme_head_angle_is_invalid():
    assessor = ObservationQualityAssessor(extreme_angle_invalid=75.0)
    result = assessor.assess(landmarks_grid(), roll=0.0, pitch=0.0, yaw=85.0)
    assert result.quality == Quality.INVALID
    assert "extreme_head_angle" in result.reasons


def test_mostly_out_of_frame_is_invalid():
    assessor = ObservationQualityAssessor(out_of_frame_invalid=0.10)
    lm = landmarks_grid(x_range=(-0.5, 1.5))  # most points outside [0, 1]
    result = assessor.assess(lm, roll=0.0, pitch=0.0, yaw=0.0)
    assert result.quality == Quality.INVALID


def test_jitter_flags_degraded_on_second_frame():
    assessor = ObservationQualityAssessor(jitter_degraded=0.01)
    first = landmarks_grid()
    second = landmarks_grid(x_range=(0.5, 0.9))  # large shift
    assessor.assess(first, roll=0.0, pitch=0.0, yaw=0.0)
    result = assessor.assess(second, roll=0.0, pitch=0.0, yaw=0.0)
    assert result.quality == Quality.DEGRADED
    assert "landmark_jitter" in result.reasons
