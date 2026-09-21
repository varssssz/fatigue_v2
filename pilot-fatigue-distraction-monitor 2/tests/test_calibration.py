import numpy as np
import pytest

from pilot_monitor.calibration import PersonalProfile


def test_fit_uses_robust_median_not_mean():
    # A blink or two of near-zero EAR shouldn't drag the "normal" baseline
    # down the way a mean would; median should stay near the bulk of values.
    values = [0.30] * 20 + [0.02, 0.03]
    profile = PersonalProfile.fit("p1", {"ear": values})

    assert profile.baselines["ear"].median == pytest.approx(0.30, abs=1e-6)


def test_robust_zscore_centers_on_this_pilots_median():
    profile = PersonalProfile.fit("p1", {"ear": [0.30] * 30})

    z_at_median = profile.robust_zscore("ear", 0.30)
    z_below = profile.robust_zscore("ear", 0.10)

    assert z_at_median == pytest.approx(0.0)
    assert z_below < 0


def test_missing_value_normalizes_to_none():
    profile = PersonalProfile.fit("p1", {"ear": [0.30] * 30})
    assert profile.robust_zscore("ear", None) is None


def test_save_and_load_round_trip(tmp_path):
    profile = PersonalProfile.fit("p1", {"ear": [0.30] * 30, "mar": [0.1] * 30})
    path = tmp_path / "p1.json"
    profile.save(path)

    loaded = PersonalProfile.load(path)
    assert loaded.pilot_id == "p1"
    assert loaded.baselines["ear"].median == pytest.approx(profile.baselines["ear"].median)


def test_sparse_feature_falls_back_without_raising():
    profile = PersonalProfile.fit("p1", {"ear": [0.1, 0.2]})
    assert profile.baselines["ear"].n_samples == 2
    # Should not raise, and should still produce a usable (if wide) baseline.
    assert profile.robust_zscore("ear", 0.5) is not None
