from pilot_monitor.events import EventType
from pilot_monitor.risk import RiskScorer


def test_no_events_means_zero_risk():
    scorer = RiskScorer()
    result = scorer.score({})
    assert result.fatigue_risk == 0.0
    assert result.distraction_risk == 0.0
    assert result.fatigue_reasons == []


def test_prolonged_closure_drives_fatigue_not_distraction():
    scorer = RiskScorer()
    stats = {
        EventType.PROLONGED_CLOSURE: {
            "count": 2,
            "rate_per_min": 2.0,
            "mean_duration": 3.0,
            "max_duration": 4.0,
            "active_fraction": 0.5,
        }
    }
    result = scorer.score(stats)
    assert result.fatigue_risk > 0.0
    assert result.distraction_risk == 0.0
    assert any("closed" in r for r in result.fatigue_reasons)


def test_gaze_deviation_drives_distraction_not_fatigue():
    scorer = RiskScorer()
    stats = {
        EventType.GAZE_DEVIATION: {
            "count": 3,
            "rate_per_min": 3.0,
            "mean_duration": 2.0,
            "max_duration": 3.0,
            "active_fraction": 0.4,
        }
    }
    result = scorer.score(stats)
    assert result.distraction_risk > 0.0
    assert result.fatigue_risk == 0.0


def test_single_yawn_does_not_dominate_fatigue_risk():
    scorer = RiskScorer()
    stats = {
        EventType.MOUTH_OPENING: {
            "count": 1,
            "rate_per_min": 1.0,
            "mean_duration": 1.0,
            "max_duration": 1.0,
            "active_fraction": 0.1,
        }
    }
    result = scorer.score(stats)
    # A single yawn must not, by itself, register as a fatigue reason.
    assert result.fatigue_reasons == []
    assert result.fatigue_risk < 0.05


def test_gru_anomaly_contributes_to_both_pathways():
    scorer = RiskScorer()
    result = scorer.score({}, gru_anomaly=3.0)
    assert result.fatigue_risk > 0.0
    assert result.distraction_risk > 0.0


def test_negative_gru_anomaly_is_ignored():
    scorer = RiskScorer()
    result = scorer.score({}, gru_anomaly=-2.0)
    assert result.fatigue_risk == 0.0
    assert result.distraction_risk == 0.0
