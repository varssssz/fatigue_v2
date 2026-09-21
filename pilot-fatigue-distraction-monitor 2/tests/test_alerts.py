from pilot_monitor.alerts import AlertState, PathwayStateMachine


def make_sm(**overrides):
    defaults = dict(
        monitoring_thresh=0.15,
        elevated_thresh=0.35,
        warning_thresh=0.6,
        recovery_thresh=0.1,
        persistence_s=1.0,
        recovery_s=1.0,
        cooldown_s=3.0,
    )
    defaults.update(overrides)
    return PathwayStateMachine(**defaults)


def test_single_frame_spike_does_not_alert():
    sm = make_sm(persistence_s=2.0)
    update = sm.update(0.0, 0.9, ["x"], quality_is_invalid=False)
    assert update.state == AlertState.NORMAL


def test_sustained_high_risk_escalates_to_warning():
    sm = make_sm(persistence_s=1.0)
    sm.update(0.0, 0.9, ["x"], quality_is_invalid=False)
    update = sm.update(1.5, 0.9, ["x"], quality_is_invalid=False)
    assert update.state == AlertState.WARNING


def test_invalid_quality_preempts_and_resumes():
    sm = make_sm(persistence_s=0.1)
    sm.update(0.0, 0.9, ["x"], quality_is_invalid=False)
    warning_update = sm.update(0.15, 0.9, ["x"], quality_is_invalid=False)
    assert warning_update.state == AlertState.WARNING

    invalid_update = sm.update(0.3, 0.9, [], quality_is_invalid=True)
    assert invalid_update.state == AlertState.DATA_INVALID

    resumed = sm.update(0.4, 0.9, ["x"], quality_is_invalid=False)
    assert resumed.state == AlertState.WARNING


def test_recovery_then_cooldown_suppresses_immediate_re_warning():
    sm = make_sm(persistence_s=0.1, recovery_s=0.1, cooldown_s=5.0)
    sm.update(0.0, 0.9, ["x"], quality_is_invalid=False)
    sm.update(0.2, 0.9, ["x"], quality_is_invalid=False)
    sm.update(0.4, 0.0, [], quality_is_invalid=False)
    recovered = sm.update(0.6, 0.0, [], quality_is_invalid=False)
    assert recovered.state == AlertState.NORMAL

    re_spike = sm.update(0.8, 0.9, ["x"], quality_is_invalid=False)
    assert re_spike.state != AlertState.WARNING
