import pytest

from pilot_monitor.events import EventEngine, EventType, WindowedEventStats


def zs(ear=0.0, dx=0.0, dy=0.0, roll=0.0, pitch=0.0, yaw=0.0, mar=0.0):
    return {"ear": ear, "gaze_dx": dx, "gaze_dy": dy, "roll": roll, "pitch": pitch, "yaw": yaw, "mar": mar}


def test_short_closure_is_classified_as_blink():
    engine = EventEngine(ear_z_thresh=1.0, blink_max_duration=0.4)
    events = []
    for t, ear in [(0.0, -2.0), (0.1, -2.0), (0.2, 0.0)]:
        events += engine.update(t, zs(ear=ear), quality_is_invalid=False)

    assert len(events) == 1
    assert events[0].event_type == EventType.BLINK


def test_long_closure_is_classified_as_prolonged():
    engine = EventEngine(ear_z_thresh=1.0, blink_max_duration=0.4)
    events = []
    for t, ear in [(0.0, -2.0), (1.0, -2.0), (2.0, -2.0), (2.1, 0.0)]:
        events += engine.update(t, zs(ear=ear), quality_is_invalid=False)

    assert len(events) == 1
    assert events[0].event_type == EventType.PROLONGED_CLOSURE
    assert events[0].duration == pytest.approx(2.1)


def test_invalid_quality_does_not_end_an_active_event():
    engine = EventEngine(ear_z_thresh=1.0)
    engine.update(0.0, zs(ear=-2.0), quality_is_invalid=False)
    # Frame is unobservable: must not be treated as "eyes opened."
    events = engine.update(0.5, zs(ear=-2.0), quality_is_invalid=True)
    assert events == []
    # Recovery only counted once observation quality is good again.
    events = engine.update(1.0, zs(ear=0.0), quality_is_invalid=False)
    assert len(events) == 1


def test_head_axes_tracked_independently():
    engine = EventEngine(head_z_thresh=1.0)
    # Large roll only; pitch/yaw stay normal.
    for t in (0.0, 0.5):
        events = engine.update(t, zs(roll=3.0, pitch=0.0, yaw=0.0), quality_is_invalid=False)
    events = engine.update(1.0, zs(roll=0.0, pitch=0.0, yaw=0.0), quality_is_invalid=False)

    assert len(events) == 1
    assert events[0].event_type == EventType.HEAD_DEVIATION_ROLL


def test_ongoing_closure_counts_toward_risk_before_it_ends():
    # Regression test: an event must affect the window's active_fraction
    # WHILE it is still ongoing, not only once it finally closes -- a real
    # alerting system cannot wait for a 30s eye closure to end before
    # reacting to it.
    engine = EventEngine(ear_z_thresh=1.0)
    stats = WindowedEventStats(window_s=60.0)

    for t in (0.0, 5.0, 10.0):
        events = engine.update(t, zs(ear=-2.0), quality_is_invalid=False)
        assert events == []  # still ongoing, nothing has closed yet

    window = stats.compute(10.0, engine.active_snapshots(10.0))
    assert EventType.PROLONGED_CLOSURE in window
    assert window[EventType.PROLONGED_CLOSURE]["active_fraction"] == pytest.approx(10.0 / 60.0)


def test_windowed_stats_expire_old_events():
    stats = WindowedEventStats(window_s=5.0)
    from pilot_monitor.events import Event

    stats.add(Event(EventType.BLINK, 0.0, 0.2, 0.2, 2.0))
    assert EventType.BLINK in stats.compute(1.0)
    assert EventType.BLINK not in stats.compute(10.0)
