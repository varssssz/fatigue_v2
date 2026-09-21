"""Temporal event engine: explicit start / duration / recovery events.

New to this project. The base project only ever tracks *how long a
condition has continuously held* (three decaying timers in
``attention_scorer.AttentionScorer``). It never produces a discrete,
countable "this was one blink, lasting 180ms, ending at t=12.4s" record.
That distinction matters for a personalized system: "the pilot blinked 40
times in the last minute, each one longer than usual" is a very different
(and more diagnostic) statement than "the pilot's eyes have been below
threshold for a total of 6 seconds," and it is exactly the kind of feature
your calibration/normalization layer needs something to compute deviation
*from* per pilot.

This module turns each behavioural signal's personalized z-score into
explicit events with a start, a duration, and a recovery, then aggregates
those events into rolling-window statistics (rate, mean/max duration,
active-time fraction) -- the actual input to the risk layer in ``risk.py``,
rather than raw per-frame values.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum


class EventType(str, Enum):
    BLINK = "BLINK"
    PROLONGED_CLOSURE = "PROLONGED_CLOSURE"
    GAZE_DEVIATION = "GAZE_DEVIATION"
    HEAD_DEVIATION_ROLL = "HEAD_DEVIATION_ROLL"
    HEAD_DEVIATION_PITCH = "HEAD_DEVIATION_PITCH"
    HEAD_DEVIATION_YAW = "HEAD_DEVIATION_YAW"
    MOUTH_OPENING = "MOUTH_OPENING"


FATIGUE_EVENT_TYPES = frozenset(
    {EventType.BLINK, EventType.PROLONGED_CLOSURE, EventType.MOUTH_OPENING}
)
DISTRACTION_EVENT_TYPES = frozenset(
    {
        EventType.GAZE_DEVIATION,
        EventType.HEAD_DEVIATION_ROLL,
        EventType.HEAD_DEVIATION_PITCH,
        EventType.HEAD_DEVIATION_YAW,
    }
)


@dataclass
class Event:
    event_type: EventType
    start_time: float
    end_time: float
    duration: float
    peak_severity: float  # max |zscore| (or equivalent) observed during the event


class _BinaryConditionTracker:
    """Turn a per-frame boolean condition into start/duration/recovery events.

    A ``None`` condition (unknown -- e.g. observation quality is INVALID)
    neither starts nor ends an event: it simply pauses judgement, mirroring
    how ``AttentionScorer._update_metric`` already treats unknown frames in
    the base project, extended here to discrete events rather than
    continuous timers.
    """

    def __init__(self, event_type):
        self.event_type = event_type
        self._active_since = None
        self._peak_severity = 0.0

    def active_duration(self, t_now):
        """Elapsed duration of the currently-active event, or ``None``."""
        if self._active_since is None:
            return None
        return t_now - self._active_since

    def update(self, t_now, condition, severity=0.0):
        """Return a completed ``Event`` if one just closed, else ``None``."""
        if condition is None:
            return None
        if condition:
            if self._active_since is None:
                self._active_since = t_now
                self._peak_severity = abs(severity)
            else:
                self._peak_severity = max(self._peak_severity, abs(severity))
            return None
        if self._active_since is not None:
            event = Event(
                event_type=self.event_type,
                start_time=self._active_since,
                end_time=t_now,
                duration=t_now - self._active_since,
                peak_severity=self._peak_severity,
            )
            self._active_since = None
            self._peak_severity = 0.0
            return event
        return None


class EventEngine:
    """Detect closure/blink, gaze-deviation, head-deviation, and mouth events.

    Parameters
    ----------
    ear_z_thresh, gaze_z_thresh, head_z_thresh, mouth_z_thresh : float
        Robust-z-score magnitude beyond which a signal counts as deviating
        from this pilot's personal baseline. EAR deviates *negative*
        (closing); the others are checked by absolute magnitude.
        ``gaze_z_thresh`` is compared against the 2D magnitude
        sqrt(zx**2 + zy**2) of two z-scores, which is Rayleigh- rather than
        Normal-distributed: a magnitude threshold numerically equal to a
        single-axis threshold is systematically too sensitive. The default
        of 3.0 (for both gaze and head axes) was set empirically against
        ``evaluation/synthetic_demo.py`` rather than derived purely from
        the i.i.d.-Gaussian approximation: real (and simulated) head/gaze
        motion is autocorrelated frame-to-frame, so multi-second "natural"
        excursions beyond 2 robust-sigma happen far more often over a long
        monitoring session than an i.i.d. calculation predicts. Retune this
        against real calibration recordings once available -- it is a
        starting point, not a validated constant (see the design document,
        Part 8's note on all risk/event weights).
    blink_max_duration : float
        Closures at or below this duration are classified as ``BLINK``;
        longer ones as ``PROLONGED_CLOSURE``. ~0.4s comfortably separates a
        voluntary/reflex blink from a closure that has become a fatigue
        signal, without needing a second, separately-tuned "asleep" timer.
    """

    def __init__(
        self,
        ear_z_thresh=1.5,
        gaze_z_thresh=3.0,
        head_z_thresh=3.0,
        mouth_z_thresh=1.5,
        blink_max_duration=0.4,
    ):
        self.ear_z_thresh = ear_z_thresh
        self.gaze_z_thresh = gaze_z_thresh
        self.head_z_thresh = head_z_thresh
        self.mouth_z_thresh = mouth_z_thresh
        self.blink_max_duration = blink_max_duration

        self._closure_tracker = _BinaryConditionTracker(EventType.PROLONGED_CLOSURE)
        self._gaze_tracker = _BinaryConditionTracker(EventType.GAZE_DEVIATION)
        self._roll_tracker = _BinaryConditionTracker(EventType.HEAD_DEVIATION_ROLL)
        self._pitch_tracker = _BinaryConditionTracker(EventType.HEAD_DEVIATION_PITCH)
        self._yaw_tracker = _BinaryConditionTracker(EventType.HEAD_DEVIATION_YAW)
        self._mouth_tracker = _BinaryConditionTracker(EventType.MOUTH_OPENING)
        self._trackers = (
            self._closure_tracker,
            self._gaze_tracker,
            self._roll_tracker,
            self._pitch_tracker,
            self._yaw_tracker,
            self._mouth_tracker,
        )

    def active_snapshots(self, t_now):
        """Elapsed duration of every currently-*ongoing* (not yet closed)
        event, keyed by the type it would close as if it ended right now.

        This matters for real-time risk scoring: without it, a closure that
        has already lasted 30 seconds contributes nothing to the current
        risk score until the pilot's eyes reopen and the event finally
        closes -- exactly backwards for an alerting system, which needs to
        know about sustained conditions *while they are happening*.
        ``WindowedEventStats.compute`` merges these in alongside completed
        events.
        """
        snapshots = {}
        for tracker in self._trackers:
            duration = tracker.active_duration(t_now)
            if duration is None:
                continue
            event_type = tracker.event_type
            if event_type == EventType.PROLONGED_CLOSURE and duration <= self.blink_max_duration:
                event_type = EventType.BLINK
            snapshots[event_type] = duration
        return snapshots

    def update(self, t_now, zscores, quality_is_invalid):
        """Advance all trackers by one frame; return newly-completed events.

        Parameters
        ----------
        zscores : dict
            Output of ``normalization.normalize(...).zscores``.
        quality_is_invalid : bool
            When True, every condition is treated as unknown (``None``) for
            this frame rather than as "cleared" -- an INVALID frame must not
            look like a recovery.
        """
        if quality_is_invalid:
            ear_c = gaze_c = roll_c = pitch_c = yaw_c = mouth_c = None
            ear_z = gaze_mag = roll_z = pitch_z = yaw_z = mouth_z = 0.0
        else:
            ear_z = zscores.get("ear")
            ear_c = None if ear_z is None else ear_z <= -self.ear_z_thresh

            zx, zy = zscores.get("gaze_dx"), zscores.get("gaze_dy")
            gaze_mag = None if zx is None or zy is None else (zx**2 + zy**2) ** 0.5
            gaze_c = None if gaze_mag is None else gaze_mag >= self.gaze_z_thresh

            roll_z = zscores.get("roll")
            roll_c = None if roll_z is None else abs(roll_z) >= self.head_z_thresh
            pitch_z = zscores.get("pitch")
            pitch_c = None if pitch_z is None else abs(pitch_z) >= self.head_z_thresh
            yaw_z = zscores.get("yaw")
            yaw_c = None if yaw_z is None else abs(yaw_z) >= self.head_z_thresh

            mouth_z = zscores.get("mar")
            mouth_c = None if mouth_z is None else mouth_z >= self.mouth_z_thresh

        closure_event = self._closure_tracker.update(t_now, ear_c, ear_z or 0.0)
        gaze_event = self._gaze_tracker.update(t_now, gaze_c, gaze_mag or 0.0)
        roll_event = self._roll_tracker.update(t_now, roll_c, roll_z or 0.0)
        pitch_event = self._pitch_tracker.update(t_now, pitch_c, pitch_z or 0.0)
        yaw_event = self._yaw_tracker.update(t_now, yaw_c, yaw_z or 0.0)
        mouth_event = self._mouth_tracker.update(t_now, mouth_c, mouth_z or 0.0)

        events = []
        if closure_event is not None:
            if closure_event.duration <= self.blink_max_duration:
                closure_event.event_type = EventType.BLINK
            events.append(closure_event)
        for event in (gaze_event, roll_event, pitch_event, yaw_event, mouth_event):
            if event is not None:
                events.append(event)
        return events


class WindowedEventStats:
    """Rolling-window statistics computed from completed events.

    Feeds the risk layer with, per event type: rate (events/minute), mean
    and max duration, and active-time fraction within the window -- the
    "how is this changing over the last N seconds" input your spec's Part 4
    and Part 19 ask for, in place of raw per-frame values.
    """

    def __init__(self, window_s=20.0):
        self.window_s = window_s
        self._events = deque()  # (end_time, Event)

    def add(self, event):
        self._events.append((event.end_time, event))

    def _trim(self, t_now):
        cutoff = t_now - self.window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def compute(self, t_now, active_durations=None):
        """Return ``{event_type: {rate_per_min, mean_duration, max_duration,
        active_fraction, count}}`` for every type seen within the window.

        Parameters
        ----------
        active_durations : dict[EventType, float] or None
            Elapsed duration of any event still ongoing right now (see
            ``EventEngine.active_snapshots``). Folded into
            ``active_fraction`` and ``max_duration`` so a sustained
            condition affects risk scoring *while it is happening*, not
            only after it ends.
        """
        self._trim(t_now)
        by_type = {}
        for _, event in self._events:
            by_type.setdefault(event.event_type, []).append(event)

        active_durations = active_durations or {}
        stats = {}
        for event_type in set(by_type) | set(active_durations):
            events = by_type.get(event_type, [])
            durations = [e.duration for e in events]
            active_time = sum(durations)

            ongoing = active_durations.get(event_type)
            duration_samples = list(durations)
            if ongoing is not None:
                active_time += min(ongoing, self.window_s)
                duration_samples.append(ongoing)
            if not duration_samples:
                continue

            stats[event_type] = {
                "count": len(events),
                "rate_per_min": len(events) / self.window_s * 60.0,
                "mean_duration": sum(duration_samples) / len(duration_samples),
                "max_duration": max(duration_samples),
                "active_fraction": min(active_time / self.window_s, 1.0),
            }
        return stats
