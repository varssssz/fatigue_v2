"""Separate, explainable fatigue and distraction risk scoring.

New to this project. The base project produces four independent booleans
(TIRED/ASLEEP/LOOKING AWAY/DISTRACTED) with no notion of "why," and mixes
signals that should stay conceptually separate (Part 11 of the design
document). This module reads the windowed event statistics from
``events.py`` -- plus, optionally, the per-pilot GRU anomaly score from
``temporal_model.py`` -- and produces two bounded [0, 1] risk scores with an
explicit, itemized reason list each, rather than a single opaque number.

Weights are a documented starting point, not a fitted or validated result:
they encode the ordering described in the design document (sustained
closure dominates fatigue risk; a single yawn should not, per the
master-prompt instruction not to overuse yawning) and should be tuned
against the evaluation framework in ``evaluation/`` once real labeled data
exists, not treated as final.
"""

from dataclasses import dataclass, field

from pilot_monitor.events import EventType


@dataclass
class RiskResult:
    fatigue_risk: float
    distraction_risk: float
    fatigue_reasons: list = field(default_factory=list)
    distraction_reasons: list = field(default_factory=list)


class RiskScorer:
    def __init__(
        self,
        closure_active_weight=0.40,
        closure_rate_weight=0.15,
        blink_duration_weight=0.15,
        mouth_weight=0.10,
        gaze_active_weight=0.40,
        gaze_rate_weight=0.15,
        head_active_weight=0.35,
        gru_fatigue_weight=0.20,
        gru_distraction_weight=0.10,
        typical_blink_duration=0.25,
        reason_threshold=0.05,
    ):
        self.closure_active_weight = closure_active_weight
        self.closure_rate_weight = closure_rate_weight
        self.blink_duration_weight = blink_duration_weight
        self.mouth_weight = mouth_weight
        self.gaze_active_weight = gaze_active_weight
        self.gaze_rate_weight = gaze_rate_weight
        self.head_active_weight = head_active_weight
        self.gru_fatigue_weight = gru_fatigue_weight
        self.gru_distraction_weight = gru_distraction_weight
        self.typical_blink_duration = typical_blink_duration
        self.reason_threshold = reason_threshold

    def score(self, window_stats, gru_anomaly=None):
        """Compute fatigue and distraction risk from one window's statistics.

        Parameters
        ----------
        window_stats : dict[EventType, dict]
            Output of ``events.WindowedEventStats.compute(t_now)``.
        gru_anomaly : float or None
            Robust z-score of GRU reconstruction error (see
            ``temporal_model.TrainedTemporalModel.anomaly_zscore``), or
            ``None`` if no trained model is available for this pilot. Only
            the positive part is used -- a *better*-than-usual
            reconstruction is not "negative risk."
        """
        fatigue_terms = []
        distraction_terms = []

        closure = window_stats.get(EventType.PROLONGED_CLOSURE)
        if closure:
            fatigue_terms.append(
                (
                    self.closure_active_weight * closure["active_fraction"],
                    f"eyes closed for {closure['active_fraction'] * 100:.0f}% of the last window "
                    f"(longest closure {closure['max_duration']:.1f}s)",
                )
            )
            fatigue_terms.append(
                (
                    self.closure_rate_weight * min(closure["rate_per_min"] / 10.0, 1.0),
                    f"{closure['rate_per_min']:.1f} prolonged-closure events/min",
                )
            )

        blink = window_stats.get(EventType.BLINK)
        if blink and blink["mean_duration"] > self.typical_blink_duration:
            excess = (blink["mean_duration"] - self.typical_blink_duration) / self.typical_blink_duration
            fatigue_terms.append(
                (
                    self.blink_duration_weight * min(excess, 1.0),
                    f"blinks averaging {blink['mean_duration']:.2f}s, longer than typical",
                )
            )

        mouth = window_stats.get(EventType.MOUTH_OPENING)
        if mouth:
            # Deliberately capped low and only from sustained/frequent
            # mouth-opening, never a single event -- per the design
            # document's instruction that yawning is a secondary signal,
            # not an independent fatigue trigger.
            contribution = self.mouth_weight * min(mouth["active_fraction"] * 2, 1.0)
            if mouth["count"] >= 2:
                fatigue_terms.append(
                    (
                        contribution,
                        f"{mouth['count']} mouth-opening events in the last window",
                    )
                )

        if gru_anomaly is not None and gru_anomaly > 0:
            fatigue_terms.append(
                (
                    self.gru_fatigue_weight * min(gru_anomaly / 3.0, 1.0),
                    f"overall behaviour pattern deviates from this pilot's normal "
                    f"(temporal anomaly z={gru_anomaly:.1f})",
                )
            )
            distraction_terms.append(
                (
                    self.gru_distraction_weight * min(gru_anomaly / 3.0, 1.0),
                    f"overall behaviour pattern deviates from this pilot's normal "
                    f"(temporal anomaly z={gru_anomaly:.1f})",
                )
            )

        gaze = window_stats.get(EventType.GAZE_DEVIATION)
        if gaze:
            distraction_terms.append(
                (
                    self.gaze_active_weight * gaze["active_fraction"],
                    f"gaze off personal-normal direction for {gaze['active_fraction'] * 100:.0f}% "
                    "of the last window",
                )
            )
            distraction_terms.append(
                (
                    self.gaze_rate_weight * min(gaze["rate_per_min"] / 10.0, 1.0),
                    f"{gaze['rate_per_min']:.1f} gaze-deviation events/min",
                )
            )

        head_fraction = 0.0
        head_axes = []
        for event_type, label in (
            (EventType.HEAD_DEVIATION_YAW, "yaw"),
            (EventType.HEAD_DEVIATION_PITCH, "pitch"),
            (EventType.HEAD_DEVIATION_ROLL, "roll"),
        ):
            stat = window_stats.get(event_type)
            if stat:
                head_fraction = max(head_fraction, stat["active_fraction"])
                head_axes.append(f"{label} {stat['active_fraction'] * 100:.0f}%")
        if head_axes:
            distraction_terms.append(
                (
                    self.head_active_weight * head_fraction,
                    f"head orientation off personal-normal ({', '.join(head_axes)} of window)",
                )
            )

        fatigue_risk = min(sum(term for term, _ in fatigue_terms), 1.0)
        distraction_risk = min(sum(term for term, _ in distraction_terms), 1.0)

        fatigue_reasons = [
            reason for term, reason in fatigue_terms if term >= self.reason_threshold
        ]
        distraction_reasons = [
            reason for term, reason in distraction_terms if term >= self.reason_threshold
        ]

        return RiskResult(
            fatigue_risk=fatigue_risk,
            distraction_risk=distraction_risk,
            fatigue_reasons=fatigue_reasons,
            distraction_reasons=distraction_reasons,
        )
