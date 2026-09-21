"""Alert state machine: persistence, hysteresis, cooldown, recovery.

New to this project. The base project has no state machine at all -- each
of its four booleans fires the instant its own timer crosses a threshold,
and clears the instant it drops back below, with no memory of having just
alerted. This module gives each risk pathway (fatigue, distraction) an
explicit state with hysteresis (a higher bar to enter WARNING than to stay
there), a cooldown after recovery (so a risk score bobbing right at the
threshold does not flap the alert on and off), and a dedicated
``DATA_INVALID`` state that takes priority over both -- an invalid
observation must never be interpreted as either NORMAL or risky.

States, per pathway, in the order the master prompt specifies:
``NORMAL -> MONITORING -> ELEVATED_RISK -> WARNING -> RECOVERY -> NORMAL``,
with ``DATA_INVALID`` reachable (and exitable back to whatever state
preceded it) from any state.
"""

from dataclasses import dataclass, field
from enum import Enum


class AlertState(str, Enum):
    NORMAL = "NORMAL"
    MONITORING = "MONITORING"
    ELEVATED_RISK = "ELEVATED_RISK"
    WARNING = "WARNING"
    RECOVERY = "RECOVERY"
    DATA_INVALID = "DATA_INVALID"


@dataclass
class AlertUpdate:
    state: AlertState
    risk: float
    reasons: list = field(default_factory=list)
    changed: bool = False


class PathwayStateMachine:
    """One risk pathway's (fatigue or distraction) alert state machine.

    Parameters
    ----------
    monitoring_thresh, elevated_thresh, warning_thresh : float
        Risk-score thresholds to *enter* each state, in increasing order.
    recovery_thresh : float
        Risk must drop below this (lower than ``monitoring_thresh``, i.e.
        real hysteresis, not just the same threshold in reverse) before a
        WARNING/ELEVATED_RISK state begins RECOVERY.
    persistence_s : float
        Risk must stay at or above a threshold for this many seconds before
        the corresponding state transition actually happens -- this is what
        stops a single noisy frame from firing an alert (the master
        prompt's "do not create alerts from one frame").
    recovery_s : float
        Time risk must stay below ``recovery_thresh`` before RECOVERY
        completes and the pathway returns to NORMAL.
    cooldown_s : float
        After returning to NORMAL from RECOVERY, minimum time before this
        pathway can re-enter WARNING, even if risk spikes again --
        suppresses immediate re-alerting on a still-noisy signal.
    """

    def __init__(
        self,
        monitoring_thresh=0.15,
        elevated_thresh=0.35,
        warning_thresh=0.6,
        recovery_thresh=0.1,
        persistence_s=5.0,
        recovery_s=5.0,
        cooldown_s=10.0,
    ):
        self.monitoring_thresh = monitoring_thresh
        self.elevated_thresh = elevated_thresh
        self.warning_thresh = warning_thresh
        self.recovery_thresh = recovery_thresh
        self.persistence_s = persistence_s
        self.recovery_s = recovery_s
        self.cooldown_s = cooldown_s

        self.state = AlertState.NORMAL
        self._pending_state = None
        self._pending_since = None
        self._recovery_since = None
        self._cooldown_until = None
        self._pre_invalid_state = AlertState.NORMAL

    def _target_state(self, risk):
        if risk >= self.warning_thresh:
            return AlertState.WARNING
        if risk >= self.elevated_thresh:
            return AlertState.ELEVATED_RISK
        if risk >= self.monitoring_thresh:
            return AlertState.MONITORING
        return AlertState.NORMAL

    def update(self, t_now, risk, reasons, quality_is_invalid):
        previous_state = self.state

        if quality_is_invalid:
            if self.state != AlertState.DATA_INVALID:
                self._pre_invalid_state = self.state
            self.state = AlertState.DATA_INVALID
            self._pending_state = None
            self._pending_since = None
            return AlertUpdate(self.state, risk, [], changed=self.state != previous_state)

        if self.state == AlertState.DATA_INVALID:
            # Observation quality recovered: resume from where we left off
            # rather than snapping straight back to NORMAL, since the risk
            # may well still be elevated once we can see the pilot again.
            self.state = self._pre_invalid_state

        target = self._target_state(risk)
        escalating = _severity(target) > _severity(self.state)
        cooling_down = (
            target == AlertState.WARNING
            and self._cooldown_until is not None
            and t_now < self._cooldown_until
        )
        if cooling_down:
            target = min(target, AlertState.ELEVATED_RISK, key=_severity)

        if escalating and target != self.state:
            if self._pending_state != target:
                self._pending_state = target
                self._pending_since = t_now
            elif t_now - self._pending_since >= self.persistence_s:
                self.state = target
                self._pending_state = None
                self._recovery_since = None
        elif target == self.state:
            self._pending_state = None
        elif _severity(target) < _severity(self.state):
            # Risk has dropped below the current state's own threshold:
            # start/continue RECOVERY rather than dropping straight down,
            # unless we're already at NORMAL/MONITORING.
            if self.state in (AlertState.ELEVATED_RISK, AlertState.WARNING):
                if risk <= self.recovery_thresh:
                    if self.state != AlertState.RECOVERY:
                        self._recovery_since = t_now
                        self.state = AlertState.RECOVERY
                    elif t_now - self._recovery_since >= self.recovery_s:
                        self.state = AlertState.NORMAL
                        self._cooldown_until = t_now + self.cooldown_s
                else:
                    self._recovery_since = None
            elif self.state == AlertState.RECOVERY:
                if risk <= self.recovery_thresh:
                    if t_now - self._recovery_since >= self.recovery_s:
                        self.state = AlertState.NORMAL
                        self._cooldown_until = t_now + self.cooldown_s
                else:
                    self._recovery_since = t_now
            else:
                self.state = target
            self._pending_state = None

        return AlertUpdate(
            state=self.state,
            risk=risk,
            reasons=list(reasons),
            changed=self.state != previous_state,
        )


_SEVERITY_ORDER = {
    AlertState.NORMAL: 0,
    AlertState.RECOVERY: 1,
    AlertState.MONITORING: 1,
    AlertState.ELEVATED_RISK: 2,
    AlertState.WARNING: 3,
    AlertState.DATA_INVALID: -1,  # handled separately; never compared for escalation
}


def _severity(state):
    return _SEVERITY_ORDER[state]


class AlertStateMachine:
    """Owns one ``PathwayStateMachine`` each for fatigue and distraction."""

    def __init__(self, **kwargs):
        self.fatigue = PathwayStateMachine(**kwargs)
        self.distraction = PathwayStateMachine(**kwargs)

    def update(self, t_now, risk_result, quality_is_invalid):
        fatigue_update = self.fatigue.update(
            t_now, risk_result.fatigue_risk, risk_result.fatigue_reasons, quality_is_invalid
        )
        distraction_update = self.distraction.update(
            t_now,
            risk_result.distraction_risk,
            risk_result.distraction_reasons,
            quality_is_invalid,
        )
        return fatigue_update, distraction_update
