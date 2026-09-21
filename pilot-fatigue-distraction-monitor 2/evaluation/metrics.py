"""Precision / recall / F1 / false-positive / false-negative rate, computed
frame-by-frame against ground truth. Deliberately dependency-free (no
scikit-learn) so the evaluation harness has no extra runtime requirement
beyond numpy.
"""

from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    n: int


def compute_metrics(y_true, y_pred):
    """``y_true``/``y_pred`` are equal-length sequences of 0/1 (or bool).
    Frames where either is ``None`` are excluded (e.g. DATA_INVALID frames
    that a system correctly declined to judge)."""
    tp = fp = tn = fn = 0
    for truth, pred in zip(y_true, y_pred):
        if truth is None or pred is None:
            continue
        truth, pred = bool(truth), bool(pred)
        if truth and pred:
            tp += 1
        elif not truth and pred:
            fp += 1
        elif not truth and not pred:
            tn += 1
        else:
            fn += 1

    n = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        n=n,
    )


def alert_flap_rate(states, minutes):
    """Number of NORMAL/MONITORING/RECOVERY -> ELEVATED_RISK/WARNING
    transitions per minute -- how often the system re-alerts, independent
    of whether those alerts were correct."""
    from pilot_monitor.alerts import AlertState

    alerting = {AlertState.ELEVATED_RISK, AlertState.WARNING}
    transitions = 0
    was_alerting = False
    for state in states:
        is_alerting = state in alerting
        if is_alerting and not was_alerting:
            transitions += 1
        was_alerting = is_alerting
    return transitions / minutes if minutes > 0 else 0.0
