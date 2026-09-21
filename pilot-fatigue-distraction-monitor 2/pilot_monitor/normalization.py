"""Robust, per-pilot normalization of behavioural features.

New to this project. Turns raw feature values into "how many robust-sigma
away from *this pilot's* normal is this," using the median/MAD baseline
from ``calibration.py``. Kept as a thin, separate module from calibration
itself so that "how we normalize" (this file) is independently swappable
from "how we estimate what's normal" (calibration.py) -- e.g. a future
adaptive/rolling baseline (see the design document's discussion of
controlled statistical adaptation) can replace calibration.py's fixed
one-shot profile without this module changing at all.
"""

from dataclasses import dataclass

from pilot_monitor.calibration import FEATURE_NAMES


@dataclass
class NormalizedObservation:
    raw: dict
    zscores: dict


def normalize(profile, features):
    """Return a ``NormalizedObservation`` for one frame's feature dict.

    Parameters
    ----------
    profile : pilot_monitor.calibration.PersonalProfile
    features : dict[str, float or None]
        One frame's raw feature values, keyed by name (a subset of
        ``FEATURE_NAMES`` is fine; missing/``None`` values normalize to
        ``None`` rather than raising).
    """
    zscores = {
        name: profile.robust_zscore(name, features.get(name))
        for name in FEATURE_NAMES
    }
    return NormalizedObservation(raw=dict(features), zscores=zscores)
