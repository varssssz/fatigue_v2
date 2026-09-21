"""Render an ``AlertUpdate`` into the explanation text the master prompt
asks for -- never a bare "AI detected fatigue," always the observations
that led there, in scientific-proxy language rather than a diagnostic claim.
"""

from pilot_monitor.alerts import AlertState


def explain(pathway_name, update):
    """Return a short, human-readable explanation string, or ``None`` if the
    pathway is not currently in an alerting state worth explaining."""
    if update.state in (AlertState.NORMAL, AlertState.MONITORING, AlertState.DATA_INVALID):
        return None

    header = {
        "fatigue": "FATIGUE-RELATED BEHAVIOURAL RISK",
        "distraction": "DISTRACTION-RELATED BEHAVIOURAL RISK",
    }[pathway_name]

    lines = [f"{header} ({update.state.value}, risk={update.risk:.2f})"]
    if update.reasons:
        lines.append("Contributing observations:")
        lines.extend(f"  • {reason}" for reason in update.reasons)
    else:
        lines.append("  • risk score elevated; no single dominant contributor")
    return "\n".join(lines)
