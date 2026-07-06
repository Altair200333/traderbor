from __future__ import annotations

from traderbot_ai.screener.config import ScreenerConfig
from traderbot_ai.screener.gates import GateResult


# Gates where a larger passing margin means a stronger signal.
HIGHER_BETTER = ("S1", "S2", "S3")
# Ceiling gates where more headroom below the threshold means a fresher, less chased setup.
LOWER_BETTER = ("S9a", "S9b", "S9c", "S10", "S11")

HIGHER_MARGIN_CAP = 2.0
LOWER_MARGIN_CAP = 1.0
MARGINAL_QUALITY_PENALTY = 1.0


def score_candidate(gates: dict[str, GateResult], quality: str | None, cfg: ScreenerConfig) -> float:
    """Rank-only score from gate margins. Internal to candidate selection; never shown to the agent."""
    total = 0.0
    weights = cfg.score_weights
    for name in HIGHER_BETTER:
        margin = _margin(gates.get(name), higher_better=True)
        if margin is not None:
            total += weights.get(name, 1.0) * min(margin, HIGHER_MARGIN_CAP)
    for name in LOWER_BETTER:
        margin = _margin(gates.get(name), higher_better=False)
        if margin is not None:
            total += weights.get(name, 1.0) * min(margin, LOWER_MARGIN_CAP)
    if quality == "marginal_extension":
        total -= MARGINAL_QUALITY_PENALTY
    return total


def _margin(gate: GateResult | None, *, higher_better: bool) -> float | None:
    if gate is None or gate.value is None or gate.threshold is None:
        return None
    try:
        value = abs(float(gate.value))
        threshold = abs(float(gate.threshold))
    except (TypeError, ValueError):
        return None
    if threshold == 0:
        return None
    if higher_better:
        return (value - threshold) / threshold
    return (threshold - value) / threshold
