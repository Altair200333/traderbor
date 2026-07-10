"""Overlay weight computation vs the frozen research rule."""
from __future__ import annotations

import math

import numpy as np
import pytest

from bot.strategy.liqrev_v2 import Overlay


def test_weight_matches_research_formula():
    scores = sorted(np.linspace(-0.05, 0.10, 100).tolist())
    ov = Overlay(scores)
    for btc in (-0.12, -0.03, 0.0, 0.02, 0.5):
        s = -btc
        p = np.searchsorted(np.array(scores), s, side="right") / len(scores)
        assert ov.weight(btc) == pytest.approx(min(2.0, 2.0 * p))


def test_weight_monotone_decreasing_in_btc_ret():
    ov = Overlay(sorted(np.random.default_rng(0).normal(0, 0.03, 500).tolist()))
    btcs = np.linspace(-0.15, 0.10, 50)
    ws = [ov.weight(b) for b in btcs]
    assert all(a >= b for a, b in zip(ws, ws[1:]))   # deeper BTC dump => bigger w


def test_weight_caps_at_two():
    ov = Overlay([0.0, 0.01, 0.02])
    assert ov.weight(-0.99) == 2.0


def test_missing_btc_is_neutral():
    ov = Overlay([0.0, 0.01])
    assert ov.weight(None) == 1.0
    assert ov.weight(float("nan")) == 1.0


def test_frozen_artifact_loads():
    ov = Overlay.load()
    assert len(ov.scores) == 839                     # frozen DEV distribution
    assert ov.scores == sorted(ov.scores)
    assert all(math.isfinite(s) for s in ov.scores)
    # spot values from the deploy spec: mean weight ~0.65 on holdout-ish inputs
    assert 0.0 < ov.weight(-0.005) < 1.0             # calm BTC -> below full size
    assert ov.weight(-0.15) == 2.0                   # deep market-wide dump -> cap
