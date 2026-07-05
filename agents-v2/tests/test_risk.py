from __future__ import annotations

import unittest

from traderbot_ai.tools.risk import _calculate_position_size_impl, _validate_order_impl


class RiskToolTests(unittest.TestCase):
    def test_validate_order_rejects_too_tight_stop(self) -> None:
        result = _validate_order_impl(
            final_decision="long",
            price=100.0,
            stop_loss=99.5,
            take_profit=102.0,
            amount=100.0,
            balance_usdt=1000.0,
        )

        self.assertFalse(result["ok"])
        self.assertTrue(any("below minimum" in error for error in result["errors"]))

    def test_validate_order_rejects_low_reward_risk(self) -> None:
        result = _validate_order_impl(
            final_decision="long",
            price=100.0,
            stop_loss=98.0,
            take_profit=102.0,
            amount=100.0,
            balance_usdt=1000.0,
        )

        self.assertFalse(result["ok"])
        self.assertTrue(any("reward:risk" in error for error in result["errors"]))

    def test_validate_order_default_loss_cap_is_point_75_percent(self) -> None:
        result = _validate_order_impl(
            final_decision="long",
            price=100.0,
            stop_loss=98.0,
            take_profit=104.0,
            amount=500.0,
            balance_usdt=1000.0,
        )

        self.assertFalse(result["ok"])
        self.assertTrue(any("exceeds allowed loss 7.5" in error for error in result["errors"]))

    def test_validate_order_rejects_fee_gate_failure(self) -> None:
        result = _validate_order_impl(
            final_decision="long",
            price=100.0,
            stop_loss=98.0,
            take_profit=103.0,
            amount=100.0,
            balance_usdt=1000.0,
            round_trip_fee_fraction=0.01,
        )

        self.assertFalse(result["ok"])
        self.assertTrue(any("fee/funding/slippage gate" in error for error in result["errors"]))

    def test_calculate_position_size_is_not_capped_at_30_percent_by_default(self) -> None:
        result = _calculate_position_size_impl(
            balance_usdt=1000.0,
            price=100.0,
            stop_loss=98.0,
        )

        self.assertEqual(result["risk_budget"], 7.5)
        self.assertEqual(result["risk_based_amount"], 375.0)
        self.assertEqual(result["amount"], 375.0)

    def test_calculate_position_size_uses_daily_budget_share(self) -> None:
        result = _calculate_position_size_impl(
            balance_usdt=1000.0,
            price=100.0,
            stop_loss=98.0,
            daily_risk_budget_remaining=5.0,
            size_pct_of_daily_budget=50.0,
        )

        self.assertEqual(result["risk_budget"], 2.5)
        self.assertEqual(result["amount"], 125.0)

    def test_calculate_position_size_defaults_to_strategy_daily_budget_share(self) -> None:
        result = _calculate_position_size_impl(
            balance_usdt=1000.0,
            price=100.0,
            stop_loss=98.0,
            daily_risk_budget_remaining=10.0,
        )

        self.assertEqual(result["risk_budget"], 3.5)
        self.assertEqual(result["amount"], 175.0)

    def test_calculate_position_size_clamps_daily_budget_share_to_strategy_range(self) -> None:
        low = _calculate_position_size_impl(
            balance_usdt=1000.0,
            price=100.0,
            stop_loss=98.0,
            daily_risk_budget_remaining=10.0,
            size_pct_of_daily_budget=1.0,
        )
        high = _calculate_position_size_impl(
            balance_usdt=1000.0,
            price=100.0,
            stop_loss=98.0,
            daily_risk_budget_remaining=10.0,
            size_pct_of_daily_budget=99.0,
        )

        self.assertEqual(low["risk_budget"], 1.5)
        self.assertEqual(high["risk_budget"], 5.0)


if __name__ == "__main__":
    unittest.main()
