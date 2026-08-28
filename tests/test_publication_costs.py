import unittest

import pandas as pd

from strategies.publication_costs import build_publication_fold_liquidity_tiers


class PublicationCostsTests(unittest.TestCase):
    def _panel(self) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=70, freq="D")
        rows = []
        for asset_id, dollar_volume in [(1, 100.0), (2, 50.0), (3, 10.0)]:
            for trade_date in dates:
                rows.append(
                    {
                        "asset_id": asset_id,
                        "trade_date": trade_date,
                        "dollar_volume": dollar_volume,
                        "member_of_universe": True,
                    }
                )
        return pd.DataFrame(rows)

    def test_formation_window_ranking_assigns_expected_tiers(self):
        prices = self._panel()
        result = build_publication_fold_liquidity_tiers(
            prices,
            formation_start=pd.Timestamp("2020-01-01"),
            formation_end=pd.Timestamp("2020-03-10"),
            evaluation_identities=[1, 2, 3],
            min_liquidity_observations=60,
        )

        # Preserve the frozen rank / N percentile convention. With only three
        # ranked identities, the percentiles are 1/3, 2/3 and 1, so the first
        # two fall in the medium tier and the third in the lower tier.
        self.assertEqual(result.tier_map.loc[1], "medium")
        self.assertEqual(result.tier_map.loc[2], "medium")
        self.assertEqual(result.tier_map.loc[3], "lower")

    def test_future_observations_do_not_change_formation_tiers(self):
        prices = self._panel()
        baseline = build_publication_fold_liquidity_tiers(
            prices,
            formation_start=pd.Timestamp("2020-01-01"),
            formation_end=pd.Timestamp("2020-03-10"),
            evaluation_identities=[1, 2, 3],
        ).tier_map

        future = pd.DataFrame(
            {
                "asset_id": [3] * 20,
                "trade_date": pd.date_range("2020-04-01", periods=20, freq="D"),
                "dollar_volume": [1_000_000.0] * 20,
                "member_of_universe": [True] * 20,
            }
        )
        altered = build_publication_fold_liquidity_tiers(
            pd.concat([prices, future], ignore_index=True),
            formation_start=pd.Timestamp("2020-01-01"),
            formation_end=pd.Timestamp("2020-03-10"),
            evaluation_identities=[1, 2, 3],
        ).tier_map

        pd.testing.assert_series_equal(baseline, altered)

    def test_only_formation_end_members_enter_cross_sectional_ranking(self):
        prices = self._panel()
        formation_end = pd.Timestamp("2020-03-10")
        prices.loc[
            (prices["asset_id"] == 1) & (prices["trade_date"] == formation_end),
            "member_of_universe",
        ] = False

        result = build_publication_fold_liquidity_tiers(
            prices,
            formation_start=pd.Timestamp("2020-01-01"),
            formation_end=formation_end,
            evaluation_identities=[1, 2, 3],
        )

        self.assertEqual(result.tier_map.loc[1], "lower")
        ranked_ids = set(result.diagnostics["asset_id"])
        self.assertNotIn(1, ranked_ids)

    def test_insufficient_history_receives_conservative_lower_tier(self):
        prices = self._panel()
        prices = prices.loc[
            ~((prices["asset_id"] == 1) & (prices["trade_date"] < "2020-02-20"))
        ].copy()

        result = build_publication_fold_liquidity_tiers(
            prices,
            formation_start=pd.Timestamp("2020-01-01"),
            formation_end=pd.Timestamp("2020-03-10"),
            evaluation_identities=[1, 2, 3],
            min_liquidity_observations=60,
        )

        self.assertEqual(result.tier_map.loc[1], "lower")
        diagnostic = result.diagnostics.set_index("asset_id").loc[1]
        self.assertFalse(bool(diagnostic["sufficient_liquidity_history"]))
        self.assertEqual(
            diagnostic["tier_assignment_reason"],
            "insufficient_history_conservative_lower",
        )

    def test_new_evaluation_identity_defaults_to_lower_tier(self):
        prices = self._panel()
        result = build_publication_fold_liquidity_tiers(
            prices,
            formation_start=pd.Timestamp("2020-01-01"),
            formation_end=pd.Timestamp("2020-03-10"),
            evaluation_identities=[1, 2, 3, 99],
        )

        self.assertEqual(result.tier_map.loc[99], "lower")

    def test_zero_dollar_volume_is_valid_liquidity_information(self):
        prices = self._panel()
        prices.loc[prices["asset_id"] == 3, "dollar_volume"] = 0.0

        result = build_publication_fold_liquidity_tiers(
            prices,
            formation_start=pd.Timestamp("2020-01-01"),
            formation_end=pd.Timestamp("2020-03-10"),
            evaluation_identities=[1, 2, 3],
        )

        diagnostic = result.diagnostics.set_index("asset_id").loc[3]
        self.assertEqual(int(diagnostic["liquidity_observation_count"]), 70)
        self.assertEqual(float(diagnostic["median_dollar_volume"]), 0.0)


if __name__ == "__main__":
    unittest.main()
