import math
import random
import unittest
from datetime import date, timedelta

from src.model import (
    FootballModel,
    apply_dixon_coles,
    dixon_coles_tau,
    market_probabilities,
    poisson_grid,
    rho_bounds,
)


def sample_matches(n=400, seed=1, rho=0.0):
    """Synthetic league. With rho != 0 the low scores are generated with a Dixon-Coles dependence."""
    rng = random.Random(seed)
    teams = [f"T{i}" for i in range(10)]
    matches = []
    for i in range(n):
        home, away = rng.sample(teams, 2)
        lam_h, lam_a = 1.5, 1.2
        grid = (
            apply_dixon_coles(poisson_grid(lam_h, lam_a), lam_h, lam_a, rho)
            if rho
            else poisson_grid(lam_h, lam_a)
        )
        flat = [(h, a, p) for h, row in enumerate(grid) for a, p in enumerate(row)]
        r, acc = rng.random() * sum(p for _, _, p in flat), 0.0
        for h, a, p in flat:
            acc += p
            if acc >= r:
                break
        matches.append(
            {
                "match_date": (date(2025, 1, 1) + timedelta(days=i // 5)).isoformat(),
                "home_team": home,
                "away_team": away,
                "home_goals": h,
                "away_goals": a,
                "is_final": 1,
            }
        )
    return matches


class GridAndMarketTests(unittest.TestCase):
    def test_independent_btts_matches_closed_form(self):
        lam_h, lam_a = 1.7, 1.1
        markets = market_probabilities(poisson_grid(lam_h, lam_a))
        expected = (1 - math.exp(-lam_h)) * (1 - math.exp(-lam_a))
        self.assertAlmostEqual(
            markets["both_teams_score_probability"], expected, places=4
        )

    def test_clean_sheets_match_closed_form(self):
        markets = market_probabilities(poisson_grid(1.7, 1.1))
        self.assertAlmostEqual(
            markets["home_clean_sheet_probability"], math.exp(-1.1), places=4
        )
        self.assertAlmostEqual(
            markets["away_clean_sheet_probability"], math.exp(-1.7), places=4
        )

    def test_expected_total_goals(self):
        markets = market_probabilities(poisson_grid(1.7, 1.1))
        self.assertAlmostEqual(markets["expected_total_goals"], 2.8, places=3)

    def test_over_lines_are_ordered(self):
        m = market_probabilities(poisson_grid(1.4, 1.2))
        self.assertGreater(m["over_1_5_probability"], m["over_2_5_probability"])
        self.assertGreater(m["over_2_5_probability"], m["over_3_5_probability"])

    def test_dixon_coles_keeps_total_mass(self):
        grid = poisson_grid(1.4, 1.1)
        for rho in (-0.15, -0.05, 0.05, 0.1):
            adjusted = apply_dixon_coles(grid, 1.4, 1.1, rho)
            self.assertAlmostEqual(
                sum(map(sum, adjusted)), sum(map(sum, grid)), places=12
            )

    def test_negative_rho_raises_low_draws_and_lowers_1_0(self):
        grid = poisson_grid(1.4, 1.1)
        adjusted = apply_dixon_coles(grid, 1.4, 1.1, -0.1)
        self.assertGreater(adjusted[0][0], grid[0][0])
        self.assertGreater(adjusted[1][1], grid[1][1])
        self.assertLess(adjusted[1][0], grid[1][0])
        self.assertLess(adjusted[0][1], grid[0][1])
        self.assertEqual(adjusted[2][3], grid[2][3])  # other scores untouched

    def test_rho_zero_is_identity_and_input_not_mutated(self):
        grid = poisson_grid(1.4, 1.1)
        copy = [row[:] for row in grid]
        self.assertEqual(apply_dixon_coles(grid, 1.4, 1.1, 0.0), grid)
        apply_dixon_coles(grid, 1.4, 1.1, -0.2)
        self.assertEqual(grid, copy)

    def test_extreme_rho_is_clipped_to_keep_probabilities_positive(self):
        for rho in (-5.0, 5.0):
            adjusted = apply_dixon_coles(poisson_grid(2.5, 2.0), 2.5, 2.0, rho)
            self.assertTrue(all(value > 0 for row in adjusted for value in row))

    def test_tau_bounds(self):
        lower, upper = rho_bounds(1.4, 1.1)
        for rho in (lower + 1e-6, upper - 1e-6):
            for h in (0, 1):
                for a in (0, 1):
                    self.assertGreater(dixon_coles_tau(h, a, 1.4, 1.1, rho), 0)


class ModelIntegrationTests(unittest.TestCase):
    def test_prediction_has_new_market_fields_and_old_ones(self):
        model = FootballModel(xg_weight=0).fit(sample_matches())
        p = model.predict("T1", "T2")
        for key in (
            "home_win_probability",
            "draw_probability",
            "away_win_probability",
            "most_likely_score",
            "top_scorelines",
            "expected_home_goals",
            "both_teams_score_probability",
            "over_2_5_probability",
            "home_clean_sheet_probability",
            "away_clean_sheet_probability",
        ):
            self.assertIn(key, p)
        self.assertAlmostEqual(
            p["home_win_probability"]
            + p["draw_probability"]
            + p["away_win_probability"],
            1.0,
            places=3,
        )
        self.assertIsNone(p["dixon_coles_rho"])

    def test_dixon_coles_off_matches_default(self):
        data = sample_matches()
        self.assertEqual(
            FootballModel(xg_weight=0).fit(data).predict("T1", "T2"),
            FootballModel(xg_weight=0, dixon_coles=False).fit(data).predict("T1", "T2"),
        )

    def test_probabilities_stay_valid_with_dixon_coles(self):
        model = FootballModel(xg_weight=0, dixon_coles=True).fit(
            sample_matches(rho=-0.2)
        )
        for home, away in (("T1", "T2"), ("T3", "T9"), ("new", "T1")):
            p = model.predict(home, away)
            total = (
                p["home_win_probability"]
                + p["draw_probability"]
                + p["away_win_probability"]
            )
            self.assertAlmostEqual(total, 1.0, places=3)
            self.assertTrue(all(s["probability"] >= 0 for s in p["top_scorelines"]))

    def test_rho_estimation_recovers_sign_and_rough_size(self):
        data = sample_matches(n=6000, seed=7, rho=-0.2)
        model = FootballModel(xg_weight=0, dixon_coles=True, half_life_days=100000).fit(
            data
        )
        self.assertLess(model.rho, -0.08)
        self.assertGreater(model.rho, -0.35)

    def test_rho_is_near_zero_for_independent_data(self):
        data = sample_matches(n=6000, seed=11, rho=0.0)
        model = FootballModel(xg_weight=0, dixon_coles=True, half_life_days=100000).fit(
            data
        )
        self.assertLess(abs(model.rho), 0.08)

    def test_tiny_dataset_leaves_rho_at_zero(self):
        model = FootballModel(xg_weight=0, dixon_coles=True).fit(sample_matches(n=30))
        self.assertEqual(model.rho, 0.0)


if __name__ == "__main__":
    unittest.main()
