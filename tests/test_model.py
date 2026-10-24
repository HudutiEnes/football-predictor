import unittest

from src.model import DEFAULT_HALF_LIFE_DAYS, FootballModel


class FootballModelTests(unittest.TestCase):
    def setUp(self):
        self.matches = [
            {
                "match_date": "2025-08-10",
                "home_team": "A",
                "away_team": "B",
                "home_goals": 2,
                "away_goals": 0,
                "home_xg": 1.4,
                "away_xg": 0.5,
                "is_final": 1,
            },
            {
                "match_date": "2025-08-17",
                "home_team": "B",
                "away_team": "A",
                "home_goals": 1,
                "away_goals": 1,
                "home_xg": 1.8,
                "away_xg": 1.2,
                "is_final": 1,
            },
            {
                "match_date": "2025-08-24",
                "home_team": "A",
                "away_team": "B",
                "home_goals": 0,
                "away_goals": 1,
                "home_xg": 0.7,
                "away_xg": 0.9,
                "is_final": 1,
            },
        ]

    def test_default_half_life_is_current_tuned_candidate(self):
        self.assertEqual(FootballModel().half_life_days, DEFAULT_HALF_LIFE_DAYS)
        self.assertEqual(DEFAULT_HALF_LIFE_DAYS, 365)

    def test_xg_weight_zero_ignores_xg(self):
        no_xg = [dict(match, home_xg=None, away_xg=None) for match in self.matches]
        with_xg_model = FootballModel(half_life_days=180, xg_weight=0).fit(self.matches)
        no_xg_model = FootballModel(half_life_days=180, xg_weight=0).fit(no_xg)
        self.assertEqual(
            with_xg_model.predict("A", "B"),
            no_xg_model.predict("A", "B"),
        )

    def test_probabilities_sum_to_one(self):
        model = FootballModel(xg_weight=0.5).fit(self.matches)
        prediction = model.predict("A", "B")
        total = (
            prediction["home_win_probability"]
            + prediction["draw_probability"]
            + prediction["away_win_probability"]
        )
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_unknown_teams_still_get_valid_prediction(self):
        model = FootballModel().fit(self.matches)
        prediction = model.predict("Unknown Home", "Unknown Away")
        self.assertGreater(prediction["expected_home_goals"], 0)
        self.assertGreater(prediction["expected_away_goals"], 0)

    def test_invalid_xg_weight_is_rejected(self):
        with self.assertRaises(ValueError):
            FootballModel(xg_weight=1.1)


if __name__ == "__main__":
    unittest.main()
