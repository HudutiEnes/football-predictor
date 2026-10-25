import math
import unittest

from src import metrics


class MetricTests(unittest.TestCase):
    def test_rps_perfect_and_worst(self):
        self.assertAlmostEqual(
            metrics.ranked_probability_score({"home": 1, "draw": 0, "away": 0}, "home"),
            0.0,
        )
        self.assertAlmostEqual(
            metrics.ranked_probability_score({"home": 1, "draw": 0, "away": 0}, "away"),
            1.0,
        )

    def test_rps_penalises_distance_between_outcomes(self):
        wrong_by_one = metrics.ranked_probability_score(
            {"home": 0, "draw": 1, "away": 0}, "home"
        )
        wrong_by_two = metrics.ranked_probability_score(
            {"home": 0, "draw": 0, "away": 1}, "home"
        )
        self.assertLess(wrong_by_one, wrong_by_two)

    def test_rps_known_value(self):
        # p = (0.5, 0.3, 0.2), home wins: 0.5 * ((0.5-1)^2 + (0.8-1)^2) = 0.145
        self.assertAlmostEqual(
            metrics.ranked_probability_score(
                {"home": 0.5, "draw": 0.3, "away": 0.2}, "home"
            ),
            0.145,
        )

    def test_binary_scores(self):
        self.assertAlmostEqual(metrics.binary_brier(0.8, True), 0.04)
        self.assertAlmostEqual(metrics.binary_log_loss(0.8, False), -math.log(0.2))
        self.assertTrue(math.isfinite(metrics.binary_log_loss(1.0, False)))

    def test_paired_difference(self):
        d = metrics.paired_difference([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(d["mean_diff"], -1.0)
        self.assertEqual(d["ci95"], 0.0)  # identical differences -> no spread

    def test_paired_difference_small_sample_is_nan_interval(self):
        self.assertTrue(math.isnan(metrics.paired_difference([1.0], [2.0])["ci95"]))

    def test_binary_calibration_counts(self):
        table = metrics.binary_calibration(
            [(0.1, False), (0.15, False), (0.9, True), (1.0, True)], bins=5
        )
        self.assertEqual(sum(row["samples"] for row in table), 4)


if __name__ == "__main__":
    unittest.main()
