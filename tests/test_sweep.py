import json
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src import sweep


def synthetic_db(path, n_teams=8, rounds=4, seed=5):
    """Small database with the project's matches / match_features layout."""
    from datetime import date, timedelta

    rng = random.Random(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    strength = {t: rng.uniform(0.7, 1.4) for t in teams}

    def pois(lam):
        limit, k, p = 2.718281828**-lam, 0, 1.0
        while True:
            p *= rng.random()
            if p <= limit:
                return k
            k += 1

    db = sqlite3.connect(path)
    db.executescript("""
    CREATE TABLE matches (provider TEXT, provider_match_id TEXT, league_id TEXT, league_name TEXT, season TEXT,
      match_date TEXT, utc_date TEXT, status TEXT, home_team_id TEXT, home_team TEXT, away_team_id TEXT,
      away_team TEXT, home_goals INTEGER, away_goals INTEGER, is_final INTEGER, raw_json TEXT);
    CREATE TABLE match_features (provider TEXT, provider_match_id TEXT, home_xg REAL, away_xg REAL);
    """)
    day = 0
    for _ in range(rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                day += 1
                d = (date(2025, 8, 1) + timedelta(days=day // 5)).isoformat()
                db.execute(
                    "INSERT INTO matches VALUES ('pitchapi',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        f"m{day}",
                        "l",
                        "Test League",
                        "2025/2026",
                        d,
                        None,
                        "FT",
                        "a",
                        h,
                        "b",
                        a,
                        pois(1.4 * strength[h] / strength[a]),
                        pois(1.1 * strength[a] / strength[h]),
                        1,
                        "{}",
                    ),
                )
                db.execute(
                    "INSERT INTO match_features VALUES ('pitchapi',?,1.3,1.1)",
                    (f"m{day}",),
                )
    db.commit()
    db.close()


def fake_records(
    config_loss,
    n_per_league=300,
    seed=0,
    leagues=("A", "B", "C", "D", "E"),
    seasons=("S1", "S2", "S3"),
):
    """Per-match records whose 1X2 log loss is config_loss(league, season) + noise shared across configs."""
    rng = random.Random(seed)
    records = []
    for league in leagues:
        for i in range(n_per_league):
            season = seasons[i % len(seasons)]
            noise = rng.gauss(0, 0.3)
            base = 1.0 + noise
            records.append(
                {
                    "league": league,
                    "match_id": f"{league}{i}",
                    "season": season,
                    "model_log_loss": base + config_loss(league, season),
                    "baseline_log_loss": 1.08,
                    "model_brier": 0.6,
                    "baseline_brier": 0.65,
                    "model_rps": 0.2,
                    "baseline_rps": 0.23,
                    "correct_model": True,
                    "correct_baseline": False,
                    "exact_hit": False,
                    "score_ll_model": 2.9,
                    "score_ll_baseline": 3.0,
                    "btts_p": 0.5,
                    "btts_base_p": 0.5,
                    "btts_y": True,
                    "over25_p": 0.5,
                    "over25_base_p": 0.5,
                    "over25_y": True,
                }
            )
    return records


def strip_noise_copy(records, delta_fn):
    copies = []
    for r in records:
        c = dict(r)
        c["model_log_loss"] = r["model_log_loss"] + delta_fn(r["league"], r["season"])
        copies.append(c)
    return copies


class AnalyzeTests(unittest.TestCase):
    def setUp(self):
        self.ref_config = (365.0, 0.75, 5.0)
        self.reference = fake_records(lambda l, s: 0.0)

    def verdict(self, delta_fn, min_gain=0.002):
        results = {
            self.ref_config: self.reference,
            (180.0, 0.75, 5.0): strip_noise_copy(self.reference, delta_fn),
            (730.0, 0.75, 5.0): strip_noise_copy(self.reference, lambda l, s: 0.0),
        }
        rows, z = sweep.analyze(results, self.ref_config, min_gain=min_gain)
        return {row["config"]: row for row in rows}, z

    def test_consistent_real_gain_is_marked_better(self):
        rows, z = self.verdict(lambda l, s: -0.01)
        self.assertEqual(rows[(180.0, 0.75, 5.0)]["verdict"], "BETTER")
        self.assertEqual(rows[self.ref_config]["verdict"], "reference")
        self.assertGreater(z, 1.96)  # widened for multiple comparisons

    def test_gain_below_minimum_is_not_better(self):
        rows, _ = self.verdict(lambda l, s: -0.0005)
        self.assertNotEqual(rows[(180.0, 0.75, 5.0)]["verdict"], "BETTER")

    def test_gain_in_only_one_league_is_not_better(self):
        rows, _ = self.verdict(lambda l, s: -0.05 if l == "A" else 0.002)
        self.assertNotEqual(rows[(180.0, 0.75, 5.0)]["verdict"], "BETTER")

    def test_gain_in_only_one_season_is_not_better(self):
        rows, _ = self.verdict(lambda l, s: -0.05 if s == "S1" else 0.004)
        self.assertNotEqual(rows[(180.0, 0.75, 5.0)]["verdict"], "BETTER")

    def test_clear_loss_is_marked_worse(self):
        rows, _ = self.verdict(lambda l, s: 0.02)
        self.assertEqual(rows[(180.0, 0.75, 5.0)]["verdict"], "worse")

    def test_identical_config_has_no_clear_difference(self):
        rows, _ = self.verdict(lambda l, s: -0.01)
        self.assertEqual(rows[(730.0, 0.75, 5.0)]["verdict"], "no clear difference")

    def test_noise_only_differences_are_not_discoveries(self):
        rng = random.Random(3)
        results = {self.ref_config: self.reference}
        for i in range(26):  # many random configs: some will look good by chance
            noise = {}
            results[(100.0 + i, 0.5, 5.0)] = [
                dict(r, model_log_loss=r["model_log_loss"] + rng.gauss(0, 0.02))
                for r in self.reference
            ]
        rows, _ = sweep.analyze(results, self.ref_config)
        self.assertEqual([r["label"] for r in rows if r["verdict"] == "BETTER"], [])

    def test_extra_comparisons_widen_the_interval(self):
        results = {
            self.ref_config: self.reference,
            (180.0, 0.75, 5.0): strip_noise_copy(self.reference, lambda l, s: -0.01),
        }
        _, z_plain = sweep.analyze(results, self.ref_config)
        rows, z_wide = sweep.analyze(results, self.ref_config, extra_comparisons=26)
        self.assertGreater(z_wide, z_plain)
        narrow = {r["config"]: r for r in sweep.analyze(results, self.ref_config)[0]}
        wide = {r["config"]: r for r in rows}
        self.assertGreater(
            wide[(180.0, 0.75, 5.0)]["adjusted_interval"],
            narrow[(180.0, 0.75, 5.0)]["adjusted_interval"],
        )

    def test_sensitivity_groups_by_parameter(self):
        results = {
            self.ref_config: self.reference,
            (180.0, 0.75, 5.0): strip_noise_copy(self.reference, lambda l, s: -0.01),
        }
        rows, _ = sweep.analyze(results, self.ref_config)
        table = sweep.sensitivity(rows)
        self.assertEqual(set(table), {"half_life_days", "xg_weight", "prior_matches"})
        self.assertLess(table["half_life_days"][180.0], table["half_life_days"][365.0])


class SweepEndToEndTests(unittest.TestCase):
    def args(self, db_path, out_dir, **overrides):
        base = [
            "--db",
            str(db_path),
            "--half-lives",
            "180,365",
            "--xg-weights",
            "0.5,1.0",
            "--priors",
            "5",
            "--reference",
            "365,0.5,5",
            "--report-from",
            "2025-01-01",
            "--report-to",
            "2026-07-31",
            "--workers",
            "1",
            "--out-dir",
            str(out_dir),
            "--min-training-matches",
            "60",
        ]
        for k, v in overrides.items():
            base += [k, v]
        return sweep.build_parser().parse_args(base)

    def test_runs_and_writes_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.db"
            synthetic_db(db)
            rows = sweep.run_sweep(self.args(db, tmp))
            self.assertEqual(len(rows), 4)
            outputs = {p.suffix for p in Path(tmp).glob("sweep_*")}
            self.assertEqual(outputs, {".txt", ".json"})
            saved = json.loads(next(Path(tmp).glob("sweep_*.json")).read_text())
            self.assertEqual(saved["reference"], "hl365_xg0.5_pm5")
            self.assertEqual(len(saved["rows"]), 4)

    def test_holdout_is_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.db"
            synthetic_db(db)
            with self.assertRaises(SystemExit) as ctx:
                sweep.run_sweep(self.args(db, tmp, **{"--report-to": "2026-09-30"}))
            self.assertIn("holdout", str(ctx.exception))

    def test_reference_added_to_grid_if_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.db"
            synthetic_db(db)
            rows = sweep.run_sweep(self.args(db, tmp, **{"--reference": "400,0.75,5"}))
            self.assertIn((400.0, 0.75, 5.0), [r["config"] for r in rows])


if __name__ == "__main__":
    unittest.main()
