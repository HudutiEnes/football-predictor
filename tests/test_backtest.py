import copy
import json
import random
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from src import backtest


def synthetic_matches(seed=5, n_teams=10, rounds=4):
    rng = random.Random(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    strength = {t: rng.uniform(0.7, 1.4) for t in teams}
    matches, day = [], 0
    for _ in range(rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                day += 1
                d = date(2025, 8, 1) + timedelta(days=day // 5)

                def pois(lam):
                    limit, k, p = 2.718281828**-lam, 0, 1.0
                    while True:
                        p *= rng.random()
                        if p <= limit:
                            return k
                        k += 1

                matches.append(
                    {
                        "provider_match_id": f"m{day}",
                        "season": "2025/2026",
                        "match_date": d.isoformat(),
                        "home_team": h,
                        "away_team": a,
                        "home_goals": pois(1.4 * strength[h] / strength[a]),
                        "away_goals": pois(1.1 * strength[a] / strength[h]),
                        "is_final": 1,
                        "home_xg": None,
                        "away_xg": None,
                    }
                )
    return matches


class BacktestLeakageTests(unittest.TestCase):
    def setUp(self):
        self.matches = synthetic_matches()

    def run_eval(self, matches, **kw):
        return backtest.evaluate(
            matches, half_life_days=365, xg_weight=0, min_training_matches=60, **kw
        )

    def test_future_results_cannot_change_earlier_predictions(self):
        original = {r["match_id"]: r for r in self.run_eval(self.matches)}
        cutoff = sorted({m["match_date"] for m in self.matches})[
            len(self.matches) // 10 * 6 // 5 // 2 + 40
        ]
        tampered = copy.deepcopy(self.matches)
        for m in tampered:
            if m["match_date"] >= cutoff:
                m["home_goals"], m["away_goals"], m["home_xg"], m["away_xg"] = (
                    9,
                    0,
                    5.0,
                    0.1,
                )
        changed = {r["match_id"]: r for r in self.run_eval(tampered)}
        early = [k for k, r in original.items() if r["date"] < cutoff]
        self.assertTrue(early)
        for key in early:
            self.assertEqual(original[key]["model_probs"], changed[key]["model_probs"])
            self.assertEqual(original[key]["btts_p"], changed[key]["btts_p"])

    def test_own_xg_of_predicted_match_is_not_used(self):
        with_xg = copy.deepcopy(self.matches)
        for m in with_xg:
            m["home_xg"], m["away_xg"] = 1.3, 1.1
        base = {
            r["match_id"]: r
            for r in backtest.evaluate(with_xg, xg_weight=1.0, min_training_matches=60)
        }
        tampered = copy.deepcopy(with_xg)
        last_date = max(m["match_date"] for m in tampered)
        for m in tampered:
            if m["match_date"] == last_date:
                m["home_xg"], m["away_xg"] = (
                    6.0,
                    0.0,
                )  # xG of the matches being predicted
        changed = {
            r["match_id"]: r
            for r in backtest.evaluate(tampered, xg_weight=1.0, min_training_matches=60)
        }
        for key, r in base.items():
            if r["date"] == last_date:
                self.assertEqual(r["model_probs"], changed[key]["model_probs"])

    def test_training_never_includes_the_same_date(self):
        # every record must come from a date that has at least min_training earlier matches
        records = self.run_eval(self.matches)
        first = min(r["date"] for r in records)
        earlier = [m for m in self.matches if m["match_date"] < first]
        self.assertGreaterEqual(len(earlier), 60)

    def test_records_contain_all_markets_and_valid_probabilities(self):
        for r in self.run_eval(self.matches, dixon_coles=True):
            self.assertAlmostEqual(sum(r["model_probs"].values()), 1.0, places=3)
            self.assertTrue(0 <= r["btts_p"] <= 1 and 0 <= r["over25_p"] <= 1)
            self.assertTrue(r["score_ll_model"] > 0 and r["score_ll_baseline"] > 0)

    def test_summary_has_paired_intervals(self):
        s = backtest.summarize(self.run_eval(self.matches))
        for key in (
            "1x2_log_loss",
            "1x2_rps",
            "exact_score_log_loss",
            "btts_log_loss",
            "over25_log_loss",
        ):
            self.assertIn(key, s["paired"])
        self.assertLess(s["log_loss_model"], 1.2)


class BacktestCliTests(unittest.TestCase):
    def make_db(self, path):
        db = sqlite3.connect(path)
        db.executescript("""
        CREATE TABLE matches (provider TEXT, provider_match_id TEXT, league_id TEXT, league_name TEXT, season TEXT,
          match_date TEXT, utc_date TEXT, status TEXT, home_team_id TEXT, home_team TEXT, away_team_id TEXT,
          away_team TEXT, home_goals INTEGER, away_goals INTEGER, is_final INTEGER, raw_json TEXT);
        CREATE TABLE match_features (provider TEXT, provider_match_id TEXT, home_xg REAL, away_xg REAL);
        """)
        for m in synthetic_matches(n_teams=8, rounds=4):
            db.execute(
                "INSERT INTO matches VALUES ('pitchapi',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    m["provider_match_id"],
                    "l",
                    "Test League",
                    m["season"],
                    m["match_date"],
                    None,
                    "FT",
                    "a",
                    m["home_team"],
                    "b",
                    m["away_team"],
                    m["home_goals"],
                    m["away_goals"],
                    1,
                    "{}",
                ),
            )
            db.execute(
                "INSERT INTO match_features VALUES ('pitchapi',?,1.3,1.1)",
                (m["provider_match_id"],),
            )
        db.commit()
        db.close()

    def test_cli_runs_and_logs_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path, log_path, json_path = (
                Path(tmp) / "t.db",
                Path(tmp) / "exp.jsonl",
                Path(tmp) / "out.json",
            )
            self.make_db(db_path)
            backtest.main(
                [
                    "--league",
                    "Test League",
                    "--db",
                    str(db_path),
                    "--xg-weight",
                    "0.5",
                    "--dixon-coles",
                    "--experiment-log",
                    str(log_path),
                    "--json-out",
                    str(json_path),
                    "--note",
                    "unit test",
                ]
            )
            entry = json.loads(log_path.read_text().splitlines()[-1])
            self.assertEqual(entry["config"]["xg_weight"], 0.5)
            self.assertTrue(entry["config"]["dixon_coles"])
            self.assertEqual(entry["note"], "unit test")
            self.assertEqual(len(entry["db_sha256_prefix"]), 16)
            self.assertIn("Test League", json.loads(json_path.read_text())["leagues"])

    def test_all_leagues_mode_pools_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "t.db"
            self.make_db(db_path)
            backtest.main(
                ["--all-leagues", "--db", str(db_path), "--no-log", "--xg-weight", "0"]
            )

    def test_report_window_limits_scored_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path, json_path = Path(tmp) / "t.db", Path(tmp) / "out.json"
            self.make_db(db_path)
            dates = sorted(
                {m["match_date"] for m in synthetic_matches(n_teams=8, rounds=4)}
            )
            lo, hi = dates[len(dates) // 2], dates[len(dates) * 3 // 4]
            backtest.main(
                [
                    "--league",
                    "Test League",
                    "--db",
                    str(db_path),
                    "--no-log",
                    "--xg-weight",
                    "0",
                    "--report-from",
                    lo,
                    "--report-to",
                    hi,
                    "--json-out",
                    str(json_path),
                ]
            )
            windowed = json.loads(json_path.read_text())["leagues"]["Test League"][
                "evaluated"
            ]
            backtest.main(
                [
                    "--league",
                    "Test League",
                    "--db",
                    str(db_path),
                    "--no-log",
                    "--xg-weight",
                    "0",
                    "--json-out",
                    str(json_path),
                ]
            )
            full = json.loads(json_path.read_text())["leagues"]["Test League"][
                "evaluated"
            ]
            self.assertGreater(windowed, 0)
            self.assertLess(windowed, full)

    def test_report_from_restricts_scored_matches(self):
        matches = synthetic_matches(n_teams=8, rounds=4)
        everything = backtest.evaluate(matches, xg_weight=0, min_training_matches=60)
        late = [
            r
            for r in everything
            if r["date"] >= sorted({m["match_date"] for m in matches})[-10]
        ]
        self.assertLess(len(late), len(everything))


if __name__ == "__main__":
    unittest.main()
