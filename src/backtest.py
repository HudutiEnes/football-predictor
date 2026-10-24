"""Chronological, date-grouped backtest with an optional xG blend."""

import argparse
import sqlite3
from collections import Counter, defaultdict
from math import log
from pathlib import Path

from src.model import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_XG_WEIGHT,
    FootballModel,
    brier_score,
    calibration_buckets,
    log_loss,
)

DB_PATH = Path("data/football.db")
MIN_TRAINING_MATCHES = 100
OUTCOMES = ("home", "draw", "away")


def load_matches(league):
    if not DB_PATH.exists():
        raise SystemExit(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """
            SELECT
                m.provider_match_id,
                m.season,
                m.match_date,
                m.home_team,
                m.away_team,
                m.home_goals,
                m.away_goals,
                m.is_final,
                f.home_xg,
                f.away_xg
            FROM matches AS m
            LEFT JOIN match_features AS f
              ON f.provider = m.provider
             AND f.provider_match_id = m.provider_match_id
            WHERE m.provider = 'pitchapi'
              AND m.league_name = ?
              AND m.is_final = 1
              AND m.home_goals IS NOT NULL
              AND m.away_goals IS NOT NULL
              AND m.match_date IS NOT NULL
              AND m.season != '2021/2022'
            ORDER BY m.match_date, m.provider_match_id
            """,
            (league,),
        ).fetchall()

    return [dict(row) for row in rows]


def actual_outcome(match):
    if match["home_goals"] > match["away_goals"]:
        return "home"
    if match["home_goals"] == match["away_goals"]:
        return "draw"
    return "away"


def prediction_probabilities(prediction):
    return {
        "home": prediction["home_win_probability"],
        "draw": prediction["draw_probability"],
        "away": prediction["away_win_probability"],
    }




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="Premier League", help="Exact league_name in the database")
    parser.add_argument("--half-life-days", type=float, default=DEFAULT_HALF_LIFE_DAYS)
    parser.add_argument("--xg-weight", type=float, default=DEFAULT_XG_WEIGHT,
                        help="Blend weight from 0 (goals only) to 1 (xG where available)")
    parser.add_argument("--min-training-matches", type=int, default=MIN_TRAINING_MATCHES)
    args = parser.parse_args()

    matches = load_matches(args.league)
    if not matches:
        raise SystemExit(
            f"No completed matches found for league {args.league!r}. "
            "Check the exact league_name stored in the database."
        )

    season_counts = Counter(match["season"] or "Unknown" for match in matches)
    print(f"League: {args.league}")
    print("Completed matches included by season:")
    for season, count in sorted(season_counts.items()):
        print(f"  {season}: {count}")
    xg_count = sum(
        match.get("home_xg") is not None and match.get("away_xg") is not None
        for match in matches
    )
    print(f"Total matches available: {len(matches)}")
    print(f"Matches with both xG values: {xg_count}/{len(matches)}")

    matches_by_date = defaultdict(list)
    for match in matches:
        matches_by_date[match["match_date"][:10]].append(match)

    previous_matches = []
    totals = {
        "correct_model": 0,
        "correct_baseline": 0,
        "exact_scores": 0,
        "model_brier": 0.0,
        "baseline_brier": 0.0,
        "model_log_loss": 0.0,
        "baseline_log_loss": 0.0,
        "evaluated": 0,
    }

    for match_date in sorted(matches_by_date):
        date_matches = matches_by_date[match_date]
        # Every match on the date is predicted before that date's results enter training.
        if len(previous_matches) >= args.min_training_matches:
            model = FootballModel(
                half_life_days=args.half_life_days,
                xg_weight=args.xg_weight,
            ).fit(previous_matches)
            outcome_counts = Counter(actual_outcome(match) for match in previous_matches)
            prior_total = sum(outcome_counts.values())
            baseline_probs = {
                outcome: outcome_counts[outcome] / prior_total for outcome in OUTCOMES
            }
            baseline_choice = max(baseline_probs, key=baseline_probs.get)

            for match in date_matches:
                prediction = model.predict(match["home_team"], match["away_team"])
                actual = actual_outcome(match)
                model_probs = prediction_probabilities(prediction)
                model_choice = max(model_probs, key=model_probs.get)

                totals["correct_model"] += model_choice == actual
                totals["correct_baseline"] += baseline_choice == actual
                totals["exact_scores"] += (
                    prediction["most_likely_score"]
                    == f'{match["home_goals"]}-{match["away_goals"]}'
                )
                totals["model_brier"] += brier_score(model_probs, actual)
                totals["baseline_brier"] += brier_score(baseline_probs, actual)
                totals["model_log_loss"] += log_loss(model_probs, actual)
                totals["baseline_log_loss"] += log_loss(baseline_probs, actual)

                # Collect per-prediction records for calibration analysis.
                if "calibration_records" not in totals:
                    totals["calibration_records"] = []
                totals["calibration_records"].append((model_probs, actual))
                totals["evaluated"] += 1

        previous_matches.extend(date_matches)

    evaluated = totals["evaluated"]
    if not evaluated:
        raise SystemExit(
            f"Not enough data to evaluate; need at least {args.min_training_matches} prior matches."
        )

    print()
    print(f"Model settings: half-life={args.half_life_days:g} days, xG weight={args.xg_weight:g}")
    print(f"Initial training window: {args.min_training_matches}")
    print(f"Matches evaluated: {evaluated}")
    print()
    print(f"Model 1X2 accuracy:     {totals['correct_model'] / evaluated:.1%}")
    print(f"Baseline 1X2 accuracy:  {totals['correct_baseline'] / evaluated:.1%}")
    print(f"Model Brier score:      {totals['model_brier'] / evaluated:.4f}")
    print(f"Baseline Brier score:   {totals['baseline_brier'] / evaluated:.4f}")
    print(f"Model log loss:         {totals['model_log_loss'] / evaluated:.4f}")
    print(f"Baseline log loss:      {totals['baseline_log_loss'] / evaluated:.4f}")
    print(f"Exact score accuracy:   {totals['exact_scores'] / evaluated:.1%}")
    print()
    print("Model calibration (predicted probability bucket vs observed rate):")
    print(f"{'Bucket':>10} {'Predicted':>10} {'Observed':>10} {'Samples':>8}")
    for bucket in calibration_buckets(totals["calibration_records"]):
        print(
            f"{bucket['range']:>10} "
            f"{bucket['predicted_rate']:>9.1%} "
            f"{bucket['observed_rate']:>9.1%} "
            f"{bucket['samples']:>8}"
        )


if __name__ == "__main__":
    main()
