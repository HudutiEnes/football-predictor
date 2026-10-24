"""Tune recency and xG blend on an earlier validation period, then compare on holdout."""

import argparse
import sqlite3
from collections import Counter, defaultdict
from math import log
from pathlib import Path

from src.model import DEFAULT_HALF_LIFE_DAYS, FootballModel

DB_PATH = Path("data/football.db")
MIN_TRAINING_MATCHES = 100
HALF_LIFE_CANDIDATES = [90, 180, 270, 365, 540, 730]
XG_WEIGHT_CANDIDATES = [0.0, 0.25, 0.5, 0.75, 1.0]
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
                m.match_date,
                m.season,
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


def score(probabilities, actual):
    brier = sum(
        (probabilities[outcome] - float(outcome == actual)) ** 2
        for outcome in OUTCOMES
    )
    logloss = -log(max(probabilities[actual], 1e-15))
    correct = max(probabilities, key=probabilities.get) == actual
    return correct, brier, logloss


def evaluate(matches, target_dates, half_life, xg_weight, min_training_matches):
    groups = defaultdict(list)
    for match in matches:
        groups[match["match_date"][:10]].append(match)

    previous = []
    totals = {
        "correct": 0,
        "brier": 0.0,
        "logloss": 0.0,
        "baseline_correct": 0,
        "baseline_brier": 0.0,
        "baseline_logloss": 0.0,
        "count": 0,
    }

    for match_date in sorted(groups):
        day_matches = groups[match_date]
        if match_date in target_dates and len(previous) >= min_training_matches:
            model = FootballModel(
                half_life_days=half_life,
                xg_weight=xg_weight,
            ).fit(previous)
            counts = Counter(actual_outcome(match) for match in previous)
            total = sum(counts.values())
            baseline = {outcome: counts[outcome] / total for outcome in OUTCOMES}

            for match in day_matches:
                prediction = model.predict(match["home_team"], match["away_team"])
                model_probs = {
                    "home": prediction["home_win_probability"],
                    "draw": prediction["draw_probability"],
                    "away": prediction["away_win_probability"],
                }
                actual = actual_outcome(match)
                correct, brier, logloss = score(model_probs, actual)
                base_correct, base_brier, base_logloss = score(baseline, actual)
                totals["correct"] += correct
                totals["brier"] += brier
                totals["logloss"] += logloss
                totals["baseline_correct"] += base_correct
                totals["baseline_brier"] += base_brier
                totals["baseline_logloss"] += base_logloss
                totals["count"] += 1

        # Same-date results are added only after all target predictions for that date.
        previous.extend(day_matches)

    count = totals["count"]
    if not count:
        return None
    return {
        **{key: value / count for key, value in totals.items() if key != "count"},
        "count": count,
    }


def show_result(label, result):
    print(
        f"{label:<34} accuracy {result['correct']:.1%} | "
        f"Brier {result['brier']:.4f} | log loss {result['logloss']:.4f} | "
        f"n={result['count']}"
    )
    print(
        f"{'  outcome-frequency baseline':<34} accuracy {result['baseline_correct']:.1%} | "
        f"Brier {result['baseline_brier']:.4f} | "
        f"log loss {result['baseline_logloss']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--league", default="Premier League", help="Exact league_name in the database")
    parser.add_argument("--min-training-matches", type=int, default=MIN_TRAINING_MATCHES)
    parser.add_argument("--validation-fraction", type=float, default=0.70,
                        help="Fraction of calendar match dates available for tuning")
    parser.add_argument("--half-lives", type=str,
                        default=",".join(map(str, HALF_LIFE_CANDIDATES)),
                        help="Comma-separated half-life candidates in days")
    parser.add_argument("--xg-weights", type=str,
                        default=",".join(map(str, XG_WEIGHT_CANDIDATES)),
                        help="Comma-separated xG blend candidates from 0 to 1")
    args = parser.parse_args()

    if not 0.5 <= args.validation_fraction <= 0.9:
        raise SystemExit("--validation-fraction must be between 0.5 and 0.9")
    try:
        half_lives = [int(value.strip()) for value in args.half_lives.split(",")]
        xg_weights = [float(value.strip()) for value in args.xg_weights.split(",")]
    except ValueError as exc:
        raise SystemExit(f"Invalid tuning candidate: {exc}") from exc
    if not half_lives or any(value <= 0 for value in half_lives):
        raise SystemExit("All half-life candidates must be positive integers")
    if not xg_weights or any(not 0 <= value <= 1 for value in xg_weights):
        raise SystemExit("All xG weights must be between 0 and 1")

    matches = load_matches(args.league)
    if len(matches) <= args.min_training_matches:
        raise SystemExit(f"Not enough completed matches for {args.league}: {len(matches)}")

    groups = defaultdict(list)
    for match in matches:
        groups[match["match_date"][:10]].append(match)
    dates = sorted(groups)
    split = int(len(dates) * args.validation_fraction)
    validation_dates = set(dates[:split])
    holdout_dates = set(dates[split:])
    xg_count = sum(
        match.get("home_xg") is not None and match.get("away_xg") is not None
        for match in matches
    )

    print(f"League: {args.league}")
    print(f"Completed matches: {len(matches)}; matches with both xG values: {xg_count}")
    print(f"Validation dates: {len(validation_dates)}; final-period dates: {len(holdout_dates)}")
    print("\nTuning on the earlier validation period (lower Brier is better):")

    results = []
    for half_life in half_lives:
        for xg_weight in xg_weights:
            result = evaluate(
                matches, validation_dates, half_life, xg_weight,
                args.min_training_matches,
            )
            if result:
                results.append((half_life, xg_weight, result))

    results.sort(key=lambda item: (item[2]["brier"], item[2]["logloss"]))
    if not results:
        raise SystemExit("No validation predictions could be evaluated.")

    print("\nBest validation configurations:")
    for half_life, xg_weight, result in results[:10]:
        show_result(f"half-life={half_life}d xG={xg_weight:.2f}", result)

    best_half_life, best_xg_weight, best_validation = results[0]
    print(
        f"\nSelected on validation: half-life={best_half_life} days, "
        f"xG weight={best_xg_weight:.2f}, Brier={best_validation['brier']:.4f}, "
        f"log loss={best_validation['logloss']:.4f}"
    )

    print("\nFinal-period comparison (do not use these results to retune):")
    holdout_variants = [
        ("goals-only control", best_half_life, 0.0),
        ("selected xG model", best_half_life, best_xg_weight),
    ]
    # Avoid printing the same model twice when validation selects xG weight zero.
    seen = set()
    for label, half_life, xg_weight in holdout_variants:
        key = (half_life, xg_weight)
        if key in seen:
            continue
        seen.add(key)
        result = evaluate(
            matches, holdout_dates, half_life, xg_weight,
            args.min_training_matches,
        )
        if result:
            show_result(f"{label} (half-life={half_life}d, xG={xg_weight:.2f})", result)

    print("\nUse the validation-selected settings only if the final-period comparison supports them.")
    print("The final-period results are a comparison, not a guarantee of future performance.")


if __name__ == "__main__":
    main()
