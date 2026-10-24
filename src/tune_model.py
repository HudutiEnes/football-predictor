"""Tune recency weighting without selecting parameters on the final holdout."""

import sqlite3
from collections import Counter, defaultdict
from math import log
from pathlib import Path

from src.model import FootballModel

DB_PATH = Path("data/football.db")
LEAGUE = "Premier League"
MIN_TRAINING_MATCHES = 100
CANDIDATES = [60, 90, 120, 180, 270, 365]
OUTCOMES = ("home", "draw", "away")


def load_matches():
    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """
            SELECT match_date, season, home_team, away_team,
                   home_goals, away_goals, is_final
            FROM matches
            WHERE provider = 'pitchapi'
              AND league_name = ?
              AND is_final = 1
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
              AND match_date IS NOT NULL
              AND season != '2021/2022'
            ORDER BY match_date
        """,
            (LEAGUE,),
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
        (probabilities[outcome] - float(outcome == actual)) ** 2 for outcome in OUTCOMES
    )
    logloss = -log(max(probabilities[actual], 1e-15))
    correct = max(probabilities, key=probabilities.get) == actual
    return correct, brier, logloss


def evaluate(matches, target_dates, half_life):
    groups = defaultdict(list)

    for match in matches:
        groups[match["match_date"][:10]].append(match)

    previous = []
    totals = {"correct": 0, "brier": 0.0, "logloss": 0.0, "count": 0}

    for match_date in sorted(groups):
        day_matches = groups[match_date]

        # Test all matches on a date before adding any results from that date.
        if match_date in target_dates and len(previous) >= MIN_TRAINING_MATCHES:
            model = FootballModel(half_life_days=half_life).fit(previous)

            counts = Counter(actual_outcome(m) for m in previous)
            baseline = {
                outcome: counts[outcome] / len(previous) for outcome in OUTCOMES
            }

            for match in day_matches:
                prediction = model.predict(match["home_team"], match["away_team"])
                model_probs = {
                    "home": prediction["home_win_probability"],
                    "draw": prediction["draw_probability"],
                    "away": prediction["away_win_probability"],
                }

                actual = actual_outcome(match)
                correct, brier, logloss = score(model_probs, actual)
                totals["correct"] += correct
                totals["brier"] += brier
                totals["logloss"] += logloss
                totals["count"] += 1

                # Only needed for the baseline comparison on the final holdout.
                if half_life == 180:
                    _, baseline_brier, baseline_logloss = score(baseline, actual)
                    totals.setdefault("baseline_correct", 0)
                    totals.setdefault("baseline_brier", 0.0)
                    totals.setdefault("baseline_logloss", 0.0)
                    totals["baseline_correct"] += (
                        max(baseline, key=baseline.get) == actual
                    )
                    totals["baseline_brier"] += baseline_brier
                    totals["baseline_logloss"] += baseline_logloss

        previous.extend(day_matches)

    count = totals["count"]
    if not count:
        return None

    return {
        "accuracy": totals["correct"] / count,
        "brier": totals["brier"] / count,
        "logloss": totals["logloss"] / count,
        "count": count,
        **{
            key: value / count
            for key, value in totals.items()
            if key.startswith("baseline_")
        },
    }


def main():
    matches = load_matches()
    if len(matches) <= MIN_TRAINING_MATCHES:
        raise SystemExit(f"Not enough completed matches: {len(matches)}")

    groups = defaultdict(list)
    for match in matches:
        groups[match["match_date"][:10]].append(match)

    dates = sorted(groups)

    # Reserve the last 30% of calendar match dates as an untouched holdout.
    split = int(len(dates) * 0.70)
    validation_dates = set(dates[:split])
    holdout_dates = set(dates[split:])

    print(f"Completed matches: {len(matches)}")
    print(f"Validation dates: {len(validation_dates)}")
    print(f"Final holdout dates: {len(holdout_dates)}")
    print("\nTuning on the earlier validation period:")

    results = []

    for half_life in CANDIDATES:
        result = evaluate(matches, validation_dates, half_life)
        if result:
            results.append((half_life, result))
            print(
                f"Half-life {half_life:>3} days | "
                f"accuracy {result['accuracy']:.1%} | "
                f"Brier {result['brier']:.4f} | "
                f"log loss {result['logloss']:.4f}"
            )

    if not results:
        raise SystemExit("No validation predictions could be evaluated.")

    # Brier score is our primary selection metric; lower is better.
    best_half_life, best_validation = min(results, key=lambda item: item[1]["brier"])

    print(
        f"\nSelected half-life: {best_half_life} days "
        f"(validation Brier {best_validation['brier']:.4f})"
    )
    print("\nFinal holdout comparison:")

    for half_life in sorted({best_half_life, 180}):
        result = evaluate(matches, holdout_dates, half_life)
        if not result:
            continue

        print(
            f"Half-life {half_life:>3} days | "
            f"accuracy {result['accuracy']:.1%} | "
            f"Brier {result['brier']:.4f} | "
            f"log loss {result['logloss']:.4f} | "
            f"matches {result['count']}"
        )

        if half_life == 180 and "baseline_brier" in result:
            print(
                f"  Simple baseline | "
                f"accuracy {result['baseline_correct']:.1%} | "
                f"Brier {result['baseline_brier']:.4f} | "
                f"log loss {result['baseline_logloss']:.4f}"
            )


if __name__ == "__main__":
    main()
