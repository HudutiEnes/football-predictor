"""Chronological, date-grouped backtest for the football model."""

import sqlite3
from collections import Counter, defaultdict
from math import log
from pathlib import Path

from src.model import FootballModel

DB_PATH = Path("data/football.db")
LEAGUE = "Premier League"
MIN_TRAINING_MATCHES = 100


def load_matches():
    if not DB_PATH.exists():
        raise SystemExit(f"Database not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """
            SELECT
                provider_match_id,
                season,
                match_date,
                home_team,
                away_team,
                home_goals,
                away_goals,
                is_final
            FROM matches
            WHERE provider = 'pitchapi'
              AND league_name = ?
              AND is_final = 1
              AND home_goals IS NOT NULL
              AND away_goals IS NOT NULL
              AND match_date IS NOT NULL
              AND season != '2021/2022'
            ORDER BY match_date, provider_match_id
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


def prediction_probabilities(prediction):
    return {
        "home": prediction["home_win_probability"],
        "draw": prediction["draw_probability"],
        "away": prediction["away_win_probability"],
    }


def brier_score(probabilities, actual):
    return sum(
        (probabilities[outcome] - float(outcome == actual)) ** 2
        for outcome in ("home", "draw", "away")
    )


def log_loss(probabilities, actual):
    # A confident prediction that is wrong gets penalized heavily.
    return -log(max(probabilities[actual], 1e-15))


def main():
    matches = load_matches()

    if not matches:
        raise SystemExit("No completed matches found in the database.")

    season_counts = Counter(match["season"] or "Unknown" for match in matches)

    print(f"League: {LEAGUE}")
    print("Completed matches included by season:")

    for season, count in sorted(season_counts.items()):
        print(f"  {season}: {count}")

    print(f"Total matches available: {len(matches)}")

    # All matches on a date are tested against results from earlier dates
    # only. No match can train the prediction of another match on that date.
    matches_by_date = defaultdict(list)

    for match in matches:
        matches_by_date[match["match_date"][:10]].append(match)

    previous_matches = []

    correct_model = 0
    correct_baseline = 0
    exact_scores = 0
    model_brier_total = 0.0
    baseline_brier_total = 0.0
    model_log_loss_total = 0.0
    baseline_log_loss_total = 0.0
    evaluated = 0

    for match_date in sorted(matches_by_date):
        date_matches = matches_by_date[match_date]

        # Only use dates strictly before the date being evaluated.
        if len(previous_matches) >= MIN_TRAINING_MATCHES:
            model = FootballModel().fit(previous_matches)

            outcome_counts = Counter(
                actual_outcome(match) for match in previous_matches
            )
            prior_total = sum(outcome_counts.values())

            baseline_probs = {
                outcome: outcome_counts[outcome] / prior_total
                for outcome in ("home", "draw", "away")
            }
            baseline_choice = max(baseline_probs, key=baseline_probs.get)

            for match in date_matches:
                prediction = model.predict(
                    match["home_team"],
                    match["away_team"],
                )

                actual = actual_outcome(match)
                model_probs = prediction_probabilities(prediction)
                model_choice = max(model_probs, key=model_probs.get)

                correct_model += model_choice == actual
                correct_baseline += baseline_choice == actual

                expected_score = f'{match["home_goals"]}-{match["away_goals"]}'
                exact_scores += prediction["most_likely_score"] == expected_score

                model_brier_total += brier_score(model_probs, actual)
                baseline_brier_total += brier_score(baseline_probs, actual)
                model_log_loss_total += log_loss(model_probs, actual)
                baseline_log_loss_total += log_loss(baseline_probs, actual)
                evaluated += 1

        # Add that day's results only after evaluating that date.
        previous_matches.extend(date_matches)

    if evaluated == 0:
        raise SystemExit(
            f"Not enough data to evaluate. Need at least "
            f"{MIN_TRAINING_MATCHES} prior matches."
        )

    print()
    print(f"Initial training window: {MIN_TRAINING_MATCHES}")
    print(f"Matches evaluated: {evaluated}")
    print()
    print(f"Model 1X2 accuracy:     {correct_model / evaluated:.1%}")
    print(f"Baseline 1X2 accuracy:  {correct_baseline / evaluated:.1%}")
    print(f"Model Brier score:      {model_brier_total / evaluated:.4f}")
    print(f"Baseline Brier score:   {baseline_brier_total / evaluated:.4f}")
    print(f"Model log loss:         {model_log_loss_total / evaluated:.4f}")
    print(f"Baseline log loss:      {baseline_log_loss_total / evaluated:.4f}")
    print(f"Exact score accuracy:   {exact_scores / evaluated:.1%}")


if __name__ == "__main__":
    main()
