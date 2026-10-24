"""Train from stored results and predict upcoming Premier League fixtures."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.model import FootballModel

DB_PATH = Path("data/football.db")
REPORT_DIR = Path("reports")
LEAGUE = "Premier League"


def load_data():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    try:
        # Exclude the known incomplete 2021/2022 import for now.
        historical = db.execute(
            """
    SELECT
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
      AND m.season != '2021/2022'
    ORDER BY m.match_date
    """,
            (LEAGUE,),
        ).fetchall()
        today = datetime.now(timezone.utc).date().isoformat()

        upcoming = db.execute(
            """
            SELECT provider_match_id, match_date, home_team, away_team
            FROM matches
            WHERE provider = 'pitchapi'
              AND league_name = ?
              AND is_final = 0
              AND match_date >= ?
              AND home_team IS NOT NULL
              AND away_team IS NOT NULL
            ORDER BY match_date
        """,
            (LEAGUE, today),
        ).fetchall()

        return [dict(row) for row in historical], [dict(row) for row in upcoming]
    finally:
        db.close()


def main():
    historical, upcoming = load_data()
    xg_count = sum(
        match.get("home_xg") is not None and match.get("away_xg") is not None
        for match in historical
    )
    print(f"Training matches: {len(historical)}")
    print(f"Training matches with both xG values: {xg_count}/{len(historical)}")

    if not historical:
        raise SystemExit(
            "No completed training matches found. Sync a completed season first."
        )

    if not upcoming:
        raise SystemExit("No upcoming fixtures found. Sync the current season first.")

    model = FootballModel().fit(historical)

    predictions = []
    for fixture in upcoming:
        prediction = model.predict(
            fixture["home_team"],
            fixture["away_team"],
        )
        prediction.update(
            {
                "match_id": fixture["provider_match_id"],
                "date": fixture["match_date"],
            }
        )
        predictions.append(prediction)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    report = {
        "generated_at": generated_at,
        "league": LEAGUE,
        "training_matches": model.trained_matches,
        "predictions": predictions,
    }

    filename = (
        "predictions_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + ".json"
    )
    output_path = REPORT_DIR / filename
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Training matches: {model.trained_matches}")
    print(f"Upcoming fixtures predicted: {len(predictions)}")
    print(f"Report saved: {output_path}")

    for p in predictions[:20]:
        print(
            f"{p['date']} | {p['home_team']} vs {p['away_team']} "
            f"| {p['most_likely_score']} "
            f"| H {p['home_win_probability']:.1%} "
            f"D {p['draw_probability']:.1%} "
            f"A {p['away_win_probability']:.1%}"
        )


if __name__ == "__main__":
    main()
