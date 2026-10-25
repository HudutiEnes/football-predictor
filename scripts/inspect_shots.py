"""Describe what the stored shot payloads contain (read-only).

    .venv/bin/python scripts/inspect_shots.py --db data/snapshots/football-backfilled-2026-10-25.db

Prints the analytics table layout, which fields each shot has, and the values of every
low-variety field (for example a shot 'situation' or 'body part'), so we can decide which
extra xG features are possible. It prints counts only, no API keys and no raw payloads.
"""

import argparse
import json
import sqlite3
from collections import Counter, defaultdict


def find_payload_column(db, columns):
    for column in columns:
        if column in ("provider", "provider_match_id", "kind"):
            continue
        row = db.execute(
            f"SELECT {column} FROM analytics WHERE kind = 'shots' AND {column} IS NOT NULL LIMIT 1"
        ).fetchone()
        if not row:
            continue
        try:
            if isinstance(json.loads(row[0]), (dict, list)):
                return column
        except (TypeError, ValueError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/football.db")
    parser.add_argument(
        "--matches", type=int, default=400, help="how many shot payloads to sample"
    )
    parser.add_argument("--max-distinct", type=int, default=15)
    args = parser.parse_args()

    db = sqlite3.connect(args.db)
    columns = [row[1] for row in db.execute("PRAGMA table_info(analytics)")]
    print(f"analytics columns: {columns}")
    print(
        "kinds:",
        db.execute("SELECT kind, COUNT(*) FROM analytics GROUP BY kind").fetchall(),
    )
    column = find_payload_column(db, columns)
    if not column:
        raise SystemExit("Could not find a JSON payload column for kind='shots'.")
    print(f"payload column: {column}")

    rows = db.execute(
        f"SELECT {column} FROM analytics WHERE kind = 'shots' ORDER BY RANDOM() LIMIT ?",
        (args.matches,),
    ).fetchall()

    payload_keys, period_keys = Counter(), Counter()
    key_presence, value_counts, types = (
        Counter(),
        defaultdict(Counter),
        defaultdict(Counter),
    )
    shots_total = 0
    for (text,) in rows:
        payload = json.loads(text)
        payload_keys.update(payload.keys() if isinstance(payload, dict) else [])
        for period in (
            (payload.get("periods") or []) if isinstance(payload, dict) else []
        ):
            period_keys.update(period.keys())
            for shot in period.get("shots") or []:
                shots_total += 1
                for key, value in shot.items():
                    key_presence[key] += 1
                    types[key][type(value).__name__] += 1
                    if isinstance(value, (str, bool)) or value is None:
                        value_counts[key][str(value)] += 1

    print(f"\nsampled {len(rows)} payloads, {shots_total} shots")
    print(f"payload top-level keys: {dict(payload_keys)}")
    print(f"period keys: {dict(period_keys)}")
    print("\nshot fields (present in how many shots, value types):")
    for key, count in key_presence.most_common():
        print(f"  {key:<32} {count / shots_total:>6.1%}  {dict(types[key])}")
    print("\nfields with few distinct text/boolean values (counts):")
    for key, counter in value_counts.items():
        if len(counter) <= args.max_distinct and key not in (
            "team_id",
            "player_id",
            "id",
        ):
            print(f"  {key}: {dict(counter.most_common())}")


if __name__ == "__main__":
    main()
