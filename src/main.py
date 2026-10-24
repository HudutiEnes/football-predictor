"""CLI:  python -m src.main competitions
python -m src.main fixtures PL --status SCHEDULED --limit 10
python -m src.main raw PL        (print one raw match, to inspect real fields)"""

import argparse
import json
import sys

from src.data.client import FootballDataClient, FootballDataError
from src.data.normalize import normalize_match


def cmd_competitions(client, args):
    data = client.get_competitions()
    comps = data.get("competitions", [])
    print(f"{len(comps)} competitions returned for your token:\n")
    for c in comps:
        area = (c.get("area") or {}).get("name") or ""
        print(
            f"{str(c.get('code')):6} id={str(c.get('id')):<5} {str(c.get('name')):28} {area:15} "
            f"type={c.get('type')} plan={c.get('plan')}"
        )
    print("\nResponse headers starting with x- (rate limit info):")
    for k, v in client.last_headers.items():
        if k.lower().startswith("x-"):
            print(f"  {k}: {v}")


def cmd_fixtures(client, args):
    data = client.get_matches(args.code, status=args.status, season=args.season)
    matches = [normalize_match(m, args.code) for m in data.get("matches", [])]
    if not matches:
        print("No matches returned (empty season, wrong status, or nothing scheduled).")
        return
    matches.sort(key=lambda m: m["utc_date"] or "")
    shown = matches[: args.limit] if args.limit else matches
    for m in shown:
        score = f"{m['home_goals']}-{m['away_goals']}" if m["is_final"] else "  -  "
        print(
            f"{m['utc_date']}  {str(m['status']):10} {m['home_team']} vs {m['away_team']}  {score}"
        )
    print(f"\nShown {len(shown)} of {len(matches)} matches.")
    if args.save:
        path = f"data/{args.code}_matches.json"
        with open(path, "w") as f:
            json.dump(matches, f, indent=2)
        print(f"Saved normalized matches to {path}")


def cmd_raw(client, args):
    data = client.get_matches(args.code, status=args.status, season=args.season)
    matches = data.get("matches", [])
    print("Top-level keys:", list(data.keys()))
    if matches:
        print(json.dumps(matches[0], indent=2))
    else:
        print("No matches returned.")


def main():
    parser = argparse.ArgumentParser(description="Football predictor data tools")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("competitions")

    for name in ("fixtures", "raw"):
        p = sub.add_parser(name)
        p.add_argument("code", help="competition code, e.g. PL, CL, BL1")
        p.add_argument("--status", default=None, help="e.g. SCHEDULED, FINISHED")
        p.add_argument("--season", default=None, help="start year, e.g. 2025")
        if name == "fixtures":
            p.add_argument("--limit", type=int, default=15)
            p.add_argument("--save", action="store_true")

    args = parser.parse_args()
    try:
        client = FootballDataClient()
        {"competitions": cmd_competitions, "fixtures": cmd_fixtures, "raw": cmd_raw}[
            args.command
        ](client, args)
    except FootballDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
