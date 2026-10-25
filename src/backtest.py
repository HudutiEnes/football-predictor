"""Chronological, date-grouped backtest with optional xG blend, Dixon-Coles and goal markets.

Every match is predicted using only matches from earlier dates. Reported per market:
1X2 (log loss, Brier, RPS, accuracy), exact score (probability given to the real score),
both-teams-to-score and over/under 2.5, each compared with a baseline that has no team
information, plus 95% intervals on the paired differences.
"""

import argparse
import hashlib
import json
import sqlite3
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from math import log
from pathlib import Path

from src import metrics
from src.model import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_XG_WEIGHT,
    MAX_GOALS,
    PRIOR_MATCHES,
    FootballModel,
    brier_score,
    calibration_buckets,
    log_loss,
    poisson_grid,
)

DB_PATH = Path("data/football.db")
EXPERIMENT_LOG = Path("reports/experiments.jsonl")
MIN_TRAINING_MATCHES = 100
OUTCOMES = ("home", "draw", "away")


def load_matches(league, db_path=None):
    db_path = Path(db_path) if db_path else DB_PATH
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    with sqlite3.connect(db_path) as db:
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


def list_leagues(db_path=None):
    db_path = Path(db_path) if db_path else DB_PATH
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            "SELECT DISTINCT league_name FROM matches WHERE provider = 'pitchapi' "
            "AND is_final = 1 AND league_name IS NOT NULL ORDER BY league_name"
        ).fetchall()
    return [row[0] for row in rows]


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


def evaluate(
    matches,
    half_life_days=DEFAULT_HALF_LIFE_DAYS,
    xg_weight=DEFAULT_XG_WEIGHT,
    min_training_matches=MIN_TRAINING_MATCHES,
    prior_matches=PRIOR_MATCHES,
    dixon_coles=False,
):
    """Walk forward through ``matches`` (ordered by date) and return one record per prediction.

    Training for a date contains only matches from strictly earlier dates, so nothing from
    the future (including the xG of the match being predicted) can influence a prediction.
    """
    matches_by_date = defaultdict(list)
    for match in matches:
        matches_by_date[match["match_date"][:10]].append(match)

    previous_matches = []
    records = []
    for match_date in sorted(matches_by_date):
        date_matches = matches_by_date[match_date]
        # Every match on the date is predicted before that date's results enter training.
        if len(previous_matches) >= min_training_matches:
            model = FootballModel(
                half_life_days=half_life_days,
                xg_weight=xg_weight,
                prior_matches=prior_matches,
                dixon_coles=dixon_coles,
            ).fit(previous_matches)

            outcome_counts = Counter(actual_outcome(m) for m in previous_matches)
            prior_total = sum(outcome_counts.values())
            baseline_probs = {o: outcome_counts[o] / prior_total for o in OUTCOMES}
            baseline_choice = max(baseline_probs, key=baseline_probs.get)

            # Baselines with no team information, built only from earlier matches.
            n_prev = len(previous_matches)
            avg_home = sum(m["home_goals"] for m in previous_matches) / n_prev
            avg_away = sum(m["away_goals"] for m in previous_matches) / n_prev
            avg_grid = poisson_grid(max(avg_home, 0.1), max(avg_away, 0.1))
            avg_total = sum(sum(row) for row in avg_grid)
            btts_base = (
                sum(
                    m["home_goals"] > 0 and m["away_goals"] > 0
                    for m in previous_matches
                )
                + 1
            ) / (n_prev + 2)
            over_base = (
                sum(m["home_goals"] + m["away_goals"] > 2 for m in previous_matches) + 1
            ) / (n_prev + 2)

            for match in date_matches:
                prediction = model.predict(match["home_team"], match["away_team"])
                _, _, grid = model.score_grid(match["home_team"], match["away_team"])
                grid_total = sum(sum(row) for row in grid)
                actual = actual_outcome(match)
                hg, ag = match["home_goals"], match["away_goals"]
                model_probs = prediction_probabilities(prediction)
                model_choice = max(model_probs, key=model_probs.get)
                inside = hg <= MAX_GOALS and ag <= MAX_GOALS
                btts = hg > 0 and ag > 0
                over25 = hg + ag > 2

                records.append(
                    {
                        "match_id": match.get("provider_match_id"),
                        "date": match_date,
                        "season": match.get("season") or "Unknown",
                        "model_probs": model_probs,
                        "baseline_probs": baseline_probs,
                        "actual": actual,
                        "correct_model": model_choice == actual,
                        "correct_baseline": baseline_choice == actual,
                        "exact_hit": prediction["most_likely_score"] == f"{hg}-{ag}",
                        "model_brier": brier_score(model_probs, actual),
                        "baseline_brier": brier_score(baseline_probs, actual),
                        "model_log_loss": log_loss(model_probs, actual),
                        "baseline_log_loss": log_loss(baseline_probs, actual),
                        "model_rps": metrics.ranked_probability_score(
                            model_probs, actual
                        ),
                        "baseline_rps": metrics.ranked_probability_score(
                            baseline_probs, actual
                        ),
                        "score_ll_model": metrics.probability_log_loss(
                            grid[hg][ag] / grid_total if inside else 0.0
                        ),
                        "score_ll_baseline": metrics.probability_log_loss(
                            avg_grid[hg][ag] / avg_total if inside else 0.0
                        ),
                        "btts_p": prediction["both_teams_score_probability"],
                        "btts_base_p": btts_base,
                        "btts_y": btts,
                        "over25_p": prediction["over_2_5_probability"],
                        "over25_base_p": over_base,
                        "over25_y": over25,
                    }
                )

        previous_matches.extend(date_matches)
    return records


def mean_of(records, key):
    return metrics.mean(r[key] for r in records)


def summarize(records):
    """Aggregate per-match records into the numbers we report and compare."""
    if not records:
        return {"evaluated": 0}
    btts_model_ll = [metrics.binary_log_loss(r["btts_p"], r["btts_y"]) for r in records]
    btts_base_ll = [
        metrics.binary_log_loss(r["btts_base_p"], r["btts_y"]) for r in records
    ]
    over_model_ll = [
        metrics.binary_log_loss(r["over25_p"], r["over25_y"]) for r in records
    ]
    over_base_ll = [
        metrics.binary_log_loss(r["over25_base_p"], r["over25_y"]) for r in records
    ]
    btts_model_b = [metrics.binary_brier(r["btts_p"], r["btts_y"]) for r in records]
    btts_base_b = [metrics.binary_brier(r["btts_base_p"], r["btts_y"]) for r in records]
    over_model_b = [metrics.binary_brier(r["over25_p"], r["over25_y"]) for r in records]
    over_base_b = [
        metrics.binary_brier(r["over25_base_p"], r["over25_y"]) for r in records
    ]
    return {
        "evaluated": len(records),
        "accuracy_model": mean_of(records, "correct_model"),
        "accuracy_baseline": mean_of(records, "correct_baseline"),
        "brier_model": mean_of(records, "model_brier"),
        "brier_baseline": mean_of(records, "baseline_brier"),
        "log_loss_model": mean_of(records, "model_log_loss"),
        "log_loss_baseline": mean_of(records, "baseline_log_loss"),
        "rps_model": mean_of(records, "model_rps"),
        "rps_baseline": mean_of(records, "baseline_rps"),
        "exact_score_accuracy": mean_of(records, "exact_hit"),
        "exact_score_log_loss_model": mean_of(records, "score_ll_model"),
        "exact_score_log_loss_baseline": mean_of(records, "score_ll_baseline"),
        "btts_log_loss_model": metrics.mean(btts_model_ll),
        "btts_log_loss_baseline": metrics.mean(btts_base_ll),
        "btts_brier_model": metrics.mean(btts_model_b),
        "btts_brier_baseline": metrics.mean(btts_base_b),
        "over25_log_loss_model": metrics.mean(over_model_ll),
        "over25_log_loss_baseline": metrics.mean(over_base_ll),
        "over25_brier_model": metrics.mean(over_model_b),
        "over25_brier_baseline": metrics.mean(over_base_b),
        "paired": {
            "1x2_log_loss": metrics.paired_difference(
                [r["model_log_loss"] for r in records],
                [r["baseline_log_loss"] for r in records],
            ),
            "1x2_rps": metrics.paired_difference(
                [r["model_rps"] for r in records], [r["baseline_rps"] for r in records]
            ),
            "exact_score_log_loss": metrics.paired_difference(
                [r["score_ll_model"] for r in records],
                [r["score_ll_baseline"] for r in records],
            ),
            "btts_log_loss": metrics.paired_difference(btts_model_ll, btts_base_ll),
            "over25_log_loss": metrics.paired_difference(over_model_ll, over_base_ll),
        },
    }


def format_diff(diff):
    return f"{diff['mean_diff']:+.4f} ± {diff['ci95']:.4f}"


def print_report(records, args, header_prefix=""):
    """Prints the original report block first (unchanged labels), then the new market blocks."""
    summary = summarize(records)
    evaluated = summary["evaluated"]
    print()
    print(
        f"Model settings: half-life={args.half_life_days:g} days, xG weight={args.xg_weight:g}"
        + (f", Dixon-Coles on" if args.dixon_coles else "")
    )
    print(f"Initial training window: {args.min_training_matches}")
    print(f"Matches evaluated: {evaluated}")
    print()
    print(f"Model 1X2 accuracy:     {summary['accuracy_model']:.1%}")
    print(f"Baseline 1X2 accuracy:  {summary['accuracy_baseline']:.1%}")
    print(f"Model Brier score:      {summary['brier_model']:.4f}")
    print(f"Baseline Brier score:   {summary['brier_baseline']:.4f}")
    print(f"Model log loss:         {summary['log_loss_model']:.4f}")
    print(f"Baseline log loss:      {summary['log_loss_baseline']:.4f}")
    print(f"Exact score accuracy:   {summary['exact_score_accuracy']:.1%}")
    print()
    print("Model calibration (predicted probability bucket vs observed rate):")
    print(f"{'Bucket':>10} {'Predicted':>10} {'Observed':>10} {'Samples':>8}")
    for bucket in calibration_buckets(
        [(r["model_probs"], r["actual"]) for r in records]
    ):
        print(
            f"{bucket['range']:>10} "
            f"{bucket['predicted_rate']:>9.1%} "
            f"{bucket['observed_rate']:>9.1%} "
            f"{bucket['samples']:>8}"
        )

    print()
    print("All markets (lower is better; model vs baseline with no team information):")
    print(f"{'Market':<28} {'Model':>9} {'Baseline':>9}   Paired diff (95% interval)")
    rows = [
        (
            "1X2 log loss",
            summary["log_loss_model"],
            summary["log_loss_baseline"],
            summary["paired"]["1x2_log_loss"],
        ),
        (
            "1X2 ranked prob. score",
            summary["rps_model"],
            summary["rps_baseline"],
            summary["paired"]["1x2_rps"],
        ),
        (
            "Exact score log loss",
            summary["exact_score_log_loss_model"],
            summary["exact_score_log_loss_baseline"],
            summary["paired"]["exact_score_log_loss"],
        ),
        (
            "Both teams score log loss",
            summary["btts_log_loss_model"],
            summary["btts_log_loss_baseline"],
            summary["paired"]["btts_log_loss"],
        ),
        (
            "Over 2.5 goals log loss",
            summary["over25_log_loss_model"],
            summary["over25_log_loss_baseline"],
            summary["paired"]["over25_log_loss"],
        ),
    ]
    for label, model_value, baseline_value, diff in rows:
        print(
            f"{label:<28} {model_value:>9.4f} {baseline_value:>9.4f}   {format_diff(diff)}"
        )
    print(
        "Negative diff = model better. If the interval includes 0 the sample is too small to tell."
    )
    print(
        "Exact-score baseline = league-average Poisson; BTTS/over baselines = historical frequency."
    )

    for label, key_p, key_y in (
        ("Both teams score", "btts_p", "btts_y"),
        ("Over 2.5 goals", "over25_p", "over25_y"),
    ):
        print()
        print(f"{label} calibration:")
        print(f"{'Bucket':>10} {'Predicted':>10} {'Observed':>10} {'Samples':>8}")
        for bucket in metrics.binary_calibration(
            [(r[key_p], r[key_y]) for r in records]
        ):
            print(
                f"{bucket['range']:>10} {bucket['predicted_rate']:>9.1%} "
                f"{bucket['observed_rate']:>9.1%} {bucket['samples']:>8}"
            )

    by_season = defaultdict(list)
    for r in records:
        by_season[r["season"]].append(r)
    print()
    print("By season (burn-in matches early in the first season are noisier):")
    print(
        f"{'Season':<12} {'N':>5} {'LogLoss':>8} {'Base':>8} {'RPS':>7} {'BTTS LL':>8} {'O2.5 LL':>8} {'Acc':>6}"
    )
    for season in sorted(by_season):
        s = summarize(by_season[season])
        print(
            f"{season:<12} {s['evaluated']:>5} {s['log_loss_model']:>8.4f} {s['log_loss_baseline']:>8.4f} "
            f"{s['rps_model']:>7.4f} {s['btts_log_loss_model']:>8.4f} {s['over25_log_loss_model']:>8.4f} "
            f"{s['accuracy_model']:>6.1%}"
        )
    return summary


def file_fingerprint(path):
    path = Path(path)
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        ).stdout.strip()
    except Exception:
        return None


def log_experiment(path, entry):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--league", default="Premier League", help="Exact league_name in the database"
    )
    parser.add_argument(
        "--all-leagues",
        action="store_true",
        help="Run every league in the database and pool the results",
    )
    parser.add_argument("--half-life-days", type=float, default=DEFAULT_HALF_LIFE_DAYS)
    parser.add_argument(
        "--xg-weight",
        type=float,
        default=DEFAULT_XG_WEIGHT,
        help="Blend weight from 0 (goals only) to 1 (xG where available)",
    )
    parser.add_argument(
        "--prior-matches",
        type=float,
        default=PRIOR_MATCHES,
        help="Shrinkage strength: pseudo-matches pulling a team towards the league average",
    )
    parser.add_argument(
        "--dixon-coles",
        action="store_true",
        help="Apply the Dixon-Coles low-score correction (rho estimated from training data)",
    )
    parser.add_argument(
        "--min-training-matches", type=int, default=MIN_TRAINING_MATCHES
    )
    parser.add_argument(
        "--report-from",
        help="Only score predictions from this date (YYYY-MM-DD) on; "
        "training still uses everything earlier",
    )
    parser.add_argument(
        "--report-to",
        help="Only score predictions up to and including this date (YYYY-MM-DD). "
        "Use it to keep the final holdout period out of tuning runs",
    )
    parser.add_argument(
        "--db",
        help=f"Database file (default {DB_PATH}); use a frozen snapshot for experiments",
    )
    parser.add_argument("--json-out", help="Write the summary as JSON to this file")
    parser.add_argument(
        "--note", default="", help="Free-text note stored in the experiment log"
    )
    parser.add_argument("--experiment-log", default=str(EXPERIMENT_LOG))
    parser.add_argument(
        "--no-log", action="store_true", help="Do not append to the experiment log"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    db_path = args.db or DB_PATH
    leagues = list_leagues(db_path) if args.all_leagues else [args.league]

    all_records = []
    per_league = {}
    for league in leagues:
        matches = load_matches(league, db_path)
        if not matches:
            if args.all_leagues:
                continue
            raise SystemExit(
                f"No completed matches found for league {league!r}. "
                "Check the exact league_name stored in the database."
            )

        season_counts = Counter(match["season"] or "Unknown" for match in matches)
        print(f"League: {league}")
        print("Completed matches included by season:")
        for season, count in sorted(season_counts.items()):
            print(f"  {season}: {count}")
        xg_count = sum(
            match.get("home_xg") is not None and match.get("away_xg") is not None
            for match in matches
        )
        print(f"Total matches available: {len(matches)}")
        print(f"Matches with both xG values: {xg_count}/{len(matches)}")

        records = evaluate(
            matches,
            half_life_days=args.half_life_days,
            xg_weight=args.xg_weight,
            min_training_matches=args.min_training_matches,
            prior_matches=args.prior_matches,
            dixon_coles=args.dixon_coles,
        )
        if args.report_from:
            records = [r for r in records if r["date"] >= args.report_from]
        if args.report_to:
            records = [r for r in records if r["date"] <= args.report_to]
        if not records:
            if args.all_leagues:
                print(f"Not enough data to evaluate {league}.\n")
                continue
            raise SystemExit(
                f"Not enough data to evaluate; need at least {args.min_training_matches} prior matches."
            )
        for r in records:
            r["league"] = league
        all_records.extend(records)
        per_league[league] = print_report(records, args)
        print()

    if not all_records:
        raise SystemExit("Nothing could be evaluated.")

    pooled = None
    if args.all_leagues:
        print("=" * 78)
        print("POOLED ACROSS LEAGUES")
        pooled = summarize(all_records)
        print(
            f"{'League':<18} {'N':>5} {'LogLoss':>8} {'Base':>8} {'Diff':>8}   BTTS LL  O2.5 LL"
        )
        for league, s in per_league.items():
            print(
                f"{league:<18} {s['evaluated']:>5} {s['log_loss_model']:>8.4f} {s['log_loss_baseline']:>8.4f} "
                f"{s['log_loss_model'] - s['log_loss_baseline']:>+8.4f}   {s['btts_log_loss_model']:.4f}   "
                f"{s['over25_log_loss_model']:.4f}"
            )
        print(
            f"{'POOLED':<18} {pooled['evaluated']:>5} {pooled['log_loss_model']:>8.4f} "
            f"{pooled['log_loss_baseline']:>8.4f} {pooled['log_loss_model'] - pooled['log_loss_baseline']:>+8.4f}"
            f"   {pooled['btts_log_loss_model']:.4f}   {pooled['over25_log_loss_model']:.4f}"
        )
        print(
            f"Pooled 1X2 log loss vs baseline: {format_diff(pooled['paired']['1x2_log_loss'])}"
        )

    final = pooled if pooled else next(iter(per_league.values()))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(
                {"leagues": per_league, "pooled": pooled}, indent=2, default=str
            ),
            encoding="utf-8",
        )
    if not args.no_log:
        log_experiment(
            args.experiment_log,
            {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "git_commit": git_commit(),
                "db": str(db_path),
                "db_sha256_prefix": file_fingerprint(db_path),
                "leagues": leagues,
                "config": {
                    "half_life_days": args.half_life_days,
                    "xg_weight": args.xg_weight,
                    "prior_matches": args.prior_matches,
                    "dixon_coles": args.dixon_coles,
                    "min_training_matches": args.min_training_matches,
                    "report_from": args.report_from,
                    "report_to": args.report_to,
                },
                "note": args.note,
                "summary": {k: v for k, v in final.items() if k != "paired"},
                "paired": final["paired"],
            },
        )
        print(f"Experiment appended to {args.experiment_log}")


if __name__ == "__main__":
    main()
