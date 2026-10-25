"""Hyper-parameter sweep on the TUNING window only.

    python -m src.sweep --db data/snapshots/football-backfilled-2026-10-25.db

Every configuration is walk-forward backtested on the same matches (default window
2023-08-01..2026-07-31, all leagues). A configuration is only called BETTER than the reference
(your current settings) when ALL of these hold:
  * the improvement is larger than --min-gain log loss,
  * its interval excludes 0 after widening it for the number of configurations compared
    (Bonferroni), so picking the luckiest of 27 tries does not count as a discovery,
  * it improves in most leagues and in most seasons.
The holdout period (default: from 2026-08-01) is protected and cannot be scored by accident.
"""

import argparse
import itertools
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

from src import backtest, metrics
from src.model import DEFAULT_HALF_LIFE_DAYS, DEFAULT_XG_WEIGHT, PRIOR_MATCHES

DEFAULT_REPORT_FROM = "2023-08-01"
DEFAULT_REPORT_TO = "2026-07-31"
DEFAULT_HOLDOUT_START = "2026-08-01"
REPORT_DIR = Path("reports")


def parse_floats(text):
    return [float(part) for part in text.split(",") if part.strip()]


def label(config):
    half_life, xg_weight, prior = config
    return f"hl{half_life:g}_xg{xg_weight:g}_pm{prior:g}"


def run_task(db_path, league, config, min_training, report_from, report_to):
    """One (configuration, league) backtest. Top-level so worker processes can import it."""
    half_life, xg_weight, prior = config
    matches = backtest.load_matches(league, db_path)
    records = backtest.evaluate(
        matches,
        half_life_days=half_life,
        xg_weight=xg_weight,
        prior_matches=prior,
        min_training_matches=min_training,
    )
    kept = [r for r in records if report_from <= r["date"] <= report_to]
    for record in kept:
        record["league"] = league
    return config, league, kept


def analyze(results, reference, min_gain=0.002, alpha=0.05, extra_comparisons=0):
    """Compare every configuration with ``reference``. ``results`` = {config: [records]}.

    ``extra_comparisons`` counts configurations already tried in earlier sweeps on the same data,
    so a follow-up sweep widens its intervals instead of quietly resetting the multiple-testing
    budget. Records must carry league / match_id / season / model_log_loss."""
    comparisons = max(1, len(results) - 1 + extra_comparisons)
    z = NormalDist().inv_cdf(1 - alpha / (2 * comparisons))
    ref_records = {(r["league"], r["match_id"]): r for r in results[reference]}
    leagues = sorted({r["league"] for r in results[reference]})
    seasons = sorted({r["season"] for r in results[reference]})
    need_leagues = math.ceil(0.6 * len(leagues))
    need_seasons = max(1, math.ceil(2 * len(seasons) / 3))

    rows = []
    for config, records in results.items():
        summary = backtest.summarize(records)
        mine, theirs = [], []
        league_diffs, season_diffs = defaultdict(list), defaultdict(list)
        for r in records:
            ref = ref_records.get((r["league"], r["match_id"]))
            if ref is None:
                continue
            mine.append(r["model_log_loss"])
            theirs.append(ref["model_log_loss"])
            diff = r["model_log_loss"] - ref["model_log_loss"]
            league_diffs[r["league"]].append(diff)
            season_diffs[r["season"]].append(diff)
        paired = metrics.paired_difference(mine, theirs, z=z)
        leagues_better = sum(1 for d in league_diffs.values() if sum(d) / len(d) < 0)
        seasons_better = sum(1 for d in season_diffs.values() if sum(d) / len(d) < 0)

        if config == reference:
            verdict = "reference"
        elif (
            paired["mean_diff"] + paired["ci"] < 0
            and paired["mean_diff"] <= -min_gain
            and leagues_better >= need_leagues
            and seasons_better >= need_seasons
        ):
            verdict = "BETTER"
        elif paired["mean_diff"] - paired["ci"] > 0:
            verdict = "worse"
        else:
            verdict = "no clear difference"
        rows.append(
            {
                "config": config,
                "label": label(config),
                "n": summary["evaluated"],
                "log_loss": summary["log_loss_model"],
                "rps": summary["rps_model"],
                "exact_score_log_loss": summary["exact_score_log_loss_model"],
                "btts_log_loss": summary["btts_log_loss_model"],
                "over25_log_loss": summary["over25_log_loss_model"],
                "diff_vs_reference": paired["mean_diff"],
                "adjusted_interval": paired["ci"],
                "leagues_better": leagues_better,
                "leagues_total": len(leagues),
                "seasons_better": seasons_better,
                "seasons_total": len(seasons),
                "by_league": {k: sum(v) / len(v) for k, v in league_diffs.items()},
                "verdict": verdict,
            }
        )
    rows.sort(key=lambda row: row["log_loss"])
    return rows, z


def sensitivity(rows):
    """Average pooled log loss for each value of each parameter (is the surface smooth?)."""
    names = ("half_life_days", "xg_weight", "prior_matches")
    table = {}
    for index, name in enumerate(names):
        groups = defaultdict(list)
        for row in rows:
            groups[row["config"][index]].append(row["log_loss"])
        table[name] = {value: sum(v) / len(v) for value, v in sorted(groups.items())}
    return table


def format_report(rows, z, reference, args, extra_note=""):
    lines = [
        f"Sweep on tuning window {args.report_from} .. {args.report_to} "
        f"({rows[0]['n']} matches per configuration, {len(rows)} configurations)",
        f"Reference (current settings): {label(reference)}",
        f"Interval half-widths use z={z:.2f} (widened for {len(rows) - 1 + args.extra_comparisons} comparisons); "
        f"'BETTER' also needs gain >= {args.min_gain} and consistency across leagues and seasons.",
        "",
        f"{'Config':<22} {'LogLoss':>8} {'vsRef':>8} {'±adj':>7} {'Lg':>4} {'Ssn':>4} {'RPS':>7} "
        f"{'Exact':>7} {'BTTS':>7} {'O2.5':>7}  Verdict",
    ]
    for row in rows:
        lines.append(
            f"{row['label']:<22} {row['log_loss']:>8.4f} {row['diff_vs_reference']:>+8.4f} "
            f"{row['adjusted_interval']:>7.4f} {row['leagues_better']}/{row['leagues_total']:<2} "
            f"{row['seasons_better']}/{row['seasons_total']:<2} {row['rps']:>7.4f} "
            f"{row['exact_score_log_loss']:>7.4f} {row['btts_log_loss']:>7.4f} {row['over25_log_loss']:>7.4f}  "
            f"{row['verdict']}"
        )
    lines += [
        "",
        "Average pooled log loss per parameter value (smooth trends are more trustworthy than a lone best cell):",
    ]
    for name, values in sensitivity(rows).items():
        lines.append(
            f"  {name}: "
            + ", ".join(f"{value:g}={loss:.4f}" for value, loss in values.items())
        )
    better = [row for row in rows if row["verdict"] == "BETTER"]
    lines.append("")
    if better:
        best = better[0]
        lines.append(
            f"Recommendation: {best['label']} beats the reference by {-best['diff_vs_reference']:.4f} "
            f"log loss (adjusted interval ±{best['adjusted_interval']:.4f})."
        )
        lines.append(
            "Do NOT confirm on the holdout yet if more experiments are planned: the holdout is used "
            "once, after every change is final."
        )
    else:
        lines.append(
            "Recommendation: no configuration clearly beats the current settings. Keep them."
        )
    if extra_note:
        lines += ["", extra_note]
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", help="database file (use a frozen snapshot)")
    parser.add_argument(
        "--leagues", help="comma separated league names (default: all in the database)"
    )
    parser.add_argument("--half-lives", default="180,365,730")
    parser.add_argument("--xg-weights", default="0.5,0.75,1.0")
    parser.add_argument("--priors", default="3,5,8", help="prior_matches values")
    parser.add_argument(
        "--reference",
        help="half_life,xg_weight,prior to compare against "
        f"(default {DEFAULT_HALF_LIFE_DAYS:g},{DEFAULT_XG_WEIGHT:g},{PRIOR_MATCHES:g})",
    )
    parser.add_argument(
        "--min-training-matches", type=int, default=backtest.MIN_TRAINING_MATCHES
    )
    parser.add_argument("--report-from", default=DEFAULT_REPORT_FROM)
    parser.add_argument("--report-to", default=DEFAULT_REPORT_TO)
    parser.add_argument("--holdout-start", default=DEFAULT_HOLDOUT_START)
    parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help="DANGER: lets the sweep score the holdout",
    )
    parser.add_argument("--min-gain", type=float, default=0.002)
    parser.add_argument(
        "--extra-comparisons",
        type=int,
        default=0,
        help="configurations already tried on this data in earlier sweeps (widens the intervals)",
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument("--note", default="")
    parser.add_argument("--out-dir", default=str(REPORT_DIR))
    return parser


def run_sweep(args):
    if args.report_to >= args.holdout_start and not args.allow_holdout:
        raise SystemExit(
            f"Refusing to sweep: --report-to {args.report_to} reaches the protected holdout "
            f"(starts {args.holdout_start}). Tune on earlier data; use --allow-holdout only for the one final check."
        )
    db_path = args.db or backtest.DB_PATH
    leagues = (
        [name.strip() for name in args.leagues.split(",")]
        if args.leagues
        else backtest.list_leagues(db_path)
    )
    grid = list(
        itertools.product(
            parse_floats(args.half_lives),
            parse_floats(args.xg_weights),
            parse_floats(args.priors),
        )
    )
    reference = (
        tuple(parse_floats(args.reference))
        if args.reference
        else (
            float(DEFAULT_HALF_LIFE_DAYS),
            float(DEFAULT_XG_WEIGHT),
            float(PRIOR_MATCHES),
        )
    )
    if reference not in grid:
        grid.append(reference)

    tasks = [
        (
            db_path,
            league,
            config,
            args.min_training_matches,
            args.report_from,
            args.report_to,
        )
        for config in grid
        for league in leagues
    ]
    print(
        f"{len(grid)} configurations x {len(leagues)} leagues = {len(tasks)} backtests, "
        f"{args.workers} worker(s)"
    )

    results = defaultdict(list)
    done = 0

    def collect(config, league, records):
        nonlocal done
        results[config].extend(records)
        done += 1
        print(
            f"[{done}/{len(tasks)}] {label(config)} {league}: {len(records)} matches",
            flush=True,
        )

    if args.workers <= 1:
        for task in tasks:
            collect(*run_task(*task))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_task, *task) for task in tasks]
            for future in as_completed(futures):
                collect(*future.result())

    if not results[reference]:
        raise SystemExit("No matches were scored in the requested window.")
    rows, z = analyze(
        results,
        reference,
        min_gain=args.min_gain,
        extra_comparisons=args.extra_comparisons,
    )
    note = "Selection happened on the tuning window only. The holdout has not been touched by this run."
    text = format_report(rows, z, reference, args, note)
    print("\n" + text)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out_dir / f"sweep_{stamp}.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / f"sweep_{stamp}.json").write_text(
        json.dumps(
            {
                "window": [args.report_from, args.report_to],
                "reference": label(reference),
                "z": z,
                "db": str(db_path),
                "db_sha256_prefix": backtest.file_fingerprint(db_path),
                "note": args.note,
                "git_commit": backtest.git_commit(),
                "rows": rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved {out_dir / ('sweep_' + stamp + '.txt')} and .json")
    return rows


def main(argv=None):
    run_sweep(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
