"""Recency-weighted Poisson model with optional xG blending.

Only completed matches supplied to ``fit`` influence the model. The caller is
responsible for supplying training matches that happened strictly before the
match being predicted (the backtest does this by date group).
"""

from collections import defaultdict
from datetime import date
from math import exp, factorial, log

MAX_GOALS = 12
PRIOR_MATCHES = 5
# Earlier tuning found 365 days slightly better than 180 on the available data.
DEFAULT_HALF_LIFE_DAYS = 365
DEFAULT_XG_WEIGHT = 1.0


def parse_match_date(value):
    """Parse a date or ISO datetime string, returning None if unavailable."""
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def poisson_probability(goals, expected):
    """Poisson probability for a non-negative goal count."""
    return exp(-expected) * expected**goals / factorial(goals)


def brier_score(probabilities, actual):
    """Multi-class Brier score for a 1X2 probability dict."""
    return sum(
        (probabilities[outcome] - float(outcome == actual)) ** 2
        for outcome in probabilities
    )


def log_loss(probabilities, actual):
    """Log loss for a 1X2 probability dict."""
    return -log(max(probabilities[actual], 1e-15))


def calibration_buckets(records, bins=5):
    """Bin predicted 1X2 probabilities and compare to observed frequencies.

    ``records`` is an iterable of ``(probabilities, actual_outcome)`` tuples,
    where ``probabilities`` is a dict like {"home": 0.5, "draw": 0.25, "away": 0.25}.
    Returns a list of dicts with predicted/observed rates and sample counts.
    """
    edges = [i / bins for i in range(bins + 1)]
    buckets = [
        {
            "low": edges[i],
            "high": edges[i + 1],
            "predicted_sum": 0.0,
            "observed_sum": 0.0,
            "count": 0,
        }
        for i in range(bins)
    ]

    for probabilities, actual in records:
        for outcome, probability in probabilities.items():
            for bucket in buckets:
                if bucket["low"] <= probability < bucket["high"] or (
                    probability == 1.0 and bucket["high"] == 1.0
                ):
                    bucket["predicted_sum"] += probability
                    bucket["observed_sum"] += float(outcome == actual)
                    bucket["count"] += 1
                    break

    result = []
    for bucket in buckets:
        count = bucket["count"]
        if count == 0:
            continue
        result.append(
            {
                "range": f"{bucket['low']:.0%} - {bucket['high']:.0%}",
                "predicted_rate": bucket["predicted_sum"] / count,
                "observed_rate": bucket["observed_sum"] / count,
                "samples": count,
            }
        )
    return result


def _number_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # NaN compares unequal to itself; infinities are not valid feature values.
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _blended_value(goals, xg, xg_weight):
    """Blend goals and xG; fall back to goals when this xG value is missing."""
    goals_value = float(goals)
    xg_value = _number_or_none(xg)
    if xg_weight <= 0 or xg_value is None:
        return goals_value
    return (1.0 - xg_weight) * goals_value + xg_weight * xg_value


class FootballModel:
    """Goal-rate model with venue splits, shrinkage, recency and optional xG.

    ``xg_weight=0`` is the goals-only baseline. ``xg_weight=1`` uses xG
    wherever available. Intermediate values blend the two signals.
    """

    def __init__(
        self,
        half_life_days=DEFAULT_HALF_LIFE_DAYS,
        xg_weight=DEFAULT_XG_WEIGHT,
        prior_matches=PRIOR_MATCHES,
    ):
        if half_life_days <= 0:
            raise ValueError("half_life_days must be greater than zero")
        if not 0.0 <= xg_weight <= 1.0:
            raise ValueError("xg_weight must be between 0 and 1")
        if prior_matches <= 0:
            raise ValueError("prior_matches must be greater than zero")

        self.half_life_days = float(half_life_days)
        self.xg_weight = float(xg_weight)
        self.prior_matches = float(prior_matches)
        self.teams = {}
        self.home_avg = 1.45
        self.away_avg = 1.15
        self.trained_matches = 0
        self.latest_training_date = None

    def fit(self, matches):
        """Fit on completed matches, applying recency weights and xG blending."""
        eligible = [
            m for m in matches
            if m.get("is_final")
            and m.get("home_goals") is not None
            and m.get("away_goals") is not None
            and m.get("home_team")
            and m.get("away_team")
        ]
        if not eligible:
            raise ValueError("No completed matches available for training.")

        dated_matches = [(m, parse_match_date(m.get("match_date"))) for m in eligible]
        known_dates = [match_date for _, match_date in dated_matches if match_date]

        if known_dates:
            latest_date = max(known_dates)
            weighted_matches = []
            for match, match_date in dated_matches:
                # An undated record cannot be placed safely in the recency window.
                if match_date is None:
                    continue
                age_days = max(0, (latest_date - match_date).days)
                weight = 0.5 ** (age_days / self.half_life_days)
                weighted_matches.append((match, weight))
            self.latest_training_date = latest_date.isoformat()
        else:
            # Useful for tiny unit tests / explicitly undated toy datasets only.
            weighted_matches = [(match, 1.0) for match in eligible]
            self.latest_training_date = None

        if not weighted_matches:
            raise ValueError("No usable dated matches available for training.")

        total_weight = sum(weight for _, weight in weighted_matches)
        self.trained_matches = len(weighted_matches)

        # Build a consistent outcome signal for both league averages and team rates.
        # If a particular xG value is missing, that side safely falls back to goals.
        enriched = []
        for match, weight in weighted_matches:
            home_signal = _blended_value(
                match["home_goals"], match.get("home_xg"), self.xg_weight
            )
            away_signal = _blended_value(
                match["away_goals"], match.get("away_xg"), self.xg_weight
            )
            enriched.append((match, weight, home_signal, away_signal))

        self.home_avg = max(
            sum(weight * home_signal for _, weight, home_signal, _ in enriched)
            / total_weight,
            0.1,
        )
        self.away_avg = max(
            sum(weight * away_signal for _, weight, _, away_signal in enriched)
            / total_weight,
            0.1,
        )

        stats = defaultdict(
            lambda: {
                "home_played": 0.0,
                "home_scored": 0.0,
                "home_conceded": 0.0,
                "away_played": 0.0,
                "away_scored": 0.0,
                "away_conceded": 0.0,
            }
        )

        for match, weight, home_signal, away_signal in enriched:
            home = match["home_team"]
            away = match["away_team"]

            stats[home]["home_played"] += weight
            stats[home]["home_scored"] += weight * home_signal
            stats[home]["home_conceded"] += weight * away_signal

            stats[away]["away_played"] += weight
            stats[away]["away_scored"] += weight * away_signal
            stats[away]["away_conceded"] += weight * home_signal

        self.teams = {}
        prior = self.prior_matches

        for team, team_stats in stats.items():
            home_played = team_stats["home_played"]
            away_played = team_stats["away_played"]

            home_scoring = (
                team_stats["home_scored"] + prior * self.home_avg
            ) / (home_played + prior)
            home_conceding = (
                team_stats["home_conceded"] + prior * self.away_avg
            ) / (home_played + prior)
            away_scoring = (
                team_stats["away_scored"] + prior * self.away_avg
            ) / (away_played + prior)
            away_conceding = (
                team_stats["away_conceded"] + prior * self.home_avg
            ) / (away_played + prior)

            self.teams[team] = {
                "home_attack": home_scoring / self.home_avg,
                "home_defence": home_conceding / self.away_avg,
                "away_attack": away_scoring / self.away_avg,
                "away_defence": away_conceding / self.home_avg,
            }

        return self

    def predict(self, home_team, away_team):
        """Return goal expectations, 1X2 probabilities and likely scorelines."""
        home = self.teams.get(home_team, {})
        away = self.teams.get(away_team, {})

        expected_home = self.home_avg * (
            home.get("home_attack", 1.0) * away.get("away_defence", 1.0)
        )
        expected_away = self.away_avg * (
            away.get("away_attack", 1.0) * home.get("home_defence", 1.0)
        )
        expected_home = min(max(expected_home, 0.05), 6.0)
        expected_away = min(max(expected_away, 0.05), 6.0)

        home_probs = [
            poisson_probability(goals, expected_home)
            for goals in range(MAX_GOALS + 1)
        ]
        away_probs = [
            poisson_probability(goals, expected_away)
            for goals in range(MAX_GOALS + 1)
        ]

        outcomes = {"home": 0.0, "draw": 0.0, "away": 0.0}
        scores = []
        for home_goals, home_probability in enumerate(home_probs):
            for away_goals, away_probability in enumerate(away_probs):
                probability = home_probability * away_probability
                scores.append((home_goals, away_goals, probability))
                if home_goals > away_goals:
                    outcomes["home"] += probability
                elif home_goals == away_goals:
                    outcomes["draw"] += probability
                else:
                    outcomes["away"] += probability

        # The grid is truncated at MAX_GOALS; normalize to make the 1X2
        # probabilities sum to exactly 1 within floating-point precision.
        outcome_total = sum(outcomes.values())
        scores.sort(key=lambda item: item[2], reverse=True)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "expected_home_goals": round(expected_home, 2),
            "expected_away_goals": round(expected_away, 2),
            "home_win_probability": round(outcomes["home"] / outcome_total, 4),
            "draw_probability": round(outcomes["draw"] / outcome_total, 4),
            "away_win_probability": round(outcomes["away"] / outcome_total, 4),
            "most_likely_score": f"{scores[0][0]}-{scores[0][1]}",
            "top_scorelines": [
                {
                    "score": f"{home_goals}-{away_goals}",
                    "probability": round(probability / outcome_total, 4),
                }
                for home_goals, away_goals, probability in scores[:5]
            ],
        }
