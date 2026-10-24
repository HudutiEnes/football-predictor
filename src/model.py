"""Recency-weighted Poisson model for football scores."""

from collections import defaultdict
from datetime import date
from math import exp, factorial

MAX_GOALS = 8
PRIOR_MATCHES = 5
HALF_LIFE_DAYS = 180


def parse_match_date(value):
    if not value:
        return None

    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def poisson_probability(goals, expected):
    return exp(-expected) * expected**goals / factorial(goals)


class FootballModel:
    def __init__(self, half_life_days=HALF_LIFE_DAYS):
        if half_life_days <= 0:
            raise ValueError("half_life_days must be greater than zero")

        self.half_life_days = half_life_days
        self.teams = {}
        self.home_avg = 1.45
        self.away_avg = 1.15
        self.trained_matches = 0

    def fit(self, matches):
        eligible = [
            m
            for m in matches
            if m.get("is_final")
            and m.get("home_goals") is not None
            and m.get("away_goals") is not None
            and m.get("home_team")
            and m.get("away_team")
        ]

        if not eligible:
            raise ValueError("No completed matches available for training.")

        dated_matches = [(m, parse_match_date(m.get("match_date"))) for m in eligible]
        known_dates = [d for _, d in dated_matches if d is not None]

        # If dates exist, exclude undated records because they cannot be
        # assigned a reliable place in the recency calculation.
        if known_dates:
            latest_date = max(known_dates)
            weighted_matches = []

            for match, match_date in dated_matches:
                if match_date is None:
                    continue

                age_days = max(0, (latest_date - match_date).days)
                weight = 0.5 ** (age_days / self.half_life_days)
                weighted_matches.append((match, weight))
        else:
            # Supports datasets that genuinely contain no dates.
            weighted_matches = [(m, 1.0) for m in eligible]

        if not weighted_matches:
            raise ValueError("No usable dated matches available for training.")

        total_weight = sum(weight for _, weight in weighted_matches)
        self.trained_matches = len(weighted_matches)

        self.home_avg = max(
            sum(weight * match["home_goals"] for match, weight in weighted_matches)
            / total_weight,
            0.1,
        )
        self.away_avg = max(
            sum(weight * match["away_goals"] for match, weight in weighted_matches)
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

        for match, weight in weighted_matches:
            home = match["home_team"]
            away = match["away_team"]
            home_goals = match["home_goals"]
            away_goals = match["away_goals"]

            stats[home]["home_played"] += weight
            stats[home]["home_scored"] += weight * home_goals
            stats[home]["home_conceded"] += weight * away_goals

            stats[away]["away_played"] += weight
            stats[away]["away_scored"] += weight * away_goals
            stats[away]["away_conceded"] += weight * home_goals

        self.teams = {}

        for team, team_stats in stats.items():
            prior = PRIOR_MATCHES
            home_played = team_stats["home_played"]
            away_played = team_stats["away_played"]

            home_scoring = (team_stats["home_scored"] + prior * self.home_avg) / (
                home_played + prior
            )

            home_conceding = (team_stats["home_conceded"] + prior * self.away_avg) / (
                home_played + prior
            )

            away_scoring = (team_stats["away_scored"] + prior * self.away_avg) / (
                away_played + prior
            )

            away_conceding = (team_stats["away_conceded"] + prior * self.home_avg) / (
                away_played + prior
            )

            self.teams[team] = {
                "home_attack": home_scoring / self.home_avg,
                "home_defence": home_conceding / self.away_avg,
                "away_attack": away_scoring / self.away_avg,
                "away_defence": away_conceding / self.home_avg,
            }

        return self

    def predict(self, home_team, away_team):
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
            poisson_probability(goals, expected_home) for goals in range(MAX_GOALS + 1)
        ]
        away_probs = [
            poisson_probability(goals, expected_away) for goals in range(MAX_GOALS + 1)
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

        total = sum(outcomes.values())
        scores.sort(key=lambda item: item[2], reverse=True)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "expected_home_goals": round(expected_home, 2),
            "expected_away_goals": round(expected_away, 2),
            "home_win_probability": round(outcomes["home"] / total, 4),
            "draw_probability": round(outcomes["draw"] / total, 4),
            "away_win_probability": round(outcomes["away"] / total, 4),
            "most_likely_score": f"{scores[0][0]}-{scores[0][1]}",
            "top_scorelines": [
                {
                    "score": f"{hg}-{ag}",
                    "probability": round(probability / total, 4),
                }
                for hg, ag, probability in scores[:5]
            ],
        }
