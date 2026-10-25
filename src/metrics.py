"""Scoring rules and comparison helpers for probabilistic predictions.

All functions are pure and dependency-free so they are easy to test.
"""

import math

EPS = 1e-15
OUTCOME_ORDER = ("home", "draw", "away")


def ranked_probability_score(probabilities, actual):
    """Ranked Probability Score for ordered outcomes home < draw < away (lower is better).

    Unlike the Brier score it knows that predicting a home win when the away team wins is
    "further" from the truth than predicting a draw. Normalised to the range 0..1."""
    cumulative_gap = 0.0
    total = 0.0
    for outcome in OUTCOME_ORDER[:-1]:
        cumulative_gap += probabilities[outcome] - float(outcome == actual)
        total += cumulative_gap**2
    return total / (len(OUTCOME_ORDER) - 1)


def binary_brier(probability, happened):
    return (probability - float(bool(happened))) ** 2


def binary_log_loss(probability, happened):
    p = probability if happened else 1.0 - probability
    return -math.log(max(p, EPS))


def probability_log_loss(probability):
    """Log loss given the probability that was assigned to what actually happened."""
    return -math.log(max(probability, EPS))


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def paired_difference(model_values, baseline_values, z=1.96):
    """Mean of (model - baseline) per match with an interval of +/- z standard errors.

    ``ci95`` is always the 95% half-width; ``ci`` uses the requested ``z`` (for example a larger
    z when many configurations are compared). For loss metrics a negative mean means the model
    is better. If the interval contains 0, the sample is too small to claim a real difference.
    """
    diffs = [m - b for m, b in zip(model_values, baseline_values)]
    n = len(diffs)
    if n < 2:
        return {
            "mean_diff": mean(diffs),
            "ci95": float("nan"),
            "ci": float("nan"),
            "n": n,
        }
    m = sum(diffs) / n
    variance = sum((d - m) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(variance / n)
    return {"mean_diff": m, "ci95": 1.96 * se, "ci": z * se, "n": n}


def binary_calibration(records, bins=5):
    """``records`` = iterable of (predicted_probability, happened). Returns per-bin summaries."""
    buckets = [[0, 0.0, 0.0] for _ in range(bins)]
    for probability, happened in records:
        index = min(bins - 1, int(probability * bins))
        buckets[index][0] += 1
        buckets[index][1] += probability
        buckets[index][2] += float(bool(happened))
    result = []
    for index, (count, predicted_sum, observed_sum) in enumerate(buckets):
        if count:
            result.append(
                {
                    "range": f"{index / bins:.0%} - {(index + 1) / bins:.0%}",
                    "predicted_rate": predicted_sum / count,
                    "observed_rate": observed_sum / count,
                    "samples": count,
                }
            )
    return result
