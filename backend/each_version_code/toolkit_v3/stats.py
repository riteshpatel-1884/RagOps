import random
from statistics import mean

from .config import BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED


def percentile(values, p):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p / 100
    lo, hi = int(pos), min(int(pos) + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def summarize(values, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Mean with a 95% bootstrap confidence interval."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    m = mean(vals)
    if len(vals) < 2:
        return {"mean": m, "ci_low": m, "ci_high": m, "n": len(vals)}
    rng = random.Random(seed)
    boots = sorted(mean(rng.choices(vals, k=len(vals))) for _ in range(resamples))
    return {
        "mean": m,
        "ci_low": boots[int(0.025 * resamples)],
        "ci_high": boots[int(0.975 * resamples) - 1],
        "n": len(vals),
    }


def latency_stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {"p50": percentile(vals, 50), "p95": percentile(vals, 95), "mean": mean(vals), "n": len(vals)}
