from __future__ import annotations

import math
import random
from typing import Callable, Iterable, Sequence

GROUP_RANGE_DEFAULT: tuple[int, ...] = (1, 2, 3, 4)
ALPHABET: str = "abcdefghijklmnopqrstuvwxyz"

# Width of each interval around 10**i (so group i covers [10**i - HALF, 10**i + HALF)).
DEFAULT_HALF_WIDTH: int = 20


def default_interval(i: int, *, half_width: int = DEFAULT_HALF_WIDTH) -> range:
    if i > 1:
        return range(10**i - half_width, 10**i + half_width)
    return range(1, 2 * half_width)


def _format_numeral_prompt(numbers: Sequence[int]) -> str:
    if not numbers:
        raise ValueError("need at least one number to build a prompt")
    body = ",".join(f"{n}={n}" for n in numbers[:-1])
    return f"{body},{numbers[-1]}=" if body else f"{numbers[-1]}="


def generate_numeral_prompts(
    *,
    k: int,
    num_examples: int,
    upper_bound: int,
    groups: Iterable[int] = GROUP_RANGE_DEFAULT,
    interval_fn: Callable[[int], range] = default_interval,
    context: str = "random",
    rng: random.Random | None = None,
) -> dict[int, list[str]]:
    rng = rng or random.Random()
    fixed_ctx = (4, 54, 432, 9543)

    sampled: dict[int, list[int]] = {
        g: [rng.choice(list(interval_fn(g))) for _ in range(k)] for g in groups
    }

    out: dict[int, list[str]] = {}
    for g, draws in sampled.items():
        out[g] = []
        for n in draws:
            if context == "random":
                ctx_nums: list[int] = [rng.randint(0, upper_bound) for _ in range(num_examples)]
            elif context == "fixed":
                ctx_nums = list(fixed_ctx[:num_examples])
            elif context == "same":
                ctx_nums = sampled[g][:num_examples]
            else:
                raise ValueError(f"unknown context option: {context!r}")
            out[g].append(_format_numeral_prompt([*ctx_nums, n]))
    return out


def generate_year_prompts(
    *,
    k: int,
    num_examples: int,
    year_min: int = 1800,
    year_max: int = 2025,
    groups: Iterable[int] = GROUP_RANGE_DEFAULT,
    context: str = "random",
    rng: random.Random | None = None,
) -> dict[int, list[str]]:
    """Sample target years from equal-width bins over [year_min, year_max]."""
    if year_max < year_min:
        raise ValueError(f"year_max must be >= year_min, got {year_min}, {year_max}")
    rng = rng or random.Random()
    groups_t = tuple(sorted(groups))
    if not groups_t:
        raise ValueError("need at least one year group")

    span = year_max - year_min + 1
    intervals: dict[int, range] = {}
    for i, g in enumerate(groups_t):
        lo = year_min + (i * span) // len(groups_t)
        hi_excl = year_min + ((i + 1) * span) // len(groups_t)
        intervals[g] = range(lo, hi_excl)

    fixed_ctx = (1800, 1900, 2000, 2020)
    sampled: dict[int, list[int]] = {
        g: [rng.choice(list(intervals[g])) for _ in range(k)] for g in groups_t
    }

    out: dict[int, list[str]] = {}
    for g, draws in sampled.items():
        out[g] = []
        for n in draws:
            if context == "random":
                ctx_nums = [rng.randint(year_min, year_max) for _ in range(num_examples)]
            elif context == "fixed":
                ctx_nums = list(fixed_ctx[:num_examples])
            elif context == "same":
                ctx_nums = sampled[g][:num_examples]
            else:
                raise ValueError(f"unknown context option: {context!r}")
            out[g].append(_format_numeral_prompt([*ctx_nums, n]))
    return out


def _random_symbol(length: int, rng: random.Random) -> str:
    return "".join(rng.choices(ALPHABET, k=length))


def generate_symbol_prompts(
    *,
    k: int,
    num_examples: int,
    groups: Iterable[int] = GROUP_RANGE_DEFAULT,
    rng: random.Random | None = None,
) -> dict[int, list[str]]:
    rng = rng or random.Random()
    groups_t = tuple(groups)
    max_g = max(groups_t)
    pool = [
        _random_symbol(rng.randint(1, max_g), rng) for _ in range(k * max(num_examples, 1))
    ]

    out: dict[int, list[str]] = {}
    for g in groups_t:
        out[g] = []
        for _ in range(k):
            ctx_syms = rng.sample(pool, num_examples)
            tail = _random_symbol(g, rng)
            body = ",".join(f"{s}={s}" for s in ctx_syms)
            out[g].append(f"{body},{tail}=" if body else f"{tail}=")
    return out


def string_to_base26(s: str) -> int:
    n = 0
    for c in s.lower():
        if not "a" <= c <= "z":
            raise ValueError(f"non-alphabetic character: {c!r}")
        n = n * 26 + (ord(c) - ord("a"))
    return n


def extract_target(prompt: str, *, alphabetic: bool) -> float:
    raw = prompt[prompt.rfind(",") + 1 : prompt.rfind("=")]
    return float(string_to_base26(raw)) if alphabetic else float(raw)


# --------------------------------------------------------------------------- #
# distribution-driven sampling (uniform / gaussian / zipf / pile)
# --------------------------------------------------------------------------- #

DISTRIBUTIONS = (
    "uniform",
    "gaussian",
    "zipf",
    "exponential_decay",
    "exponential_growth",
)


def sample_integers(
    distribution: str,
    n: int,
    *,
    lo: int = 1,
    hi: int = 1000,
    rng: random.Random | None = None,
    pile_counts: dict[int, int] | None = None,
) -> list[int]:
    """Draw `n` integers from one of {uniform, gaussian, zipf, pile} on [lo, hi].

    `gaussian` is N(mu=(lo+hi)/2, sigma=(hi-lo)/4) clipped to [lo, hi].
    `zipf` is the discrete Zipf-like distribution with weight 1/k.
    `pile` resamples in proportion to a supplied corpus-frequency dict (used
    to tie the synthetic sweep back to Stage 1 directly).
    """
    rng = rng or random.Random()
    name = distribution.lower()

    if name == "uniform":
        return [rng.randint(lo, hi) for _ in range(n)]

    if name == "gaussian":
        mu = (lo + hi) / 2.0
        sigma = (hi - lo) / 4.0
        out: list[int] = []
        guard = 0
        while len(out) < n and guard < 64 * n:
            x = int(round(rng.gauss(mu, sigma)))
            guard += 1
            if lo <= x <= hi:
                out.append(x)
        return out

    if name == "zipf":
        ks = list(range(lo, hi + 1))
        weights = [1.0 / k for k in ks]
        return rng.choices(ks, weights=weights, k=n)

    if name in ("exponential_decay", "exponential", "exp_decay"):
        # P(N) ~ exp(-lambda * (N - lo)), lambda chosen so 95% of mass in [lo, hi]
        ks = list(range(lo, hi + 1))
        lam = 3.0 / max(1, hi - lo)
        weights = [math.exp(-lam * (k - lo)) for k in ks]
        return rng.choices(ks, weights=weights, k=n)

    if name in ("exponential_growth", "exp_growth"):
        # P(N) ~ exp(+lambda * (N - lo)) -- *low* N rare, *high* N frequent
        ks = list(range(lo, hi + 1))
        lam = 3.0 / max(1, hi - lo)
        weights = [math.exp(+lam * (k - lo)) for k in ks]
        return rng.choices(ks, weights=weights, k=n)

    if name == "pile":
        if pile_counts is None:
            raise ValueError("pile distribution requires pile_counts dict")
        ks = list(range(lo, hi + 1))
        weights = [float(max(0, pile_counts.get(k, 0))) for k in ks]
        if sum(weights) == 0:
            raise ValueError("pile_counts produced an all-zero distribution")
        return rng.choices(ks, weights=weights, k=n)

    raise ValueError(f"unknown distribution: {distribution!r}")


def generate_distribution_prompts(
    distribution: str,
    *,
    n_prompts: int,
    num_examples: int,
    lo: int = 1,
    hi: int = 1000,
    context: str = "random",
    rng: random.Random | None = None,
    pile_counts: dict[int, int] | None = None,
) -> list[tuple[str, int]]:
    """Return `[(prompt, target_int), ...]` with target_int sampled from `distribution`."""
    rng = rng or random.Random()
    fixed_ctx = (4, 54, 432, 9543)

    targets = sample_integers(
        distribution, n_prompts, lo=lo, hi=hi, rng=rng, pile_counts=pile_counts,
    )

    out: list[tuple[str, int]] = []
    for n in targets:
        if context == "random":
            ctx_nums: list[int] = [rng.randint(lo, hi) for _ in range(num_examples)]
        elif context == "fixed":
            ctx_nums = list(fixed_ctx[:num_examples])
        elif context == "matched":
            ctx_nums = sample_integers(
                distribution, num_examples, lo=lo, hi=hi, rng=rng,
                pile_counts=pile_counts,
            )
        else:
            raise ValueError(f"unknown context option: {context!r}")
        out.append((_format_numeral_prompt([*ctx_nums, n]), n))
    return out
