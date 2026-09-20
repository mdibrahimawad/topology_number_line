"""Paper-matched numerical prompt sampling copied from the user's project."""
from __future__ import annotations

import random
from typing import Iterable, Sequence

GROUP_RANGE_DEFAULT: tuple[int, ...] = (1, 2, 3, 4)
DEFAULT_HALF_WIDTH = 20


def default_interval(i: int, *, half_width: int = DEFAULT_HALF_WIDTH) -> range:
    if i > 1:
        return range(10**i - half_width, 10**i + half_width)
    return range(1, 2 * half_width)


def _format_numeral_prompt(numbers: Sequence[int]) -> str:
    if not numbers:
        raise ValueError("need at least one number")
    body = ",".join(f"{n}={n}" for n in numbers[:-1])
    return f"{body},{numbers[-1]}=" if body else f"{numbers[-1]}="


def generate_numeral_prompts(
    *,
    k: int,
    num_examples: int,
    upper_bound: int,
    groups: Iterable[int] = GROUP_RANGE_DEFAULT,
    context: str = "random",
    rng: random.Random | None = None,
) -> dict[int, list[str]]:
    rng = rng or random.Random()
    fixed_ctx = (4, 54, 432, 9543)
    groups = tuple(groups)
    sampled = {g: [rng.choice(list(default_interval(g))) for _ in range(k)] for g in groups}
    out: dict[int, list[str]] = {}
    for g, draws in sampled.items():
        out[g] = []
        for n in draws:
            if context == "random":
                ctx_nums = [rng.randint(0, upper_bound) for _ in range(num_examples)]
            elif context == "fixed":
                ctx_nums = list(fixed_ctx[:num_examples])
            elif context == "same":
                ctx_nums = sampled[g][:num_examples]
            else:
                raise ValueError(f"unknown context option: {context!r}")
            out[g].append(_format_numeral_prompt([*ctx_nums, n]))
    return out


def extract_target(prompt: str) -> float:
    raw = prompt[prompt.rfind(",") + 1 : prompt.rfind("=")]
    return float(raw)


def build_ordered_prompt_records(
    *,
    samples_per_group: int,
    num_examples: int,
    upper_bound: int,
    groups: tuple[int, ...],
    context: str,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    grouped = generate_numeral_prompts(
        k=samples_per_group,
        num_examples=num_examples,
        upper_bound=upper_bound,
        groups=groups,
        context=context,
        rng=rng,
    )
    rows: list[dict] = []
    point_id = 0
    for group in sorted(grouped):
        for prompt in grouped[group]:
            rows.append(
                {
                    "point_id": point_id,
                    "group": int(group),
                    "target": float(extract_target(prompt)),
                    "prompt": prompt,
                }
            )
            point_id += 1
    return rows
