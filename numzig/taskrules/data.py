"""Paired stimuli for digit permutations and number-word copying."""
from __future__ import annotations

from itertools import zip_longest
import random


TASKS = ('copy4', 'reverse4', 'swap_first4', 'swap_last4', 'numeric_copy', 'word_copy')
FOUR_DIGIT_DEMOS = (
    (1379, 4820, 6053, 7248, 9186, 3502, 8614, 2097),
    (2468, 9130, 5072, 3694, 8521, 4306, 7815, 1029),
)
COPY_DEMOS = (
    (7, 46, 234, 817, 3, 92, 506, 681),
    (9, 58, 361, 926, 2, 74, 405, 713),
)
INSTRUCTIONS = {
    'copy4': 'Copy the four digits in their original order. Keep every digit, including zeroes.',
    'reverse4': 'Reverse the order of the four digits. Keep every digit, including leading zeroes.',
    'swap_first4': 'Swap the first two digits and keep the last two unchanged. Keep every digit, including leading zeroes.',
    'swap_last4': 'Swap the last two digits and keep the first two unchanged. Keep every digit, including zeroes.',
    'numeric_copy': 'Copy the number exactly as written.',
    'word_copy': 'Copy the number words exactly as written.',
}


def number_words(n: int) -> str:
    """British English, lowercase, hyphenated tens; the target range is 1..1000."""
    if not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= 1000:
        raise ValueError('number_words supports integers from 0 through 1000')
    small = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
             'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen',
             'sixteen', 'seventeen', 'eighteen', 'nineteen')
    tens = ('', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety')
    if n < 20:
        return small[n]
    if n < 100:
        return tens[n // 10] + ('-' + small[n % 10] if n % 10 else '')
    if n < 1000:
        return small[n // 100] + ' hundred' + (' and ' + number_words(n % 100) if n % 100 else '')
    return 'one thousand'


def transform_digits(text: str, task: str) -> str:
    if len(text) != 4 or any(c not in '0123456789' for c in text):
        raise ValueError('digit permutation inputs must contain exactly four ASCII digits')
    if task == 'copy4':
        return text
    if task == 'reverse4':
        return text[::-1]
    if task == 'swap_first4':
        return text[1] + text[0] + text[2:]
    if task == 'swap_last4':
        return text[:2] + text[3] + text[2]
    raise ValueError(f'Unknown digit task: {task}')


def four_digit_targets(n_targets: int = 1000, seed: int = 42) -> list[int]:
    """Balanced leading-digit strata, fixed boundaries, and disjoint demonstrations."""
    excluded = set(sum(FOUR_DIGIT_DEMOS, ()))
    if not isinstance(n_targets, int) or isinstance(n_targets, bool) or not 1 <= n_targets <= 9000 - len(excluded):
        raise ValueError(f'n_targets must be between 1 and {9000 - len(excluded)}')
    rng = random.Random(seed)
    strata = []
    for digit in range(1, 10):
        candidates = [n for n in range(digit * 1000, (digit + 1) * 1000) if n not in excluded]
        rng.shuffle(candidates)
        boundary = 1000 if digit == 1 else 9999 if digit == 9 else None
        if boundary is not None:
            candidates.remove(boundary)
            candidates.insert(0, boundary)
        strata.append(candidates)
    ordered = [n for row in zip_longest(*strata) for n in row if n is not None]
    return sorted(ordered[:n_targets])


def make_dataset(n_targets: int = 1000, seed: int = 42, prompt_style: str = 'examples') -> list[dict]:
    """Two contexts per target; n_targets affects the four-digit cohort only.

    Numeric/word copying always covers every integer 1..1000. Their demonstration
    collisions are retained and flagged, so later analyses can exclude them.
    """
    if prompt_style not in ('examples', 'instruction'):
        raise ValueError('prompt_style must be examples or instruction')
    targets4 = four_digit_targets(n_targets, seed)
    records = []
    for task in TASKS:
        digit_task = task.endswith('4')
        targets = targets4 if digit_task else range(1, 1001)
        contexts = FOUR_DIGIT_DEMOS if digit_task else COPY_DEMOS
        for context_id, demos in enumerate(contexts):
            def pair(n):
                text = number_words(n) if task == 'word_copy' else str(n)
                return text, transform_digits(text, task) if digit_task else text

            prefix = ','.join(f'{a}={b}' for a, b in map(pair, demos)) + ','
            if prompt_style == 'instruction':
                prefix = INSTRUCTIONS[task] + '\n' + prefix
            for target in targets:
                input_text, expected_output = pair(target)
                source_digits = [int(c) for c in str(target)]
                # Word-copy labels describe its underlying numerical value,
                # rather than characters or tokens in the written English.
                output_digits = [int(c) for c in (expected_output if digit_task else str(target))]
                records.append(dict(
                    point_id=len(records), task=task, context_id=context_id, target=target,
                    input_text=input_text, expected_output=expected_output,
                    prompt=prefix + input_text + '=', demo_values=list(demos),
                    source_digits=source_digits, output_digits=output_digits,
                    leading_digit=source_digits[0], digit_count=len(source_digits),
                    expected_leading_digit=output_digits[0],
                    demonstration_equals_target=target in demos,
                    transformation_changed=input_text != expected_output,
                    output_starts_zero=expected_output.startswith('0'),
                    prompt_style=prompt_style, seed=seed,
                ))
    return records


def pilot_records(dataset: list[dict], n_per_group: int = 32) -> list[dict]:
    """Evenly spaced saved stimuli in each task/context, retaining original IDs."""
    if not isinstance(n_per_group, int) or isinstance(n_per_group, bool) or n_per_group < 1:
        raise ValueError('n_per_group must be a positive integer')
    groups = {}
    for row in dataset:
        groups.setdefault((row['task'], row['context_id']), []).append(row)
    selected = []
    for rows in groups.values():
        count = min(n_per_group, len(rows))
        indices = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)] if count > 1 else [0]
        # Include a zero-containing source when available, preserving count and
        # the first/last boundary examples.
        if count > 2 and not any('0' in str(rows[i]['target']) for i in indices):
            zero = next((i for i, row in enumerate(rows) if '0' in str(row['target'])), None)
            if zero is not None:
                replacement = min(range(1, len(indices) - 1), key=lambda j: abs(indices[j] - zero))
                indices[replacement] = zero
        selected.extend(rows[i] for i in sorted(indices))
    return selected
