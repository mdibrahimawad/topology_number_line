from collections import Counter

import pytest

from numzig.taskrules.data import (
    COPY_DEMOS, FOUR_DIGIT_DEMOS, TASKS, four_digit_targets, make_dataset,
    number_words, pilot_records, transform_digits,
)


def test_digit_permutations_preserve_zeroes_and_every_character():
    assert transform_digits('1234', 'copy4') == '1234'
    assert transform_digits('1234', 'reverse4') == '4321'
    assert transform_digits('1234', 'swap_first4') == '2134'
    assert transform_digits('1234', 'swap_last4') == '1243'
    assert transform_digits('1200', 'reverse4') == '0021'
    assert transform_digits('1020', 'swap_first4') == '0120'
    for text in ('1000', '1010', '9999', '3402'):
        for task in TASKS[:4]:
            output = transform_digits(text, task)
            assert len(output) == 4 and Counter(output) == Counter(text)
    with pytest.raises(ValueError):
        transform_digits('123', 'reverse4')


def test_number_words_boundaries_and_convention():
    expected = {0: 'zero', 1: 'one', 19: 'nineteen', 20: 'twenty', 21: 'twenty-one',
                40: 'forty', 100: 'one hundred', 101: 'one hundred and one',
                234: 'two hundred and thirty-four', 999: 'nine hundred and ninety-nine',
                1000: 'one thousand'}
    assert {n: number_words(n) for n in expected} == expected
    words = [number_words(n) for n in range(1, 1001)]
    assert len(set(words)) == 1000
    assert all(word == word.lower() and word.isascii() for word in words)
    with pytest.raises(ValueError):
        number_words(1001)


def test_paired_dataset_coverage_contexts_prompts_and_labels():
    rows = make_dataset()
    assert len(rows) == 12000
    assert rows == make_dataset()
    assert [r['point_id'] for r in rows] == list(range(len(rows)))
    groups = {(task, context): [r for r in rows if r['task'] == task and r['context_id'] == context]
              for task in TASKS for context in (0, 1)}
    reference = [r['target'] for r in groups['copy4', 0]]
    assert len(set(reference)) == 1000 and reference[0] == 1000 and reference[-1] == 9999
    counts = Counter(str(n)[0] for n in reference)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert not set(reference).intersection(sum(FOUR_DIGIT_DEMOS, ()))
    for (task, context), group in groups.items():
        assert len(group) == 1000
        assert [r['target'] for r in group] == (reference if task.endswith('4') else list(range(1, 1001)))
        demos = FOUR_DIGIT_DEMOS[context] if task.endswith('4') else COPY_DEMOS[context]
        assert len(demos) == len(set(demos)) == 8
        assert all(r['demo_values'] == list(demos) for r in group)
        assert all(r['prompt'].endswith(r['input_text'] + '=') for r in group)
        assert all(len(r['prompt'].split(',')) == 9 for r in group)
        for r in group:
            assert r['source_digits'] == list(map(int, str(r['target'])))
            assert r['leading_digit'] == int(str(r['target'])[0])
            if task.endswith('4'):
                assert r['expected_output'] == transform_digits(r['input_text'], task)
                assert r['output_digits'] == list(map(int, r['expected_output']))
            else:
                assert r['output_digits'] == r['source_digits']
                assert r['expected_output'] == r['input_text']
                assert r['demonstration_equals_target'] == (r['target'] in demos)
    for context in (0, 1):
        for numeric, word in zip(groups['numeric_copy', context], groups['word_copy', context]):
            assert numeric['target'] == word['target']
            assert numeric['demo_values'] == word['demo_values']
            assert word['input_text'] == number_words(numeric['target'])


def test_pilot_is_unique_original_subset_and_revision_preserves_pairing():
    rows = make_dataset()
    pilot = pilot_records(rows)
    assert len(pilot) == 384
    assert len({r['point_id'] for r in pilot}) == len(pilot)
    for row in pilot:
        assert row is rows[row['point_id']]
    for task in TASKS:
        for context in (0, 1):
            selected = [r for r in pilot if (r['task'], r['context_id']) == (task, context)]
            full = [r for r in rows if (r['task'], r['context_id']) == (task, context)]
            assert len(selected) == 32
            assert selected[0] is full[0] and selected[-1] is full[-1]
            assert any('0' in str(r['target']) for r in selected)
    revised = make_dataset(prompt_style='instruction')
    for original, revision in zip(rows, revised):
        assert revision['prompt'].endswith('\n' + original['prompt'])
        assert all(revision[key] == original[key] for key in original if key not in ('prompt', 'prompt_style'))
    small = make_dataset(n_targets=9)
    assert len(small) == 4072
    assert four_digit_targets(10) == sorted(four_digit_targets(10))
    assert set(four_digit_targets(100)).issubset(four_digit_targets(1000))
    with pytest.raises(ValueError):
        pilot_records(rows, 0)
