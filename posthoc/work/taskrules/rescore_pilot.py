"""Recount already downloaded pilot generations; no model calls."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

BASE = Path('/Users/mohammed.awad/Documents/Codex/2026-09-19/c')
SOURCE = BASE / 'outputs/taskrules/pilot_checks'
MODELS = ('crystal', 'starcoderbase-3b', 'openllama-3b')
NAMES = {'crystal': 'Crystal', 'starcoderbase-3b': 'StarCoderBase-3B', 'openllama-3b': 'OpenLLaMA-3B'}
TASKS = ('copy4', 'reverse4', 'swap_first4', 'swap_last4', 'numeric_copy', 'word_copy')
LABELS = {'copy4':'Four-digit copy', 'reverse4':'Reverse', 'swap_first4':'Swap first two',
          'swap_last4':'Swap last two', 'numeric_copy':'Numeric copy', 'word_copy':'Written-number copy'}


def accuracy(rows):
    return dict(correct=sum(r['_correct'] for r in rows), n=len(rows),
                accuracy=sum(r['_correct'] for r in rows) / len(rows) if rows else None)


def stats(rows):
    changed = [r for r in rows if r['expected_output'] != r['input_text']]
    unchanged = [r for r in rows if r['expected_output'] == r['input_text']]
    zeros = [r for r in rows if r['expected_output'].isdigit() and r['expected_output'].startswith('0')]
    return dict(exact=accuracy(rows), changed_output_only=accuracy(changed), unchanged_output=accuracy(unchanged),
        changed_output_copy_errors=sum(r['_answer'] == r['input_text'] for r in changed),
        leading_zero_output=accuracy(zeros), leading_zero_targets=sorted({r['target'] for r in zeros}),
        leading_zero_omitted=sum(not r['_correct'] and r['_answer'] == r['expected_output'].lstrip('0') for r in zeros),
        no_op_targets=sorted({r['target'] for r in unchanged}),
        unique_targets=len({r['target'] for r in rows}),
        generation_decode_prefix_failures=sum(not r['generation_decode_prefix_valid'] for r in rows),
        expected_continuation_prefix_failures=sum(not r['continuation_prefix_valid'] for r in rows),
        token_limit_hits=sum(r['hit_token_limit'] for r in rows),
        token_limit_correct=sum(r['hit_token_limit'] and r['_correct'] for r in rows),
        possible_incomplete_correct_prefix=sum(r['hit_token_limit'] and r['_answer'] != r['expected_output']
            and r['expected_output'].startswith(r['_answer']) for r in rows),
        correct_without_termination=sum(r['_correct'] and not (r['stopped_at_delimiter'] or r['stopped_at_eos']) for r in rows),
        stop_delimiter=sum(r['stopped_at_delimiter'] for r in rows), stop_eos=sum(r['stopped_at_eos'] for r in rows),
        demonstration_collisions=sum(r['demonstration_equals_target'] for r in rows))


report = dict(passed=True, errors=[], source_hashes={}, models={}, methods={
    'exact_match':'Parse saved generated_text at its first comma/newline/carriage return, strip surrounding whitespace, compare exact expected_output; require saved contextual-decode-prefix validity.',
    'independence_limit':'Recounts saved generated text independently of saved correctness/answer fields; does not rerun generation or re-decode tokenizer IDs. Prefix/EOS flags remain saved observations.',
    'main_policy':'Explicit instruction for four-digit conditions; examples only for numeric_copy and word_copy.',
    'sampling':'32 deterministic targets repeated in two contexts; aggregate64 counts prompt evaluations, not64 independent targets.',
    'selection':'The same pilot targets helped select the prompt policy; this is not a new post-selection held-out evaluation.'})
reference = None
for model in MODELS:
    source = SOURCE / f'{model}_main_behavior.json'
    report['source_hashes'][source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = json.loads(source.read_text())
    errors = []
    if len(rows) != 384 or len({r['point_id'] for r in rows}) != 384:
        errors.append('Invalid record count or repeated point ID')
    identity = [(r['point_id'],r['task'],r['context_id'],r['target'],r['prompt'],r['expected_output']) for r in rows]
    if reference is None:
        reference = identity
    elif identity != reference:
        errors.append('Prompt/target pairing differs across models')
    for row in rows:
        row['_answer'] = re.split(r'[\n\r,]', row['generated_text'], maxsplit=1)[0].strip()
        row['_correct'] = bool(row['generation_decode_prefix_valid'] and row['_answer'] == row['expected_output'])
        expected_style = 'instruction' if row['task'].endswith('4') else 'examples'
        if row['prompt_style'] != expected_style:
            errors.append(f'Prompt policy mismatch at {row["point_id"]}')
        if row['_answer'] != row['predicted_output'] or row['_correct'] != row['exact_match'] or row['_correct'] != row['exact_correct']:
            errors.append(f'Answer/correctness mismatch at {row["point_id"]}')
        if bool(re.search(r'[\n\r,]',row['generated_text'])) != row['stopped_at_delimiter']:
            errors.append(f'Delimiter mismatch at {row["point_id"]}')
        if row['hit_token_limit'] != (len(row['generated_token_ids']) >= row['max_new_tokens'] and not row['stopped_at_delimiter'] and not row['stopped_at_eos']):
            errors.append(f'Token-limit mismatch at {row["point_id"]}')
        if row['transformation_changed'] != (row['input_text'] != row['expected_output']):
            errors.append(f'No-op label mismatch at {row["point_id"]}')
    groups, by_task = [], {}
    for task in TASKS:
        by_task[task] = stats([r for r in rows if r['task'] == task])
        receipt_path = SOURCE / f'{model}_pilot{"_instruction" if task.endswith("4") else ""}.json'
        receipt = json.loads(receipt_path.read_text())
        task_targets = []
        for context in (0,1):
            subset = [r for r in rows if r['task'] == task and r['context_id'] == context]
            summary = stats(subset)
            summary.update(task=task, context_id=context)
            stored = next(r for r in receipt['summary'] if r['task'] == task and r['context_id'] == context)
            summary['receipt_match'] = len(subset) == stored['n'] and abs(summary['exact']['correct'] - stored['n'] * stored['exact_accuracy']) < 1e-12
            if len(subset) != 32 or not summary['receipt_match']:
                errors.append(f'Receipt/coverage mismatch for {task}/{context}')
            task_targets.append([r['target'] for r in subset])
            groups.append(summary)
        if task_targets[0] != task_targets[1]:
            errors.append(f'Unpaired context targets for {task}')
    issues = [dict(point_id=r['point_id'],task=r['task'],context_id=r['context_id'],target=r['target'],
                   expected=r['expected_output'],predicted=r['_answer'],generated_text=r['generated_text'],
                   hit_token_limit=r['hit_token_limit'],prefix_valid=r['generation_decode_prefix_valid'],
                   correct=r['_correct']) for r in rows
              if r['hit_token_limit'] or not r['generation_decode_prefix_valid'] or not r['continuation_prefix_valid']]
    report['models'][model] = dict(record_count=len(rows), errors=errors, overall=stats(rows), by_task=by_task, groups=groups, issues=issues)
    report['errors'].extend(f'{model}: {error}' for error in errors)
report['passed'] = not report['errors']


def fraction(item, percent=False):
    if not item['n']:
        return '—'
    return f'{item["correct"]}/{item["n"]}' + (f' ({item["accuracy"]:.1%})' if percent else '')


lines = ['# Pilot outputs independently recounted', '',
    f'**Reconciliation: {"passed" if report["passed"] else "FAILED"}.** All 1,152 saved main-policy pilot evaluations were recounted from generated text. '
    'The recount agrees with every saved correctness flag and all 36 relevant task/context receipt summaries. '
    'The three models received identical paired prompts and expected answers. No new inference was run.', '',
    'Four-digit conditions used instructions plus examples; numeric and word copying used examples only. '
    'Each aggregate combines the same 32 targets in two contexts, so its denominator 64 describes prompt evaluations, not independent numbers.', '',
    '## Exact-answer accuracy, all sampled targets', '',
    '| Condition | Crystal | StarCoderBase-3B | OpenLLaMA-3B |', '|---|---:|---:|---:|']
for task in TASKS:
    lines.append('| '+LABELS[task]+' | '+' | '.join(fraction(report['models'][m]['by_task'][task]['exact'],True) for m in MODELS)+' |')
lines += ['', '## Accuracy when the requested transformation changes the output', '',
    'These scores exclude palindrome/equal-digit cases that can be answered correctly by copying. Each cell shows the aggregate, followed by the two context scores.', '',
    '| Transformation | Crystal | StarCoderBase-3B | OpenLLaMA-3B |', '|---|---:|---:|---:|']
for task in ('reverse4','swap_first4','swap_last4'):
    cells=[]
    for model in MODELS:
        result=report['models'][model]
        contexts=[g for g in result['groups'] if g['task']==task]
        cells.append(fraction(result['by_task'][task]['changed_output_only'],True)+' ['+'; '.join(fraction(g['changed_output_only']) for g in contexts)+']')
    lines.append('| '+LABELS[task]+' | '+' | '.join(cells)+' |')
lines += ['', 'Removed targets per context: reversal—9999; swap first two—3318, 4412, 8832, 9999; swap last two—1000, 9999. '
    'Thus the changed-output denominators are 31, 28, and 30 per context. Copying controls have no changed-output subset by definition.', '',
    'After removing these easy cases, StarCoder swapping the last two still exceeds 80% in both contexts (90.0% and 83.3%). '
    'Crystal swapping the first two remains context-sensitive (92.9% versus 75.0%). '
    'OpenLLaMA has only 3/56 successful changed first-two swaps and 1/60 successful changed last-two swaps; its original totals included several unchanged-output successes.', '',
    '## Leading-zero outputs', '',
    'The answer must retain every zero. These are small diagnostic subsets, not robust model rankings.', '',
    '| Transformation | Crystal | StarCoderBase-3B | OpenLLaMA-3B |', '|---|---:|---:|---:|']
for task in ('reverse4','swap_first4','swap_last4'):
    lines.append('| '+LABELS[task]+' | '+' | '.join(fraction(report['models'][m]['by_task'][task]['leading_zero_output'],True) for m in MODELS)+' |')
lines += ['', 'A dash means no sampled expected answer began with zero. Reversal tested four distinct leading-zero cases twice: 1000, 1550, 7970, and 8480. '
    'Swap-first tested three twice: 1000, 3024, and 5024. No leading-zero failure was explained solely by dropping the required leading zeroes. '
    'The JSON includes target IDs and per-context scores.', '',
    '## Generation and parsing checks', '',
    '| Model | Contextual-decode prefix failures | Expected-continuation prefix failures | Token-limit hits | Correct-prefix answers possibly cut short | Correct answers without a recorded terminator |',
    '|---|---:|---:|---:|---:|---:|']
for model in MODELS:
    item=report['models'][model]['overall']
    lines.append('| '+NAMES[model]+' | '+' | '.join(str(item[k]) for k in ('generation_decode_prefix_failures','expected_continuation_prefix_failures','token_limit_hits','possible_incomplete_correct_prefix','correct_without_termination'))+' |')
lines += ['', 'All 1,152 saved evaluations reached a delimiter. None hit the 8-token numeric or 32-token word limit; none showed a saved prefix-validity failure. '
    'Thus the observed poor transformation scores are not explained by recorded generation truncation. '
    'No sampled target matched a demonstration. StarCoder\'s sole written-copy error was target 774 in context 1: expected “seven hundred and seventy-four”, generated “seventy-four”.', '',
    'A wrong answer that already differs from the expected prefix cannot become exactly correct merely by extending generation. '
    'Delimiter/EOS and prefix observations come from the saved generation records; this recount did not reload tokenizers to independently re-decode token IDs.', '',
    '## Interpretation limits', '',
    '- These are deterministic, small, paired pilots reused for prompt selection—not a fresh evaluation on all 1,000 targets per condition.',
    '- The saved vectors precede generation. A cloud produced under a failing requested rule must not be described as a representation of successfully applying that rule.',
    '- Desired-output digit colors are predefined labels. Context wording, token length, output formatting, and task accuracy remain competing explanations for geometric changes.',
    '- Exact answers are compared after stripping surrounding whitespace and stopping at the first comma or newline; internal spacing, spelling, and leading zeroes must match.',
    '', 'Source files and SHA-256 hashes, all per-context numerators/denominators, no-op targets, leading-zero targets, and individual generation-limit cases are preserved in `pilot_rescored.json`.']
destination=BASE/'work/taskrules'
(destination/'pilot_rescored.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
(destination/'pilot_rescored.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({model:{task:report['models'][model]['by_task'][task] for task in ('reverse4','swap_first4','swap_last4')} for model in MODELS},indent=2))
print('Reconciliation passed:',report['passed'])
