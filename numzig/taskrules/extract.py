"""Pinned cached models and query-state/behavior checks; never downloads assets."""
from pathlib import Path
import copy
import inspect
import re
import time
import numpy as np
from numzig.fullrange.storage import digest
from numzig.multimodel.backend import forward, compare, padded_inputs
from numzig.multimodel import MODELS

CRYSTAL_REVISION = '34fc9cd58acd87002560379a95b432147cc9135a'
ALIASES = {'starcoder': 'starcoderbase-3b', 'openllama': 'openllama-3b', 'llm360-crystal': 'crystal'}

def key_for(model_key):
    key = ALIASES.get(model_key, model_key)
    if key not in ('crystal', *MODELS): raise ValueError(f'Unsupported model: {model_key}')
    return key

def load_model(model_key, cache_root='/cache/huggingface'):
    """Load exact cached snapshots onto CUDA; coordinator must warm them first."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    key = key_for(model_key)
    if not torch.cuda.is_available(): raise RuntimeError('Pretrained extraction requires CUDA')
    torch.set_num_threads(2); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    repo = 'LLM360/Crystal' if key == 'crystal' else MODELS[key]['model_id']
    revision = CRYSTAL_REVISION if key == 'crystal' else MODELS[key]['revision']
    hub_cache = str(Path(cache_root) / 'hub')
    snapshot = snapshot_download(repo, revision=revision, cache_dir=hub_cache, local_files_only=True)
    started = time.perf_counter()
    if key == 'crystal':
        wrapper = snapshot_download('LLM360/CrystalCoder', revision=revision, cache_dir=hub_cache, local_files_only=True)
        cls = get_class_from_dynamic_module('tokenization_crystalcoder_fast.CrystalCoderTokenizerFast', wrapper, local_files_only=True)
        tokenizer = cls.from_pretrained(snapshot, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(snapshot, trust_remote_code=True,
            torch_dtype=torch.bfloat16, local_files_only=True).to('cuda').eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(snapshot, use_fast=MODELS[key]['use_fast'], trust_remote_code=False, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(snapshot, trust_remote_code=False,
            torch_dtype=torch.float32, attn_implementation='eager', local_files_only=True).to('cuda').eval()
    if (key == 'openllama-3b' and tokenizer.is_fast) or (key == 'crystal' and not tokenizer.is_fast):
        raise ValueError('Pinned tokenizer implementation mismatch')
    model._taskrules_validated_batch_sizes = {1}
    levels, dimension = int(model.config.num_hidden_layers)+1, int(model.config.hidden_size)
    meta = dict(model_key=key, model_id=repo, revision=revision, expected_levels=levels,
        expected_hidden_dim=dimension, model_loads=1, load_seconds=time.perf_counter()-started,
        dtype=str(next(model.parameters()).dtype), saved_dtype='float32', tokenizer_class=type(tokenizer).__name__,
        is_fast=tokenizer.is_fast, gpu=torch.cuda.get_device_name(), tf32=False, eval=True,
        special_token_policy=('native Crystal add_special_tokens=True' if key == 'crystal' else
                             'exactly one BOS' if key == 'openllama-3b' else 'no inserted BOS/EOS'),
        inference_source_sha256=digest(__file__), model_source_sha256=digest(inspect.getfile(type(model))),
        hidden_state_semantics=('L0: token plus learned absolute position embedding' if key == 'starcoderbase-3b'
                               else 'L0: token embedding (Crystal includes native embedding scale)') +
            f'; L1..L{levels-2}: block residuals; L{levels-1}: final block after native ' +
            ('RMSNorm' if key == 'openllama-3b' else 'LayerNorm'),
        extraction='final real prompt token ending in equals, before generated answer',
        output_logits_scale=float(getattr(model, 'output_logits_scale', 1.)))
    return model, tokenizer, meta

def expected_text(row):
    for name in ('expected_output', 'expected', 'output_text', 'answer', 'output'):
        if name in row:
            value = row[name]
            if not isinstance(value, str) or not value: raise ValueError('Expected output must be a nonempty string, preserving leading zeros')
            return value
    raise ValueError('Missing explicit expected_output')

def tokenize_rows(tokenizer, rows, model_key):
    key = key_for(model_key)
    def encode(text):
        ids = list(tokenizer(text, add_special_tokens=(key == 'crystal'))['input_ids'])
        if key == 'openllama-3b':
            if tokenizer.bos_token_id is None or tokenizer.bos_token_id in ids: raise ValueError('Missing or duplicate BOS')
            ids = [tokenizer.bos_token_id] + ids
        return ids
    result = []
    for raw in rows:
        row = copy.deepcopy(raw); prompt, expected = row['prompt'], expected_text(row)
        if not prompt.endswith('='): raise ValueError('Query must end at meaningful equals')
        ids = encode(prompt)
        last = tokenizer.decode(ids[-1:], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if ids[-1] in tokenizer.all_special_ids or not last.endswith('=') or decoded.strip() != prompt.strip():
            raise ValueError('Tokenizer changed prompt or final meaningful token')
        full = encode(prompt + expected); prefix = full[:len(ids)] == ids
        continuation = full[len(ids):] if prefix else None
        row.update(expected_output=expected, input_token_ids=ids, token_ids=ids, token_count=len(ids),
            extraction_position=len(ids)-1, final_token_id=ids[-1], final_token_text=last,
            final_equals_shares_token=(last != '='), decoded_input=decoded,
            continuation_prefix_valid=prefix, continuation_token_ids=continuation,
            first_continuation_token=(continuation[0] if continuation else None))
        result.append(row)
    return result

def _pad_id(tokenizer):
    return tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

def _check_context(model, rows, extra=0):
    capacity = getattr(model.config, 'max_position_embeddings', getattr(model.config, 'n_positions', None))
    if capacity and any(len(r['input_token_ids'])+extra > capacity for r in rows):
        raise ValueError('Native context capacity exceeded; truncation forbidden')

def infer_rows(model, tokenizer, rows, model_key, batch_size=1):
    """One native full-LM forward per prompt; states and logits from that call.

    Query-only head GEMMs are intentionally excluded: changing the vocabulary
    projection's matrix shape can round differently even when states are exact.
    """
    import torch
    key = key_for(model_key)
    if not rows: raise ValueError('At least one row required')
    if batch_size != 1: raise ValueError('Current task extraction requires one unpadded full-LM forward per prompt')
    records = tokenize_rows(tokenizer, rows, key); _check_context(model, records); model.eval()
    hidden = []
    with torch.inference_mode():
        for row in records:
            inputs = padded_inputs([row], _pad_id(tokenizer), next(model.parameters()).device)
            result = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            states = torch.stack([state[0, row['extraction_position']] for state in result.hidden_states]).float().cpu().numpy()
            expected_shape = (int(model.config.num_hidden_layers)+1, int(model.config.hidden_size))
            if states.shape != expected_shape or not np.isfinite(states).all(): raise ValueError('Invalid native hidden states')
            hidden.append(states)
            logits = result.logits[0, row['extraction_position']].float()
            if not torch.isfinite(logits).all(): raise ValueError('Nonfinite native query logits')
            p = torch.softmax(logits, dim=-1); top = int(logits.argmax()); expected = row['first_continuation_token']
            row.update(next_token_id=top,
                next_token_text=tokenizer.decode([top], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                next_token_probability=float(p[top]),
                expected_first_token_probability=float(p[expected]) if expected is not None else None,
                expected_first_token_correct=(top == expected if expected is not None else None),
                logits_source='same native full-LM forward as hidden states; no separate head projection')
    return np.stack(hidden, axis=1), records

def equivalence_gate(model, tokenizer, rows, model_key, batch_sizes=(1,)):
    """Fresh-prompt bitwise comparison with single unpadded full-LM reference."""
    import torch
    key = key_for(model_key)
    if not rows or len({r['point_id'] for r in rows}) != len(rows): raise ValueError('Distinct gate point IDs required')
    if tuple(batch_sizes) != (1,): raise ValueError('Current run supports only serial full-LM extraction')
    records = tokenize_rows(tokenizer, rows, key); _check_context(model, records); model.eval()
    pad = _pad_id(tokenizer)
    reference = np.concatenate([forward(model, [r], pad, backbone=False) for r in records], axis=1)
    ids = np.array([r['point_id'] for r in records])
    candidate, measured = infer_rows(model, tokenizer, rows, key)
    check = compare(reference, candidate, ids)
    check['bitwise_equal'] = bool(np.array_equal(reference, candidate))
    check['passed'] = check['passed'] and check['bitwise_equal']
    checks = [dict(batch_size=1, **check)]
    logit_checks = []
    with torch.inference_mode():
        for i, row in enumerate(records):
            inputs = padded_inputs([row], pad, next(model.parameters()).device)
            full = model(**inputs, use_cache=False, return_dict=True).logits[0,-1].float()
            probability = torch.softmax(full, dim=-1); top = int(full.argmax())
            expected = records[i]['first_continuation_token']
            argmax_equal = top == measured[i]['next_token_id']
            probability_equal = float(probability[top]) == measured[i]['next_token_probability']
            expected_equal = expected is None or float(probability[expected]) == measured[i]['expected_first_token_probability']
            logit_checks.append(dict(passed=argmax_equal and probability_equal and expected_equal,
                argmax_equal=argmax_equal, next_probability_exact=probability_equal, expected_probability_exact=expected_equal))
    passed = all(c['passed'] for c in checks) and all(c['passed'] for c in logit_checks)
    report = dict(passed=passed, checks=checks, logit_checks=logit_checks, points=len(records),
        scope='selected new prompts; serial full-LM states bitwise and native probabilities exact',
        validated_batch_sizes=list(batch_sizes) if passed else [])
    if not passed: raise ValueError(f'New-task equivalence failed: {report}')
    model._taskrules_validated_batch_sizes = set(batch_sizes)
    return report

def generated_answer(tokenizer, prompt_ids, generated_ids):
    """Contextual decode; full-text tokenization need not preserve a prefix."""
    before = tokenizer.decode(prompt_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    after = tokenizer.decode(generated_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    valid = after.startswith(before)
    text = after[len(before):] if valid else tokenizer.decode(generated_ids[len(prompt_ids):],skip_special_tokens=True,clean_up_tokenization_spaces=False)
    boundary = re.search(r'[\n\r,]',text)
    return (text[:boundary.start()] if boundary else text).strip(), text, valid, bool(boundary)

def behavior(model, tokenizer, rows, model_key, max_new_tokens=None):
    """Bounded greedy cached generation; exact accuracy is not teacher forced."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    records=tokenize_rows(tokenizer,rows,model_key); model.eval(); output=[]
    for row in records:
        cap=32 if any(c.isalpha() for c in row['expected_output']) else 8
        limit=cap if max_new_tokens is None else int(max_new_tokens)
        if not 1<=limit<=cap: raise ValueError(f'Generation limit must be 1..{cap}')
        _check_context(model,[row],extra=limit); prompt_ids=row['input_token_ids']
        class DelimiterStop(StoppingCriteria):
            def __call__(self,input_ids,scores,**kwargs):
                return generated_answer(tokenizer,prompt_ids,input_ids[0].tolist())[3]
        ids=torch.tensor([prompt_ids],dtype=torch.long,device=next(model.parameters()).device)
        with torch.inference_mode():
            result=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),do_sample=False,
                num_beams=1,max_new_tokens=limit,use_cache=True,pad_token_id=_pad_id(tokenizer),
                eos_token_id=tokenizer.eos_token_id,stopping_criteria=StoppingCriteriaList([DelimiterStop()]))
        generated=result[0].tolist(); answer,text,valid,boundary=generated_answer(tokenizer,prompt_ids,generated)
        new_ids=generated[len(prompt_ids):]
        eos=bool(new_ids and tokenizer.eos_token_id is not None and new_ids[-1]==tokenizer.eos_token_id)
        output.append(dict(**row,generated_token_ids=new_ids,generated_text=text,predicted_output=answer,
            exact_correct=valid and answer==row['expected_output'],exact_match=valid and answer==row['expected_output'],generation_decode_prefix_valid=valid,
            stopped_at_delimiter=boundary,stopped_at_eos=eos,hit_token_limit=len(new_ids)>=limit and not boundary and not eos,
            max_new_tokens=limit,decoding='greedy; one unpadded prompt; native KV cache'))
    return output
