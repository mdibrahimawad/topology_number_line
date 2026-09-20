"""No pretrained weights, cloud calls or external downloads in these tests."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import copy
import numpy as np
import pytest
from numzig.multimodel import MODELS, CRYSTAL_ROOT, configuration, resources
from numzig.multimodel.data import import_text, prepare, load, token_rows, semantics
from numzig.multimodel.backend import (padded_inputs, forward, reference, batched, equivalence_gate,
                                     extract_assigned, ensure_contract, compare)
from numzig.multimodel.pipeline import recover_extraction, layer_cache, recover_analysis, analysis_worker, finalize_analysis
from numzig.fullrange.storage import Store, digest, read_json, write_json
from numzig.fullrange.parallel import make_plan, WorkerStore
from numzig.fullrange.extract import extraction_dependency, chunks, read_layer
from numzig.fullrange.analysis import analysis_dependency

class CharTokenizer:
    all_special_ids = [0, 1, 2]
    bos_token_id, eos_token_id, is_fast = 1, 2, False
    def __call__(self, text, **kwargs):
        return dict(input_ids=[ord(c) for c in text])
    def decode(self, ids, skip_special_tokens=False, **kwargs):
        return ''.join(chr(i) for i in ids if not skip_special_tokens or i not in self.all_special_ids)
    def convert_ids_to_tokens(self, ids):
        return [chr(i) for i in ids]


def fixture(path, model='openllama-3b', chunk_size=8):
    cfg = configuration(model, True); cfg['chunk_size'] = chunk_size
    store = Store(path, cfg)
    from transformers import LlamaConfig, GPTBigCodeConfig
    mc = (LlamaConfig(hidden_size=12, num_hidden_layers=2, intermediate_size=24, num_attention_heads=3)
          if model == 'openllama-3b' else GPTBigCodeConfig(n_embd=12, n_layer=3, n_head=3))
    prepare(store, CRYSTAL_ROOT, tokenizer=CharTokenizer(), model_config=mc, synthetic=True)
    ensure_contract(store)
    return store


def vectors(rows, levels=3):
    return np.stack([np.stack([np.random.default_rng(r['point_id'] + layer*10001).normal(size=12).astype(np.float32)
                    for r in rows]) for layer in range(levels)])


def run_extract(store, workers=3, fail=False):
    missing = recover_extraction(store)
    if not missing:
        return []
    plan = make_plan(store, 'chunk', missing, workers, extraction_dependency(store))
    meta = read_json(store.root / 'tokenizer_provenance.json')
    before = store.path.read_bytes()
    def job(slot):
        worker = WorkerStore(store.root, plan['id'], slot)
        count = 0
        def infer(rows):
            nonlocal count
            count += 1
            if fail and slot == 0 and count == 2:
                raise RuntimeError('interrupted extraction')
            return vectors(rows, meta['expected_levels'])
        extract_assigned(worker, infer, (meta['expected_levels'], 12))
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(job, i) for i in range(len(plan['assignments']))]
        for f in futures:
            try: f.result()
            except RuntimeError as exc: errors.append(exc)
    assert store.path.read_bytes() == before
    return errors


def checkpoint_bytes(store, prefix):
    return {p: (digest(store.root/p), (store.root/p).stat().st_mtime_ns)
            for k, v in store.manifest['artifacts'].items() if k.startswith(prefix) for p in v['files']}


def test_authoritative_text_and_independent_tokens():
    rows, h = import_text(CRYSTAL_ROOT)
    assert len(rows) == 10000 and [r['point_id'] for r in rows] == list(range(10000))
    assert rows == [{k: r[k] for k in rows[0]} for r in read_json(CRYSTAL_ROOT/'dataset.json')['records']]
    a = token_rows(rows, CharTokenizer(), configuration('starcoderbase-3b'))
    b = token_rows(rows, CharTokenizer(), configuration('openllama-3b'))
    for x, y, raw in zip(a, b, rows):
        assert x['prompt'] == y['prompt'] == raw['prompt']
        assert y['input_token_ids'] == [1] + x['input_token_ids']
        assert x['final_token_text'] == y['final_token_text'] == '='
        assert y['continuation_token_ids'] == [ord(c) for c in raw['target_string']]
    assert len(h) == 64


def test_merged_equals_and_context_prefix_mismatch():
    class Merge(CharTokenizer):
        def __call__(self, text, **kwargs):
            return dict(input_ids=[ord(c) for c in text[:-2]] + [127] if text.endswith('1=') else [ord(c) for c in text])
        def decode(self, ids, **kwargs):
            return ''.join('1=' if i == 127 else chr(i) for i in ids if i not in self.all_special_ids)
    raw = import_text(CRYSTAL_ROOT)[0][:1]
    row = token_rows(raw, Merge(), configuration('starcoderbase-3b'))[0]
    assert row['final_equals_shares_token'] and row['final_token_text'] == '1='
    assert not row['continuation_prefix_valid'] and row['continuation_token_ids'] is None


@pytest.mark.parametrize('architecture', ['llama', 'gpt_bigcode'])
def test_tiny_native_reference_batching_masks_semantics(architecture):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM, GPTBigCodeConfig, GPTBigCodeForCausalLM
    torch.set_num_threads(1); torch.manual_seed(7)
    if architecture == 'llama':
        cfg = LlamaConfig(vocab_size=128, hidden_size=24, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4, max_position_embeddings=128)
        cls, name = LlamaForCausalLM, 'openllama-3b'
    else:
        cfg = GPTBigCodeConfig(vocab_size=128, n_embd=24, n_layer=2, n_head=4, n_positions=128, multi_query=True)
        cls, name = GPTBigCodeForCausalLM, 'starcoderbase-3b'
    cfg._attn_implementation = 'eager'
    model = cls(cfg).eval()
    raw = import_text(CRYSTAL_ROOT)[0]
    rows = token_rows([raw[i] for i in [0, 4, 9, 98, 99, 998, 999, 9998, 9999]], CharTokenizer(), configuration(name))
    inputs = padded_inputs(rows, 2, 'cpu', 60)
    for i, r in enumerate(rows):
        n = len(r['input_token_ids'])
        assert inputs['attention_mask'][i].tolist() == [1]*n + [0]*(60-n)
        assert inputs['position_ids'][i].tolist() == list(range(60))
    ref = reference(model, rows, 2)
    out = batched(model, rows, 2, 8)
    assert out.shape == (3, len(rows), 24)
    np.testing.assert_allclose(ref, out, atol=2e-5, rtol=2e-5)
    gate, _ = equivalence_gate(model, rows, 2)
    assert gate['passed'], gate
    if architecture == 'llama':
        assert np.array_equal(out[0, 0], out[0, -1])
    else:
        assert not np.array_equal(out[0, 0], out[0, -1])
    info = semantics(architecture, 2, 24)
    assert len(info['levels']) == 3 and 'final-normalized' in info['levels'][-1]['state']


def test_resource_bounds_and_disjoint_models():
    for gpu in (1, 5, 10):
        for cpu in (1, 24, 31):
            assert resources(gpu, cpu)['conservative_aggregate_cpu_ceiling'] <= 100
    for values in [(11, 24, 16), (10, 32, 16), (10, 24, 17)]:
        with pytest.raises(ValueError): resources(*values)
    from numzig.fullrange.parallel import partition
    raw = import_text(CRYSTAL_ROOT)[0]
    keys = []
    for model in MODELS:
        cfg = configuration(model)
        for _, part, key in chunks(SimpleNamespace(manifest={'config': cfg}), raw):
            keys.append((model, key, tuple(r['point_id'] for r in part)))
    assignments = partition(keys, 10)
    for model in MODELS:
        assert sorted(i for worker in assignments for m,k,ids in worker if m == model for i in ids) == list(range(10000))


@pytest.mark.parametrize('model', list(MODELS))
def test_interruption_recovery_changed_workers_and_scientific_equivalence(tmp_path, model):
    s = fixture(tmp_path/'parallel', model, 4)
    assert run_extract(s, 3, fail=True)
    s = Store(s.root); missing = recover_extraction(s); assert missing
    before = checkpoint_bytes(s, 'chunk/')
    assert not run_extract(s, 2)
    s = Store(s.root); assert not recover_extraction(s)
    assert all(checkpoint_bytes(s, 'chunk/')[p] == h for p,h in before.items())
    serial = fixture(tmp_path/'serial', model, 4); run_extract(serial, 1)
    serial = Store(serial.root); recover_extraction(serial)
    shape = layer_cache(s)
    for layer in range(shape[0]):
        np.testing.assert_array_equal(read_layer(s, layer, load(s)), read_layer(serial, layer, load(serial)))
    missing = recover_analysis(s)
    plan = make_plan(s, 'layer', missing, 2, analysis_dependency(s))
    w = WorkerStore(s.root, plan['id'], 0)
    # Commit first layer, then interrupt before the second; peer may complete.
    w.assigned = w.assigned[:1]; analysis_worker(w)
    s = Store(s.root); assert recover_analysis(s)
    completed = checkpoint_bytes(s, 'layer/')
    plan = make_plan(s, 'layer', recover_analysis(s), 3, analysis_dependency(s))
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda slot: analysis_worker(WorkerStore(s.root, plan['id'], slot)), range(len(plan['assignments']))))
    s = Store(s.root); finalize_analysis(s)
    assert all(checkpoint_bytes(s, 'layer/')[p] == h for p,h in completed.items())
    layer_cache(serial)
    plan = make_plan(serial, 'layer', recover_analysis(serial), 1, analysis_dependency(serial))
    analysis_worker(WorkerStore(serial.root, plan['id'], 0)); serial = Store(serial.root); finalize_analysis(serial)
    for layer in range(shape[0]):
        with np.load(s.root/'analysis'/f'layer_{layer:02d}.npz') as a, np.load(serial.root/'analysis'/f'layer_{layer:02d}.npz') as b:
            for key in a.files: np.testing.assert_allclose(a[key], b[key], equal_nan=True, atol=1e-12)
    bad = configuration('openllama-3b' if model == 'starcoderbase-3b' else 'starcoderbase-3b', True)
    with pytest.raises(ValueError): Store(s.root, bad)


def test_gate_rejects_geometry_change_and_near_tie():
    x = np.arange(18, dtype=np.float32).reshape(1,6,3)
    y = x.copy(); y[:,1] += 1
    assert not compare(x,y,np.arange(6))['passed']
    x = np.zeros((1,6,3), dtype=np.float32); y = x.copy(); y[:,1,0] = 1e-6
    assert not compare(x,y,np.arange(6))['passed']


def test_plot_interruption_recovery_model_labels_and_packaging(tmp_path):
    from numzig.multimodel import plots
    s = fixture(tmp_path/'plots'); run_extract(s, 2)
    s = Store(s.root); recover_extraction(s); layer_cache(s)
    plan = make_plan(s, 'layer', recover_analysis(s), 1, analysis_dependency(s))
    analysis_worker(WorkerStore(s.root, plan['id'], 0)); s = Store(s.root); finalize_analysis(s)
    dep = plots.dependency(s)
    plan = plot_plan(s, plots.recover(s), 2)
    w = WorkerStore(s.root, plan['id'], 0)
    w.assigned = w.assigned[:1]
    plots.worker(w)  # Interrupted between figures; other worker not yet started.
    manifest = s.path.read_bytes()
    s = Store(s.root); assert plots.recover(s)
    done = checkpoint_bytes(s, 'figure/')
    plan = plot_plan(s, plots.recover(s), 3)
    # Matplotlib must run in separate processes, never shared threads.
    from concurrent.futures import ProcessPoolExecutor
    from numzig.multimodel.__main__ import local_plot
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(local_plot, [(str(s.root), plan['id'], i) for i in range(len(plan['assignments']))]))
    s = Store(s.root); plots.finalize(s)
    from numzig.multimodel.validation import audit
    assert audit(s)['passed']
    plots.package(s)
    assert all(checkpoint_bytes(s, 'figure/')[p] == h for p,h in done.items())
    html = (s.root/'viewer/index.html').read_text()
    assert 'OpenLLaMA-3B' in html and 'Crystal' not in html
    index = read_json(s.root/'viewer/index.json'); assert len(index['rows']) == 40 and len(index['layers']) == 3
    assert not plots.recover(s)
    before = checkpoint_bytes(s, 'lightweight_export')
    plots.package(s); assert checkpoint_bytes(s, 'lightweight_export') == before
    # A single broken output resumes alone and never touches hidden vectors.
    hidden = checkpoint_bytes(s, 'chunk/')
    path = s.root/'figures/layer_00_graph.png'; path.write_bytes(b'partial')
    with pytest.raises(ValueError, match='Missing or invalid'):
        audit(s)  # A saved validation record cannot hide later artifact corruption.
    assert plots.recover(s) == ['figure/00/graph']
    plan = plot_plan(s, plots.recover(s), 1)
    plots.worker(WorkerStore(s.root, plan['id'], 0))
    assert checkpoint_bytes(s, 'chunk/') == hidden
    s = Store(s.root); assert not plots.recover(s)
    unaffected = checkpoint_bytes(s, 'figure/01/')
    path = s.root/'analysis/layer_00.json'; metric = read_json(path); metric['seconds'] += 1
    write_json(path, metric)
    record = s.manifest['artifacts']['analysis/00']
    s.finish('analysis/00', [s.root/p for p in record['files']], record['dependency'])
    assert set(plots.recover(s)) == {*(f'figure/00/{mode}' for mode in plots.MODES), 'viewer/00'}
    assert checkpoint_bytes(s, 'figure/01/') == unaffected


@pytest.mark.parametrize('fail_at', [1,2])
def test_native_receipt_commit_interruption(tmp_path, fail_at):
    s = fixture(tmp_path/'interruption', chunk_size=4)
    plan = make_plan(s, 'chunk', recover_extraction(s), 2, extraction_dependency(s))
    commits = 0
    def commit():
        nonlocal commits
        commits += 1
        if commits == fail_at: raise RuntimeError('commit interrupted')
    w = WorkerStore(s.root, plan['id'], 0, commit)
    with pytest.raises(RuntimeError): extract_assigned(w, vectors, (3,12))
    first = plan['assignments'][0][0]
    before = {p: digest(s.root/p) for p in read_json(s.root/'receipts'/(first+'.json'))['files']}
    s = Store(s.root); recover_extraction(s)
    assert s.valid(first)
    assert all(digest(s.root/p) == h for p,h in before.items())


def test_crystal_inference_and_plot_fingerprints_untouched():
    for name in ('__init__.py', 'dataset.py', 'extract.py', 'plots.py', 'viewer.html'):
        relative = Path('numzig/fullrange')/name
        assert digest(relative) == digest(CRYSTAL_ROOT/'source'/relative)


def test_overlap_guard_uses_actual_modal_json_schema():
    from numzig.multimodel import reject_overlapping_apps
    old = dict(app_id='old', description='numeral-native-fullrange', state='stopped', tasks='0')
    reject_overlapping_apps([old], 'new')
    for state, tasks in [('stopping','0'), ('detached','2'), ('stopped','1')]:
        with pytest.raises(RuntimeError): reject_overlapping_apps([old | dict(state=state, tasks=tasks)], 'new')
    with pytest.raises(ValueError): reject_overlapping_apps([{'App ID': 'old'}], 'new')


def test_bounded_oom_halving_and_reference_reuse(monkeypatch):
    import torch
    import numzig.multimodel.backend as backend
    rows = token_rows(import_text(CRYSTAL_ROOT)[0][:9], CharTokenizer(), configuration('starcoderbase-3b'))
    calls, fallback = [], []
    def pretend(model, part, pad_id, **kwargs):
        calls.append(len(part))
        if len(part) > 2: raise torch.cuda.OutOfMemoryError('synthetic OOM')
        return vectors(part)
    monkeypatch.setattr(backend, 'forward', pretend)
    actual = backend.batched(None, rows, 0, 8, on_backoff=fallback.append)
    assert fallback == [4,2] and calls[:3] == [8,4,2]
    np.testing.assert_array_equal(actual, vectors(rows))
    def fail(*args, **kwargs): raise torch.cuda.OutOfMemoryError('always')
    monkeypatch.setattr(backend, 'forward', fail)
    with pytest.raises(torch.cuda.OutOfMemoryError): backend.batched(None, rows, 0, 2)


def test_cross_model_receipts_rejected(tmp_path):
    import shutil
    a = fixture(tmp_path/'a', 'openllama-3b'); run_extract(a, 1)
    b = fixture(tmp_path/'b', 'starcoderbase-3b')
    shutil.copytree(a.root/'hidden', b.root/'hidden')
    shutil.copytree(a.root/'receipts', b.root/'receipts')
    with pytest.raises(ValueError, match='Incompatible'): recover_extraction(b)


def test_source_scientific_kernels_unchanged():
    import ast
    names = {'degeneration', 'distribution', 'exact_knn', 'graph_metrics', 'projection', 'union_graph'}
    def kernels(p):
        return [ast.dump(n, include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    relative = Path('numzig/fullrange/analysis.py')
    assert kernels(relative) == kernels(CRYSTAL_ROOT/'source'/relative)


def test_partial_smoke_resume_reuses_gate_without_reinferring_completed_targets(tmp_path, monkeypatch):
    import torch
    import transformers
    import huggingface_hub
    import numzig.multimodel.backend as backend
    from numzig.multimodel.pipeline import saved_gate
    s = fixture(tmp_path/'smoke', chunk_size=8)
    model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=2, hidden_size=12))
    model.to = lambda device: model
    model.eval = lambda: model
    model.parameters = lambda: iter([torch.zeros(1)])
    loads = []
    snapshot = str(tmp_path / 'pinned_snapshot')
    def cached_snapshot(repo, revision, local_files_only):
        assert repo == s.manifest['config']['model_id']
        assert revision == s.manifest['config']['model_revision'] and local_files_only
        return snapshot
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', cached_snapshot)
    def load_fake(path, **kwargs):
        assert path == snapshot and kwargs['local_files_only']
        loads.append(1)
        return model
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', load_fake)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'SYNTHETIC GPU')
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda index: SimpleNamespace(total_memory=0))
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(backend, 'equivalence_gate', lambda model, rows, *args: ({'passed':True}, vectors(rows)))
    plan = make_plan(s, 'chunk', recover_extraction(s), 1, extraction_dependency(s))
    commits = 0
    def commit():
        nonlocal commits
        commits += 1
        if commits == 4: raise RuntimeError('interrupt after first smoke chunk committed')
    with pytest.raises(RuntimeError): backend.extract_model(WorkerStore(s.root, plan['id'], 0, commit))
    s = Store(s.root); missing = recover_extraction(s)
    completed = checkpoint_bytes(s, 'chunk/')
    gate = saved_gate(s); assert gate['passed']
    monkeypatch.setattr(backend, 'equivalence_gate', lambda *a, **k: pytest.fail('Completed smoke prompts re-inferred by gate'))
    called = []
    def infer(model, rows, *args, **kwargs): called.extend(r['point_id'] for r in rows); return vectors(rows)
    monkeypatch.setattr(backend, 'batched', infer)
    plan = make_plan(s, 'chunk', missing, 1, extraction_dependency(s))
    backend.extract_model(WorkerStore(s.root, plan['id'], 0), smoke_gate=gate)
    s = Store(s.root); assert not recover_extraction(s)
    assert not set(called) & set(r['point_id'] for r in load(s)[:8])
    assert all(checkpoint_bytes(s, 'chunk/')[p] == h for p,h in completed.items())
    assert len(loads) == 2  # one per invocation, never per batch/chunk


def test_cloud_scheduler_bounded_global_pool_drains_failed_peers():
    # Importing definitions does not build an image or resolve Modal resources.
    from modal_multimodel_fullrange import bounded_calls
    import threading,time
    state = dict(active=0, peak=0, done=[])
    lock = threading.Lock()
    class Fake:
        def remote(self, model, layer):
            with lock:
                state['active'] += 1; state['peak'] = max(state['peak'],state['active'])
            try:
                time.sleep(.01)
                if layer == 2: raise RuntimeError('synthetic peer failure')
                with lock: state['done'].append((model,layer))
            finally:
                with lock: state['active'] -= 1
        def get_current_stats(self):
            return SimpleNamespace(num_running_inputs=state['active'],num_total_runners=state['active'],backlog=0)
    jobs=[(model,layer) for model in MODELS for layer in range(4)]
    errors=bounded_calls(Fake(),jobs,3)
    assert len(errors)==2 and len(state['done'])==6 and state['peak']<=3 and state['active']==0


def plot_plan(store, missing, workers):
    from numzig.multimodel import plots
    return make_plan(store, 'plot', missing, workers, plots.dependency(store),
                     dependencies={key: plots.dependency(store, key) for key in missing})


def test_actual_module_cli_uses_spawn_importable_workers(tmp_path):
    import subprocess,sys,os
    from numzig.multimodel.__main__ import local_analysis,local_plot
    assert local_analysis.__module__ == local_plot.__module__ == 'numzig.multimodel.pipeline'
    s=fixture(tmp_path/'cli');run_extract(s,2);s=Store(s.root);recover_extraction(s)
    result=subprocess.run([sys.executable,'-m','numzig.multimodel','analyze','--root',str(s.root),'--cpu-workers','2'],
        text=True,capture_output=True,env=os.environ | {'MPLCONFIGDIR': str(tmp_path/'mpl')},timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    assert Store(s.root).manifest['stages']['analysis']=='complete'
