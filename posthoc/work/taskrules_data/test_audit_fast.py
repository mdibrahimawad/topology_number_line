"""I/O refactor regression fixture; small synthetic dimensions, full coverage."""
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import numpy as np
from scipy.spatial.distance import cdist

REPO = Path('/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL')
sys.path.insert(0, str(REPO))
from numzig.taskrules.data import make_dataset
from numzig.taskrules.runtime import chunk_rows
from numzig.fullrange.storage import fingerprint


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


original = load('audit_original', REPO / 'numzig/taskrules/audit.py')
fast = load('audit_fast', Path(__file__).with_name('audit_fast.py'))
specs = {name: (levels, 3) for name, (levels, _) in original.SPECS.items()}
rows = make_dataset()
groups = {(task, ctx): [r for r in rows if (r['task'], r['context_id']) == (task, ctx)]
          for task in original.TASKS for ctx in (0, 1)}
a = np.random.default_rng(5).normal(size=(1000, 3)).astype(np.float32)
centered = a.astype(np.float64) - a.astype(np.float64).mean(axis=0)
_, _, components = np.linalg.svd(centered, full_matrices=False)
scores = centered @ components.T
variance = scores.var(axis=0, ddof=1)
distance = cdist(a, a)
np.fill_diagonal(distance, np.inf)
nearest = np.argsort(distance, axis=1, kind='stable')[:, :4]
payload = b'fixture artifact bytes\n'
sha = hashlib.sha256(payload).hexdigest()
archives = {}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


with tempfile.TemporaryDirectory(prefix='audit-io-fixture-') as temp:
    root = Path(temp)
    shared = root / 'shared'
    shared.write_bytes(payload)

    def link(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        os.link(shared, path)

    write_json(root / 'dataset.json', rows)
    write_json(root / 'contract.json', {'fingerprint': 'fixture', 'dataset_hash': fingerprint(rows)})
    for model, (levels, dimension) in specs.items():
        base = root / model
        for key, part in chunk_rows(rows):
            files = {f'hidden/{key}.npz': sha, f'tokens/{key}.json': sha}
            for name in files:
                link(base / name)
            write_json(base / 'receipts' / f'{key}.json', dict(contract='fixture', model=model,
                ids=[r['point_id'] for r in part], shape=[levels, len(part), dimension], files=files))
        for (task, ctx), selected in groups.items():
            directory = base / 'analysis' / f'{task}_ctx{ctx}'
            config = dict(model=model, task=task, context_id=str(ctx), shape=[levels, 1000, dimension])
            ids = np.array([r['point_id'] for r in selected])
            z = dict(point_ids=ids, mean=a.astype(np.float64).mean(axis=0), components=components,
                scores3=scores, explained_variance=variance, explained_variance_ratio=variance/variance.sum(),
                neighbors=ids[nearest], distances=np.take_along_axis(distance, nearest, axis=1),
                h0=np.array([[0., np.inf]]), h1=np.empty((0, 2)))
            artifacts = {}
            for layer in range(levels):
                name = f'layers/layer_{layer:02d}.npz'
                link(directory / name)
                archives[str(directory / name)] = z
                artifacts[f'layer/{layer:02d}'] = dict(status='complete', files={name: sha})
            write_json(directory / 'manifest.json', dict(config=config, config_fingerprint=fingerprint(config), artifacts=artifacts))
            write_json(directory / 'summary.json', dict(status='complete', layer_count=levels))
        for ctx in (0, 1):
            for family, tasks in (('permutations', original.TASKS[:4]), ('notation', original.TASKS[4:])):
                directory = base / 'comparison' / f'ctx{ctx}'
                z = dict(scores=np.repeat(scores[None], len(tasks), axis=0), task_names=np.array(tasks),
                    targets=np.array([r['target'] for r in groups[tasks[0], ctx]]),
                    mean=a.astype(np.float64).mean(axis=0), components=components)
                for layer in range(levels):
                    name = f'{family}/layer_{layer:02d}'
                    files = {name+'.npz': sha, name+'_input.png': sha}
                    if family == 'permutations':
                        files[name+'_output.png'] = sha
                    for path in files:
                        link(directory / path)
                    archives[str(directory / (name+'.npz'))] = z
                    write_json(directory / (name+'.json'), dict(model=model, context_id=ctx, layer=layer,
                        tasks=list(tasks), count_per_condition=1000, files=files))
    # The source loader and NPZ reader are unchanged by the refactor. Substitute
    # small in-memory vectors so the complete control flow is cheap to compare.
    for module in (original, fast):
        module.SPECS = specs
        module.load_saved_group = lambda base, model, rows, task, ctx, contract: (
            np.repeat(a[None], specs[model][0], axis=0), groups[task, ctx])
    real_load = np.load
    np.load = lambda path, **kwargs: contextlib.nullcontext(archives[str(path)])
    try:
        before = original.audit_run(root)
        after = fast.audit_run(root)
        assert before['passed'] and after == before, (before, after)
        assert after['counts']['sampled_geometry_layers'] == 18
        assert after['counts']['sampled_knn_queries'] == 288
        # Replace one hard link; do not corrupt the shared fixture payload.
        corrupt = root / 'crystal/analysis/copy4_ctx0/layers/layer_00.npz'
        corrupt.unlink()
        corrupt.write_bytes(b'corrupt')
        before_bad = original.audit_run(root)
        after_bad = fast.audit_run(root)
        assert not before_bad['passed'] and after_bad == before_bad, (before_bad, after_bad)
        assert 'Analysis hash:' in after_bad['errors'][0]
    finally:
        np.load = real_load
    active = peak = 0
    lock = threading.Lock()

    def blocking_read(value):
        global active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(.01)
        with lock:
            active -= 1
        return value

    with fast.ThreadPoolExecutor(max_workers=16) as pool:
        assert list(fast._io_map(pool, blocking_read, range(49))) == list(range(49))
    assert 1 < peak <= 16
print('PASS: complete original/fast reports identical; all 18 numerical layers and 288 kNN queries run; corrupted artifact rejected identically; I/O ordering and 16-reader bound verified.')
