"""Disjoint work plans and independent receipts; only the coordinator writes manifest.json."""
from __future__ import annotations

import copy
from pathlib import Path
import uuid

from .storage import Store, digest, fingerprint, read_json, write_json

MAX_GPU_WORKERS = 10
MAX_CPU_WORKERS = 12
CPU_PER_WORKER = 4
CPU_MEMORY_MIB = 8192
BLAS_THREADS = 4


def partition(keys, workers):
    if workers < 1 or len(keys) != len(set(keys)):
        raise ValueError('Workers must be positive and work keys unique')
    count = min(workers, len(keys))
    return [list(keys[i::count]) for i in range(count)]


def execution_settings(gpu_workers=10, cpu_workers=4):
    if not 1 <= gpu_workers <= MAX_GPU_WORKERS or not 1 <= cpu_workers <= MAX_CPU_WORKERS:
        raise ValueError('gpu_workers must be 1..10; cpu_workers must be 1..12')
    return dict(gpu_workers=gpu_workers, cpu_workers=cpu_workers, cpu_per_worker=CPU_PER_WORKER,
                memory_mib_per_cpu_worker=CPU_MEMORY_MIB, blas_threads=BLAS_THREADS,
                analysis_cpu_ceiling=cpu_workers * CPU_PER_WORKER,
                coordinator_cpus=1, aggregate_analysis_cpu_ceiling=cpu_workers * CPU_PER_WORKER + 1)


def files_valid(root, item):
    try:
        return bool(item['files']) and all((root / p).is_file() and digest(root / p) == h
                                           for p, h in item['files'].items())
    except (KeyError, OSError):
        return False


def expected_paths(key):
    kind, number = key.split('/')
    if kind == 'chunk' and number.isdigit() and len(number) == 5:
        return {f'hidden/{number}.npy', f'hidden/{number}_ids.npy'}
    if kind == 'layer' and number.isdigit() and len(number) == 2:
        return {f'layers/layer_{number}.npz', f'layers/layer_{number}.json'}
    raise ValueError(f'Not a worker artifact: {key}')


def make_plan(store, kind, missing, workers, dependency):
    limit = MAX_GPU_WORKERS if kind == 'chunk' else MAX_CPU_WORKERS
    if kind not in ('chunk', 'layer') or not 1 <= workers <= limit:
        raise ValueError('Invalid worker count or work kind')
    for key in missing:
        if key.split('/')[0] != kind:
            raise ValueError('Mixed assignment kinds')
        expected_paths(key)
    plan_id = uuid.uuid4().hex
    plan = dict(id=plan_id, kind=kind, config_fingerprint=store.manifest['config_fingerprint'],
                dependency=dependency, assignments=partition(missing, workers))
    path = store.root / 'plans' / f'{plan_id}.json'
    write_json(path, plan)
    store.finish(f'plan/{plan_id}', [path], fingerprint(plan))
    return plan


class WorkerStore(Store):
    """Read shared state, write ONLY own outputs/receipts/private metadata.

    This deliberately does not call Store.__init__: a worker must never create,
    recover or replace the coordinator manifest, even on an error path.
    """
    def __init__(self, root, plan_id, slot, commit=lambda: None):
        self.root = Path(root)
        self.manifest = read_json(self.root / 'manifest.json')
        self.commit = commit
        if fingerprint(self.manifest['config']) != self.manifest['config_fingerprint']:
            raise ValueError('Invalid shared configuration')
        if not self.valid(f'plan/{plan_id}'):
            raise ValueError('Missing or corrupt committed work plan')
        self.plan = read_json(self.root / 'plans' / f'{plan_id}.json')
        if self.plan['config_fingerprint'] != self.manifest['config_fingerprint']:
            raise ValueError('Work plan configuration mismatch')
        self.assigned = self.plan['assignments'][slot]
        self.worker_id = f'{plan_id}/{slot:02d}'
        self.worker_dir = self.root / 'workers' / self.worker_id
        self.worker_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.worker_dir / 'unused_manifest.json'
        # Container restarts can reuse the same plan without a coordinator merge.
        # Read receipts without touching shared state or publishing completion.
        for key in self.assigned:
            path = self.root / 'receipts' / (key + '.json')
            if not path.exists():
                continue
            try:
                item = read_json(path)
            except (ValueError, OSError):
                continue
            if not isinstance(item, dict) or not all(k in item for k in ('key', 'dependency', 'files', 'status')):
                continue
            if item.get('dependency') != self.plan['dependency']:
                raise ValueError('Incompatible worker receipt')
            if (item.get('key') == key and set(item.get('files', {})) == expected_paths(key)
                and item.get('status') in ('prepared', 'complete') and files_valid(self.root, item)):
                self.manifest['artifacts'][key] = {**item, 'status': 'complete'}


    def save(self):
        raise RuntimeError('Workers cannot save the shared manifest')

    def output_path(self, name):
        return self.worker_dir / name

    def log(self, message):
        print(f'[{self.worker_id}] {message}', flush=True)
        with (self.worker_dir / 'progress.log').open('a') as f:
            f.write(message + '\n')

    def stage(self, name, status):
        write_json(self.worker_dir / 'status.json', dict(stage=name, status=status, assigned=self.assigned))
        self.commit()

    def valid(self, key, dependency=None):
        item = self.manifest['artifacts'].get(key)
        return bool(item and item['status'] == 'complete' and
                    (dependency is None or item['dependency'] == dependency) and files_valid(self.root, item))

    def finish(self, key, paths, dependency, metadata=None):
        private = key in ('extraction_runtime', 'extraction_benchmark')
        if private:
            receipt_key = f'worker/{self.worker_id}/{key}'
            allowed = {str(self.output_path(key + '.json').relative_to(self.root))}
        else:
            if key not in self.assigned or dependency != self.plan['dependency']:
                raise ValueError('Worker attempted an unassigned/incompatible write')
            receipt_key, allowed = key, expected_paths(key)
        files = {str(Path(p).relative_to(self.root)): digest(p) for p in paths}
        if set(files) != allowed:
            raise ValueError('Worker output paths escape its assignment')
        receipt = dict(key=receipt_key, status='prepared', dependency=dependency, files=files,
                       metadata=metadata or {}, worker=self.worker_id, plan=self.plan['id'])
        receipt_path = self.root / 'receipts' / (receipt_key + '.json')
        write_json(receipt_path, receipt)
        self.commit()  # Durable data + pending receipt, no completion yet.
        receipt['status'] = 'complete'
        write_json(receipt_path, receipt)
        self.commit()
        self.manifest['artifacts'][key] = receipt
        self.log(f'[COMPLETE] {key}')


def merge_receipts(store, kind, dependency, validate):
    """Run at a barrier (all workers stopped), and before assigning work on resume.

    Recover prepared receipts only after checksums and scientific validation pass.
    Never rewrite chunk/layer bytes, and never let a bad receipt replace valid data.
    """
    merged, invalid = [], []
    for path in sorted((store.root / 'receipts' / kind).glob('*.json')):
        try:
            item = read_json(path)
        except (ValueError, OSError):
            invalid.append(path.name)
            continue
        if not isinstance(item, dict) or not all(name in item for name in ('key', 'dependency', 'files', 'status')):
            invalid.append(path.name)
            continue
        key = item['key']
        if item.get('dependency') != dependency:
            raise ValueError(f'Incompatible {kind} receipt {path.name}; refusing to mix results')
        if key != f'{kind}/{path.stem}' or set(item.get('files', {})) != expected_paths(key):
            raise ValueError(f'Invalid receipt paths: {path}')
        if item.get('status') not in ('prepared', 'complete') or not files_valid(store.root, item):
            invalid.append(key)
            continue
        old = store.manifest['artifacts'].get(key)
        if old and store.valid(key, dependency) and old['files'] != item['files']:
            raise ValueError(f'Conflicting valid results for {key}; overlapping launches are not supported')
        candidate = copy.deepcopy(item)
        candidate['status'] = 'complete'
        store.manifest['artifacts'][key] = candidate
        try:
            validate(key)
        except (ValueError, OSError, KeyError):
            if old is None:
                store.manifest['artifacts'].pop(key)
            else:
                store.manifest['artifacts'][key] = old
            invalid.append(key)
            continue
        if item['status'] == 'prepared':
            store.commit()
            item['status'] = candidate['status'] = 'complete'
            write_json(path, item)
            store.commit()
            store.log(f'[RECOVER] {key}: durable prepared receipt; no recomputation')
        candidate['status'] = 'complete'
        if old != candidate:
            merged.append(key)
    store.save()
    store.commit()
    store.log(f'[MERGE {kind}] merged={len(merged)} invalid={len(invalid)}')
    return merged, invalid


def merge_worker_metadata(store):
    """Private metadata has unique keys and never replaces historical runtime/benchmark files."""
    for path in sorted((store.root / 'receipts' / 'worker').rglob('*.json')):
        try:
            item = read_json(path)
        except (ValueError, OSError):
            continue
        if not isinstance(item, dict) or not all(k in item for k in ('key', 'worker', 'files', 'status')):
            continue
        key = item['key']
        if not key.startswith('worker/') or path.relative_to(store.root / 'receipts').as_posix() != key + '.json':
            raise ValueError('Invalid worker receipt identity')
        prefix = 'workers/' + item['worker'] + '/'
        if not all(p.startswith(prefix) and '..' not in Path(p).parts for p in item['files']):
            raise ValueError('Invalid worker metadata path')
        if item['status'] == 'complete' and files_valid(store.root, item):
            store.manifest['artifacts'][key] = item
    store.save()
    store.commit()
