"""Future opt-in native-model experiment. Definitions/imports do not start jobs."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import subprocess
import time
import modal
from numzig.multimodel import MODELS, configuration, resources, CRYSTAL_NAME, BATCH_SIZES

ROOT = Path(__file__).resolve().parent
REMOTE = '/root/numeral_native'
image = (modal.Image.debian_slim(python_version='3.11')
    .pip_install_from_requirements(str(ROOT / 'requirements-crystal-fullrange.txt'))
    .env({'PYTHONPATH': REMOTE, 'HF_HOME': '/cache/huggingface', 'TOKENIZERS_PARALLELISM': 'false',
        'PYTHONUNBUFFERED': '1', 'MPLBACKEND': 'Agg', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    .add_local_dir(ROOT / 'numzig', REMOTE + '/numzig', ignore=['__pycache__', '*.pyc'])
    .add_local_dir(ROOT / 'tests', REMOTE + '/tests', ignore=['__pycache__', '*.pyc']))
for name in ('modal_multimodel_fullrange.py', 'modal_crystal_fullrange.py', 'requirements-crystal-fullrange.txt',
             'requirements-multimodel.txt', 'MULTIMODEL_FULLRANGE.md', 'CRYSTAL_FULLRANGE.md'):
    image = image.add_local_file(ROOT / name, REMOTE + '/' + name)
app = modal.App('numeral-native-fullrange', image=image)
results = modal.Volume.from_name('numberline-zigzag-results', create_if_missing=False)
cache = modal.Volume.from_name('numberline-zigzag-hf-cache', create_if_missing=False)
secret = modal.Secret.from_name('numberline-hf-token')


def claim_launch(directory, launch_id, commit):
    """Automatic coordinator retries must fail closed before touching results.

    A deliberate resume gets a fresh ID only after the local stopped-app guard.
    The marker is committed before preparing datasets or launching any workers.
    """
    from numzig.fullrange.storage import write_json, digest
    marker = Path(directory) / (launch_id + '.json')
    if marker.exists():
        raise RuntimeError('Coordinator attempt restarted. Stop this app, wait for zero tasks, then resume with a fresh launch; saved receipts are preserved.')
    write_json(marker, dict(launch_id=launch_id, coordinator_sha256=digest(__file__), started=time.time()))
    commit()


def root_for(model, smoke):
    return Path('/artifacts') / configuration(model, smoke)['experiment']


@app.function(gpu='L40S', cpu=(2, 2), memory=32768, timeout=86400, retries=0,
              max_containers=10, scaledown_window=2,
              volumes={'/artifacts': results, '/cache': cache.with_mount_options(read_only=True)})
def extraction(model: str, smoke: bool, plan_id: str, slot: int, batch_size: int, gate: dict):
    from numzig.fullrange.parallel import WorkerStore
    from numzig.multimodel.backend import extract_model
    results.reload(); cache.reload()
    store = WorkerStore(root_for(model, smoke), plan_id, slot, results.commit)
    extract_model(store, batch_size, gate)


@app.function(cpu=(2, 2), memory=6144, timeout=86400, retries=0, max_containers=31, scaledown_window=2,
              volumes={'/artifacts': results})
def analysis_job(root: str, plan_id: str, slot: int):
    from numzig.fullrange.parallel import WorkerStore
    from numzig.multimodel.pipeline import analysis_worker
    results.reload()
    analysis_worker(WorkerStore(root, plan_id, slot, results.commit))


@app.function(cpu=(1, 1), memory=4096, timeout=86400, retries=0, max_containers=16, scaledown_window=2,
              volumes={'/artifacts': results})
def plot_job(root: str, plan_id: str, slot: int):
    from numzig.fullrange.parallel import WorkerStore
    from numzig.multimodel.plots import worker
    results.reload()
    worker(WorkerStore(root, plan_id, slot, results.commit))


def bounded_calls(function, jobs, workers):
    # Each future blocks on its remote result. At most workers invocations active;
    # all peers drain before the coordinator reloads or merges receipts.
    errors = []
    observed = []
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs)) or 1) as pool:
        pending = [pool.submit(function.remote, *job) for job in jobs]
        for future in as_completed(pending):
            try:
                stats = function.get_current_stats()
                observation = dict(timestamp=time.time(), running_inputs=stats.num_running_inputs,
                                   containers=stats.num_total_runners, backlog=stats.backlog)
                observed.append(observation)
                print(f'[OBSERVED] {observation}', flush=True)
            except Exception as exc:
                print(f'[OBSERVED unavailable] {type(exc).__name__}', flush=True)
            try:
                future.result()
            except Exception as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
    bounded_calls.observed = observed
    return errors


@app.function(cpu=(1, 1), memory=8192, timeout=86400, retries=0, max_containers=1, nonpreemptible=True,
              volumes={'/artifacts': results, '/cache': cache}, secrets=[secret])
def pipeline(stage: str, smoke: bool, models: str, gpu_workers: int, cpu_workers: int, plot_workers: int, batch_size: int, launch_id: str):
    from threadpoolctl import threadpool_limits
    from huggingface_hub import snapshot_download, HfApi
    from numzig.fullrange.storage import Store, write_json
    from numzig.fullrange.parallel import make_plan
    from numzig.fullrange.extract import extraction_dependency
    from numzig.fullrange.analysis import analysis_dependency
    from numzig.multimodel.data import prepare
    from numzig.multimodel.backend import ensure_contract, contract
    from numzig.fullrange.storage import fingerprint
    from numzig.multimodel.pipeline import (recover_extraction, saved_gate, layer_cache, recover_analysis,
        global_jobs, finalize_analysis, record_allocation)
    from numzig.multimodel import plots
    threadpool_limits(limits=1)
    settings = resources(gpu_workers, cpu_workers, plot_workers)
    names = list(MODELS) if models == 'both' else [models]
    if any(m not in MODELS for m in names) or batch_size not in BATCH_SIZES:
        raise ValueError('Invalid model or batch size')
    results.reload(); cache.reload()
    claim_launch('/artifacts/_native_launches', launch_id, results.commit)
    def stores():
        if stage not in ('all', 'prepare') and any(not (root_for(m, smoke) / 'manifest.json').exists() for m in names):
            raise ValueError('Missing saved experiment; this stage never prepares or extracts implicitly')
        return [Store(root_for(m, smoke), configuration(m, smoke), results.commit) for m in names]
    current = stores()
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if stage in ('all', 'prepare'):
        for s in current:
            prepare(s, Path('/artifacts') / CRYSTAL_NAME, token=token)
        cache.commit()
    if stage in ('all', 'extract'):
        # Sequential models: never 10 GPUs per model concurrently, no model swaps
        # inside workers. Warm all required weights before any GPU fan-out.
        needed = []
        for s in current:
            ensure_contract(s)
            missing = recover_extraction(s)
            if missing:
                needed.append(s)
                if not smoke:
                    smoke_store = Store(root_for(s.manifest['config']['model_key'], True), commit=results.commit)
                    gate = saved_gate(smoke_store)
                    smoke_store.require('validation')
                    from numzig.fullrange.storage import read_json
                    smoke_validation = read_json(smoke_store.root / 'validation.json')
                    if not smoke_validation['passed'] or smoke_validation['synthetic'] or smoke_store.manifest['stages'].get('plots') != 'complete':
                        raise ValueError('Full extraction requires completed real smoke geometry/plot audit')
                    if not gate or gate['contract'] != fingerprint(contract(s)):
                        raise ValueError('Run and pass compatible smoke before requesting full extraction')
        for s in needed:
            cfg = s.manifest['config']
            names_in_repo = HfApi().list_repo_files(cfg['model_id'], revision=cfg['model_revision'], token=token)
            weights = '*.safetensors' if any(n.endswith('.safetensors') for n in names_in_repo) else 'pytorch_model*.bin'
            snapshot_download(cfg['model_id'], revision=cfg['model_revision'], token=token,
                allow_patterns=['*.json', weights, '*.model', 'merges.txt', 'vocab.json'],
                ignore_patterns=['optimizer*', 'training_args*'])
        cache.commit()
        for s in needed:
            model = s.manifest['config']['model_key']
            missing = recover_extraction(s)
            # One smoke model copy validates every smoke point under all compositions.
            gate = saved_gate(s if smoke else Store(root_for(model, True), commit=results.commit))
            plan = make_plan(s, 'chunk', missing, 1 if smoke else gpu_workers, extraction_dependency(s))
            record_allocation(s, 'extraction', settings, len(missing), len(plan['assignments']))
            results.commit()
            jobs = [(model, smoke, plan['id'], i, batch_size, gate) for i in range(len(plan['assignments']))]
            errors = bounded_calls(extraction, jobs, gpu_workers)
            results.reload()
            s = Store(s.root, commit=results.commit)
            save_observations(s, 'extraction', bounded_calls.observed)
            remaining = recover_extraction(s)
            if errors or remaining:
                raise RuntimeError(f'Extraction interrupted; completed peers preserved; remaining={len(remaining)}; {errors}')
        current = stores()
    if stage in ('all', 'analyze'):
        # Hash source representations at coordinator barrier, then one transpose.
        for s in current:
            if recover_extraction(s):
                raise ValueError('Missing extraction; analysis never invokes inference')
            from numzig.fullrange.storage import read_json
            meta = read_json(s.root / 'tokenizer_provenance.json')
            layer_cache(s, (meta['expected_levels'], meta['expected_hidden_dim']))
        jobs = global_jobs(current, recover_analysis, cpu_workers, 'layer', analysis_dependency)
        for s in current:
            record_allocation(s, 'analysis', settings, len(jobs), cpu_workers)
        results.commit()
        errors = bounded_calls(analysis_job, jobs, cpu_workers)
        results.reload(); current = stores()
        for s in current:
            save_observations(s, 'analysis', bounded_calls.observed)
        remaining = sum(len(recover_analysis(s)) for s in current)
        if errors or remaining:
            raise RuntimeError(f'Analysis interrupted; remaining={remaining}; completed peers preserved; {errors}')
        for s in current:
            finalize_analysis(s)
    if stage in ('all', 'plot', 'finish'):
        jobs = global_jobs(current, plots.recover, plot_workers, 'plot', plots.dependency)
        for s in current:
            record_allocation(s, 'plot', settings, len(jobs), plot_workers)
        results.commit()
        errors = bounded_calls(plot_job, jobs, plot_workers)
        results.reload(); current = stores()
        for s in current:
            save_observations(s, 'plot', bounded_calls.observed)
        remaining = sum(len(plots.recover(s)) for s in current)
        if errors or remaining:
            raise RuntimeError(f'Plotting interrupted; remaining={remaining}; resume plot only; {errors}')
        for s in current:
            plots.finalize(s)
    if stage in ('all', 'validate', 'finish'):
        from numzig.multimodel.validation import audit
        for s in current:
            audit(s)
    if stage in ('all', 'package', 'finish'):
        for s in current:
            plots.package(s)
    if stage == 'compare' or (stage == 'all' and not smoke and models == 'both'):
        plots.comparison([Path('/artifacts') / CRYSTAL_NAME, *[root_for(m, False) for m in MODELS]],
                         Path('/artifacts') / 'numerical_three_model_comparison')
        results.commit()
    return dict(settings=settings, experiments=[dict(name=s.manifest['config']['experiment'], stages=s.manifest['stages']) for s in current])


def save_observations(store, stage, observations):
    from numzig.fullrange.storage import write_json, fingerprint
    path = store.root / 'execution' / f'{time.time_ns()}_{stage}_observed.json'
    write_json(path, dict(stage=stage, observations=observations, limitation='Snapshots at task completions; may miss peak concurrency'))
    store.finish(f'execution/{path.stem}', [path], fingerprint(observations))


@app.local_entrypoint()
def main(stage: str = 'all', smoke: bool = False, models: str = 'both', gpu_workers: int = 10,
         cpu_workers: int = 24, plot_workers: int = 16, batch_size: int = 1):
    if stage not in ('all', 'prepare', 'extract', 'analyze', 'plot', 'validate', 'package', 'compare', 'finish'):
        raise ValueError('Invalid stage')
    resources(gpu_workers, cpu_workers, plot_workers)
    if stage in ('all', 'extract'):
        for model in MODELS if models == 'both' else [models]:
            if batch_size not in configuration(model)['allowed_batch_sizes']:
                raise ValueError(f'{model} supports batch sizes {configuration(model)["allowed_batch_sizes"]}')
    # Read-only guard before remote work, including restarts; two launches cannot
    # intentionally share writers. The current launch itself is excluded.
    import sys
    raw = subprocess.check_output([sys.executable, '-m', 'modal', 'app', 'list', '--env', 'main', '--json'])
    active = json.loads(raw)
    from numzig.multimodel import reject_overlapping_apps
    reject_overlapping_apps(active, app.app_id)
    import uuid
    print(pipeline.remote(stage, smoke, models, gpu_workers, cpu_workers, plot_workers, batch_size, uuid.uuid4().hex))
