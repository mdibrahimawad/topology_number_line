"""Opt-in cloud entrypoint. Import creates no resources and starts no jobs.

Run only after workspace confirmation. Never overlap launches for the same results
experiment; stop the previous app before resuming. Worker counts are operational.
"""
from pathlib import Path
import os

import modal
from numzig.fullrange.parallel import (execution_settings, CPU_PER_WORKER, CPU_MEMORY_MIB,
                                      BLAS_THREADS, MAX_CPU_WORKERS, MAX_GPU_WORKERS)

ROOT = Path(__file__).resolve().parent
REMOTE = '/root/crystal_fullrange'
image = (modal.Image.debian_slim(python_version='3.11')
    .pip_install_from_requirements(str(ROOT / 'requirements-crystal-fullrange.txt'))
    .env({'PYTHONPATH': REMOTE, 'HF_HOME': '/cache/huggingface', 'HF_MODULES_CACHE': '/tmp/hf_modules',
          'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1', 'MPLBACKEND': 'Agg',
          **{name: str(BLAS_THREADS) for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']}})
    .add_local_dir(ROOT / 'numzig', REMOTE + '/numzig', ignore=['__pycache__', '*.pyc'])
    .add_local_dir(ROOT / 'tests', REMOTE + '/tests', ignore=['__pycache__', '*.pyc'])
    .add_local_file(ROOT / 'modal_crystal_fullrange.py', REMOTE + '/modal_crystal_fullrange.py')
    .add_local_file(ROOT / 'requirements-crystal-fullrange.txt', REMOTE + '/requirements-crystal-fullrange.txt')
    .add_local_file(ROOT / 'CRYSTAL_FULLRANGE.md', REMOTE + '/CRYSTAL_FULLRANGE.md'))
app = modal.App('crystal-fullrange', image=image)
results = modal.Volume.from_name('numberline-zigzag-results', create_if_missing=False)
cache = modal.Volume.from_name('numberline-zigzag-hf-cache', create_if_missing=False)
secret = modal.Secret.from_name(os.environ.get('MODAL_HF_SECRET_NAME', 'numberline-hf-token'))


def hf_token():
    return os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')


def root_for(smoke):
    from numzig.fullrange import configuration
    return Path('/artifacts') / configuration(smoke)['experiment']


@app.function(gpu='L40S', cpu=(2, 2), memory=32768, timeout=24 * 60 * 60,
              max_containers=MAX_GPU_WORKERS, retries=0, scaledown_window=2,
              volumes={'/artifacts': results, '/cache': cache.with_mount_options(read_only=True)})
def extraction(smoke: bool, plan_id: str, slot: int):
    from numzig.fullrange.parallel import WorkerStore
    from numzig.fullrange.extract import extract_model
    results.reload()
    cache.reload()
    store = WorkerStore(root_for(smoke), plan_id, slot, results.commit)
    # A worker has one model on its one GPU for its entire disjoint assignment.
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        return extract_model(store, assigned_keys=store.assigned)


@app.function(cpu=(CPU_PER_WORKER, CPU_PER_WORKER), memory=CPU_MEMORY_MIB,
              timeout=24 * 60 * 60, max_containers=MAX_CPU_WORKERS, retries=0,
              volumes={'/artifacts': results})
def analysis_worker(smoke: bool, plan_id: str, slot: int):
    from numzig.fullrange.parallel import WorkerStore
    from numzig.fullrange.analysis import analyze_assignments
    results.reload()
    store = WorkerStore(root_for(smoke), plan_id, slot, results.commit)
    analyze_assignments(store, BLAS_THREADS)


@app.function(cpu=(1, 1), memory=8192, timeout=24 * 60 * 60, max_containers=1,
              volumes={'/artifacts': results, '/cache': cache}, secrets=[secret])
def pipeline(stage: str, smoke: bool, gpu_workers: int = 10, cpu_workers: int = 4):
    from numzig.fullrange import configuration
    from numzig.fullrange.storage import Store
    from numzig.fullrange.dataset import prepare
    from numzig.fullrange.parallel import make_plan
    from numzig.fullrange.coordinator import recover_extraction, recover_analysis, drain_calls
    from threadpoolctl import threadpool_limits
    settings = execution_settings(gpu_workers, cpu_workers)
    threadpool_limits(limits=1)  # Coordinator metrics/plots never inherit the workers' BLAS budget.
    cfg = configuration(smoke)
    results.reload()
    cache.reload()
    store = Store(root_for(smoke), cfg, results.commit)
    if stage in ('all', 'prepare'):
        prepare(store, hf_token())
        cache.commit()
    if stage in ('all', 'extract'):
        from numzig.fullrange.runtime import ensure_extraction_contract, warm_model_cache
        from numzig.fullrange.extract import extraction_dependency
        # Validate compatibility even when all representations are already complete.
        ensure_extraction_contract(store)
        missing = recover_extraction(store)
        if missing:
            warm_model_cache(store, hf_token())
            cache.commit()  # All weights/tokenizer assets visible BEFORE GPU fan-out.
            store.stage('extraction', 'running')
            plan = make_plan(store, 'chunk', missing, gpu_workers, extraction_dependency(store))
            store.log(f'[FAN-OUT GPU] workers={len(plan["assignments"])} one L40S/model each; missing_chunks={len(missing)}')
            results.commit()  # Shared plan/progress durable before workers start.
            calls = []
            failures = []
            try:
                for slot in range(len(plan['assignments'])):
                    calls.append(extraction.spawn(smoke, plan['id'], slot))
            except Exception as exc:
                failures.append(f'submission: {exc}')
            failures.extend(drain_calls(calls))
            results.reload()  # Barrier: no worker can still be writing when we merge.
            store = Store(root_for(smoke), cfg, results.commit)
            remaining = recover_extraction(store)
            if failures or remaining:
                raise RuntimeError(f'Extraction stopped; successful chunks retained. Remaining={len(remaining)}. {failures}')
        else:
            store.log('[SKIP] all hidden chunks complete; no GPU workers submitted')
    if stage in ('all', 'analyze'):
        from numzig.fullrange.analysis import analyze, analysis_dependency
        threadpool_limits(limits=1)
        missing = recover_analysis(store)
        if missing:
            store.stage('analysis', 'running')
            plan = make_plan(store, 'layer', missing, cpu_workers, analysis_dependency(store))
            store.log(f'[FAN-OUT CPU] workers={len(plan["assignments"])} settings={settings}; every layer uses ALL points')
            results.commit()
            calls, failures = [], []
            try:
                for slot in range(len(plan['assignments'])):
                    calls.append(analysis_worker.spawn(smoke, plan['id'], slot))
            except Exception as exc:
                failures.append(f'submission: {exc}')
            failures.extend(drain_calls(calls))
            results.reload()
            store = Store(root_for(smoke), cfg, results.commit)
            remaining = recover_analysis(store)
            if failures or remaining:
                raise RuntimeError(f'Analysis stopped; successful layers retained. Remaining={len(remaining)}. {failures}')
        analyze(store, allow_compute=False)  # Comparisons only after full-layer outputs are available.
    if stage in ('all', 'plot'):
        from numzig.fullrange.plots import render
        render(store)
    return dict(experiment=cfg['experiment'], stages=store.manifest['stages'], execution=settings)


@app.local_entrypoint()
def main(stage: str = 'all', smoke: bool = False, gpu_workers: int = 10, cpu_workers: int = 4):
    if stage not in ('all', 'prepare', 'extract', 'analyze', 'plot'):
        raise ValueError('stage must be all, prepare, extract, analyze, or plot')
    settings = execution_settings(gpu_workers, cpu_workers)
    print(f'Crystal stage={stage} smoke={smoke} execution={settings}; resume valid checkpoints')
    print(pipeline.remote(stage, smoke, gpu_workers, cpu_workers))
