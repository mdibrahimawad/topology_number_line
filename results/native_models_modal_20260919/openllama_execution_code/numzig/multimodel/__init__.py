"""Native StarCoderBase/OpenLLaMA extension; never mutates Crystal outputs."""
from pathlib import Path
from numzig.fullrange import configuration as crystal_configuration

MODELS = {
    'starcoderbase-3b': dict(model_id='bigcode/starcoderbase-3b', revision='e1c5ef4ebb97afa0db09ec3e520f0487ca350bbe',
        label='StarCoderBase-3B', model_type='gpt_bigcode', bos=False, use_fast=True,
        public_access='gated: unauthenticated pinned config returned HTTP 401',
        dimension=None, blocks=None),
    'openllama-3b': dict(model_id='openlm-research/open_llama_3b', revision='141067009124b9c0aea62c76b3eb952174864057',
        label='OpenLLaMA-3B', model_type='llama', bos=True, use_fast=False,
        public_access='public: pinned config and tokenizer metadata verified', dimension=3200, blocks=26),
}
CRYSTAL_NAME = 'crystal_fullrange_1_10000_ctx1234_k4_seed42'
CRYSTAL_ROOT = Path(__file__).resolve().parents[2] / 'results/crystal_fullrange_modal_20260919T123307Z/full_complete' / CRYSTAL_NAME
CRYSTAL_DATASET_SHA256 = 'bf0616805c5f4740fb3e790b0f7d5e7735ac89639696c271a5e6e2a626c12016'
RAW_FIELDS = ('point_id', 'target', 'target_string', 'leading_digit', 'digit_count', 'demonstrations',
              'demonstration_digit_lengths', 'demonstration_equals_target', 'prompt', 'character_count', 'seed')
BATCH_SIZES = (1, 2, 4, 8, 16, 32)
# Declared before real weights: fp32 eager, TF32 disabled. Any kNN rank change fails,
# including near ties; margins are diagnostics, never permission to alter neighbors.
TOLERANCES = dict(vector_atol=2e-5, vector_rtol=2e-5, relative_l2=2e-5,
                  cosine_error=2e-6, distance_atol=2e-5, distance_rtol=2e-5)


def configuration(model, smoke=False):
    spec = MODELS[model]
    legacy = crystal_configuration(smoke)
    return {k: legacy[k] for k in ('seed', 'k', 'chunk_size', 'prompt_policy', 'distance', 'pca')} | dict(
        schema=1, experiment=f'{model}_fullrange_1_10000_ctx1234_k4_seed42' + ('_smoke' if smoke else ''),
        model_key=model, model_id=spec['model_id'], model_revision=spec['revision'], tokenizer_revision=spec['revision'],
        smoke=smoke, model_dtype='float32', saved_dtype='float32', trust_remote_code=False,
        bos=spec['bos'], use_fast=spec['use_fast'], special_policy='explicit one BOS' if spec['bos'] else 'no inserted BOS/EOS',
        extraction='all native HF returned states at final real token; no trailing EOS',
        backend='native Transformers eager backbone; right padding; arange positions; TF32 disabled',
        allowed_batch_sizes=list(BATCH_SIZES), tolerances=TOLERANCES, maximum_padded_length=128,
        crystal_dataset_sha256=CRYSTAL_DATASET_SHA256, paper_checkpoint_match='unverified')


def resources(gpu_workers=10, cpu_workers=24, plot_workers=16):
    # Reserve headroom for scaled-down-but-still-resident GPU/plot containers.
    # Workers have strict CPU limits. Phase barriers prevent concurrent writers.
    if not 1 <= gpu_workers <= 10 or not 1 <= cpu_workers <= 31 or not 1 <= plot_workers <= 16:
        raise ValueError('Global limits: GPU 1..10, analysis 1..31, plot 1..16')
    peak = 1 + 2 * gpu_workers + 2 * cpu_workers + plot_workers
    if peak > 100:
        raise ValueError('CPU budget includes coordinator and residual warm containers')
    return dict(gpu_workers=gpu_workers, gpu='L40S', gpu_host_cpus=2, gpu_host_memory_mib=32768,
        cpu_workers=cpu_workers, cpu_cores_per_worker=2, cpu_memory_mib=6144, blas_threads=2,
        plot_workers=plot_workers, plot_cpus=1, plot_memory_mib=4096,
        coordinator_cpus=1, coordinator_memory_mib=8192,
        analysis_phase_cpus=1 + 2 * cpu_workers, analysis_phase_memory_mib=8192 + cpu_workers * 6144,
        conservative_aggregate_cpu_ceiling=peak,
        conservative_memory_mib=8192 + gpu_workers * 32768 + cpu_workers * 6144 + plot_workers * 4096,
        scheduling='sequential models for GPU; global independent model/layer CPU pool; no point subsets',
        quota_status='not verified; explicit smaller worker counts on resume; never overlap launches')


def reject_overlapping_apps(rows, current_app_id):
    """Fail closed on CLI schema drift; a stopping app can still have writers."""
    for row in rows:
        if not all(k in row for k in ('description', 'app_id', 'state', 'tasks')):
            raise ValueError('Unknown Modal app-list schema; inspect manually before launch')
        if (row['description'] == 'numeral-native-fullrange' and row['app_id'] != current_app_id
                and (str(row['state']).lower() != 'stopped' or int(row['tasks']) != 0)):
            raise RuntimeError('Another native experiment app has not stopped. No new writers started; wait before resume.')
