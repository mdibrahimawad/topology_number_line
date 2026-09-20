"""One-time empty-run archival after the cached-loader bug; never migrates vectors."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import modal
from modal_multimodel_fullrange import image, results
from numzig.multimodel import MODELS, configuration, reject_overlapping_apps

app = modal.App('native-empty-smoke-reprepare', image=image)


@app.function(cpu=(1, 1), memory=1024, timeout=600, volumes={'/artifacts': results})
def reprepare(models: str, reason: str, current_config: bool):
    from numzig.fullrange.storage import Store, read_json, digest, write_json, fingerprint
    results.reload()
    report = {}
    for model in MODELS if models == 'both' else [models]:
        root = Path('/artifacts') / configuration(model, True)['experiment']
        archive = root.with_name(root.name + '_' + reason)
        source = archive if archive.exists() else root
        old = read_json(source / 'manifest.json')
        if (source / 'hidden').exists() or any(k.startswith(('chunk/', 'layer/', 'analysis/')) for k in old['artifacts']):
            raise ValueError('Refusing contract reset: representations or analyses exist')
        saved = old['artifacts']['dataset']
        assert saved['status'] == 'complete'
        for path, sha in saved['files'].items():
            assert digest(source / path) == sha
        if not archive.exists():
            root.rename(archive)
            results.commit()
        fresh = Store(root, configuration(model, True) if current_config else old['config'], results.commit)
        if any(k != 'dataset' for k in fresh.manifest['artifacts']):
            raise ValueError('Fresh destination already used; do not replace its contract')
        for path, sha in saved['files'].items():
            dest = root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive / path, dest)
            assert digest(dest) == sha
        original = read_json(archive / 'dataset.json')
        if current_config:
            updated = dict(original, config=fresh.manifest['config'])
            write_json(root / 'dataset.json', updated)
        assert read_json(root / 'dataset.json')['records'] == original['records']
        fresh.finish('dataset', [root / p for p in saved['files']], fresh.manifest['config_fingerprint'], saved['metadata'])
        fresh.stage('dataset', 'complete')
        report[model] = dict(archive=str(archive), reused_dataset_sha256=digest(root / 'dataset.json'),
                             reused_records_sha256=fingerprint(original['records']), published_representations=0,
                             configuration_unchanged=not current_config, current_config=fresh.manifest['config'])
    return report


@app.local_entrypoint()
def main(models: str = 'both', reason: str = 'failed_loader_20260919', current_config: bool = False):
    if models != 'both' and models not in MODELS:
        raise ValueError('Invalid model')
    if reason not in ('failed_loader_20260919', 'failed_batch_gate_20260919'):
        raise ValueError('Unexpected archive reason')
    rows = json.loads(subprocess.check_output([sys.executable, '-m', 'modal', 'app', 'list', '--env', 'main', '--json']))
    reject_overlapping_apps(rows, app.app_id)
    report = reprepare.remote(models, reason, current_config)
    Path(f'results/native_models_modal_20260919/{reason}_archival.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
