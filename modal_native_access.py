"""Authorized read-only model access check; never returns secret values."""
import json
import os
from pathlib import Path
import modal
from modal_multimodel_fullrange import image, secret
from numzig.multimodel import MODELS

app = modal.App('native-model-access-preflight', image=image)


@app.function(cpu=(1, 1), memory=1024, timeout=600, secrets=[secret])
def check():
    from huggingface_hub import HfApi, hf_hub_url, get_hf_file_metadata, hf_hub_download
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    report = {'secret_token_present': bool(token), 'models': {}}
    for name, spec in MODELS.items():
        item = {'model_id': spec['model_id'], 'revision': spec['revision']}
        try:
            files = HfApi().list_repo_files(spec['model_id'], revision=spec['revision'], token=token)
            weights = [f for f in files if f.endswith('.safetensors')]
            weights = weights or [f for f in files if f.startswith('pytorch_model') and f.endswith('.bin')]
            assets = ['config.json', 'tokenizer_config.json', *weights]
            assets += [f for f in ('tokenizer.json', 'tokenizer.model') if f in files]
            item['assets'] = []
            for filename in assets:
                meta = get_hf_file_metadata(hf_hub_url(spec['model_id'], filename, revision=spec['revision']), token=token)
                item['assets'].append({'file': filename, 'bytes': meta.size, 'commit': meta.commit_hash})
            cfg = json.loads(Path(hf_hub_download(spec['model_id'], 'config.json', revision=spec['revision'], token=token)).read_text())
            item.update(access=True, model_type=cfg['model_type'], dimension=cfg.get('hidden_size', cfg.get('n_embd')),
                        blocks=cfg.get('num_hidden_layers', cfg.get('n_layer')))
        except Exception as exc:
            response = getattr(exc, 'response', None)
            item.update(access=False, error_type=type(exc).__name__, http_status=getattr(response, 'status_code', None))
        report['models'][name] = item
    return report


@app.local_entrypoint()
def main():
    report = check.remote()
    Path('results/native_models_modal_20260919/access.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
