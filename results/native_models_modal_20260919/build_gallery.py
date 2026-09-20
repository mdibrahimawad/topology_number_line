"""Link validated saved outputs without modifying any experiment directory."""
from pathlib import Path
import hashlib
import html
import json
import os

B = Path(__file__).resolve().parent
CRYSTAL = B.parent / 'crystal_fullrange_modal_20260919T123307Z/full_complete/crystal_fullrange_1_10000_ctx1234_k4_seed42'
ROOTS = [B / 'full_complete' / (m + '_fullrange_1_10000_ctx1234_k4_seed42')
         for m in ('starcoderbase-3b', 'openllama-3b')] + [CRYSTAL]


def build():
    output = B / 'gallery'
    output.mkdir(exist_ok=True)
    def link(path):
        assert path.is_file(), path
        return html.escape(os.path.relpath(path, output), quote=True)
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Numerical representation graphs</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;color:#18212b}a{color:#125ab5}img{width:100%;height:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:24px}figure{margin:0}nav a{margin-right:24px}section{margin:40px 0}figcaption{padding:8px}summary{cursor:pointer;font-weight:600}table{border-collapse:collapse}td,th{padding:8px 14px;border:1px solid #ddd;text-align:left}</style>
<h1>Numerical representations · 10,000 targets</h1>
<p>Exact k=4 neighbors are selected in each layer’s original hidden space; PCA only supplies the display coordinates. Every layer retains all 10,000 targets. Colors indicate leading decimal digit.</p>
<nav><a href="#starcoderbase-3b">StarCoderBase-3B</a><a href="#openllama-3b">OpenLLaMA-3B</a><a href="#crystal">Crystal</a>''',
             f'<a href="{link(B / "comparison/index.html")}">Three-model comparison</a></nav><div class="grid">']
    for root in ROOTS[:2]:
        meta = json.loads((root / 'tokenizer_provenance.json').read_text())
        final = meta['expected_levels'] - 1
        image = root / f'figures/layer_{final:02d}_graph.png'
        parts.append(f'<figure><a href="{link(image)}"><img src="{link(image)}" alt="{root.name.split("_fullrange")[0]} final-layer PCA and kNN graph"></a></figure>')
    parts += ['</div><p>These are descriptive comparisons. Architectures, tokenizers, widths, layer counts and precision differ: the new models use float32; Crystal used bfloat16 inference. PCA frames are independent. The paper’s tokenization controls have not been reproduced.</p>']
    for root in ROOTS:
        manifest = json.loads((root / 'manifest.json').read_text())
        assert all(manifest['stages'].get(k) == 'complete' for k in ('extraction', 'analysis', 'plots'))
        name = root.name.split('_fullrange')[0]
        meta = json.loads((root / 'tokenizer_provenance.json').read_text())
        parts += [f'<section id="{name}"><h2>{html.escape(manifest["config"]["model_id"])}</h2>',
                  f'<p>{html.escape(meta["hidden_state_semantics"])}</p>',
                  f'<p><a href="{link(root / "viewer/index.html")}">Interactive viewer</a> · <a href="{link(root / "figures/overview.png")}">All-layer contact sheet</a> · <a href="{link(root / "figures/depth.png")}">Depth metrics</a> · <a href="{link(root / "layer_metrics.json")}">Numerical metrics</a> · <a href="{link(root / "manifest.json")}">Manifest</a></p>',
                  '<div class="grid">']
        for layer in range(meta['expected_levels']):
            image = root / f'figures/layer_{layer:02d}_graph.png'
            parts += [f'<figure><a href="{link(image)}"><img loading="lazy" src="{link(image)}" alt="{name} level {layer} PCA and kNN"></a><figcaption>Level {layer} · ']
            for mode, label in [('digits', 'Digit panels'), ('heatmap', 'Transition heatmap'), ('magnitude', 'Magnitude colors')]:
                path = root / f'figures/layer_{layer:02d}_{mode}.png'
                parts.append(f'<a href="{link(path)}">{label}</a> · ')
            parts.append('</figcaption></figure>')
        parts.append('</div></section>')
    parts.append('</html>')
    path = output / 'index.html'
    path.write_text('\n'.join(parts))
    print(json.dumps({'gallery': str(path), 'layers': 97, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    build()
