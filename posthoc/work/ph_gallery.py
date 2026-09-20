from pathlib import Path
import json, sys
from html import escape
import numpy as np

repo = Path('/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL')
sys.path.insert(0, str(repo))
from numzig.fullrange.storage import Store, digest

root = Path('/Users/mohammed.awad/Documents/Codex/2026-09-19/c/outputs/ph_fullrange_overnight')
data = root / 'results/results'
models = {'crystal': ('Crystal', 33), 'starcoderbase-3b': ('StarCoderBase-3B', 37), 'openllama-3b': ('OpenLLaMA-3B', 27)}
rows = []
for p in sorted(data.glob('*/layer_*/manifest.json')):
    s = Store(p.parent)
    if not all(s.valid(k) for k in ('h0', 'h1', 'plots')):
        continue
    r = json.loads((p.parent/'metrics.json').read_text())
    cfg = s.manifest['config']
    assert cfg['code_sha256'] == digest(repo/'numzig/ph_overnight.py')
    assert r['point_count'] == 10000 and cfg['source']['shape'][0] == 10000
    assert cfg['coeff'] == 2 and r['filtration_dtype'] == 'float32' and r['normalized_threshold'] is None
    assert r['model'] == p.parent.parent.name and r['layer'] == int(p.parent.name.split('_')[1])
    with np.load(p.parent/'h0.npz') as a, np.load(p.parent/'h1.npz') as b:
        assert a['raw'].shape == (10000,2) and np.isinf(a['raw'][:,1]).sum() == 1
        assert np.array_equal(a['point_ids'], np.arange(10000))
        assert np.isfinite(b['raw']).all() and not b['right_censored'].any()
        assert len(b['raw']) == r['h1_finite_count'] and np.all(b['raw'][:,1] >= b['raw'][:,0])
        scale = r['normalization']['scale']
        assert np.array_equal(b['normalized'], b['raw']/scale if scale else b['raw'])
    rows.append(dict(model=r['model'], layer=r['layer'], path=p.parent.relative_to(root).as_posix()))
expected = {(m,l) for m,(_,n) in models.items() for l in range(n)}
found = {(r['model'],r['layer']) for r in rows}
assert len(rows) == len(found) == 96
assert expected-found == {('starcoderbase-3b',0)}
validation = dict(validated_layers=len(rows), expected_layers=97, missing=sorted(expected-found),
    checks='Saved artifact checksums, configuration, all 10,000 IDs, diagram finiteness and normalization. No independent full H1 recomputation.', layers=rows)
(root/'validation.json').write_text(json.dumps(validation,indent=2))
sections=[]
for model,(label,n) in models.items():
    cards=[]
    for r in rows:
        if r['model'] != model: continue
        path=escape(r['path']); layer=r['layer']; title=f'Layer {layer}' + (' · embedding' if layer==0 else '')
        cards.append(f'<article><h3>{title}</h3><a href="{path}/persistence.png"><img loading="lazy" src="{path}/persistence.png" alt="{escape(label)} {title} PH graphs"></a><p><a href="{path}/persistence.png">Open full-size graph</a> · <a href="{path}/persistence.svg">SVG</a> · <a href="{path}/metrics.json">Saved metrics</a></p></article>')
    sections.append(f'<section id="{model}"><h2>{label} <small>{len(cards)}/{n} levels</small></h2>' + ('<p class="note">Layer 0 H1 timed out; its completed H0 is saved, but a full PH figure is unavailable.</p>' if model=='starcoderbase-3b' else '') + '<div class="grid">'+''.join(cards)+'</div></section>')
html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PH results · three models</title><style>body{font:16px/1.5 system-ui;margin:0;background:#f4f6f9;color:#182238}main{max-width:1450px;margin:auto;padding:32px}h1{font-size:36px;margin-bottom:8px}h2{margin-top:45px}small{font-size:17px;color:#536278}a{color:#1557a0}nav{display:flex;gap:24px;flex-wrap:wrap;padding:16px 0}.note{background:#fff1d7;padding:14px;border-radius:8px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,450px),1fr));gap:18px}article{background:white;padding:16px;border-radius:12px;border:1px solid #dde3ec}article h3{margin:0 0 12px}img{width:100%;height:auto}article p{font-size:14px}section{scroll-margin-top:15px}</style><main><h1>Persistent homology results</h1><p>96 of 97 levels · 3 models · 10,000 targets per level · H0 and H1</p><p>Click any figure to see it at full size. Each saved figure contains a persistence diagram, barcodes and Betti curves. Layer 0 is the embedding level.</p><p class="note">All completed figures and diagrams are downloaded and checksum-validated. StarCoder layer 0 H1 remains incomplete. The original cloud packaging failed; this gallery was assembled locally from saved artifacts, without rerunning PH.</p><nav><a href="#crystal">Crystal</a><a href="#starcoderbase-3b">StarCoderBase-3B</a><a href="#openllama-3b">OpenLLaMA-3B</a><a href="validation.json">Validation record</a></nav>'''+''.join(sections)+'</main></html>'
(root/'index.html').write_text(html)
print(json.dumps({k:v for k,v in validation.items() if k!='layers'},indent=2))
