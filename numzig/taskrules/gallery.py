"""Portable local gallery for completed task-rule analyses; no inference or cloud calls."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from numzig.fullrange.storage import atomic, read_json


MODELS = ('crystal', 'starcoderbase-3b', 'openllama-3b')
MODEL_LEVELS = {'crystal': 33, 'starcoderbase-3b': 37, 'openllama-3b': 27}
TASKS = ('copy4', 'reverse4', 'swap_first4', 'swap_last4', 'numeric_copy', 'word_copy')
TASK_NAMES = dict(copy4='Copy four digits', reverse4='Reverse four digits', swap_first4='Swap first two digits',
                  swap_last4='Swap last two digits', numeric_copy='Copy numbers 1–1000', word_copy='Copy words 1–1000')


def _write(path, content):
    content = content.encode() if isinstance(content, str) else content
    if not path.exists() or path.read_bytes() != content:
        atomic(path, lambda stream: stream.write(content))


def _json(path, value):
    _write(path, json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False))


def _behavior(root):
    summaries = {}
    for model in MODELS:
        path = root / 'pilot' / model / 'behavior.json'
        if not path.exists():
            continue
        groups = defaultdict(list)
        for row in read_json(path):
            groups[(row['task'], str(row['context_id']))].append(row)
        for (task, context), rows in groups.items():
            correct = sum(bool(r.get('exact_match', r.get('exact_correct', False))) for r in rows)
            summaries[(model, task, context)] = dict(model=model, task=task, context_id=context,
                correct=correct, n=len(rows), exact_accuracy=correct / len(rows),
                generation='greedy pilot output; complete answer exact match',
                warning='Below 80% pilot exact accuracy; geometry is not evidence of successful rule execution.'
                        if correct / len(rows) < .8 else None)
    return summaries


def build_gallery(root):
    """Create index.html and gallery assets, keeping original analyses unchanged.

    Groups without a complete summary or required files are omitted and listed
    as missing. A partial export is clearly labeled, never declared complete.
    """
    root = Path(root)
    index = root / 'index.html'
    if index.exists() and '<title>Digit rules and written numbers</title>' not in index.read_text():
        raise ValueError('Refusing to replace an unrelated existing index.html; use the new task-rule result directory')
    from plotly.offline import get_plotlyjs
    gallery = root / 'gallery'
    gallery.mkdir(parents=True, exist_ok=True)
    _write(gallery / 'plotly.min.js', get_plotlyjs())
    dataset_path = root / 'dataset.json'
    dataset = read_json(dataset_path) if dataset_path.exists() else []
    lookup = {(r['task'], str(r['context_id']), int(r['target'])): r for r in dataset}
    behavior = _behavior(root)
    groups, missing, warnings, comparisons = [], [], [], []
    for model in MODELS:
        for task in TASKS:
            for context in ('0', '1'):
                path = root / model / 'analysis' / f'{task}_ctx{context}' / 'summary.json'
                group_id = f'{model}/{task}/ctx{context}'
                if not path.exists():
                    missing.append(group_id)
                    continue
                summary = read_json(path)
                levels = summary.get('layers', [])
                groupdir = path.parent
                if (summary.get('status') != 'complete' or summary.get('model') != model or summary.get('task') != task
                    or str(summary.get('context_id')) != context or len(levels) != summary.get('layer_count')
                    or not levels or any(not (groupdir / layer['arrays']).is_file()
                        or any(not (groupdir / figure).is_file() for figure in layer.get('figures', [])) for layer in levels)):
                    missing.append(group_id)
                    warnings.append(f'{group_id}: incomplete summary or missing artifacts')
                    continue
                layer_records = []
                for layer in levels:
                    level = int(layer['layer'])
                    figures = {}
                    for figure in layer.get('figures', []):
                        for kind in ('pca', 'graph', 'ph'):
                            if str(figure).endswith(f'_{kind}.png'):
                                figures[kind] = str((groupdir / figure).relative_to(root))
                    record = dict(layer=level, status=layer['status'], figures=figures,
                        arrays=str((groupdir / layer['arrays']).relative_to(root)),
                        explained_variance_ratio=layer.get('pca', {}).get('explained_variance_ratio', []),
                        rho=layer.get('pca', {}).get('spearman_absolute', []), ph=layer.get('ph', {}))
                    if layer['status'] == 'valid':
                        with np.load(groupdir / layer['arrays'], allow_pickle=False) as archive:
                            scores = archive['scores3'] if 'scores3' in archive else archive['scores']
                            targets = archive['targets'].astype(int)
                            if scores.shape != (len(targets), 3) or not np.isfinite(scores).all():
                                raise ValueError(f'Invalid PCA3 coordinates for {group_id} level {level}')
                            rows = [lookup.get((task, context, int(t)), {}) for t in targets]
                            values = dict(scores=scores.tolist(), targets=targets.tolist(),
                                point_ids=archive['point_ids'].astype(int).tolist(),
                                input_text=[r.get('input_text', str(t)) for r, t in zip(rows, targets)],
                                expected_output=[r.get('expected_output', '?') for r in rows],
                                input_first_digit=archive['input_first_digit'].astype(str).tolist(),
                                output_first_label=archive['output_first_label'].astype(str).tolist(),
                                token_counts=archive['token_counts'].astype(int).tolist(),
                                explained_variance_ratio=archive['explained_variance_ratio'].tolist(),
                                output_label_kind=layer.get('output_label_kind', 'expected output first label'))
                        data_path = gallery / 'data' / f'{model}_{task}_ctx{context}_layer{level:02d}.json'
                        _json(data_path, values)
                        record['data3d'] = str(data_path.relative_to(root))
                    layer_records.append(record)
                groups.append(dict(model=model, task=task, task_name=TASK_NAMES[task], context_id=context,
                    point_count=summary['point_count'], layers=layer_records,
                    behavior=behavior.get((model, task, context)), summary=str(path.relative_to(root))))
        comparison_root = root / model / 'comparison'
        for path in sorted(comparison_root.glob('ctx*/*/layer_*.json')):
            receipt = read_json(path)
            family = receipt.get('family')
            if family not in ('permutations', 'notation'):
                continue
            figures = {color: str(path.with_name(path.stem + f'_{color}.png').relative_to(root))
                       for color in ('input', 'output') if path.with_name(path.stem + f'_{color}.png').is_file()}
            if not figures or not path.with_suffix('.npz').exists():
                continue
            comparisons.append(dict(model=model, context_id=str(receipt['context_id']), family=family,
                layer=int(receipt['layer']), tasks=receipt['tasks'], figures=figures,
                point_count=receipt['count_per_condition'],
                explained_variance_ratio=receipt['shared_pca_variance_ratio'],
                metrics=receipt['metrics'], arrays=str(path.with_suffix('.npz').relative_to(root))))
    expected_groups = len(MODELS) * len(TASKS) * 2
    figure_count = sum(len(layer['figures']) for group in groups for layer in group['layers'])
    missing_levels = [f'{g["model"]}/{g["task"]}/ctx{g["context_id"]}: missing expected saved levels'
                      for g in groups if {l['layer'] for l in g['layers']} != set(range(MODEL_LEVELS[g['model']]))]
    expected_comparisons = sum(MODEL_LEVELS.values()) * 2 * 2
    complete = not missing and not missing_levels and len(comparisons) == expected_comparisons
    manifest = dict(schema=1, title='Digit rules and written numbers', groups=groups, comparisons=comparisons,
        completed_groups=len(groups), expected_groups=expected_groups, status='complete' if complete else 'partial',
        missing_groups=missing, warnings=warnings, figure_count=figure_count,
        missing_levels=missing_levels, expected_comparison_levels=expected_comparisons,
        completed_comparison_levels=len(comparisons),
        comparison_figure_count=sum(len(c['figures']) for c in comparisons),
        three_d_view_count=sum('data3d' in layer for group in groups for layer in group['layers']),
        behavior=list(behavior.values()), limitations=[
            'Expected-answer colors are predefined labels, not generated answers or learned cluster assignments.',
            'Pilot accuracy is measured on a small subset; it is not accuracy for every plotted target.',
            'Each task has two distinct saved demonstration contexts, analyzed separately.',
            'Independent PCA axes are not shared across tasks; matched comparisons use one joint PCA fit.',
            'PH and nearest neighbors use original hidden-space distances. The 3D plot is a projection.',
            'Written-number decimal digit colors refer to numeric values; output colors show the first output word.',
            'The last saved state includes the model\'s native final normalization.',
            'These descriptive comparisons do not establish a causal digit-routing mechanism.',
        ])
    _json(gallery / 'manifest.json', manifest)
    _write(root / 'index.html', HTML)
    lines = ['# Digit rules and written numbers', '',
        f'**Export status: {manifest["status"]}.** {len(groups)}/{expected_groups} saved task/context/model group summaries; '
        f'{len(comparisons)}/{expected_comparisons} shared-comparison levels.', '',
        f'{figure_count} per-layer figures; {manifest["comparison_figure_count"]} shared-PCA comparison figures; '
        f'{manifest["three_d_view_count"]} interactive 3D views.', '',
        '[Open gallery](index.html)', '', '| Model | Task | Context | Pilot exact answers | Completed layers |',
        '|---|---|---:|---:|---:|']
    by_group = {(g['model'], g['task'], g['context_id']): g for g in groups}
    for model in MODELS:
        for task in TASKS:
            for context in ('0', '1'):
                b = behavior.get((model, task, context))
                accuracy = f'{b["correct"]}/{b["n"]} ({b["exact_accuracy"]:.1%})' if b else 'Unavailable'
                if b and b['warning']:
                    accuracy += ' — below 80%'
                g = by_group.get((model, task, context))
                lines.append(f'| {model} | {TASK_NAMES[task]} | {context} | {accuracy} | {len(g["layers"]) if g else "Pending"} |')
    lines += ['', '## Interpretation limits', '', *('- ' + item for item in manifest['limitations'])]
    if missing:
        lines += ['', '## Incomplete groups', '', *('- ' + item for item in missing)]
    if missing_levels:
        lines += ['', '## Incomplete depth coverage', '', *('- ' + item for item in missing_levels)]
    _write(root / 'SUMMARY.md', '\n'.join(lines) + '\n')
    return manifest


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Digit rules and written numbers</title><script src="gallery/plotly.min.js"></script>
<style>
:root{color-scheme:light;font-family:Inter,system-ui,sans-serif;color:#1b2a32;background:#f2f5f6}*{box-sizing:border-box}
body{margin:0}header{padding:24px 30px 18px;background:#102e38;color:white}h1{margin:0 0 7px;font-size:27px}
p{line-height:1.5;margin:7px 0}header a{color:#baf1ed}.shell{padding:20px 28px;max-width:1800px;margin:auto}
.controls{display:flex;gap:13px;flex-wrap:wrap;background:white;border:1px solid #d9e2e5;border-radius:9px;padding:15px}
label{display:flex;flex-direction:column;gap:5px;font-size:13px;font-weight:600}select,button{font:inherit;padding:9px;border:1px solid #a7bbc3;border-radius:5px;background:white;color:#132e39}
button{cursor:pointer}button:hover{background:#e8f5f3}#behavior,#details,#notice{margin:13px 0;padding:11px 14px;border-radius:6px;background:white}
#behavior.warn,#notice{background:#fff1d6;color:#724b08}#behavior.good{background:#e4f3eb;color:#155037}
.viewbox{min-height:500px;background:white;border:1px solid #d9e2e5;border-radius:9px;padding:10px;position:relative}
#plot{height:72vh;min-height:500px}#figure{display:block;max-width:100%;height:auto;margin:auto}.hidden{display:none!important}
#colors{display:flex;align-items:center;gap:10px;margin:11px 0}#colors span{font-size:13px;color:#435e69}
#empty{padding:120px 20px;text-align:center;color:#49626e}.footer{font-size:13px;color:#415761;margin-top:15px}
details{margin-top:18px;background:white;padding:13px;border-radius:6px}li{margin:7px 0;line-height:1.45}a{color:#086c73}
@media(max-width:700px){.shell{padding:12px}header{padding:18px}#plot{height:65vh}select{max-width:100%}.controls label{flex:1 1 140px}}
</style></head><body>
<header><h1>Digit rules and written numbers</h1><p>Matched input numbers, changed output rules, and two saved demonstration contexts.</p>
<p id="status">Loading saved results…</p><a href="SUMMARY.md">Results inventory and pilot accuracy</a> · <a href="gallery/manifest.json">Complete gallery metadata</a></header>
<main class="shell"><div id="notice" class="hidden"></div>
<div class="controls">
<label>Model<select id="model"></select></label><label>Experiment<select id="task"></select></label>
<label>Context<select id="context"></select></label><label>Saved level<select id="layer"></select></label>
<label>View<select id="view"><option value="pca">2D · three label panels</option><option value="graph">2D · original k=4 graph</option>
<option value="ph">Persistent homology · H0 / H1</option><option value="3d">Interactive 3D PCA</option>
<option value="comparison_input">Matched tasks · input colors</option><option value="comparison_output">Matched tasks · output colors</option></select></label>
<label>&nbsp;<button id="prev" aria-label="Previous saved level">← Previous</button></label><label>&nbsp;<button id="next" aria-label="Next saved level">Next →</button></label></div>
<div id="behavior"></div><div id="details"></div>
<div id="colors" class="hidden"><label>3D colors<select id="color"><option value="input_first_digit">Input leading decimal digit</option>
<option value="output_first_label">Expected output first digit / word</option><option value="token_counts">Prompt token count</option></select></label>
<span>Drag to rotate · scroll to zoom · double-click to reset · hover to inspect a target</span></div>
<div class="viewbox"><img id="figure" alt="Selected saved analysis figure" class="hidden"><div id="plot" class="hidden"></div><div id="empty" class="hidden"></div></div>
<p class="footer" id="links"></p><details><summary>What these views do and do not show</summary><ul id="limits"></ul></details></main>
<script>
const $=id=>document.getElementById(id);let M=null,requestId=0,cache=new Map();
const digitColors=['#333333','#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22'];
const start=new URLSearchParams(location.hash.slice(1));
function options(id,items,wanted){const e=$(id);e.replaceChildren();for(const [value,text] of items){const o=document.createElement('option');o.value=value;o.textContent=text;e.append(o)}if([...e.options].some(o=>o.value===String(wanted)))e.value=String(wanted)}
function unique(values){return [...new Set(values)]}
function modelChange(){const gs=M.groups.filter(g=>g.model===$('model').value);options('task',gs.filter((g,i)=>gs.findIndex(h=>h.task===g.task)===i).map(g=>[g.task,g.task_name]),$('task').value||start.get('task'));taskChange()}
function taskChange(){const gs=M.groups.filter(g=>g.model===$('model').value&&g.task===$('task').value);options('context',gs.map(g=>[g.context_id,'Context '+g.context_id]),$('context').value||start.get('context'));contextChange()}
function group(){return M.groups.find(g=>g.model===$('model').value&&g.task===$('task').value&&g.context_id===$('context').value)}
function contextChange(){const g=group();options('layer',g?g.layers.map(l=>[String(l.layer),l.layer===0?'0 · input embedding':String(l.layer)]):[],$('layer').value||start.get('layer'));render()}
function empty(text){$('empty').textContent=text;$('empty').classList.remove('hidden')}
function addLink(text,url){const a=document.createElement('a');a.href=url;a.textContent=text;a.target='_blank';a.rel='noopener';if($('links').childNodes.length)$('links').append(' · ');$('links').append(a)}
async function render(){const mine=++requestId,g=group();for(const id of ['figure','plot','empty','colors'])$(id).classList.add('hidden');$('links').replaceChildren();if(!g){empty('No completed analysis groups are available yet. The pilot table remains in the results inventory.');return}
 const l=g.layers.find(l=>l.layer===Number($('layer').value)),view=$('view').value,b=g.behavior;
 $('behavior').className=b&&b.exact_accuracy>=.8?'good':'warn';$('behavior').textContent=b?`Pilot: ${b.correct}/${b.n} exact answers (${(100*b.exact_accuracy).toFixed(1)}%). ${b.warning||'Expected-answer colors still describe the requested answer, not each generated answer.'}`:'Pilot accuracy is unavailable. Do not interpret expected-answer coloring as task success.';
 let ev=l.explained_variance_ratio;let text=`${g.point_count.toLocaleString()} points · one task/context · saved level ${l.layer}. `;
 if(l.status==='valid')text+=`PCA variance: 2D ${(100*(ev[0]+ev[1])).toFixed(1)}%; 3D ${(100*ev.reduce((a,b)=>a+b,0)).toFixed(1)}%. `;else text+='Constant or numerically degenerate cloud: geometry unavailable. ';
 if(l.ph.status==='complete')text+=`At normalized connection distance 1: β₀=${l.ph.betti_at_one[0]}, β₁=${l.ph.betti_at_one[1]}.`;
 $('details').textContent=text;addLink('Group metrics',g.summary);addLink('Original saved arrays',l.arrays);
 history.replaceState(null,'','#'+new URLSearchParams({model:g.model,task:g.task,context:g.context_id,layer:l.layer,view}).toString());
 if(view.startsWith('comparison')){const family=g.task.endsWith('4')?'permutations':'notation',c=M.comparisons.find(c=>c.model===g.model&&c.context_id===g.context_id&&c.family===family&&c.layer===l.layer);const color=view==='comparison_input'?'input':'output';if(!c||!c.figures[color]){empty('This shared-coordinate comparison is not available. Written-number comparison uses underlying input-value colors.');return}showImage(c.figures[color]);$('details').textContent=`Shared PCA fit across ${c.tasks.length} matched conditions, ${c.point_count.toLocaleString()} targets per condition. Same axes and scale in every panel. Joint 2D explained variance: ${(100*(c.explained_variance_ratio[0]+c.explained_variance_ratio[1])).toFixed(1)}%.`;
 const low=M.groups.filter(x=>x.model===g.model&&x.context_id===g.context_id&&c.tasks.includes(x.task)&&(!x.behavior||x.behavior.exact_accuracy<.8)).map(x=>x.task_name);if(low.length){$('behavior').className='warn';$('behavior').textContent='Comparison includes absent or below-80% pilot accuracy: '+low.join(', ')+'. These shapes cannot establish successful transformation.'}addLink('Matched comparison arrays',c.arrays);return}
 if(view!=='3d'){if(l.figures[view])showImage(l.figures[view]);else empty('This figure is unavailable for the selected level. Constant clouds do not have informative PCA, kNN or PH.');return}
 if(!l.data3d){empty('No valid 3D projection for this constant or degenerate level.');return}
 $('colors').classList.remove('hidden');$('plot').classList.remove('hidden');
 try{let data=cache.get(l.data3d);if(!data){const response=await fetch(l.data3d);if(!response.ok)throw Error(response.status);data=await response.json();cache.set(l.data3d,data);if(cache.size>8)cache.delete(cache.keys().next().value)}if(mine!==requestId)return;plot3d(data,g,l)}catch(e){$('plot').classList.add('hidden');empty('Could not load saved 3D data: '+e.message)}
}
function showImage(path){$('figure').src=path;$('figure').classList.remove('hidden');addLink('Open full-size figure',path)}
function plot3d(d,g,l){const key=$('color').value,values=d[key],traces=[];const make=(indices,label,color)=>({type:'scatter3d',mode:'markers',name:label,x:indices.map(i=>d.scores[i][0]),y:indices.map(i=>d.scores[i][1]),z:indices.map(i=>d.scores[i][2]),customdata:indices.map(i=>[d.targets[i],d.input_text[i],d.expected_output[i],d.token_counts[i]]),hovertemplate:'Target %{customdata[0]}<br>Input: %{customdata[1]}<br>Expected: %{customdata[2]}<br>Prompt tokens: %{customdata[3]}<extra>%{fullData.name}</extra>',marker:{size:3,opacity:.72,color}});
 if(key==='token_counts'){const t=make(d.targets.map((_,i)=>i),'Prompt tokens',values);t.marker.colorscale='Viridis';t.marker.showscale=true;t.marker.colorbar={title:'Tokens'};traces.push(t)}else{const labels=unique(values).sort((a,b)=>String(a).localeCompare(String(b),undefined,{numeric:true}));labels.forEach((label,j)=>{const indices=values.map((v,i)=>v===label?i:-1).filter(i=>i>=0);const color=/^[0-9]$/.test(String(label))?digitColors[Number(label)]:`hsl(${(j*137.508)%360},65%,45%)`;traces.push(make(indices,String(label),color))})}
 const ev=d.explained_variance_ratio;Plotly.react('plot',traces,{margin:{l:0,r:0,t:38,b:0},title:{text:`${g.task_name} · ${g.model} · level ${l.layer}`,font:{size:17}},scene:{xaxis:{title:`PC1 (${(100*ev[0]).toFixed(1)}%)`},yaxis:{title:`PC2 (${(100*ev[1]).toFixed(1)}%)`},zaxis:{title:`PC3 (${(100*ev[2]).toFixed(1)}%)`},aspectmode:'data'},legend:{itemsizing:'constant'},uirevision:`${g.model}/${g.task}/${g.context_id}/${l.layer}`,paper_bgcolor:'white'},{responsive:true,displaylogo:false});
}
function shift(step){const e=$('layer');if(!e.options.length)return;e.selectedIndex=Math.max(0,Math.min(e.options.length-1,e.selectedIndex+step));render()}
$('model').onchange=modelChange;$('task').onchange=taskChange;$('context').onchange=contextChange;$('layer').onchange=render;$('view').onchange=render;$('color').onchange=render;$('prev').onclick=()=>shift(-1);$('next').onclick=()=>shift(1);
fetch('gallery/manifest.json').then(r=>{if(!r.ok)throw Error(r.status);return r.json()}).then(m=>{M=m;$('status').textContent=`${m.completed_groups}/${m.expected_groups} saved groups · ${m.figure_count.toLocaleString()} layer figures · ${m.three_d_view_count.toLocaleString()} 3D views · ${m.status.toUpperCase()}`;if(m.status!=='complete'){$('notice').textContent=`Partial export: ${m.missing_groups.length} groups absent or incomplete; ${m.missing_levels.length} groups have incomplete depth coverage; ${m.completed_comparison_levels}/${m.expected_comparison_levels} shared-comparison levels saved. Available completed summaries are selectable.`;$('notice').classList.remove('hidden')}for(const text of m.limitations){const li=document.createElement('li');li.textContent=text;$('limits').append(li)}options('model',unique(m.groups.map(g=>g.model)).map(x=>[x,x]),start.get('model'));if(start.has('view'))$('view').value=start.get('view');modelChange()}).catch(e=>{$('status').textContent='Gallery could not load.';empty('Serve this folder using a local HTTP server, then open index.html. Details: '+e.message)});
</script></body></html>'''
