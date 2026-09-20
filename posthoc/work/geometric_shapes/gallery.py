"""Build a local, read-only viewer for saved geometric registrations."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os

import numpy as np
from scipy.spatial import ConvexHull

from inventory import ROOT, load_cloud
from templates import make_templates


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))
    os.replace(tmp, path)


def build(out=None, limit=None):
    source = ROOT / "outputs/geometric_shapes"
    out = Path(out) if out else source
    (out / "data").mkdir(parents=True, exist_ok=True)
    entries = []
    files = sorted((source / "fits").glob("*.json"))
    if limit is not None:
        files = files[:limit]
    for path in files:
        fit = json.loads(path.read_text())
        entry = dict(fit["entry"])
        entry["fit_url"] = os.path.relpath(path, out)
        entry["source_url"] = os.path.relpath(ROOT / entry["source"], out)
        entry["screen_matches"] = fit.get("screen_matches", [])
        entry["best_name"] = fit.get("fits", [{}])[0].get("name")
        entry["best_rms"] = fit.get("fits", [{}])[0].get("symmetric_rms")
        if "normalization" in fit:
            scores, targets, labels, evr = load_cloud(entry)
            norm = fit["normalization"]
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                normalized = ((scores - norm["center"]) / norm["radius"]) @ np.asarray(norm["basis"]).T
            assert np.isfinite(normalized).all()
            ids = np.arange(len(scores))
            if len(ids) > 2500:
                ids = np.sort(np.random.default_rng(42).choice(ids, 2500, replace=False))
            data = dict(id=entry["id"], coordinates=np.round(normalized[ids], 7).tolist(),
                        targets=targets[ids].tolist(), labels=labels[ids].tolist(),
                        point_indices=ids.tolist(), point_count=len(scores),
                        displayed_count=len(ids), evr=evr.tolist())
            write_json(out / "data" / (entry["id"] + ".json"), data)
            entry["data_url"] = "data/" + entry["id"] + ".json"
        entries.append(entry)
    templates = []
    for template in make_templates():
        serialized = {**template, "points": template["points"].tolist()}
        if template["family"] in ("polyhedron_surface", "sphere"):
            triangles = ConvexHull(template["points"]).simplices
            serialized["mesh"] = {key: triangles[:, i].tolist()
                                  for i, key in enumerate(("i", "j", "k"))}
        templates.append(serialized)
    write_json(out / "alltemplates.json", templates)
    manifest = dict(generated_at=datetime.now(timezone.utc).isoformat(), entries=entries,
                    counts=dict(total=len(entries), drawable=sum("data_url" in e for e in entries),
                                groups=dict(Counter(e["group"] for e in entries))))
    write_json(out / "manifest.json", manifest)
    (out / "index.html").write_text(HTML)
    print(json.dumps(manifest["counts"]))
    return manifest


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Shape comparisons · Number representations</title>
<script src="../pca3d/plotly.min.js"></script>
<style>
:root{font:15px/1.5 system-ui,sans-serif;color:#172334;background:#f4f6fa}*{box-sizing:border-box}body{margin:0}header,main{max-width:1500px;margin:auto;padding:24px}header{padding-bottom:10px}h1{font-size:30px;line-height:1.15;margin:6px 0 12px}h2{font-size:18px;margin:0 0 12px}p{margin:7px 0}a{color:#2850a0}small,.muted{color:#586679}.eyebrow{font-size:12px;font-weight:700;letter-spacing:.1em;color:#5b667a;text-transform:uppercase}.card{background:white;border:1px solid #dce2ec;border-radius:12px;padding:18px;margin-bottom:16px}.controls{display:flex;gap:12px;flex-wrap:wrap}.controls label{flex:1;min-width:135px}label{font-size:12px;font-weight:650;color:#43516a}select,button,input{font:inherit;border:1px solid #ccd5e3;background:white;border-radius:7px;color:#172334;padding:8px}select{width:100%;margin-top:5px}button{cursor:pointer}button:hover{background:#eef3fc}button:disabled{opacity:.4;cursor:default}.twocol{display:grid;grid-template-columns:minmax(0,2.4fr) minmax(275px,1fr);gap:16px}.plotcard{padding:4px 8px 12px}#plot{height:650px;width:100%}.plotcontrols{padding:8px 12px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}.plotcontrols label{font-weight:400}.plotcontrols input{vertical-align:middle}.status{padding:12px;border-radius:8px;background:#fff6de;border:1px solid #efd397;margin:12px 0}.status.pass{background:#e9f1fb;border-color:#aac4e6}.stats{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:16px 0}.stat{background:#f5f7fb;padding:12px;border-radius:8px}.stat b{display:block;font-size:23px;line-height:1.2}.stat span{font-size:12px;color:#5b687b}.flag{display:inline-block;font-size:12px;padding:3px 7px;background:#f3eafd;color:#68418b;border-radius:5px;margin:0 4px 4px 0}.legend{display:flex;gap:12px;padding:2px 12px;font-size:12px;flex-wrap:wrap}details{margin-top:12px}summary{cursor:pointer;font-weight:650}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 10px;border-bottom:1px solid #e5eaf1;text-align:left}th{font-size:11px;color:#59667a;background:#f6f8fc;position:sticky;top:0}tbody tr:hover{background:#edf3fc;cursor:pointer}tbody tr.chosen{background:#e8effb}#ranking{max-height:510px;overflow:auto;margin-top:12px}.links{font-size:13px;display:flex;gap:12px;flex-wrap:wrap}.error{color:#a22323}.note{font-size:13px;color:#536075}.noresult{padding:150px 25px;text-align:center}.nav{display:flex;align-items:center;gap:9px;margin-top:12px}.direct{display:flex;gap:8px;margin-top:12px}.direct input{flex:1;font-family:monospace;font-size:12px;min-width:120px}@media(max-width:900px){.twocol{grid-template-columns:1fr}#plot{height:540px}header,main{padding:16px}.stats{grid-template-columns:repeat(4,1fr)}}@media(max-width:550px){.stats{grid-template-columns:1fr 1fr}#plot{height:430px}}
</style>
<header><div class="eyebrow">Saved-data analysis · No new model inference</div><h1>Do the point clouds resemble familiar shapes?</h1><p>Compare each cloud with 32 fixed-proportion geometric templates. These are fits to the <strong>first three PCA coordinates</strong>, not proof of a unique topology in the full hidden space.</p><div class="links" style="margin:14px 0"><a href="REPORT.html"><strong>Read the findings →</strong></a><a href="REPORT.html#examples">See the strongest examples →</a></div><p class="muted" id="totals">Loading completed analyses…</p></header>
<main>
<section class="card"><div class="controls">
<label>Experiment<select id="group"></select></label><label>Model<select id="model"></select></label><label>Task / comparison<select id="task"></select></label><label>Demonstrations<select id="context"></select></label><label>Saved layer<select id="layer"></select></label>
</div><div class="nav"><button id="previous">← Previous layer</button><button id="next">Next layer →</button><span id="identity" class="note"></span></div>
<details><summary>Open a particular cloud</summary><div class="direct"><input id="direct" aria-label="Cloud identifier" placeholder="original_crystal_L07"><button id="openid">Open</button></div></details>
</section>
<div class="twocol">
<section class="card plotcard"><div class="plotcontrols"><label><input id="showshape" type="checkbox" checked> Show fitted shape</label><label><input id="showdata" type="checkbox" checked> Show data</label><button id="resetcamera">Reset view</button><span class="note">Drag to rotate · scroll to zoom</span></div><div id="plot"></div><div class="legend"><span>● Colored points: saved observations</span><span style="color:#121824">◆ Dark points: fitted template</span></div><p class="note" id="sample" style="padding:0 12px"></p></section>
<aside class="card"><h2>Candidate shape</h2><label>Choose a shape<select id="candidate"></select></label><div id="flags" style="margin-top:12px"></div><div id="status" class="status"></div><div id="metrics" class="stats"></div><div id="detail" class="note"></div><div id="projection" class="note" style="margin-top:12px"></div><div id="links" class="links" style="margin-top:15px"></div></aside>
</div>
<section class="card"><h2>All candidate fits for this cloud</h2><p class="note">Lower residuals mean a closer fit. The first row is simply the smallest error among this library: it can still be a poor match. Click any row to view that candidate.</p><div id="ranking"></div></section>
<section class="card"><h2>How to read this comparison</h2><p><strong>A resemblance is not a shape identification.</strong> Projection can hide directions, separate or overlap clusters, and create apparent loops. A good projected fit does not establish the same shape in the model’s full hidden space.</p>
<p><strong>Two distances prevent an easy false match.</strong> “Data residual” measures how far observations fall from the fitted shape. “Coverage residual” checks whether the shape extends into empty regions. Both are measured on held-out points, in units of the training cloud’s RMS radius.</p>
<p><strong>The descriptive screen is deliberately explicit.</strong> Both RMS residuals must be at most 0.15, both 95th-percentile distances at most 0.30, and scale must not hit the permitted bounds. Passing is an exploratory screen, not a statistical significance test or proof of a unique object. See the analysis report for null comparisons and the interpretation of winners.</p>
<p><strong>Fixed proportions, flexible pose.</strong> We fit translation, rotation/reflection, and one uniform scale. We never stretch axes separately. Vertex, frame, and surface versions of a polyhedron are different candidates. Cylinders and cones omit end caps; surfaces are not filled solids. A finite helix is an open curve, not a topological loop. Faint shells connect sampled points on convex-surface templates for visibility; they do not change fit scores.</p>
<p><strong>Points shown versus points tested.</strong> Pose fitting uses at most 384 training observations and a 384-point approximation of each continuous template. Evaluation uses the remaining half of the observations. The viewer displays at most 2,500 saved observations, but the plotted subset does not determine the scores.</p>
<p><strong>Colored groups are predefined labels.</strong> Individual clouds use the input’s leading digit; shared comparisons use the task. Colors do not come from clustering. Shared comparisons reuse the individual observations and are not independent experiments.</p>
<div class="links"><a href="manifest.json">Cloud index</a><a href="alltemplates.json">Template coordinates</a><a href="summary.json">Analysis summary data</a><a href="REPORT.html">Written findings</a><a href="REPORT.md">Markdown report</a></div>
</section><p class="note" id="error" role="alert"></p>
</main>
<script>
const $=id=>document.getElementById(id), controls=['group','model','task','context','layer'];
const names={original:'Original · 10,000 numbers',new:'New tasks · 1,000 targets',overlay:'Shared task comparisons',crystal:'Crystal',starcoder:'StarCoderBase-3B',openllama:'OpenLLaMA-3B',original_copy:'Original number copying',copy4:'Copy four digits',reverse4:'Reverse four digits',swap_first4:'Swap first two digits',swap_last4:'Swap last two digits',numeric_copy:'Copy 1–1,000 as digits',word_copy:'Copy 1–1,000 as words',permutations:'Four digit-transformation tasks',notation:'Digits and written words'};
const familyNames={polyhedron_vertices:'Polyhedron vertices',polyhedron_edges:'Polyhedron frames',polyhedron_surface:'Polyhedron surfaces',line:'Straight line',circle:'Circle',disk:'Disk',sphere:'Sphere',cylinder:'Cylinder surface',cone:'Cone surface',torus:'Torus surface',helix:'Open helix'};
const colors=['#547da5','#e99848','#59a85c','#cf5557','#8c6bb0','#997261','#cf83bc','#888888','#b0ad52'];
let manifest,templates,entry,fit,data,selected,revision=0;
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const val=(e,key)=>e[key]===null?'none':String(e[key]);
const pretty=x=>x.replaceAll('_',' ').replace('pitch','pitch ').replace(/r(0\.)/,'tube $1');
const num=x=>Number.isFinite(x)?x.toFixed(3):'—';
async function json(url){const r=await fetch(url);if(!r.ok)throw Error(url+': HTTP '+r.status);return r.json()}
function optionLabel(key,v){if(key==='context')return v==='none'?'Original context':'Context '+v;if(key==='layer')return 'Layer '+v;return names[v]||v}
function filterThrough(level){return manifest.entries.filter(e=>controls.slice(0,level).every(key=>val(e,key)===$(key).value))}
function populate(start=0,wanted={}){
 for(let i=start;i<controls.length;i++){const key=controls[i],previous=wanted[key]===undefined?$(key).value:String(wanted[key]===null?'none':wanted[key]);let values=[...new Set(filterThrough(i).map(e=>val(e,key)))];if(key==='layer')values.sort((a,b)=>Number(a)-Number(b));else if(key==='group')values.sort((a,b)=>['original','new','overlay'].indexOf(a)-['original','new','overlay'].indexOf(b));else values.sort();$(key).innerHTML=values.map(v=>`<option value="${esc(v)}">${esc(optionLabel(key,v))}</option>`).join('');if(values.includes(previous))$(key).value=previous;}
}
function resolve(){return filterThrough(controls.length)[0]}
function setHash(){if(!entry)return;history.replaceState(null,'','#id='+encodeURIComponent(entry.id)+(selected?'&shape='+encodeURIComponent(selected):''))}
async function loadEntry(desiredCandidate){
 const token=++revision;entry=resolve();if(!entry)return;fit=null;data=null;selected=null;$('identity').textContent=entry.id;$('direct').value=entry.id;$('error').textContent='';$('plot').innerHTML='<div class="noresult">Loading saved coordinates…</div>';
 try{const results=await Promise.all([json(entry.fit_url),entry.data_url?json(entry.data_url):Promise.resolve(null)]);if(token!==revision)return;[fit,data]=results;
 const layerOptions=[...$('layer').options];$('previous').disabled=$('layer').selectedIndex===0;$('next').disabled=$('layer').selectedIndex===layerOptions.length-1;
 $('flags').innerHTML=[entry.positional?'Positional embedding':null,entry.group==='overlay'?'Reused observations':null,entry.layer===0?'Input embedding level':null].filter(Boolean).map(t=>`<span class="flag">${t}</span>`).join('');
 $('links').innerHTML=`<a href="${esc(entry.fit_url)}">All fit measurements</a><a href="${esc(entry.source_url)}">Original PCA arrays</a>`;
 if(!data||!fit.fits){$('candidate').innerHTML='<option>No informative shape</option>';$('candidate').disabled=true;$('status').className='status';$('status').textContent='Constant cloud: no geometric shape can be meaningfully fitted.';$('metrics').innerHTML='';$('detail').textContent='The saved points coincide. This is recorded as a degenerate layer, not silently omitted.';$('projection').textContent='';$('ranking').innerHTML='No fit measurements.';$('sample').textContent='';$('plot').innerHTML='<div class="noresult">All saved coordinates coincide.<br>No meaningful shape comparison.</div>';setHash();return;}
 $('candidate').disabled=false;let options='';for(const family of [...new Set(templates.map(t=>t.family))]){options+=`<optgroup label="${esc(familyNames[family]||family)}">`;for(const t of templates.filter(t=>t.family===family))options+=`<option value="${esc(t.name)}">${esc(pretty(t.name))}</option>`;options+='</optgroup>'}$('candidate').innerHTML=options;
 selected=fit.fits.some(f=>f.name===desiredCandidate)?desiredCandidate:fit.fits[0].name;$('candidate').value=selected;$('plot').innerHTML='';render(true);
 }catch(e){if(token===revision)$('error').textContent='Unable to load this analysis: '+e.message}
}
function render(reset=false){
 if(!data||!fit)return;selected=$('candidate').value;const f=fit.fits.find(f=>f.name===selected),template=templates.find(t=>t.name===selected);setHash();
 $('status').className='status'+(f.screen_pass?' pass':'');$('status').innerHTML=f.screen_pass?'<strong>Passes the descriptive fit screen.</strong><br>This remains a candidate resemblance, not an identified topology.':'<strong>Does not pass the match screen.</strong><br>This candidate leaves too much mismatch, missing coverage, or a scale-bound issue.';
 const evr=entry.evr.reduce((a,b)=>a+b,0);$('metrics').innerHTML=[['Data residual',num(f.data_rms)],['Coverage residual',num(f.coverage_rms)],['Combined residual',num(f.symmetric_rms)],['Variance in 3 PCs',(100*evr).toFixed(1)+'%']].map(([label,value])=>`<div class="stat"><b>${value}</b><span>${label}</span></div>`).join('');
 const matches=fit.screen_matches||[];let detail=`95% of held-out points are within <b>${num(f.data_p95)}</b> radius units of this template. 95% of template points are within <b>${num(f.coverage_p95)}</b> of the held-out cloud.<br><br><b>${matches.length}</b> of 32 candidates pass the descriptive screen. Lowest combined residual: <b>${esc(pretty(fit.fits[0].name))}</b>.`;
 if(entry.positional)detail+='<p><strong>Position warning:</strong> this embedding-level arrangement can reflect token position, not learned numerical meaning.</p>';
 if(entry.group==='overlay')detail+='<p>This view combines tasks in one PCA basis; task offsets can create apparent structure.</p>';
 if(/reverse|swap|permutations/.test(entry.task))detail+='<p>Transformation accuracy varied by model. A shape here does not establish that the requested rule was executed correctly.</p>';
 $('detail').innerHTML=detail;
 let projection='';const p=fit.projection||{};if(p.neighbor4_retention3d!==undefined)projection='Only '+(100*p.neighbor4_retention3d).toFixed(1)+'% of original-space 4-neighbor links are retained in this 3D projection.';else if(p.distance_spearman!==undefined&&p.distance_spearman!==null)projection='Correlation with saved full-space pair-distance ranks: '+num(p.distance_spearman)+' (512 sampled pairs).';else projection='Full-space projection check: '+(p.reason||'not available')+'.';$('projection').textContent=projection;
 $('sample').textContent=`Showing ${data.displayed_count.toLocaleString()} of ${data.point_count.toLocaleString()} observations. Pose fitted on ${fit.train_count} training points; scores evaluated on ${fit.test_count.toLocaleString()} held-out points. Colors show ${entry.group==='overlay'?'tasks':'the input’s leading digit'}.`;
 $('ranking').innerHTML='<table><thead><tr><th>Candidate</th><th>Data residual</th><th>Coverage</th><th>Combined</th><th>Screen</th></tr></thead><tbody>'+fit.fits.map(x=>`<tr data-shape="${esc(x.name)}" class="${x.name===selected?'chosen':''}"><td>${esc(pretty(x.name))}</td><td>${num(x.data_rms)}</td><td>${num(x.coverage_rms)}</td><td>${num(x.symmetric_rms)}</td><td>${x.screen_pass?'Pass':'No match'}</td></tr>`).join('')+'</tbody></table>';
 for(const row of $('ranking').querySelectorAll('tr[data-shape]'))row.onclick=()=>{$('candidate').value=row.dataset.shape;render()};
 const traces=[];if($('showdata').checked){for(const label of [...new Set(data.labels)].sort((a,b)=>a-b)){const ids=data.labels.map((v,i)=>v===label?i:-1).filter(i=>i>=0);const labelName=entry.group==='overlay'?(names[entry.task_names[label-1]]||entry.task_names[label-1]):'Starts with '+label;traces.push({type:'scatter3d',mode:'markers',name:labelName,x:ids.map(i=>data.coordinates[i][0]),y:ids.map(i=>data.coordinates[i][1]),z:ids.map(i=>data.coordinates[i][2]),customdata:ids.map(i=>data.targets[i]),marker:{size:2.5,opacity:.55,color:colors[(label-1)%colors.length]},hovertemplate:esc(labelName)+'<br>Target %{customdata}<extra></extra>'})}}
 const points=template.points.map(v=>[0,1,2].map(j=>f.scale*(v[0]*f.rotation[0][j]+v[1]*f.rotation[1][j]+v[2]*f.rotation[2][j])+f.offset[j]));if($('showshape').checked){if(template.mesh)traces.push({type:'mesh3d',x:points.map(v=>v[0]),y:points.map(v=>v[1]),z:points.map(v=>v[2]),i:template.mesh.i,j:template.mesh.j,k:template.mesh.k,color:'#59708c',opacity:.12,flatshading:template.family==='polyhedron_surface',showlegend:false,hoverinfo:'skip'});const curve=['line','circle','helix'].includes(template.family);let p=points;if(template.family==='circle')p=[...points,points[0]];traces.push({type:'scatter3d',mode:curve?'lines':'markers',name:'Fitted '+pretty(f.name),x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),line:{color:'#121824',width:6},marker:{size:template.dimension===0?7:3,color:'#121824',opacity:.8,symbol:'diamond'},hovertemplate:'Fitted '+esc(pretty(f.name))+'<extra></extra>'})}
 const bounds=[...data.coordinates,...points];const mins=[0,1,2].map(j=>Math.min(...bounds.map(p=>p[j]))),maxs=[0,1,2].map(j=>Math.max(...bounds.map(p=>p[j])));const center=mins.map((v,j)=>(v+maxs[j])/2),half=Math.max(...maxs.map((v,j)=>(v-mins[j])/2))*1.07||1;const axis=j=>({title:'Normalized axis '+(j+1),range:[center[j]-half,center[j]+half],gridcolor:'#e1e5ed',zerolinecolor:'#bcc7d6',backgroundcolor:'#fcfdff'});
 const layout={margin:{l:0,r:0,t:6,b:0},paper_bgcolor:'white',font:{family:'system-ui',size:11,color:'#43516a'},legend:{orientation:'h',y:-.03},scene:{xaxis:axis(0),yaxis:axis(1),zaxis:axis(2),aspectmode:'cube',camera:{eye:{x:1.5,y:1.5,z:1.1}}},uirevision:reset?entry.id+'-'+Date.now():entry.id};if(!reset&&$('plot').layout?.scene?.camera)layout.scene.camera=$('plot').layout.scene.camera;Plotly.react('plot',traces,layout,{responsive:true,displaylogo:false,modeBarButtonsToRemove:['toImage']});
}
controls.forEach((key,index)=>$(key).onchange=()=>{populate(index+1);loadEntry()});$('candidate').onchange=()=>render();$('showdata').onchange=()=>render();$('showshape').onchange=()=>render();$('resetcamera').onclick=()=>render(true);
for(const [button,delta] of [['previous',-1],['next',1]])$(button).onclick=()=>{const layer=$('layer');layer.selectedIndex+=delta;loadEntry()};
$('openid').onclick=()=>{const e=manifest.entries.find(e=>e.id===$('direct').value.trim());if(!e){$('error').textContent='No saved cloud has that identifier.';return}populate(0,e);loadEntry()};
window.addEventListener('hashchange',()=>{const p=new URLSearchParams(location.hash.slice(1)),e=manifest.entries.find(e=>e.id===p.get('id'));if(e){populate(0,e);loadEntry(p.get('shape'))}});
(async()=>{try{[manifest,templates]=await Promise.all([json('manifest.json'),json('alltemplates.json')]);$('totals').textContent=`${manifest.counts.total.toLocaleString()} saved views indexed · ${manifest.counts.drawable.toLocaleString()} nonconstant clouds · 32 candidate shapes. Multiple views of the same observations are not independent replications.`;const p=new URLSearchParams(location.hash.slice(1));let e=manifest.entries.find(e=>e.id===p.get('id'))||manifest.entries.find(e=>e.id==='original_crystal_L07')||manifest.entries.find(e=>e.data_url)||manifest.entries[0];if(!e)throw Error('No completed fits yet');populate(0,e);await loadEntry(p.get('shape'))}catch(e){$('error').textContent=e.message}})();
</script></html>'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    build(args.out, args.limit)
