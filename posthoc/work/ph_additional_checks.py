from pathlib import Path
import json,csv,itertools
import numpy as np
from scipy.stats import spearmanr
out=Path('/Users/mohammed.awad/Documents/Codex/2026-09-19/c/outputs/ph_analysis')
rows=json.loads((out/'layer_metrics.json').read_text());selections=json.loads((out/'selections.json').read_text())
pairs=list(csv.DictReader((out/'comparisons/pairs.csv').open()))
lookup={frozenset((r['a'],r['b'])):r for r in pairs}
models=list(selections);result={'secondary_anchors':[],'peak_neighborhoods':[],'same_index':{},'scale_sensitivity':{},'cutoff_sensitivity':[]}
for kind in ['best_rho_pc2','best_ev_pc1']:
 for a,b in itertools.combinations(models,2):
  pa=f'{a}:L{selections[a][kind]:02d}';pb=f'{b}:L{selections[b][kind]:02d}';p=lookup[frozenset([pa,pb])]
  result['secondary_anchors'].append(dict(selection=kind,a=pa,b=pb,**{k:float(p[k]) for k in ['h0','h1','landscape5']}))
for model in models:
 rr=[r for r in rows if r['model']==model]
 for key,metric in [('best_rho_pc1','rho_pc1'),('best_rho_pc2','rho_pc2'),('best_ev_pc1','ev_pc1'),('best_ev_pc12','ev_pc12')]:
  center=selections[model][key];window=[r for r in rr if abs(r['layer']-center)<=2]
  result['peak_neighborhoods'].append(dict(model=model,selection=key,center=center,layers=[r['layer'] for r in window],selector_range=[min(r[metric] for r in window),max(r[metric] for r in window)],h1_max_range=[min(r['h1_max'] for r in window),max(r['h1_max'] for r in window)],h1_count_range=[min(r['h1_count'] for r in window),max(r['h1_count'] for r in window)]))
 result['scale_sensitivity'][model]=dict(scale_min=min(r['distance_scale'] for r in rr),scale_max=max(r['distance_scale'] for r in rr),raw_longest_lifetime_depth_spearman=float(spearmanr([r['depth'] for r in rr],[r['h1_raw_max'] for r in rr]).statistic),normalized_longest_lifetime_depth_spearman=float(spearmanr([r['depth'] for r in rr],[r['h1_max'] for r in rr]).statistic))
 for part in ['early','middle','late','final']:
  subset=[r for r in rr if r['phase']==part] if part!='final' else rr[-1:]
  result['cutoff_sensitivity'].append(dict(model=model,phase=part,**{str(t):float(np.median([r[f'h1_count_gt_{t}'] for r in subset])) for t in [.01,.02,.05,.1,.2]}))
cross=[r for r in pairs if r['same_model']=='False'];same=[r for r in cross if r['a'].split(':L')[1]==r['b'].split(':L')[1]]
for k in ['h0','h1','landscape5']:result['same_index'][k]=dict(pairs=len(same),median=float(np.median([float(r[k]) for r in same])),all_cross_median=float(np.median([float(r[k]) for r in cross])))
(out/'additional_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['peak_neighborhoods','cutoff_sensitivity']},indent=2))
