from pathlib import Path
import json, hashlib
import numpy as np
source=Path('outputs/digit_order_check/metrics.json')
records=json.loads(source.read_text())
out=Path('outputs/centroid_neighbors');out.mkdir(exist_ok=True)
models=[('Crystal',32),('StarCoder',36),('OpenLLaMA',26)]
results=[]
for name,last in models:
 for layer in [last-1,last]:
  selected=[r for r in records if r['model']==name and r['layer']==layer and r['space']=='hidden' and r['subset']=='all'];assert len(selected)==1
  d=np.array(selected[0]['pair_distances'],dtype=float)
  assert d.shape==(9,9) and np.isfinite(d).all() and np.allclose(d,d.T) and np.allclose(np.diag(d),0)
  groups=[]
  for i in range(9):
   order=sorted((j for j in range(9) if j!=i),key=lambda j:(d[i,j],j))
   ranked=[dict(rank=k+1,leading_digit=j+1,euclidean_distance=float(d[i,j])) for k,j in enumerate(order)]
   assert len(set(x['leading_digit'] for x in ranked[:4]))==4 and all(x['leading_digit']!=i+1 for x in ranked)
   groups.append(dict(leading_digit=i+1,ranked_other_centroids=ranked,top4=[j+1 for j in order[:4]],fourth_to_fifth_distance_gap=float(d[i,order[4]]-d[i,order[3]])))
  results.append(dict(model=name,layer=layer,stage='final' if layer==last else 'second-last',groups=groups,pairwise_euclidean_distances=d.tolist()))
metadata=dict(source=str(source.resolve()),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),method='Euclidean distance between arithmetic-mean leading-digit group centroids in the original saved hidden space. All targets 1..10000. Self excluded. Ascending distance; exact ties broken by leading digit. Final saved layers include the native final model normalization.',results=results)
(out/'centroid_neighbors.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
lines=['# Four nearest leading-digit group centroids','','Distances are computed between group means in the original saved hidden space, before PCA. Each mean includes every target with that leading digit among 1–10,000. The source group itself is excluded. Lists run from closest to fourth closest; this is a directed neighbor ranking, so being neighbors need not be mutual. Final layers include each model\'s native final normalization.','']
for stage in ['final','second-last']:
 chosen=[next(r for r in results if r['model']==name and r['stage']==stage) for name,_ in models]
 lines += [f'## {stage.title()} layers','','| Source leading digit | '+' | '.join(f"{r['model']} L{r['layer']}" for r in chosen)+' |','|---:|---|---|---|']
 for i in range(9):
  cells=[', '.join(str(x) for x in r['groups'][i]['top4']) for r in chosen]
  lines.append(f"| {i+1} | "+' | '.join(cells)+' |')
 lines.append('')
for r in results:
 lines += [f"## Distances: {r['model']} layer {r['layer']}",'','Distances below are raw hidden-space units; absolute units should not be compared across models or layers.','','| Source | Closest | Second | Third | Fourth | Fifth |','|---:|---|---|---|---|---|']
 for g in r['groups']:
  lines.append(f"| {g['leading_digit']} | "+' | '.join(f"{n['leading_digit']} ({n['euclidean_distance']:.6f})" for n in g['ranked_other_centroids'][:5])+' |')
 lines.append('')
lines += ['## Reading the results','','These are neighborhoods of nine group centroids, not the earlier k=4 graph on 10,000 individual targets. A centroid discards internal group structure; these rankings do not describe every pair of points. Small fourth–fifth distance gaps can make top-four membership sensitive to prompt variation. The JSON includes all eight ranks, full precision distances, full distance matrices, and fourth–fifth gaps.','','All results were derived from previously saved centroid-distance matrices. No model inference, PH reruns, or cloud jobs were used.']
(out/'TOP4_CENTROIDS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:31]))
for r in results:
 print(r['model'],r['layer'],'nearest',[g['top4'][0] for g in r['groups']])
