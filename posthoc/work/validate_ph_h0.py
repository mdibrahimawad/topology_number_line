from pathlib import Path
import sys,json,time
import numpy as np
repo=Path('/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL');sys.path.insert(0,str(repo))
from numzig.ph import source_info,read_vectors,local_roots
from numzig.fullrange.storage import read_json,write_json,digest
root=repo/'results/ph_execution_20260919/lightweight';sources=local_roots(repo);reports=[]
for model,layer in [('crystal',1),('crystal',7),('starcoderbase-3b',1),('openllama-3b',1)]:
 p=root/model/f'layer_{layer:02d}';m=read_json(p/'manifest.json')
 for name,sha in m['artifacts']['h0']['files'].items():assert digest(p/name)==sha
 source,rows,meta,identity=source_info(sources[model],layer)
 x=read_vectors(sources[model],layer,source,identity['shape'])
 with np.load(p/'h0.npz') as data:
  a=data['raw'];edges=data['mst_edges'];dist=data['mst_distances'];norm=data['normalized']
  assert a.shape==(10000,2) and edges.shape==(9999,2) and np.isinf(a[:,1]).sum()==1
  error=0.
  for start in range(0,len(edges),256):
   e=edges[start:start+256];delta=x[e[:,0]].astype(float)-x[e[:,1]].astype(float)
   calculated=np.sqrt(np.sum(delta*delta,axis=1))
   error=max(error,float(np.max(np.abs(calculated-dist[start:start+256]))))
  assert error<1e-9
  # Independent graph check: 9999 edges, connected and acyclic by union-find.
  parent=np.arange(10000)
  def find(v):
   while parent[v]!=v:parent[v]=parent[parent[v]];v=parent[v]
   return v
  for a0,b0 in edges:
   ra,rb=find(a0),find(b0);assert ra!=rb;parent[ra]=rb
  assert len({find(i) for i in range(10000)})==1
  reports.append(dict(model=model,layer=layer,point_count=10000,checked_mst_edges=9999,max_edge_distance_error=error,connected_tree=True,normalized_merge_quantiles=np.quantile(norm[:-1,1],[0,.5,.9,.99,1]).tolist()))
 del x
 print(reports[-1],flush=True)
write_json(repo/'results/ph_execution_20260919/h0_validation.json',reports)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
for r in reports:
 p=root/r['model']/f"layer_{r['layer']:02d}"
 with np.load(p/'h0.npz') as f:a=f['normalized']
 t=np.linspace(0,max(x['normalized_merge_quantiles'][-1] for x in reports),1000)
 counts=10000-np.searchsorted(a[:-1,1],t,side='right')
 label=f"{r['model']} L{r['layer']}"
 axes[0].plot(t,counts,label=label)
 axes[1].plot(np.linspace(0,100,9999),a[:-1,1],label=label)
axes[0].set(xlabel='Distance / layer median distance',ylabel='Connected components',yscale='log',title='H0: components merge as distance increases')
axes[1].set(xlabel='Percentile of finite merge distances',ylabel='Distance / layer median distance',title='H0 merge-distance distributions')
for ax in axes:ax.legend(fontsize=8)
fig.suptitle('Modal PH pilot: all 10,000 targets per layer — H0 only; H1 not yet complete')
fig.savefig(repo/'results/ph_execution_20260919/h0_pilot.png',dpi=170)
