from pathlib import Path
import json,shutil,hashlib,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
repo=Path('/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL');sys.path.insert(0,str(repo))
from numzig.ph_acceleration import compare
src=repo/'results/ph_execution_20260919/acceleration'
a=src/'1789845920693323941';b=src/'1789846256274332641'
for run in [a,b]:
 assert json.loads((run/'report.json').read_text())['passed']
 for p in run.glob('layer_*/manifest.json'):
  m=json.loads(p.read_text())
  for item in m['artifacts'].values():
   for name,sha in item['files'].items():assert hashlib.sha256((p.parent/name).read_bytes()).hexdigest()==sha
with np.load(a/'layer_07/h1.npz') as x,np.load(b/'layer_07/h1.npz') as y:
 assert np.array_equal(x['raw'],y['raw'])
 agreement={'full_layer7_bars_compared':len(x['raw']),'bitwise_identical_raw_diagrams':True}
with np.load(a/'layer_01/h1.npz') as x,np.load(repo/'results/ph_execution_20260919/direct_lightweight/crystal/layer_01/h1.npz') as y:
 assert np.array_equal(x['raw'],y['raw'])
 agreement.update(crystal_layer1_cpu_gpu_bars_compared=len(x['raw']),crystal_layer1_bitwise_identical=True)
apps=json.loads((src/'final_apps.json').read_text())
assert all(r['state']=='stopped' and int(r['tasks'])==0 for r in apps if r['description'] in ('numberline-layer-ph','numberline-ph-acceleration'))
out=Path('/Users/mohammed.awad/Documents/Codex/2026-09-19/c/outputs/ph_acceleration')
shutil.copytree(src,out,dirs_exist_ok=True)
(out/'validation.json').write_text(json.dumps(agreement,indent=2))
with np.load(b/'layer_07/h1.npz') as f:h1=f['normalized'].copy()
h0path=repo/'results/ph_execution_20260919/direct_lightweight/crystal/layer_07/h0.npz'
with np.load(h0path) as f:h0=f['normalized'].copy()
shutil.copy2(h0path,out/'crystal_layer07_h0.npz')
end=max(h0[np.isfinite(h0[:,1]),1].max(),h1[:,1].max())*1.04
fig,ax=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
for d,col,name in [(h0,'#3366aa','H0: components'),(h1,'#cc6633','H1: loops')]:
 finite=d[np.isfinite(d[:,1])]
 ax[0].scatter(finite[:,0],finite[:,1],s=7,alpha=.5,c=col,label=name)
 t=np.linspace(0,end,512)
 alive=np.searchsorted(np.sort(d[:,0]),t,side='right')-np.searchsorted(np.sort(d[:,1]),t,side='right')
 ax[2].plot(t,alive,c=col,label=name)
ax[0].plot([0,end],[0,end],c='gray',lw=.5)
ax[0].set(xlabel='Birth / layer scale',ylabel='Death / layer scale',title='Persistence diagram');ax[0].legend(fontsize=8)
for rank,i in enumerate(np.argsort(-(h1[:,1]-h1[:,0]))[:30]):ax[1].plot(h1[i],[rank,rank],c='#cc6633')
ax[1].set(xlabel='Distance / layer scale',ylabel='Rank (0 = longest)',title='Longest 30 loop bars')
ax[2].set(xlabel='Distance / layer scale',ylabel='Features alive',yscale='symlog',title='Component and loop counts');ax[2].legend(fontsize=8)
fig.suptitle('Crystal layer 7 — all 10,000 targets\nExact-point Rips H1, float32 filtration; H0 from verified saved Euclidean distances',fontsize=11)
for ext in ['png','svg']:fig.savefig(out/f'crystal_layer07_persistence.{ext}',dpi=160)
plt.close(fig)
(out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Faster PH: Crystal layer 7</title><style>body{font:18px system-ui;margin:40px auto;max-width:1250px;line-height:1.5}img{width:100%}table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}</style><h1>The slow layer now finishes</h1><p>Crystal layer 7, all 10,000 numbers: H1 finished in <b>129 seconds</b> using one L40S GPU, four CPUs and 12 GiB RAM. The CPU attempt had timed out after 15 minutes.</p><p>Its 24,712 loop bars match the earlier eight-CPU GPU result exactly. On layer 1, all 12,634 bars match the completed CPU result exactly. Layer 7 has no completed full CPU reference.</p><img src="crystal_layer07_persistence.png"><p>H0 shows components merging; H1 shows loops appearing and disappearing as distance grows. These results do not by themselves establish meaningful numerical loops or shared model topology.</p><p><b>All test apps stopped. A ten-GPU bulk run has not been launched.</b></p><p><a href="../PH_ACCELERATION_RESULTS.md">Performance and validation report</a> · <a href="validation.json">Exact result comparisons</a></p>''')
print(agreement)
