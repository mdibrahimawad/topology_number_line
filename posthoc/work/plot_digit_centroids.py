from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('outputs/digit_order_check')
records=json.loads((OUT/'metrics.json').read_text())
base=Path('/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL/results')
paths={
'Crystal':base/'crystal_fullrange_modal_20260919T123307Z/full_complete/crystal_fullrange_1_10000_ctx1234_k4_seed42',
'StarCoder':base/'native_models_modal_20260919/full_complete/starcoderbase-3b_fullrange_1_10000_ctx1234_k4_seed42',
'OpenLLaMA':base/'native_models_modal_20260919/full_complete/openllama-3b_fullrange_1_10000_ctx1234_k4_seed42'}
offsets={
'Crystal':[(-15,12),(8,12),(-5,-24),(7,12),(10,-23),(10,-10),(15,8),(7,10),(7,10)],
'StarCoder':[(-22,-8),(-5,-27),(-10,18),(15,12),(7,-22),(10,-2),(-5,15),(15,-5),(8,12)],
'OpenLLaMA':[(-15,-20),(-20,0),(-25,12),(-10,15),(13,14),(15,-20),(20,0),(10,10),(10,10)]}
colors=plt.get_cmap('tab10')(np.array([0,1,2,3,4,5,6,7,9]))
fig,axs=plt.subplots(1,3,figsize=(15,5.4))
for ax,(name,path) in zip(axs,paths.items()):
 row=max((r for r in records if r['model']==name and r['space']=='PCA2' and r['subset']=='all'),key=lambda r:r['layer'])
 n=row['layer'];ev=row['variance_retained'];c=np.array(row['centroids'])
 ds=json.loads((path/'dataset.json').read_text())['records'];leads=np.array([r['leading_digit'] for r in sorted(ds,key=lambda r:r['point_id'])])
 with np.load(path/'analysis'/f'layer_{n:02}.npz') as z:s=z['scores']
 scale=np.sqrt((s*s).sum(1).mean());s=s/scale;c=c/scale
 for d in range(1,10):
  ax.scatter(*s[leads==d].T,s=3,color=colors[d-1],alpha=.055,rasterized=True)
  ax.scatter(*c[d-1],s=95,color=colors[d-1],edgecolor='white',linewidth=1.2,zorder=5)
  ax.annotate(str(d),c[d-1],xytext=offsets[name][d-1],textcoords='offset points',fontsize=12,fontweight='bold',ha='center',va='center',zorder=6,bbox=dict(boxstyle='round,pad=.1',facecolor='white',edgecolor='none',alpha=.85),arrowprops=dict(arrowstyle='-',color='#777777',lw=.65))
 ax.set_aspect('equal',adjustable='datalim');ax.set_title(f'{name} · final layer {n}\nTwo PCs retain {100*ev:.1f}% of variance',fontsize=12)
 ax.set_xlabel('PC1 / RMS projected radius');ax.set_ylabel('PC2 / RMS projected radius');ax.grid(alpha=.15)
fig.suptitle('Where are the leading-digit group centers?',fontsize=17)
fig.text(.5,.025,'Large dots = group means; faint points = all 10,000 targets. Labels are offset for readability; equal axis scales.',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.06,1,.91));fig.savefig(OUT/'final_centroids.png',dpi=180);fig.savefig(OUT/'final_centroids.svg');plt.close(fig)
def check(obj):
 if isinstance(obj,dict):
  for v in obj.values():check(v)
 elif isinstance(obj,list):
  for v in obj:check(v)
 elif isinstance(obj,float):assert np.isfinite(obj)
check(records)
print('All saved metrics finite. Relabeled figure saved.')
