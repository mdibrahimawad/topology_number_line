import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pca3d-analysis-mpl')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
ROOT=Path(__file__).resolve().parents[2];SRC=ROOT/'outputs/pca3d';OUT=ROOT/'outputs/pca3d_analysis';(OUT/'atlas').mkdir(exist_ok=True)
index=json.loads((SRC/'index.json').read_text());colors=index['colors'];cmap=ListedColormap(colors)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11})

def draw(ax,y,c,title,elev=23,azim=-55,paths=False):
 ax.set_proj_type('ortho');ax.view_init(elev=elev,azim=azim)
 ax.scatter(*y.T,c=c,s=.55,alpha=.55,depthshade=False,rasterized=True)
 ranges=np.ptp(y,axis=0)
 if np.max(ranges)>0:ax.set_box_aspect(np.maximum(ranges,.005*np.max(ranges)))
 ax.set_title(title,pad=0);ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([])
 ax.set_xlabel('PC1',labelpad=-12);ax.set_ylabel('PC2',labelpad=-12);ax.set_zlabel('PC3',labelpad=-12)
 if paths:
  q=y[999:9999].reshape(90,100,3).mean(1);ax.plot(*q.T,color='black',lw=1,alpha=.9)

for model in index['models']:
 for page,start in enumerate(range(0,len(model['layers']),12),1):
  dest=OUT/'atlas'/f"{model['id']}_{page}.png"
  if dest.exists():continue
  fig=plt.figure(figsize=(15,17),facecolor='white')
  levels=model['layers'][start:start+12]
  for i,l in enumerate(levels):
   with np.load(SRC/l['download']) as z:y=z['scores'];labels=z['leading_digits']
   ax=fig.add_subplot(4,3,i+1,projection='3d')
   title=f"Layer {l['layer']} · 3D variance {100*sum(l['variance']):.1f}%" if l['status']=='valid' else 'Layer 0 · coincident vectors'
   draw(ax,y,np.array(colors)[labels-1],title)
  fig.suptitle(f"{model['name']} · all 10,000 points · leading-digit colors\nEqual spatial units; each panel has its own scale and PCA frame",fontsize=16,y=.987)
  fig.subplots_adjust(left=.025,right=.98,top=.95,bottom=.025,wspace=.0,hspace=.06)
  fig.savefig(dest,dpi=120);plt.close(fig);print(dest.name,flush=True)
# Two additional views of the early/best-rho/late/final examples, plus digit-length coloring.
for mid,layer in [('crystal',7),('crystal',16),('crystal',31),('crystal',32),('starcoder',21),('starcoder',35),('starcoder',36),('openllama',8),('openllama',25),('openllama',26)]:
 with np.load(SRC/'data'/f'{mid}_{layer:02}.npz') as z:y=z['scores'];labels=z['leading_digits'];length=z['digit_counts']
 fig=plt.figure(figsize=(15,5),facecolor='white')
 draw(fig.add_subplot(131,projection='3d'),y,np.array(colors)[labels-1],'Leading digit · view A',paths=True)
 draw(fig.add_subplot(132,projection='3d'),y,np.array(colors)[labels-1],'Leading digit · view B',elev=25,azim=40,paths=True)
 draw(fig.add_subplot(133,projection='3d'),y,plt.get_cmap('viridis')((length-1)/4),'Digit length · view B',elev=25,azim=40)
 fig.suptitle(f'{mid.capitalize()} · layer {layer} | black path = means of consecutive 100-number bins, targets 1000–9999',fontsize=13)
 fig.subplots_adjust(left=.01,right=.99,top=.89,bottom=.03,wspace=0)
 fig.savefig(OUT/f'{mid}_{layer:02}_views.png',dpi=130);plt.close(fig)
print('PLOTS COMPLETE',flush=True)
