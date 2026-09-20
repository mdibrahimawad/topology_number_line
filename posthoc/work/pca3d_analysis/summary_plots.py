import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pca3d-analysis-mpl')
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'outputs/pca3d_analysis';S=ROOT/'outputs/pca3d/data'
R=[r for r in json.loads((O/'metrics.json').read_text()) if r['layer']];D=json.loads((O/'digit_contributions.json').read_text())
models=['crystal','starcoder','openllama'];names=['Crystal','StarCoderBase-3B','OpenLLaMA-3B'];cols=['#2962a3','#dd7c2d','#289275','#93488e']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(15,8),sharex='col',sharey='row')
for j,m in enumerate(models):
 rows=[r for r in R if r['model']==m];x=[r['layer'] for r in rows]
 for k,label,c in [('leading_digit','First digit only',cols[0]),('generalized_decimal','Linear + repeating decimal waves',cols[1]),('decimal_digits','Four separate digit categories',cols[2])]:
  axes[0,j].plot(x,[100*r['fits'][k]['block100'] for r in rows],label=label,c=c,lw=2)
 axes[0,j].set_title(names[j]);axes[0,j].set_ylim(-5,103);axes[0,j].grid(alpha=.15)
 axes[1,j].plot(x,[100*r['variance3'] for r in rows],label='Original variance retained in 3D',c=cols[0],lw=2)
 axes[1,j].plot(x,[100*r['neighbor4_retention3d'] for r in rows],label='Original nearest neighbors retained',c=cols[1],lw=2)
 axes[1,j].set_ylim(0,100);axes[1,j].set_xlabel('Saved layer');axes[1,j].grid(alpha=.15)
axes[0,0].set_ylabel('Held-out 3D variance explained (%)');axes[1,0].set_ylabel('Percent')
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False)
axes[1,1].legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,frameon=False)
fig.suptitle('Digit structure and projection quality across all layers',fontsize=16,y=.995)
fig.subplots_adjust(top=.83,bottom=.13,hspace=.28,wspace=.13);fig.savefig(O/'fits_and_projection.png',dpi=150);plt.close(fig)
# Closest Crystal/StarCoder comparison with rotation/scale/reflection removed; data-derived overlay.
a=np.load(S/'crystal_25.npz')['scores'][999:9999].reshape(90,100,3).mean(1);b=np.load(S/'starcoder_20.npz')['scores'][999:9999].reshape(90,100,3).mean(1)
a-=a.mean(0);b-=b.mean(0);a/=np.linalg.norm(a);b/=np.linalg.norm(b)
with np.errstate(over='ignore',divide='ignore',invalid='ignore'):u,sv,vt=np.linalg.svd(b.T@a);b=b@(u@vt)
fig=plt.figure(figsize=(14,5))
for j,(x,y,lab) in enumerate([(0,1,'Aligned axes 1–2'),(0,2,'Aligned axes 1–3'),(1,2,'Aligned axes 2–3')]):
 ax=fig.add_subplot(1,3,j+1);ax.plot(a[:,x],a[:,y],'-o',ms=2,lw=1.4,label='Crystal L25');ax.plot(b[:,x],b[:,y],'-o',ms=2,lw=1.4,label='StarCoder L20');ax.set_title(lab);ax.set_aspect('equal');ax.grid(alpha=.2)
fig.legend(*fig.axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False);fig.suptitle('Similar ordered shapes at different depths\n90 means of 100-number bins; similarity 0.978 after alignment (1 = identical)',fontsize=15);fig.subplots_adjust(top=.77,bottom=.17,wspace=.3);fig.savefig(O/'aligned_shapes.png',dpi=150);plt.close(fig)
# Pure period-10 ellipse vs actual last-digit centroids.
fig=plt.figure(figsize=(14,5));t=np.linspace(0,10,200)
for j,(m,l) in enumerate([('crystal',3),('starcoder',7),('openllama',2)]):
 r=next(r for r in D if r['model']==m and r['layer']==l)['digit_contributions']['units'];p=np.array(r['means']);digit=np.arange(10);f=np.column_stack([np.ones(10),np.cos(2*np.pi*digit/10),np.sin(2*np.pi*digit/10)])
 with np.errstate(over='ignore',divide='ignore',invalid='ignore'):coef=np.linalg.lstsq(f,p,rcond=None)[0];curve=np.column_stack([np.ones(200),np.cos(2*np.pi*t/10),np.sin(2*np.pi*t/10)])@coef
 ax=fig.add_subplot(1,3,j+1,projection='3d');ax.set_proj_type('ortho');ax.view_init(22,-55);ax.scatter(*p.T,s=35,c=cols[0]);ax.plot(*curve.T,c=cols[1],lw=1.7)
 for k in range(10):ax.text(*p[k],str(k),fontsize=10)
 ax.set_box_aspect(np.maximum(np.ptp(p,axis=0),.001));ax.set_xticks([]);ax.set_yticks([]);ax.set_zticks([]);ax.set_title(f'{names[j]} L{l}\nEllipse explains {r["circular_first_harmonic_centroid_R2"]:.0%} of centroid spread',fontsize=11)
fig.suptitle('The last digit repeats every ten numbers — its geometry is not a perfect circle\nBlue = actual digit means; orange = best period-10 ellipse in the saved 3D coordinates',fontsize=14)
fig.subplots_adjust(top=.74,bottom=.02,wspace=.0);fig.savefig(O/'periodic_digit_shapes.png',dpi=150);plt.close(fig)
H=json.loads((O/'hidden_digit_contributions.json').read_text());assert len(H)==94
fig,axes=plt.subplots(2,3,figsize=(15,8),sharex='col',sharey=True)
for j,m in enumerate(models):
 for row,source in [(0,H),(1,D)]:
  rr=[r for r in source if r['model']==m]
  for k,name in enumerate(['thousands','hundreds','tens','units']):
   values=[r['digit_variance_fractions'][name] if row==0 else r['digit_contributions'][name]['variance_fraction'] for r in rr]
   axes[row,j].plot([r['layer'] for r in rr],100*np.array(values),label=['First digit (thousands)','Hundreds','Tens','Last digit (units)'][k],c=cols[k],lw=2)
  axes[row,j].set_ylim(0,100);axes[row,j].grid(alpha=.15)
  if row==0:axes[row,j].set_title(names[j])
  else:axes[row,j].set_xlabel('Saved layer')
axes[0,0].set_ylabel('Original hidden-space variance (%)');axes[1,0].set_ylabel('Variance within 3D view (%)')
fig.suptitle('The dominant digit changes with depth\nMeasured on the same 9,000 four-digit targets, 1000–9999',fontsize=16,y=.99)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False)
fig.subplots_adjust(top=.84,bottom=.12,hspace=.2,wspace=.13);fig.savefig(O/'digit_shift.png',dpi=150);plt.close(fig)
print('Summary charts complete')
