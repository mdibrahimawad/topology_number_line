"""Separate 3D PCA artifacts; only reads the completed original experiments."""
import os
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse,hashlib,json,time,shutil
import numpy as np
from scipy.sparse.linalg import LinearOperator,eigsh
from scipy.stats import spearmanr
ROOT=Path('/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL/results')
OUT=Path(__file__).resolve().parents[2]/'outputs/pca3d'
MODELS={
 'crystal':dict(name='Crystal',last=32,rho=7,normalization='LayerNorm',path=ROOT/'crystal_fullrange_modal_20260919T123307Z/full_complete/crystal_fullrange_1_10000_ctx1234_k4_seed42'),
 'starcoder':dict(name='StarCoderBase-3B',last=36,rho=21,normalization='LayerNorm',path=ROOT/'native_models_modal_20260919/full_complete/starcoderbase-3b_fullrange_1_10000_ctx1234_k4_seed42'),
 'openllama':dict(name='OpenLLaMA-3B',last=26,rho=8,normalization='RMSNorm',path=ROOT/'native_models_modal_20260919/full_complete/openllama-3b_fullrange_1_10000_ctx1234_k4_seed42')}
COLORS=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#17becf']

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save_json(path,data):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,separators=(',',':'),allow_nan=False));os.replace(tmp,path)
def save_npz(path,**data):
 tmp=path.with_suffix('.npz.tmp')
 with tmp.open('wb') as f:np.savez_compressed(f,**data)
 os.replace(tmp,path)

def pc3(x,basis):
 """Leading eigenvector after removing saved exact PC1/PC2; all points, no sampling."""
 calls=0
 def mv(v):
  nonlocal calls
  calls+=1
  v=v-basis.T@(basis@v)
  y=x.T@(x@v)
  return y-basis.T@(basis@y)
 # Accelerate's NumPy floating-point status flags can be stale; actual arrays and
 # the unprojected eigen-residual are independently checked below.
 with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
  values,vectors=eigsh(LinearOperator((x.shape[1],x.shape[1]),matvec=mv,dtype=np.float64),k=1,which='LA',tol=1e-10,maxiter=3000,v0=np.random.default_rng(42).normal(size=x.shape[1]))
  v=vectors[:,0];v-=basis.T@(basis@v);v/=np.linalg.norm(v)
  score=x@v;lam=float(score@score)
  residual=float(np.linalg.norm(x.T@score-lam*v)/max(lam,1e-300))
 assert np.isfinite(v).all() and np.isfinite(score).all() and residual<1e-7,(values,residual)
 return v,score,dict(relative_eigen_residual=residual,matvec_calls=calls)

def write_layer(model,layer,x,rows):
 tic=time.monotonic();cfg=MODELS[model];path=cfg['path'];base=OUT/'data'/f'{model}_{layer:02}'
 analysis=path/'analysis'/f'layer_{layer:02}.npz';meta=json.loads(analysis.with_suffix('.json').read_text())
 with np.load(analysis) as z:old={k:z[k] for k in z.files}
 assert np.array_equal(old['point_ids'],np.arange(10000)) and np.isfinite(x).all()
 targets=np.arange(1,10001);labels=np.array([r['leading_digit'] for r in rows]);lens=np.array([r['digit_count'] for r in rows])
 status=meta['status'];total=0.;extra={};nn=old.get('neighbors',np.empty((10000,0),int));dist=old.get('distances',np.empty((10000,0)))
 if status=='valid':
  mean=old['mean'];assert np.allclose(x.mean(0),mean,atol=1e-10,rtol=1e-10)
  x-=mean;total=float(np.einsum('ij,ij->',x,x));basis=old['components']
  with np.errstate(over='ignore',divide='ignore',invalid='ignore'):error=float(np.max(np.abs(x@basis.T-old['scores'])))
  assert error<1e-6
  v,s3,extra=pc3(x,basis);rho=float(spearmanr(targets,s3).statistic)
  if np.isfinite(rho) and rho<0:v=-v;s3=-s3;rho=-rho
  components=np.vstack((basis,v));scores=np.column_stack((old['scores'],s3))
  eigen3=float(s3@s3/(len(x)-1));variance=np.r_[old['explained_variance'],eigen3];ratio=np.r_[old['explained_variance_ratio'],float(s3@s3/total)]
  with np.errstate(over='ignore',divide='ignore',invalid='ignore'):ortho=float(np.max(abs(components@components.T-np.eye(3))))
  assert ortho<1e-8 and 0<=eigen3<=variance[1]*(1+1e-7) and ratio.sum()<=1+1e-8
  assert np.array_equal(scores[:,:2],old['scores']) and np.isfinite(scores).all()
  extra.update(pc12_projection_max_error=error,pc12_scores_preserved_exactly=True,component_orthogonality_error=ortho,pc3_spearman=rho if np.isfinite(rho) else None)
 else:
  mean=x.mean(0);x-=mean;total=float(np.einsum('ij,ij->',x,x));assert total<1e-12
  scores=np.zeros((10000,3));components=np.zeros((3,x.shape[1]));variance=np.zeros(3);ratio=np.zeros(3)
  extra['interpretation']='All 10,000 hidden vectors coincide; PCA axes and variance percentages are undefined.'
 centers=np.stack([scores[labels==d].mean(0) for d in range(1,10)])
 save_npz(base.with_suffix('.npz'),point_ids=targets-1,targets=targets,scores=scores,mean=mean,components=components,explained_variance=variance,explained_variance_ratio=ratio,leading_digits=labels,digit_counts=lens,centroids=centers,neighbors=nn,distances=dist)
 with np.load(base.with_suffix('.npz')) as z:assert np.array_equal(z['scores'],scores) and np.array_equal(z['components'],components)
 payload=dict(model=model,layer=layer,status=status,scores=np.round(scores,7).tolist(),variance=ratio.tolist(),centroids=np.round(centers,7).tolist(),neighbors=nn.tolist(),distances=np.round(dist,7).tolist())
 save_json(base.with_suffix('.json'),payload)
 result=dict(model=model,layer=layer,status=status,point_count=10000,hidden_dimension=x.shape[1],variance=ratio.tolist(),pc3_additional_variance=float(ratio[2]),total_3d_variance=float(ratio.sum()),source_analysis_sha256=digest(analysis),source_dataset_sha256=digest(path/'dataset.json'),scores_npz_sha256=digest(base.with_suffix('.npz')),viewer_json_sha256=digest(base.with_suffix('.json')),seconds=time.monotonic()-tic,**extra)
 save_json(base.with_suffix('.validation.json'),result)
 return result

def batch(job):
 model,layers=job;cfg=MODELS[model];p=cfg['path'];rows=sorted(json.loads((p/'dataset.json').read_text())['records'],key=lambda r:r['point_id'])
 assert [r['target'] for r in rows]==list(range(1,10001));first=np.load(p/'hidden/00000.npy',mmap_mode='r');dim=first.shape[2]
 tic=time.monotonic();x=np.empty((len(layers),10000,dim),dtype=np.float64);seen=np.zeros(10000,int)
 for f in sorted((p/'hidden').glob('*_ids.npy')):
  ids=np.load(f);a=np.load(f.with_name(f.name.replace('_ids','')),mmap_mode='r')
  assert a.shape[0]==cfg['last']+1 and a.shape[1]==len(ids) and a.shape[2]==dim
  x[:,ids,:]=a[layers];seen[ids]+=1
 assert np.all(seen==1)
 loaded=time.monotonic()-tic;results=[]
 for i,layer in enumerate(layers):
  result=write_layer(model,layer,x[i],rows);result['batch_read_seconds']=loaded;results.append(result)
 return results

def index():
 groups=[];validations=[]
 for model,cfg in MODELS.items():
  levels=[]
  for layer in range(cfg['last']+1):
   base=OUT/'data'/f'{model}_{layer:02}';p=base.with_suffix('.validation.json')
   if p.exists():
    v=json.loads(p.read_text());validations.append(v);levels.append(dict(layer=layer,status=v['status'],variance=v['variance'],file=f'data/{base.name}.json',download=f'data/{base.name}.npz'))
  groups.append(dict(id=model,name=cfg['name'],last=cfg['last'],best_rho=cfg['rho'],best_variance=1,final_normalization=cfg['normalization'],layers=levels))
 save_json(OUT/'index.json',dict(title='3D numerical representations',models=groups,colors=COLORS,point_count=10000,total_expected=97,completed=len(validations),contextual_layers=94,method='Existing full-SVD PC1/PC2 unchanged; PC3 computed from all original hidden vectors by a converged deflated symmetric eigensolver. Centered, unscaled, unwhitened.'))
 save_json(OUT/'validation.json',dict(completed=len(validations),expected=97,all_points_per_level=10000,solver='ARPACK eigsh, largest algebraic eigenpair in orthogonal complement of original full-SVD PC1/PC2; float64; tolerance 1e-10',layers=validations))
 return len(validations)

def selfcheck():
 rng=np.random.default_rng(5);x=rng.normal(size=(120,12))*np.linspace(8,1,12);x-=x.mean(0)
 _,s,vh=np.linalg.svd(x,full_matrices=False);v,score,_=pc3(x,vh[:2]);assert np.allclose(abs(v@vh[2]),1,atol=1e-8) and np.isclose(score@score,s[2]**2,rtol=1e-8)
 return dict(synthetic_full_svd_equivalence=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--batch-size',type=int,default=4);args=ap.parse_args()
 (OUT/'data').mkdir(parents=True,exist_ok=True);save_json(OUT/'solver_check.json',selfcheck())
 shutil.copyfile(MODELS['crystal']['path']/'viewer/plotly.min.js',OUT/'plotly.min.js')
 jobs=[]
 for model,cfg in MODELS.items():
  pending=[]
  for l in range(cfg['last']+1):
   base=OUT/'data'/f'{model}_{l:02}';v=base.with_suffix('.validation.json');ok=False
   if v.exists():
    r=json.loads(v.read_text());ok=(base.with_suffix('.npz').exists() and base.with_suffix('.json').exists() and digest(base.with_suffix('.npz'))==r['scores_npz_sha256'] and digest(base.with_suffix('.json'))==r['viewer_json_sha256'] and digest(cfg['path']/'analysis'/f'layer_{l:02}.npz')==r['source_analysis_sha256'])
   if not ok:pending.append(l)
  jobs.extend((model,pending[i:i+args.batch_size]) for i in range(0,len(pending),args.batch_size))
 print('Starting',len(jobs),'batches;',index(),'levels already saved.',flush=True)
 with ProcessPoolExecutor(max_workers=args.workers) as pool:
  futures={pool.submit(batch,job):job for job in jobs}
  for future in as_completed(futures):
   rows=future.result();n=index();print(json.dumps(dict(completed=n,batch=futures[future],read_seconds=round(rows[0]['batch_read_seconds'],2),solve_seconds=round(sum(r['seconds'] for r in rows),2))),flush=True)
 assert index()==97
 print('COMPLETE: 94 contextual layers and 3 embedding levels.',flush=True)
if __name__=='__main__':main()
