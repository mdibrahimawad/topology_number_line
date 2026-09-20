"""Exact zero-distance quotient for StarCoder L0; never alters historical results."""
from pathlib import Path
import json, hashlib, time, importlib.metadata
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from ripser import ripser
from gph import ripser_parallel
ROOT=Path('/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL/results/native_models_modal_20260919/full_complete/starcoderbase-3b_fullrange_1_10000_ctx1234_k4_seed42')
ORIG=Path('outputs/ph_fullrange_overnight/results/results/starcoderbase-3b/layer_00')
OUT=Path('outputs/ph_analysis/audit/star_embedding_supplement');OUT.mkdir(parents=True,exist_ok=True)
start=time.perf_counter()
rows=json.loads((ROOT/'dataset.json').read_text())['records'];unique={};counts={};point_group=np.full(10000,-1,dtype=np.int32);coverage=np.zeros(10000,dtype=np.int8);representative_ids={};violations=0
for f in sorted((ROOT/'hidden').glob('[0-9][0-9][0-9][0-9][0-9].npy')):
 a=np.load(f,mmap_mode='r');ids=np.load(f.with_name(f.stem+'_ids.npy'))
 assert a.shape==(37,len(ids),2816)
 for v,i in zip(a[0],ids):
  assert 0<=i<10000 and np.isfinite(v).all()
  l=rows[i]['token_count'];coverage[i]+=1;point_group[i]=l
  if l not in unique:unique[l]=v.copy();counts[l]=0;representative_ids[l]=int(i)
  else:violations+=not np.array_equal(v.view(np.uint32),unique[l].view(np.uint32))
  counts[l]+=1
assert np.all(coverage==1) and violations==0 and len(unique)==5
lengths=sorted(unique);v=np.stack([unique[l]for l in lengths]);assert len({r.tobytes()for r in v})==5
point_group=np.array([lengths.index(int(x))for x in point_group],dtype=np.int32)
d64=cdist(v.astype(np.float64),v.astype(np.float64),metric='euclidean');d32=d64.astype(np.float32)
assert np.array_equal(d64,d64.T) and np.all(np.diag(d64)==0)
original=np.load(ORIG/'h0.npz');norm=json.loads((ORIG/'normalization.json').read_text());scale=norm['scale']
assert np.isclose(np.max(d64),scale,atol=0,rtol=0)
positive=np.sort(original['mst_distances'][original['mst_distances']>0]);mst=np.sort(minimum_spanning_tree(d64).data)
assert np.array_equal(mst,positive) and np.count_nonzero(original['mst_distances']==0)==9995
r=ripser(d32,distance_matrix=True,maxdim=1,coeff=2)['dgms'];g=ripser_parallel(d32,metric='precomputed',maxdim=1,coeff=2,n_threads=1,collapse_edges=False)['dgms']
def sort_dgm(a):
 a=np.asarray(a,dtype=np.float64).reshape(-1,2);return a[np.lexsort((a[:,1],a[:,0]))] if len(a) else a
r=[sort_dgm(a)for a in r];g=[sort_dgm(a)for a in g]
assert all(np.array_equal(a,b)for a,b in zip(r,g))
assert np.array_equal(np.sort(r[0][np.isfinite(r[0][:,1]),1]),np.sort(positive.astype(np.float32).astype(np.float64)))
# Independent small-complex H1 rank at each distinct edge threshold (F2).
from itertools import combinations
def rank_f2(mat):
 a=np.array(mat,dtype=np.uint8,copy=True);r=0
 for col in range(a.shape[1]):
  piv=np.flatnonzero(a[r:,col])
  if not len(piv):continue
  p=r+int(piv[0]);a[[r,p]]=a[[p,r]]
  for i in range(a.shape[0]):
   if i!=r and a[i,col]:a[i]^=a[r]
  r+=1
  if r==a.shape[0]:break
 return r
rank_checks=[]
for t in np.unique(d32):
 edges=[(i,j)for i,j in combinations(range(5),2)if d32[i,j]<=t]
 triangles=[z for z in combinations(range(5),3)if all(d32[i,j]<=t for i,j in combinations(z,2))]
 b1=np.zeros((5,len(edges)),dtype=np.uint8)
 for j,(a,b)in enumerate(edges):b1[[a,b],j]=1
 b2=np.zeros((len(edges),len(triangles)),dtype=np.uint8)
 for j,z in enumerate(triangles):
  for e in combinations(z,2):b2[edges.index(e),j]=1
 betti1=len(edges)-rank_f2(b1)-rank_f2(b2)
 diagram_betti=int(np.sum((r[1][:,0]<=t)&(t<r[1][:,1])))
 assert betti1==diagram_betti
 rank_checks.append(dict(threshold=float(t),edges=len(edges),triangles=len(triangles),betti1=int(betti1)))
np.savez_compressed(OUT/'quotient_geometry.npz',vectors=v,token_lengths=np.array(lengths),multiplicities=np.array([counts[l]for l in lengths]),representative_point_ids=np.array([representative_ids[l]for l in lengths]),point_to_group=point_group,point_ids=np.arange(10000),distance_float64=d64,distance_float32=d32)
np.savez_compressed(OUT/'h1.npz',raw=r[1],normalized=r[1]/scale,right_censored=np.zeros(len(r[1]),dtype=bool))
np.savez_compressed(OUT/'h0.npz',**{k:original[k]for k in original.files})
np.savez_compressed(OUT/'quotient_diagrams.npz',h0_float32=r[0],h1_float32=r[1],gph_h0=g[0],gph_h1=g[1])
report={'model':'starcoderbase-3b','layer':0,'status':'supplementary_complete','historical_full_duplicate_run_status':'H1 timed out; left unchanged','points_original':10000,'points_distinct':5,'bitwise_equality_checked_across_all_points':True,'within_length_bitwise_mismatches':violations,'length_groups':lengths,'multiplicities':[counts[l]for l in lengths],'extraction_position_equals_token_count_minus_one':True,'source':'exact saved embedding vectors; no inference','method':'exact zero-distance quotient of full Vietoris-Rips filtration; not approximate subsampling','distance_dtype':'float64 direct-difference cdist; cast to float32 filtration','coefficient_field':2,'filtration_threshold':None,'diagram_backends':['ripser','giotto-ph'],'backend_bitwise_diagram_equivalence':True,'independent_F2_boundary_rank_checks':rank_checks,'normalization':norm,'h1_bars':r[1].tolist(),'h1_normalized':(r[1]/scale).tolist(),'h1_count':len(r[1]),'full_H0_zero_deaths':9995,'full_H0_nonzero_deaths':positive.tolist(),'full_H0_essential_count':int(np.sum(np.isinf(original['raw'][:,1]))),'H0_quotient_MST_float64_equals_archived_nonzero_weights':True,'H0_float32_equals_cast_archived_nonzero_weights':True,'packages':{x:importlib.metadata.version(x)for x in ['numpy','scipy','ripser','giotto-ph']},'seconds_including_source_reads':time.perf_counter()-start,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'original_manifest_sha256':hashlib.sha256((ORIG/'manifest.json').read_bytes()).hexdigest(),'original_h0_sha256':hashlib.sha256((ORIG/'h0.npz').read_bytes()).hexdigest(),'source_dataset_sha256':hashlib.sha256((ROOT/'dataset.json').read_bytes()).hexdigest()}
(OUT/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
