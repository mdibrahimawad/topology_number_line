from pathlib import Path
import sys,time,json
import numpy as np
from scipy.spatial.distance import cdist
repo=Path('/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL');sys.path.insert(0,str(repo))
from numzig.ph import source_info,read_vectors,local_roots,h1
from persim import bottleneck
p=local_roots(repo)['crystal'];m,_,_,identity=source_info(p,7)
x=read_vectors(p,7,m,identity['shape']);ids=np.sort(np.random.default_rng(42).choice(len(x),128,replace=False))
d=cdist(x[ids].astype(float),x[ids].astype(float));del x
results={};diagrams=[]
for backend in ['gph','ripser']:
 t=time.perf_counter();a=h1(d,backend,threads=1);results[backend]={'bars':len(a),'seconds':time.perf_counter()-t};diagrams.append(a)
 print(backend,results[backend],flush=True)
t=time.perf_counter();results['gate']={'bottleneck':float(bottleneck(*diagrams)),'seconds':time.perf_counter()-t,'array_equal':bool(np.array_equal(*diagrams))}
(repo/'results/ph_execution_20260919/crystal_l7_gate_diagnostic.json').write_text(json.dumps(results,indent=2))
print(results,flush=True)
