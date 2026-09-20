"""Bounded similarity registration; exploratory projected geometry, not topology inference."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import itertools
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def initial_rotations(template, seed=43):
    # Align principal axes, but search axis permutations/signs and arbitrary poses.
    _, _, vt = np.linalg.svd(template, full_matrices=False)
    starts=[]
    for p in itertools.permutations(range(3)):
        # PCA eigenvector signs are arbitrary, including a reflection. Search all
        # eight signs for every permutation, consistent with similarity().
        for sign in itertools.product((-1,1), repeat=3):
            starts.append(vt.T[:,p] * np.array(sign))
    starts.extend(Rotation.random(8,random_state=seed).as_matrix())
    return starts


def similarity(a, b, weights):
    """Weighted least squares allowing reflection; no anisotropic deformation."""
    weights=weights/weights.sum()
    ma=weights@a; mb=weights@b
    ac=a-ma;bc=b-mb
    u,s,vt=np.linalg.svd((ac*weights[:,None]).T@bc)
    r=u@vt
    scale=float(np.clip(s.sum()/max(np.sum(weights[:,None]*ac*ac),1e-20),.25,3.0))
    offset=mb-scale*ma@r
    return scale,r,offset


def score(y, z, tree_y=None):
    d=cKDTree(z).query(y)[0]
    e=(tree_y or cKDTree(y)).query(z)[0]
    return float((np.mean(d*d)+np.mean(e*e))/2)


def register(y, template, starts=4, iterations=24):
    """Multi-start symmetric ICP with training-only pose selection."""
    tree_y=cKDTree(y)
    transforms=[]
    for r in initial_rotations(template):
        z=template@r
        transforms.append((score(y,z,tree_y),1.,r,np.zeros(3)))
    transforms.sort(key=lambda t:t[0])
    best=None
    for _,scale,r,offset in transforms[:starts]:
        previous=np.inf
        for k in range(iterations):
            z=scale*template@r+offset
            a=cKDTree(z).query(y)[1]
            b=tree_y.query(z)[1]
            source=np.concatenate([template[a],template])
            dest=np.concatenate([y,y[b]])
            w=np.r_[np.full(len(y),.5/len(y)),np.full(len(template),.5/len(template))]
            scale,r,offset=similarity(source,dest,w)
            value=score(y,scale*template@r+offset,tree_y)
            if abs(previous-value)<1e-7:break
            previous=value
        item=(value,scale,r,offset,k+1)
        if best is None or item[0]<best[0]:best=item
    return best


def evaluate(test,template,pose,train=None):
    value,scale,r,offset,iters=pose
    z=scale*template@r+offset
    d,ids=cKDTree(z).query(test)
    e=cKDTree(test).query(z)[0]
    counts=np.bincount(ids,minlength=len(z))
    out=dict(train_symmetric_rms=float(np.sqrt(value)),
        symmetric_rms=float(np.sqrt((np.mean(d*d)+np.mean(e*e))/2)),
        data_rms=float(np.sqrt(np.mean(d*d))),coverage_rms=float(np.sqrt(np.mean(e*e))),
        data_p95=float(np.quantile(d,.95)),data_p99=float(np.quantile(d,.99)),
        coverage_p95=float(np.quantile(e,.95)),data_within_015=float(np.mean(d<=.15)),
        shape_within_015=float(np.mean(e<=.15)),occupied_template_fraction=float(np.mean(counts>0)),
        scale=float(scale),scale_at_bound=bool(scale<=.25001 or scale>=2.99999),
        rotation=r.tolist(),offset=offset.tolist(),iterations=iters)
    if len(z)<=20:out['vertex_occupancy']=(counts/len(test)).tolist()
    return out


def split_normalize(y,seed=1729,cap=384):
    rng=np.random.default_rng(seed);order=rng.permutation(len(y));h=len(y)//2
    train_ids=order[:h];test_ids=order[h:]
    center=y[train_ids].mean(0)
    radius=float(np.sqrt(np.mean(np.sum((y[train_ids]-center)**2,axis=1))))
    if radius<1e-12:return None
    normalized=(y-center)/radius
    # PCA orient training sample to stabilize pose starts under changed coordinate bases.
    _,_,vt=np.linalg.svd(normalized[train_ids],full_matrices=False)
    normalized=normalized@vt.T
    return normalized[train_ids[:cap]],normalized[test_ids],dict(center=center.tolist(),radius=radius,basis=vt.tolist(),train_ids=train_ids[:cap].tolist(),test_ids=test_ids.tolist())


@np.errstate(over='ignore',divide='ignore',invalid='ignore')
def fit_all(y,templates,seed=1729,cap=384,starts=4,iterations=24):
    assert np.isfinite(y).all() and y.shape[1]==3
    split=split_normalize(y,seed,cap)
    if split is None:return None
    train,test,normalization=split
    fits=[]
    for t in templates:
        pose=register(train,t['points'],starts,iterations)
        fit=evaluate(test,t['points'],pose)
        assert all(np.isfinite(fit[k]) for k in ('data_rms','coverage_rms','symmetric_rms','scale'))
        fit.update({k:v for k,v in t.items() if k!='points'})
        fit['screen_pass']=bool(fit['data_rms']<=.15 and fit['coverage_rms']<=.15 and fit['data_p95']<=.30 and fit['coverage_p95']<=.30 and not fit['scale_at_bound'])
        fits.append(fit)
    fits.sort(key=lambda f:f['symmetric_rms'])
    # Infinite linear baselines distinguish near-line/near-plane spread from named bounded shapes.
    _,_,vt=np.linalg.svd(train-train.mean(0),full_matrices=False)
    e=test-train.mean(0)
    line=e-(e@vt[0])[:,None]*vt[0]
    baselines=dict(line_rms=float(np.sqrt(np.mean(np.sum(line*line,axis=1)))),plane_rms=float(np.sqrt(np.mean((e@vt[2])**2))))
    return dict(fits=fits,normalization=normalization,baselines=baselines,train_count=len(train),test_count=len(test))
