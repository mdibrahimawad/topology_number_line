"""Scientific and recovery checks using synthetic vectors, never model inference."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.distance import cdist

pytest.importorskip('gph', reason='Run PH tests in the isolated .venv-ph environment')
pytest.importorskip('ripser', reason='Run PH tests in the isolated .venv-ph environment')

from numzig import ph
from numzig.fullrange.storage import Store,atomic,digest,write_json


def circle(n=32):
    a=np.linspace(0,2*np.pi,n,endpoint=False)
    return np.column_stack([np.cos(a),np.sin(a)]).astype(np.float32)


def source(tmp_path,x,cache=True):
    root=tmp_path/'source'
    s=Store(root,dict(experiment='synthetic_validation_only'))
    write_json(root/'dataset.json',dict(records=[dict(point_id=i,target=i+1,prompt=f'{i+1}=',leading_digit=int(str(i+1)[0])) for i in range(len(x))]))
    write_json(root/'tokenizer_provenance.json',dict(expected_levels=2,expected_hidden_dim=x.shape[1],hidden_state_semantics='synthetic'))
    s.finish('dataset',[root/'dataset.json',root/'tokenizer_provenance.json'],'synthetic')
    if cache:
        p=root/'analysis_cache/layer_01.npy'
        atomic(p,lambda f:np.save(f,x));s.finish('cache/01',[p],'synthetic')
    else:
        for i,start in enumerate(range(0,len(x),7)):
            p=root/f'hidden/{i:05d}.npy';q=root/f'hidden/{i:05d}_ids.npy'
            a=np.stack([np.zeros_like(x[start:start+7]),x[start:start+7]])
            atomic(p,lambda f:np.save(f,a));atomic(q,lambda f:np.save(f,np.arange(start,min(start+7,len(x)),dtype=np.int32)))
            s.finish(f'chunk/{i:05d}',[p,q],'synthetic')
    return root


@pytest.mark.parametrize('backend',['gph','ripser'])
def test_known_shapes(backend):
    line=np.arange(20).reshape(-1,1)
    assert len(ph.h1(cdist(line,line),backend))==0
    x=circle();a=ph.h1(cdist(x,x),backend)
    assert len(a)==1 and (a[0,1]-a[0,0])>1.4
    d=cdist(x,x)
    cut=ph.h1(d,backend,threshold=.5)
    assert len(cut)==1 and np.isinf(cut[0,1])


def test_backend_equivalence_and_scaling():
    from persim import bottleneck
    rng=np.random.default_rng(3)
    x=circle()+rng.normal(0,.01,(32,2))
    d=cdist(x,x)
    a=ph.h1(d,'gph',threads=1);b=ph.h1(d,'ripser')
    assert bottleneck(a,b)<1e-6
    assert bottleneck(a,ph.h1(d,'gph',threads=2))<1e-6
    norm=ph.normalization(d);norm2=ph.normalization(7*d)
    assert norm2['scale']==pytest.approx(7*norm['scale'])
    assert bottleneck(a/norm['scale'],ph.h1(7*d)/norm2['scale'])<1e-6


def test_mst_duplicate_and_tiny_distances():
    x=np.array([[0.],[0.],[1e-12],[1.]])
    a,edges,w=ph.mst_h0(cdist(x,x))
    np.testing.assert_allclose(a[:-1,1],[0.,1e-12,1.-1e-12],rtol=1e-13,atol=0)
    assert edges.shape==(3,2) and np.isinf(a[-1,1])


def test_mst_matches_backend_h0():
    from ripser import ripser
    x=np.random.default_rng(2).normal(size=(35,4));d=cdist(x,x)
    a,_,_=ph.mst_h0(d)
    b=ripser(d,distance_matrix=True,maxdim=0)['dgms'][0]
    np.testing.assert_allclose(a[:-1,1],np.sort(b[np.isfinite(b[:,1]),1]),rtol=1e-6)


@pytest.mark.parametrize('cache',[True,False])
def test_saved_source_and_resume(tmp_path,cache):
    x=circle();root=source(tmp_path,x,cache)
    before={str(p):digest(p) for p in root.rglob('*') if p.is_file()}
    out=tmp_path/'out/crystal/layer_01'
    report=ph.run_layer(root,out,1,expected_points=len(x))
    assert report['point_count']==32 and report['h1_finite_count']==1
    snapshots={str(p):p.stat().st_mtime_ns for p in out.rglob('*') if p.is_file()}
    assert ph.run_layer(root,out,1,threads=1,expected_points=len(x))==report
    assert snapshots=={str(p):p.stat().st_mtime_ns for p in out.rglob('*') if p.is_file()}
    assert before=={str(p):digest(p) for p in root.rglob('*') if p.is_file()}
    assert ph.aggregate(tmp_path/'out')['completed']==1
    assert (tmp_path/'out/index.html').exists()


def test_interrupted_distance_resume(tmp_path):
    x=circle(20);s=Store(tmp_path/'out',{'synthetic':True})
    calls=[]
    def interrupt():
        calls.append(1)
        if len(calls)==3:raise RuntimeError('simulated interruption')
    s.commit=interrupt
    with pytest.raises(RuntimeError,match='simulated'):ph.dense_distances(s,x,block=7)
    p=s.root/'distances/00000.npy';before=(digest(p),p.stat().st_mtime_ns)
    d=ph.dense_distances(Store(s.root),x,block=7)
    np.testing.assert_array_equal(d,cdist(x,x))
    assert before==(digest(p),p.stat().st_mtime_ns)


def test_corrupt_distance_recomputed(tmp_path):
    x=circle();s=Store(tmp_path/'out',{'synthetic':True})
    ph.dense_distances(s,x,block=10)
    bad=s.root/'distances/00000.npy';bad.write_bytes(b'bad')
    good=s.root/'distances/00010.npy';stamp=good.stat().st_mtime_ns
    np.testing.assert_array_equal(ph.dense_distances(s,x,block=10),cdist(x,x))
    assert stamp==good.stat().st_mtime_ns


def test_degenerate_and_zero_median(tmp_path):
    d=np.zeros((20,20));assert ph.normalization(d)['degenerate']
    x=np.zeros((20,2),dtype=np.float32);root=source(tmp_path,x)
    r=ph.run_layer(root,tmp_path/'out',1,expected_points=20)
    assert r['status']=='degenerate' and r['h1_finite_count']==0
    x[-1]=1;norm=ph.normalization(cdist(x,x))
    assert not norm['degenerate'] and norm['sampled_median']==0 and 'fallback' in norm['method']


def test_censored_bars(tmp_path):
    x=circle();root=source(tmp_path,x)
    r=ph.run_layer(root,tmp_path/'out',1,threshold=.4,expected_points=32)
    assert r['status']=='truncated' and r['h1_right_censored_count']==1


def test_source_corruption_and_output_safety(tmp_path):
    root=source(tmp_path,circle())
    with pytest.raises(ValueError,match='separate'):ph.run_layer(root,root/'ph',1,expected_points=32)
    with pytest.raises(ValueError,match='canonical'):ph.source_info(root,1,expected_points=10000)
    (root/'analysis_cache/layer_01.npy').write_bytes(b'broken')
    with pytest.raises(ValueError,match='corrupt'):ph.run_layer(root,tmp_path/'out',1,expected_points=32)


def test_budget_fail_closed_and_no_refund():
    ledger=[]
    for _ in range(10):ph.reserve(ledger,'pilot','job',4,16,600)
    count=len(ledger)
    with pytest.raises(RuntimeError):ph.reserve(ledger,'pilot','expensive',100,100,3600)
    assert len(ledger)==count and all(r['status']=='reserved' for r in ledger)
    assert sum(r['reserved_usd'] for r in ledger)<3
    for value in (0,-1,float('inf'),float('nan')):
        with pytest.raises(ValueError):ph.reserve(ledger,'pilot','invalid',4,16,value)


def test_diagram_comparison_error_bound():
    from persim import bottleneck
    a=np.array([[0,2],[.3,.5],[.6,.65]])
    b=np.array([[.1,2.1],[.1,.25],[.7,.75]])
    aa,ea=ph.diagram_sketch(a,1);bb,eb=ph.diagram_sketch(b,1)
    approx=bottleneck(aa,bb);actual=bottleneck(a,b)
    assert max(0,approx-ea-eb)-1e-12 <= actual <= approx+ea+eb+1e-12
    with pytest.raises(ValueError):ph.diagram_sketch([[0,np.inf]],3)


def test_h0_only_then_h1_and_plot_recovery(tmp_path):
    root=source(tmp_path,circle());out=tmp_path/'out'
    ph.run_layer(root,out,1,stage='h0',expected_points=32)
    stamp=(out/'h0.npz').stat().st_mtime_ns
    assert not (out/'h1.npz').exists()
    ph.run_layer(root,out,1,expected_points=32)
    assert stamp==(out/'h0.npz').stat().st_mtime_ns
    (out/'persistence.png').unlink()
    h1stamp=(out/'h1.npz').stat().st_mtime_ns
    ph.run_layer(root,out,1,expected_points=32)
    assert (out/'persistence.png').exists() and h1stamp==(out/'h1.npz').stat().st_mtime_ns


def test_audited_preprocessing_reuse_is_read_only(tmp_path):
    from numzig.fullrange.storage import read_json,fingerprint
    root=source(tmp_path,circle());old=tmp_path/'old'
    ph.run_layer(root,old,1,stage='h0',expected_points=32)
    # Synthetic fixture representing the archived v1 producer, not a production migration.
    m=read_json(old/'manifest.json');m['config']['code_sha256']=ph.PREPROCESSING_V1
    m['config_fingerprint']=fingerprint(m['config']);write_json(old/'manifest.json',m)
    hashes={str(p):digest(p) for p in old.rglob('*') if p.is_file()}
    result=ph.run_layer(root,tmp_path/'new',1,expected_points=32,reuse_preprocessing=old)
    assert result['h1_finite_count']==1 and result['edge_collapse'] is False
    assert hashes=={str(p):digest(p) for p in old.rglob('*') if p.is_file()}
    assert not (tmp_path/'new/distances').exists()
    with np.load(old/'h0.npz') as a,np.load(tmp_path/'new/h0.npz') as b:
        np.testing.assert_array_equal(a['raw'],b['raw'])
    (old/'distances/00000.npy').write_bytes(b'broken')
    with pytest.raises(ValueError,match='corrupt'):
        ph.run_layer(root,tmp_path/'bad',1,expected_points=32,reuse_preprocessing=old)


def test_cli_and_modal_import():
    repo=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,'-m','numzig.ph','--help'],cwd=repo,capture_output=True,text=True,check=True)
    assert 'inventory' in result.stdout
    subprocess.run([sys.executable,'-c','import modal_ph; print(modal_ph.APP_NAME)'],cwd=repo,check=True,capture_output=True)
