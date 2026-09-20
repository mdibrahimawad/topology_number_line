import numpy as np
import pytest
from scipy.spatial.distance import cdist
from numzig import ph,ph_overnight as p
from numzig.fullrange.storage import Store,atomic,digest,write_json
from test_ph import source,circle

def test_parallel_distances_interruption_and_corruption(tmp_path):
    x=np.random.default_rng(1).normal(size=(530,4)).astype(np.float32)
    s=Store(tmp_path/'distances',{'test':True});calls=[]
    def stop():
        calls.append(1)
        if len(calls)==2:raise RuntimeError('interrupted')
    s.commit=stop
    with pytest.raises(RuntimeError):p.parallel_distances(s,x)
    preserved=tmp_path/'distances/distances/00000.npy';before=(digest(preserved),preserved.stat().st_mtime_ns)
    s=Store(s.root);d=p.parallel_distances(s,x,threads=2)
    assert np.array_equal(d,cdist(x.astype(float),x.astype(float)))
    assert before==(digest(preserved),preserved.stat().st_mtime_ns)
    bad=s.root/'distances/00512.npy';bad.write_bytes(b'corrupt')
    with pytest.raises(ValueError):p.load_distances(s,len(x))
    assert np.array_equal(p.parallel_distances(s,x),d)

def test_prepare_resume_and_degenerate(tmp_path):
    x=circle(40);src=source(tmp_path,x)
    out=tmp_path/'prep';r=p.prepare(src,out,1,points=40)
    stamps={str(f):f.stat().st_mtime_ns for f in out.rglob('*') if f.is_file()}
    assert p.prepare(src,out,1,points=40)==r
    assert stamps=={str(f):f.stat().st_mtime_ns for f in out.rglob('*') if f.is_file()}
    zero=tmp_path/'zero';zero.mkdir();src=source(zero,np.ones((30,2),dtype=np.float32))
    r=p.prepare(src,zero/'out',1,points=30)
    assert r['implicit_zero_distances'] and r['normalization']['degenerate']
    assert not (zero/'out/distances').exists()
    with np.load(zero/'out/h0.npz') as a:
        assert a['raw'].shape==(30,2) and np.sum(np.isinf(a['raw']))==1

def test_budget_reserve_settle_and_no_overspend():
    ledger=[];i=p.reserve_budget(ledger,'gpu',1080,6,40,1)
    initial=ledger[i]['charged_usd'];p.settle_budget(ledger,i,140)
    assert ledger[i]['charged_usd']<initial
    with pytest.raises(ValueError):p.settle_budget(ledger,i,1)
    with pytest.raises(RuntimeError):p.reserve_budget(ledger,'huge',100000,100,100,1)
    while True:
        try:p.reserve_budget(ledger,'more',1080,6,40,1)
        except RuntimeError:break
    assert sum(r['charged_usd'] for r in ledger)<=20

def test_crystal_cache_reads_all_ids_once(tmp_path,monkeypatch):
    n=10000;root=tmp_path/'original';s=Store(root,{'experiment':'synthetic_cache'})
    write_json(root/'dataset.json',{'records':[{'point_id':i,'target':i+1,'prompt':f'{i+1}='} for i in range(n)]})
    write_json(root/'tokenizer_provenance.json',dict(expected_levels=2,expected_hidden_dim=2))
    s.finish('dataset',[root/'dataset.json',root/'tokenizer_provenance.json'],'test')
    ids=np.arange(n,dtype=np.int32);x=np.stack([np.ones((n,2)),np.c_[ids,ids*2]]).astype(np.float32)
    a=root/'hidden/00000.npy';b=root/'hidden/00000_ids.npy'
    atomic(a,lambda f:np.save(f,x));atomic(b,lambda f:np.save(f,ids));s.finish('chunk/00000',[a,b],'test')
    monkeypatch.setitem(ph.MODELS,'crystal',('original',2,2))
    before=digest(a);cache=p.crystal_cache(tmp_path,tmp_path/'output')
    assert np.array_equal(np.load(cache+'/layer_01.npy'),x[1]) and digest(a)==before
    mtime=(tmp_path/'output/crystal_cache/layer_01.npy').stat().st_mtime_ns
    p.crystal_cache(tmp_path,tmp_path/'output')
    assert (tmp_path/'output/crystal_cache/layer_01.npy').stat().st_mtime_ns==mtime
    result=p.prepare(root,tmp_path/'zero_prepare',0,cache=cache)
    assert result['implicit_zero_distances']
