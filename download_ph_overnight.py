"""Download/validate the finished lightweight archive; starts no compute jobs."""
from pathlib import Path
import argparse,hashlib,json,tarfile
import modal
import numpy as np
from numzig import ph
from numzig.fullrange.storage import Store,digest,write_json

def main(output):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    volume=modal.Volume.from_name('numberline-zigzag-results',environment_name='main')
    remote='/ph_overnight_v1'
    def read(name):return b''.join(volume.read_file(remote+'/'+name))
    report=json.loads(read('report.json'));package=json.loads(read('package.json'))
    path=output/'lightweight.tar.gz'
    if not path.exists() or digest(path)!=package['sha256']:
        partial=path.with_suffix('.partial')
        with partial.open('wb') as f:
            for block in volume.read_file(remote+'/lightweight.tar.gz'):f.write(block)
        if digest(partial)!=package['sha256']:raise ValueError('Archive checksum mismatch')
        partial.replace(path)
    with tarfile.open(path,'r:gz') as tar:
        for m in tar.getmembers():
            if m.issym() or m.islnk() or m.isdev() or not (output/m.name).resolve().is_relative_to(output):
                raise ValueError('Unsafe archive member')
        tar.extractall(output,filter='data')
    rows=[]
    for p in sorted((output/'results').glob('*/layer_*/manifest.json')):
        store=Store(p.parent)
        if not all(store.valid(k) for k in ['h0','h1','plots']):continue
        config=store.manifest['config'];r=json.loads((p.parent/'metrics.json').read_text())
        assert config['source']['shape'][0]==10000 and r['point_count']==10000 and r['normalized_threshold'] is None
        assert config['coeff']==2 and r['filtration_dtype']=='float32'
        assert config['code_sha256']==digest(Path(__file__).parent/'numzig/ph_overnight.py')
        scale=r['normalization']['scale']
        with np.load(p.parent/'h0.npz') as a,np.load(p.parent/'h1.npz') as b:
            assert a['raw'].shape==(10000,2) and np.isinf(a['raw'][:,1]).sum()==1
            assert np.array_equal(a['point_ids'],np.arange(10000))
            assert np.isfinite(b['raw']).all() and not b['right_censored'].any()
            assert len(b['raw'])==r['h1_finite_count'] and np.all(b['raw'][:,1]>=b['raw'][:,0])
            assert np.array_equal(b['normalized'],b['raw']/scale if scale else b['raw'])
        rows.append((p.parent.parent.name,r['layer']))
    expected={(m,l) for m,v in ph.MODELS.items() for l in range(v[1])}
    complete=set(rows)==expected and len(rows)==97
    assert len(rows)==package['completed']
    if report['state']=='complete' and not complete:raise ValueError('Completion report disagrees with downloaded layers')
    validation=dict(validated_layers=len(rows),expected_layers=97,all_layers_complete=complete,
        missing=sorted(expected-set(rows)),scope='Artifact hashes, configurations, point IDs, diagrams and normalization; not independent full-cloud H1 recomputation.')
    for name,value in [('validation.json',validation),('cloud_report.json',report),('package.json',package)]:write_json(output/name,value)
    (output/'budget.json').write_bytes(read('budget.json'))
    print(json.dumps(validation,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    main(parser.parse_args().output)
