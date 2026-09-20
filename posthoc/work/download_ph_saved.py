from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json
import modal

out=Path('/Users/mohammed.awad/Documents/Codex/2026-09-19/c/outputs/ph_fullrange_overnight/results/results')
volume=modal.Volume.from_name('numberline-zigzag-results',environment_name='main')
def fetch(remote,local,sha=None):
    if local.is_file() and sha and hashlib.sha256(local.read_bytes()).hexdigest()==sha:return
    local.parent.mkdir(parents=True,exist_ok=True)
    partial=local.with_name(local.name+'.download')
    with partial.open('wb') as f:
        for block in volume.read_file(remote):f.write(block)
    if sha and hashlib.sha256(partial.read_bytes()).hexdigest()!=sha:raise ValueError('Checksum mismatch: '+remote)
    partial.replace(local)
def layer(item):
    model,level=item
    relative=f'{model}/layer_{level:02d}'
    remote='/ph_overnight_v1/results/'+relative
    local=out/relative
    fetch(remote+'/manifest.json',local/'manifest.json')
    manifest=json.loads((local/'manifest.json').read_text())
    for record in manifest['artifacts'].values():
        for name,sha in record['files'].items():
            dest=(local/name).resolve()
            assert dest.is_relative_to(local.resolve())
            fetch(remote+'/'+name,dest,sha)
    return relative
items=[(m,l) for m,n in [('crystal',33),('starcoderbase-3b',37),('openllama-3b',27)] for l in range(n)]
with ThreadPoolExecutor(max_workers=8) as pool:
    for i,name in enumerate(pool.map(layer,items),1):print(f'{i}/97 downloaded and checksummed: {name}',flush=True)
