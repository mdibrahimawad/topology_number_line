"""Download completed small PH artifacts; never copy large input/distance arrays."""
import hashlib,json,sys
from pathlib import Path
import modal
repo=Path('/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL')
sys.path.insert(0,str(repo))
from numzig.ph import PILOT
v=modal.Volume.from_name('numberline-zigzag-results',environment_name='main')
remote='/ph_fullrange_v1'
out=repo/'results/ph_execution_20260919/lightweight'
out.mkdir(parents=True,exist_ok=True)
def read(path):return b''.join(v.read_file(path))
for name in ['budget_ledger.json','pilot_report.json','pilot_progress.json']:
 try:(out/name).write_bytes(read(remote+'/'+name))
 except (FileNotFoundError,modal.exception.NotFoundError):pass
for model,layers in PILOT.items():
 for layer in layers:
  rel=f'{model}/layer_{layer:02d}'
  try:raw=read(remote+'/'+rel+'/manifest.json')
  except (FileNotFoundError,modal.exception.NotFoundError):continue
  m=json.loads(raw);dest=out/rel;dest.mkdir(parents=True,exist_ok=True)
  for key,item in m['artifacts'].items():
   if key not in ('h0','h1','plots') or item['status']!='complete':continue
   for name,sha in item['files'].items():
    p=dest/name
    if not p.resolve().is_relative_to(dest.resolve()):raise ValueError('Unsafe artifact path')
    if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==sha:continue
    data=read(remote+'/'+rel+'/'+name)
    if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Download hash mismatch: '+name)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
  (dest/'manifest.json').write_bytes(raw)
  print(model,layer,[k for k in ('h0','h1','plots') if k in m['artifacts'] and m['artifacts'][k]['status']=='complete'],flush=True)
print(out)
