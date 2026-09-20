"""Saved-pilot adoption uses synthetic arrays only, with no model loading."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from numzig.fullrange.storage import atomic,digest,read_json,write_json
from numzig.taskrules.data import make_dataset,pilot_records
from numzig.taskrules.runtime import summarize_behavior,run_pilot
from numzig.taskrules import extract

stage=Path(__file__).with_name('adopt.py')
if stage.exists():
    spec=importlib.util.spec_from_file_location('taskrules_adopt',stage)
    adoption=importlib.util.module_from_spec(spec);spec.loader.exec_module(adoption)
else:
    from numzig.taskrules import adopt as adoption

def fixture(tmp_path):
    examples=make_dataset(n_targets=40,prompt_style='examples')
    instructions=make_dataset(n_targets=40,prompt_style='instruction')
    mixed=[instructions[i] if row['task'].endswith('4') else row for i,row in enumerate(examples)]
    for style,data in (('examples',examples),('instruction',instructions)):
        root=tmp_path/'source'/('pilot' if style=='examples' else 'pilot_instruction')/'crystal'
        selected=pilot_records(data);ids=np.array([r['point_id'] for r in selected],dtype=np.int32)
        hidden=np.broadcast_to(ids[None,:,None]+(10000 if style=='instruction' else 0),(3,len(ids),4)).astype(np.float32).copy()
        meta=dict(model_key='crystal',model_id='LLM360/Crystal',revision=extract.CRYSTAL_REVISION,
            model_source_sha256='synthetic-model',dtype='torch.float32',saved_dtype='float32',
            tokenizer_class='Synthetic',is_fast=True,special_token_policy='synthetic',expected_levels=3,
            expected_hidden_dim=4,hidden_state_semantics='synthetic',output_logits_scale=1.,model_loads=1,
            load_seconds=1. if style=='examples' else 2.,inference_source_sha256=digest(extract.__file__))
        behavior=[dict(row,exact_match=True) for row in selected]
        gate=dict(passed=True,scope='synthetic fixture')
        atomic(root/'hidden.npz',lambda f:np.savez(f,hidden=hidden,point_ids=ids))
        for name,value in (('tokens.json',selected),('behavior.json',behavior),('model.json',meta),('gate.json',gate)):
            write_json(root/name,value)
        receipt=dict(contract='old-'+style,files={name:digest(root/name) for name in adoption.FILES},
            summary=summarize_behavior(behavior),model='crystal',rows=len(ids),meta=meta)
        write_json(root/'receipt.json',receipt)
    return mixed

def test_adoption_selection_provenance_and_resume(tmp_path,monkeypatch):
    rows=fixture(tmp_path)
    monkeypatch.setattr(extract,'load_model',lambda *a,**k:pytest.fail('Adoption must not load a model'))
    commits=[]
    result=adoption.adopt_pilot(tmp_path/'source',tmp_path/'target','crystal',rows,'new-contract',lambda:commits.append(1))
    root=tmp_path/'target'/'pilot'/'crystal';selected=pilot_records(rows)
    assert len(commits)==2 and result['adopted_without_inference']
    with np.load(root/'hidden.npz') as z:
        assert z['point_ids'].tolist()==[r['point_id'] for r in selected]
        for i,row in enumerate(selected):
            assert np.all(z['hidden'][:,i]==row['point_id']+(10000 if row['task'].endswith('4') else 0))
    assert read_json(root/'tokens.json')==selected
    receipt=read_json(root/'receipt.json');assert receipt['contract']=='new-contract'
    meta=read_json(root/'model.json');assert meta['model_loads']==0 and meta['source_model_metadata']['instruction']['load_seconds']==2.
    provenance=read_json(root/'adoption.json')
    assert len(provenance['selection'])==len(selected)
    assert set(provenance['sources'])=={'examples','instruction'}
    before={p.name:(digest(p),p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file()}
    assert adoption.adopt_pilot(tmp_path/'source',tmp_path/'target','crystal',rows,'new-contract')['skipped']
    # It is a valid standard runtime pilot receipt, so run_pilot also skips inference.
    assert run_pilot(root,'crystal',selected,'new-contract')['skipped']
    assert before=={p.name:(digest(p),p.stat().st_mtime_ns) for p in root.iterdir() if p.is_file()}

def test_source_checksum_tampering_rejected(tmp_path):
    rows=fixture(tmp_path);path=tmp_path/'source'/'pilot'/'crystal'/'tokens.json'
    path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='corrupt pilot artifact'):
        adoption.adopt_pilot(tmp_path/'source',tmp_path/'target','crystal',rows,'new-contract')
    assert not (tmp_path/'target'/'pilot'/'crystal'/'receipt.json').exists()

def test_matching_raw_fields_required(tmp_path):
    rows=fixture(tmp_path);rows[0]=dict(rows[0],prompt=rows[0]['prompt']+'WRONG')
    with pytest.raises(ValueError,match='Saved stimulus differs'):
        adoption.adopt_pilot(tmp_path/'source',tmp_path/'target','crystal',rows,'new-contract')

def test_scientific_metadata_mismatch_rejected(tmp_path):
    rows=fixture(tmp_path);root=tmp_path/'source'/'pilot_instruction'/'crystal'
    meta=read_json(root/'model.json');meta['dtype']='torch.bfloat16';write_json(root/'model.json',meta)
    receipt=read_json(root/'receipt.json');receipt['meta']=meta;receipt['files']['model.json']=digest(root/'model.json');write_json(root/'receipt.json',receipt)
    with pytest.raises(ValueError,match='metadata differs: dtype'):
        adoption.adopt_pilot(tmp_path/'source',tmp_path/'target','crystal',rows,'new-contract')
