import numpy as np
import pytest
from numzig.taskrules import runtime, extract


def test_chunk_recovery_preserves_valid_bytes(tmp_path, monkeypatch):
    rows=[dict(point_id=i,prompt=f'{i}=',expected_output=str(i)) for i in range(70)]
    calls=[]
    monkeypatch.setattr(extract,'load_model',lambda k:(None,None,{'test':True}))
    def infer(model,tok,part,key,batch_size=1):
        calls.append([r['point_id'] for r in part])
        return np.tile(np.array([r['point_id'] for r in part],dtype=np.float32)[None,:,None],(3,1,4)),part
    monkeypatch.setattr(extract,'infer_rows',infer)
    runtime.extract_shard(tmp_path,'crystal',rows,['00000'],'contract')
    saved=(tmp_path/'hidden/00000.npz').read_bytes()
    runtime.extract_shard(tmp_path,'crystal',rows,['00000','00001','00002'],'contract')
    assert len(calls)==3 and (tmp_path/'hidden/00000.npz').read_bytes()==saved
    runtime.extract_shard(tmp_path,'crystal',rows,['00001'],'contract')
    assert len(calls)==3
    (tmp_path/'hidden/00001.npz').write_bytes(b'incomplete')
    runtime.extract_shard(tmp_path,'crystal',rows,['00001'],'contract')
    assert len(calls)==4 and (tmp_path/'hidden/00000.npz').read_bytes()==saved
    with pytest.raises(ValueError):runtime.chunk_valid(tmp_path,'00000',rows[:32],'other')
