"""No downloads/cloud calls: synthetic native models and boundary edge cases."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

# The same tests also run from the read-only-development staging directory.
stage = Path(__file__).with_name('extract.py')
if stage.exists():
    spec = importlib.util.spec_from_file_location('taskrules_extract', stage)
    extraction = importlib.util.module_from_spec(spec); spec.loader.exec_module(extraction)
else:
    from numzig.taskrules import extract as extraction

class CharTokenizer:
    all_special_ids=[0,1,2]
    bos_token_id,eos_token_id,is_fast=1,2,False
    def __call__(self,text,**kwargs): return dict(input_ids=[ord(c) for c in text])
    def decode(self,ids,skip_special_tokens=False,**kwargs):
        return ''.join(chr(i) for i in ids if not skip_special_tokens or i not in self.all_special_ids)

def rows():
    return [dict(point_id=i,prompt=f'12=21,{n}=',expected_output=str(n)[::-1]) for i,n in enumerate([1,23,456,120,930,12])]

def test_tokenization_boundary_and_bos():
    tok=CharTokenizer(); data=rows()
    a=extraction.tokenize_rows(tok,data,'starcoder')
    b=extraction.tokenize_rows(tok,data,'openllama')
    assert all(y['input_token_ids']==[1]+x['input_token_ids'] for x,y in zip(a,b))
    assert a[3]['expected_output']=='021' and a[3]['first_continuation_token']==ord('0')
    assert 'token_ids' not in data[0]
    class Merge(CharTokenizer):
        def __call__(self,text,**kwargs):
            return dict(input_ids=[ord(c) for c in text[:-2]]+[127] if text.endswith('1=') else [ord(c) for c in text])
        def decode(self,ids,**kwargs): return ''.join('1=' if i==127 else chr(i) for i in ids)
    merged=extraction.tokenize_rows(Merge(),data[:1],'starcoder')[0]
    assert merged['final_equals_shares_token'] and not merged['continuation_prefix_valid']
    assert merged['first_continuation_token'] is None

@pytest.mark.parametrize('architecture',['llama','gpt_bigcode'])
def test_tiny_reference_and_rows_preserved(architecture):
    import torch
    from transformers import LlamaConfig,LlamaForCausalLM,GPTBigCodeConfig,GPTBigCodeForCausalLM
    torch.manual_seed(17);torch.set_num_threads(1)
    if architecture=='llama':
        cfg=LlamaConfig(vocab_size=128,hidden_size=24,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,max_position_embeddings=128)
        model=LlamaForCausalLM(cfg).eval();key='openllama'
    else:
        cfg=GPTBigCodeConfig(vocab_size=128,n_embd=24,n_layer=2,n_head=4,n_positions=128)
        model=GPTBigCodeForCausalLM(cfg).eval();key='starcoder'
    tok=CharTokenizer();data=rows()
    values,records=extraction.infer_rows(model,tok,data,key)
    assert values.shape==(3,6,24) and values.dtype==np.float32
    for i,r in enumerate(records):
        inputs=torch.tensor([r['input_token_ids']])
        with torch.inference_mode(): reference=model(input_ids=inputs,output_hidden_states=True,use_cache=False)
        expected=torch.stack([h[0,-1] for h in reference.hidden_states]).numpy()
        assert np.allclose(values[:,i],expected,atol=2e-5,rtol=2e-5)
        assert r['next_token_id']==int(reference.logits[0,-1].argmax())
    gate=extraction.equivalence_gate(model,tok,data,key)
    assert gate['passed']
    with pytest.raises(ValueError,match='one unpadded'): extraction.infer_rows(model,tok,data,key,batch_size=2)

def test_native_wrapper_output_scaling_is_preserved():
    import torch
    from transformers import LlamaConfig,LlamaForCausalLM
    model=LlamaForCausalLM(LlamaConfig(vocab_size=128,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2)).eval()
    native_forward=model.forward
    def scaled_forward(**kwargs):
        result=native_forward(**kwargs);result.logits*=.13875;return result
    model.forward=scaled_forward
    _,records=extraction.infer_rows(model,CharTokenizer(),rows()[:1],'openllama')
    with torch.inference_mode():
        expected=model(input_ids=torch.tensor([records[0]['input_token_ids']])).logits[0,-1].softmax(-1)
    assert records[0]['next_token_probability']==float(expected.max())
    assert records[0]['logits_source'].startswith('same native full-LM forward')

def test_behavior_preserves_zeroes_and_bounds_generation():
    import torch
    class Generator(torch.nn.Module):
        def __init__(self,text):
            super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self.config=SimpleNamespace(max_position_embeddings=128);self.text=text
        def generate(self,input_ids,**kwargs):
            assert kwargs['use_cache'] and not kwargs['do_sample'] and kwargs['num_beams']==1
            assert kwargs['max_new_tokens']<=32
            assert kwargs['stopping_criteria'](torch.tensor([input_ids[0].tolist()+[ord(c) for c in self.text]]),None)
            return torch.tensor([input_ids[0].tolist()+[ord(c) for c in self.text]])
    tok=CharTokenizer();row=dict(point_id=0,prompt='120=',expected_output='021')
    result=extraction.behavior(Generator('021,23='),tok,[row],'starcoder')[0]
    assert result['exact_correct'] and result['predicted_output']=='021' and result['stopped_at_delimiter']
    assert result['max_new_tokens']==8 and not result['hit_token_limit']
    word=dict(point_id=0,prompt='one hundred=',expected_output='one hundred')
    result=extraction.behavior(Generator('one hundred\n'),tok,[word],'starcoder')[0]
    assert result['exact_correct'] and result['max_new_tokens']==32
    with pytest.raises(ValueError,match='Generation limit'): extraction.behavior(Generator(''),tok,[row],'starcoder',max_new_tokens=9)

def test_decode_prefix_failure_is_explicit():
    class Context(CharTokenizer):
        def decode(self,ids,**kwargs):
            s=super().decode(ids,**kwargs)
            return s.replace('1=2','1 =2')
    answer,text,valid,boundary=extraction.generated_answer(Context(),[49,61],[49,61,50,44])
    assert not valid and answer=='2' and boundary
