from pathlib import Path
import json,collections
import numpy as np
from scipy.stats import spearmanr
BASE=Path('/Users/mohammed.awad/Documents/numberline_zigzag_modal_FINAL')
OUT=Path('outputs/ph_analysis/audit')
paths={
'crystal':BASE/'results/crystal_fullrange_modal_20260919T123307Z/full_complete/crystal_fullrange_1_10000_ctx1234_k4_seed42',
'starcoderbase-3b':BASE/'results/native_models_modal_20260919/full_complete/starcoderbase-3b_fullrange_1_10000_ctx1234_k4_seed42',
'openllama-3b':BASE/'results/native_models_modal_20260919/full_complete/openllama-3b_fullrange_1_10000_ctx1234_k4_seed42'}
result={}
for model,p in paths.items():
 ds=json.loads((p/'dataset.json').read_text())['records']; ms=json.loads((p/'layer_metrics.json').read_text());pr=json.loads((p/'tokenizer_provenance.json').read_text());base=json.loads((p/'baselines.json').read_text());val=json.loads((p/'dataset_validation.json').read_text())
 valid=[r for r in ms if r['layer']>0 and r['status']=='valid']
 selects={}
 for name,fn in [('best_pc1_rho',lambda x:x['pca']['spearman_absolute'][0]),('best_pc2_rho',lambda x:x['pca']['spearman_absolute'][1]),('best_ev1',lambda x:x['pca']['explained_variance_ratio'][0]),('best_ev2',lambda x:sum(x['pca']['explained_variance_ratio']))]:
  r=max(valid,key=fn);selects[name]={'layer':r['layer'],'value':fn(r),'relative_depth':r['layer']/(len(ms)-1)}
 demo=np.array([[r['demonstrations'][k] for k in 'ABCD']for r in ds]);targets=np.array([r['target']for r in ds]);length=np.array([r['token_count']for r in ds])
 rows={}
 for r in valid:
  ph=Path('outputs/ph_fullrange_overnight/results/results')/model/f"layer_{r['layer']:02d}"/'metrics.json'
  m=json.loads(ph.read_text());rows[r['layer']]={'rho':r['pca']['spearman_absolute'],'ev':r['pca']['explained_variance_ratio'],'same_leading_digit':r['same_leading_digit_fraction'],'mean_numeric_gap':r['absolute_numerical_gaps']['mean'],'cross_length_edges':r['cross_digit_length_count'],'same_digit_given_cross_length':r['same_leading_digit_given_cross_length'],'h1_count':m['h1_finite_count'],'h1_max_lifetime':m['h1_max_finite_lifetime'],'distance_scale':m['normalization']['scale'],'rounding_error':m['max_filtration_rounding_error'],'backend_gate_pass':m['backend_gate']['passed']}
 result[model]={'source':str(p),'levels':len(ms),'transformer_blocks':len(ms)-1,'hidden_dim':pr['expected_hidden_dim'],'embedding_semantics':pr['hidden_state_semantics'],'model_revision':pr['model_revision'],'dtype':'bfloat16'if model=='crystal'else pr['inference_dtype'],'tokenizer':pr['tokenizer_class'],'paper_checkpoint_match':pr.get('paper_checkpoint_match','unverified'),'selection':selects,'prompt_example':ds[0]['prompt'],'character_count_distribution':dict(collections.Counter(r['character_count']for r in ds)),'token_count_distribution':dict(collections.Counter(length.tolist())),'final_token_text_distribution':dict(collections.Counter(r['final_token_text']for r in ds)),'digit_count_distribution':dict(collections.Counter(r['digit_count']for r in ds)),'token_count_by_target_digit_count':{d:dict(collections.Counter(r['token_count']for r in ds if r['digit_count']==d))for d in range(1,6)},'leading_digit_counts':dict(collections.Counter(r['leading_digit']for r in ds)),'contexts_with_target_match':sum(any(r['demonstration_equals_target'])for r in ds),'context_target_match_counts':sum(np.array([r['demonstration_equals_target']for r in ds])).tolist(),'target_vs_demo_spearman':[float(spearmanr(targets,demo[:,i]).statistic)for i in range(4)],'target_vs_token_length_spearman':float(spearmanr(targets,length).statistic),'all_prompts_identical_to_crystal':all(a['prompt']==b['prompt']for a,b in zip(ds,json.loads((paths['crystal']/'dataset.json').read_text())['records'])),'numerical_knn_baseline':base['numerical_distance'],'random_label_baseline':base['frequency'],'layer_records':rows}
(OUT/'verified_facts.json').write_text(json.dumps(result,indent=2))
for k,v in result.items():
 print(k,json.dumps({a:v[a]for a in ['selection','prompt_example','token_count_distribution','contexts_with_target_match','context_target_match_counts','target_vs_demo_spearman','target_vs_token_length_spearman','all_prompts_identical_to_crystal']}))
 print('FINAL',v['layer_records'][v['transformer_blocks']])
