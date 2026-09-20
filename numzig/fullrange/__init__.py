"""Crystal full-range experiment; deliberately independent of the topology pipeline."""

NAME = 'crystal_fullrange_1_10000_ctx1234_k4_seed42'
MODEL_ID = 'LLM360/Crystal'
REVISION = '34fc9cd58acd87002560379a95b432147cc9135a'
POLICY = 'A=A,B=B,C=C,D=D,E=; independent uniform 1/2/3/4-digit demonstrations'


def configuration(smoke=False):
    return dict(schema=1, experiment=NAME + ('_smoke' if smoke else ''), smoke=smoke,
                model_id=MODEL_ID, model_revision=REVISION, tokenizer_revision=REVISION,
                tokenizer_code_repo='LLM360/CrystalCoder', tokenizer_code_revision=REVISION,
                seed=42, k=4, chunk_size=32, prompt_policy=POLICY,
                model_dtype='bfloat16', saved_dtype='float32', batch_size=1,
                prepend_bos=False, add_special_tokens=True, trust_remote_code=True,
                extraction='final meaningful equals; all returned states',
                distance='scipy.cdist euclidean float64 direct differences; distance then point_id',
                pca='full SVD float64, centered, no scaling/whitening; positive target rho signs')
