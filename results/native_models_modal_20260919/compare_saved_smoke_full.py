"""Compare common saved prompts across one-worker smoke and ten-worker full runs."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist

B = Path(__file__).resolve().parent
reports = {}
for model in ('starcoderbase-3b', 'openllama-3b'):
    name = model + '_fullrange_1_10000_ctx1234_k4_seed42'
    smoke, full = B / 'smoke_complete' / (name + '_smoke'), B / 'full_complete' / name
    paths = sorted((smoke / 'hidden').glob('*_ids.npy'))
    ids = np.concatenate([np.load(p) for p in paths])
    reference = np.concatenate([np.load(str(p).replace('_ids.npy', '.npy')) for p in paths], axis=1).astype(float)
    actual = np.stack([np.load(full / f'hidden/{int(i)//32:05d}.npy', mmap_mode='r')[:, int(i)%32, :] for i in ids], axis=1).astype(float)
    np.testing.assert_allclose(actual, reference, atol=2e-5, rtol=2e-5)
    delta = actual - reference
    relative = np.linalg.norm(delta, axis=-1) / np.maximum(np.linalg.norm(reference, axis=-1), 1e-30)
    cosine = np.abs(1 - np.sum(actual*reference, axis=-1) / (np.linalg.norm(actual, axis=-1)*np.linalg.norm(reference, axis=-1)))
    assert relative.max() <= 2e-5 and cosine.max() <= 2e-6
    for x, y in zip(actual, reference):
        d, ref_d = cdist(x, x), cdist(y, y)
        np.testing.assert_allclose(d, ref_d, atol=2e-5, rtol=2e-5)
        np.fill_diagonal(d, np.inf); np.fill_diagonal(ref_d, np.inf)
        np.testing.assert_array_equal(np.argsort(d, axis=1, kind='stable')[:, :4], np.argsort(ref_d, axis=1, kind='stable')[:, :4])
    reports[model] = dict(passed=True, points=len(ids), smoke_workers=1, full_workers=10,
        exact_equal=bool(np.array_equal(actual, reference)), max_absolute=float(np.abs(delta).max()),
        maximum_relative_l2=float(relative.max()), maximum_cosine_error=float(cosine.max()),
        vector_distance_and_ordered_knn_checks_passed=True,
        scope='Same 40 saved prompts; neighbor comparison uses the common 40-point candidate set. Full 10,000-point graphs audited separately.')
(B / 'real_smoke_full_vector_comparison.json').write_text(json.dumps(reports, indent=2))
print(json.dumps(reports, indent=2))
