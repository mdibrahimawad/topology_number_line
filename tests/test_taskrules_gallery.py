import json

import numpy as np

from numzig.fullrange.storage import digest, write_json, write_npz
from numzig.taskrules.gallery import build_gallery


def test_partial_gallery_behavior_labels_and_source_preservation(tmp_path):
    group = tmp_path / 'crystal/analysis/reverse4_ctx0'
    archive = group / 'layers/layer_01.npz'
    write_npz(archive, scores3=np.arange(18, dtype=float).reshape(6, 3), targets=np.arange(1234, 1240),
              point_ids=np.arange(6), input_first_digit=np.array(['1'] * 6),
              output_first_label=np.array(['4', '5', '6', '7', '8', '9']),
              token_counts=np.arange(20, 26), explained_variance_ratio=np.array([.5, .3, .1]))
    picture = group / 'layers/layer_01_pca.png'
    picture.write_bytes(b'fixture image not rendered by this test')
    write_json(group / 'summary.json', dict(model='crystal', task='reverse4', context_id='0', status='complete',
        layer_count=1, point_count=6, layers=[dict(layer=1, status='valid', arrays='layers/layer_01.npz',
            figures=['layers/layer_01_pca.png'], pca=dict(explained_variance_ratio=[.5, .3, .1],
            spearman_absolute=[.8, .2, .1]), ph=dict(status='not_requested'))]))
    rows = [dict(task='reverse4', context_id=0, target=n, input_text=str(n), expected_output=str(n)[::-1]) for n in range(1234, 1240)]
    write_json(tmp_path / 'dataset.json', rows)
    write_json(tmp_path / 'pilot/crystal/behavior.json', [dict(**r, exact_match=i == 0) for i, r in enumerate(rows)])
    original = {p: digest(p) for p in group.rglob('*') if p.is_file()}
    result = build_gallery(tmp_path)
    assert result['status'] == 'partial'
    assert result['completed_groups'] == 1 and len(result['missing_groups']) == 35
    assert result['completed_comparison_levels'] == 0
    assert result['missing_levels']
    assert result['groups'][0]['behavior']['exact_accuracy'] == 1 / 6
    assert result['groups'][0]['behavior']['warning']
    path = tmp_path / result['groups'][0]['layers'][0]['data3d']
    data = json.loads(path.read_text())
    assert data['expected_output'][0] == '4321'
    assert data['input_text'][0] == '1234'
    assert data['point_ids'] == list(range(6))
    assert all(digest(p) == checksum for p, checksum in original.items())
    assert (tmp_path / 'gallery/plotly.min.js').stat().st_size > 1_000_000
    before = path.stat().st_mtime_ns
    assert build_gallery(tmp_path) == result
    assert path.stat().st_mtime_ns == before
    picture.unlink()
    assert build_gallery(tmp_path)['completed_groups'] == 0


def test_empty_gallery_is_explicitly_partial(tmp_path):
    result = build_gallery(tmp_path)
    assert result['status'] == 'partial' and result['completed_groups'] == 0
    assert len(result['missing_groups']) == 36
    assert 'partial' in (tmp_path / 'SUMMARY.md').read_text()
