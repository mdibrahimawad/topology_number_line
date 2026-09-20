"""Synthetic geometry checks; these calibrate code, not real-data significance."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from fit import fit_all, split_normalize
from templates import make_templates, polyhedron_mesh, polyhedron_vertices


def main():
    rng = np.random.default_rng(24601)
    templates = {t['name']: t for t in make_templates()}
    tests = {}

    def run(name, points, shape):
        result = fit_all(points, [templates[shape]])
        tests[name] = result['fits'][0]
        return result['fits'][0]

    vertices = templates['tetrahedron_vertices']['points']
    tetra = np.repeat(vertices, 250, axis=0) + rng.normal(0, .012, (1000, 3))
    tetra_fit = run('tetrahedron', tetra, 'tetrahedron_vertices')
    assert tetra_fit['screen_pass'], tetra_fit
    assert min(tetra_fit['vertex_occupancy']) > .15, tetra_fit

    # Independent, continuous samples rather than resampling the fitter's template.
    sphere = rng.normal(size=(2000, 3))
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    sphere += rng.normal(0, .005, sphere.shape)
    sphere_fit = run('sphere', sphere, 'sphere')
    assert sphere_fit['screen_pass'], sphere_fit

    theta = rng.uniform(-2*np.pi, 2*np.pi, 1800)
    helix = np.column_stack((np.cos(theta), np.sin(theta), theta/(2*np.pi)))
    helix += rng.normal(0, .004, helix.shape)
    helix_fit = run('helix_2turn', helix, 'helix_2turn_pitch1')
    assert helix_fit['screen_pass'], helix_fit

    # Similarity transforms must not change a shape classification. Reflection is allowed.
    rotation = Rotation.from_rotvec([.7, -.4, 1.1]).as_matrix()
    rotation[:, 0] *= -1
    transformed = 4.7*helix@rotation + np.array([19., -5., 43.])
    invariant = run('helix_transformed', transformed, 'helix_2turn_pitch1')
    assert invariant['screen_pass'], invariant
    assert abs(helix_fit['symmetric_rms']-invariant['symmetric_rms']) < .03

    gaussian = rng.normal(size=(1800, 3))
    for shape in ('tetrahedron_vertices', 'circle'):
        fit = run('gaussian_vs_'+shape, gaussian, shape)
        assert not fit['screen_pass'], fit

    # A dense 70-degree arc must not be accepted as an entire populated circle.
    theta = rng.uniform(-.61, .61, 1800)
    arc = np.column_stack((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
    arc += rng.normal(0, .002, arc.shape)
    fit = run('partial_circle', arc, 'circle')
    assert not fit['screen_pass'], fit
    assert fit['coverage_rms'] > .15 or fit['coverage_p95'] > .30, fit

    # Fit and normalization use training data only, with a deterministic disjoint test half.
    train, test, info = split_normalize(helix)
    train_ids, test_ids = set(info['train_ids']), set(info['test_ids'])
    assert train_ids.isdisjoint(test_ids)
    assert len(train_ids) == len(train) == 384
    assert len(test_ids) == len(test) == len(helix)//2
    repeated = split_normalize(helix)
    np.testing.assert_array_equal(train, repeated[0])
    np.testing.assert_array_equal(test, repeated[1])
    changed = helix.copy()
    changed[list(test_ids)] *= 100
    changed_split = split_normalize(changed)
    np.testing.assert_array_equal(train, changed_split[0])
    assert info == changed_split[2]
    assert split_normalize(np.ones((1000, 3))) is None
    tests['split_and_degeneracy'] = {'passed': True}

    expected = {'tetrahedron':(4,6),'cube':(8,12),'octahedron':(6,12),
                'icosahedron':(12,30),'dodecahedron':(20,30)}
    for name, verts in polyhedron_vertices().items():
        _, edges = polyhedron_mesh(verts)
        assert (len(verts),len(edges)) == expected[name]
        lengths = np.linalg.norm(verts[edges[:,0]]-verts[edges[:,1]],axis=1)
        np.testing.assert_allclose(lengths,lengths[0])
    assert len(templates) == 32
    tests['template_edge_counts'] = {'passed':True,'expected':expected}
    out = Path(__file__).with_name('synthetic_validation.json')
    out.write_text(json.dumps({'passed':True,'tests':tests}, indent=2)+'\n')
    print(json.dumps({'passed':True,'file':str(out),'metrics':{
        name:{key:result[key] for key in ('data_rms','coverage_rms','screen_pass')}
        for name,result in tests.items() if 'data_rms' in result}},indent=2))


if __name__ == '__main__':
    main()
