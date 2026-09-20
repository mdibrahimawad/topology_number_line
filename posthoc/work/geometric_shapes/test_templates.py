"""Run directly with an environment containing NumPy and SciPy."""
import numpy as np
from scipy.spatial.distance import pdist

from templates import make_templates, polyhedron_mesh, polyhedron_vertices


def main():
    templates = make_templates()
    assert len(templates) == 32
    assert len({t["name"] for t in templates}) == len(templates)
    for a, b in zip(templates, make_templates()):
        p = a["points"]
        assert p.shape[1] == 3 and 4 <= len(p) <= 384
        assert np.isfinite(p).all()
        assert np.max(np.abs(p.mean(axis=0))) < 1e-12
        assert abs(np.mean(np.sum(p * p, axis=1)) - 1) < 1e-12
        assert np.array_equal(p, b["points"])
        assert pdist(p).min() > 1e-10
    expected = {"tetrahedron": (4, 6), "cube": (8, 12), "octahedron": (6, 12),
                "icosahedron": (12, 30), "dodecahedron": (20, 30)}
    for name, p in polyhedron_vertices().items():
        triangles, edges = polyhedron_mesh(p)
        assert (len(p), len(edges)) == expected[name]
        lengths = np.linalg.norm(p[edges[:, 0]] - p[edges[:, 1]], axis=1)
        assert np.ptp(lengths) < 1e-12, name
    for name in ("line", "circle", "disk"):
        p = next(t["points"] for t in templates if t["name"] == name)
        assert np.max(np.abs(p[:, 2])) == 0
    assert all(len(t["points"]) <= 64 for t in make_templates(64))
    for invalid in (0, 63, 385, 3.5):
        try:
            make_templates(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid sample size accepted")
    print("Passed: 32 deterministic normalized templates, regular-polyhedron edges, "
          "sampling integrity and validation.")


if __name__ == "__main__":
    main()
