"""Fixed-proportion geometric templates, sampled reproducibly in three dimensions.

Templates distinguish vertices, one-dimensional frames and two-dimensional
surfaces. Centering and isotropic RMS scaling never change shape proportions.
"""

from itertools import product

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import qmc


def normalize(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Expected finite N by 3 coordinates")
    centered = points - points.mean(axis=0)
    rms = np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
    if rms == 0:
        raise ValueError("Cannot normalize a constant cloud")
    return centered / rms


def polyhedron_vertices():
    phi = (1 + np.sqrt(5)) / 2
    signs = list(product((-1., 1.), repeat=2))
    ico = [[0, a, b * phi] for a, b in signs]
    ico += [[a, b * phi, 0] for a, b in signs]
    ico += [[a * phi, 0, b] for a, b in signs]
    dodeca = list(product((-1., 1.), repeat=3))
    dodeca += [(0, a / phi, b * phi) for a, b in signs]
    dodeca += [(a / phi, b * phi, 0) for a, b in signs]
    dodeca += [(a * phi, 0, b / phi) for a, b in signs]
    return {
        "tetrahedron": np.array([[1, 1, 1], [1, -1, -1],
                                 [-1, 1, -1], [-1, -1, 1]], dtype=float),
        "cube": np.array(list(product((-1., 1.), repeat=3))),
        "octahedron": np.concatenate((np.eye(3), -np.eye(3))),
        "icosahedron": np.array(ico),
        "dodecahedron": np.array(dodeca),
    }


def polyhedron_mesh(vertices):
    """Return hull triangles and actual edges, excluding face diagonals."""
    hull = ConvexHull(vertices)
    incident = {}
    for face_id, face in enumerate(hull.simplices):
        for i, j in ((0, 1), (1, 2), (0, 2)):
            edge = tuple(sorted((int(face[i]), int(face[j]))))
            incident.setdefault(edge, []).append(face_id)
    edges = []
    for edge, faces in sorted(incident.items()):
        if len(faces) != 2:
            raise ValueError("Expected a closed convex hull")
        if np.linalg.norm(hull.equations[faces[0], :3]
                          - hull.equations[faces[1], :3]) > 1e-7:
            edges.append(edge)
    return hull.simplices, np.array(edges, dtype=int)


def _counts(weights, n):
    expected = n * np.asarray(weights) / np.sum(weights)
    counts = np.floor(expected).astype(int)
    order = np.argsort(-(expected - counts), kind="stable")
    counts[order[:n - counts.sum()]] += 1
    return counts


def _uv(n, offset=0):
    # Scrambled Sobol avoids grid symmetries; each invocation has a fixed seed.
    return qmc.Sobol(2, scramble=True, seed=24680 + offset).random_base2(
        int(np.ceil(np.log2(max(1, n))))
    )[:n]


def _edge_samples(vertices, edges, n):
    segments = vertices[edges]
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    out = []
    for segment, count in zip(segments, _counts(lengths, n)):
        t = (np.arange(count) + 0.5) / count
        out.append(segment[0] + t[:, None] * (segment[1] - segment[0]))
    return np.concatenate(out)


def _surface_samples(vertices, triangles, n):
    faces = vertices[triangles]
    areas = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0],
                                    faces[:, 2] - faces[:, 0]), axis=1) / 2
    out = []
    for i, (face, count) in enumerate(zip(faces, _counts(areas, n))):
        uv = _uv(count, i)
        root = np.sqrt(uv[:, 0])
        out.append((1 - root[:, None]) * face[0]
                   + (root * (1 - uv[:, 1]))[:, None] * face[1]
                   + (root * uv[:, 1])[:, None] * face[2])
    return np.concatenate(out)


def make_templates(n=384):
    """Return 32 templates; sampled models have n points, vertices have 4–20.

    ``dimension`` describes the ideal template, not an estimated data dimension.
    Sphere/cylinder/cone/torus are surfaces, not filled solids. Cylinder and cone
    have radius 1 and height 2 before isotropic normalization. Helix pitch is
    axial advance per full turn, divided by radius. Curves retain endpoints.
    """
    if not isinstance(n, int) or not 64 <= n <= 384:
        raise ValueError("n must be an integer between 64 and 384")
    templates = []

    def add(name, family, dimension, points, **metadata):
        templates.append(dict(name=name, family=family, dimension=dimension,
                              points=normalize(points), **metadata))

    for name, vertices in polyhedron_vertices().items():
        triangles, edges = polyhedron_mesh(vertices)
        add(name + "_vertices", "polyhedron_vertices", 0, vertices,
            solid=name, n_vertices=len(vertices), n_edges=len(edges))
        add(name + "_edges", "polyhedron_edges", 1,
            _edge_samples(vertices, edges, n), solid=name,
            n_vertices=len(vertices), n_edges=len(edges))
        add(name + "_surface", "polyhedron_surface", 2,
            _surface_samples(vertices, triangles, n), solid=name,
            n_vertices=len(vertices), n_edges=len(edges))

    t = np.linspace(-1, 1, n)
    zero = np.zeros(n)
    angle = 2 * np.pi * np.arange(n) / n
    add("line", "line", 1, np.column_stack((t, zero, zero)))
    add("circle", "circle", 1,
        np.column_stack((np.cos(angle), np.sin(angle), zero)))
    golden = np.pi * (3 - np.sqrt(5))
    spiral_angle = np.arange(n) * golden
    radius = np.sqrt((np.arange(n) + .5) / n)
    add("disk", "disk", 2, np.column_stack((radius * np.cos(spiral_angle),
                                            radius * np.sin(spiral_angle), zero)))
    z = 1 - 2 * (np.arange(n) + .5) / n
    radius = np.sqrt(1 - z * z)
    add("sphere", "sphere", 2, np.column_stack((radius * np.cos(spiral_angle),
                                                radius * np.sin(spiral_angle), z)))
    uv = _uv(n)
    angle = 2 * np.pi * uv[:, 0]
    add("cylinder", "cylinder", 2,
        np.column_stack((np.cos(angle), np.sin(angle), 2 * uv[:, 1] - 1)),
        height_over_radius=2., caps=False)
    radius = np.sqrt(uv[:, 1])
    add("cone", "cone", 2,
        np.column_stack((radius * np.cos(angle), radius * np.sin(angle),
                         1 - 2 * radius)), height_over_radius=2., base=False)

    for tube_radius in (.25, .5):
        phi = 2 * np.pi * uv[:, 1]
        theta = phi.copy()
        # Invert the tube angle's area-weighted CDF: theta + r*sin(theta).
        for _ in range(8):
            theta -= (theta + tube_radius * np.sin(theta) - phi) / (
                1 + tube_radius * np.cos(theta))
        radial = 1 + tube_radius * np.cos(theta)
        add(f"torus_r{tube_radius:g}", "torus", 2,
            np.column_stack((radial * np.cos(angle), radial * np.sin(angle),
                             tube_radius * np.sin(theta))),
            tube_over_major_radius=tube_radius)

    for turns in (1, 2, 3):
        theta = np.linspace(-np.pi * turns, np.pi * turns, n)
        for pitch in (.5, 1., 2.):
            add(f"helix_{turns}turn_pitch{pitch:g}", "helix", 1,
                np.column_stack((np.cos(theta), np.sin(theta),
                                 pitch * theta / (2 * np.pi))),
                turns=turns, pitch_per_turn_over_radius=pitch,
                closed=False, handedness="right")
    return templates


if __name__ == "__main__":
    for template in make_templates():
        print(template["name"], len(template["points"]), template["dimension"])
