"""Independent final audit of all saved geometric registrations.

Run only after the scan completes. This checks saved results, never refits or
changes them; direct pairwise distances independently reproduce held-out scores.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import time

import numpy as np
from scipy.spatial.distance import cdist

from inventory import ROOT, inventory, load_cloud
from templates import make_templates

OUT = ROOT / "outputs/geometric_shapes"
VERSION = "regular32-v2"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def assert_close(actual, expected, label, atol=1e-9, rtol=1e-8):
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def check_rotation(rotation, label):
    rotation = np.asarray(rotation, float)
    assert rotation.shape == (3, 3) and np.isfinite(rotation).all(), label
    assert_close(rotation.T @ rotation, np.eye(3), label + " orthogonality")
    assert_close(abs(np.linalg.det(rotation)), 1., label + " determinant")
    return rotation


def distances(points, shape):
    """Direct pairwise distances, independent of production cKDTree scoring."""
    pairwise = cdist(points, shape)
    return pairwise.min(axis=1), pairwise.min(axis=0), pairwise.argmin(axis=1)


def recompute_metrics(result, normalized, template, saved):
    scale = saved["scale"]
    shape = scale * template @ np.array(saved["rotation"]) + np.array(saved["offset"])
    norm = result["normalization"]
    train, test = normalized[norm["train_ids"]], normalized[norm["test_ids"]]
    d, coverage, closest = distances(test, shape)
    td, tc, _ = distances(train, shape)
    metrics = dict(
        train_symmetric_rms=float(np.sqrt((np.mean(td ** 2) + np.mean(tc ** 2)) / 2)),
        symmetric_rms=float(np.sqrt((np.mean(d ** 2) + np.mean(coverage ** 2)) / 2)),
        data_rms=float(np.sqrt(np.mean(d ** 2))),
        coverage_rms=float(np.sqrt(np.mean(coverage ** 2))),
        data_p95=float(np.quantile(d, .95)), data_p99=float(np.quantile(d, .99)),
        coverage_p95=float(np.quantile(coverage, .95)),
        data_within_015=float(np.mean(d <= .15)),
        shape_within_015=float(np.mean(coverage <= .15)),
    )
    # Exact duplicate vertex distances can choose different tied nearest indices;
    # compare occupancy only when the nearest assignment is unambiguous.
    pairwise = cdist(test, shape)
    nearest = np.partition(pairwise, 1, axis=1)[:, :2]
    unambiguous = bool(np.all(abs(nearest[:, 1] - nearest[:, 0]) > 1e-10))
    counts = np.bincount(closest, minlength=len(shape))
    if unambiguous:
        metrics["occupied_template_fraction"] = float(np.mean(counts > 0))
        if "vertex_occupancy" in saved:
            assert_close(counts / len(test), saved["vertex_occupancy"], "vertex occupancy")
    for key, value in metrics.items():
        assert_close(value, saved[key], f"{result['entry']['id']} / {saved['name']} / {key}")
    return {"view": result["entry"]["id"], "template": saved["name"],
            "metrics_compared": list(metrics), "tie_free_occupancy": unambiguous,
            "data_rms": metrics["data_rms"], "coverage_rms": metrics["coverage_rms"]}


def main():
    start = time.monotonic()
    entries = inventory()
    assert len(entries) == 1649
    expected_counts = {
        "original": {"degenerate": 2, "valid": 95},
        "new": {"degenerate": 32, "valid": 1132},
        "overlay": {"degenerate": 8, "valid": 380},
    }
    counts = {group: dict(Counter(e["status"] for e in entries if e["group"] == group))
              for group in expected_counts}
    assert counts == expected_counts, counts
    expected_files = {f"{e['id']}.json" for e in entries}
    existing_files = {p.name for p in (OUT / "fits").glob("*.json")}
    assert existing_files == expected_files, {
        "missing": sorted(expected_files - existing_files),
        "unexpected": sorted(existing_files - expected_files),
    }
    templates = {t["name"]: t for t in make_templates()}
    assert len(templates) == 32
    saved_templates = json.loads((OUT / "templates.json").read_text())
    assert saved_templates == [{k: v for k, v in t.items() if k != "points"}
                               for t in templates.values()]
    valid = [e for e in entries if e["status"] == "valid"]
    assert len(valid) == 1607
    # Fixed evenly spaced coverage is chosen without looking at fit quality.
    sample_ids = {valid[i]["id"] for i in np.linspace(0, len(valid) - 1, 20, dtype=int)}
    residual_checks = []
    status_counts = Counter()
    source_hashes = {}
    fit_hashes = {}
    max_normalization_error = 0.
    for i, entry in enumerate(entries):
        path = OUT / "fits" / f"{entry['id']}.json"
        result = json.loads(path.read_text())
        assert result["entry"] == entry, entry["id"]
        assert result.get("version") == VERSION, entry["id"]
        assert not result.get("paused") and not result.get("placeholder"), entry["id"]
        source_hash = digest(ROOT / entry["source"])
        assert result["source_sha256"] == source_hash, entry["id"] + " changed source"
        source_hashes[entry["source"]] = source_hash
        fit_hashes[path.name] = digest(path)
        y, targets, labels, evr = load_cloud(entry)
        assert_close(evr, entry["evr"] if entry["evr"] else [0, 0, 0], entry["id"] + " EVR")
        status_counts[entry["status"]] += 1
        if entry["status"] == "degenerate":
            assert "fits" not in result and "normalization" not in result, entry["id"]
            assert np.max(np.linalg.norm(y - y.mean(0), axis=1)) < 1e-9, entry["id"]
            continue
        fits = result["fits"]
        assert result["fits_count"] == len(fits) == 32, entry["id"]
        assert {fit["name"] for fit in fits} == set(templates), entry["id"]
        assert len({fit["name"] for fit in fits}) == 32, entry["id"]
        assert [fit["symmetric_rms"] for fit in fits] == sorted(fit["symmetric_rms"] for fit in fits)
        norm = result["normalization"]
        train_ids = np.array(norm["train_ids"], dtype=int)
        test_ids = np.array(norm["test_ids"], dtype=int)
        assert len(np.unique(train_ids)) == len(train_ids)
        assert len(np.unique(test_ids)) == len(test_ids)
        assert not np.intersect1d(train_ids, test_ids).size, entry["id"]
        assert np.all((train_ids >= 0) & (train_ids < len(y)))
        assert np.all((test_ids >= 0) & (test_ids < len(y)))
        order = np.random.default_rng(1729).permutation(len(y))
        full_training = order[:len(y) // 2]
        assert np.array_equal(train_ids, full_training[:384]), entry["id"]
        assert np.array_equal(test_ids, order[len(y) // 2:]), entry["id"]
        assert result["train_count"] == len(train_ids) and result["test_count"] == len(test_ids)
        expected_center = y[full_training].mean(axis=0)
        expected_radius = np.sqrt(np.mean(np.sum((y[full_training] - expected_center) ** 2, axis=1)))
        assert norm["radius"] > 0 and np.isfinite(norm["radius"])
        assert_close(norm["center"], expected_center, entry["id"] + " center")
        assert_close(norm["radius"], expected_radius, entry["id"] + " radius")
        max_normalization_error = max(max_normalization_error, float(np.max(abs(np.array(norm["center"]) - expected_center))), abs(norm["radius"] - expected_radius))
        basis = check_rotation(norm["basis"], entry["id"] + " normalization basis")
        normalized = ((y - np.array(norm["center"])) / norm["radius"]) @ basis.T
        assert np.isfinite(normalized).all()
        for saved in fits:
            name = saved["name"]
            assert all(saved[k] == v for k, v in templates[name].items() if k != "points")
            check_rotation(saved["rotation"], entry["id"] + " " + name)
            assert np.array(saved["offset"]).shape == (3,) and np.isfinite(saved["offset"]).all()
            scale = saved["scale"]
            assert np.isfinite(scale) and .25 - 1e-12 <= scale <= 3. + 1e-12
            assert saved["scale_at_bound"] == (scale <= .25001 or scale >= 2.99999)
            for key in ("data_rms", "coverage_rms", "symmetric_rms", "train_symmetric_rms", "data_p95", "data_p99", "coverage_p95"):
                assert np.isfinite(saved[key]) and saved[key] >= 0, (entry["id"], name, key)
            assert_close(saved["symmetric_rms"] ** 2, (saved["data_rms"] ** 2 + saved["coverage_rms"] ** 2) / 2, "symmetric score identity")
            for key in ("data_within_015", "shape_within_015", "occupied_template_fraction"):
                assert np.isfinite(saved[key]) and 0 <= saved[key] <= 1
            expected_pass = (saved["data_rms"] <= .15 and saved["coverage_rms"] <= .15
                             and saved["data_p95"] <= .30 and saved["coverage_p95"] <= .30
                             and not saved["scale_at_bound"])
            assert saved["screen_pass"] == expected_pass
        assert result["screen_matches"] == [f["name"] for f in fits if f["screen_pass"]]
        if entry["id"] in sample_ids:
            chosen = [fits[0]["name"], "tetrahedron_vertices", "helix_2turn_pitch1"]
            chosen = list(dict.fromkeys(chosen))
            if len(chosen) < 3:
                chosen += [name for name in ("sphere", "cube_surface") if name not in chosen][:3 - len(chosen)]
            for name in chosen:
                saved = next(f for f in fits if f["name"] == name)
                residual_checks.append(recompute_metrics(result, normalized, templates[name]["points"], saved))
        if (i + 1) % 200 == 0:
            print(f"Checked {i + 1}/{len(entries)} views", flush=True)
    assert dict(status_counts) == {"degenerate": 42, "valid": 1607}
    assert len(residual_checks) == 60
    validation = dict(
        status="passed", version=VERSION, checked_at=datetime.now(timezone.utc).isoformat(),
        views=len(entries), valid_views=1607, degenerate_views=42,
        fits_per_valid_view=32, registrations_checked=1607 * 32,
        group_status_counts=counts, source_hashes=source_hashes, fit_file_hashes=fit_hashes,
        independent_distance_recomputations=residual_checks,
        sample_selection="20 evenly spaced nondegenerate inventory views; top fit plus tetrahedron vertices and two-turn pitch-1 helix, with a sphere fallback to keep three distinct templates",
        normalization="Seed1729 half-split; all training-half points set center/radius; first384 training indices fit poses; all remaining-half indices test",
        max_center_or_radius_recompute_error=max_normalization_error,
        uniform_similarity_transforms=True, reflection_allowed=True,
        scalar_scale_bounds=[.25, 3.], candidate_metadata_verified=True,
        finite_coordinates_and_scores=True, train_test_ids_disjoint=True,
        sources_unchanged=True, seconds=time.monotonic() - start,
        auditor_sha256=digest(__file__), template_generator_sha256=digest(Path(__file__).with_name("templates.py")),
    )
    temporary = OUT / "validation.json.tmp"
    temporary.write_text(json.dumps(validation, indent=2, allow_nan=False))
    temporary.replace(OUT / "validation.json")
    print(json.dumps({k: validation[k] for k in ("status", "views", "registrations_checked", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
