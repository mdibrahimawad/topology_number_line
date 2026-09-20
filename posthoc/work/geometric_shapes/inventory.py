"""Read-only adapters for existing individual and shared-PCA point clouds.

Every NPZ is authoritative; PNG/JSON browser coordinates are never fitted.
Shared-PCA overlays reuse the individual observations and are descriptive views,
not independent experiments. Embedding-level StarCoder clouds encode position.
"""
from functools import lru_cache
from pathlib import Path
import json

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "outputs/pca3d"
NEW = ROOT / "outputs/taskrules/run"
MODEL_NAMES = {"crystal": "crystal", "starcoderbase-3b": "starcoder", "openllama-3b": "openllama"}


def inventory():
    """Return 97 original + 1164 new + 388 secondary overlay entries."""
    entries = []
    old = json.loads((OLD / "index.json").read_text())
    for group in old["models"]:
        model = group["id"]
        for level in group["layers"]:
            layer = level["layer"]
            entries.append(dict(
                id=f"original_{model}_L{layer:02d}",
                source=str((OLD / level["download"]).relative_to(ROOT)),
                group="original", model=model, model_source=model,
                task="original_copy", context=None, layer=layer,
                evr=level["variance"], layer_kind="embedding" if layer == 0 else "contextual",
                positional=model == "starcoder" and layer == 0,
                status=level["status"], point_count=10000,
                primary_view=True,
            ))
    new = json.loads((NEW / "gallery/manifest.json").read_text())
    for group in new["groups"]:
        model_source = group["model"]
        model = MODEL_NAMES[model_source]
        for level in group["layers"]:
            layer = level["layer"]
            task, context = group["task"], int(group["context_id"])
            entries.append(dict(
                id=f"new_{model}_{task}_ctx{context}_L{layer:02d}",
                source=str((NEW / level["arrays"]).relative_to(ROOT)),
                group="new", model=model, model_source=model_source,
                task=task, context=context, layer=layer,
                evr=level["explained_variance_ratio"],
                layer_kind="embedding" if layer == 0 else "contextual",
                positional=model == "starcoder" and layer == 0 and level["status"] == "valid",
                status=level["status"], point_count=group["point_count"],
                primary_view=True,
            ))
    for comparison in new["comparisons"]:
        model_source = comparison["model"]
        model = MODEL_NAMES[model_source]
        layer, family, context = comparison["layer"], comparison["family"], int(comparison["context_id"])
        evr = comparison["explained_variance_ratio"]
        status = "valid" if sum(evr) > 1e-12 else "degenerate"
        entries.append(dict(
            id=f"overlay_{model}_{family}_ctx{context}_L{layer:02d}",
            source=str((NEW / comparison["arrays"]).relative_to(ROOT)),
            group="overlay", model=model, model_source=model_source,
            task=family, context=context, layer=layer, evr=evr,
            layer_kind="embedding" if layer == 0 else "contextual",
            positional=model == "starcoder" and layer == 0,
            status=status,
            point_count=comparison["point_count"] * len(comparison["tasks"]),
            task_names=comparison["tasks"], primary_view=False,
        ))
    assert len(entries) == 1649
    assert len({e["id"] for e in entries}) == len(entries)
    assert all((ROOT / e["source"]).is_file() for e in entries)
    return entries


def load_cloud(entry):
    """Return (scores Nx3, numeric targets N, labels N, EVR length3).

    Original/new labels are predefined input leading digits 1..9; overlay labels
    are 1-based task indices. Degenerate sources lacking scores return zeros and
    stay explicitly marked degenerate in the inventory.
    """
    with np.load(ROOT / entry["source"], allow_pickle=False) as arrays:
        targets = arrays["targets"].copy()
        scores = arrays["scores"].copy() if "scores" in arrays else np.zeros((len(targets), 3))
        evr = arrays["explained_variance_ratio"].copy() if "explained_variance_ratio" in arrays else np.zeros(3)
        if entry["group"] == "overlay":
            task_count, point_count, dimensions = scores.shape
            assert dimensions == 3 and point_count == len(targets)
            scores = scores.reshape(-1, 3)
            targets = np.tile(targets, task_count)
            labels = np.repeat(np.arange(1, task_count + 1), point_count)
        elif entry["group"] == "original":
            labels = arrays["leading_digits"].copy()
        else:
            labels = arrays["input_first_digit"].astype(int)
    assert scores.shape == (entry["point_count"], 3)
    assert len(targets) == len(labels) == len(scores)
    assert np.isfinite(scores).all() and np.isfinite(evr).all()
    return scores, targets, labels, evr


@lru_cache(maxsize=1)
def _old_metrics():
    data = json.loads((ROOT / "outputs/pca3d_analysis/metrics.json").read_text())
    return {(row["model"], row["layer"]): row for row in data}


def projection_diagnostics(entry, scores=None, targets=None):
    """Reuse full-space diagnostics; no original high-D vector load/download.

    Old views reuse validated all-point 4NN retention. New individual views use
    the 512 saved full-space pair distances from the matched comparison file,
    but compare these against the *individual* PCA scores being shape-fitted.
    No full-space result is inferred from the displayed shape alone.
    """
    if entry["status"] != "valid":
        return {"status": "unavailable", "reason": "degenerate cloud"}
    if entry["group"] == "original":
        row = _old_metrics()[(entry["model"], entry["layer"])]
        if "neighbor4_retention3d" not in row:
            return {"status": "unavailable", "reason": "positional embedding view"}
        keys = ("neighbor4_retention3d", "same_leading_neighbor4_3d", "same_leading_neighbor4_hidden")
        return {"status": "complete", "kind": "existing all-point original-space 4NN retention", **{k: row[k] for k in keys}}
    if entry["group"] == "overlay":
        return {"status": "unavailable", "reason": "saved pairs are within tasks; overlay includes between-task offsets"}
    if scores is None or targets is None:
        scores, targets, _, _ = load_cloud(entry)
    family = "notation" if entry["task"] in ("numeric_copy", "word_copy") else "permutations"
    path = NEW / entry["model_source"] / "comparison" / f"ctx{entry['context']}" / family / f"layer_{entry['layer']:02d}.npz"
    with np.load(path, allow_pickle=False) as arrays:
        assert np.array_equal(targets, arrays["targets"])
        task_index = arrays["task_names"].tolist().index(entry["task"])
        a, b = arrays["pair_a"], arrays["pair_b"]
        full = arrays["fullspace_pair_distances"][task_index]
        projected = np.linalg.norm(scores[a] - scores[b], axis=1)
    denominator = float(full @ full)
    if denominator <= 0:
        return {"status": "unavailable", "reason": "zero full-space sampled pair distances"}
    scale = float(projected @ full / (projected @ projected)) if projected @ projected > 0 else None
    full_variation, projected_variation = np.std(full), np.std(projected)
    correlatable = full_variation > 1e-12 and projected_variation > 1e-12
    return dict(
        status="complete", kind="512 existing sampled within-task full-space distances",
        reference=str(path.relative_to(ROOT)), pair_count=len(full),
        distance_pearson=float(np.corrcoef(projected, full)[0, 1]) if correlatable else None,
        distance_spearman=float(spearmanr(projected, full).statistic) if correlatable else None,
        distance_energy_fraction=float(projected @ projected / denominator),
        unscaled_stress=float(np.linalg.norm(projected - full) / np.sqrt(denominator)),
        scale_fitted_stress=float(np.linalg.norm(scale * projected - full) / np.sqrt(denominator)) if scale is not None else None,
        fitted_distance_scale=scale,
        projection_exceeds_full_max=float(np.max(projected - full)),
    )


if __name__ == "__main__":
    from collections import Counter
    entries = inventory()
    path = ROOT / "outputs/geometric_shapes/inventory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, allow_nan=False))
    print(json.dumps({"entries": len(entries), "group_status": {group: dict(Counter(e["status"] for e in entries if e["group"] == group)) for group in ("original", "new", "overlay")}}, indent=2))
