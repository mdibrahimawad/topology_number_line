"""Shared PCA coordinates for paired task conditions, without whitening."""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.decomposition import PCA

from numzig.fullrange.storage import atomic, digest, fingerprint, read_json, write_json, write_npz


def compare_groups(xs, rows, outdir, model_key, context_id, commit=lambda: None):
    """Compare conditions at each saved level; caller supplies a model/context directory.

    xs maps task names to (levels, targets, hidden dimensions). Rows must contain
    the same ordered targets within each family. Returns one receipt per level.
    """
    outdir = Path(outdir)
    code_hash = digest(__file__)
    summaries = []
    families = {'permutations': ('copy4', 'reverse4', 'swap_first4', 'swap_last4'),
                'notation': ('numeric_copy', 'word_copy')}
    palette = ListedColormap(['#808080', *plt.get_cmap('tab10').colors[:9]])
    for family, available in families.items():
        tasks = [task for task in available if task in xs]
        if len(tasks) < 2:
            continue
        levels, count, dimensions = xs[tasks[0]].shape
        targets = [row['target'] for row in rows[tasks[0]]]
        if count < 2 or len(targets) != count or len(set(targets)) != count:
            raise ValueError('Shared PCA requires at least two unique matched targets')
        for task in tasks:
            if xs[task].shape != (levels, count, dimensions):
                raise ValueError('Conditions must have matching activation shapes')
            if [row['target'] for row in rows[task]] != targets:
                raise ValueError('Conditions must have identical ordered targets')
            if any(row['task'] != task or row['context_id'] != context_id for row in rows[task]):
                raise ValueError('Rows do not match the requested task/context')
        input_labels = np.array([row['leading_digit'] for row in rows[tasks[0]]])
        output_labels = np.array([[row['output_digits'][0] for row in rows[task]] for task in tasks])
        metadata_hash = fingerprint({task: rows[task] for task in tasks})
        rng = np.random.default_rng(42)
        pair_a = rng.integers(0, count, 512)
        pair_b = (pair_a + rng.integers(1, count, 512)) % count
        for layer in range(levels):
            base = outdir / family / f'layer_{layer:02d}'
            receipt_path = base.with_suffix('.json')
            arrays = [np.asarray(xs[task][layer], dtype=np.float64) for task in tasks]
            source = hashlib.sha256(metadata_hash.encode())
            for array in arrays:
                if not np.isfinite(array).all():
                    raise ValueError('Non-finite hidden states in shared PCA')
                source.update(np.ascontiguousarray(array).tobytes())
            dependency = fingerprint(dict(input_sha256=source.hexdigest(), code_sha256=code_hash,
                                          model=model_key, context=context_id, layer=layer))
            if receipt_path.exists():
                previous = read_json(receipt_path)
                if previous.get('dependency') == dependency and all(
                    (outdir / name).is_file() and digest(outdir / name) == sha
                    for name, sha in previous.get('files', {}).items()
                ) and previous.get('files'):
                    summaries.append(previous)
                    continue
            joint = np.concatenate(arrays, axis=0)
            mean = joint.mean(axis=0)
            centered = joint - mean
            total_variance = float(np.einsum('ij,ij->', centered, centered) / (len(joint) - 1))
            scores = np.zeros((len(joint), 3), dtype=np.float64)
            components = np.zeros((3, dimensions), dtype=np.float64)
            evr = np.zeros(3)
            if total_variance > 0:
                components_count = min(3, min(joint.shape) - 1)
                if components_count:
                    pca = PCA(n_components=components_count, svd_solver='arpack', random_state=42,
                              whiten=False, tol=1e-8)
                    scores[:, :components_count] = pca.fit_transform(joint)
                    components[:components_count] = pca.components_
                    evr[:components_count] = pca.explained_variance_ratio_
                    mean = pca.mean_
            if not np.isfinite(scores).all() or not np.isfinite(evr).all():
                raise ValueError('Non-finite shared PCA result')
            scores = scores.reshape(len(tasks), count, 3)
            pair_distances = [np.linalg.norm(array[pair_a] - array[pair_b], axis=1) for array in arrays]
            reference_scale = float(np.median(pair_distances[0]))
            metrics = {}
            for task, array, distances in zip(tasks, arrays, pair_distances):
                displacement = float(np.median(np.linalg.norm(array - arrays[0], axis=1)))
                correlation = (float(np.corrcoef(pair_distances[0], distances)[0, 1])
                               if np.std(pair_distances[0]) > 0 and np.std(distances) > 0 else None)
                metrics[task] = dict(reference_task=tasks[0], median_paired_displacement=displacement,
                    displacement_over_reference_median_distance=displacement / reference_scale if reference_scale > 0 else None,
                    sampled_pair_distance_pearson=correlation)
            archive = base.with_suffix('.npz')
            write_npz(archive, scores=scores, task_names=np.array(tasks), targets=np.array(targets),
                      mean=mean, components=components, explained_variance_ratio=evr,
                      input_leading_digit=input_labels, output_leading_digit=output_labels,
                      pair_a=pair_a, pair_b=pair_b, fullspace_pair_distances=np.array(pair_distances))
            files = [archive]
            limits = []
            for axis in range(2):
                lo, hi = float(scores[:, :, axis].min()), float(scores[:, :, axis].max())
                margin = max((hi - lo) * .06, 1e-9) if hi > lo else 1.
                limits.append((lo - margin, hi + margin))
            colorings = ('input', 'output') if family == 'permutations' else ('input',)
            for coloring in colorings:
                figure, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4), squeeze=False,
                                            sharex=True, sharey=True, constrained_layout=True)
                for index, (task, axis) in enumerate(zip(tasks, axes[0])):
                    labels = input_labels if coloring == 'input' else output_labels[index]
                    scatter = axis.scatter(scores[index, :, 0], scores[index, :, 1], c=labels,
                                           cmap=palette, vmin=-.5, vmax=9.5, s=5, alpha=.65, linewidths=0)
                    axis.set(title=task, xlabel=f'Joint PC1 ({evr[0]:.1%})', xlim=limits[0], ylim=limits[1])
                axes[0, 0].set_ylabel(f'Joint PC2 ({evr[1]:.1%})')
                figure.colorbar(scatter, ax=list(axes[0]), ticks=range(10), shrink=.8,
                                label=f'{coloring.capitalize()} leading numerical digit')
                figure.suptitle(f'{model_key} · level {layer} · context {context_id} · shared PCA axes')
                png = base.parent / f'{base.name}_{coloring}.png'
                try:
                    atomic(png, lambda f: figure.savefig(f, format='png', dpi=130))
                finally:
                    plt.close(figure)
                files.append(png)
            receipt = dict(dependency=dependency, input_sha256=source.hexdigest(), code_sha256=code_hash,
                model=model_key, context_id=context_id, family=family, layer=layer, tasks=tasks,
                count_per_condition=count, shared_pca_variance_ratio=evr.tolist(),
                joint_total_variance=total_variance, metrics=metrics,
                distance_sampling='512 deterministic ordered distinct-index pairs, seed 42',
                files={str(path.relative_to(outdir)): digest(path) for path in files})
            commit()
            write_json(receipt_path, receipt)
            commit()
            summaries.append(receipt)
    return summaries
