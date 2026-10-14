"""Cut Crystal hidden vectors (all 10,000 numbers, selected layers) into GitHub-sized files.
Source: Modal volume numberline-zigzag-results:/crystal_fullrange_1_10000_ctx1234_k4_seed42/hidden,
downloaded to ~/Documents/crystal_dl. Vectors are centred per layer (subtracting the mean does not
change any distance), then stored as float16; the script checks that distances survive this."""
import glob, json, os
import numpy as np

SRC = os.path.expanduser("~/Documents/crystal_dl")
OUT = os.path.dirname(os.path.abspath(__file__))
LAYERS = [8, 16, 24, 27, 32]

files = sorted(f for f in glob.glob(SRC + "/**/*.npy", recursive=True) if not f.endswith("_ids.npy"))
assert len(files) == 313, f"expected 313 vector files, found {len(files)} under {SRC}"
ids, parts = [], {L: [] for L in LAYERS}
for f in files:
    x = np.load(f)                    # [level, point, feature]
    assert x.shape[0] == 33 and x.shape[2] == 4096, f"unexpected shape {x.shape} in {f}"
    ids.append(np.load(f[:-4] + "_ids.npy"))
    for L in LAYERS:
        parts[L].append(np.asarray(x[L], dtype=np.float32))
ids = np.concatenate(ids); order = np.argsort(ids); target = ids[order] + 1
assert (target == np.arange(1, 10001)).all(), "expected every number 1..10000 exactly once"

rng = np.random.default_rng(0); a, b = rng.integers(0, 10000, (2, 2000))
report = {}
for L in LAYERS:
    X = np.concatenate(parts[L])[order]
    mean = X.mean(0); Xc = X - mean; X16 = Xc.astype(np.float16)
    assert np.isfinite(X16).all(), f"layer {L}: values too large for float16"
    d32 = np.linalg.norm(Xc[a] - Xc[b], axis=1)
    d16 = np.linalg.norm(X16[a].astype(np.float32) - X16[b].astype(np.float32), axis=1)
    ok = d32 > 0
    err = float(np.max(np.abs(d16[ok] - d32[ok]) / d32[ok]))
    path = os.path.join(OUT, f"layer_{L:02d}.npz")
    np.savez_compressed(path, X=X16, mean=mean, target=target)
    mb = os.path.getsize(path) / 1e6
    report[L] = dict(shape=list(X16.shape), max_rel_distance_error=err, file_mb=round(mb, 1))
    print(f"layer {L:2d}: {X16.shape}, max distance error {err:.2e}, file {mb:.0f} MB"
          + ("   <-- TOO BIG for GitHub" if mb > 95 else ""))
json.dump(dict(source="modal numberline-zigzag-results:/crystal_fullrange_1_10000_ctx1234_k4_seed42/hidden",
               layers=LAYERS, point_order="target 1..10000", centred=True, dtype="float16",
               checks=report), open(os.path.join(OUT, "meta.json"), "w"), indent=2)
print("done")
