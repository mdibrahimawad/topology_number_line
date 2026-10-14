"""Cut one model's saved hidden vectors (all numbers 1..10000, chosen levels) into GitHub-sized
files: centred per level (distances unchanged), float16, with a distance check. Also saves the
model's tokenization of each prompt (from dataset.json) as tokenization.json.gz."""
import argparse, glob, gzip, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--layers", required=True); ap.add_argument("--levels", type=int, required=True)
a = ap.parse_args()
LAYERS = [int(x) for x in a.layers.split(",")]
files = sorted(f for f in glob.glob(a.src + "/**/*.npy", recursive=True) if not f.endswith("_ids.npy"))
assert len(files) == 313, f"expected 313 vector files, found {len(files)}"
ids, parts, width = [], {L: [] for L in LAYERS}, None
for f in files:
    x = np.load(f)
    assert x.shape[0] == a.levels, f"{f}: {x.shape[0]} levels, expected {a.levels}"
    width = x.shape[2]
    ids.append(np.load(f[:-4] + "_ids.npy"))
    for L in LAYERS:
        parts[L].append(x[L].astype(np.float32))
ids = np.concatenate(ids); order = np.argsort(ids); target = ids[order] + 1
assert (target == np.arange(1, 10001)).all(), "expected every number 1..10000 exactly once"
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(0); i, j = rng.integers(0, 10000, (2, 2000)); report = {}
for L in LAYERS:
    X = np.concatenate(parts[L])[order]; Xc = X - X.mean(0); X16 = Xc.astype(np.float16)
    assert np.isfinite(X16).all(), f"level {L}: values too large for float16"
    d32 = np.linalg.norm(Xc[i] - Xc[j], axis=1)
    d16 = np.linalg.norm(X16[i].astype(np.float32) - X16[j].astype(np.float32), axis=1)
    ok = d32 > 0; err = float(np.max(np.abs(d16[ok] - d32[ok]) / d32[ok]))
    path = os.path.join(a.out, f"layer_{L:02d}.npz")
    np.savez_compressed(path, X=X16, mean=X.mean(0), target=target)
    mb = os.path.getsize(path) / 1e6
    report[L] = dict(shape=list(X16.shape), max_rel_distance_error=err, file_mb=round(mb, 1))
    print(f"level {L:2d}: {X16.shape}, max distance error {err:.2e}, file {mb:.0f} MB" + ("   <-- TOO BIG" if mb > 95 else ""))
ds = glob.glob(a.src + "/**/dataset.json", recursive=True)
assert len(ds) == 1, f"expected one dataset.json, found {ds}"
with gzip.open(os.path.join(a.out, "tokenization.json.gz"), "wt") as g:
    g.write(open(ds[0]).read())
json.dump(dict(levels=a.levels, width=width, layers=LAYERS, point_order="target 1..10000", centred=True,
               dtype="float16", checks=report), open(os.path.join(a.out, "meta.json"), "w"), indent=2)
print("done")
