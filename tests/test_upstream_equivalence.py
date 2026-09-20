import importlib.util
import random
from pathlib import Path

import numpy as np

from numzig.prompts import generate_numeral_prompts
from numzig.geometry import fit_spacing_direct
from numzig.zigzag import _contiguous_ranges

ROOT = Path(__file__).resolve().parents[1]

def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

def test_prompt_sampling_matches_uploaded_source():
    upstream = _load("prompts_original", ROOT / "upstream_reference" / "prompts_original.py")
    a = generate_numeral_prompts(k=30, num_examples=3, upper_bound=10000, groups=(1,2,3,4), context="random", rng=random.Random(42))
    b = upstream.generate_numeral_prompts(k=30, num_examples=3, upper_bound=10000, groups=(1,2,3,4), interval_fn=upstream.default_interval, context="random", rng=random.Random(42))
    assert a == b

def test_beta_fit_matches_uploaded_source():
    upstream = _load("spacing_fit_original", ROOT / "upstream_reference" / "spacing_fit_original.py")
    gaps = np.array([3.2, 1.7, 0.91])
    assert np.allclose(fit_spacing_direct(gaps), upstream.fit_spacing_direct(gaps), equal_nan=True)

def test_zigzag_time_ranges_match_uploaded_source_convention():
    # Uploaded fclaux.ranges maps contiguous [0,1,2] to (1,4) and [5,6] to (6,8).
    assert _contiguous_ranges(np.array([0,1,2,5,6])) == [(1,4),(6,8)]
