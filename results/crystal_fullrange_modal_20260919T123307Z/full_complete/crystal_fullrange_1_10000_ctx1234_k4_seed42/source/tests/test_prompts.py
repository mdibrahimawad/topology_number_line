from numzig.prompts import build_ordered_prompt_records


def test_prompt_count_and_determinism():
    a = build_ordered_prompt_records(samples_per_group=3, num_examples=3, upper_bound=10000, groups=(1,2,3,4), context="random", seed=42)
    b = build_ordered_prompt_records(samples_per_group=3, num_examples=3, upper_bound=10000, groups=(1,2,3,4), context="random", seed=42)
    assert a == b
    assert len(a) == 12
    assert [x["point_id"] for x in a] == list(range(12))
    assert all(x["prompt"].endswith("=") for x in a)
