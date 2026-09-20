from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .config import ModelSpec
from .prompts import build_ordered_prompt_records


def _torch_dtype(name: str):
    import torch
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def _tokenize_one(tokenizer, prompt: str, prepend_bos: bool):
    import torch
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=not prepend_bos)
    if prepend_bos:
        if tokenizer.bos_token_id is None:
            raise ValueError("prepend_bos=True but tokenizer has no BOS token")
        bos = torch.full((1, 1), int(tokenizer.bos_token_id), dtype=inputs["input_ids"].dtype)
        inputs["input_ids"] = torch.cat((bos, inputs["input_ids"]), dim=1)
        if "attention_mask" in inputs:
            one = torch.ones((1, 1), dtype=inputs["attention_mask"].dtype)
            inputs["attention_mask"] = torch.cat((one, inputs["attention_mask"]), dim=1)
    inputs.pop("token_type_ids", None)
    return inputs


def extract_hidden_states(
    spec: ModelSpec,
    *,
    output_dir: str | Path,
    seed: int,
    samples_per_group: int = 30,
    num_examples: int = 3,
    groups: tuple[int, ...] = (1, 2, 3, 4),
    context: str = "random",
    upper_bound: int = 10000,
    prepend_bos: bool = False,
    hf_token: str | None = None,
) -> dict:
    """Run the user's numerical prompts and save last-token hidden states at every layer.

    Scientific lock: one prompt at a time, output_hidden_states=True, use_cache=False,
    and hidden_states[layer][0, -1, :] converted to float32, matching the uploaded code.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for extraction")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_ordered_prompt_records(
        samples_per_group=samples_per_group,
        num_examples=num_examples,
        upper_bound=upper_bound,
        groups=groups,
        context=context,
        seed=seed,
    )

    hf_kwargs: dict = {}
    if hf_token:
        hf_kwargs["token"] = hf_token
    if spec.revision:
        hf_kwargs["revision"] = spec.revision
    if spec.trust_remote_code:
        hf_kwargs["trust_remote_code"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id, use_fast=spec.tokenizer_use_fast, **hf_kwargs
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        spec.model_id,
        torch_dtype=_torch_dtype(spec.dtype),
        **hf_kwargs,
    ).to("cuda")
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    n_layers = int(model.config.num_hidden_layers) + 1
    n_points = len(records)
    layer_rows: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    diag_samples: list[dict] = []
    equals_ok = 0

    with torch.inference_mode():
        for idx, rec in enumerate(records):
            inputs = _tokenize_one(tokenizer, rec["prompt"], prepend_bos)
            ids = inputs["input_ids"][0].tolist()
            last_text = tokenizer.decode(
                [ids[-1]],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            ends_equals = bool(rec["prompt"].endswith("=") and last_text == "=")
            equals_ok += int(ends_equals)
            if len(diag_samples) < 12:
                diag_samples.append(
                    {
                        "point_id": rec["point_id"],
                        "prompt": rec["prompt"],
                        "target": rec["target"],
                        "group": rec["group"],
                        "last_token_id": int(ids[-1]),
                        "last_token_text": last_text,
                        "ends_with_equals_token": ends_equals,
                    }
                )
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            hs = outputs.hidden_states
            if len(hs) != n_layers:
                raise RuntimeError(f"expected {n_layers} hidden states, got {len(hs)}")
            for layer in range(n_layers):
                vec = hs[layer][0, -1, :].detach().to(torch.float32).cpu().numpy()
                layer_rows[layer].append(vec)
            if (idx + 1) % 20 == 0 or idx + 1 == n_points:
                print(f"[{spec.alias}] extracted {idx + 1}/{n_points}", flush=True)

    hidden = np.stack([np.stack(rows, axis=0) for rows in layer_rows], axis=0)
    targets = np.asarray([r["target"] for r in records], dtype=np.float64)
    group_ids = np.asarray([r["group"] for r in records], dtype=np.int16)
    point_ids = np.asarray([r["point_id"] for r in records], dtype=np.int32)

    # Float32 is intentional: the user's original collector converts to float32 before PCA.
    np.savez_compressed(
        output_dir / "hidden_states.npz",
        hidden_states=hidden.astype(np.float32, copy=False),
        targets=targets,
        group_ids=group_ids,
        point_ids=point_ids,
    )
    (output_dir / "prompts.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    diagnostics = {
        "model": spec.to_dict(),
        "seed": seed,
        "n_points": n_points,
        "n_layers_including_embedding": n_layers,
        "hidden_dim": int(hidden.shape[2]),
        "shape": list(hidden.shape),
        "prompt_method": {
            "groups": list(groups),
            "samples_per_group": samples_per_group,
            "num_examples": num_examples,
            "context": context,
            "upper_bound": upper_bound,
            "prepend_bos": prepend_bos,
        },
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "prompts_ending_with_equals_as_single_last_token": equals_ok,
        "diagnostic_samples": diag_samples,
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")),
    }
    (output_dir / "extraction_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    del model
    torch.cuda.empty_cache()
    return diagnostics
