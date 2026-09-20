from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ModelSpec:
    alias: str
    model_id: str
    trust_remote_code: bool
    tokenizer_use_fast: bool = True
    dtype: str = "float16"
    gpu: str = "L40S"
    revision: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "ModelSpec":
        return cls(**value)


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        alias="falcon-rw-1b",
        model_id="tiiuae/falcon-rw-1b",
        trust_remote_code=True,
    ),

    ModelSpec(
        alias="falcon-rw-7b",
        model_id="tiiuae/falcon-rw-7b",
        trust_remote_code=True,
    ),

    ModelSpec(
        alias="btlm-3b-8k",
        model_id="cerebras/btlm-3b-8k-base",
        trust_remote_code=True,
        tokenizer_use_fast=False,
        revision="eff52176f1aac5b352045d04b83b6957a9ef1b47",
    ),

    ModelSpec(
        alias="llm360-crystal",
        model_id="LLM360/Crystal",
        trust_remote_code=True,
        dtype="bfloat16",
    ),
)


_BY_ALIAS = {
    model.alias: model
    for model in MODELS
}


def resolve_models(selection: str) -> list[ModelSpec]:
    parts = [
        item.strip().lower()
        for item in selection.split(",")
        if item.strip()
    ]

    if not parts or "all" in parts:
        return list(MODELS)

    unknown = [
        item
        for item in parts
        if item not in _BY_ALIAS
    ]

    if unknown:
        raise ValueError(
            f"Unknown model alias(es): {unknown}. "
            f"Allowed: {sorted(_BY_ALIAS)}"
        )

    return [
        _BY_ALIAS[item]
        for item in parts
    ]