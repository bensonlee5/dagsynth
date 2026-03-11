"""Categorical converter implementations."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from dagzoo.core.execution_semantics import sample_converter_plan
from dagzoo.core.fixed_layout.batched import FixedLayoutBatchRng, _apply_categorical_group_batch
from dagzoo.core.fixed_layout.plan_types import CategoricalConverterPlan
from dagzoo.core.layout_types import MechanismFamily
from dagzoo.rng import keyed_rng_from_generator


def apply_categorical_converter(
    x: torch.Tensor,
    generator: torch.Generator,
    *,
    n_categories: int,
    method: str | None = None,
    function_family_mix: dict[MechanismFamily, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Categorical converter in torch via the shared typed-plan semantics."""
    y = x.to(torch.float32)
    if y.dim() == 1:
        y = y.unsqueeze(1)

    c = max(2, int(n_categories))
    root = keyed_rng_from_generator(generator, "apply_categorical_converter")
    plan = sample_converter_plan(
        SimpleNamespace(key="value", kind="cat", dim=int(y.shape[1]), cardinality=c),
        keyed_rng=root.keyed("plan"),
        mechanism_logit_tilt=0.0,
        function_family_mix=function_family_mix,
        method_override=method,
        device=str(generator.device),
    )
    if not isinstance(plan, CategoricalConverterPlan):
        raise RuntimeError("Expected categorical converter plan for categorical converter.")

    rng = FixedLayoutBatchRng.from_keyed_rng(
        root.keyed("execution"),
        batch_size=1,
        device=str(y.device),
    )
    out, labels = _apply_categorical_group_batch(
        y.unsqueeze(0).unsqueeze(2),
        rng,
        plan,
        n_categories=c,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    return out[0, :, 0, :], labels[0, :, 0]
