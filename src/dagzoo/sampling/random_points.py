"""Random points mechanism."""

import torch

from dagzoo.core.execution_semantics import sample_root_source_plan
from dagzoo.core.fixed_layout_batched import (
    FixedLayoutBatchRng,
    _sample_random_points_batch,
    apply_function_plan_batch,
)
from dagzoo.core.layout_types import MechanismFamily
from dagzoo.rng import keyed_rng_from_generator
from dagzoo.sampling.noise import NoiseSamplingSpec


def sample_random_points(
    n_rows: int,
    dim: int,
    generator: torch.Generator,
    device: str,
    *,
    mechanism_logit_tilt: float = 0.0,
    function_family_mix: dict[MechanismFamily, float] | None = None,
    noise_sigma_multiplier: float = 1.0,
    noise_spec: NoiseSamplingSpec | None = None,
) -> torch.Tensor:
    """Sample one typed random-points source and execute it in torch."""
    if n_rows <= 0 or dim <= 0:
        raise ValueError(f"n_rows and dim must be > 0. Got n_rows={n_rows}, dim={dim}")

    root = keyed_rng_from_generator(generator, "sample_random_points")
    source = sample_root_source_plan(
        keyed_rng=root.keyed("plan"),
        out_dim=dim,
        mechanism_logit_tilt=mechanism_logit_tilt,
        function_family_mix=function_family_mix,
        device=str(generator.device),
    )
    rng = FixedLayoutBatchRng.from_keyed_rng(
        root.keyed("execution", "source"),
        batch_size=1,
        device=device,
    )
    base = _sample_random_points_batch(
        rng.keyed("base"),
        n_rows=n_rows,
        dim=dim,
        base_kind=source.base_kind,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
    out = apply_function_plan_batch(
        base,
        rng.keyed("function"),
        source.function,
        out_dim=dim,
        noise_sigma_multiplier=noise_sigma_multiplier,
        noise_spec=noise_spec,
    )
    return out.squeeze(0)
