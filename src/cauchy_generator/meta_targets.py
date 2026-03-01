"""Shared helpers for meta-feature target specs and diagnostics bands."""

from __future__ import annotations

from dataclasses import fields
import math

from cauchy_generator.diagnostics.types import DatasetMetrics

TargetBand = tuple[float, float]
SUPPORTED_METRICS: frozenset[str] = frozenset(
    field_info.name for field_info in fields(DatasetMetrics) if field_info.name != "task"
)


def coerce_target_bands(raw: object) -> dict[str, TargetBand]:
    """Normalize target band mappings into finite `(lo, hi)` tuples."""

    normalized: dict[str, TargetBand] = {}
    if not isinstance(raw, dict):
        return normalized
    for metric_name, value in raw.items():
        if not isinstance(metric_name, str):
            continue
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        try:
            lo = float(value[0])
            hi = float(value[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(lo) and math.isfinite(hi)):
            continue
        normalized[metric_name] = (lo, hi) if lo <= hi else (hi, lo)
    return normalized


def merge_target_bands(*raw_values: object) -> dict[str, TargetBand]:
    """Merge target bands in order, where later entries win."""

    merged: dict[str, TargetBand] = {}
    for raw in raw_values:
        merged.update(coerce_target_bands(raw))
    return merged


def coerce_quantiles(raw: object) -> tuple[float, ...]:
    """Normalize quantile payload into finite float values."""

    if not isinstance(raw, (list, tuple)):
        return ()
    normalized: list[float] = []
    for item in raw:
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            normalized.append(value)
    return tuple(normalized)
