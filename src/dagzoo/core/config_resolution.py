"""Shared config-resolution helpers for generate and benchmark command paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dagzoo.config import (
    MISSINGNESS_MECHANISM_MAR,
    MISSINGNESS_MECHANISM_MCAR,
    MISSINGNESS_MECHANISM_MNAR,
    MISSINGNESS_MECHANISM_NONE,
    REQUEST_PROFILE_SMOKE,
    GeneratorConfig,
    RequestFileConfig,
    clone_generator_config,
)
from dagzoo.hardware import HardwareInfo, detect_hardware
from dagzoo.hardware_policy import (
    apply_hardware_policy,
    resolve_cuda_fixed_layout_target_cells_limits,
)

_MISSING_VALUE = "<missing>"
_DEFAULT_CUDA_FIXED_LAYOUT_TARGET_SOURCE = "hardware.default_cuda_fixed_layout_target_cells"
_REQUEST_REPO_ROOT = Path(__file__).resolve().parents[3]
_REQUEST_BASE_CONFIG_PATH = _REQUEST_REPO_ROOT / "configs" / "default.yaml"
_REQUEST_MISSINGNESS_PRESET_PATHS = {
    MISSINGNESS_MECHANISM_MCAR: _REQUEST_REPO_ROOT / "configs" / "preset_missingness_mcar.yaml",
    MISSINGNESS_MECHANISM_MAR: _REQUEST_REPO_ROOT / "configs" / "preset_missingness_mar.yaml",
    MISSINGNESS_MECHANISM_MNAR: _REQUEST_REPO_ROOT / "configs" / "preset_missingness_mnar.yaml",
}
_REQUEST_MISSINGNESS_DATASET_PATHS = (
    "dataset.missing_rate",
    "dataset.missing_mechanism",
    "dataset.missing_mar_observed_fraction",
    "dataset.missing_mar_logit_scale",
    "dataset.missing_mnar_logit_scale",
)


@dataclass(slots=True, frozen=True)
class ResolutionEvent:
    """One field-level config override event."""

    path: str
    source: str
    old_value: Any
    new_value: Any


@dataclass(slots=True, frozen=True)
class BenchmarkSmokeCaps:
    """Hard caps applied to benchmark preset configs in smoke mode."""

    n_train: int
    n_test: int
    n_features: int
    n_nodes: int


@dataclass(slots=True, frozen=True)
class RequestSmokeProfile:
    """Hard caps applied to request configs for the public smoke profile."""

    n_train: int
    n_test: int
    n_features_min: int
    n_features_max: int
    n_nodes_min: int
    n_nodes_max: int


@dataclass(slots=True)
class ResolvedGenerateConfig:
    """Fully resolved generate config with hardware info and override trace."""

    config: GeneratorConfig
    hardware: HardwareInfo
    requested_device: str
    trace_events: list[ResolutionEvent]


@dataclass(slots=True)
class ResolvedRequestConfig:
    """Fully resolved request config with hardware info and override trace."""

    request: RequestFileConfig
    config: GeneratorConfig
    hardware: HardwareInfo
    requested_device: str
    trace_events: list[ResolutionEvent]


@dataclass(slots=True)
class ResolvedBenchmarkPresetConfig:
    """Fully resolved benchmark preset config with hardware info and override trace."""

    preset_key: str
    config: GeneratorConfig
    hardware: HardwareInfo
    requested_device: str
    trace_events: list[ResolutionEvent]


def _load_repo_config(path: Path) -> GeneratorConfig:
    """Load one repo-owned config file with standard validation."""

    return GeneratorConfig.from_yaml(path)


def _clone_config(config: GeneratorConfig) -> GeneratorConfig:
    """Clone nested config state so resolution does not mutate caller-owned objects."""

    return clone_generator_config(config, revalidate=True)


def _append_event(
    *,
    events: list[ResolutionEvent],
    path: str,
    source: str,
    old_value: Any,
    new_value: Any,
) -> None:
    """Append one trace event when values differ."""

    if old_value == new_value:
        return
    events.append(
        ResolutionEvent(
            path=path,
            source=source,
            old_value=old_value,
            new_value=new_value,
        )
    )


def _set_config_path(
    config: GeneratorConfig,
    *,
    path: str,
    value: Any,
    source: str,
    events: list[ResolutionEvent],
) -> None:
    """Set a dotted dataclass path and emit one trace event when changed."""

    parts = path.split(".")
    target: Any = config
    for part in parts[:-1]:
        target = getattr(target, part)
    field_name = parts[-1]
    old_value = getattr(target, field_name)
    if old_value == value:
        return
    setattr(target, field_name, value)
    _append_event(
        events=events,
        path=path,
        source=source,
        old_value=old_value,
        new_value=value,
    )


def _append_diff_events(
    before: Any,
    after: Any,
    *,
    source: str,
    events: list[ResolutionEvent],
    path: str = "",
) -> None:
    """Append trace events by diffing serialized config payloads."""

    if isinstance(before, dict) and isinstance(after, dict):
        keys = sorted(set(before) | set(after))
        for key in keys:
            child_path = key if not path else f"{path}.{key}"
            old_value = before.get(key, _MISSING_VALUE)
            new_value = after.get(key, _MISSING_VALUE)
            _append_diff_events(
                old_value,
                new_value,
                source=source,
                events=events,
                path=child_path,
            )
        return
    _append_event(
        events=events,
        path=path,
        source=source,
        old_value=before,
        new_value=after,
    )


def _normalize_requested_device(device: str | None, fallback: str | None) -> str:
    """Normalize device text into the same lower-cased runtime shape used by CLI paths."""

    if device is not None:
        candidate = str(device).strip().lower()
        return candidate or "auto"
    if fallback is None:
        return "auto"
    candidate = str(fallback).strip().lower()
    return candidate or "auto"


def _apply_missingness_overrides(
    config: GeneratorConfig,
    *,
    missing_rate: float | None,
    missing_mechanism: str | None,
    missing_mar_observed_fraction: float | None,
    missing_mar_logit_scale: float | None,
    missing_mnar_logit_scale: float | None,
    events: list[ResolutionEvent],
) -> None:
    """Apply generate missingness overrides."""

    overrides = (
        ("dataset.missing_rate", missing_rate),
        ("dataset.missing_mechanism", missing_mechanism),
        ("dataset.missing_mar_observed_fraction", missing_mar_observed_fraction),
        ("dataset.missing_mar_logit_scale", missing_mar_logit_scale),
        ("dataset.missing_mnar_logit_scale", missing_mnar_logit_scale),
    )
    has_override = any(value is not None for _, value in overrides)
    if not has_override:
        return

    for path, value in overrides:
        if value is None:
            continue
        _set_config_path(
            config,
            path=path,
            value=value,
            source="cli.missingness_override",
            events=events,
        )


def _apply_default_cuda_fixed_layout_target_floor(
    config: GeneratorConfig,
    *,
    hw: HardwareInfo,
    events: list[ResolutionEvent],
) -> None:
    """Fill the fixed-layout target from the default CUDA floor when the config leaves it unset."""

    target_floor, _ = resolve_cuda_fixed_layout_target_cells_limits(hw)
    if target_floor is None:
        return
    current_target = config.runtime.fixed_layout_target_cells
    if current_target is not None:
        return
    _set_config_path(
        config,
        path="runtime.fixed_layout_target_cells",
        value=int(target_floor),
        source=_DEFAULT_CUDA_FIXED_LAYOUT_TARGET_SOURCE,
        events=events,
    )


def _apply_rows_override(
    config: GeneratorConfig,
    *,
    rows: str | None,
    events: list[ResolutionEvent],
) -> None:
    """Apply generate rows override."""

    if rows is None:
        return
    _set_config_path(
        config,
        path="dataset.rows",
        value=rows,
        source="cli.rows",
        events=events,
    )


def _apply_smoke_caps(
    config: GeneratorConfig,
    *,
    smoke_caps: BenchmarkSmokeCaps,
    events: list[ResolutionEvent],
) -> None:
    """Apply benchmark smoke caps and trace each field-level cap event."""

    cap_specs = (
        ("dataset.n_train", int(smoke_caps.n_train)),
        ("dataset.n_test", int(smoke_caps.n_test)),
        ("dataset.n_features_min", int(smoke_caps.n_features)),
        ("dataset.n_features_max", int(smoke_caps.n_features)),
        ("graph.n_nodes_min", int(smoke_caps.n_nodes)),
        ("graph.n_nodes_max", int(smoke_caps.n_nodes)),
    )
    for path, cap_value in cap_specs:
        parts = path.split(".")
        target: Any = config
        for part in parts[:-1]:
            target = getattr(target, part)
        field_name = parts[-1]
        old_value = int(getattr(target, field_name))
        new_value = min(old_value, cap_value)
        if old_value == new_value:
            continue
        setattr(target, field_name, new_value)
        _append_event(
            events=events,
            path=path,
            source="benchmark.suite_smoke_caps",
            old_value=old_value,
            new_value=new_value,
        )


def _apply_request_smoke_profile(
    config: GeneratorConfig,
    *,
    smoke_profile: RequestSmokeProfile,
    events: list[ResolutionEvent],
) -> None:
    """Apply request smoke profile overrides."""

    smoke_specs = (
        ("dataset.n_train", int(smoke_profile.n_train)),
        ("dataset.n_test", int(smoke_profile.n_test)),
        ("dataset.n_features_min", int(smoke_profile.n_features_min)),
        ("dataset.n_features_max", int(smoke_profile.n_features_max)),
        ("graph.n_nodes_min", int(smoke_profile.n_nodes_min)),
        ("graph.n_nodes_max", int(smoke_profile.n_nodes_max)),
    )
    for path, value in smoke_specs:
        _set_config_path(
            config,
            path=path,
            value=value,
            source="request.profile_smoke",
            events=events,
        )


def _apply_request_missingness_profile(
    config: GeneratorConfig,
    *,
    missingness_profile: str,
    events: list[ResolutionEvent],
) -> None:
    """Apply missingness profile defaults without leaking preset-local fields."""

    if missingness_profile == MISSINGNESS_MECHANISM_NONE:
        overlay = _load_repo_config(_REQUEST_BASE_CONFIG_PATH)
    else:
        overlay_path = _REQUEST_MISSINGNESS_PRESET_PATHS[missingness_profile]
        overlay = _load_repo_config(overlay_path)

    for path in _REQUEST_MISSINGNESS_DATASET_PATHS:
        target = config.dataset
        source = overlay.dataset
        field_name = path.split(".")[-1]
        old_value = getattr(target, field_name)
        new_value = getattr(source, field_name)
        if old_value == new_value:
            continue
        setattr(target, field_name, new_value)
        _append_event(
            events=events,
            path=path,
            source="request.missingness_profile",
            old_value=old_value,
            new_value=new_value,
        )


def resolve_generate_config(
    config: GeneratorConfig,
    *,
    device_override: str | None,
    rows: str | None,
    hardware_policy: str,
    missing_rate: float | None,
    missing_mechanism: str | None,
    missing_mar_observed_fraction: float | None,
    missing_mar_logit_scale: float | None,
    missing_mnar_logit_scale: float | None,
    diagnostics_enabled: bool,
) -> ResolvedGenerateConfig:
    """Resolve effective config for one generate command invocation."""

    resolved = _clone_config(config)
    trace_events: list[ResolutionEvent] = []

    requested_device = _normalize_requested_device(device_override, resolved.runtime.device)
    _set_config_path(
        resolved,
        path="runtime.device",
        value=requested_device,
        source="cli.device",
        events=trace_events,
    )

    hw = detect_hardware(requested_device)
    before_policy = resolved.to_dict()
    resolved = apply_hardware_policy(
        resolved,
        hw,
        policy_name=hardware_policy,
        validate=False,
    )
    after_policy = resolved.to_dict()
    _append_diff_events(
        before_policy,
        after_policy,
        source=f"hardware_policy.{str(hardware_policy).strip().lower()}",
        events=trace_events,
    )
    _apply_default_cuda_fixed_layout_target_floor(
        resolved,
        hw=hw,
        events=trace_events,
    )

    _apply_rows_override(
        resolved,
        rows=rows,
        events=trace_events,
    )

    _apply_missingness_overrides(
        resolved,
        missing_rate=missing_rate,
        missing_mechanism=missing_mechanism,
        missing_mar_observed_fraction=missing_mar_observed_fraction,
        missing_mar_logit_scale=missing_mar_logit_scale,
        missing_mnar_logit_scale=missing_mnar_logit_scale,
        events=trace_events,
    )

    if diagnostics_enabled:
        _set_config_path(
            resolved,
            path="diagnostics.enabled",
            value=True,
            source="cli.diagnostics",
            events=trace_events,
        )

    resolved.validate_generation_constraints()
    return ResolvedGenerateConfig(
        config=resolved,
        hardware=hw,
        requested_device=requested_device,
        trace_events=trace_events,
    )


def resolve_request_config(
    *,
    request: RequestFileConfig,
    device_override: str | None,
    hardware_policy: str,
) -> ResolvedRequestConfig:
    """Resolve effective config for one request-file execution."""

    base_config = _load_repo_config(_REQUEST_BASE_CONFIG_PATH)
    trace_events: list[ResolutionEvent] = []

    if request.profile == REQUEST_PROFILE_SMOKE:
        _apply_request_smoke_profile(
            base_config,
            smoke_profile=RequestSmokeProfile(
                n_train=128,
                n_test=32,
                n_features_min=8,
                n_features_max=12,
                n_nodes_min=2,
                n_nodes_max=12,
            ),
            events=trace_events,
        )

    _set_config_path(
        base_config,
        path="dataset.task",
        value=request.task,
        source="request.task",
        events=trace_events,
    )
    if request.seed is not None:
        _set_config_path(
            base_config,
            path="seed",
            value=int(request.seed),
            source="request.seed",
            events=trace_events,
        )
    _set_config_path(
        base_config,
        path="dataset.rows",
        value=request.rows,
        source="request.rows",
        events=trace_events,
    )
    _apply_request_missingness_profile(
        base_config,
        missingness_profile=request.missingness_profile,
        events=trace_events,
    )
    _set_config_path(
        base_config,
        path="output.out_dir",
        value=str(Path(request.output_root) / "generated"),
        source="request.output_root",
        events=trace_events,
    )

    resolved = resolve_generate_config(
        base_config,
        device_override=device_override,
        rows=None,
        hardware_policy=hardware_policy,
        missing_rate=None,
        missing_mechanism=None,
        missing_mar_observed_fraction=None,
        missing_mar_logit_scale=None,
        missing_mnar_logit_scale=None,
        diagnostics_enabled=False,
    )
    return ResolvedRequestConfig(
        request=request,
        config=resolved.config,
        hardware=resolved.hardware,
        requested_device=resolved.requested_device,
        trace_events=trace_events + list(resolved.trace_events),
    )


def resolve_benchmark_preset_config(
    *,
    preset_key: str,
    config: GeneratorConfig,
    preset_device: str | None,
    suite: str,
    hardware_policy: str,
    smoke_caps: BenchmarkSmokeCaps | None,
) -> ResolvedBenchmarkPresetConfig:
    """Resolve effective config for one benchmark preset run."""

    resolved = _clone_config(config)
    trace_events: list[ResolutionEvent] = []

    requested_device = _normalize_requested_device(preset_device, resolved.runtime.device)
    _set_config_path(
        resolved,
        path="runtime.device",
        value=requested_device,
        source="benchmark.preset_device",
        events=trace_events,
    )

    hw = detect_hardware(requested_device)
    before_policy = resolved.to_dict()
    resolved = apply_hardware_policy(resolved, hw, policy_name=hardware_policy)
    after_policy = resolved.to_dict()
    _append_diff_events(
        before_policy,
        after_policy,
        source=f"hardware_policy.{str(hardware_policy).strip().lower()}",
        events=trace_events,
    )
    _apply_default_cuda_fixed_layout_target_floor(
        resolved,
        hw=hw,
        events=trace_events,
    )

    normalized_suite = str(suite).strip().lower()
    if normalized_suite == "smoke":
        if smoke_caps is None:
            raise ValueError("Benchmark smoke suite config resolution requires smoke cap values.")
        _apply_smoke_caps(resolved, smoke_caps=smoke_caps, events=trace_events)

    resolved.validate_generation_constraints()
    return ResolvedBenchmarkPresetConfig(
        preset_key=preset_key,
        config=resolved,
        hardware=hw,
        requested_device=requested_device,
        trace_events=trace_events,
    )


def serialize_resolution_events(events: list[ResolutionEvent]) -> list[dict[str, Any]]:
    """Convert trace dataclasses into JSON/YAML-safe dictionaries."""

    return [asdict(event) for event in events]


def append_config_diff_events(
    before: GeneratorConfig,
    after: GeneratorConfig,
    *,
    source: str,
    events: list[ResolutionEvent],
) -> None:
    """Append trace events describing config differences between two states."""

    _append_diff_events(
        before.to_dict(),
        after.to_dict(),
        source=source,
        events=events,
    )


__all__ = [
    "append_config_diff_events",
    "BenchmarkSmokeCaps",
    "RequestSmokeProfile",
    "ResolutionEvent",
    "ResolvedBenchmarkPresetConfig",
    "ResolvedGenerateConfig",
    "ResolvedRequestConfig",
    "resolve_benchmark_preset_config",
    "resolve_generate_config",
    "resolve_request_config",
    "serialize_resolution_events",
]
