"""Tests for converters/numeric.py numeric path."""

import pytest
import torch
from conftest import make_generator as _make_generator

import dagzoo.converters.numeric as numeric_mod
from dagzoo.converters.numeric import apply_numeric_converter
from dagzoo.core.fixed_layout.plan_types import NumericConverterPlan
from dagzoo.rng import KeyedRng


def test_output_shapes() -> None:
    g = _make_generator(0)
    x = torch.randn(64, 3, generator=g)
    x_prime, v = apply_numeric_converter(x, g)
    assert x_prime.shape[0] == 64
    assert v.shape == (64,)


def test_deterministic() -> None:
    x = torch.randn(32, 2, generator=_make_generator(99))
    xp1, v1 = apply_numeric_converter(x.clone(), _make_generator(0))
    xp2, v2 = apply_numeric_converter(x.clone(), _make_generator(0))
    torch.testing.assert_close(xp1, xp2)
    torch.testing.assert_close(v1, v2)


def test_finite_outputs() -> None:
    g = _make_generator(7)
    x = torch.randn(64, 4, generator=g)
    x_prime, v = apply_numeric_converter(x, g)
    assert torch.all(torch.isfinite(x_prime))
    assert torch.all(torch.isfinite(v))


def test_1d_input() -> None:
    g = _make_generator(1)
    x = torch.randn(64, generator=g)
    x_prime, v = apply_numeric_converter(x, g)
    assert x_prime.dim() == 2
    assert x_prime.shape[0] == 64


def test_multi_column_inputs_share_one_warp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.linspace(-1.0, 1.0, steps=32).unsqueeze(1).repeat(1, 3)
    plan = NumericConverterPlan(kind="num", warp_enabled=True)
    monkeypatch.setattr(numeric_mod, "sample_converter_plan", lambda *_args, **_kwargs: plan)

    x_prime, v = apply_numeric_converter(x, _make_generator(2))

    torch.testing.assert_close(x_prime[:, 0], x_prime[:, 1])
    torch.testing.assert_close(x_prime[:, 1], x_prime[:, 2])
    torch.testing.assert_close(v, x[:, 0])


def test_apply_numeric_converter_passes_generator_device_to_plan_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_devices: list[str] = []

    class FakeGenerator:
        device = "cuda"

    monkeypatch.setattr(numeric_mod, "keyed_rng_from_generator", lambda *_args: KeyedRng(7))

    def _stub_sample_converter_plan(*_args, **kwargs) -> NumericConverterPlan:
        seen_devices.append(str(kwargs["device"]))
        return NumericConverterPlan(kind="num", warp_enabled=False)

    monkeypatch.setattr(numeric_mod, "sample_converter_plan", _stub_sample_converter_plan)

    out, values = apply_numeric_converter(
        torch.randn(8, 3),
        FakeGenerator(),  # type: ignore[arg-type]
    )

    assert seen_devices == ["cuda"]
    assert out.shape == (8, 3)
    assert values.shape == (8,)
