"""Tests for functions/multi.py."""

import pytest
import torch
from conftest import make_generator as _make_generator
from conftest import make_keyed_rng as _make_keyed_rng

from dagzoo.core.fixed_layout.batched import FixedLayoutBatchRng, apply_function_plan_batch
from dagzoo.core.fixed_layout.plan_types import (
    ConcatNodeSource,
    GaussianMatrixPlan,
    LinearFunctionPlan,
    StackedNodeSource,
)
from dagzoo.core.layout_types import AggregationKind
from dagzoo.functions.multi import apply_multi_function
from dagzoo.functions.random_functions import apply_random_function
from dagzoo.math import sanitize_and_standardize


def test_single_input() -> None:
    g = _make_generator(0)
    x = torch.randn(64, 4, generator=g)
    y = apply_multi_function([x], g, out_dim=5)
    assert y.shape == (64, 5)


def test_single_input_matches_apply_random_function() -> None:
    x = torch.randn(64, 4, generator=_make_generator(3))
    actual_generator = _make_generator(4)
    reference_generator = _make_generator(4)

    actual = apply_multi_function([x.clone()], actual_generator, out_dim=5)
    expected = apply_random_function(x.clone(), reference_generator, out_dim=5)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_multiple_inputs() -> None:
    g = _make_generator(1)
    a = torch.randn(64, 3, generator=g)
    b = torch.randn(64, 2, generator=g)
    y = apply_multi_function([a, b], g, out_dim=4)
    assert y.shape == (64, 4)


@pytest.mark.parametrize("aggregation_kind", ["sum", "product", "max", "logsumexp"])
def test_multiple_inputs_support_explicit_aggregation_kind(
    aggregation_kind: AggregationKind,
) -> None:
    g = _make_generator(2)
    a = torch.randn(64, 3, generator=g)
    b = torch.randn(64, 2, generator=g)
    y = apply_multi_function([a, b], g, out_dim=4, aggregation_kind=aggregation_kind)
    assert y.shape == (64, 4)
    assert torch.all(torch.isfinite(y))


@pytest.mark.parametrize("aggregation_kind", ["sum", "product", "max", "logsumexp"])
def test_multiple_inputs_are_deterministic_for_explicit_aggregation_kind(
    aggregation_kind: AggregationKind,
) -> None:
    a = torch.randn(64, 3, generator=_make_generator(11))
    b = torch.randn(64, 2, generator=_make_generator(12))
    y1 = apply_multi_function(
        [a.clone(), b.clone()],
        _make_generator(13),
        out_dim=4,
        aggregation_kind=aggregation_kind,
    )
    y2 = apply_multi_function(
        [a.clone(), b.clone()],
        _make_generator(13),
        out_dim=4,
        aggregation_kind=aggregation_kind,
    )
    torch.testing.assert_close(y1, y2)


def test_multiple_inputs_concat_sanitizes_non_finite_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ConcatNodeSource(function=LinearFunctionPlan(matrix=GaussianMatrixPlan()))
    monkeypatch.setattr(
        "dagzoo.functions.multi.sample_multi_source_plan", lambda *_args, **_kwargs: source
    )
    inputs = [
        torch.tensor(
            [[0.0, float("nan")], [1.0, 2.0], [float("inf"), -1.0]],
            dtype=torch.float32,
        ),
        torch.tensor([[1.0], [2.0], [-float("inf")]], dtype=torch.float32),
    ]

    actual = apply_multi_function(inputs, _make_generator(18), out_dim=3)

    assert actual.shape == (3, 3)
    assert torch.all(torch.isfinite(actual))


def test_multiple_inputs_stacked_sanitizes_non_finite_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = StackedNodeSource(
        aggregation_kind="sum",
        parent_functions=(
            LinearFunctionPlan(matrix=GaussianMatrixPlan()),
            LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    monkeypatch.setattr(
        "dagzoo.functions.multi.sample_multi_source_plan", lambda *_args, **_kwargs: source
    )
    inputs = [
        torch.tensor(
            [[0.0, float("nan")], [1.0, 2.0], [float("inf"), -1.0]],
            dtype=torch.float32,
        ),
        torch.tensor([[1.0], [2.0], [-float("inf")]], dtype=torch.float32),
    ]

    actual = apply_multi_function(inputs, _make_generator(19), out_dim=3)

    assert actual.shape == (3, 3)
    assert torch.all(torch.isfinite(actual))


def test_multi_function_matches_explicit_stacked_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = [
        torch.randn(64, 3, generator=_make_generator(21)),
        torch.randn(64, 2, generator=_make_generator(22)),
        torch.randn(64, 4, generator=_make_generator(23)),
    ]
    source = StackedNodeSource(
        aggregation_kind="logsumexp",
        parent_functions=(
            LinearFunctionPlan(matrix=GaussianMatrixPlan()),
            LinearFunctionPlan(matrix=GaussianMatrixPlan()),
            LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    monkeypatch.setattr(
        "dagzoo.functions.multi.sample_multi_source_plan", lambda *_args, **_kwargs: source
    )

    actual_generator = _make_generator(24)
    reference_generator = _make_generator(24)

    actual = apply_multi_function(
        [inp.clone() for inp in inputs],
        actual_generator,
        out_dim=5,
    )
    root = _make_keyed_rng(reference_generator, "apply_multi_function")
    rng = FixedLayoutBatchRng.from_keyed_rng(root.keyed("execution"), batch_size=1, device="cpu")
    transformed = [
        apply_function_plan_batch(
            sanitize_and_standardize(inp).unsqueeze(0),
            rng.keyed("parent", plan_index),
            source.parent_functions[plan_index],
            out_dim=5,
            noise_sigma_multiplier=1.0,
            noise_spec=None,
            standardize_input=False,
        ).squeeze(0)
        for plan_index, inp in enumerate(inputs)
    ]
    expected = torch.logsumexp(torch.stack(transformed, dim=1), dim=1)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


@pytest.mark.parametrize("aggregation_kind", ["sum", "product", "max"])
def test_multi_function_matches_explicit_stacked_plan_for_associative_reducers(
    aggregation_kind: AggregationKind,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = [
        torch.randn(64, 3, generator=_make_generator(31)),
        torch.randn(64, 2, generator=_make_generator(32)),
        torch.randn(64, 4, generator=_make_generator(33)),
    ]
    source = StackedNodeSource(
        aggregation_kind=aggregation_kind,
        parent_functions=(
            LinearFunctionPlan(matrix=GaussianMatrixPlan()),
            LinearFunctionPlan(matrix=GaussianMatrixPlan()),
            LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    monkeypatch.setattr(
        "dagzoo.functions.multi.sample_multi_source_plan", lambda *_args, **_kwargs: source
    )

    actual_generator = _make_generator(34)
    reference_generator = _make_generator(34)
    actual = apply_multi_function(
        [inp.clone() for inp in inputs],
        actual_generator,
        out_dim=5,
    )

    root = _make_keyed_rng(reference_generator, "apply_multi_function")
    rng = FixedLayoutBatchRng.from_keyed_rng(root.keyed("execution"), batch_size=1, device="cpu")
    transformed = [
        apply_function_plan_batch(
            sanitize_and_standardize(inp).unsqueeze(0),
            rng.keyed("parent", plan_index),
            source.parent_functions[plan_index],
            out_dim=5,
            noise_sigma_multiplier=1.0,
            noise_spec=None,
            standardize_input=False,
        ).squeeze(0)
        for plan_index, inp in enumerate(inputs)
    ]
    expected = torch.stack(transformed, dim=1)
    if aggregation_kind == "sum":
        expected_out = torch.sum(expected, dim=1)
    elif aggregation_kind == "product":
        expected_out = torch.prod(expected, dim=1)
    else:
        expected_out = torch.max(expected, dim=1).values

    torch.testing.assert_close(actual, expected_out)
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_empty_raises() -> None:
    g = _make_generator(0)
    with pytest.raises(ValueError, match="non-empty"):
        apply_multi_function([], g, out_dim=3)


def test_deterministic() -> None:
    x = torch.randn(32, 4, generator=_make_generator(99))
    y1 = apply_multi_function([x.clone()], _make_generator(0), out_dim=3)
    y2 = apply_multi_function([x.clone()], _make_generator(0), out_dim=3)
    torch.testing.assert_close(y1, y2)


def test_finite_outputs() -> None:
    g = _make_generator(7)
    x = torch.randn(64, 4, generator=g)
    y = apply_multi_function([x], g, out_dim=3)
    assert torch.all(torch.isfinite(y))
