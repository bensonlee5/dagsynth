import pytest
import torch

from dagzoo.converters.categorical import apply_categorical_converter
from dagzoo.converters.numeric import apply_numeric_converter
from dagzoo.core.execution_semantics import typed_converter_specs
from dagzoo.core.fixed_layout_batched import (
    FixedLayoutBatchRng,
    _apply_categorical_group_batch,
    _apply_node_plan_batch,
    _sample_random_points_batch,
    apply_numeric_converter_plan_batch,
    apply_function_plan_batch,
)
from dagzoo.core.fixed_layout_plan_types import (
    ActivationMatrixPlan,
    CategoricalConverterPlan,
    FixedActivationPlan,
    FixedLayoutLatentPlan,
    FixedLayoutNodePlan,
    GaussianMatrixPlan,
    GpFunctionPlan,
    LinearFunctionPlan,
    NeuralNetFunctionPlan,
    NumericConverterPlan,
    ProductFunctionPlan,
    QuadraticFunctionPlan,
    RandomPointsNodeSource,
    StackedNodeSource,
    TreeFunctionPlan,
    DiscretizationFunctionPlan,
    EmFunctionPlan,
    fixed_layout_converter_groups,
)
from dagzoo.core.node_pipeline import ConverterSpec, apply_node_pipeline
from dagzoo.diagnostics.effective_diversity import AblationArm, _runtime_override_context
from dagzoo.functions.multi import apply_multi_function
from dagzoo.functions.random_functions import apply_random_function
from dagzoo.sampling.random_points import sample_random_points
from conftest import make_generator as _make_generator, make_keyed_rng as _make_keyed_rng
import dagzoo.converters.categorical as categorical_mod
import dagzoo.converters.numeric as numeric_mod
import dagzoo.core.fixed_layout_batched as fixed_layout_batched_mod
import dagzoo.core.execution_semantics as execution_semantics_mod
import dagzoo.core.node_pipeline as node_pipeline_mod
import dagzoo.functions.multi as multi_mod
import dagzoo.functions.random_functions as random_functions_mod
import dagzoo.sampling.random_points as random_points_mod


@pytest.mark.parametrize(
    ("family", "plan"),
    [
        ("linear", LinearFunctionPlan(matrix=GaussianMatrixPlan())),
        ("quadratic", QuadraticFunctionPlan(matrix=GaussianMatrixPlan())),
        (
            "nn",
            NeuralNetFunctionPlan(
                n_layers=2,
                hidden_width=5,
                input_activation=FixedActivationPlan(name="relu"),
                output_activation=None,
                layer_matrices=(GaussianMatrixPlan(), GaussianMatrixPlan()),
                hidden_activations=(FixedActivationPlan(name="tanh"),),
            ),
        ),
        ("tree", TreeFunctionPlan(n_trees=2, depths=(2, 3))),
        (
            "discretization",
            DiscretizationFunctionPlan(
                n_centers=4,
                linear_matrix=GaussianMatrixPlan(),
            ),
        ),
        ("gp", GpFunctionPlan(branch_kind="ha")),
        (
            "em",
            EmFunctionPlan(
                m_val=4,
                linear_matrix=GaussianMatrixPlan(),
            ),
        ),
        (
            "product",
            ProductFunctionPlan(
                lhs=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
                rhs=QuadraticFunctionPlan(matrix=GaussianMatrixPlan()),
            ),
        ),
    ],
)
def test_apply_random_function_matches_explicit_plan(
    family: str,
    plan: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(32, 4, generator=_make_generator(1))
    monkeypatch.setattr(
        random_functions_mod,
        "sample_function_plan_for_family",
        lambda *_args, **_kwargs: plan,
    )

    actual_generator = _make_generator(2)
    reference_generator = _make_generator(2)
    actual = apply_random_function(x.clone(), actual_generator, out_dim=3, function_type=family)  # type: ignore[arg-type]

    root = _make_keyed_rng(reference_generator, "apply_random_function")
    rng = FixedLayoutBatchRng.from_keyed_rng(root.keyed("execution"), batch_size=1, device="cpu")
    expected = apply_function_plan_batch(
        random_functions_mod._standardize(x).unsqueeze(0),
        rng,
        plan,  # type: ignore[arg-type]
        out_dim=3,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
        standardize_input=False,
    ).squeeze(0)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


@pytest.mark.parametrize("family", ["nn", "tree", "discretization", "em"])
def test_sample_function_plan_for_family_uses_generator_device_for_log_uniform(
    family: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _make_generator(123)
    calls: list[str] = []
    monkeypatch.setattr(
        execution_semantics_mod,
        "_log_uniform",
        lambda *_args: calls.append(_args[3]) or 5.0,
    )
    monkeypatch.setattr(
        execution_semantics_mod.KeyedRng,
        "torch_rng",
        lambda _self, *args, **kwargs: _make_generator(999),
    )
    monkeypatch.setattr(execution_semantics_mod, "_generator_device", lambda *_args: "cuda")
    monkeypatch.setattr(execution_semantics_mod, "_randint_scalar", lambda *_args, **_kwargs: 2)
    monkeypatch.setattr(
        execution_semantics_mod, "_sample_bool_keyed", lambda *_args, **_kwargs: False
    )
    monkeypatch.setattr(
        execution_semantics_mod,
        "sample_activation_plan",
        lambda *_args, **_kwargs: FixedActivationPlan(name="relu"),
    )
    monkeypatch.setattr(
        execution_semantics_mod,
        "sample_matrix_plan",
        lambda *_args, **_kwargs: GaussianMatrixPlan(),
    )

    execution_semantics_mod.sample_function_plan_for_family(
        generator,
        family=family,  # type: ignore[arg-type]
        out_dim=4,
        mechanism_logit_tilt=0.0,
        function_family_mix=None,
    )

    assert calls == ["cuda"]


def test_keyed_discrete_plan_sampling_uses_generator_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = _make_generator(124)
    devices: list[str] = []

    monkeypatch.setattr(execution_semantics_mod, "_generator_device", lambda *_args: "cuda")
    monkeypatch.setattr(
        execution_semantics_mod.KeyedRng,
        "torch_rng",
        lambda _self, *args, **kwargs: (
            devices.append(str(kwargs["device"])) or _make_generator(1000)
        ),
    )
    monkeypatch.setattr(
        execution_semantics_mod,
        "sample_function_plan",
        lambda *_args, **_kwargs: LinearFunctionPlan(matrix=GaussianMatrixPlan()),
    )

    execution_semantics_mod.sample_function_family(
        generator,
        mechanism_logit_tilt=0.0,
        function_family_mix=None,
    )
    execution_semantics_mod.sample_matrix_plan(generator)
    execution_semantics_mod.sample_activation_plan(generator)
    execution_semantics_mod.sample_converter_plan(
        ConverterSpec(key="feature", kind="cat", dim=3, cardinality=5),
        generator,
        mechanism_logit_tilt=0.0,
        function_family_mix=None,
    )
    execution_semantics_mod.sample_multi_source_plan(
        generator,
        parent_count=2,
        out_dim=3,
        mechanism_logit_tilt=0.0,
        function_family_mix=None,
    )
    execution_semantics_mod.sample_root_source_plan(
        generator,
        out_dim=3,
        mechanism_logit_tilt=0.0,
        function_family_mix=None,
    )

    assert devices
    assert all(device == "cuda" for device in devices)


@pytest.mark.parametrize(
    ("family", "plan", "mapped_plan"),
    [
        (
            "nn",
            NeuralNetFunctionPlan(
                n_layers=2,
                hidden_width=5,
                input_activation=FixedActivationPlan(name="relu"),
                output_activation=FixedActivationPlan(name="relu"),
                layer_matrices=(GaussianMatrixPlan(), GaussianMatrixPlan()),
                hidden_activations=(FixedActivationPlan(name="relu"),),
            ),
            NeuralNetFunctionPlan(
                n_layers=2,
                hidden_width=5,
                input_activation=FixedActivationPlan(name="tanh"),
                output_activation=FixedActivationPlan(name="tanh"),
                layer_matrices=(GaussianMatrixPlan(), GaussianMatrixPlan()),
                hidden_activations=(FixedActivationPlan(name="tanh"),),
            ),
        ),
        (
            "linear",
            LinearFunctionPlan(
                matrix=ActivationMatrixPlan(
                    base_kind="gaussian",
                    activation=FixedActivationPlan(name="relu"),
                )
            ),
            LinearFunctionPlan(
                matrix=ActivationMatrixPlan(
                    base_kind="gaussian",
                    activation=FixedActivationPlan(name="tanh"),
                )
            ),
        ),
    ],
)
def test_apply_random_function_respects_activation_override(
    family: str,
    plan: object,
    mapped_plan: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(32, 4, generator=_make_generator(6))
    monkeypatch.setattr(
        random_functions_mod,
        "sample_function_plan_for_family",
        lambda *_args, **_kwargs: plan,
    )

    actual_generator = _make_generator(7)
    with _runtime_override_context(
        AblationArm(arm_id="act_relu_to_tanh", description="x", activation_map={"relu": "tanh"})
    ):
        actual = apply_random_function(
            x.clone(),
            actual_generator,
            out_dim=3,
            function_type=family,  # type: ignore[arg-type]
        )

    monkeypatch.setattr(
        random_functions_mod,
        "sample_function_plan_for_family",
        lambda *_args, **_kwargs: mapped_plan,
    )
    expected_generator = _make_generator(7)
    expected = apply_random_function(
        x.clone(),
        expected_generator,
        out_dim=3,
        function_type=family,  # type: ignore[arg-type]
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_generator.get_state(), expected_generator.get_state())


def test_apply_numeric_converter_matches_explicit_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(24, 1, generator=_make_generator(10))
    plan = NumericConverterPlan(kind="num", warp_enabled=True)
    monkeypatch.setattr(numeric_mod, "sample_converter_plan", lambda *_args, **_kwargs: plan)

    actual_generator = _make_generator(11)
    reference_generator = _make_generator(11)
    actual_x, actual_values = apply_numeric_converter(x.clone(), actual_generator)

    root = _make_keyed_rng(reference_generator, "apply_numeric_converter")
    rng = FixedLayoutBatchRng.from_keyed_rng(root.keyed("execution"), batch_size=1, device="cpu")
    expected_x, expected_values = apply_numeric_converter_plan_batch(
        x.unsqueeze(0),
        rng,
        plan,
    )

    torch.testing.assert_close(actual_x, expected_x.squeeze(0))
    torch.testing.assert_close(actual_values, expected_values.squeeze(0))
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_apply_multi_function_respects_logsumexp_aggregation_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = [
        torch.randn(32, 3, generator=_make_generator(14)),
        torch.randn(32, 2, generator=_make_generator(15)),
    ]
    source = StackedNodeSource(
        aggregation_kind="logsumexp",
        parent_functions=(
            LinearFunctionPlan(matrix=GaussianMatrixPlan()),
            LinearFunctionPlan(matrix=GaussianMatrixPlan()),
        ),
    )
    mapped_source = StackedNodeSource(
        aggregation_kind="max",
        parent_functions=source.parent_functions,
    )
    monkeypatch.setattr(
        multi_mod,
        "sample_multi_source_plan",
        lambda *_args, **_kwargs: source,
    )

    actual_generator = _make_generator(16)
    with _runtime_override_context(
        AblationArm(
            arm_id="agg_logsumexp_to_max",
            description="x",
            aggregation_map={"logsumexp": "max"},
        )
    ):
        actual = apply_multi_function(
            [inp.clone() for inp in inputs],
            actual_generator,
            out_dim=4,
        )

    monkeypatch.setattr(
        multi_mod,
        "sample_multi_source_plan",
        lambda *_args, **_kwargs: mapped_source,
    )
    expected_generator = _make_generator(16)
    expected = apply_multi_function(
        [inp.clone() for inp in inputs],
        expected_generator,
        out_dim=4,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_generator.get_state(), expected_generator.get_state())


def test_apply_categorical_converter_matches_explicit_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.randn(24, 3, generator=_make_generator(20))
    plan = CategoricalConverterPlan(
        kind="cat",
        method="neighbor",
        variant="center_random_fn",
        function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
    )
    monkeypatch.setattr(categorical_mod, "sample_converter_plan", lambda *_args, **_kwargs: plan)

    actual_generator = _make_generator(21)
    reference_generator = _make_generator(21)
    actual_x, actual_labels = apply_categorical_converter(
        x.clone(), actual_generator, n_categories=5
    )

    root = _make_keyed_rng(reference_generator, "apply_categorical_converter")
    rng = FixedLayoutBatchRng.from_keyed_rng(root.keyed("execution"), batch_size=1, device="cpu")
    expected_x, expected_labels = _apply_categorical_group_batch(
        x.unsqueeze(0).unsqueeze(2),
        rng,
        plan,
        n_categories=5,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    torch.testing.assert_close(actual_x, expected_x[0, :, 0, :])
    torch.testing.assert_close(actual_labels, expected_labels[0, :, 0])
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_sample_random_points_matches_explicit_root_source_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = RandomPointsNodeSource(
        base_kind="uniform",
        function=LinearFunctionPlan(matrix=GaussianMatrixPlan()),
    )
    monkeypatch.setattr(
        random_points_mod, "sample_root_source_plan", lambda *_args, **_kwargs: source
    )

    actual_generator = _make_generator(30)
    reference_generator = _make_generator(30)
    actual = sample_random_points(32, 4, actual_generator, "cpu")

    root = _make_keyed_rng(reference_generator, "sample_random_points")
    rng = FixedLayoutBatchRng.from_keyed_rng(
        root.keyed("execution", "source"),
        batch_size=1,
        device="cpu",
    )
    base = _sample_random_points_batch(
        rng.keyed("base"),
        n_rows=32,
        dim=4,
        base_kind=source.base_kind,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )
    expected = apply_function_plan_batch(
        base,
        rng.keyed("function"),
        source.function,
        out_dim=4,
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    ).squeeze(0)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_apply_node_pipeline_matches_explicit_node_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [
        ConverterSpec(key="feature_0", kind="num", dim=1),
        ConverterSpec(key="feature_1", kind="cat", dim=3, cardinality=4),
    ]
    typed_specs = typed_converter_specs(specs)
    converter_plans = (
        NumericConverterPlan(kind="num", warp_enabled=True),
        CategoricalConverterPlan(
            kind="cat",
            method="softmax",
            variant="softmax_points",
        ),
    )
    node_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(0,),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
        latent=FixedLayoutLatentPlan(required_dim=4, extra_dim=2, total_dim=6),
        source=StackedNodeSource(
            aggregation_kind="sum",
            parent_functions=(LinearFunctionPlan(matrix=GaussianMatrixPlan()),),
        ),
    )
    monkeypatch.setattr(node_pipeline_mod, "sample_node_plan", lambda **_kwargs: node_plan)

    parent = torch.randn(16, 4, generator=_make_generator(40))
    actual_generator = _make_generator(41)
    reference_generator = _make_generator(41)
    actual_latent, actual_extracted = apply_node_pipeline(
        [parent.clone()],
        16,
        specs,
        actual_generator,
        "cpu",
    )

    root = _make_keyed_rng(reference_generator, "apply_node_pipeline")
    rng = FixedLayoutBatchRng.from_keyed_rng(root.keyed("execution"), batch_size=1, device="cpu")
    expected_latent, expected_extracted = _apply_node_plan_batch(
        None,
        node_plan,
        [parent.unsqueeze(0)],
        n_rows=16,
        rng=rng,
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    torch.testing.assert_close(actual_latent, expected_latent.squeeze(0))
    for key, value in actual_extracted.items():
        torch.testing.assert_close(value, expected_extracted[key].squeeze(0))
    torch.testing.assert_close(actual_generator.get_state(), reference_generator.get_state())


def test_apply_node_pipeline_respects_logsumexp_aggregation_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [ConverterSpec(key="value", kind="num", dim=1)]
    typed_specs = typed_converter_specs(specs)
    converter_plans = (NumericConverterPlan(kind="num", warp_enabled=False),)
    node_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(0, 1),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
        latent=FixedLayoutLatentPlan(required_dim=1, extra_dim=1, total_dim=2),
        source=StackedNodeSource(
            aggregation_kind="logsumexp",
            parent_functions=(
                LinearFunctionPlan(matrix=GaussianMatrixPlan()),
                LinearFunctionPlan(matrix=GaussianMatrixPlan()),
            ),
        ),
    )
    mapped_plan = FixedLayoutNodePlan(
        node_index=node_plan.node_index,
        parent_indices=node_plan.parent_indices,
        converter_specs=node_plan.converter_specs,
        converter_plans=node_plan.converter_plans,
        converter_groups=node_plan.converter_groups,
        latent=node_plan.latent,
        source=StackedNodeSource(
            aggregation_kind="max",
            parent_functions=node_plan.source.parent_functions,  # type: ignore[attr-defined]
        ),
    )
    parents = [
        torch.randn(16, 3, generator=_make_generator(42)),
        torch.randn(16, 2, generator=_make_generator(43)),
    ]
    monkeypatch.setattr(node_pipeline_mod, "sample_node_plan", lambda **_kwargs: node_plan)

    actual_generator = _make_generator(44)
    with _runtime_override_context(
        AblationArm(
            arm_id="agg_logsumexp_to_max",
            description="x",
            aggregation_map={"logsumexp": "max"},
        )
    ):
        actual_latent, actual_extracted = apply_node_pipeline(
            [parent.clone() for parent in parents],
            16,
            specs,
            actual_generator,
            "cpu",
        )

    monkeypatch.setattr(node_pipeline_mod, "sample_node_plan", lambda **_kwargs: mapped_plan)
    expected_generator = _make_generator(44)
    expected_latent, expected_extracted = apply_node_pipeline(
        [parent.clone() for parent in parents],
        16,
        specs,
        expected_generator,
        "cpu",
    )

    torch.testing.assert_close(actual_latent, expected_latent)
    for key, value in actual_extracted.items():
        torch.testing.assert_close(value, expected_extracted[key])
    torch.testing.assert_close(actual_generator.get_state(), expected_generator.get_state())


@pytest.mark.parametrize(("kind", "key"), [("num", "value"), ("target_reg", "target")])
def test_apply_node_plan_replaces_full_multi_column_numeric_slices(
    kind: str,
    key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = [ConverterSpec(key=key, kind=kind, dim=3)]  # type: ignore[arg-type]
    typed_specs = typed_converter_specs(specs)
    converter_plans = (NumericConverterPlan(kind=kind, warp_enabled=True),)  # type: ignore[arg-type]
    node_plan = FixedLayoutNodePlan(
        node_index=0,
        parent_indices=(0,),
        converter_specs=typed_specs,
        converter_plans=converter_plans,
        converter_groups=fixed_layout_converter_groups(typed_specs, converter_plans),
        latent=FixedLayoutLatentPlan(required_dim=3, extra_dim=0, total_dim=3),
        source=StackedNodeSource(
            aggregation_kind="sum",
            parent_functions=(LinearFunctionPlan(matrix=GaussianMatrixPlan()),),
        ),
    )
    pattern = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).view(1, 1, 3)
    monkeypatch.setattr(
        fixed_layout_batched_mod,
        "apply_numeric_converter_plan_batch",
        lambda x, _rng, _plan: (
            pattern.expand(x.shape[0], x.shape[1], -1).to(x.device),
            torch.full((x.shape[0], x.shape[1]), 5.0, device=x.device),
        ),
    )

    rng = FixedLayoutBatchRng.from_generator(_make_generator(45), batch_size=1, device="cpu")
    latent, extracted = _apply_node_plan_batch(
        None,
        node_plan,
        [torch.randn(1, 8, 3, generator=_make_generator(46))],
        n_rows=8,
        rng=rng,
        device="cpu",
        noise_sigma_multiplier=1.0,
        noise_spec=None,
    )

    torch.testing.assert_close(latent[0, :, 1], 2.0 * latent[0, :, 0])
    torch.testing.assert_close(latent[0, :, 2], 3.0 * latent[0, :, 0])
    torch.testing.assert_close(
        extracted[key],
        torch.full((1, 8), 5.0, dtype=torch.float32),
    )
