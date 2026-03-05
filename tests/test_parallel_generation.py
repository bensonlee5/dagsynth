import pytest

from dagzoo.config import GeneratorConfig
from dagzoo.core.parallel_generation import generate_parallel_batch_iter
from dagzoo.rng import SeedManager


def test_generate_parallel_batch_iter_preserves_global_seed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 3
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    observed_seeds: list[int] = []

    def _stub_generate_one_seeded(
        config, *, seed: int, requested_device: str, resolved_device: str
    ):
        _ = config
        _ = requested_device
        _ = resolved_device
        observed_seeds.append(seed)
        return seed

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        _stub_generate_one_seeded,
    )

    produced = list(generate_parallel_batch_iter(cfg, num_datasets=7, seed=777, device="cpu"))
    manager = SeedManager(777)
    expected = [manager.child("dataset", idx) for idx in range(7)]

    assert produced == expected
    assert sorted(observed_seeds) == sorted(expected)


def test_generate_parallel_batch_iter_propagates_worker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    failing_seed = SeedManager(777).child("dataset", 2)

    def _stub_generate_one_seeded(
        config, *, seed: int, requested_device: str, resolved_device: str
    ):
        _ = config
        _ = requested_device
        _ = resolved_device
        if seed == failing_seed:
            raise RuntimeError("boom")
        return seed

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_engine._generate_one_seeded",
        _stub_generate_one_seeded,
    )

    with pytest.raises(RuntimeError, match="boom"):
        list(generate_parallel_batch_iter(cfg, num_datasets=6, seed=777, device="cpu"))


def test_generate_parallel_batch_iter_rejects_nonzero_worker_index() -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 1
    cfg.runtime.device = "cpu"

    with pytest.raises(ValueError, match=r"runtime\.worker_index == 0"):
        list(generate_parallel_batch_iter(cfg, num_datasets=2, seed=7, device="cpu"))


def test_generate_parallel_batch_iter_rejects_non_cpu_resolved_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_context._resolve_device",
        lambda _config, _device: "cuda",
    )

    with pytest.raises(ValueError, match=r"supports resolved device 'cpu' only"):
        list(generate_parallel_batch_iter(cfg, num_datasets=2, seed=7, device="auto"))
