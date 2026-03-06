import queue
import threading

import dagzoo.core.parallel_generation as parallel_generation_mod
import pytest

from dagzoo.bench.metrics import reproducibility_signature
from dagzoo.config import GeneratorConfig
from dagzoo.core.dataset import generate_batch_iter
from dagzoo.core.parallel_generation import (
    ParallelGenerationWorkerError,
    generate_parallel_batch_iter,
)


def _tiny_parallel_config() -> GeneratorConfig:
    cfg = GeneratorConfig.from_yaml("configs/default.yaml")
    cfg.dataset.task = "regression"
    cfg.dataset.n_train = 32
    cfg.dataset.n_test = 8
    cfg.dataset.n_features_min = 4
    cfg.dataset.n_features_max = 6
    cfg.graph.n_nodes_min = 3
    cfg.graph.n_nodes_max = 5
    cfg.runtime.worker_count = 2
    cfg.runtime.worker_index = 0
    cfg.runtime.device = "cpu"
    cfg.filter.enabled = False
    cfg.benchmark.preset_name = "parallel_test"
    return cfg


def test_generate_parallel_batch_iter_matches_serial_output_and_seed_order() -> None:
    cfg = _tiny_parallel_config()
    cfg.runtime.worker_count = 3

    produced = list(generate_parallel_batch_iter(cfg, num_datasets=6, seed=777, device="cpu"))
    serial = list(generate_batch_iter(cfg, num_datasets=6, seed=777, device="cpu"))
    expected_seeds = [int(bundle.metadata["seed"]) for bundle in serial]

    assert [int(bundle.metadata["seed"]) for bundle in produced] == expected_seeds
    assert reproducibility_signature(produced) == reproducibility_signature(serial)


def test_generate_parallel_batch_iter_preserves_root_worker_index_in_bundle_metadata() -> None:
    cfg = _tiny_parallel_config()
    cfg.runtime.worker_count = 3

    produced = list(generate_parallel_batch_iter(cfg, num_datasets=4, seed=777, device="cpu"))

    assert len(produced) == 4
    assert all(bundle.metadata["config"]["runtime"]["worker_index"] == 0 for bundle in produced)


def test_generate_parallel_batch_iter_yields_nothing_for_zero_datasets() -> None:
    cfg = _tiny_parallel_config()

    assert list(generate_parallel_batch_iter(cfg, num_datasets=0, seed=777, device="cpu")) == []


def test_generate_parallel_batch_iter_propagates_worker_error() -> None:
    cfg = _tiny_parallel_config()
    cfg.filter.enabled = True

    with pytest.raises(ParallelGenerationWorkerError, match="filter.enabled"):
        list(generate_parallel_batch_iter(cfg, num_datasets=4, seed=777, device="cpu"))


def test_generate_parallel_batch_iter_rejects_single_worker_request() -> None:
    cfg = _tiny_parallel_config()
    cfg.runtime.worker_count = 1

    with pytest.raises(ValueError, match=r"runtime\.worker_count > 1"):
        list(generate_parallel_batch_iter(cfg, num_datasets=1, seed=7, device="cpu"))


def test_generate_parallel_batch_iter_rejects_nonzero_worker_index() -> None:
    cfg = _tiny_parallel_config()
    cfg.runtime.worker_index = 1

    with pytest.raises(ValueError, match=r"runtime\.worker_index == 0"):
        list(generate_parallel_batch_iter(cfg, num_datasets=2, seed=7, device="cpu"))


def test_generate_parallel_batch_iter_rejects_non_cpu_resolved_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_parallel_config()

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._generation_context._resolve_device",
        lambda _config, _device: "cuda",
    )

    with pytest.raises(ValueError, match=r"supports resolved device 'cpu' only"):
        list(generate_parallel_batch_iter(cfg, num_datasets=2, seed=7, device="auto"))


def test_generate_parallel_batch_iter_caps_local_worker_count_to_cpu_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_parallel_config()
    cfg.runtime.worker_count = 64
    spawned_worker_counts: list[int] = []
    original_spawn = parallel_generation_mod._spawn_parallel_workers

    def _recording_spawn_parallel_workers(**kwargs):
        spawned_worker_counts.append(len(kwargs["worker_specs"]))
        return original_spawn(**kwargs)

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._local_parallel_worker_capacity",
        lambda: 2,
    )
    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._spawn_parallel_workers",
        _recording_spawn_parallel_workers,
    )

    produced = list(generate_parallel_batch_iter(cfg, num_datasets=6, seed=777, device="cpu"))

    assert len(produced) == 6
    assert spawned_worker_counts == [2]


def test_generate_parallel_batch_iter_handles_single_dataset_with_many_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _tiny_parallel_config()
    cfg.runtime.worker_count = 64
    spawned_worker_counts: list[int] = []
    original_spawn = parallel_generation_mod._spawn_parallel_workers

    def _recording_spawn_parallel_workers(**kwargs):
        spawned_worker_counts.append(len(kwargs["worker_specs"]))
        return original_spawn(**kwargs)

    monkeypatch.setattr(
        "dagzoo.core.parallel_generation._spawn_parallel_workers",
        _recording_spawn_parallel_workers,
    )

    produced = list(generate_parallel_batch_iter(cfg, num_datasets=1, seed=777, device="cpu"))
    serial = list(generate_batch_iter(cfg, num_datasets=1, seed=777, device="cpu"))

    assert len(produced) == 1
    assert int(produced[0].metadata["seed"]) == int(serial[0].metadata["seed"])
    assert spawned_worker_counts == [1]


def test_generate_parallel_batch_iter_close_does_not_hang() -> None:
    cfg = _tiny_parallel_config()
    cfg.runtime.worker_count = 2

    iterator = generate_parallel_batch_iter(
        cfg,
        num_datasets=12,
        seed=777,
        device="cpu",
        max_buffered_results=1,
    )
    _ = next(iterator)

    close_result: queue.Queue[BaseException | None] = queue.Queue()

    def _close_iterator() -> None:
        try:
            iterator.close()
        except BaseException as exc:  # pragma: no cover - surfaced via queue assertion
            close_result.put(exc)
            return
        close_result.put(None)

    close_thread = threading.Thread(target=_close_iterator, daemon=True)
    close_thread.start()
    close_thread.join(timeout=5.0)

    assert not close_thread.is_alive()
    assert close_result.get_nowait() is None
