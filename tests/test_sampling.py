import numpy as np
import torch

from cauchy_generator.sampling.random_weights import (
    sample_random_weights,
    sample_random_weights_torch,
)


def test_random_weights_normalized() -> None:
    rng = np.random.default_rng(42)
    w = sample_random_weights(32, rng)
    assert w.shape == (32,)
    assert np.all(w > 0)
    np.testing.assert_allclose(np.sum(w), 1.0, atol=1e-6, rtol=1e-6)


def test_random_weights_deterministic() -> None:
    a = sample_random_weights(16, np.random.default_rng(0))
    b = sample_random_weights(16, np.random.default_rng(0))
    np.testing.assert_array_equal(a, b)


def test_random_weights_positive() -> None:
    rng = np.random.default_rng(7)
    w = sample_random_weights(64, rng)
    assert np.all(w > 0)


def test_random_weights_torch_shape_and_sum() -> None:
    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    w = sample_random_weights_torch(16, g, "cpu")
    assert w.shape == (16,)
    assert torch.all(w > 0)
    torch.testing.assert_close(w.sum(), torch.tensor(1.0), atol=1e-5, rtol=1e-5)


def test_random_weights_torch_deterministic() -> None:
    g1 = torch.Generator(device="cpu")
    g1.manual_seed(0)
    a = sample_random_weights_torch(16, g1, "cpu")

    g2 = torch.Generator(device="cpu")
    g2.manual_seed(0)
    b = sample_random_weights_torch(16, g2, "cpu")

    torch.testing.assert_close(a, b)
