"""Tests for graph/cauchy_graph.py — Appendix E.4."""

import numpy as np

from cauchy_generator.graph.cauchy_graph import sample_cauchy_dag


def test_dag_shape() -> None:
    rng = np.random.default_rng(42)
    adj = sample_cauchy_dag(5, rng)
    assert adj.shape == (5, 5)
    assert adj.dtype == bool


def test_dag_upper_triangular() -> None:
    rng = np.random.default_rng(7)
    adj = sample_cauchy_dag(8, rng)
    for i in range(8):
        for j in range(i + 1):
            assert adj[i, j] == False  # noqa: E712


def test_dag_deterministic() -> None:
    a = sample_cauchy_dag(6, np.random.default_rng(99))
    b = sample_cauchy_dag(6, np.random.default_rng(99))
    np.testing.assert_array_equal(a, b)


def test_dag_single_node() -> None:
    rng = np.random.default_rng(0)
    adj = sample_cauchy_dag(1, rng)
    assert adj.shape == (1, 1)
    assert not adj[0, 0]
