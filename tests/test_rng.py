"""Tests for rng.py — seed derivation and SeedManager."""

from cauchy_generator.rng import SeedManager, derive_seed


def test_derive_seed_deterministic() -> None:
    assert derive_seed(42, "a", 1) == derive_seed(42, "a", 1)


def test_derive_seed_varies_with_base() -> None:
    assert derive_seed(1, "x") != derive_seed(2, "x")


def test_derive_seed_varies_with_component() -> None:
    assert derive_seed(42, "a") != derive_seed(42, "b")


def test_derive_seed_returns_valid_32bit() -> None:
    seed = derive_seed(99, "comp", 7)
    assert 0 <= seed < 2**32


def test_seed_manager_child_matches_derive() -> None:
    sm = SeedManager(seed=42)
    assert sm.child("node", 3) == derive_seed(42, "node", 3)
