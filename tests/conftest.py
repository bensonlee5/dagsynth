import torch

from dagzoo.rng import KeyedRng, keyed_rng_from_generator


def make_generator(seed: int = 42) -> torch.Generator:
    """Create a seeded torch Generator on CPU for deterministic tests."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def make_keyed_rng(generator: torch.Generator, *components: str | int) -> KeyedRng:
    """Consume one ambient draw and derive the same keyed root used by helpers."""

    return keyed_rng_from_generator(generator, *components)
