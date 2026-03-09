"""RNG helpers for seeded, component-level reproducibility."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

SEED32_MIN = 0
SEED32_MAX = (2**32) - 1
_SEED32_MODULUS = 2**32


def validate_seed32(seed: int, *, field_name: str = "seed") -> int:
    """Validate an external seed against the supported unsigned 32-bit range."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(
            f"{field_name} must be an integer in [{SEED32_MIN}, {SEED32_MAX}], got {seed!r}."
        )
    if seed < SEED32_MIN or seed > SEED32_MAX:
        raise ValueError(
            f"{field_name} must be an integer in [{SEED32_MIN}, {SEED32_MAX}], got {seed!r}."
        )
    return int(seed)


def offset_seed32(seed: int, offset: int) -> int:
    """Derive a wrapped child seed via 32-bit modular arithmetic."""

    return (int(seed) + int(offset)) % _SEED32_MODULUS


def derive_seed(base_seed: int, *components: str | int) -> int:
    """Derive a deterministic 32-bit seed from a base seed and components."""

    h = hashlib.blake2s(digest_size=8)
    h.update(str(base_seed).encode("utf-8"))
    for comp in components:
        h.update(b"|")
        h.update(str(comp).encode("utf-8"))
    return int.from_bytes(h.digest(), "little") % SEED32_MAX


@dataclass(slots=True, frozen=True)
class KeyedRng:
    """A keyed RNG namespace rooted at one deterministic base seed."""

    seed: int
    path: tuple[str | int, ...] = ()

    def __post_init__(self) -> None:
        """Normalize public path input to an immutable tuple."""

        path = self.path
        normalized = (path,) if isinstance(path, str | int) else tuple(path)
        object.__setattr__(self, "path", normalized)

    def keyed(self, *components: str | int) -> "KeyedRng":
        """Return a child namespace with the provided semantic path appended."""

        return KeyedRng(seed=self.seed, path=self.path + tuple(components))

    def child_seed(self, *components: str | int) -> int:
        """Return a deterministic seed for this namespace and child components."""

        return derive_seed(self.seed, *self.path, *components)

    def torch_rng(self, *components: str | int, device: str = "cpu") -> torch.Generator:
        """Return a torch Generator for this namespace and child components."""

        g = torch.Generator(device=device)
        g.manual_seed(self.child_seed(*components))
        return g


def keyed_rng_from_generator(generator: torch.Generator, *components: str | int) -> KeyedRng:
    """Consume one ambient generator draw and convert it into a keyed RNG root."""

    seed = int(
        torch.randint(
            0,
            SEED32_MAX + 1,
            (1,),
            generator=generator,
            device=str(generator.device),
        ).item()
    )
    return KeyedRng(validate_seed32(seed)).keyed(*components)


@dataclass(slots=True)
class SeedManager:
    """Creates reproducible child seeds from a run-level seed."""

    seed: int

    def child(self, *components: str | int) -> int:
        """Return a deterministic child seed for the provided component path."""

        return KeyedRng(self.seed).child_seed(*components)

    def torch_rng(self, *components: str | int, device: str = "cpu") -> torch.Generator:
        """Return a torch Generator seeded from a deterministic child seed."""

        return KeyedRng(self.seed).torch_rng(*components, device=device)
