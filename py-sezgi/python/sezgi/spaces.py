"""Pythonic search-space builders (M4-1 Task 1).

Light dataclasses mirroring `Block` (`crates/core/src/space.rs:13-19`) 1:1,
one per block kind:

    Float(lo, hi, n)         -- continuous block, n coordinates in [lo, hi]
    Int(lo, hi, n)           -- integer block, n coordinates in [lo, hi]
    Categorical(k, n)        -- n categorical genes, each in 0..k
    Binary(n)                -- n bits
    Permutation(n)           -- one permutation of range(n)

`Space(*blocks)` composes one or more blocks, in the given order, into a
search space. A bare block is accepted wherever a `Space` is expected (see
`as_space`) -- `sezgi.Problem.space()` may return either.

These builders are pure Python: `Space._to_json()` serializes to `Block`'s
own JSON wire shape (`#[serde(tag = "type", rename_all = "snake_case")]`),
consumed directly by `_sezgi.from_callable_spaced(...)`
(`py-sezgi/src/lib.rs`) -- no new Rust-side space-parsing code, reusing
`Block`'s existing `Deserialize` impl.
"""
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Float:
    """A continuous block: `n` coordinates, each in `[lo, hi]`."""
    lo: float
    hi: float
    n: int

    def _to_dict(self):
        return {"type": "float", "lo": float(self.lo), "hi": float(self.hi), "n": int(self.n)}


@dataclass(frozen=True)
class Int:
    """An integer block: `n` coordinates, each in `[lo, hi]` (inclusive)."""
    lo: int
    hi: int
    n: int

    def _to_dict(self):
        return {"type": "int", "lo": int(self.lo), "hi": int(self.hi), "n": int(self.n)}


@dataclass(frozen=True)
class Categorical:
    """A categorical block: `n` genes, each a category index in `0..k`."""
    k: int
    n: int

    def _to_dict(self):
        return {"type": "categorical", "k": int(self.k), "n": int(self.n)}


@dataclass(frozen=True)
class Binary:
    """A binary block: `n` bits."""
    n: int

    def _to_dict(self):
        return {"type": "binary", "n": int(self.n)}


@dataclass(frozen=True)
class Permutation:
    """A permutation block: one permutation of `range(n)`."""
    n: int

    def _to_dict(self):
        return {"type": "permutation", "n": int(self.n)}


#: The set of concrete block builder types -- used by `Space`/`as_space` to
#: recognize a "bare block" argument.
_BLOCK_TYPES = (Float, Int, Categorical, Binary, Permutation)


class Space:
    """Composes one or more blocks into a search space, in the given order.

    A single-block space's `evaluate(x)` receives that block's own
    converted value, passed bare; a multi-block space's `x` is a tuple of
    per-block values, in the order the blocks were given here -- see
    `sezgi.Problem`'s own docstring for the exact conversion table.
    """

    def __init__(self, *blocks):
        if not blocks:
            raise ValueError("Space() requires at least one block")
        for b in blocks:
            if not isinstance(b, _BLOCK_TYPES):
                raise TypeError(
                    "Space() blocks must be Float/Int/Categorical/Binary/Permutation, "
                    f"got {type(b)!r}")
        self.blocks = tuple(blocks)

    def _to_json(self) -> str:
        return json.dumps([b._to_dict() for b in self.blocks])

    def __repr__(self):
        return f"Space{self.blocks!r}"

    def __eq__(self, other):
        return isinstance(other, Space) and self.blocks == other.blocks


def as_space(obj) -> Space:
    """Accepts a `Space`, or a bare block (wrapped into a single-block
    `Space`). Raises `TypeError` for anything else."""
    if isinstance(obj, Space):
        return obj
    if isinstance(obj, _BLOCK_TYPES):
        return Space(obj)
    raise TypeError(
        "expected a sezgi.Space or a bare block (Float/Int/Categorical/Binary/"
        f"Permutation), got {type(obj)!r}")
