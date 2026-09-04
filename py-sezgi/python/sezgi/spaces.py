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
    """A continuous block: `n` coordinates, each in `[lo, hi]`.

    Accepted bare wherever a `sezgi.Space` is expected -- `sezgi.Problem.
    space()` may return a `Float(...)` directly instead of wrapping it in
    `Space(Float(...))` (see `as_space`). Converts to `list[float]` when it
    becomes (all or part of) `evaluate`'s `x` argument (see
    `sezgi.problem`'s own conversion table).

        Float(-5.0, 5.0, 10)   # 10 coordinates, each in [-5.0, 5.0]
    """
    lo: float
    hi: float
    n: int

    def _to_dict(self):
        return {"type": "float", "lo": float(self.lo), "hi": float(self.hi), "n": int(self.n)}


@dataclass(frozen=True)
class Int:
    """An integer block: `n` coordinates, each in `[lo, hi]` (inclusive).

    Accepted bare wherever a `sezgi.Space` is expected -- see `Float`'s own
    docstring. Converts to `list[int]` (see `sezgi.problem`'s conversion
    table).

        Int(0, 9, 5)   # 5 coordinates, each an integer in [0, 9]
    """
    lo: int
    hi: int
    n: int

    def _to_dict(self):
        return {"type": "int", "lo": int(self.lo), "hi": int(self.hi), "n": int(self.n)}


@dataclass(frozen=True)
class Categorical:
    """A categorical block: `n` genes, each a category index in `0..k`.

    Accepted bare wherever a `sezgi.Space` is expected -- see `Float`'s own
    docstring. Converts to `list[int]` -- category INDICES (`0..k`), not
    labels (see `sezgi.problem`'s conversion table).

        Categorical(k=4, n=3)   # 3 genes, each an index in {0, 1, 2, 3}
    """
    k: int
    n: int

    def _to_dict(self):
        return {"type": "categorical", "k": int(self.k), "n": int(self.n)}


@dataclass(frozen=True)
class Binary:
    """A binary block: `n` bits.

    Accepted bare wherever a `sezgi.Space` is expected -- see `Float`'s own
    docstring. Converts to `list[bool]` (see `sezgi.problem`'s conversion
    table).

        Binary(8)   # 8 bits
    """
    n: int

    def _to_dict(self):
        return {"type": "binary", "n": int(self.n)}


@dataclass(frozen=True)
class Permutation:
    """A permutation block: one permutation of `range(n)`.

    Accepted bare wherever a `sezgi.Space` is expected -- see `Float`'s own
    docstring. Converts to `list[int]`, a 0-based permutation of `range(n)`
    (see `sezgi.problem`'s conversion table).

        Permutation(10)   # one permutation of range(10), e.g. a TSP tour
    """
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
        """Composes `blocks`, in the given order, into one search space.

        blocks: one or more `Float`/`Int`/`Categorical`/`Binary`/
            `Permutation` instances, in the order `evaluate`'s `x` tuple
            should present them (see the class docstring).

        Raises `ValueError` if called with no blocks, `TypeError` if any
        argument is not one of the five block types."""
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

    def __hash__(self):
        """Consistent with `__eq__`: two `Space`s with equal `blocks` tuples
        hash equal (final-review INFO fix -- defining `__eq__` without
        `__hash__` makes Python set `__hash__ = None`, so `Space` was
        unhashable while the five frozen block dataclasses it composes are
        all hashable on their own). `blocks` is a tuple of frozen
        dataclasses, so it is itself hashable whenever every block in it
        is."""
        return hash(self.blocks)


def as_space(obj) -> Space:
    """Normalizes a `sezgi.Problem.space()` return value to a `Space`.

    obj: a `Space` (returned unchanged), or a bare block instance
        (`Float`/`Int`/`Categorical`/`Binary`/`Permutation`, wrapped into a
        single-block `Space`).

    Raises `TypeError` for anything else. Used by `Problem._to_native()`
    and `Algorithm`'s own space handling wherever a `space()` result needs
    to become an actual `Space`."""
    if isinstance(obj, Space):
        return obj
    if isinstance(obj, _BLOCK_TYPES):
        return Space(obj)
    raise TypeError(
        "expected a sezgi.Space or a bare block (Float/Int/Categorical/Binary/"
        f"Permutation), got {type(obj)!r}")
