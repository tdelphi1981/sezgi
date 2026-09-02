"""sezgi.Problem -- subclassable Python ABC for a search-space problem
(M4-1 Task 1).

Sits on top of the block-typed callable-problem bridge
(`_sezgi.from_callable_spaced`, `py-sezgi/src/lib.rs`), the widening of the
existing `sezgi.from_callable(...)` bridge to any `sezgi.Space` (not just a
single Float block). A subclass declares its search space (`space()`) and
objective (`evaluate(x)`); `_to_native()` builds the native `Problem` handle
every existing entry point (`sezgi.solve`, `EvalSession.for_problem`, ...)
already accepts.

Genotype -> Python conversion table (PINNED; `block_value_to_py`/
`genotype_to_py` in `py-sezgi/src/lib.rs`):

    Float block        -> list[float]
    Int block           -> list[int]
    Categorical block    -> list[int]   (category INDICES 0..k, not labels)
    Binary block         -> list[bool]
    Permutation block     -> list[int]

A single-block space's `x` is that one block's own converted value, passed
BARE. A multi-block space's `x` is a Python tuple of per-block converted
values, in `space()`'s block order.
"""
import abc

from sezgi import _sezgi
from sezgi.spaces import Space, as_space


class Problem(abc.ABC):
    """Subclass and implement `evaluate`/`space`; `optimum`/`batch_evaluate`
    are optional overrides."""

    @abc.abstractmethod
    def evaluate(self, x):
        """x -> float (the fitness/objective value at x).

        For a single-block space, `x` is that block's own converted value,
        passed bare. For a multi-block space, `x` is a tuple of per-block
        values, in `space()`'s block order. See the module docstring's
        conversion table for each block kind's exact Python type."""
        raise NotImplementedError

    @abc.abstractmethod
    def space(self):
        """Declares the search space: a `sezgi.Space(...)`, or a bare block
        (`sezgi.Float(...)`, `sezgi.Binary(...)`, ...)."""
        raise NotImplementedError

    def optimum(self):
        """The problem's known optimum (a float), or `None` (default) if it
        has none.

        Design decision (M4-1 Task 1): plumbed through to the native
        handle's own `.optimum()` accessor (`_to_native()` passes this
        value to `_sezgi.from_callable_spaced(..., optimum=...)`), so
        `problem._to_native().optimum()` and `problem.optimum()` agree --
        the native surface DOES support carrying an arbitrary Python-
        supplied optimum for a callable-bridge handle (a one-field addition
        to `Inner::CallableSpaced`, no core change), so there was no reason
        to leave it Python-side only."""
        return None

    def batch_evaluate(self, xs):
        """xs -> list[float], one entry per x in xs (default: loop
        `evaluate`). Override for a vectorized objective; called ONCE per
        generation with the whole population as `xs` (see
        `_sezgi.from_callable_spaced`'s own doc)."""
        return [self.evaluate(x) for x in xs]

    def _to_native(self):
        """Builds the native `Problem` handle via the block-typed callable
        bridge (`_sezgi.from_callable_spaced`), `vectorized=True` -- the
        native side calls `self.batch_evaluate` once per generation with
        the whole population; the default `batch_evaluate` above loops
        `self.evaluate`, so a subclass that only overrides `evaluate` still
        works correctly and efficiently (a single GIL-held Python call per
        generation, not per individual)."""
        space = as_space(self.space())
        return _sezgi.from_callable_spaced(
            self.batch_evaluate, space._to_json(), True, self.optimum())


def as_native_problem(obj):
    """Accepts anything `sezgi`'s entry points (`solve`, future `.run()`
    methods, ...) should accept as "a problem": a native `Problem` handle
    passes through unchanged; a `sezgi.Problem` subclass instance is
    converted via `_to_native()`; anything else raises `TypeError`."""
    if isinstance(obj, _sezgi.Problem):
        return obj
    if isinstance(obj, Problem):
        return obj._to_native()
    raise TypeError(
        "expected a sezgi.Problem subclass instance or a native Problem handle "
        f"(e.g. sezgi.bbob(...)), got {type(obj)!r}")
