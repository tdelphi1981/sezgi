"""sezgi.Algorithm -- engine-hosted, class-first algorithm authoring base
(M4-1 Task 3). Hook taxonomy modeled after pymoo 0.6.2 (Apache-2.0) and
jMetal v7.5 (MIT) public APIs.

A subclass implements `generate(self, pop, ctx)` (REQUIRED -- offspring
production) and may optionally override `initialize(self, n, ctx)`
(population seeding) and `validate_space(self, space)` (a build-time veto).
Unlike `sezgi.AskTellAlgorithm` (formerly `sezgi.Algorithm`,
`py-sezgi/python/sezgi/algo.py`, M3-4 Task 2) -- which drives a pure-Python
`setup()`/`step()` loop OVER the ask/tell `EvalSession`, entirely in
Python -- this base's hooks run INSIDE the Rust engine's own generation
loop (population init, boundary repair, replacement, and termination all
happen Rust-side, exactly like a built-in `Generator`/`Initializer`
component): Python is called back once per stage per generation (or once
for `initialize`), not once per evaluation batch. `run()` is the sole
entry point, wiring a subclass instance through Task 2's `PyGenerator`
bridge (`generate`) and Task 3's `PyInitializer` bridge (`initialize`,
RULING A) over `_sezgi.solve_with_py_generator`.
"""
import abc

from sezgi import _sezgi
from sezgi.algo import SolveResult
from sezgi.problem import as_native_problem


class Algorithm(abc.ABC):
    """Subclass, implement `generate(self, pop, ctx)`; call `run(...)`.

    `pop` (`generate`'s first argument) is a population view:
    `pop.individuals` -- a Python list, one entry per genotype, each
    converted the SAME way `sezgi.Problem.evaluate(x)`'s own `x` is (bare
    for a single-block space, a tuple of per-block values for a
    multi-block one); `pop.fitness` -- a numpy 1-D `float64` array,
    parallel to `pop.individuals`.

    `ctx` (both `generate`'s and `initialize`'s second argument):
    `ctx.iteration` (int, the current generation count -- always `0` for
    `initialize`'s own single call), `ctx.space` (`list[dict]`, one
    descriptor per block in the problem's own block order -- e.g.
    `{"kind": "float", "lo": ..., "hi": ..., "n": ...}`), `ctx.rng` (a
    handle over the engine's OWN seeded stream for this stage/call --
    `ctx.rng.next_f64()` / `ctx.rng.next_below(n)`; drawing from it is
    what keeps a `run()` deterministic under a fixed `seed`, exactly like
    a built-in Rust component drawing from its own stage stream).

    `generate`'s and `initialize`'s return value: an iterable of x-values,
    the SAME bare/tuple convention as `pop.individuals` -- offspring/
    population COUNT is the hook's own choice (not checked against
    `pop.len()`/`n`, mirroring every Rust component's own freedom here).

    Spec persistence is deliberately absent: an engine-hosted Python
    algorithm is a live `generate`/`initialize`/`validate_space` Python
    callback registered into the Rust engine's per-call `Registry`
    (`_sezgi.solve_with_py_generator`) -- it is process-local (there is no
    Python-callback wire format this crate's `AlgorithmSpec` TOML can
    express), unlike a purely Rust-component spec, which is always
    serializable. This is an honest degrade, not a gap to silently paper
    over: subclass instances of this base cannot be saved to / loaded from
    a `.toml` spec file the way `sezgi.presets.*`/a hand-written spec can.
    """

    @abc.abstractmethod
    def generate(self, pop, ctx):
        """REQUIRED: produce this generation's offspring. See the class
        docstring for `pop`/`ctx`'s shape and the return-value convention."""
        raise NotImplementedError

    def initialize(self, n, ctx):
        """Optional: produce the initial population of size `n` (an
        iterable of `n` x-values, same convention as `generate`'s return
        value). Default: NOT overridden -- see `run()`'s own
        override-detection note; a subclass that does not override this
        method never has it called at all by `run()`: the engine's own
        Rust `init/uniform` initializer runs instead, with NO Python call
        involved (byte-identical to a plain `_sezgi.solve_with_py_generator`
        call with `initializer=None`). This default body is therefore
        never reached by `run()` itself -- it only exists so `Algorithm`
        need not be abstract over this hook, and so a direct call (outside
        `run()`) fails loudly rather than silently returning something
        iterable-shaped."""
        raise NotImplementedError(
            "Algorithm.initialize was not overridden by "
            f"{type(self).__name__} -- run() never calls this default "
            "body (see its own override-detection note); the engine's "
            "own init/uniform initializer is used instead")

    def validate_space(self, space):
        """Optional build-time veto: raise an exception to reject `space`
        (a `list[dict]`, the SAME shape as `ctx.space`) BEFORE any
        `generate()`/`initialize()` call -- `run()` surfaces it as a
        `ValueError` at build time. Default: no-op (accepts any space)."""
        return None

    def run(self, problem, budget, seed=0, pop_size=50, log_dir=None):
        """Runs this algorithm's `generate`/`initialize`/`validate_space`
        hooks INSIDE the Rust engine loop.

        problem: a `sezgi.Problem` subclass instance OR a native handle
            (anything `sezgi.solve()` itself accepts: `sezgi.bbob(...)`,
            `sezgi.problems.cec2022(...)`, `sezgi.problems.tsp(...)`,
            `sezgi.from_callable(...)`, `sezgi.bias.f0(...)`, ...) --
            converted via `sezgi.as_native_problem`.
        budget, seed, pop_size: as in every other sezgi entry point
            (`pop_size` default 50, distinct from
            `_sezgi.solve_with_py_generator`'s own lower-level default 20 --
            this base's own pinned default).
        log_dir: forwarded to `_sezgi.solve_with_py_generator`'s own
            `log_dir` (RULING B, M4-1 Task 3 -- wired end-to-end, mirroring
            `sezgi.solve()`'s own per-problem-type IOH-logging support):
            IOH-logged for a BBOB / CEC 2022 / CEC 2014 / CEC 2017
            `problem`; raises `ValueError` for anything else (TSP, the
            typed diagnostics, `from_callable(...)`, a `sezgi.Problem`
            subclass, `bias.f0(...)`) -- same support boundary
            `sezgi.solve()` itself has, for the same reason (no fid/
            instance/known-optimum identity to log against).

        Returns the SAME `SolveResult` dataclass `AskTellAlgorithm.solve()`
        returns (`sezgi.algo.SolveResult`) -- one result shape everywhere,
        per this milestone's own design ruling (no new result type for the
        new base). `f_opt`/`gap` are populated whenever the native
        handle's own `.optimum()` is not `None`.

        Override detection (RULING A): whether `initialize` was actually
        overridden by `type(self)` is decided by comparing the resolved
        method's underlying function against `Algorithm.initialize` itself
        (function-identity, not a sentinel return value or name/string
        match -- a legitimate override may itself choose to return
        anything, including an empty sequence, so only identity is a safe
        signal here). NOT overridden -> `initializer=None` is passed to
        `_sezgi.solve_with_py_generator`, which then keeps its own default
        `init_kind="init/uniform"` in full, unchanged control -- the exact
        same pure-Rust init path Task 2's raw-callback bridge always used,
        with NO Python call for `initialize` at all. Overridden -> `self`
        itself (already has a matching `initialize(n, ctx)` method) is
        passed as the `initializer` callback, registered under
        `"py/initializer"` (`PyInitializer`, Task 3's `Initializer`-trait
        counterpart of Task 2's `PyGenerator`).
        """
        native = as_native_problem(problem)
        algo_name = type(self).__name__.lower()

        overridden = type(self).initialize is not Algorithm.initialize
        initializer = self if overridden else None

        result = _sezgi.solve_with_py_generator(
            self, native, budget, master_seed=seed, run_id=0,
            pop_size=pop_size, log_dir=log_dir, initializer=initializer,
            algo_name=algo_name)

        f_opt = native.optimum()
        best_f = result["best_f"]
        return SolveResult(
            algo=algo_name, seed=seed, budget=budget,
            evals_used=result["evals_used"], best_x=result["best_x"],
            best_f=best_f, f_opt=f_opt,
            gap=None if f_opt is None else best_f - f_opt)
