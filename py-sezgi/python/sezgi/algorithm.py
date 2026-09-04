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
from sezgi.algo import _wrap_result
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
        """REQUIRED: produce this generation's offspring.

        pop: the current population view -- `pop.individuals` (a Python
            list, one entry per genotype, bare/tuple-converted the same way
            `sezgi.Problem.evaluate(x)`'s own `x` is) and `pop.fitness` (a
            numpy 1-D `float64` array, parallel to `pop.individuals`).
        ctx: this call's context -- `ctx.iteration`, `ctx.space`, `ctx.rng`.

        See the class docstring for the exact shape of both arguments.

        Returns an iterable of x-values, the SAME bare/tuple convention as
        `pop.individuals` -- offspring count is this hook's own choice, not
        checked against `pop`'s size."""
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

    def run(self, problem, budget, seed=0, pop_size=50, log_dir=None,
            run_id=0, name=None):
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
        run_id: forwarded to `_sezgi.solve_with_py_generator`'s own
            `run_id` (default 0, additive -- final-review fix 2). Every
            builtin wrapper class's own `run()` already exposes this same
            knob (`sezgi.builtins`); this closes the gap so both front
            doors agree for a caller doing multi-run experiments.
        name: optional label overriding `type(self).__name__.lower()` for
            the result's `algo` field and the engine's own `algo_name`
            (default `None`, additive -- final-review fix 4). Mirrors
            `sezgi.AskTellAlgorithm`'s own `name` class-attribute semantics
            (`self.name or type(self).__name__.lower()`), exposed here as a
            `run()` parameter since `Algorithm` has no equivalent
            instance-level override point.

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
        algo_name = name or type(self).__name__.lower()

        overridden = type(self).initialize is not Algorithm.initialize
        initializer = self if overridden else None

        result = _sezgi.solve_with_py_generator(
            self, native, budget, master_seed=seed, run_id=run_id,
            pop_size=pop_size, log_dir=log_dir, initializer=initializer,
            algo_name=algo_name)

        f_opt = native.optimum()
        return _wrap_result(algo_name, seed, budget, result, f_opt)


class PopulationAlgorithm(Algorithm):
    """Family base for population-style algorithms (GA/DE/ES-shaped): a
    subclass implements `vary(self, parents, ctx)` (REQUIRED -- turns a
    parent pool into offspring) and may optionally override
    `select(self, pop, k, ctx)` (default: k-fold binary tournament,
    tournament size 2, minimization). `generate(self, pop, ctx)` is NOT
    meant to be overridden by a subclass of this base -- it is composed
    here from `select` + `vary` (see `generate`'s own docstring for the
    exact arity contract). Does NOT override `initialize`: a
    `PopulationAlgorithm` subclass that itself does not override
    `initialize` inherits `Algorithm`'s own override-detection default
    (the engine's Rust `init/uniform` path runs, with no Python call for
    `initialize` at all) -- exactly like any other `Algorithm` subclass;
    this base adds no special-casing there.

    Hook taxonomy modeled after pymoo 0.6.2 (Apache-2.0) and jMetal v7.5
    (MIT) selection/variation operator split, same attribution as the
    module docstring above.
    """

    def select(self, pop, k, ctx):
        """Default: k-fold binary tournament selection (tournament size
        2; minimization -- lower fitness wins).

        For each of the `k` requested parent slots, INDEPENDENTLY draws
        exactly two values from `ctx.rng` (with `n = len(pop.individuals)`):

            i = ctx.rng.next_below(n)
            j = ctx.rng.next_below(n - 1); if j >= i: j += 1      (n > 1)

        This is the standard "index-shift" trick for drawing two
        DISTINCT indices from `[0, n)` using exactly two draws and no
        rejection/retry loop -- the draw count per slot is always
        EXACTLY 2 for n > 1, so the whole sequence is hand-traceable
        given a fixed RNG seed and a known `n`. The individuals at `i`
        and `j` then "fight": `pop.fitness[i] <= pop.fitness[j]` wins as
        `i` -- this is also the EXACT tie-break rule: on an exact tie,
        the first-drawn contestant (`i`) always wins, never `j`. `n == 1`
        (a population of one) is a degenerate edge case with no second
        index to draw: only `i = ctx.rng.next_below(1)` (always `0`) is
        drawn, and `i` wins trivially, costing 1 draw instead of 2.

        Returns a list of exactly `k` parents (`pop.individuals` entries,
        one per slot, in slot order) -- ties/duplicates across slots are
        possible and expected, exactly like any other tournament
        selection scheme.
        """
        n = len(pop.individuals)
        parents = []
        for _ in range(k):
            i = ctx.rng.next_below(n)
            if n > 1:
                j = ctx.rng.next_below(n - 1)
                if j >= i:
                    j += 1
                winner = i if pop.fitness[i] <= pop.fitness[j] else j
            else:
                winner = i
            parents.append(pop.individuals[winner])
        return parents

    @abc.abstractmethod
    def vary(self, parents, ctx):
        """REQUIRED: turn a parent pool (a list of `len(parents)`
        genotypes, the SAME bare/tuple convention as `pop.individuals` --
        see the module docstring) into this generation's offspring (an
        iterable of x-values, same convention). Offspring COUNT is
        entirely this method's own choice, independent of
        `len(parents)` -- e.g. a DE-style `vary` can consume
        `len(parents) == len(pop)` parents and still emit exactly
        `len(pop)` offspring (one mutant per target index), while a
        crossover-pair `vary` might consume the same `k` parents two at a
        time and emit `k // 2 * 2` offspring."""
        raise NotImplementedError

    def generate(self, pop, ctx):
        """Composes `select()` then `vary()` -- NOT meant to be
        overridden by a `PopulationAlgorithm` subclass (override
        `select`/`vary` instead).

        Arity contract (PINNED): always calls
        `self.select(pop, len(pop.individuals), ctx)` -- i.e. the
        requested `k` is the CURRENT population's own size, one
        tournament result per current member (the standard "mating pool"
        convention: a full-size parent pool, not a fixed pair). The
        resulting `parents` list (always exactly `len(pop.individuals)`
        entries) is passed to `self.vary(parents, ctx)` UNCHANGED, and
        `vary`'s return value is returned UNCHANGED as this generation's
        offspring -- offspring count is entirely `vary`'s own choice (see
        `vary`'s own docstring), never checked against `len(parents)` or
        `pop.len()`.
        """
        parents = self.select(pop, len(pop.individuals), ctx)
        return self.vary(parents, ctx)


class LocalSearch(Algorithm):
    """Family base for single-trajectory local search (hill-climbing/
    SA-shaped): a subclass implements `neighbor(self, x, ctx)` (REQUIRED
    -- proposes one perturbed candidate from the current point) and may
    optionally override `accept(self, f_old, f_new, ctx)` (default:
    greedy, `f_new <= f_old`). Designed for `pop_size=1` (a population of
    exactly one point) -- `run()` is overridden here to default AND
    ENFORCE `pop_size=1` (`ValueError` otherwise): this base's whole
    design only makes sense for a single search trajectory, not a
    population.

    ACCEPT() DESIGN (read carefully -- this is not the naive reading of
    "accept decides whether the engine keeps the point"):

    This base runs over the engine's default `replace/mu-plus-lambda`
    replacer (`Algorithm.run`'s own default). At `pop_size=1` (mu=1),
    with `generate()` here always returning exactly ONE offspring
    (lambda=1 for this base), that replacer ALWAYS keeps the
    strictly-better (or, on an exact tie, the OLD) of {current, neighbor}
    as the NEXT call's `pop.individuals[0]`/`pop.fitness[0]` -- this is a
    structural, UNCONDITIONAL guarantee of `replace/mu-plus-lambda`
    itself (drains the pool, sorts, truncates to `mu`), not a decision
    `accept()` makes. Two direct consequences:

    1. `pop.fitness[0]`, as read by THIS base on the call following a
       `neighbor()` proposal, is ALWAYS `<= f_old` (the fitness this
       base proposed that neighbor from). `accept()` can therefore
       NEVER actually be called with an `f_new` that is objectively
       WORSE than `f_old` -- the replacer has already filtered that
       case out before Python ever sees the population again. A worse
       neighbor's exact fitness value is not even recoverable: the base
       can tell a neighbor LOST (because `pop.individuals[0]` still
       equals its own previously-tracked point, not the neighbor it
       proposed), but the lost neighbor's own fitness is never
       surfaced back to Python by this bridge. **This base's `accept()`
       therefore cannot implement true simulated-annealing-style
       "sometimes accept a worse move" semantics -- that would require
       intercepting the replacer's own decision, which this design
       deliberately does NOT attempt** (fighting the replacer -- e.g.
       trying to smuggle a worse point back into the reported
       population -- is not attempted; `replace/mu-plus-lambda` would
       simply overrule it again on the very next comparison anyway).
    2. Given (1), the DEFAULT `accept` (`f_new <= f_old`) is a
       TAUTOLOGY over every `(f_old, f_new)` pair this base can ever
       actually present to it -- it is ALWAYS true, faithfully
       mirroring (not fighting) what `replace/mu-plus-lambda` already
       unconditionally enforces at `pop_size=1`. This is intentional
       and documented rather than hidden: for the default
       configuration, `accept()`'s call is a confirmation of the
       engine's own greedy behavior, not an independent gate.

    What `accept()` DOES meaningfully control: whether this base's OWN
    internally-tracked anchor (`self.current_x`/`self.current_f`, the
    point the NEXT `neighbor()` call is proposed from) ADVANCES to match
    the engine's already-decided outcome, or stays where it was. A
    stricter-than-default `accept` (e.g. `f_new < f_old`, REJECTING an
    exact tie that the engine's own population has already moved to)
    makes the base re-propose `neighbor()` from the SAME previous anchor
    again on the next call, even though `pop.individuals[0]` itself has
    already moved -- i.e. `accept()` governs this base's own SEARCH
    TRAJECTORY (what x seeds the next `neighbor()` call), never
    population membership (which the replacer alone decides,
    unconditionally, at `pop_size=1`). This is the "return the accepted
    point as the next proposal base" design: `accept()` returning
    `False` means "propose from `current_x` again", never an attempt to
    veto or override what the engine's own population already contains.
    """

    current_x = None
    current_f = None

    @abc.abstractmethod
    def neighbor(self, x, ctx):
        """REQUIRED: propose ONE perturbed candidate from the current
        point `x` (the SAME bare/tuple convention as `pop.individuals` --
        see the module docstring). Returns a SINGLE x-value (NOT a list
        -- `generate()` wraps it into the required one-offspring list)."""
        raise NotImplementedError

    def accept(self, f_old, f_new, ctx):
        """Default: greedy (`f_new <= f_old`). See the class docstring's
        ACCEPT() DESIGN note for exactly what this controls (and does
        NOT control) at `pop_size=1` under `replace/mu-plus-lambda`."""
        return f_new <= f_old

    def run(self, problem, budget, seed=0, pop_size=1, log_dir=None,
            run_id=0, name=None):
        """Same contract as `Algorithm.run()` (including `run_id`/`name`,
        forwarded unchanged -- final-review fixes 2/4), with `pop_size`
        defaulting to (and REQUIRED to remain) 1 -- see the class
        docstring. Also RESETS this instance's tracked current-point state
        (`current_x`/`current_f`) at the START of every call, so the
        SAME `LocalSearch` instance can be `run()` more than once (e.g.
        across seeds) without leaking state from a previous run."""
        if pop_size != 1:
            raise ValueError(
                "LocalSearch requires pop_size=1 (a single search "
                f"trajectory) -- got pop_size={pop_size}")
        self.current_x = None
        self.current_f = None
        return super().run(problem, budget, seed=seed, pop_size=pop_size,
                            log_dir=log_dir, run_id=run_id, name=name)

    def generate(self, pop, ctx):
        """NOT meant to be overridden by a `LocalSearch` subclass
        (override `neighbor`/`accept` instead). See the class
        docstring's ACCEPT() DESIGN note for the full reasoning; summary:

        First call (`self.current_x is None`): adopts the initial
        population's own single member
        (`pop.individuals[0]`/`pop.fitness[0]`, produced by whatever
        `initialize()` path is in effect -- the inherited Rust
        `init/uniform` default, unless this subclass itself overrides
        `initialize()`) as the starting anchor.

        Every later call: `pop.individuals[0]`/`pop.fitness[0]` reflect
        `replace/mu-plus-lambda`'s own already-greedy choice between the
        PREVIOUS `current_x` and the neighbor this method proposed last
        call (see the class docstring) -- calls
        `self.accept(self.current_f, pop.fitness[0], ctx)`; if `True`,
        ADVANCES this instance's own anchor to match
        (`self.current_x, self.current_f = pop.individuals[0],
        pop.fitness[0]`); if `False`, leaves the anchor unchanged (the
        next `neighbor()` is proposed from the SAME point again).

        Either way, returns exactly one offspring:
        `[self.neighbor(self.current_x, ctx)]`.
        """
        if self.current_x is None:
            self.current_x = pop.individuals[0]
            self.current_f = float(pop.fitness[0])
        else:
            f_candidate = float(pop.fitness[0])
            if self.accept(self.current_f, f_candidate, ctx):
                self.current_x = pop.individuals[0]
                self.current_f = f_candidate
        return [self.neighbor(self.current_x, ctx)]
