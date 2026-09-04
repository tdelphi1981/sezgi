# Class hierarchy

sezgi's Python surface has three independent authoring paths into the same
Rust engine, plus one large family of ready-to-use wrapper classes. They
are **not** one inheritance tree — `AskTellAlgorithm` in particular is a
parallel base, not a subclass or superclass of `Algorithm`, because it
drives a completely different execution model (see
[The Python-callback bridge](rng-and-determinism.md#the-python-callback-bridge)
below for why).

## The four families

- **`sezgi.Problem`** (`problem.py`) — an ABC you subclass to declare a
  search space (`space()`) and an objective (`evaluate(x)`); optionally
  override `optimum()`/`batch_evaluate()`. `sezgi.recipes.FeatureSelection`
  and `sezgi.recipes.MixedTuning` (`recipes.py`) are its two concrete
  subclasses shipped in this milestone.
- **`sezgi.Algorithm`** (`algorithm.py`) — an ABC whose `generate(pop,
  ctx)` hook (required) runs *inside* the Rust engine loop, once per stage
  per generation, via `_sezgi.solve_with_py_generator`'s `PyGenerator`
  bridge. Two family bases sit below it: `PopulationAlgorithm` (adds
  `select`/`vary`, composing them into `generate`) and `LocalSearch` (adds
  `neighbor`/`accept`, pinned to `pop_size=1`).
- **`sezgi.AskTellAlgorithm`** (`algo.py`, formerly the top-level
  `sezgi.Algorithm` before this milestone's name repurposing) — an ABC
  whose `setup(ctx)`/`step(ctx)` hooks drive a pure-Python loop *over* the
  ask/tell `AlgoContext`/`EvalSession` core, entirely outside the Rust
  engine's own generation loop. Population bookkeeping, replacement, and
  termination are all the subclass's own responsibility here — the engine
  only supplies evaluation counting, budget enforcement, and a
  deterministic RNG stream.
- **The 29 built-in wrapper classes** (`builtins.py`) — one per
  `crates/components/src/presets.rs` preset builder (34 builders; some
  presets, like DE's three variants, are collapsed under one wrapper via a
  `variant=` kwarg), plus `NSGA2` (a thin `sezgi.mo.nsga2` skin, not a
  `presets.rs` builder). Every one of them is a plain `object` subclass,
  not a `Problem`/`Algorithm` subclass at all — `__init__(pop_size=...,
  **preset_kwargs)` then `run(problem, budget, seed=0, ...)`, internally
  building an `AlgorithmSpec` and calling `sezgi.solve()` (the same
  compat-layer entry point `sezgi.Algorithm.run` and `AskTellAlgorithm.
  solve` both funnel their own results through, via the shared
  `_wrap_result` helper in `algo.py`).

All four families converge on one result shape: `sezgi.algo.SolveResult`
(`algo` name, `seed`, `budget`, `evals_used`, `best_x`, `best_f`, `f_opt`,
`gap`) — there is exactly one result type across the whole class-first
surface, by design.

## As a class diagram

```mermaid
classDiagram
    class Problem {
        <<ABC>>
        +evaluate(x) float
        +space() Space
        +optimum() float|None
        +batch_evaluate(xs) list~float~
    }
    Problem <|-- FeatureSelection
    Problem <|-- MixedTuning

    class Algorithm {
        <<ABC>>
        +generate(pop, ctx) iterable
        +initialize(n, ctx) iterable
        +validate_space(space)
        +run(problem, budget, seed, pop_size, ...) SolveResult
    }
    Algorithm <|-- PopulationAlgorithm
    Algorithm <|-- LocalSearch
    class PopulationAlgorithm {
        +select(pop, k, ctx) list
        +vary(parents, ctx) iterable
    }
    class LocalSearch {
        +neighbor(x, ctx) x
        +accept(f_old, f_new, ctx) bool
    }

    class AskTellAlgorithm {
        <<ABC, PARALLEL base -- NOT related to Algorithm>>
        +setup(ctx)
        +step(ctx)
        +solve(problem, budget, seed) SolveResult
    }

    class BuiltinWrapper {
        <<29 classes, e.g. GeneticAlgorithm,\nDifferentialEvolution, ParticleSwarm,\nRandomSearch, SimulatedAnnealing, NSGA2, ...>>
        +__init__(pop_size, **preset_kwargs)
        +run(problem, budget, seed, run_id, log_dir) SolveResult
    }

    class SolveResult {
        <<shared result shape>>
        algo
        seed
        budget
        evals_used
        best_x
        best_f
        f_opt
        gap
    }
    Algorithm ..> SolveResult : run() returns
    AskTellAlgorithm ..> SolveResult : solve() returns
    BuiltinWrapper ..> SolveResult : run() returns
```

<!-- Source: py-sezgi/python/sezgi/problem.py (`Problem` ABC); py-sezgi/python/sezgi/recipes.py (`FeatureSelection`, `MixedTuning`); py-sezgi/python/sezgi/algorithm.py (`Algorithm`, `PopulationAlgorithm`, `LocalSearch`); py-sezgi/python/sezgi/algo.py (`AskTellAlgorithm`, `SolveResult`, the module-level `Algorithm = AskTellAlgorithm` compat alias); py-sezgi/python/sezgi/builtins.py (the 29 wrapper classes, `_PRESET_TABLE`, `_run_spec`) -->

## The compat layer underneath

None of these four families is a reimplementation of the others — they are
four different front doors onto the *same* `sezgi.solve()` /
`_sezgi.solve_with_py_generator()` compat internals, which in turn build an
`AlgorithmSpec` and hand it to the one `Engine::run` loop described in
[Engine flow](engine-flow.md). See
[Solve / compat internals](../api/compat.md) for that layer's own reference.

## Next

- [Determinism and the RNG model](rng-and-determinism.md) explains how
  `Algorithm.run`'s `generate()` callback draws from the *exact same*
  seeded stream a built-in Rust `Generator` would have used at that call.
