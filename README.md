# sezgi

**sezgi** (Turkish for "intuition") is a Rust-core, component-based metaheuristic
optimization library with Python and R frontends. Design doc: `docs/superpowers/specs/2026-08-27-sezgi-design.md`.

## Statement of need

Metaheuristic-optimization research has a reproducibility problem: RNG
draw order and operator details are rarely pinned precisely enough for a
third party to reproduce a run byte-for-byte, and a large share of
"novel" nature-inspired algorithms turns out, on close inspection, to be
a known method with new vocabulary (see the equivalence-critique
literature cited throughout `examples/README.md`'s algorithm catalog).
sezgi exists for optimization researchers who need cross-language,
bit-exact reproducibility; students who want a library that teaches the
field's actual structure rather than hiding it; and practitioners who
want a dependable, class-first library rather than a loose collection of
reference scripts. One Rust engine drives both the Python and R
frontends, so a shared seed reproduces byte-identical runs across
languages. See `docs/site/about/statement-of-need.md` (or the built
docs site) for the full statement, including a state-of-the-field
comparison to pymoo, jMetal, and ecr.

## Documentation

The full documentation site (installation, a 6-page Learn track, 8
executed tutorials, architecture diagrams, the auto-generated Python API
reference, and a 12-entry examples gallery) is built locally with MkDocs:

    uv pip install --python py-sezgi/.venv/bin/python \
      mkdocs==1.6.1 mkdocs-material==9.7.7 mkdocstrings==1.0.6 \
      mkdocstrings-python==2.0.8 markdown-exec==1.12.3 \
      pymdown-extensions==11.0.2 matplotlib==3.11.1
    py-sezgi/.venv/bin/mkdocs build --strict   # or: ... serve
    open site/index.html

See `CONTRIBUTING.md` for the full development setup and `docs/site/`
for the page sources.

## Quickstart (Python)

Every built-in algorithm is a class; every problem is either a native
handle (`sezgi.bbob(...)`, `sezgi.problems.onemax(...)`, ...) or a
`sezgi.Problem` subclass you author yourself — `.run()` accepts either:

    import sezgi

    problem = sezgi.problems.onemax(100)   # diagnostic Binary-space problem
    result = sezgi.GeneticAlgorithm(pop_size=20).run(problem, budget=400, seed=7)
    print(result.evals_used, result.best_f)

Output (live-run):

    400 16.0

`GeneticAlgorithm` auto-dispatches on the problem's space kind (here,
Binary → `presets.ga_bin`, bit-identically — see "Built-in algorithm
classes" below); every wrapper's `.run()` returns the same `SolveResult`
dataclass (`algo, seed, budget, evals_used, best_x, best_f, f_opt, gap`)
regardless of which class produced it. `sezgi.solve()`/`presets.*` still
work underneath, unchanged — see "Internals & spec files" below.

Your own problem, by subclassing `sezgi.Problem` (see "Define your own
problem" below for the full space-builder table):

    import sezgi

    class Sphere(sezgi.Problem):
        def __init__(self, n=3, lo=-5.0, hi=5.0):
            self.n, self.lo, self.hi = n, lo, hi

        def space(self):
            return sezgi.Float(self.lo, self.hi, self.n)

        def evaluate(self, x):
            return sum(v * v for v in x)

    result = sezgi.GeneticAlgorithm(pop_size=20).run(Sphere(n=3), budget=500, seed=1)
    print(f"evals_used={result.evals_used} best_f={result.best_f:.6g}")

Output (live-run):

    evals_used=500 best_f=0.005517

## Author your own algorithm (Python, class-first) (M4-1)

Subclass `sezgi.Algorithm` and override `generate(self, pop, ctx)` — it
runs INSIDE the Rust engine loop as a real `Generator` component (a
captured Python callback registered into the per-call component registry,
NOT a Python-owned ask/tell loop): the engine still owns the budget,
boundary repair, evaluation, and logging. `pop.individuals`/`pop.fitness`
are read-only views of the current population (`pop.individuals[i]` uses
the same bare/tuple convention `sezgi.Problem.evaluate(x)` does);
`ctx.rng` draws from the SAME house `RngStream` a Rust generator would use
at that call, so a Python-authored algorithm is exactly as deterministic
and reproducible as a built-in one — same seed, byte-identical run.
`generate` is the only required hook; `initialize`/`validate_space` are
optional (default: the engine's own `init/uniform`, no build-time veto):

    import sezgi

    class RandomMutation(sezgi.Algorithm):
        """Every offspring is a copy of a random parent with one perturbed coordinate."""

        def generate(self, pop, ctx):
            offspring = []
            for x in pop.individuals:
                i = ctx.rng.next_below(len(x))
                y = list(x)
                y[i] += ctx.rng.next_f64() - 0.5
                offspring.append(y)
            return offspring

    result = RandomMutation().run(sezgi.bbob(fid=1, dim=5, instance=1),
                                   budget=2000, seed=42, pop_size=20)
    print(f"evals_used={result.evals_used} best_f={result.best_f:.6g} "
          f"gap={result.gap:.6g}")

Output (live-run):

    evals_used=2000 best_f=-125.949 gap=0.000604428

`sezgi.PopulationAlgorithm` (adds `select(pop, k, ctx)`/`vary(parents,
ctx)`, default `select` a seeded binary tournament) and `sezgi.LocalSearch`
(adds `neighbor(x, ctx)`/`accept(f_old, f_new, ctx)`, default `accept`
greedy, runs at `pop_size=1`) are two family bases over the same
`Algorithm` — override only the hook that differs, exactly like overriding
only ONE of pymoo's `sampling`/`selection`/`crossover`/`mutation`/
`survival` components (see `docs/DECISIONS.md`'s M4-1 record for the
attribution). `examples/python/oop/custom_de_variant.py` overrides only
`vary()` on `PopulationAlgorithm` (a ~14-line DE/rand/1-shaped mutation);
`examples/python/oop/custom_local_search.py` overrides only `neighbor()`
on `LocalSearch`. **`LocalSearch.accept()` cannot express true SA-style
"sometimes accept a worse move"** — the default `replace/mu-plus-lambda`
replacer has already filtered out any worse candidate before Python ever
sees the population again at `pop_size=1`; `accept()` controls whether
this base's OWN search anchor advances to the engine's already-decided
outcome, not what the engine's population itself contains.

Spec persistence is deliberately absent on this path: a Python-authored
algorithm is a live callback captured per-call by the Rust component
registry — process-local, with no wire format `AlgorithmSpec::to_toml()`
can express. `sezgi.solve()`/`presets.*` remain the only path that
produces a persistable, shareable spec file (see "Internals & spec files"
below).

## Data recipes: feature selection (M4-1)

`sezgi.recipes.FeatureSelection(X, y, scorer, penalty=0.0)` is a `Problem`
subclass wrapping a binary-mask feature-selection objective: `space()` is
`Binary(n_features)`, `evaluate(mask)` is `scorer(X[:, mask], y) +
penalty * popcount / n_features` (minimize; an empty mask returns `+inf`
without ever calling `scorer`). Data enters through the objective — no ML
framework dependency; `scorer` is any callable `(X_sub, y) -> float`.
Paired with `GeneticAlgorithm`'s Binary auto-dispatch, this recovers a
known informative-column subset from a synthetic dataset
(`examples/python/oop/feature_selection.py`: a fixed 20x8 matrix, 3
informative columns, an OLS-residual-sum-of-squares scorer, `penalty=0.3`
so the penalty — not just the raw score — is what makes the 3-column
subset the actual global minimum):

    ./py-sezgi/.venv/bin/python examples/python/oop/feature_selection.py

Output (live-run):

    feature_selection (oop): evals_used=200 best_f=0.3607495483 popcount=3 mask=01010010 recovered=True

`mask=01010010` sets exactly bits 1, 3, 6 — the dataset's own informative
columns; `recovered=True` confirms the search found the unique global
minimum (independently cross-checked by a full 256-mask brute-force
enumeration in `py-sezgi/tests/test_oop_recipes.py`).
`sezgi.recipes.MixedTuning(space, objective)` is the general-purpose
sibling — a one-line `Problem` binding an arbitrary objective over an
arbitrary declared space (the "tune anything" door); a genuinely Mixed
space through it still needs the `gen/compound` hand-spec workaround
`GeneticAlgorithm` itself needs (see "Built-in algorithm classes" below).

## Define your own problem: Problem subclassing (M4-1)

`sezgi.Problem` is an ABC over the same callable-problem bridge
`from_callable` has always used, widened beyond Float (see the Sphere
example in "Quickstart" above): `evaluate(self, x)` and `space(self)` are
the two required hooks; `optimum(self)` (default `None`) and
`batch_evaluate(self, xs)` (default: loop `evaluate`) are optional.
`space()` returns one of five block builders, or a `sezgi.Space(*blocks)`
composing several of them:

| Builder | Block kind | `x` type passed to `evaluate` |
|---|---|---|
| `sezgi.Float(lo, hi, n)` | Float | `list[float]` |
| `sezgi.Int(lo, hi, n)` | Int | `list[int]` |
| `sezgi.Categorical(k, n)` | Categorical | `list[int]` (category indices `0..k`, not labels) |
| `sezgi.Binary(n)` | Binary | `list[bool]` |
| `sezgi.Permutation(n)` | Permutation | `list[int]` |

A single-block space passes `x` bare (that block's own value); a
multi-block `Space(...)` passes `x` as a `tuple` of per-block values, in
declared order — the same convention `pop.individuals[i]` uses in the
engine-hosted `Algorithm.generate` hook above.
`sezgi.as_native_problem(obj)` accepts either a `Problem` subclass instance
or a native handle (`sezgi.bbob(...)`, `sezgi.problems.onemax(...)`, ...),
so every built-in class and every user-authored `Algorithm` interoperate
with both kinds of problem uniformly. It is used internally by every scalar
wrapper class's and `Algorithm`'s (and its `PopulationAlgorithm`/
`LocalSearch` family bases') `.run()`, and by `sezgi.solve()` /
`AskTellAlgorithm.solve()` (both route their `problem` argument through it
too) — the one exception is `NSGA2.run`, which takes `mo.nsga2`'s problem
STRINGS (`"zdt1"`, ...), not a handle or a `Problem` subclass, and so never
calls `as_native_problem` at all (see "Built-in algorithm classes" below).

## Built-in algorithm classes (M4-1)

Every preset in `presets.rs` also has a configurable class:
`__init__(pop_size=..., **preset_kwargs)` (the preset's own kwargs pass
through unchanged), `.run(problem, budget, seed=0, run_id=0, log_dir=None)
-> SolveResult` — proven bit-identical to the equivalent
`sezgi.solve(presets.X(...), problem, master_seed=seed, ...)` call at the
same seed (the wrapper adds nothing; it is a class SKIN, not a new
algorithm):

| Class | Preset(s) |
|---|---|
| `GeneticAlgorithm` | `ga_real`/`ga_perm`/`ga_bin`/`ga_int`/`ga_cat` (space-kind auto-dispatch; `representation=` overrides) |
| `DifferentialEvolution` | `de_rand_1`/`de_best_1`/`jde` (`variant=`) |
| `EvolutionStrategy`, `ParticleSwarm`, `SimulatedAnnealing`, `SHADE`, `LSHADE`, `CMAES`, `CMAESIpop`, `NelderMead`, `RandomSearch` | one preset each |
| `GreyWolfOptimizer`, `WhaleOptimization`, `HarmonySearch`, `CuckooSearch`, `GrasshopperOptimization`, `SineCosineAlgorithm`, `JAYA`, `MothFlameOptimization`, `SalpSwarm`, `FireflyAlgorithm`, `BatAlgorithm`, `FlowerPollination`, `TLBO`, `HarrisHawks`, `AntLion`, `ArtificialBeeColony`, `GravitationalSearch` | one preset each (all 17 labeled-metaphor algorithms) |
| `NSGA2` | `mo.nsga2`/`nsga2_run` — **run-only, no `generate()` override point** (MO authoring is out of scope this milestone) |

`GeneticAlgorithm` auto-dispatches on the problem's space kind
(Float/Permutation/Binary/Int/Categorical → `ga_real`/`ga_perm`/`ga_bin`/
`ga_int`/`ga_cat`); a genuinely **Mixed space is rejected** with a
`NotImplementedError` naming `gen/compound` (the hand-spec workaround, see
"Typed operators, mixed spaces, and diagnostic problems (M3-8)" above) —
`representation=` forces a single-kind preset instead of introspecting the
space. `NSGA2(pop_size=..., **nsga2_kwargs).run(problem, dim, budget, m=,
k=, l=, ...)` mirrors `mo.nsga2`'s own contract exactly and returns
`mo.nsga2`'s own dict shape (`individuals`/`objectives`/`front0`/
`evals_used`/`violations`), NOT `SolveResult` — a Pareto front has no
single "best point" for `SolveResult`'s fields to describe.

## Internals & spec files: solve() and presets.* (compat)

`sezgi.solve(spec, problem, master_seed=..., ...)` and `sezgi.presets.*`
are the ORIGINAL surface every class above builds on, still fully
supported — every built-in wrapper's `.run()` assembles a preset's JSON
spec and calls `sezgi.solve()` internally, unchanged. Reach for this path
directly when you need:

- a **spec file** (TOML/JSON) you can save, diff, or hand-author component
  by component (`[[stages]]`, `gen/...`, `replace/...` — see "Typed
  operators, mixed spaces, and diagnostic problems (M3-8)" above for the
  `gen/compound` mixed-space example). A Python-authored `sezgi.Algorithm`
  has **no such spec**: it is a live callback captured per-call by the Rust
  component registry, process-local, with no wire format `to_toml()`/
  `to_json()` can express — the class-first `.run()` path never exposes
  spec persistence for a Python component, deliberately (honest degrade,
  not a gap to be closed).
- **benchmarking/experiments** — `sezgi.run_experiment`, IOH logging,
  `per_budget_packages`, bias scanning all consume a spec/preset, not a
  class instance; every section below this one (Experiments & Statistics,
  Anytime analysis, Bias scanning, Multi-objective, CEC suites, TSP,
  Typed operators) is written against `solve()`/`presets.*` directly,
  unchanged by this milestone.
- **R spec files** — R has its own mirror of this class-first surface
  (M4-2 — see "Author your own algorithm (R, class-first)" below), built
  over the SAME `sz_solve_*`/`sz_preset_*` compat internals; `sz_solve_*`/
  `sz_preset_*`/`sz_algorithm` remain R's own persistable-spec path (see
  "Internals & spec files (R)" and "Write your own algorithm (R) (M3-5)"
  below).

`solve()`/`presets.*` are not deprecated and are not scheduled for removal
— they stay compat internals for spec files, benchmarking, and R, per
`docs/DECISIONS.md`'s M4-1 record.

## Experiments & Statistics (M2c)

Run multiple algorithms across multiple problems, seeds, and budgets with a TOML grid; get comparative statistics:

    import sezgi
    
    spec_toml = """
        name = "example"
        seeds = [1, 2, 3]
        budgets = [1000, 5000]
        
        [[algorithms]]
        name = "de"
        preset = { kind = "de_rand_1", pop_size = 20 }
        
        [[algorithms]]
        name = "cmaes"
        preset = { kind = "cmaes", pop_size = 20 }
        
        [[problems]]
        suite = "bbob"
        fid = 1
        dim = 10
        instances = [1, 2, 3, 4, 5]
    """
    
    # Run experiment (TOML grid × seeds × budgets × instances)
    records = sezgi.run_experiment(spec_toml, parallel=True)

    # One paper_package PER DISTINCT BUDGET present in records, in ascending
    # budget order — no hand-rolled filtering/aggregation needed. At least 5
    # problems is recommended for meaningful comparisons (the exact Wilcoxon
    # test now handles fewer if no ties or zeros are present).
    packages = sezgi.per_budget_packages(records, rope=0.01, samples=10000, seed=42)

    for budget, pkg in packages:
        print(f"--- budget={budget} ---")
        print(pkg["latex_summary"])  # Friedman ranks + pairwise Wilcoxon–Holm

`per_budget_packages` reports statistics per budget, never pooled across
budgets: rankings can flip between small and large budgets (Piotrowski et al.
2025), so each budget gets its own package rather than one arbitrarily-chosen
budget standing in for all of them. `sezgi.results_matrix(records, budget)`
is available separately if you need just the `(algo_names, problem_labels,
matrix)` triple for one budget (e.g. to feed a custom analysis).

## Anytime analysis: IOH logs, ECDF, COCO export (M2d-2)

Pass `log_dir=` to `run_experiment` to also write an IOH-profiler-format
log tree (readable directly by IOHinspector/IOHanalyzer) alongside the
in-memory records, then read anytime-performance curves, export a
COCO/BBOB archive, or reconstruct records straight off disk.

IOH logging supports only a SINGLE budget: the on-disk archive records no
budget, so it cannot tell apart two runs of the same `(instance, seed)`
logged at different budgets. `run_experiment(..., log_dir=...)` (and the
`log_dir` pass-through on the checkpoint path) raises `ValueError` naming
"multiple budgets" if `spec_toml` declares more than one. Log at the single
largest budget you need instead, and derive any smaller budgets on read via
`read_ioh_records`:

    import sezgi

    # Multi-budget specs (like `spec_toml` above) can't be logged directly --
    # log at the largest budget only, then derive [1000, 5000] on read.
    single_budget_toml = spec_toml.replace("budgets = [1000, 5000]", "budgets = [5000]")

    records = sezgi.run_experiment(single_budget_toml, log_dir="logs/", parallel=False)

    # ECDF (anytime performance) curves, one per algorithm by default.
    curves = sezgi.ecdf("logs/")
    for algo, curve in curves:
        print(algo, curve["evals"][-1], curve["proportion"][-1])

    # COCO/BBOB "old format" export, ready for cocopp post-processing.
    written = sezgi.coco_export("logs/", "coco_out/")

    # Reconstruct RunRecords directly from the on-disk archive -- no
    # in-memory `records` object required -- deriving BOTH original budgets
    # from the single largest-budget archive that was actually logged.
    disk_records = sezgi.read_ioh_records("logs/", [1000, 5000])

    # Same record-dict shape run_experiment returns, so it feeds straight
    # into per_budget_packages (or results_matrix) unchanged.
    packages = sezgi.per_budget_packages(disk_records, rope=0.01, samples=10000, seed=42)

This makes the on-disk IOH archive a first-class, self-contained
alternative to the in-memory `records` list: a single `log_dir=` run today
can be re-analyzed later (different targets, a different budget subset, a
COCO export for a separate tool) without re-running any algorithm, by
reading `logs/` back with `read_ioh_records`/`ecdf`/`coco_export` alone.

R mirrors this exactly with `sz_` names: `sz_run_experiment(spec_toml,
log_dir = "logs/")` writes the same IOH tree, `sz_ecdf("logs/")` returns
the same per-algorithm curves as a named list, `sz_coco_export("logs/",
"coco_out/")` writes the same COCO archive, and `sz_read_ioh_records("logs/",
2000)` reconstructs the same data.frame shape `sz_run_experiment` returns,
feeding directly into `sz_per_budget_packages`/`sz_results_matrix`.

## Quickstart (R)

Install from the repo root (the Rust core builds via `cargo` on install):

    R CMD INSTALL r-sezgi

Then:

    library(sezgi)

    spec <- sz_preset_de_rand_1(pop_size = 50, budget = 20000)
    result <- sz_solve_bbob(spec, fid = 1L, dim = 10L, instance = 1L,
                             master_seed = 42, run_id = 0)
    print(result$best_f)

Experiments and statistics mirror the Python bindings exactly (same TOML
grid schema, same checkpoint/resume semantics, bit-identical results for the
same spec and seed):

    spec_toml <- '
        name = "example"
        seeds = [1, 2, 3]
        budgets = [1000, 5000]

        [[algorithms]]
        name = "de"
        preset = { kind = "de_rand_1", pop_size = 20 }

        [[algorithms]]
        name = "cmaes"
        preset = { kind = "cmaes", pop_size = 20 }

        [[problems]]
        suite = "bbob"
        fid = 1
        dim = 10
        instances = [1, 2, 3, 4, 5]
    '

    records <- sz_run_experiment(spec_toml, parallel = TRUE)

    # One paper_package PER DISTINCT BUDGET present in records, named by
    # budget, in ascending budget order.
    packages <- sz_per_budget_packages(records, rope = 0.01, samples = 10000, seed = 42)

    for (budget_name in names(packages)) {
      cat("--- budget=", budget_name, " ---\n", sep = "")
      cat(packages[[budget_name]]$latex_summary, "\n")
    }

`sz_results_matrix(records, budget)` is available separately if you need
just the `(algo_names, problem_labels, matrix)` triple for one budget.

R also has a class-first surface now (M4-2, mirroring Python's — see
"Author your own algorithm (R, class-first)" below): every preset has a
configurable R6 class, and `sezgi::Problem`/`sezgi::Algorithm` are
subclassable bases. `sz_solve_*`/`sz_preset_*`/`sz_algorithm` above remain
fully supported — see "Internals & spec files (R)" further down.

## Author your own algorithm (R, class-first) (M4-2)

Subclass `sezgi::Algorithm` (an R6 base, via `R6::R6Class(..., inherit =
Algorithm, ...)`) and override `generate(pop, ctx)` — like Python's
`sezgi.Algorithm`, it runs INSIDE the Rust engine loop as a real
`Generator` component (an R closure captured into the per-call component
registry, not an R-owned loop): the engine still owns the budget, boundary
repair, evaluation, and logging. `pop` is `list(x, f)`: `f` is the current
population's fitness (a numeric vector); `x` is an n×dim numeric **matrix**
(rows = individuals) for a single-Float-block space — the common,
ergonomic case — and falls back to a plain `list` of per-individual values
(the same bare/list convention `Problem$evaluate(x)` uses) for every other
space shape. **This `pop$x`-is-a-matrix convenience is a deliberate
R-idiomatic divergence from Python's `PopView`, which is always a plain
list** — see `r-sezgi/R/algorithm.R`'s own class doc for the exact rule.
`ctx$rng` (`next_f64()`, `next_below(n)`, `split(child_id)`) draws from the
SAME house `RngStream` a Rust generator would use, so an R-authored
algorithm is exactly as reproducible as a built-in one — same seed,
byte-identical run. `generate` is the only required hook;
`initialize_population`/`validate_space` are optional (default: the
engine's own `init/uniform`, no build-time veto):

    library(sezgi)

    RandomMutation <- R6::R6Class("RandomMutation",
      inherit = Algorithm,
      public = list(
        generate = function(pop, ctx) {
          n <- nrow(pop$x); d <- ncol(pop$x)
          out <- pop$x
          for (i in seq_len(n)) {
            j <- ctx$rng$next_below(as.double(d)) + 1
            out[i, j] <- out[i, j] + ctx$rng$next_f64() - 0.5
          }
          out
        }
      )
    )

    result <- RandomMutation$new()$run(sz_builtin_bbob(1, 5, 1),
                                        budget = 2000, seed = 42, pop_size = 20)
    cat(sprintf("evals_used=%d best_f=%.6g\n", result$evals_used, result$best_f))

Output (live-run):

    evals_used=2000 best_f=-125.949

**Naming divergence (RULING 8):** the population-initializer hook is
`initialize_population(n, space, ctx)`, **not** `initialize` — R6 reserves
`$initialize` for the constructor, so Python's `initialize(n, ctx)` hook
name cannot be reused here. This is a deliberate, documented divergence
from the Python surface, not an oversight.

`sezgi::PopulationAlgorithm` (adds `select(pop, k, ctx)`/`vary(parents,
ctx)`, default `select` a seeded binary tournament) and
`sezgi::LocalSearch` (adds `neighbor(x, ctx)`/`accept(f_old, f_new, ctx)`,
default `accept` greedy, runs at `pop_size = 1`) are two family bases over
the same `Algorithm`, mirroring Python's identical two bases (same
pymoo/jMetal-modeled hook taxonomy — see `docs/DECISIONS.md`'s M4-1
record for the attribution). `examples/r/oop/custom_de_variant.R`
overrides only `vary()` on `PopulationAlgorithm`; `examples/r/oop/
custom_local_search.R` overrides only `neighbor()` on `LocalSearch`. Like
Python, **`LocalSearch$accept()` cannot express true SA-style "sometimes
accept a worse move"** — the default `replace/mu-plus-lambda` replacer has
already filtered out any worse candidate before R ever sees the population
again at `pop_size = 1`; see `r-sezgi/R/algorithm.R`'s own class doc
("ACCEPT() DESIGN") for the full structural argument.

**`log_dir` is honestly rejected, not silently ignored.** Unlike Python
(where `Algorithm.run(..., log_dir=...)` wires IOH logging through),
r-sezgi's engine-hosted generator bridge (`sz_solve_r_generator`/
`sz_solve_r_generator_bbob`) has no `log_dir` parameter at all — there is
no R-side logging path to wire `run()`'s own `log_dir` through this
milestone. A non-`NULL` `log_dir` raises a clear error naming the gap
rather than being silently swallowed.

**Only one built-in problem descriptor exists: `sz_builtin_bbob(fid, dim,
instance = 1)`.** r-sezgi has no general native-problem-handle type the
way Python's `as_native_problem()` does, so `Algorithm$run()`'s `problem`
argument is EITHER an `sz_builtin_bbob()` descriptor OR a `Problem`
subclass instance (see "Define your own problem" below) — no
CEC2014/CEC2017/CEC2022/TSP/onemax native form exists for the class
surface. **`sz_builtin_bbob()` also has no known-optimum accessor**, so
`f_opt`/`gap` on the returned `sz_result` are always `NULL` for that path
(`gap` was omitted from the example above for exactly this reason) —
unlike Python's `sezgi.bbob(...)`, which does carry a known optimum. A
`Problem` subclass's own `optimum()` is used for `f_opt`/`gap` on the
other path.

Spec persistence is deliberately absent on this path, same as Python: an
R-authored algorithm is a live callback captured per-call by the Rust
component registry — process-local, with no wire format
`AlgorithmSpec::to_toml()`/`to_json()` can express. `sz_solve_*`/
`sz_preset_*`/`sz_algorithm` remain the only path that produces a
persistable, shareable spec file (see "Internals & spec files (R)" below).

## Data recipes: feature selection (R) (M4-2)

`FeatureSelection$new(X, y, scorer, penalty = 0)` (an R6 class, inherits
`Problem`) mirrors `sezgi.recipes.FeatureSelection` exactly: `space()` is
`sz_binary(ncol(X))`, `evaluate(mask)` is `scorer(X[, mask, drop = FALSE],
y) + penalty * (popcount / n_features)` (minimize; an empty mask returns
`Inf` without ever calling `scorer`). Data enters through the objective —
no ML framework dependency; `scorer` is any function `(X_sub, y) ->
numeric(1)`. Paired with `GeneticAlgorithm`'s Binary auto-dispatch, this
recovers a known informative-column subset from a synthetic dataset
(`examples/r/oop/feature_selection.R`: an independently-derived — not
copied from the Python twin's own seed/columns — fixed 20×8 matrix, 3
informative columns, an OLS-residual-sum-of-squares scorer, `penalty =
0.3`):

    Rscript examples/r/oop/feature_selection.R

Output (live-run):

    feature_selection (oop): evals_used=200 best_f=0.6562182758 popcount=3 mask=01010010 recovered=TRUE

`mask=01010010` sets exactly bits 2, 4, 7 (1-based R column indices) — the
dataset's own informative columns; `recovered=TRUE` confirms the search
found the unique global minimum (independently cross-checked by a full
256-mask brute-force enumeration in `r-sezgi/tests/testthat/
test-oop-recipes.R`). `best_f` differs from the Python twin's own number —
the two datasets are independent derivations (base-R `rnorm()` vs. numpy),
not a shared fixture, so no cross-language numeric anchor is claimed here.

`MixedTuning$new(space, objective)` is the general-purpose sibling — a
one-line `Problem` binding an arbitrary objective over an arbitrary
declared space. A single-kind (e.g. Float-only) space runs through
`GeneticAlgorithm`'s own auto-dispatch directly; a genuinely **Mixed**
space (more than one distinct block kind) has no `ga_*` preset to
dispatch to, so it needs a hand-built `gen/compound` spec passed to
`sezgi:::sz_solve_r_problem()` directly — the same workaround
`GeneticAlgorithm` itself needs (see "Built-in algorithm classes (R)"
below). **`sz_solve_r_problem()`'s `spec_json` parameter accepts JSON
only** (not TOML) — this is specific to that one entry point, not a
statement about R generally: `sz_solve_mixed_diagnostic` (M3-8) is a
pre-existing TOML-accepting export elsewhere in r-sezgi. The hand-built
spec for a Mixed `MixedTuning` run is therefore written as an equivalent
JSON literal, not TOML.

## Define your own problem: Problem subclassing (R) (M4-2)

`sezgi::Problem` is an R6 base over the new R-callable problem bridge
(`sz_solve_r_problem`, closing the long-standing "R callable-objective
sessions" deferral — see `docs/DECISIONS.md`'s M4-2 record): subclass it
and override `evaluate(x)` and `space()` (both required — the defaults
`stop()` with a "not implemented" message); `optimum()` (default `NULL`)
and `batch_evaluate(xs)` (default: loop `evaluate`, called ONCE PER
GENERATION with the whole population, mirroring Python's own vectorized
default) are optional. `space()` returns one of five block builders, or an
`sz_space(...)` composing several of them:

| Builder | Block kind | `x` type passed to `evaluate` |
|---|---|---|
| `sz_float(lo, hi, n)` | Float | numeric (double) vector |
| `sz_int(lo, hi, n)` | Int | integer vector |
| `sz_categorical(k, n)` | Categorical | integer vector (category indices `0..k`, not labels) |
| `sz_binary(n)` | Binary | logical vector |
| `sz_permutation(n)` | Permutation | integer vector, **0-based** (not R's usual 1-based convention) |

A single-block space passes `x` bare (that block's own value); a
multi-block `sz_space(...)` passes `x` as an (unnamed) `list` of per-block
values, in declared order — the same convention `Algorithm$generate`'s
`pop$x` (list form) uses. `sz_as_problem(obj)` gives a friendly error for
a non-`Problem` argument; unlike Python's `as_native_problem()`, r-sezgi
has no separate native-problem-handle type to convert to, so a `Problem`
subclass instance already IS what the bridge needs:

    library(sezgi)

    Sphere <- R6::R6Class("Sphere",
      inherit = Problem,
      public = list(
        n = 3,
        space = function() sz_space(sz_float(-5, 5, self$n)),
        evaluate = function(x) sum(x^2)
      )
    )

    result <- GeneticAlgorithm$new(pop_size = 20)$run(Sphere$new(), budget = 500, seed = 1)
    cat(sprintf("evals_used=%d best_f=%.6g\n", result$evals_used, result$best_f))

Output (live-run):

    evals_used=500 best_f=0.005517

## Built-in algorithm classes (R) (M4-2)

Every preset in `presets.rs` also has a configurable R6 class:
`$new(pop_size = ..., ...)` (the preset's own kwargs pass through
unchanged), `$run(problem, budget, seed = 0, run_id = 0) -> sz_result` —
proven bit-identical to the equivalent `sz_solve_bbob(sz_preset_X(...),
...)`/`sz_solve_r_problem(sz_preset_X(...), ...)` call at the same seed
(the wrapper adds nothing; it is a class SKIN, not a new algorithm), and
returns the SAME 8-field `sz_result` shape (`algo, seed, budget,
evals_used, best_x, best_f, f_opt, gap`) `Algorithm$run()` and
`sz_algo_solve()` both return:

| Class | Preset(s) |
|---|---|
| `GeneticAlgorithm` | `ga_real`/`ga_perm`/`ga_bin`/`ga_int`/`ga_cat` (space-kind auto-dispatch; `representation=` overrides) |
| `DifferentialEvolution` | `de_rand_1`/`de_best_1`/`jde` (`variant=`) |
| `EvolutionStrategy`, `ParticleSwarm`, `SimulatedAnnealing`, `SHADE`, `LSHADE`, `CMAES`, `CMAESIpop`, `NelderMead`, `RandomSearch` | one preset each |
| `GreyWolfOptimizer`, `WhaleOptimization`, `HarmonySearch`, `CuckooSearch`, `GrasshopperOptimization`, `SineCosineAlgorithm`, `JAYA`, `MothFlameOptimization`, `SalpSwarm`, `FireflyAlgorithm`, `BatAlgorithm`, `FlowerPollination`, `TLBO`, `HarrisHawks`, `AntLion`, `ArtificialBeeColony`, `GravitationalSearch` | one preset each (all 17 labeled-metaphor algorithms) |
| `NSGA2` | `sz_nsga2()` — **run-only, no `generate()` override point** (MO authoring is out of scope this milestone) |

29 exported classes total (26 table-driven + `GeneticAlgorithm` +
`DifferentialEvolution` = 28 preset-backed, plus the run-only `NSGA2`
skin), reconciling all 34 exported `sz_preset_*` builders with zero
skipped and zero orphans — live-verified (`length(sezgi:::.sz_preset_table)
== 26`, `getNamespaceExports("sezgi")` filtered to `sz_preset_*` == 34).
`GeneticAlgorithm` auto-dispatches on the problem's space kind
(Float/Permutation/Binary/Int/Categorical); a genuinely **Mixed space is
rejected** with a clear error naming the limitation —
`representation=` forces a single-kind preset instead of introspecting the
space. `NSGA2$new(pop_size = ..., ...)$run(problem, dim, budget, m = ...,
k = ..., l = ..., ...)` delegates to `sz_nsga2()` VERBATIM and returns
`sz_nsga2()`'s own list shape (`individuals`/`objectives`/`front0`/
`evals_used`), **NOT** `sz_result` — a Pareto front has no single "best
point" for `sz_result`'s fields to describe.

**Problem-form support matrix, honestly stated: BBOB-only natively.**
Every class in this family accepts EITHER a `Problem` subclass instance
(routed through `sz_solve_r_problem()`) OR an `sz_builtin_bbob()`
descriptor (routed through `sz_solve_bbob()`) — the SAME two forms
`Algorithm$run()` accepts above. **28 of these 29 classes (everything
except `NSGA2`, which takes `sz_nsga2()`'s own problem-name strings) reach
only BBOB natively** — there is no CEC2014/CEC2017/CEC2022/TSP/onemax
native-descriptor path for this class family; a `Problem` subclass
wrapping one of those suites' own `sz_solve_*`/`sz_eval_session_*`
functions is the workaround. r-sezgi has no general native-problem-handle
type at all (unlike Python's `as_native_problem()`), so this scope is
narrower than the Python surface's own `sezgi.bbob(...)`/
`sezgi.problems.onemax(...)`/etc. built-in family:

    library(sezgi)

    result <- GreyWolfOptimizer$new(pop_size = 30)$run(
      sz_builtin_bbob(fid = 1, dim = 10, instance = 1), budget = 4000, seed = 7)
    cat(sprintf("evals_used=%d best_f=%.6g\n", result$evals_used, result$best_f))

Output (live-run):

    evals_used=3990 best_f=-84.342

## Internals & spec files (R): sz_solve_*/sz_preset_*/sz_algorithm (compat, extended M4-2)

Every class above is a skin over `sz_solve_*`/`sz_preset_*`, still fully
supported and unchanged — reach for them directly (or for
`sz_algorithm()`/`sz_algo_solve()`, the SEPARATE ask/tell scripting
surface, see "Write your own algorithm (R) (M3-5)" below) when you need a
**spec file** (TOML/JSON) to save, diff, or hand-author component by
component (see "Typed operators, mixed spaces, and diagnostic problems
(M3-8)" above for a `gen/compound` mixed-space example) — an R-authored
`Algorithm`/`Problem` subclass has no such spec, exactly like Python's
class surface (see "Author your own algorithm (R, class-first)" above).
`sz_run_experiment`, IOH logging, `sz_per_budget_packages`, bias scanning
(every section below this one) all consume a spec/preset, not a class
instance, and are entirely unaffected by this milestone. All 86
pre-existing exports keep their names and behavior unchanged (M4-2 ruling
5) — nothing here is deprecated or scheduled for removal.

## Bias scanning (M3-1)

`sezgi.bias`/`sz_bias_*` scans a preset spec for structural bias (does the
algorithm's own search operators pull final positions toward particular
regions of the domain, independent of the objective — the BIAS-toolbox
method, Kononova et al. 2015 / Vermetten, van Stein, Caraffini, Minku &
Kononova 2022) and center bias (does the algorithm perform suspiciously
better when the optimum sits at the domain center than when it sits at its
natural off-center location — the Kůdela method, *Nature Machine
Intelligence* 2022, pinned here via the author's own 2023 restatement,
arXiv:2301.01984). A one-call `bias.report`/`sz_bias_report` runs both scans
and renders a NaN-free LaTeX summary table:

    import sezgi

    spec = sezgi.presets.random_search(pop_size=5, budget=50)
    report = sezgi.bias.report(spec, dim=2, budget=50, seed=20260830,
                                structural_runs=30, central_fids=[1],
                                central_instances=[1], central_runs_per=5)
    print(report["latex_summary"])

Output (tiny budgets, for illustration — see `docs/DECISIONS.md`'s M3-1
"Method-provenance table" for the numbers a real scan should use):

    \begin{tabular}{llll}
    \toprule
    Test & Statistic & $p$ & Verdict \\
    \midrule
    Structural bias (KS, Holm-corrected) & 1.221e-1 & $1.0000$ & no evidence of structural bias \\
    Structural bias (AD, Holm-corrected) & 4.949e-1 & $1.0000$ & no evidence of structural bias \\
    Central bias (Wilcoxon) & 7.000e0 & $1.0000$ & no evidence of center-bias exploitation \\
    Signature (Rajwar-Deep) & -- & -- & not run: the Rajwar-Deep method could not be pinned from accessible sources \\
    \bottomrule
    \end{tabular}

`sezgi.bias.structural`/`sezgi.bias.central` are also available individually
(each returns the raw per-dimension KS/AD rows or the paired gap vectors plus
the Wilcoxon/Cliff's-delta decision, not just the rendered table). R mirrors
this with `sz_bias_structural`/`sz_bias_central`/`sz_bias_report`, same field
names:

    library(sezgi)

    spec <- sz_preset_random_search(5, 50)
    r <- sz_bias_structural(spec, dim = 2, budget = 50, runs = 30, seed = 20260830)
    cat("verdict:", r$verdict, "\n")
    cat("per_dim_ks[[1]]$p_value:", r$per_dim_ks[[1]]$p_value, "\n")

Output:

    verdict: no_evidence
    per_dim_ks[[1]]$p_value: 0.8931464

**Results are statistical evidence, not accusations.** A `NoEvidence`
verdict means this scan, at this config, found nothing — it is not proof the
algorithm is unbiased (absence of evidence is not evidence of absence), and
an `Evidence` verdict describes a measured statistical departure from
uniformity/parity, not a claim about the algorithm's intent or general
quality. A third planned test — the Rajwar-Deep Generalized Signature Test —
is deferred: its primary source is paywalled with no accessible preprint or
reference implementation, so `bias.report()["signature"]`/`sz_bias_report()$signature`
is always `None`/`NULL` until the method can be verified from a real source.
See `docs/DECISIONS.md`'s M3-1 record for the full method-provenance table,
pinned KS/AD formulas, and the deferral's search log.

## Multi-objective optimization (M3-2, extended M3-7)

The Python `sezgi.mo` namespace and the R `sz_nsga2`/`sz_mo_*` functions run NSGA-II (Deb, Pratap,
Agarwal & Meyarivan 2002) against the ZDT (Zitzler, Deb & Thiele 2000) and
DTLZ (Deb, Thiele, Laumanns & Zitzler 2005) test-problem suites, plus the
2-objective hypervolume and IGD quality indicators:

    import sezgi

    result = sezgi.mo.nsga2("zdt1", dim=10, pop_size=40, budget=4000, seed=20260830)
    front0 = [result["objectives"][i] for i in result["front0"]]
    ref_front = sezgi.mo.pareto_front("zdt1", dim=10, n=200)
    print(len(front0), result["evals_used"])
    print(sezgi.mo.hypervolume_2d(front0, [1.1, 1.1]))
    print(sezgi.mo.igd(front0, ref_front))

Output (live-run, same scenario as `examples/python/nsga2_zdt1.py`):

    40 4000
    0.8580576535101335
    0.01254540902919091

R mirrors this 1:1 with `sz_nsga2()`/`sz_mo_hypervolume_2d()`/
`sz_mo_igd()`/`sz_mo_pareto_front()`, same keys, same numbers (both
bindings call the same Rust core, so a same-seed run is bit-identical
across languages) — see `examples/r/nsga2_zdt1.R` for the full matched
example, including the R-idiomatic matrix inputs `hypervolume_2d`/`igd`
expect.

Honestly: NSGA-II ships as a self-contained, seeded reference runner over
a parallel `MoProblem`/`MoEvaluator` surface, not as a component graph you
assemble through an `ExperimentSpec` — the engine, `Ctx`, `Population`,
and every `Replacer`/`Adapter` are scalar-fitness-pinned surfaces, and
generalizing the executor to multi-objective fitness is v2-scale surgery
across all 25 presets, so MO component-graph spec integration is deferred
to v2 (see `docs/DECISIONS.md`'s M3-2 record for the full ruling).
`pop_size` must be a multiple of 4, not merely even — a KanGAL-faithful
tightening of the naive "even, >= 4" rule that NSGA-II's own reference C
code (`nsga2r.c`) enforces for its double-permutation tournament pairing.

### Multi-objective remainders (M3-7)

M3-7 closes every MO capability the M3-2 record above left deferred: a
constraint channel plus Deb's constrained-domination in NSGA-II (feasible
beats infeasible; among infeasible, smaller total violation wins; among
feasible, plain dominance — KanGAL `nsga2r.c`'s own `check_dominance`), a
binary genotype path (two-point crossover + bit-flip mutation, per the
SAME reference C — the task brief's "one-point crossover" sketch was
corrected to the C's actual two-point structure, code-over-report),
DTLZ8/DTLZ9 (the constraint-surface pair from the 2005 book chapter),
ZDT5 (the 80-bit binary-coded T5), the full WFG1-WFG9 scalable toolkit
(Huband, Hingston, Barone & While), a general-`M` exact hypervolume (the
WFG algorithm, While, Bradstreet & Barone 2012), and archive-first
"sezgi-moa v1" run logging:

    import sezgi, tempfile

    with tempfile.TemporaryDirectory() as log_dir:
        result = sezgi.mo.nsga2("wfg4", dim=None, m=2, pop_size=40, budget=4000,
                                 seed=20260830, log_dir=log_dir, label="demo")
        archive = sezgi.mo.read_moa(f"{log_dir}/demo-s20260830.moa")
    print(len(archive["archive"]), sezgi.mo.hypervolume(archive["archive"], [2.2, 4.4]))

Output (live-run, same scenario as `examples/python/wfg4_nsga2.py`):

    217 3.0620256270488344

New problem strings: `"zdt5"` (binary-coded, `dim` rejected — fixed
80-bit layout), `"dtlz8"`/`"dtlz9"` (constrained — `mo.nsga2(...)`'s
result dict gains a `"violations"` key, present ONLY for a constrained
problem), `"wfg1"`..`"wfg9"` (`dim` rejected — derived from `k`/`l`;
`k`/`l` default to the toolkit's own recommended values, `k=4` for `m=2`/
`k=2*(m-1)` for `m>=3`, `l=20`, when omitted). `mo.hypervolume(front,
ref_point)` is the general-`M` counterpart to the frozen 2-objective
`mo.hypervolume_2d` — `ref_point` is REQUIRED, with no default (see
`docs/DECISIONS.md`'s M3-7 record, ruling 5, and Ishibuchi, Imada,
Setoguchi & Nojima 2018's critique of the choice). `mo.nsga2(...,
log_dir=, label=)` streams every feasible archive insertion to
`<log_dir>/<label>-s<seed>.moa` (sezgi-moa v1: a versioned header plus one
record per archive insertion, no `.dat`/`.info` IOH mimicry, no COCO
compatibility claim — COCO bbob-biobj and MO-IOHinspector cited as design
precedent, not reproduced); `mo.read_moa(path, at=None)` reconstructs the
archive at any evaluation budget.

Binary-coded MO (zdt5) was scoped to the KanGAL reference exactly at M3-7
time — all-Float OR all-Binary spaces only, a MIXED space rejected naming
the M3-8 deferral explicitly. **M3-8 closes that deferral at the Rust-core
level**: the `nsga2_run` core now accepts any space combining
Float/Int/Categorical/Binary blocks, in any mix (see "Typed operators,
mixed spaces, and diagnostic problems (M3-8)" below) — only a space
containing a `Permutation` block stays rejected (MO permutation search is
out of scope, the error names it explicitly). This is core-only, though:
both `sezgi.mo.nsga2` and `sz_nsga2` still take a problem-name string
(zdt1-6/dtlz1-9/wfg1-9), and every constructible one of those is all-Float
or all-Binary (zdt5) — no mixed MO problem is constructible from either
binding today, so frontend reachability of mixed-space NSGA-II is
deferred (see `docs/DECISIONS.md`). Everything ZDT1-4/6/DTLZ1-7/
`hypervolume_2d` documented in the M3-2 section above, and the
all-Float/all-Binary paths documented here, stay frozen byte-for-byte
throughout.

R mirrors this 1:1 (`sz_nsga2`, `sz_mo_hypervolume`, `sz_mo_read_moa`),
including two new cross-language bit-equality anchors (an nsga2-on-wfg4
and an nsga2-on-zdt5 scenario) and a `.moa` file byte-identity check
across languages. `examples/python/wfg4_nsga2.py`/`examples/r/wfg4_nsga2.R`
is the matched pair (NSGA-II on WFG4, `log_dir`-logged, reporting archive
size + hypervolume at the nadir x 1.1 reference point). See
`docs/DECISIONS.md`'s M3-7 record for the full provenance table (KanGAL
C, the DTLZ 2005 chapter, the ZDT 2000 paper, the WFG EMO2005 paper plus
official toolkit via a dead-host Wayback chain, the 2012 hypervolume
paper), every toolkit-vs-paper/C-vs-paper/secondary-library divergence
found, and the closed carry-forward items.

## CEC 2022 benchmark suite (M3-3)

`sezgi.problems.cec2022(fid, dim)` / direct evaluation
(`sezgi.problems.cec2022_evaluate`, `sz_cec2022_evaluate`) implement all
**12 fids** of the CEC 2022 Special Session and Competition suite (Kumar,
Price, Mohamed, Hadi & Suganthan 2021) — basic functions f1-f5, hybrids
f6-f8, and compositions f9-f12 — at dims `{2,10,20}` (fid 6-8 reject
dim=2: the official data has no dim-2 shuffle data for hybrids). The
vendored shift/rotation/shuffle data (`crates/problems/data/cec2022/`,
~868KB) comes from the competition's own official repository
(`P-N-Suganthan/2022-SO-BO`), which carries **no LICENSE file anywhere in
the repo or the data archive** (checked directly, not assumed) — vendored
here with prominent attribution rather than withheld, per this project's
scope ruling; see `docs/DECISIONS.md`'s M3-3 record for the full finding.

Run a reference-tier preset (SHADE, Tanabe & Fukunaga 2013) against a CEC
2022 function through the same `sezgi.solve()` path every other preset
uses:

    import sezgi

    problem = sezgi.problems.cec2022(fid=3, dim=10)
    spec = sezgi.presets.shade(pop_size=20, budget=5000)
    result = sezgi.solve(spec, problem, master_seed=20260830, run_id=0)
    print(result["best_f"] - sezgi.problems.cec2022_f_star(3))

Output (live-run, same scenario as `examples/python/cec2022_shade.py`):

    600.0040181437115 - 600.0 -> gap 0.004018143711505218  (evals_used=5000)

**Where the printed CEC 2022 report and the official C code disagree,
sezgi follows the C code** — the code is what scored the competition and
produced its published results, not the report's (imperfect) prose. This
affects f3 (the report's "Expanded Schaffer's f6" name and its printed
scale/rotation do not match what the code actually computes — plain
Schaffer's F7, no scale, and, due to a verified buffer-reuse bug, no
effective rotation either), f4 (the report's "Non-Continuous Rastrigin"
step-quantization is dead code in the reference C — the shipped behavior
is plain Rastrigin), f5 (the printed `5.12/100` scale is a copy-paste
duplicate of f4's own scale — the code applies no scale), and f7 (the
report's printed 7-value weight array has one duplicate entry too many for
its own `N=6`, and the SchafferF7 hybrid component's assigned segment is
provably inert — a global-buffer indexing bug in the reference C).
Independent cross-validation against a freshly compiled copy of the
official C reference (primary) and `opfunu==1.0.4` (secondary; several new
opfunu-side bugs were root-caused and documented along the way) confirms
sezgi matches the C reference to machine precision at every probed point.
See `docs/DECISIONS.md`'s M3-3 record for the full method-provenance
table, every discrepancy quoted verbatim from the C source, and the
opfunu cross-check's complete findings.

R mirrors the direct-evaluation half 1:1 (`sz_cec2022_evaluate`,
`sz_cec2022_f_star`) and, since **M3-5**, also has a `solve()`-integrated
CEC 2022 binding: `sz_solve_cec2022(spec_json, fid, dim, master_seed,
run_id)` (mirrors `sz_solve_bbob`/`sz_solve_tsp` exactly) runs any built-in
preset — including `sz_preset_shade` — against a CEC2022 problem, closing
the gap M3-3 disclosed (`docs/DECISIONS.md`'s M3-3 record, ruling (g)).
`examples/r/cec2022_shade.R` now runs the SAME SHADE preset through the
SAME Rust core as `examples/python/cec2022_shade.py`; their `best_f`
outputs are bit-identical (verified via `writeBin`/`struct.pack`, not a
decimal-literal comparison — see that test in
`r-sezgi/tests/testthat/test-cec-tsp.R`).

## CEC 2014 / CEC 2017 benchmark suites (M3-6)

`sezgi.problems.cec2014(fid, dim)` / `sezgi.problems.cec2017(fid, dim)` and
their direct-evaluation counterparts (`*_evaluate`, `*_f_star`) implement
the **full CEC 2014 suite** (Liang, Qu & Suganthan 2013 — 30 fids:
unimodal f1-f3, simple multimodal f4-f16, hybrid f17-f22, composition
f23-f30) and the **CEC 2017 suite** (Awad, Ali, Liang, Qu & Suganthan
2016 — fid `{1} ∪ {3..=30}`, 29 usable fids), both at dims `{10,30}`
(the two dims this project vendors data for). The vendored shift/
rotation/shuffle data (`crates/problems/data/cec2014/`, 106 files,
2,811,834 bytes; `crates/problems/data/cec2017/`, 111 files, 3,280,439
bytes, both as stored in git after LF normalization of the upstream CRLF
endings) comes from each suite's own official repository, neither of which
carries a LICENSE file anywhere in the repo or its data archive (checked
directly, not assumed) — vendored here with prominent attribution rather
than withheld, the same scope ruling CEC 2022's data follows; see
`docs/DECISIONS.md`'s M3-6 record for the full finding.

**CEC 2017 fid 2 was officially withdrawn from the competition after
publication** ("Sum of Different Powers"); the official C reference's
`case 2` prints `"Error: This function (F2) has been deleted"` and leaves
its result unset. sezgi follows the C: `cec2017(2, dim)` raises a
dedicated withdrawn error (Python `ValueError`, R error) quoting the C's
own message, distinct from an ordinary out-of-range fid — the valid fid
set stays gapped at `{1} ∪ {3..=30}`, never renumbered/compacted.

Run a reference-tier preset (L-SHADE, Tanabe & Fukunaga 2014 — the CEC
2014 competition's own 1st-place algorithm) against a CEC 2014 function
through the same `sezgi.solve()` path every other preset uses:

    import sezgi

    problem = sezgi.problems.cec2014(fid=1, dim=10)
    spec = sezgi.presets.lshade(dim=10, budget=9000)
    result = sezgi.solve(spec, problem, master_seed=20260830, run_id=0)
    print(result["best_f"] - sezgi.problems.cec2014_f_star(1))

Output (live-run, same scenario as `examples/python/cec2014_lshade.py`):

    180.23000508877507 - 100.0 -> gap 80.23000508877507  (evals_used=9000)

**Where the printed report and the official C code disagree, sezgi
follows the C code**, per the same standing ruling CEC 2022 established.
For CEC 2014: F16 (the report's printed eq. adds a spurious `+1`
shift-to-origin the compiled `escaffer6_func` never applies) and CF1
(the report prints its `g2`/`g5` components under the identical name, but
the C hardcodes `g5`'s rotation flag off while `g2` rotates). For CEC
2017: fid 6 (the report's name, "Expanded Schaffer's F6," does not match
the C's actual dispatch to plain Schaffer's F7, with the loaded rotation
matrix silently discarded), fid 8 (the report's non-continuous
pre-transform is dead code in the reference C, same bug class as CEC
2022's own F4), fid 9/Levy (a genuine cross-generation constant
divergence from CEC 2022's own Levy core, `w=1+z/4` vs. `w=1+(z-1)/4`,
proven exact both algebraically and numerically), fid 20 (the report
names its first hybrid component "Happycat" where the C calls
`hgbat_func`), two VERIFIED reference-C bugs replicated deliberately
(`schaffer_F7_func`'s prefix-read bug in hf04/hf10, and `bi_rastrigin`'s
unshuffled-prefix-read bug in hf03 — both confirmed by instrumented-C
differential probes, not merely inferred), cf06's printed `lambda` array
(which matches no permutation of the C's own rescale factors), and items
28/29's swapped printed titles (a documentation-only finding, no code
impact). Independent cross-validation against a freshly compiled copy of
each suite's own official C reference (primary) and `opfunu==1.0.4`
(secondary) confirms sezgi matches the C reference to machine precision
at every probed point; opfunu itself agrees with sezgi on CEC 2014 fid
1-16+28 and on CEC 2017 fid 1 only, with four source-evidenced
opfunu-side divergence classes documented per suite. See
`docs/DECISIONS.md`'s M3-6 record for the full method-provenance table,
every discrepancy quoted verbatim, and the complete opfunu findings.

R mirrors both the direct-evaluation and `solve()`-integrated halves:
`sz_cec2014_evaluate`/`sz_cec2014_f_star`/`sz_solve_cec2014` and
`sz_cec2017_evaluate`/`sz_cec2017_f_star`/`sz_solve_cec2017` (the latter
pair mirroring `sz_solve_bbob`/`sz_solve_cec2022` exactly), plus generic
ask/tell sessions (`sz_eval_session_cec2014`, `sz_eval_session_cec2017`).
`examples/r/cec2014_lshade.R` runs the SAME L-SHADE preset through the
SAME Rust core as `examples/python/cec2014_lshade.py`; their `best_f`
outputs are bit-identical (verified via `writeBin`/`struct.pack`, not a
decimal-literal comparison — see that test in
`r-sezgi/tests/testthat/test-cec1417.R`).

Both suites' IOH logging uses the same suite-discriminator machinery
M3-5 built for CEC 2022 (`RunKey.suite`, `"sezgi-cec2014"`/
`"sezgi-cec2017"`), with `cec2014-f{fid}d{dim}i{instance}`/
`cec2017-f{fid}d{dim}i{instance}` labels — no changes were needed to the
IOH logger or the labeling helper itself, only the two frontends'
`solve()`/`for_problem` match arms.

## Permutation problems and TSP (M3-3)

Permutation-typed search spaces (`init/perm-random`, `gen/ox` order
crossover, `gen/perm-swap` swap mutation, and the fused `gen/ga-perm`
preset — a permutation genetic algorithm mirroring `presets::ga_real`'s
composition) plus a TSPLIB95 (Reinelt) `EUC_2D` loader with three vendored
instances and their published-optimal tour lengths as goldens: `berlin52`
(7542.0), `eil51` (426.0), `st70` (675.0).

    import sezgi

    problem = sezgi.problems.tsp("berlin52")
    spec = sezgi.presets.ga_perm(pop_size=32, budget=5000)
    result = sezgi.solve(spec, problem, master_seed=20260830, run_id=0)
    print(result["best_f"], sezgi.problems.tsp_load("berlin52")["known_optimum"])

Output (live-run, same scenario as `examples/python/tsp_ga_perm.py`):

    11771.0 7542.0   (gap 4229.0, ratio 1.5607, evals_used=4992)

`ga-perm` is a baseline permutation GA with no local search (no 2-opt), so
a gap of this size against the optimum is expected, not a defect — this
single-seed run is a SMOKE demonstration of the binding, not a quality
claim (see the single-seed-ban wording in `examples/README.md`). R mirrors
this through `sz_preset_ga_perm`/`sz_solve_tsp`, using **1-based** tour
indices throughout (matching TSPLIB's own node numbering and
`sz_bayesian_plackett_luce`'s existing 1-based item-id precedent) — unlike
Python's 0-based convention. Both bindings agree exactly on this scenario
(same master_seed, same Rust core underneath): `examples/r/tsp_ga_perm.R`
reproduces `11771.0` too. See `docs/DECISIONS.md`'s M3-3 record for OX1's
own provenance finding (Davis's actual 1985 paper describes a different,
single-cut-point operator; the two-cut-point cyclic "OX" implemented here
is the field's later, still Davis-attributed, synthesis — pinned to
Cicirello's 2023 worked numeric example) and the TSPLIB `nint` rounding
rule.

## Typed operators, mixed spaces, and diagnostic problems (M3-8)

Beyond the Float-block (`presets.ga_real`, DE, CMA-ES, ...) and
Permutation-block (`presets.ga_perm`) representations, sezgi now has fused
genetic-algorithm presets for the three remaining block kinds the design
spec's `SearchSpace` enum names (`docs/superpowers/specs/2026-08-27-sezgi-design.md`
§3): **`presets.ga_bin`** (Binary — KanGAL two-point crossover + bit-flip
mutation, Deb, Pratap, Agarwal & Meyarivan 2002 Sec. IV.A, `p_c=0.9`/
`p_m=1/L`, the same formulas M3-7's binary NSGA-II path already validated),
**`presets.ga_int`** (Int — real-coded SBX + polynomial mutation computed
in float then rounded and bound-repaired, pymoo 0.6.2's `Integer`
convention — "integer SBX" has no dedicated primary paper, pymoo is the
reference implementation and oracle), and **`presets.ga_cat`** (Categorical
— uniform crossover at 0.5 + random-reset mutation, pymoo 0.6.2's `Choice`
convention, resets do NOT exclude the current value). Standalone
half-operators ship alongside every fused preset (`gen/bin-2pt`/
`gen/bit-flip`, `gen/int-sbx`/`gen/int-pm`, `gen/cat-ux`/`gen/cat-reset`),
mirroring `gen/ox`/`gen/perm-swap`'s own precedent.

Three diagnostic problems (`sezgi.problems.onemax(n_bits)`,
`.int_quadratic(lo, hi, n)`, `.cat_match(k, n, seed)`) make every typed
preset reachable end to end — minimal, hand-verifiable, single-optimum
landscapes, explicitly a **diagnostic, not a benchmark suite** (no
published literature behind them, no quality claims beyond "did the
preset converge"):

    import sezgi

    problem = sezgi.problems.onemax(100)
    spec = sezgi.presets.ga_bin(pop_size=20, budget=400)
    result = sezgi.solve(spec, problem, master_seed=7, run_id=0)
    print(result["evals_used"], result["best_f"])

Output (live-run, same scenario as `examples/python/onemax_ga.py`):

    400 16.0

`solve()`'s typed-genotype `best_x` mirrors each block's own natural
Python/R type: Float stays `list[float]`/numeric (byte-identical to every
result before this milestone), Int/Categorical become `list[int]`/integer
(categorical values are indices `0..k`, not labels), Binary becomes
`list[bool]`/logical. This deliberately differs from `sezgi.mo.nsga2`'s/
`sz_nsga2`'s own Binary encoding (bits flattened to `0.0`/`1.0` floats, for
cross-family uniformity in that MO-specific result dict) — a documented,
per-surface choice, not an inconsistency to fix.

**Mixed spaces: `gen/compound` and mixed NSGA-II.** A search space
combining several block kinds is served by `gen/compound`, the design
spec §3's own "per block (compound operator)" mechanism: an ordered list
of one FUSED sub-generator per block (`gen/ga-real`, `gen/ga-int`,
`gen/ga-cat`, `gen/ga-bin`, or `gen/ga-perm`), each dispatched against its
own block's single-block view and stitched back into one offspring
genotype — composition adds nothing beyond routing (verified by an
exactness test replaying every sub-generator standalone against the same
RNG stream and asserting byte-identical output). Reachable from both
bindings through the normal spec path (JSON or TOML — the same
`AlgorithmSpec` schema either way):

    [stages.generator]
    kind = "gen/compound"
    blocks = [
      { kind = "gen/ga-real", tournament_k = 2, pc = 0.9, eta_c = 15.0, eta_m = 20.0 },
      { kind = "gen/ga-int",  tournament_k = 2, p_c = 0.9, eta_c = 15.0, eta_m = 20.0 },
      { kind = "gen/ga-cat",  tournament_k = 2, p_c = 0.9 },
      { kind = "gen/ga-bin",  tournament_k = 2, p_c = 0.9 },
    ]

Note the spelling shown mixed above: both `pc` and `p_c` are now accepted on
every family, canonical `p_c` (and `p_m` for the analogous mutation key,
where a family has one) — `gen/ga-real`/`gen/ox`/`gen/ga-perm` (the legacy
generators) accept `pc` as an alias for `p_c`, just like the M3-8 typed
families always did the other way; if a block sets both spellings, the
canonical key wins.

The `nsga2_run` core gained the same per-block support (a space combining
any of Float/Int/Categorical/Binary, in any mix — see "Multi-objective
optimization" above); every existing Float/Binary/constrained NSGA-II
golden stays bit-identical throughout — M3-8 added a THIRD `Representation`
case (`Mixed`), it did not touch the frozen two. This is a core-level
capability only: `sezgi.mo.nsga2`/`sz_nsga2` still accept a problem-name
string, and no mixed-space named problem exists, so mixed spaces are not
reachable through either binding yet (frontend reachability is a recorded
deferral, see `docs/DECISIONS.md`).

**ABC-TSP/permutation authoring** (both `sezgi.AskTellAlgorithm` and
`sz_algorithm`) is now supported too — see "Write your own algorithm,
ask/tell style (Python) (M3-4)" and "Write your own algorithm (R) (M3-5)"
below for the `ctx.kind`/`ctx.n`/`ctx.random_permutation()`/
`ctx.two_opt(tour, i, j)` surface and the `tsp_two_opt` worked example
pair.

See `docs/DECISIONS.md`'s "M3-8 completed" record for the full
method-provenance table (KanGAL binary reuse, pymoo 0.6.2 source shas,
Eiben & Smith 2015 taxonomy citation with its print-edition caveat, the
"Deb & Deb 2014 does not cover integers" research finding), the operator
inventory as shipped, and every deferral born this milestone.

## Write your own algorithm, ask/tell style (Python) (`AskTellAlgorithm`, M3-4)

**Renamed in M4-1.** `sezgi.Algorithm` now names the NEW engine-hosted
class-first base described in "Author your own algorithm (Python,
class-first)" above; the ask/tell surface described in this section is
`sezgi.AskTellAlgorithm` (`sezgi.algo.Algorithm` remains as a compat alias
for the same class — `sezgi.algo.Algorithm is sezgi.AskTellAlgorithm`).
Nothing below changed behavior, only the name.

`sezgi.AskTellAlgorithm` is a subclassable ABC for authoring a
metaheuristic entirely in Python (no Rust component graph, no
`ExperimentSpec`) while still getting evaluation counting, all-or-nothing
budget enforcement, best tracking, and optional IOH logging for free from
the SAME `EvalSession` core that backs the `examples/python/oop/` twins.
A subclass implements two methods — `setup(ctx)` (run once) and
`step(ctx)` (run repeatedly until the budget is exhausted) — over an
`AlgoContext` (`ctx.dim`, `ctx.bounds`, `ctx.rng`, `ctx.random_point()`,
`ctx.evaluate(points)`, `ctx.best()`, `ctx.remaining`). A random-search
subclass, in full:

    import sezgi


    class RandomSearch(sezgi.AskTellAlgorithm):
        """Draw batches of random points; keep the best (EvalSession does that)."""

        def setup(self, ctx):
            self.batch = 10

        def step(self, ctx):
            points = [ctx.random_point() for _ in range(self.batch)]
            ctx.evaluate(points)


    result = RandomSearch().solve(
        sezgi.bbob(fid=1, dim=5, instance=1), budget=2000, seed=42)
    print(f"evals_used={result.evals_used} best_f={result.best_f:.6g} "
          f"gap={result.gap:.6g}")

Output (live run):

    evals_used=2000 best_f=-124.971 gap=0.97896

`ctx.evaluate(points)` raises `sezgi.algo.BudgetExhausted` when a batch
would overrun the remaining budget — the driver catches it and ends the run
cleanly, so `step()` can be written as if the budget were unlimited, the
same `while used + N <= BUDGET` idiom the pure `examples/python/*.py`
scripts already use, just expressed as a boundary condition instead of a
loop guard. `solve()` returns a `SolveResult` (`algo`, `seed`, `budget`,
`evals_used`, `best_x`, `best_f`, `f_opt`, `gap` — `f_opt`/`gap` are `None`
for a problem with no known optimum, e.g. a raw `from_callable` handle).
`solve()` always calls `session.finish()` exactly once before returning
OR raising, so a run's IOH archive (if `log_dir=` was passed) is flushed
even when `setup()`/`step()` raises.

`ctx.evaluate(points)` has two distinct error types, not one: `BudgetExhausted`
(above) for a batch that doesn't fit the remaining budget — checked before
the session is touched, so a rejected batch charges nothing — and a plain
`ValueError`, raised by the underlying session itself, for a row with the
wrong length or a non-finite (NaN/inf) coordinate. The second is the
failure an author is most likely to hit in practice (a diverging custom
algorithm producing NaN), and it is NOT `BudgetExhausted` — catch
`ValueError` separately if a subclass wants to handle it.

**Feeding a custom algorithm into the stats pipeline.** `sezgi.algo.bbob_records`
sweeps a factory-constructed `AskTellAlgorithm` across combinations of BBOB
functions, dimensions, instances, and seeds, and records each run in the
exact same dict shape `run_experiment` produces (`algo`, `fid`, `dim`,
`instance`, `seed`, `budget`, `best_f`, `f_opt`, `gap`, `evals_used`,
`wall_secs`) — the two sources mix freely in one call to
`sezgi.results_matrix`/`sezgi.per_budget_packages`. Comparing a custom
Python algorithm against a built-in preset (or against another custom
algorithm) therefore goes through the SAME multi-seed, per-budget
machinery as any other comparison in this project — never a single-seed
run (see the single-seed-comparison ban stated in `examples/README.md` and
the "Experiments & Statistics" section above), and never pooled across
budgets (`per_budget_packages` builds one statistical package PER budget
present in the records, per Piotrowski et al. 2025's finding that
rankings can flip depending on which budget is examined):

    import sezgi
    from sezgi.algo import AskTellAlgorithm, bbob_records

    # RandomSearch as defined above; a second, genuinely different algorithm:
    class HillClimber(AskTellAlgorithm):
        def setup(self, ctx):
            ctx.evaluate([ctx.random_point()])

        def step(self, ctx):
            best_x, best_f = ctx.best()
            spread = 0.1 * (ctx.bounds[1] - ctx.bounds[0])
            p = [max(ctx.bounds[0], min(ctx.bounds[1],
                     x + ctx.rng.gauss(0, spread))) for x in best_x]
            ctx.evaluate([p])

    records = (bbob_records(RandomSearch, fids=[1, 2], dims=[2], instances=[1],
                             seeds=[0, 1, 2, 3, 4], budget=200)
               + bbob_records(HillClimber, fids=[1, 2], dims=[2], instances=[1],
                               seeds=[0, 1, 2, 3, 4], budget=200))
    packages = sezgi.per_budget_packages(records)  # one package per budget present
    budget, pkg = packages[0]
    print(budget, sorted(pkg.keys()))

Output (live run):

    200 ['bayes', 'cliffs', 'friedman', 'latex_summary', 'latex_tests', 'nemenyi_cd', 'pairwise_wilcoxon_holm', 'plackett_luce']

**Bias-scanning a custom algorithm.** `sezgi.bias.f0(dim, seed)` is a
`Problem` handle over the BIAS-toolbox's own `[0,1]^d` random test function
(no known optimum, and — like every non-BBOB problem — `log_dir`/IOH
logging is rejected for it, see the scope ruling below);
`sezgi.bias.structural_positions(final_positions)`
runs the SAME statistical KS/AD structural-bias scan described in the "Bias
scanning (M3-1)" section above, but over a plain list of final-position
vectors collected from ANY externally-driven algorithm, not just a
spec-driven engine run:

    import sezgi
    from sezgi.algo import AskTellAlgorithm

    class RandomSearch(AskTellAlgorithm):
        def setup(self, ctx):
            self.batch = 10

        def step(self, ctx):
            ctx.evaluate([ctx.random_point() for _ in range(self.batch)])

    positions = []
    for run in range(30):
        r = RandomSearch().solve(sezgi.bias.f0(dim=3, seed=run), budget=60, seed=run)
        positions.append(r.best_x)

    verdict = sezgi.bias.structural_positions(positions)
    print("verdict:", verdict["verdict"])

Output (live run):

    verdict: no_evidence

**Permutation/TSP authoring (M3-8).** `AlgoContext` also supports
permutation-typed problems now: `ctx.kind` is `"float"` or `"permutation"`
(`ctx.bounds` is `None` for the latter), `ctx.n` is the tour length,
`ctx.random_permutation()` draws a uniformly random 0-based tour (a
Rust-side Fisher-Yates seeded from the session's own RNG stream — NOT
`ctx.rng`, which stays the plain `random.Random` handle a Float-typed
algorithm uses), and `ctx.two_opt(tour, i, j)` reverses the inclusive
segment `tour[i:j+1]` (Eiben & Smith 2015: inversion is "the basic move
behind 2-opt" — the one neighborhood helper TSP authoring genuinely
needs). `EvalSession.for_problem` accepts a `sezgi.problems.tsp(name)`
handle directly; the Float surface stays byte-compatible throughout (every
pre-M3-8 example/test passes unmodified). `examples/python/oop/tsp_two_opt.py`
is the worked example — a first-improvement 2-opt local search from one
random start on TSPLIB berlin52:

    evals_used=2000 best_f=9077 gap=1535 tour_length=9077

R mirrors this exactly via `sz_algorithm`'s `ctx` (`ctx$kind()`, `ctx$n()`,
`ctx$random_permutation()`, `ctx$two_opt(tour, i, j)`) with **1-based**
tour indices (matching `sz_solve_tsp`'s own convention) — see "Write your
own algorithm (R) (M3-5)" below and `examples/r/oop/tsp_two_opt.R`, which
reproduces the SAME numbers above (tour length is index-convention-invariant;
`random_permutation()`'s own draw is bit-identical too, both languages
deriving from the same `RngStream`/Fisher-Yates core).

**Scope rulings.** `AskTellAlgorithm`/`AlgoContext` cover Float and
Permutation problems (v1) — Binary/Int/Categorical and mixed-typed
problems have no ask/tell session type yet (`EvalSession.for_problem`
rejects `sezgi.problems.onemax`/`.int_quadratic`/`.cat_match`/
`.mixed_diagnostic` with a `ValueError` naming the reason; `sezgi.solve()`
still runs them end to end, see "Typed operators, mixed spaces, and
diagnostic problems (M3-8)" above), deferred onward. IOH logging from a
custom `AskTellAlgorithm` covers BBOB and CEC 2022 problems (widened in
M3-5 — see `docs/DECISIONS.md`'s M3-5 record; this superseded an earlier
BBOB-only narrowing) — `sezgi.bbob(...)` and `sezgi.problems.cec2022(...)`
both work with `log_dir=`, but `sezgi.bias.f0(...)` and a raw
`from_callable` handle still raise `ValueError`, matching `sezgi.solve()`'s
own policy for the identical handles exactly (neither has a known optimum,
and `EvalSession.with_log` itself requires one). A known optimum (`f_opt`)
is necessary but not sufficient on its own for `log_dir` — the on-disk IOH
record key also needed a suite discriminator (`RunKey.suite`, M3-5) so a
CEC 2022 run and a BBOB run sharing `(fid, dim, instance, seed, budget)` no
longer silently merge into one `results_matrix` cell.

## Write your own algorithm (R) (M3-5)

`sz_algorithm(setup, step, name)`/`sz_algo_solve(algo, session, seed)` are
r-sezgi's **R-owned ask/tell scripting surface** — a SEPARATE, unchanged
surface from the engine-hosted class-first `sezgi::Algorithm` above (M4-2
ruling 5): unlike Python (which had to rename its own ask/tell ABC to
`AskTellAlgorithm` to free up the `Algorithm` name), R had no name
collision to begin with, so `sz_algorithm`/`sz_algo_solve` keep their
original names and behavior, unchanged by M4-2. They mirror
`sezgi.AskTellAlgorithm` above (the Python ask/tell surface, renamed in
M4-1 — see "Write your own algorithm, ask/tell style (Python)" above) —
same driver semantics (`setup(ctx)` once, `step(ctx)` repeatedly until the
budget is exhausted), expressed as two plain closures instead of a
subclass — this ask/tell surface itself is unaffected by M4-2's R6
addition (see the scope ruling below for the class surface's own R6
supersession). `ctx` is an
`environment` of callables (`ctx$dim()`, `ctx$bounds()`,
`ctx$random_point()`, `ctx$evaluate(points)`, `ctx$best()`, `ctx$f_opt()`,
`ctx$evals_used()`, `ctx$budget()`, `ctx$remaining()`) — every member is a
function, not a field, since R has no property/descriptor syntax to keep
`dim`/`bounds` and `evaluate`/`best` uniform otherwise.
`sz_algo_solve(algo, session, seed)` calls `set.seed(seed)` exactly once,
up front — R's global RNG stream IS the `ctx` RNG for the whole run, so
`ctx$random_point()` and any direct `runif()`/`sample()` call inside
`setup()`/`step()` are both driven by the one seeded stream. A random
search, in full:

    library(sezgi)

    rs_setup <- function(ctx) invisible(NULL)
    rs_step  <- function(ctx) ctx$evaluate(t(replicate(10, ctx$random_point())))
    rs <- sz_algorithm(rs_setup, rs_step, name = "random-search")

    s <- sz_eval_session(fid = 1, dim = 5, instance = 1, budget = 2000)
    res <- sz_algo_solve(rs, s, seed = 42)
    cat(sprintf("evals_used=%d best_f=%.6g gap=%.6g\n",
                res$evals_used, res$best_f, res$gap))

Output (live run):

    evals_used=2000 best_f=-124.389 gap=1.5604

`ctx$evaluate(points)` signals the custom condition class
`"sz_budget_exhausted"` (a base-R condition object, `c("sz_budget_exhausted",
"error", "condition")` — the same three trailing classes `simpleError()`
builds, so any generic handler catches it as an error too) when `points`
would overrun the remaining budget — checked BEFORE the session is
touched, so a rejected batch spends nothing; `sz_algo_solve()`'s own
`tryCatch` catches ONLY this condition class around the whole
`setup()`/`step()`-loop body, ending the run cleanly, same
`while used + N <= budget` idiom the pure `examples/r/*.R` scripts already
use, expressed as a boundary condition instead of a loop guard. `sz_algo_solve()`
returns a named `list` (`algo`, `seed`, `budget`, `evals_used`, `best_x`,
`best_f`, `f_opt`, `gap` — mirroring `SolveResult`'s field names exactly;
`f_opt`/`gap` are `NULL` for a problem with no known optimum, e.g. an f0
session) and GUARANTEES `session$finish()` runs exactly once on every exit
path — success, a driver-error `stop()` (a `step()` that consumed no
budget, or a run that evaluated nothing at all), or any other error
`setup()`/`step()` raises — via `on.exit(session$finish(), add = TRUE)`,
the base-R `finally` equivalent to py-sezgi's own `try`/`finally` guarantee.

**Sessions and bias scanning.** The same `sz_algorithm`/`sz_algo_solve`
surface drives any of r-sezgi's generic `EvalSession` constructors —
`sz_eval_session()` (BBOB), `sz_eval_session_cec2022()`, or
`sz_eval_session_f0()` (the BIAS-toolbox's own `[0,1]^d` random test
function, no known optimum, no `log_dir`/IOH logging argument at all —
`f_opt`/`gap` come back `NULL`) — so a custom R algorithm can be run and
compared across suites exactly like a Python one. `sz_bias_structural_positions(final_positions)`
runs the SAME statistical KS/AD structural-bias scan as the "Bias scanning
(M3-1)" section above, but over a plain matrix/list of final-position
vectors collected from ANY externally-driven algorithm — e.g. 30
`sz_algo_solve()` runs of the random search above over `sz_eval_session_f0()`:

    positions <- vector("list", 30)
    for (run in 1:30) {
      s <- sz_eval_session_f0(dim = 3, f0_seed = as.double(run), budget = 60)
      res <- sz_algo_solve(rs, s, seed = run)
      positions[[run]] <- res$best_x
    }
    verdict <- sz_bias_structural_positions(do.call(rbind, positions))
    cat("verdict:", verdict$verdict, "\n")

Output (live run):

    verdict: no_evidence

**Permutation/TSP authoring (M3-8).** `sz_eval_session_tsp(name, budget,
seed)` mirrors py-sezgi's TSP session; `ctx` gains `ctx$kind()`
(`"float"`/`"permutation"`), `ctx$n()`, `ctx$random_permutation()` (a
uniformly random **1-based** tour, matching `sz_solve_tsp`'s own
convention), and `ctx$two_opt(tour, i, j)` (inclusive `1 <= i < j <= n`
segment reversal) — the exact R mirror of `sezgi.AskTellAlgorithm`'s own
`ctx.kind`/`ctx.n`/`ctx.random_permutation()`/`ctx.two_opt()` surface
above, byte-compatible on the Float side throughout.
`examples/r/oop/tsp_two_opt.R` is the R twin of
`examples/python/oop/tsp_two_opt.py`: same instance/seed/budget, and (since
both languages derive `random_permutation()` from the identical
`RngStream`/Fisher-Yates core) it reproduces the SAME
`evals_used=2000 best_f=9077 gap=1535 tour_length=9077` numbers, gated by a
cross-language hex anchor (`r-sezgi/tests/testthat/test-tsp-two-opt.R`).

**Scope rulings.** This ask/tell surface itself stays base-R closures/
environments (unchanged by M4-2 — no R6 involved anywhere in
`sz_algorithm`/`sz_algo_solve`/`R/algo.R`); M3-5 scope ruling 4 ("R stays
base-R, no R6") was SUPERSEDED for the class-first surface only (M4-2
ruling 1, `Imports: R6 (>= 2.4.0)` — see `docs/DECISIONS.md`'s M4-2
record), not for this ask/tell surface, which needed no class system to
begin with and was not touched. Float and Permutation problems (v1,
widened from Float-only by M3-8) — Binary/Int/Categorical and mixed-typed
problems have no ask/tell session type yet, matching
`AskTellAlgorithm`/`AlgoContext`'s own scope above exactly. `sz_algorithm`/
`sz_algo_solve` port the pinned `examples/r/gwo.R` script onto this surface
verbatim as `examples/r/oop/gwo.R` — the ONE worked twin proving the
surface (not a full 17-algorithm R wave like `examples/python/oop/`'s —
see `examples/README.md`'s "R authoring example (M3-5)" section and
`docs/DECISIONS.md`'s M3-5 record for the draw-order analysis and gate).
**R callable-objective sessions: NARROWED by M4-2, not closed** (M3-5
scope ruling 5). M4-2 closed the "no engine-solve path for an R callable
at all" half of this deferral: `sezgi::Problem` + `sz_solve_r_problem()`
(above) let an R researcher's own function run INSIDE the Rust engine
loop, per-generation batched, with full determinism. What is STILL open:
a `Problem` subclass does not plug into `sz_algorithm`/`sz_algo_solve`'s
own SESSION-backed ask/tell surface here — there is no
`sz_eval_session`-style counting/logging session type for an arbitrary R
callable, so an R-authored ask/tell algorithm (as opposed to an
engine-hosted `Algorithm`/`PopulationAlgorithm`/`LocalSearch` subclass)
still cannot evaluate its own R function through THIS surface. See
`docs/DECISIONS.md`'s M4-2 record for the precise wording.

## Examples

`examples/` holds a catalog of all **17** labeled-metaphor algorithms (GWO,
WOA, Harmony Search, Cuckoo Search, GOA, SCA, JAYA, MFO, SSA, FA, BA, FPA,
TLBO, HHO, ALO, ABC, GSA), each as a triplet: a pure-Python teaching
implementation (stdlib only, driven through `sezgi.EvalSession`), the same
in pure R (`sz_eval_session`), and an `ExperimentSpec` TOML running sezgi's
own built-in, RNG-stream-pinned preset for that algorithm — **51 example
artifacts** in total. All three forms implement the same pinned update
equations documented in the algorithm's Rust component
(`crates/components/src/{gwo,woa,hs,cs,goa,sca,jaya,mfo,ssa,fa,ba,fpa,tlbo,
hho,alo,abc,gsa}.rs`). See `examples/README.md` for the full catalog table
(primary + equivalence-critique references, and what each pure script
teaches vs. its preset).

`examples/python/nsga2_zdt1.py` / `examples/r/nsga2_zdt1.R` are a separate
matched PAIR (M3-2): NSGA-II on ZDT1, run directly through the Rust core in
both languages (bit-identical output, not merely statistically
comparable). No `specs/nsga2_zdt1.toml` exists — see "Multi-objective
optimization (M3-2)" above for why.

`examples/python/cec2022_shade.py`/`examples/r/cec2022_shade.R` and
`examples/python/tsp_ga_perm.py`/`examples/r/tsp_ga_perm.R` are two more
matched PAIRs (M3-3; the CEC pair's R-side gap closed in M3-5): SHADE on
CEC 2022 f3, and ga-perm on TSPLIB berlin52, both through
`sezgi.solve()`/`sz_solve_*`, both bit-identical between languages. See
"CEC 2022 benchmark suite (M3-3)" and "Permutation problems and TSP
(M3-3)" above.

`examples/python/cec2014_lshade.py`/`examples/r/cec2014_lshade.R` (M3-6)
are a matched PAIR for the CEC 2014 suite: L-SHADE (the CEC 2014
competition's own 1st-place algorithm) on CEC 2014 f1, through
`sezgi.solve()`/`sz_solve_cec2014`, bit-identical between languages. See
"CEC 2014 / CEC 2017 benchmark suites (M3-6)" above.

`examples/python/wfg4_nsga2.py`/`examples/r/wfg4_nsga2.R` (M3-7) is a
matched PAIR for the MO remainders: NSGA-II on WFG4, called directly
through `sezgi.mo.nsga2()`/`sz_nsga2()` (like `nsga2_zdt1.py`/`.R`, not
`sezgi.solve()` — see "Multi-objective optimization" above), with
`log_dir`-logged sezgi-moa v1 output read back via `mo.read_moa()`/
`sz_mo_read_moa()` and reported through the general-`M` `mo.hypervolume()`/
`sz_mo_hypervolume()`, bit-identical between languages (gated by a
committed testthat anchor, `r-sezgi/tests/testthat/test-mo.R`).

`examples/python/onemax_ga.py`/`examples/r/onemax_ga.R` (M3-8) is a matched
PAIR for the typed-operator milestone: `ga_bin` on the `OneMax` diagnostic
problem, through `sezgi.solve()`/`sz_solve_onemax`, bit-identical between
languages (a deliberately non-converging parameter set, per the M3-8 Task
10 cross-language-anchor convention, so the anchor pins the run's actual
trajectory, not just "reached the trivial known optimum"). See "Typed
operators, mixed spaces, and diagnostic problems (M3-8)" above.

`examples/python/oop/tsp_two_opt.py`/`examples/r/oop/tsp_two_opt.R` (M3-8)
is a matched PAIR for ABC-TSP authoring: a first-improvement 2-opt local
search (`sezgi.AskTellAlgorithm`/`sz_algorithm`, `ctx.random_permutation()` +
`ctx.two_opt()`) on TSPLIB berlin52 from one random start, bit-identical
between languages despite the 0-based/1-based tour-indexing difference
(tour LENGTH is index-convention-invariant). Unlike every other pair
above, both sides sit OUTSIDE their own 17-pair OOP-twin parity gate (no
pure-script counterpart exists to reproduce) — each gets its own
determinism/anchored-output/cross-language-hex test file instead
(`py-sezgi/tests/test_tsp_two_opt_example.py`,
`r-sezgi/tests/testthat/test-tsp-two-opt.R`). See "Write your own
algorithm, ask/tell style (Python) (M3-4)" above for the worked walkthrough.

`examples/python/oop/custom_de_variant.py` and
`examples/python/oop/custom_local_search.py` (M4-1) are two more
engine-hosted worked examples, over the NEW class-first `sezgi.Algorithm`
family bases (`PopulationAlgorithm`/`LocalSearch`, not the ask/tell
surface above): a DE/rand/1-shaped `vary()` override and a
`neighbor()`-only local search, both on `sezgi.bbob(1, 10, 1)`. Like
`tsp_two_opt.py`, both sit outside the 17-pair OOP-twin parity gate (no
pure-script counterpart), each gated by its own anchored pytest.
`examples/python/oop/feature_selection.py` (M4-1) demonstrates
`sezgi.recipes.FeatureSelection` recovering a known informative-column
mask via `GeneticAlgorithm`'s Binary auto-dispatch. `tsp_two_opt.py` (and
its 18 `examples/python/oop/*.py` siblings) also now import
`sezgi.AskTellAlgorithm` instead of `sezgi.Algorithm` (M4-1's rename — see
"Write your own algorithm, ask/tell style (Python)" above); no example's
printed numbers changed. See "Author your own algorithm (Python,
class-first)" and "Data recipes: feature selection" above, and
`examples/README.md`'s own catalog rows, for the full walkthroughs.

`examples/r/oop/custom_de_variant.R` and `examples/r/oop/
custom_local_search.R` (M4-2) are the R twins of the two Python examples
above, over the NEW class-first `sezgi::Algorithm` family bases
(`PopulationAlgorithm`/`LocalSearch`, not the `sz_algorithm`/
`sz_algo_solve` ask/tell surface above): the same DE/rand/1-shaped
`vary()` override and `neighbor()`-only local search, both on
`sz_builtin_bbob(1, 10, 1)`. Like `tsp_two_opt.R`, both sit outside the
17-pair OOP-twin parity gate (no pure-script counterpart), each gated by
its own `stopifnot()` anchor. `examples/r/oop/feature_selection.R` (M4-2)
demonstrates `FeatureSelection` recovering a known informative-column mask
via `GeneticAlgorithm`'s Binary auto-dispatch, over an independently
derived R dataset (not shared with the Python twin's own fixture). See
"Author your own algorithm (R, class-first)" and "Data recipes: feature
selection (R)" above, and `examples/README.md`'s own catalog rows, for the
full walkthroughs.

## Development

    cargo test --workspace --release        # Rust tests
    cd py-sezgi && maturin develop && pytest # Python tests
    R CMD INSTALL --preclean r-sezgi && Rscript -e 'testthat::test_dir("r-sezgi/tests/testthat", package = "sezgi")' # R tests

## Status

M4-1 (Python class-first front door + engine-hosted authoring) **complete**
— a course correction (user direction 2026-09-02) turning the Python
interface into a class-first front door: built-in algorithms as
configurable classes, new algorithms authored by subclassing the new
engine-hosted `sezgi.Algorithm` template base (hooks run INSIDE the Rust
engine loop as real `Generator`/`Initializer` components via a captured
Python callback, NOT a Python-owned loop), and user problems authored by
subclassing `sezgi.Problem` (with two data recipes) — while `solve()`/
`presets.*` remain fully working compat internals and **the Rust core
stayed byte-untouched throughout** (`git diff --stat -- crates/` empty at
every one of the six tasks' gates, confirming the research doc's §B6
zero-core-change feasibility verdict HELD in practice, not just in
theory). Delivered: five space builders (`Float`/`Int`/`Categorical`/
`Binary`/`Permutation`/`Space`) and the `Problem` ABC over a
newly-block-typed callable bridge; the `PyRng`/`EngineCtx`/`PopView`
bridge (clone-out/mutate/write-back RNG protocol, verified byte-identical
to the engine's own per-stage stream reconstruction); `Algorithm`/
`PopulationAlgorithm`/`LocalSearch` family bases; 28 built-in wrapper
classes (one per `presets.rs` builder) plus `GeneticAlgorithm` auto-dispatch,
`DifferentialEvolution` variants, and a run-only `NSGA2` class skin (29
classes total, table-driven, proven bit-identical to the underlying
`solve()`/preset call at the same seed); the `FeatureSelection`/
`MixedTuning` data recipes. `sezgi.Algorithm`'s old ask/tell surface is
renamed `sezgi.AskTellAlgorithm` (`sezgi.algo.Algorithm` kept as a compat
alias; 18 `examples/python/oop/*.py` scripts and their tests updated,
import/name-only). See "Quickstart (Python)" through "Internals & spec
files" above for the full walkthrough and `docs/DECISIONS.md`'s "M4-1
completed" record for the six scope rulings, the provenance table (hook
taxonomy modeled after pymoo 0.6.2 and jMetal v7.5's public APIs, no code
copied), the honest-degrade note (Python-component specs are
process-local, no TOML persistence), and all eight deferrals born this
milestone (R mirror of the class surface, MO authoring, `ctx.eval`
mid-generate evaluations, Mixed auto-dispatch for `GeneticAlgorithm`, and
four more, including two pre-existing engine gaps found along the way).
Next: v1.0 prep (see the checklist), or the R mirror of this milestone's
class surface.

M3-8 (mixed-type operators) **complete** — closes deferred group D from the
M3-2/M3-7 records: Binary/Int/Categorical typed operator families
(`gen/ga-bin`/`gen/ga-int`/`gen/ga-cat` plus standalone halves, KanGAL
binary reuse + pymoo 0.6.2 Integer/Choice conventions as the executable
oracles, Eiben & Smith 2015 §4.2-4.3 as the citable textbook taxonomy);
three diagnostic problems (`OneMax`/`IntQuadratic`/`CatMatch`, honestly
NOT a benchmark suite) making every typed preset reachable end to end; a
per-block `gen/compound` generator implementing the design spec §3's own
"compound operator" mechanism for mixed search spaces, as a COMPONENT (no
engine/spec-validator surgery); mixed-space NSGA-II support (Float+Int+
Categorical+Binary in any combination, Permutation still rejected as
out-of-scope for MO search) built on an M3-7 carry-forward that unifies
`nsga2_run`'s float/binary main loops first (bit-identity re-proved against
every frozen golden in the same commit) and dedupes the
`axis_grid`/`grid_r`/`cartesian_product` sampling helpers; permutation/TSP
authoring in both `sezgi.AskTellAlgorithm` (`ctx.kind`/`ctx.n`/
`ctx.random_permutation()`/`ctx.two_opt()`) and `sz_algorithm` (1-based
mirror); Python and R bindings for the whole typed surface plus the four
M3-7-parked binding tests (nsga2-on-WFG, a constrained logged dtlz8 run,
degenerate hypervolume fronts, one nsga2 m>3 run); and a matched
`onemax_ga`/`tsp_two_opt` example-pair set. See `docs/DECISIONS.md`'s
"M3-8 completed" record for the full method-provenance table (including
pymoo's exact re-fetched source shas, quoted verbatim), the six scope
rulings, the operator inventory as shipped, and every deferral born this
milestone (a documented pymoo-vs-sezgi Integer-mutation gate divergence, a
proposed-but-not-implemented `validate_space` build hook, the Binary
dual-encoding split between `solve()` and `mo.nsga2`, and the
single-block-per-kind typed-generator convention). Next: v1.0 prep (see
the checklist).

M3-7 (multi-objective remainders) **complete** — closes every deferred MO
capability from M3-2: a constraint channel (`MoProblem::evaluate_constraints_batch`)
and Deb's constrained-domination in NSGA-II, transcribed from KanGAL
`nsga2r.c`'s own `check_dominance`; a binary genotype path (two-point
crossover + bit-flip mutation, also from the C — the plan's own
"one-point crossover" sketch was wrong, corrected under the standing
code-over-report ruling); DTLZ8/DTLZ9 (the constraint-surface pair, Eq.
6.26/6.27 of the 2005 book chapter, including a documented `// sezgi
decision:` for DTLZ8's undefined `M=2` corner); ZDT5 (the 80-bit
binary-coded T5, Zitzler-Deb-Thiele 2000 Definition 4); the full
WFG1-WFG9 scalable toolkit (Huband, Hingston, Barone & While, recovered
via a dead-host Wayback Machine chain after IEEE/ResearchGate/Semantic
Scholar all failed, cross-checked against a compiled copy of the official
C++ toolkit at ~1e-15 and against pymoo 0.6.2 with zero divergences across
288 comparisons); a general-`M` exact hypervolume (the WFG algorithm,
While, Bradstreet & Barone 2012, also Wayback-recovered) with an explicit,
never-defaulted reference point (Ishibuchi, Imada, Setoguchi & Nojima
2018 cited for why); and archive-first "sezgi-moa v1" run logging
(COCO bbob-biobj/MO-IOHinspector cited as design precedent, no
compatibility claim). Python (`sezgi.mo.*`) and R (`sz_nsga2`/`sz_mo_*`)
bindings expose the full surface with cross-language bit-equal output; a
matched `wfg4_nsga2` Python/R example pair, gated by a committed testthat
anchor. WFG/hv reference code stayed ORACLE-ONLY throughout — never
vendored (see `docs/DECISIONS.md`'s M3-7 record for the license findings
on each). MO component-graph spec integration remains deferred to v2
(unchanged since M3-2); `Int`/`Categorical`/`Binary` typed operators and
mixed-representation NSGA-II remain M3-8 scope, deliberately not
pre-empted by this milestone's binary-only NSGA-II path. See
`docs/DECISIONS.md`'s "M3-7 completed" record for the full
method-provenance table, every divergence found, and the closed
carry-forward items. Next: v1.0 prep (see the checklist), or M3-8
(Int/Categorical/Binary typed operators, mixed spaces).

M3-6 (CEC 2014 + CEC 2017 benchmark suites) **complete** — the full CEC
2014 suite (30 fids: unimodal, simple multimodal, hybrid F17-F22,
composition F23-F30) and the CEC 2017 suite (fid `{1} ∪ {3..=30}`, fid 2
officially withdrawn and rejected with a dedicated error), both at dims
`{10,30}` (vendored data: CEC 2014 106 files/2,811,834 bytes, CEC 2017
111 files/3,280,439 bytes as stored in git — both superseding the plan's
pre-research byte estimates), reusing the CEC 2022-derived shared basic-function library
(`crates/problems/src/cec_basics.rs`) byte-for-byte; every
report-vs-official-C divergence found and resolved per the standing
code-over-report ruling, including two VERIFIED reference-C bugs
replicated deliberately; Python (`sezgi.problems.cec2014`/`cec2017`) and
R (`sz_solve_cec2014`/`sz_solve_cec2017`, `sz_eval_session_cec2014`/
`sz_eval_session_cec2017`) bindings with suite-aware IOH logging
(`"sezgi-cec2014"`/`"sezgi-cec2017"`, riding M3-5's `RunKey.suite`
machinery unchanged, no logger changes needed); a matched Python/R
example pair (L-SHADE, the CEC 2014 competition's own winner, on CEC
2014 f1, bit-identical between languages). Independent cross-validation
against a freshly compiled official C reference (primary) and
`opfunu==1.0.4` (secondary) confirms sezgi matches the C reference to
machine precision; opfunu itself agrees with sezgi only on CEC 2014 fid
1-16+28 and CEC 2017 fid 1 — four source-evidenced opfunu-side
divergence classes per suite, none a sezgi defect. A `R CMD build
r-sezgi` tarball measurement found the current build far under CRAN's
~5MB guideline only because it does not yet vendor its path-dependency
crates or any CEC/TSPLIB data at all (the M2d-3 structural blocker,
still open); the actual vendored-data total across all CEC suites plus
TSPLIB now measures 7.2MB, over the guideline, with a dim-10-only trim
recorded as a fallback decision for the user at v1.0, not applied here.
See `docs/DECISIONS.md`'s "M3-6 completed" record for the full
method-provenance table, every divergence quoted, and the CRAN
measurement's full detail. Next: v1.0 prep (see the checklist).

M3-5 (frontend parity and logging gaps) **complete** — closed every deferred
parity/logging gap from M3-3/M3-4: a `suite` discriminator on `RunKey`/
record reconstruction (label-stable for BBOB, suite-prefixed otherwise,
backward-compatible with old journals/record dicts) fixing the M3-4 final
review's silent BBOB/CEC-2022 collision at its root; custom-session IOH
logging widened from BBOB-only to BBOB + CEC 2022, at both entry points
(`EvalSession.for_problem` and the module-level `solve()`); `log_dir`
threaded through `bbob_records`; `sz_solve_cec2022` (r-sezgi), closing the
M3-3 R-side CEC solve gap — `examples/r/cec2022_shade.R` now runs the SAME
SHADE preset as its Python twin, bit-identical; generic R eval sessions
(`sz_eval_session_cec2022`, `sz_eval_session_f0`) with `$dim()`/`$bounds()`
accessors on every session; and the pure-R algorithm-authoring surface
(`sz_algorithm`/`sz_algo_solve`, base R only — closures and condition
classes, no R6/S4) plus `sz_bias_structural_positions`, proven by ONE
worked twin (`examples/r/oop/gwo.R`) reproducing `examples/r/gwo.R`'s
`evals_used`/`best_f`/`gap` output STRING-EXACTLY at the same seed. See
`docs/DECISIONS.md`'s "M3-5 completed" record for the full ruling list,
every closed v1.0 item, and the new deferral (R callable-objective
sessions). Next: v1.0 prep (see the checklist), or the next approved group
of deferred milestones (CEC 2014/2017, MO remainder, mixed-type problems).

M3-4 (Python algorithm authoring + OOP example twins) **complete** — the
ask/tell `Algorithm` ABC (`py-sezgi/python/sezgi/algo.py`, renamed
`sezgi.AskTellAlgorithm` in M4-1): a subclassable
`setup(ctx)`/`step(ctx)` template over `AlgoContext`/`EvalSession`, chosen
over pure ask/tell because mid-generation evaluation patterns (TLBO's
teacher/learner passes, HHO's dives) cannot be expressed as a single ask;
`bbob_records`, a multi-scenario sweep helper feeding custom-algorithm runs
into `results_matrix`/`per_budget_packages` in the same record shape
`run_experiment` produces; a generalized `EvalSession` (`SessionMeta`,
`f_opt: Option<f64>`, `EvalSession.for_problem` accepting BBOB/CEC2022/
callable problems) with the calling-convention (`vectorized`) carried on
the `from_callable` handle, honored identically by `solve()` and
`for_problem`; the `bias.f0`/`bias.structural_positions` bridge letting a
structural-bias scan run over final positions collected from ANY
externally-driven algorithm; and OOP twins of **all 17** example
algorithms (`examples/python/oop/`) behind a 17-pair parity gate
(`py-sezgi/tests/test_examples_oop_parity.py`) comparing each twin's
printed `%.6g` output fields string-exactly against its pre-existing pure
script, which remains untouched (raw f64 bit-pattern equality of
`best_f`/`gap`/`best_x` for all 17 pairs was additionally verified at the
2026-08-30 final whole-branch review — see `examples/README.md`'s "OOP
twins" section). Scope: continuous problems only in
v1 (TSP/permutation authoring deferred); Python authoring only (R deferred
to the v1.0 checklist below). See `docs/DECISIONS.md`'s "M3-4 completed"
record for the full ruling list. Next: v1.0 prep (see the checklist).

M3-3 (CEC 2022 benchmark suite + permutation problems/TSP) **complete** —
all 12 CEC 2022 fids (`crates/problems/src/cec2022/`, ~868KB vendored
data, no upstream LICENSE file, attributed) with five adjudicated
report-vs-official-C discrepancies (F3, F4, F5, the fid-7 weight-array
misprint, and the fid-7 SchafferF7 dead-segment bug) resolved in the C
code's favor per the standing code-over-report ruling; independent
cross-validation against a freshly compiled official C reference (primary)
and `opfunu==1.0.4` (secondary, six new opfunu bug classes root-caused);
permutation operators (`init/perm-random`, `gen/ox`, `gen/perm-swap`) and
the fused `gen/ga-perm` preset; a TSPLIB95 `EUC_2D` loader with three
vendored instances (berlin52/eil51/st70) and their published-optimal-tour
goldens; Python (`sezgi.problems.*`) and R (`sz_cec2022_*`/`sz_tsp_*`)
bindings, with R disclosed as lacking a `solve()`-integrated CEC2022
binding (direct evaluation only); four live example scripts (two matched
Python/R pairs). See `docs/DECISIONS.md`'s "M3-3 completed" record for the
full method-provenance table, every discrepancy quoted verbatim, and the
v1.0 readiness checklist. Next: v1.0 prep (see the checklist).

M3-2 (multi-objective optimization) **complete** — NSGA-II (Deb, Pratap,
Agarwal & Meyarivan 2002) as a self-contained, seeded reference runner
(`crates/components/src/nsga2.rs`) over a new parallel MO core surface
(`MoProblem`/`MoEvaluator`/`MoPopulation`, `crates/core/src/mo.rs`); the
ZDT (Zitzler, Deb & Thiele 2000) and DTLZ (Deb, Thiele, Laumanns & Zitzler
2005) test-problem suites (`crates/problems`); 2-objective hypervolume and
IGD quality indicators (`crates/stats/src/moo_indicators.rs`); a KanGAL
(`nsga2r.c`) source-code finding that the reference tournament uses raw
pairwise dominance rather than reading rank (a paper-vs-code divergence,
C behavior implemented and documented); Python (`sezgi.mo.*`) and R
(`sz_nsga2`/`sz_mo_*`) bindings with 1:1 key mirroring and cross-language
bit-equal output; a matched Python/R NSGA-II-on-ZDT1 example pair. MO
component-graph spec integration (an `ExperimentSpec` you assemble NSGA-II
from) is deferred to v2 — see `docs/DECISIONS.md`'s "M3-2 completed"
record for the full method-provenance table, every ruling, and the
deferrals list. Next: M3-3 (CEC benchmark suites, mixed-type problems).

M3-1 (bias-scanning module) **complete** — `crates/bias` (structural bias,
central bias, and a one-call `bias_report`), exposed as `sezgi.bias.*` /
`sz_bias_*` in both frontends with cross-language bit-equal output; a
pre-M3 engine follow-up (`RunResult::best_f` unified with the
Evaluator-observed minimum, closing the M2d-4 HHO divergence caveat). The
signature/Rajwar-Deep test (a third planned bias check) is BLOCKED and
deferred — its primary source is paywalled with no accessible preprint or
reference implementation. See `docs/DECISIONS.md`'s "M3-1 completed" record
for the full method-provenance table, pinned statistical formulas, and every
ruling made along the way. Next: M3-2 (multi-objective: NSGA-II, ZDT/DTLZ).

M2d-4 (second and final labeled-metaphor wave) **complete** — twelve more algorithm presets (SCA, JAYA, MFO, SSA, FA, BA, FPA, TLBO, HHO, ALO, ABC, GSA) with pinned deterministic draw orders and primary-source citations, completing the labeled-metaphor catalog at **17 algorithms**, plus their pure-Python/pure-R/spec example triplets under `examples/` (**51 example artifacts** total across all 17); new reusable components `replace/bat-loudness-greedy`, `replace/abc-trial-greedy`, `adapter/abc-onlooker-scout`; TLBO as the project's first multi-stage-per-generation preset (`gen/tlbo-teacher` + `gen/tlbo-learner`); HHO as the project's first in-generator (not adapter) mid-evaluation generator. See `docs/DECISIONS.md`'s "M2d-4 completed" record for the full per-algorithm provenance table (source artifact + sketch-vs-verified deltas found) and the wave's consolidated rulings, including the wave-wide current-pop-argmin parked convention and its MFO-flame/ALO-antlion carve-out.

M2d-3 (labeled metaphor presets, ask/tell `EvalSession`, CRAN dry run) **complete** — five labeled-metaphor algorithm presets (GWO, WOA, Harmony Search, Cuckoo Search/Lévy, GOA) with pinned deterministic draw orders and primary-source citations; ask/tell `EvalSession`/`sz_eval_session()` exposed in both Python and R (no internal RNG, seed-as-label, constructor-only IOH logging); a budget-meta-key lift in IOH logging with two read-side reconciliation policies (`dedupe_same_budget`, `canonical_anytime`); a core-engine `global_best` fix; an `R CMD check --as-cran` dry run and a `cargo vendor` dry run (both recorded, not resolved — see below); `f64_to_u64` strictness and an `ExperimentError::Parse`/`InvalidSpec` diagnostics split. See `docs/DECISIONS.md` for the full M2d-3 record, including the CRAN check's verbatim output and the vendoring restructuring options (decision deferred to v1.0 prep — the R package currently depends on sibling workspace crates by path, which is not CRAN-submittable as-is). Next: the bias-scanning showcase (M3). License: MIT.

## Algorithms

sezgi M2b ships 13 reference algorithm presets (with Rust function names):

| Algorithm | Preset Function |
|-----------|-----------------|
| Differential Evolution (rand/1) | `presets::de_rand_1` |
| Differential Evolution (best/1) | `presets::de_best_1` |
| jDE (self-adaptive DE) | `presets::jde` |
| SHADE | `presets::shade` |
| L-SHADE | `presets::lshade` |
| CMA-ES | `presets::cmaes` |
| CMA-ES with IPOP restarts | `presets::cmaes_ipop` |
| Particle Swarm Optimization | `presets::pso` |
| Genetic Algorithm (real-coded) | `presets::ga_real` |
| (μ+λ)-Evolution Strategy | `presets::es_mu_plus_lambda` |
| Simulated Annealing | `presets::sa` |
| Nelder–Mead Simplex | `presets::nelder_mead` |
| Random Search (baseline) | `presets::random_search` |

Both the Python and R bindings expose every preset above, including `es_mu_plus_lambda` (its `Distribution` argument is bridged via a distribution-name string plus per-family parameters — see `sezgi.presets.es_mu_plus_lambda` / `sz_preset_es_mu_plus_lambda`).
