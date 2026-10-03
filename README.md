# sezgi

[![PyPI version](https://img.shields.io/pypi/v/sezgi.svg)](https://pypi.org/project/sezgi/)
[![CRAN status](https://www.r-pkg.org/badges/version/sezgi)](https://cran.r-project.org/package=sezgi)

**sezgi** (Turkish for "intuition") is a Rust-core, component-based metaheuristic
optimization library with Python and R frontends.

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
reference, and a 12-entry examples gallery) is published at
**<https://tdelphi1981.github.io/sezgi/>** (rebuilt from `main` on every
push by `.github/workflows/docs.yml`). To build it locally with MkDocs:

    uv pip install --python py-sezgi/.venv/bin/python \
      mkdocs==1.6.1 mkdocs-material==9.7.7 mkdocstrings==1.0.6 \
      mkdocstrings-python==2.0.8 markdown-exec==1.12.3 \
      pymdown-extensions==11.0.2 matplotlib==3.11.1
    py-sezgi/.venv/bin/mkdocs build --strict   # or: ... serve
    open site/index.html

See `CONTRIBUTING.md` for the full development setup and `docs/site/`
for the page sources.

## Team

sezgi is developed at the Department of Computer Science, Faculty of
Science, Karadeniz Technical University (KTU), Trabzon, Türkiye:

- **Tolga Berber** (Assoc. Prof.) —
  [AVESIS](https://avesis.ktu.edu.tr/tberber) ·
  [ORCID 0000-0002-6487-5581](https://orcid.org/0000-0002-6487-5581)
- **Beyzanur Siyah** (Research Assistant) —
  [AVESIS](https://avesis.ktu.edu.tr/beyzanursiyah) ·
  [ORCID 0000-0002-2071-3724](https://orcid.org/0000-0002-2071-3724)
- **Emir Karayağız** (Research Assistant) —
  [AVESIS](https://avesis.ktu.edu.tr/emirkarayagiz) ·
  [ORCID 0009-0005-1673-621X](https://orcid.org/0009-0005-1673-621X)

## Quickstart (Python)

Install from PyPI (prebuilt wheels for Linux x86_64/aarch64, macOS
universal2, and Windows x64, Python >= 3.9 — no Rust toolchain needed):

    pip install sezgi
    # or
    uv pip install sezgi

See "Development install (Python)" below to build from a repo checkout
instead.

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

## Author your own algorithm (Python, class-first)

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
`survival` components. `examples/python/oop/custom_de_variant.py` overrides only
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

## Data recipes: feature selection

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
space through it runs with any compound-eligible preset (`DifferentialEvolution`,
`GreyWolfOptimizer`, `TLBO`, ... -- a hybrid, see "Built-in algorithm classes"
below); `GeneticAlgorithm` itself still needs a single-kind space.

## Define your own problem: Problem subclassing

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

## Built-in algorithm classes

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
`ga_int`/`ga_cat`); a multi-block space whose blocks share one kind is
auto-dispatched via `gen/compound` (one generator copy per block); a
genuinely **Mixed space is rejected** with a
`NotImplementedError` naming `gen/compound`; since 0.1.3 (13 since 0.1.4: `AntLion` and
`EvolutionStrategy` joined; 15 since 0.1.5: `GravitationalSearch` and
`BatAlgorithm` joined) the compound-eligible
presets (`DifferentialEvolution` rand1/best1, `GreyWolfOptimizer`,
`WhaleOptimization`, `SineCosineAlgorithm`, `JAYA`, `GrasshopperOptimization`,
`SalpSwarm`, `FireflyAlgorithm`, `FlowerPollination`, `TLBO`, `CuckooSearch`)
instead auto-dispatch on a mixed space to a **hybrid** (the preset's own
generator on float blocks, GA variation on every other block; a space with no
float block gets zero preset-specific variation) —
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
  operators, mixed spaces, and diagnostic problems" above for the
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
  (see "Author your own algorithm (R, class-first)" below), built
  over the SAME `sz_solve_*`/`sz_preset_*` compat internals; `sz_solve_*`/
  `sz_preset_*`/`sz_algorithm` remain R's own persistable-spec path (see
  "Internals & spec files (R)" and "Write your own algorithm (R)"
  below).

`solve()`/`presets.*` are not deprecated and are not scheduled for removal
— they stay compat internals for spec files, benchmarking, and R.

## Experiments & Statistics

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

## Anytime analysis: IOH logs, ECDF, COCO export

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

### Development install (Python)

Building from a repo checkout instead of PyPI (needed only if you're
modifying sezgi itself, or want an unreleased change):

    uv venv py-sezgi/.venv
    uv pip install --python py-sezgi/.venv/bin/python maturin
    cd py-sezgi
    uv run --python .venv/bin/python maturin develop --release

See [Install](https://tdelphi1981.github.io/sezgi/install/) on the
documentation site for the full prerequisites (Rust toolchain) and a
non-`uv` path.

## Quickstart (R)

sezgi is [on CRAN](https://cran.r-project.org/package=sezgi):

    install.packages("sezgi")

For the development version (tracks `main`), install from
[r-universe](https://tdelphi1981.r-universe.dev) instead:

    install.packages("sezgi",
      repos = c("https://tdelphi1981.r-universe.dev", "https://cloud.r-project.org"))

See "Development install (R)" below to build from a repo checkout.

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

R also has a class-first surface now (mirroring Python's — see
"Author your own algorithm (R, class-first)" below): every preset has a
configurable R6 class, and `sezgi::Problem`/`sezgi::Algorithm` are
subclassable bases. `sz_solve_*`/`sz_preset_*`/`sz_algorithm` above remain
fully supported — see "Internals & spec files (R)" further down.

### Development install (R)

Building from a repo checkout instead of r-universe/CRAN (needed only if
you're modifying sezgi itself, or want an unreleased change) — the Rust
core builds via `cargo` as part of R's own install step:

    R CMD INSTALL r-sezgi

## Author your own algorithm (R, class-first)

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
pymoo/jMetal-modeled hook taxonomy). `examples/r/oop/custom_de_variant.R`
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

## Data recipes: feature selection (R)

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
`GeneticAlgorithm`'s own auto-dispatch directly (multi-block single-kind
spaces included, via `gen/compound`); a genuinely **Mixed**
space (more than one distinct block kind) has no `ga_*` preset to
dispatch to, so it needs a hand-built `gen/compound` spec passed to
`sezgi:::sz_solve_r_problem()` directly — the same workaround
`GeneticAlgorithm` itself needs (see "Built-in algorithm classes (R)"
below). **`sz_solve_r_problem()`'s `spec_json` parameter accepts JSON
only** (not TOML) — this is specific to that one entry point, not a
statement about R generally: `sz_solve_mixed_diagnostic` is a
pre-existing TOML-accepting export elsewhere in r-sezgi. The hand-built
spec for a Mixed `MixedTuning` run is therefore written as an equivalent
JSON literal, not TOML.

## Define your own problem: Problem subclassing (R)

`sezgi::Problem` is an R6 base over the new R-callable problem bridge
(`sz_solve_r_problem`, closing the long-standing "R callable-objective
sessions" deferral): subclass it
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

## Built-in algorithm classes (R)

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
rejected** with a clear error naming the limitation; since 0.1.3 (13 since 0.1.4, 15 since 0.1.5) the
compound-eligible presets (same list as the Python frontend) auto-dispatch on
a mixed space to a hybrid (preset generator on float blocks, GA variation on
the rest) —
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

## Internals & spec files (R): sz_solve_*/sz_preset_*/sz_algorithm (compat)

Every class above is a skin over `sz_solve_*`/`sz_preset_*`, still fully
supported and unchanged — reach for them directly (or for
`sz_algorithm()`/`sz_algo_solve()`, the SEPARATE ask/tell scripting
surface, see "Write your own algorithm (R)" below) when you need a
**spec file** (TOML/JSON) to save, diff, or hand-author component by
component (see "Typed operators, mixed spaces, and diagnostic problems"
above for a `gen/compound` mixed-space example) — an R-authored
`Algorithm`/`Problem` subclass has no such spec, exactly like Python's
class surface (see "Author your own algorithm (R, class-first)" above).
`sz_run_experiment`, IOH logging, `sz_per_budget_packages`, bias scanning
(every section below this one) all consume a spec/preset, not a class
instance, and are entirely unaffected by this. All 86
pre-existing exports keep their names and behavior unchanged
— nothing here is deprecated or scheduled for removal.

## Bias scanning

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

Output (tiny budgets, for illustration; consult a method-provenance table
for the numbers a real scan should use):

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

## Multi-objective optimization

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
to v2.
`pop_size` must be a multiple of 4, not merely even — a KanGAL-faithful
tightening of the naive "even, >= 4" rule that NSGA-II's own reference C
code (`nsga2r.c`) enforces for its double-permutation tournament pairing.

### Multi-objective remainders

This closes every MO capability left deferred above: a
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
Ishibuchi, Imada, Setoguchi & Nojima 2018's critique of the choice).
`mo.nsga2(...,
log_dir=, label=)` streams every feasible archive insertion to
`<log_dir>/<label>-s<seed>.moa` (sezgi-moa v1: a versioned header plus one
record per archive insertion, no `.dat`/`.info` IOH mimicry, no COCO
compatibility claim — COCO bbob-biobj and MO-IOHinspector cited as design
precedent, not reproduced); `mo.read_moa(path, at=None)` reconstructs the
archive at any evaluation budget.

Binary-coded MO (zdt5) was scoped to the KanGAL reference exactly at
first — all-Float OR all-Binary spaces only, a MIXED space rejected naming
the mixed-space deferral explicitly. **A later change closes that deferral at the Rust-core
level**: the `nsga2_run` core now accepts any space combining
Float/Int/Categorical/Binary blocks, in any mix (see "Typed operators,
mixed spaces, and diagnostic problems" below) — only a space
containing a `Permutation` block stays rejected (MO permutation search is
out of scope, the error names it explicitly). This is core-only, though:
both `sezgi.mo.nsga2` and `sz_nsga2` still take a problem-name string
(zdt1-6/dtlz1-9/wfg1-9), and every constructible one of those is all-Float
or all-Binary (zdt5) — no mixed MO problem is constructible from either
binding today, so frontend reachability of mixed-space NSGA-II is
deferred. Everything ZDT1-4/6/DTLZ1-7/
`hypervolume_2d` documented in the Multi-objective optimization section above, and the
all-Float/all-Binary paths documented here, stay frozen byte-for-byte
throughout.

R mirrors this 1:1 (`sz_nsga2`, `sz_mo_hypervolume`, `sz_mo_read_moa`),
including two new cross-language bit-equality anchors (an nsga2-on-wfg4
and an nsga2-on-zdt5 scenario) and a `.moa` file byte-identity check
across languages. `examples/python/wfg4_nsga2.py`/`examples/r/wfg4_nsga2.R`
is the matched pair (NSGA-II on WFG4, `log_dir`-logged, reporting archive
size + hypervolume at the nadir x 1.1 reference point). The full
provenance table draws on KanGAL C, the DTLZ 2005 chapter, the ZDT 2000
paper, the WFG EMO2005 paper plus official toolkit via a dead-host Wayback
chain, and the 2012 hypervolume paper, with every toolkit-vs-paper/
C-vs-paper/secondary-library divergence found and closed.

## CEC 2022 benchmark suite

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
scope ruling.

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
sezgi matches the C reference to machine precision at every probed point,
with every discrepancy quoted verbatim from the C source.

R mirrors the direct-evaluation half 1:1 (`sz_cec2022_evaluate`,
`sz_cec2022_f_star`) and also has a `solve()`-integrated
CEC 2022 binding: `sz_solve_cec2022(spec_json, fid, dim, master_seed,
run_id)` (mirrors `sz_solve_bbob`/`sz_solve_tsp` exactly) runs any built-in
preset — including `sz_preset_shade` — against a CEC2022 problem: r-sezgi
previously had direct evaluation only for CEC 2022, with no
engine-solve path.
`examples/r/cec2022_shade.R` now runs the SAME SHADE preset through the
SAME Rust core as `examples/python/cec2022_shade.py`; their `best_f`
outputs are bit-identical (verified via `writeBin`/`struct.pack`, not a
decimal-literal comparison — see that test in
`r-sezgi/tests/testthat/test-cec-tsp.R`).

## CEC 2014 / CEC 2017 benchmark suites

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
than withheld, the same scope ruling CEC 2022's data follows.

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
opfunu-side divergence classes documented per suite, with every
discrepancy quoted verbatim.

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
built for CEC 2022 (`RunKey.suite`, `"sezgi-cec2014"`/
`"sezgi-cec2017"`), with `cec2014-f{fid}d{dim}i{instance}`/
`cec2017-f{fid}d{dim}i{instance}` labels — no changes were needed to the
IOH logger or the labeling helper itself, only the two frontends'
`solve()`/`for_problem` match arms.

## Permutation problems and TSP

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
reproduces `11771.0` too. OX1's own provenance: Davis's actual 1985 paper
describes a different, single-cut-point operator; the two-cut-point cyclic
"OX" implemented here is the field's later, still Davis-attributed,
synthesis — pinned to Cicirello's 2023 worked numeric example. See also
the TSPLIB `nint` rounding rule.

## Typed operators, mixed spaces, and diagnostic problems

Beyond the Float-block (`presets.ga_real`, DE, CMA-ES, ...) and
Permutation-block (`presets.ga_perm`) representations, sezgi now has fused
genetic-algorithm presets for the three remaining block kinds the design
spec's `SearchSpace` enum names: **`presets.ga_bin`** (Binary — KanGAL two-point crossover + bit-flip
mutation, Deb, Pratap, Agarwal & Meyarivan 2002 Sec. IV.A, `p_c=0.9`/
`p_m=1/L`, the same formulas the binary NSGA-II path already validated),
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
of one sub-generator per block (the fused `gen/ga-real`, `gen/ga-int`,
`gen/ga-cat`, `gen/ga-bin`, `gen/ga-perm`, or since 0.1.3 any other
registered generator that is stateless, pop-to-pop and free of internal
evaluation, e.g. `gen/de`, `gen/gwo`), each dispatched against its
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
generators) accept `pc` as an alias for `p_c`, just like the newer typed
families always did the other way; if a block sets both spellings, the
canonical key wins.

The `nsga2_run` core gained the same per-block support (a space combining
any of Float/Int/Categorical/Binary, in any mix — see "Multi-objective
optimization" above); every existing Float/Binary/constrained NSGA-II
golden stays bit-identical throughout — this work added a THIRD `Representation`
case (`Mixed`), without touching the frozen two. This is a core-level
capability only: `sezgi.mo.nsga2`/`sz_nsga2` still accept a problem-name
string, and no mixed-space named problem exists, so mixed spaces are not
reachable through either binding yet (frontend reachability is a recorded
deferral).

**ABC-TSP/permutation authoring** (both `sezgi.AskTellAlgorithm` and
`sz_algorithm`) is now supported too — see "Write your own algorithm,
ask/tell style (Python)" and "Write your own algorithm (R)"
below for the `ctx.kind`/`ctx.n`/`ctx.random_permutation()`/
`ctx.two_opt(tour, i, j)` surface and the `tsp_two_opt` worked example
pair.

The full method-provenance table draws on KanGAL binary reuse, pymoo 0.6.2
source shas, the Eiben & Smith 2015 taxonomy citation with its
print-edition caveat, and the "Deb & Deb 2014 does not cover integers"
research finding, alongside the operator inventory as shipped and every
deferral born from this work.

## Write your own algorithm, ask/tell style (Python) (`AskTellAlgorithm`)

**Renamed.** `sezgi.Algorithm` now names the NEW engine-hosted
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
scanning" section above, but over a plain list of final-position
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

**Permutation/TSP authoring.** `AlgoContext` also supports
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
pre-existing example/test passes unmodified). `examples/python/oop/tsp_two_opt.py`
is the worked example — a first-improvement 2-opt local search from one
random start on TSPLIB berlin52:

    evals_used=2000 best_f=9077 gap=1535 tour_length=9077

R mirrors this exactly via `sz_algorithm`'s `ctx` (`ctx$kind()`, `ctx$n()`,
`ctx$random_permutation()`, `ctx$two_opt(tour, i, j)`) with **1-based**
tour indices (matching `sz_solve_tsp`'s own convention) — see "Write your
own algorithm (R)" below and `examples/r/oop/tsp_two_opt.R`, which
reproduces the SAME numbers above (tour length is index-convention-invariant;
`random_permutation()`'s own draw is bit-identical too, both languages
deriving from the same `RngStream`/Fisher-Yates core).

**Scope rulings.** `AskTellAlgorithm`/`AlgoContext` cover Float and
Permutation problems (v1) — Binary/Int/Categorical and mixed-typed
problems have no ask/tell session type yet (`EvalSession.for_problem`
rejects `sezgi.problems.onemax`/`.int_quadratic`/`.cat_match`/
`.mixed_diagnostic` with a `ValueError` naming the reason; `sezgi.solve()`
still runs them end to end, see "Typed operators, mixed spaces, and
diagnostic problems" above), deferred onward. IOH logging from a
custom `AskTellAlgorithm` covers BBOB and CEC 2022 problems (widened
later on; this superseded an earlier BBOB-only narrowing) —
`sezgi.bbob(...)` and `sezgi.problems.cec2022(...)`
both work with `log_dir=`, but `sezgi.bias.f0(...)` and a raw
`from_callable` handle still raise `ValueError`, matching `sezgi.solve()`'s
own policy for the identical handles exactly (neither has a known optimum,
and `EvalSession.with_log` itself requires one). A known optimum (`f_opt`)
is necessary but not sufficient on its own for `log_dir` — the on-disk IOH
record key also needed a suite discriminator (`RunKey.suite`) so a
CEC 2022 run and a BBOB run sharing `(fid, dim, instance, seed, budget)` no
longer silently merge into one `results_matrix` cell.

## Write your own algorithm (R)

`sz_algorithm(setup, step, name)`/`sz_algo_solve(algo, session, seed)` are
r-sezgi's **R-owned ask/tell scripting surface** — a SEPARATE, unchanged
surface from the engine-hosted class-first `sezgi::Algorithm` above:
unlike Python (which had to rename its own ask/tell ABC to
`AskTellAlgorithm` to free up the `Algorithm` name), R had no name
collision to begin with, so `sz_algorithm`/`sz_algo_solve` keep their
original names and behavior, unchanged by the R6 class-first addition.
They mirror `sezgi.AskTellAlgorithm` above (the Python ask/tell surface,
renamed when the class-first door was added — see "Write your own
algorithm, ask/tell style (Python)" above) —
same driver semantics (`setup(ctx)` once, `step(ctx)` repeatedly until the
budget is exhausted), expressed as two plain closures instead of a
subclass — this ask/tell surface itself is unaffected by the R6
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
runs the SAME statistical KS/AD structural-bias scan as the "Bias scanning"
section above, but over a plain matrix/list of final-position
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

**Permutation/TSP authoring.** `sz_eval_session_tsp(name, budget,
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
environments (unchanged by the R6 addition — no R6 involved anywhere in
`sz_algorithm`/`sz_algo_solve`/`R/algo.R`); the earlier scope ruling
("R stays base-R, no R6") was SUPERSEDED for the class-first surface only
(`Imports: R6 (>= 2.4.0)`), not for this ask/tell surface, which needed no class system to
begin with and was not touched. Float and Permutation problems (v1,
widened from Float-only later on) — Binary/Int/Categorical and mixed-typed
problems have no ask/tell session type yet, matching
`AskTellAlgorithm`/`AlgoContext`'s own scope above exactly. `sz_algorithm`/
`sz_algo_solve` port the pinned `examples/r/gwo.R` script onto this surface
verbatim as `examples/r/oop/gwo.R` — the ONE worked twin proving the
surface (not a full 17-algorithm R wave like `examples/python/oop/`'s —
see `examples/README.md`'s "R authoring example" section for the
draw-order analysis and gate).
**R callable-objective sessions: NARROWED by the class-first addition, not
closed.** It closed the "no engine-solve path for an R callable
at all" half of this deferral: `sezgi::Problem` + `sz_solve_r_problem()`
(above) let an R researcher's own function run INSIDE the Rust engine
loop, per-generation batched, with full determinism. What is STILL open:
a `Problem` subclass does not plug into `sz_algorithm`/`sz_algo_solve`'s
own SESSION-backed ask/tell surface here — there is no
`sz_eval_session`-style counting/logging session type for an arbitrary R
callable, so an R-authored ask/tell algorithm (as opposed to an
engine-hosted `Algorithm`/`PopulationAlgorithm`/`LocalSearch` subclass)
still cannot evaluate its own R function through THIS surface.

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
matched PAIR: NSGA-II on ZDT1, run directly through the Rust core in
both languages (bit-identical output, not merely statistically
comparable). No `specs/nsga2_zdt1.toml` exists — see "Multi-objective
optimization" above for why.

`examples/python/cec2022_shade.py`/`examples/r/cec2022_shade.R` and
`examples/python/tsp_ga_perm.py`/`examples/r/tsp_ga_perm.R` are two more
matched PAIRs (the CEC pair's R-side gap closed later on): SHADE on
CEC 2022 f3, and ga-perm on TSPLIB berlin52, both through
`sezgi.solve()`/`sz_solve_*`, both bit-identical between languages. See
"CEC 2022 benchmark suite" and "Permutation problems and TSP" above.

`examples/python/cec2014_lshade.py`/`examples/r/cec2014_lshade.R`
are a matched PAIR for the CEC 2014 suite: L-SHADE (the CEC 2014
competition's own 1st-place algorithm) on CEC 2014 f1, through
`sezgi.solve()`/`sz_solve_cec2014`, bit-identical between languages. See
"CEC 2014 / CEC 2017 benchmark suites" above.

`examples/python/wfg4_nsga2.py`/`examples/r/wfg4_nsga2.R` is a
matched PAIR for the MO remainders: NSGA-II on WFG4, called directly
through `sezgi.mo.nsga2()`/`sz_nsga2()` (like `nsga2_zdt1.py`/`.R`, not
`sezgi.solve()` — see "Multi-objective optimization" above), with
`log_dir`-logged sezgi-moa v1 output read back via `mo.read_moa()`/
`sz_mo_read_moa()` and reported through the general-`M` `mo.hypervolume()`/
`sz_mo_hypervolume()`, bit-identical between languages (gated by a
committed testthat anchor, `r-sezgi/tests/testthat/test-mo.R`).

`examples/python/onemax_ga.py`/`examples/r/onemax_ga.R` is a matched
PAIR for the typed-operator work: `ga_bin` on the `OneMax` diagnostic
problem, through `sezgi.solve()`/`sz_solve_onemax`, bit-identical between
languages (a deliberately non-converging parameter set, per the
cross-language-anchor convention, so the anchor pins the run's actual
trajectory, not just "reached the trivial known optimum"). See "Typed
operators, mixed spaces, and diagnostic problems" above.

`examples/python/oop/tsp_two_opt.py`/`examples/r/oop/tsp_two_opt.R`
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
algorithm, ask/tell style (Python)" above for the worked walkthrough.

`examples/python/oop/custom_de_variant.py` and
`examples/python/oop/custom_local_search.py` are two more
engine-hosted worked examples, over the NEW class-first `sezgi.Algorithm`
family bases (`PopulationAlgorithm`/`LocalSearch`, not the ask/tell
surface above): a DE/rand/1-shaped `vary()` override and a
`neighbor()`-only local search, both on `sezgi.bbob(1, 10, 1)`. Like
`tsp_two_opt.py`, both sit outside the 17-pair OOP-twin parity gate (no
pure-script counterpart), each gated by its own anchored pytest.
`examples/python/oop/feature_selection.py` demonstrates
`sezgi.recipes.FeatureSelection` recovering a known informative-column
mask via `GeneticAlgorithm`'s Binary auto-dispatch. `tsp_two_opt.py` (and
its 18 `examples/python/oop/*.py` siblings) also now import
`sezgi.AskTellAlgorithm` instead of `sezgi.Algorithm` (renamed when the
class-first door was added — see "Write your own algorithm, ask/tell style
(Python)" above); no example's
printed numbers changed. See "Author your own algorithm (Python,
class-first)" and "Data recipes: feature selection" above, and
`examples/README.md`'s own catalog rows, for the full walkthroughs.

`examples/r/oop/custom_de_variant.R` and `examples/r/oop/
custom_local_search.R` are the R twins of the two Python examples
above, over the NEW class-first `sezgi::Algorithm` family bases
(`PopulationAlgorithm`/`LocalSearch`, not the `sz_algorithm`/
`sz_algo_solve` ask/tell surface above): the same DE/rand/1-shaped
`vary()` override and `neighbor()`-only local search, both on
`sz_builtin_bbob(1, 10, 1)`. Like `tsp_two_opt.R`, both sit outside the
17-pair OOP-twin parity gate (no pure-script counterpart), each gated by
its own `stopifnot()` anchor. `examples/r/oop/feature_selection.R`
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

sezgi is at version 0.1.5. The library ships 29 built-in algorithm
classes over one deterministic Rust engine, Python and R frontends that
are bit-exact against each other for shared algorithms, BBOB/CEC
2014/2017/2022 benchmark suites, NSGA-II multi-objective optimization
over ZDT/DTLZ/WFG, IOH-format logging with ECDF/COCO export, a
statistical-comparison toolkit, and a structural-bias scanner. The full
documentation site is at <https://tdelphi1981.github.io/sezgi/>.
License: MIT. Third-party notices: see
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

## Algorithms

sezgi ships 13 reference algorithm presets (with Rust function names):

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
