# sezgi

**sezgi** (Turkish for "intuition") is a Rust-core, component-based metaheuristic
optimization library with Python and R frontends. Design doc: `docs/superpowers/specs/2026-08-27-sezgi-design.md`.

## Quickstart (Python)

    import sezgi

    problem = sezgi.bbob(fid=1, dim=10, instance=1)
    spec = sezgi.presets.de_rand_1(pop_size=50, budget=20_000)
    result = sezgi.solve(spec, problem, master_seed=42, log_dir="logs/")
    print(result["best_f"])   # IOH-format log under logs/

Your own problem (batch evaluation — a single call per population):

    import numpy as np

    def f(X):                       # X: np.ndarray (n, d) float64
        return ((X - 1.0) ** 2).sum(axis=1)

    problem = sezgi.from_callable(f, lo=-5.0, hi=5.0, dim=10)

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

## Examples

`examples/` holds a catalog of 5 labeled-metaphor algorithms (GWO, WOA,
Harmony Search, Cuckoo Search, GOA), each as a triplet: a pure-Python
teaching implementation (stdlib only, driven through `sezgi.EvalSession`),
the same in pure R (`sz_eval_session`), and an `ExperimentSpec` TOML running
sezgi's own built-in, RNG-stream-pinned preset for that algorithm. All three
forms implement the same pinned update equations documented in the
algorithm's Rust component (`crates/components/src/{gwo,woa,hs,cs,goa}.rs`).
See `examples/README.md` for the full catalog table (primary + equivalence-
critique references, and what each pure script teaches vs. its preset).

## Development

    cargo test --workspace --release        # Rust tests
    cd py-sezgi && maturin develop && pytest # Python tests
    R CMD INSTALL --preclean r-sezgi && Rscript -e 'testthat::test_dir("r-sezgi/tests/testthat", package = "sezgi")' # R tests

## Status

M2d-2 (IOH anytime logging + analysis, COCO export, Bayesian Plackett–Luce) **complete** — `run_experiment(..., log_dir=)`/`sz_run_experiment(..., log_dir = )` write a self-contained IOH-profiler-format log tree; `ecdf`/`sz_ecdf` compute anytime-performance curves and `coco_export`/`sz_coco_export` export a `cocopp`-loadable COCO/BBOB archive straight from that tree; `read_ioh_records`/`sz_read_ioh_records` reconstruct run records from disk alone, feeding `per_budget_packages`/`sz_per_budget_packages` unchanged; full Bayesian Plackett–Luce posterior via Gibbs sampling (`stats.bayesian_plackett_luce`/`sz_bayesian_plackett_luce`), replacing the M2d-1 point estimate. See `docs/DECISIONS.md` for the full M2d-2 record. Next: M2d-3 (example triplets: GWO/WOA/HS/CS/GOA, problem-evaluate/ask-tell FFI, `R CMD check --as-cran` + CRAN vendoring dry run). License: MIT.

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
