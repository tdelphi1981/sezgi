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

Output (tiny budgets, for illustration — see "Method provenance and
defaults" below for the numbers a real scan should use):

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

## Development

    cargo test --workspace --release        # Rust tests
    cd py-sezgi && maturin develop && pytest # Python tests
    R CMD INSTALL --preclean r-sezgi && Rscript -e 'testthat::test_dir("r-sezgi/tests/testthat", package = "sezgi")' # R tests

## Status

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
