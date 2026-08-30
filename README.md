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

## Multi-objective optimization (M3-2)

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
`sz_cec2022_f_star`) but does **not** yet have a `solve()`-integrated CEC
2022 binding (no built-in preset can be pointed at a CEC2022 problem from
R today) — see `examples/r/cec2022_shade.R`'s header and the v1.0
readiness checklist below for this disclosed gap.

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
matched PAIRs (M3-3): SHADE on CEC 2022 f3, and ga-perm on TSPLIB
berlin52, both through `sezgi.solve()`/`sz_solve_*`. See "CEC 2022
benchmark suite (M3-3)" and "Permutation problems and TSP (M3-3)" above —
including the disclosed R/CEC2022 solve()-binding gap the first pair's R
script works around.

## Development

    cargo test --workspace --release        # Rust tests
    cd py-sezgi && maturin develop && pytest # Python tests
    R CMD INSTALL --preclean r-sezgi && Rscript -e 'testthat::test_dir("r-sezgi/tests/testthat", package = "sezgi")' # R tests

## Status

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
