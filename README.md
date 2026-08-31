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
2,816,016 bytes; `crates/problems/data/cec2017/`, 111 files, 3,285,318
bytes) comes from each suite's own official repository, neither of which
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

## Write your own algorithm (Python) (M3-4)

`sezgi.Algorithm` is a subclassable ABC for authoring a metaheuristic
entirely in Python (no Rust component graph, no `ExperimentSpec`) while
still getting evaluation counting, all-or-nothing budget enforcement, best
tracking, and optional IOH logging for free from the SAME `EvalSession`
core that backs the `examples/python/oop/` twins. A subclass implements two
methods — `setup(ctx)` (run once) and `step(ctx)` (run repeatedly until the
budget is exhausted) — over an `AlgoContext` (`ctx.dim`, `ctx.bounds`,
`ctx.rng`, `ctx.random_point()`, `ctx.evaluate(points)`, `ctx.best()`,
`ctx.remaining`). A random-search subclass, in full:

    import sezgi


    class RandomSearch(sezgi.Algorithm):
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
sweeps a factory-constructed `Algorithm` across combinations of BBOB
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
    from sezgi.algo import Algorithm, bbob_records

    # RandomSearch as defined above; a second, genuinely different algorithm:
    class HillClimber(Algorithm):
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
    from sezgi.algo import Algorithm

    class RandomSearch(Algorithm):
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

**Scope rulings.** `Algorithm`/`AlgoContext` cover continuous (`Float`-block)
problems only in v1 — a TSP/permutation problem raises `ValueError` from
`EvalSession.for_problem` ("EvalSession supports continuous (float)
problems only"), and authoring a custom permutation-space algorithm this
way is out of scope, deferred onward (a base-R equivalent of this
authoring surface now exists too — `sz_algorithm`/`sz_algo_solve`, see
"Write your own algorithm (R) (M3-5)" below — it shares the same
continuous-only scope). IOH logging from a custom `Algorithm` covers BBOB
and CEC 2022 problems (widened in M3-5 — see `docs/DECISIONS.md`'s M3-5
record; this superseded an earlier BBOB-only narrowing) —
`sezgi.bbob(...)` and `sezgi.problems.cec2022(...)` both work with
`log_dir=`, but `sezgi.bias.f0(...)` and a raw `from_callable` handle still
raise `ValueError`, matching `sezgi.solve()`'s own policy for the identical
handles exactly (neither has a known optimum, and `EvalSession.with_log`
itself requires one). A known optimum (`f_opt`) is necessary but not
sufficient on its own for `log_dir` — the on-disk IOH record key also
needed a suite discriminator (`RunKey.suite`, M3-5) so a CEC 2022 run and a
BBOB run sharing `(fid, dim, instance, seed, budget)` no longer silently
merge into one `results_matrix` cell.

## Write your own algorithm (R) (M3-5)

`sz_algorithm(setup, step, name)`/`sz_algo_solve(algo, session, seed)` are
the base-R mirror of `sezgi.Algorithm` above — same driver semantics
(`setup(ctx)` once, `step(ctx)` repeatedly until the budget is exhausted),
expressed as two plain closures instead of a subclass, since this project
stays base-R only (no R6/S4 — see the scope ruling below). `ctx` is an
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

**Scope rulings.** Base-R only (no R6/S4/other new dependency — CRAN
posture); continuous (`Float`-block) problems only, matching
`Algorithm`/`AlgoContext`'s own v1 scope exactly. `sz_algorithm`/
`sz_algo_solve` port the pinned `examples/r/gwo.R` script onto this surface
verbatim as `examples/r/oop/gwo.R` — the ONE worked twin proving the
surface (not a full 17-algorithm R wave like `examples/python/oop/`'s —
see `examples/README.md`'s "R authoring example (M3-5)" section and
`docs/DECISIONS.md`'s M3-5 record for the draw-order analysis and gate).
**R callable-objective sessions are deferred** (M3-5 scope ruling 5): an R
researcher can evaluate their own R function directly, but a
session-backed counting/logging path for an R callable (the R analogue of
`sezgi.from_callable`) needs a savvy callback design not undertaken this
milestone — on the v1.0 readiness checklist below.

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

## Development

    cargo test --workspace --release        # Rust tests
    cd py-sezgi && maturin develop && pytest # Python tests
    R CMD INSTALL --preclean r-sezgi && Rscript -e 'testthat::test_dir("r-sezgi/tests/testthat", package = "sezgi")' # R tests

## Status

M3-6 (CEC 2014 + CEC 2017 benchmark suites) **complete** — the full CEC
2014 suite (30 fids: unimodal, simple multimodal, hybrid F17-F22,
composition F23-F30) and the CEC 2017 suite (fid `{1} ∪ {3..=30}`, fid 2
officially withdrawn and rejected with a dedicated error), both at dims
`{10,30}` (vendored data: CEC 2014 106 files/2,816,016 bytes, CEC 2017
111 files/3,285,318 bytes — both superseding the plan's pre-research byte
estimates), reusing the CEC 2022-derived shared basic-function library
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
`sezgi.Algorithm` ABC (`py-sezgi/python/sezgi/algo.py`): a subclassable
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
