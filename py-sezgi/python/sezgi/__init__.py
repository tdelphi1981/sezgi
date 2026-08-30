"""sezgi — Rust-core, component-based metaheuristic optimization."""
import json
from types import SimpleNamespace

from sezgi._sezgi import Problem, EvalSession, bbob, from_callable
from sezgi import _sezgi


# EvalSession / Problem surface (M3-4 Task 1): EvalSession's ask/tell core
# (evaluate/evals_used/budget/best/f_opt/finish) is no longer BBOB-only.
#
# EvalSession(fid, dim, instance, budget, log_dir=None, algo_name="custom",
#   seed=0) -- the original, frozen BBOB constructor: unchanged.
#
# EvalSession.for_problem(problem, budget, log_dir=None, algo_name="custom",
#   seed=0) -- staticmethod building a session over any continuous (float)
#   Problem handle: bbob(...), problems.cec2022(...), from_callable(...), or
#   bias.f0(...). Raises ValueError for problems.tsp(...) (a permutation
#   space, not continuous) and for log_dir on any non-BBOB problem: IOH
#   logging is currently supported for BBOB problems only, matching
#   solve()'s pre-existing policy -- a known optimum (which cec2022(...)
#   also has) is necessary but not sufficient, since the on-disk IOH record
#   key carries no suite discriminator and a non-BBOB run would silently
#   merge with a BBOB run at the same (fid, dim, instance, seed, budget).
#
# f_opt() now returns float | None (was always float, since only BBOB
#   existed): None for a problem with no analytically known optimum (e.g.
#   from_callable(...)); a session built via the BBOB constructor above
#   always returns a float, unchanged.
#
# Problem handle accessors, usable on any handle above:
#   p.dim() -> int: the search space's dimensionality.
#   p.bounds() -> (float, float): the uniform (lo, hi) bounds of a
#     continuous (float) space. ValueError for a non-continuous space (e.g.
#     problems.tsp(...)'s permutation space).
#   p.optimum() -> float | None: the problem's known optimum, or None (a
#     from_callable(...) handle always returns None).
#
# from_callable(f, lo, hi, dim, vectorized=True) -- vectorized fixes f's
#   calling convention for EVERY consumer of the returned handle (solve()
#   AND EvalSession.for_problem alike: one handle, one contract everywhere).
#   vectorized=True (default, unchanged from before M3-4): f is called ONCE
#     per evaluate_batch call, with the whole population as a single 2-D
#     (n, dim) float64 numpy array, and must return n values.
#   vectorized=False: f is called ONCE PER POINT, with a 1-D length-dim
#     float64 numpy array, and must return a scalar float -- natural for
#     EvalSession's ask/tell callers (which evaluate individually-generated
#     points) and for an ordinary single-point objective function.


def solve(spec, problem, master_seed=0, run_id=0, log_dir=None, algo_name=None):
    """Run an algorithm spec against a problem.

    spec: dict (JSON-compatible algorithm spec) or a JSON string.

    Returns a dict including `best_x`: the best EVALUATED point (paired with
    `best_f`). For most algorithms this always lies within `problem`'s
    declared bounds. It is not guaranteed to for every algorithm: some (e.g.
    HHO) charge raw, pre-boundary-repair trial points against the budget
    before boundary repair, and such a point can become the reported best if
    it happens to be the run's own minimum.
    """
    if isinstance(spec, dict):
        spec = json.dumps(spec)
    return _sezgi.solve(spec, problem, master_seed=master_seed, run_id=run_id,
                        log_dir=log_dir, algo_name=algo_name)


def run_experiment(spec_toml, journal=None, parallel=True, threads=None, log_dir=None):
    """Run an [`ExperimentSpec`] (parsed from TOML) and return its run
    records as a list of dicts with keys: algo, fid, dim, instance, seed,
    budget, best_f, f_opt, gap, evals_used, wall_secs.

    spec_toml: TOML string of an ExperimentSpec.
    journal: optional path to a JSONL checkpoint file; if given, runs via
        the checkpoint/resume executor (already-completed runs, by key,
        are skipped and loaded from the journal instead of re-executed).
    parallel: if True, runs via rayon in parallel; if False, runs
        sequentially. Applies both with and without a journal (a journaled
        parallel run checkpoints each record incrementally as it
        completes). Both modes produce bit-identical results.
    threads: thread pool size for the parallel executor; None uses rayon's
        global pool.
    log_dir: optional directory to also write an IOH-profiler-format log
        tree to. Only runs this call actually EXECUTES are logged -- a run
        resumed from an existing `journal` was executed in a prior process
        and is never re-logged, so the on-disk IOH tree does not grow on
        resume. Read back with `read_ioh_records`/`ecdf`/`coco_export`.
    """
    return _sezgi.run_experiment(spec_toml, journal=journal, parallel=parallel,
                                 threads=threads, log_dir=log_dir)


def read_ioh_records(log_root, budgets):
    """Reconstruct run records from an on-disk IOH archive at `log_root`
    (as written by `run_experiment(..., log_dir=...)` or
    `solve(..., log_dir=...)`), one record per (run, budget) pair.

    log_root: root directory of the IOH archive.
    budgets: list of evaluation budgets to reconstruct a best-so-far value
        at (see `sezgi_bench::ioh_records`'s doc comment for the exact
        `best_f`/`evals_used` semantics, and the curtailed-view-vs-
        independent-run distinction for a budget smaller than a run's
        logged budget).

    Returns the SAME record-dict shape `run_experiment` returns, so
    `results_matrix`/`per_budget_packages` accept it unchanged.
    """
    return _sezgi.read_ioh_records(log_root, budgets)


def ecdf(log_root, targets=None, per_algo=True):
    """ECDF (anytime performance) curve(s) over an on-disk IOH archive.

    log_root: root directory of the IOH archive.
    targets: precision targets; None uses the COCO-convention 51-value
        default target set (10^(2 - 0.2*k) for k = 0..=50).
    per_algo: if True (default), returns a list of `(algo, curve)` pairs,
        one per distinct algorithm in the archive, in first-appearance
        order; if False, returns a single pooled curve over every scenario.

    Each curve is a dict `{"evals": [...], "proportion": [...]}`: `evals`
    ascending, `proportion` in [0, 1] and monotonically nondecreasing.
    """
    return _sezgi.ecdf(log_root, targets=targets, per_algo=per_algo)


def coco_export(log_root, out_dir):
    """Export an on-disk IOH archive at `log_root` as a COCO/BBOB "old
    format" archive rooted at `out_dir`, so it can be post-processed with
    `cocopp`. Returns the list of written file paths (as strings), sorted
    for determinism.
    """
    return _sezgi.coco_export(log_root, out_dir)


def results_matrix(records, budget, aggregate="mean"):
    """Build a `sezgi.stats.paper_package`-shaped results matrix for one
    `budget` from `run_experiment`'s record dicts.

    records: list of dicts as returned by `run_experiment` (or any list of
        dicts with the same fields: algo, fid, dim, instance, seed, budget,
        best_f, f_opt, evals_used).
    budget: only records with this budget are used.
    aggregate: how to combine a (problem, algorithm) cell's per-seed gaps
        (`best_f - f_opt`) into one number: "mean" or "median".

    Returns (algo_names, problem_labels, matrix): `matrix[i][j]` is the
    aggregated gap of `algo_names[j]` on `problem_labels[i]`. Problem labels
    are `f{fid}d{dim}i{instance}`; both lists are ordered by first
    appearance in `records`. Raises ValueError if a (problem, algorithm)
    pair present for one algorithm/problem is missing for another at this
    budget (an incomplete experiment) -- see
    `sezgi_bench::reporting::results_matrix`.
    """
    return _sezgi.results_matrix(records, budget, aggregate=aggregate)


def per_budget_packages(records, rope=0.0, samples=20000, seed=1, aggregate="mean"):
    """Build one `sezgi.stats.paper_package` PER DISTINCT BUDGET present in
    `records`, in ascending budget order.

    Piotrowski et al. (2025) show algorithm rankings on benchmark
    comparisons can flip depending on which evaluation budget is examined,
    so this makes multi-budget reporting the default rather than a single,
    arbitrarily-chosen budget's report: compare algorithms per budget, never
    pooled across budgets.

    records: list of dicts as returned by `run_experiment`.
    rope, samples, seed: forwarded to the Bayesian signed-rank test inside
        each budget's paper_package (same as `sezgi.stats.paper_package`).
    aggregate: "mean" or "median" -- see `results_matrix`.

    Returns a list of `[budget, package_dict]` pairs; each `package_dict`
    has exactly the shape `sezgi.stats.paper_package` returns.
    """
    return _sezgi.per_budget_packages(records, rope=rope, samples=samples, seed=seed,
                                      aggregate=aggregate)


def _preset(fn):
    def wrapper(*args, **kwargs):
        return json.loads(fn(*args, **kwargs))
    return wrapper


# All presets from crates/components/src/presets.rs are exposed here.
#
# es_mu_plus_lambda takes the mutation distribution as a `dist` string plus
# its (flattened, all-distributions-superimposed) params, parsed on the Rust
# side by `parse_distribution` in py-sezgi/src/lib.rs:
#   dist="uniform"                    -> no extra params
#   dist="gaussian", mean=0.0, sigma=0.5
#   dist="cauchy",   loc=0.0,  scale=1.0
#   dist="levy",     alpha=1.5
#   dist="student_t", nu=3.0
#   dist="laplace",  loc=0.0,  scale=1.0
# An unrecognized `dist` raises ValueError ("unknown distribution ...").
presets = SimpleNamespace(
    de_rand_1=_preset(_sezgi.preset_de_rand_1),
    de_best_1=_preset(_sezgi.preset_de_best_1),
    jde=_preset(_sezgi.preset_jde),
    shade=_preset(_sezgi.preset_shade),
    lshade=_preset(_sezgi.preset_lshade),
    ga_real=_preset(_sezgi.preset_ga_real),
    pso=_preset(_sezgi.preset_pso),
    gwo=_preset(_sezgi.preset_gwo),
    woa=_preset(_sezgi.preset_woa),
    harmony_search=_preset(_sezgi.preset_harmony_search),
    cuckoo_search=_preset(_sezgi.preset_cuckoo_search),
    goa=_preset(_sezgi.preset_goa),
    sca=_preset(_sezgi.preset_sca),
    jaya=_preset(_sezgi.preset_jaya),
    mfo=_preset(_sezgi.preset_mfo),
    ssa=_preset(_sezgi.preset_ssa),
    firefly=_preset(_sezgi.preset_firefly),
    bat=_preset(_sezgi.preset_bat),
    fpa=_preset(_sezgi.preset_fpa),
    tlbo=_preset(_sezgi.preset_tlbo),
    hho=_preset(_sezgi.preset_hho),
    alo=_preset(_sezgi.preset_alo),
    abc=_preset(_sezgi.preset_abc),
    gsa=_preset(_sezgi.preset_gsa),
    sa=_preset(_sezgi.preset_sa),
    random_search=_preset(_sezgi.preset_random_search),
    nelder_mead=_preset(_sezgi.preset_nelder_mead),
    cmaes=_preset(_sezgi.preset_cmaes),
    cmaes_ipop=_preset(_sezgi.preset_cmaes_ipop),
    es_mu_plus_lambda=_preset(_sezgi.preset_es_mu_plus_lambda),
    ga_perm=_preset(_sezgi.preset_ga_perm),
)

def _paper_package(algo_names, problem_names, results, rope=0.0, samples=20000, seed=1):
    return _sezgi.stats_paper_package(algo_names, problem_names, results,
                                      rope=rope, samples=samples, seed=seed)


def _bayesian_signed_rank(a, b, rope=0.0, samples=20000, seed=1):
    return _sezgi.stats_bayesian_signed_rank(a, b, rope=rope, samples=samples, seed=seed)


def _bayesian_plackett_luce(rankings, samples=2000, burn_in=500, seed=1):
    return _sezgi.stats_bayesian_plackett_luce(rankings, samples=samples, burn_in=burn_in, seed=seed)


# Statistics namespace: mirrors crates/stats's public functions. Accepts
# Python lists (list of lists for matrices) or numpy arrays.
#
# stats.wilcoxon(a, b) returns a dict with keys w_statistic, z, p_value,
# n_effective, and method ("exact" or "normal_approx"). "exact" is used when
# n_effective <= 25 and there are no zero differences or tied |d| ranks; the
# exact p-value formula is semver-pinned (see
# crates/stats/src/pairwise.rs::wilcoxon_signed_rank doc comment).
stats = SimpleNamespace(
    friedman=_sezgi.stats_friedman,
    wilcoxon=_sezgi.stats_wilcoxon,
    cliffs_delta=_sezgi.stats_cliffs_delta,
    cliffs_magnitude=_sezgi.stats_cliffs_magnitude,
    bayesian_signed_rank=_bayesian_signed_rank,
    plackett_luce=_sezgi.stats_plackett_luce,
    bayesian_plackett_luce=_bayesian_plackett_luce,
    paper_package=_paper_package,
)


def _spec_json(spec):
    """spec: dict (JSON-compatible algorithm spec) or a JSON string —
    same convention as `solve`'s own `spec` argument."""
    return json.dumps(spec) if isinstance(spec, dict) else spec


def _bias_structural(spec, dim, budget, runs=30, seed=0):
    return _sezgi.bias_structural(_spec_json(spec), dim, budget, runs=runs, seed=seed)


def _bias_structural_positions(final_positions):
    return _sezgi.bias_structural_positions(final_positions)


def _bias_central(spec, dim, budget, fids=None, instances_shifted=None, runs_per=20, seed=0):
    return _sezgi.bias_central(_spec_json(spec), dim, budget, fids=fids,
                               instances_shifted=instances_shifted, runs_per=runs_per, seed=seed)


def _bias_report(spec, dim, budget, seed=0, structural_runs=None, central_fids=None,
                  central_instances=None, central_runs_per=None):
    return _sezgi.bias_report(_spec_json(spec), dim, budget, seed=seed,
                              structural_runs=structural_runs, central_fids=central_fids,
                              central_instances=central_instances,
                              central_runs_per=central_runs_per)


# Bias-scanning namespace (M3-1 Task 8): mirrors crates/bias's public
# structs 1:1 by field name.
#
# Every dict returned here that carries a verdict has two flat keys:
# `verdict` ("no_evidence" or "evidence") and `detail` (None for
# "no_evidence", the evidence-not-accusation detail string otherwise).
#
# bias.structural(spec, dim, budget, runs=30, seed=0) -> dict with keys
#   per_dim_ks (list of {d, p_value, n}), per_dim_ad (list of
#   {a2, p_value, n}), holm_rejections_ks, holm_rejections_ad,
#   verdict, detail, final_positions (runs x dim).
#
# bias.f0(dim, seed) -> Problem (M3-4 Task 4): a handle for the BIAS-toolbox
#   null problem (every evaluation an independent U(0,1) draw over [0,1]^dim,
#   uncorrelated with the queried point) -- usable with
#   EvalSession.for_problem / Algorithm.solve exactly like any other
#   continuous Problem handle. optimum() is always None (no landscape to have
#   an optimum), so log_dir is rejected the same way it is for
#   from_callable(...).
#
# bias.structural_positions(final_positions) -> dict (M3-4 Task 4): the
#   bias bridge for externally-authored algorithms -- runs the SAME
#   KS/AD/Holm battery as bias.structural, but over caller-supplied
#   final_positions (each row one run's final best x, e.g. collected from
#   repeated Algorithm.solve(bias.f0(...)) calls) instead of an
#   AlgorithmSpec-driven engine run. dim is inferred from row length. Same
#   return shape as bias.structural. Raises ValueError for fewer than 5 rows
#   or ragged rows (mirrors crates/bias's own MIN_RUNS floor).
#
# bias.central(spec, dim, budget, fids=None, instances_shifted=None,
#   runs_per=20, seed=0) -> dict with keys gap_centered, gap_shifted,
#   wilcoxon ({w_statistic, z, p_value, n_effective, method}), effect,
#   verdict, detail. fids/instances_shifted default to crates/bias's own
#   [1, 4, 13] / [1, 2] when omitted. Raises ValueError for a fid in
#   {5, 6, 20, 24} (not translation-invariant).
#
# bias.report(spec, dim, budget, seed=0, structural_runs=None,
#   central_fids=None, central_instances=None, central_runs_per=None) ->
#   dict with keys structural (bias.structural's own shape), central
#   (bias.central's own shape), signature (always None today -- the T6
#   Rajwar-Deep signature-bias test is DEFERRED in crates/bias itself:
#   no reachable source fully specifies its statistical procedure),
#   latex_summary (str, never containing the literal "NaN"), plot_data
#   ({final_positions, gap_centered, gap_shifted}).
bias = SimpleNamespace(
    structural=_bias_structural,
    structural_positions=_bias_structural_positions,
    f0=_sezgi.bias_f0,
    central=_bias_central,
    report=_bias_report,
)

def _mo_nsga2(problem, dim, pop_size, budget, m=None, seed=0,
              eta_c=20.0, eta_m=20.0, p_c=0.9, p_m=None):
    return _sezgi.mo_nsga2(problem, dim, pop_size, budget, m=m, seed=seed,
                           eta_c=eta_c, eta_m=eta_m, p_c=p_c, p_m=p_m)


def _mo_pareto_front(problem, dim, n, m=None):
    return _sezgi.mo_pareto_front(problem, dim, n, m=m)


# Multi-objective namespace (M3-2 Task 9): binds T6's NSGA-II runner, the
# ZDT/DTLZ benchmark suites, and the exact 2-objective hypervolume/IGD
# indicators. Every f64 is passed through EXACTLY as the Rust core computed
# it (bit-equality mandate: T10's R bindings assert against these same
# values), so nothing here rounds or reformats a number.
#
# Problem strings: "zdt1", "zdt2", "zdt3", "zdt4", "zdt6" (ZDT5 is a
# binary-coded problem, out of scope) and "dtlz1".."dtlz7". `m` (number of
# objectives) is DTLZ-only and REQUIRED there; passing `m` for a zdt problem
# raises ValueError (zdt problems are always 2-objective).
#
# mo.nsga2(problem, dim, pop_size, budget, m=None, seed=0, eta_c=20.0,
#   eta_m=20.0, p_c=0.9, p_m=None) -> dict with keys individuals (list of
#   float-lists, one per final-population member), objectives (list of
#   float-lists, parallel to individuals), front0 (list of ints: indices of
#   the final population's non-dominated set), evals_used (int).
#   eta_c/eta_m/p_c default to the NSGA-II paper's own pinned experimental
#   settings (Deb et al. 2002, Sec. IV.A); p_m=None resolves on the Rust
#   side to 1/n_variables. pop_size must be >= 4 AND a multiple of 4 (a
#   KanGAL-faithful tightening of the naive "even, >= 4" rule) or this
#   raises ValueError.
#
# mo.hypervolume_2d(front, ref_point) -> float: the exact 2-objective
#   S-metric hypervolume (front: list of [f1, f2] rows; ref_point: a
#   2-element list).
#
# mo.igd(front, reference_front) -> float: Inverted Generational Distance
#   (Ishibuchi et al. 2015, eq. 12, p=1). Any equal, consistent number of
#   objectives.
#
# mo.pareto_front(problem, dim, n, m=None) -> list of float-lists, or None
#   when the problem has no known analytic front sample at this m (e.g.
#   DTLZ5/DTLZ6 with m > 3).
mo = SimpleNamespace(
    nsga2=_mo_nsga2,
    hypervolume_2d=_sezgi.mo_hypervolume_2d,
    igd=_sezgi.mo_igd,
    pareto_front=_mo_pareto_front,
)

# Problems namespace (M3-3 Task 9): CEC 2022 + TSPLIB, mirroring
# `bias`/`mo`'s own SimpleNamespace-of-bound-functions convention. Every f64
# here is passed through EXACTLY as the Rust core computed it -- no
# rounding/formatting anywhere in this section (T10's R bindings assert
# bit-equality against these same values).
#
# problems.cec2022(fid, dim) -> Problem: a solve()-eligible handle for a CEC
#   2022 function (crates/problems/src/cec2022/mod.rs Cec2022::new), usable
#   exactly like sezgi.bbob(...): `sezgi.solve(spec, sezgi.problems.cec2022(1,
#   10), ...)`. fid in 1..=12; dim in {2, 10, 20} (fid 6-8, the hybrid
#   functions, reject dim=2). ValueError for any out-of-domain (fid, dim).
#
# problems.cec2022_evaluate(fid, dim, x) -> float: direct, one-shot
#   evaluation at x (a length-dim list of floats), bypassing solve()'s
#   budget/engine machinery. Same (fid, dim) domain as cec2022(...); also
#   ValueError if len(x) != dim.
#
# problems.cec2022_f_star(fid) -> float: the report's pinned F_i* bias
#   (Cec2022::f_star). Does not depend on dim. ValueError if fid is outside
#   1..=12.
#
# problems.tsp(name) -> Problem: a solve()-eligible handle for a VENDORED
#   TSPLIB instance (Tsp::vendored: "berlin52", "eil51", or "st70"), usable
#   exactly like sezgi.bbob(...) / problems.cec2022(...). ValueError for any
#   other name.
#
# problems.tsp_load(name_or_text) -> dict: loads a TSPLIB EUC_2D instance,
#   either a vendored instance name OR raw TSPLIB ".tsp" file text (a
#   vendored name is tried first; only if that name is unrecognized does it
#   fall back to parsing name_or_text as raw TSPLIB text). Returns
#   {"name": str, "n_cities": int, "coords": [(x, y), ...], "known_optimum":
#   float or None (None unless name_or_text is a vendored name)}.
#
# problems.tsp_tour_length(name_or_text, tour) -> float: closed-tour length
#   of a 0-based tour (a permutation of range(n_cities)) on the instance
#   named/parsed by name_or_text, via the module's nint-rounded EUC_2D sum
#   (see tsp.rs's module doc). ValueError if name_or_text does not resolve,
#   or if tour is not a valid permutation of range(n_cities) (wrong length,
#   an out-of-range entry, or a repeated entry).
#
# ga-perm (crates/components/src/perm.rs's fused OX-crossover + swap-mutation
# generator) runs through the SAME solve()/presets path as every other
# labeled preset -- no problems.* entry point of its own:
#   sezgi.solve(sezgi.presets.ga_perm(pop_size, budget),
#               sezgi.problems.tsp("berlin52"), master_seed=...)
problems = SimpleNamespace(
    cec2022=_sezgi.cec2022,
    cec2022_evaluate=_sezgi.cec2022_evaluate,
    cec2022_f_star=_sezgi.cec2022_f_star,
    tsp=_sezgi.tsp,
    tsp_load=_sezgi.tsp_load,
    tsp_tour_length=_sezgi.tsp_tour_length,
)

__all__ = ["Problem", "EvalSession", "bbob", "from_callable", "solve", "run_experiment", "presets",
           "stats", "results_matrix", "per_budget_packages", "read_ioh_records", "ecdf",
           "coco_export", "bias", "mo", "problems", "Algorithm", "algo"]


# Algorithm-authoring surface (M3-4 Task 2): sezgi.Algorithm is the pure-
# Python ABC for subclassing an algorithm's setup()/step() over the
# EvalSession ask/tell core. sezgi.algo also exposes AlgoContext,
# SolveResult, BudgetExhausted for direct import. Imported last since
# sezgi/algo.py itself does `import sezgi` (module-level attribute access
# happens only inside Algorithm.solve(), at call time, well after this
# module has finished initializing).
from sezgi import algo
from sezgi.algo import Algorithm
