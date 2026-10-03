"""sezgi — Rust-core, component-based metaheuristic optimization."""
import functools
import json
from types import SimpleNamespace

from sezgi._sezgi import EvalSession, bbob, from_callable
from sezgi import _sezgi

# Tracks `py-sezgi/Cargo.toml`'s `[package] version` (via `version.workspace
# = true`, resolved from the workspace root `Cargo.toml`'s `[workspace.
# package] version`) -- kept in sync BY HAND, there is no build-time
# templating wiring the two together, so bump this whenever that version
# changes.
__version__ = "0.1.5"

# `sezgi.Problem` is now the subclassable Python ABC
# (`py-sezgi/python/sezgi/problem.py`), NOT the native `#[pyclass]` handle
# type that `bbob(...)`/`problems.onemax(...)`/`from_callable(...)` return
# -- that native type is still fully reachable (as `sezgi._sezgi.Problem`;
# `problem.py`'s `as_native_problem` uses it directly), just no longer
# aliased under this same top-level name. No existing test or example
# referenced `sezgi.Problem` before this change (verified), so this is a
# name-repurposing, not a behavior change to anything working code already
# depended on: the new class hierarchy is meant to become THE documented
# `Problem` symbol.
from sezgi.problem import Problem, as_native_problem
from sezgi.spaces import Float, Int, Categorical, Binary, Permutation, Space


# EvalSession / Problem surface (M3-4 Task 1): EvalSession's ask/tell core
# (evaluate/evals_used/budget/best/f_opt/finish) is no longer BBOB-only.
#
# EvalSession(fid, dim, instance, budget, log_dir=None, algo_name="custom",
#   seed=0) -- the original, frozen BBOB constructor: unchanged.
#
# EvalSession.for_problem(problem, budget, log_dir=None, algo_name="custom",
#   seed=0) -- staticmethod building a session over any Problem handle:
#   continuous (bbob(...), problems.cec2022(...), problems.cec2014(...),
#   problems.cec2017(...), from_callable(...), bias.f0(...)) OR, as of M3-8
#   Task 7, permutation-typed (problems.tsp(...)). Raises ValueError for
#   log_dir on a from_callable(...)/bias.f0(...)/problems.tsp(...) problem:
#   IOH logging is supported for BBOB, CEC 2022, CEC 2014, and CEC 2017
#   problems only (M3-5 widened this from BBOB-only to include CEC 2022, now
#   that the on-disk IOH record key carries a "suite" discriminator -- see
#   read_ioh_records/results_matrix -- so a non-BBOB run no longer silently
#   merges with a BBOB run at the same (fid, dim, instance, seed, budget);
#   M3-6 widened it again to CEC 2014/CEC 2017, riding the same suite-aware
#   machinery). Callable/F0/Tsp have no fid identity or known optimum to log
#   against (Tsp has a known optimum but no suite/fid identity), so they
#   still raise.
#
# EvalSession.kind() -> str: "float" for a continuous-problem session,
#   "permutation" for a problems.tsp(...) session (M3-8 Task 7).
#
# EvalSession.random_permutation() -> list[int]: a uniformly random 0-based
#   tour (a permutation of range(n)), drawn from THIS session's own seeded
#   RNG stream -- deterministic under the session's `seed`, a fresh draw on
#   each call. ValueError if kind() == "float" (M3-8 Task 7).
#
# evaluate(xs) accepts either shape now: a list of length-dim float rows for
#   a "float" session, or a list of 0-based tours (permutations of range(n))
#   for a "permutation" session -- ValueError naming the defect (wrong
#   length, out-of-range city, repeated city) for an invalid tour.
#
# f_opt() now returns float | None (was always float, since only BBOB
#   existed): None for a problem with no analytically known optimum (e.g.
#   from_callable(...)); a session built via the BBOB constructor above
#   always returns a float, unchanged. A problems.tsp(...) session returns
#   the vendored instance's published optimum.
#
# Problem handle accessors, usable on any handle above:
#   p.dim() -> int: the search space's dimensionality (for problems.tsp(...),
#     the number of cities).
#   p.bounds() -> (float, float): the uniform (lo, hi) bounds of a
#     continuous (float) space. ValueError for a non-continuous space (e.g.
#     problems.tsp(...)'s permutation space).
#   p.optimum() -> float | None: the problem's known optimum, or None (a
#     from_callable(...) handle always returns None).
#
# sezgi.AskTellAlgorithm/AlgoContext (py-sezgi/python/sezgi/algo.py, M3-8
#   Task 7; class renamed from sezgi.Algorithm at M4-1 Task 3 -- see the
#   "Algorithm-authoring surfaces" comment near the bottom of this file):
#   AlgoContext now also works over a problems.tsp(...) problem -- ctx.kind
#   ("float"/"permutation"), ctx.n (same value as ctx.dim, a kind-neutral
#   name), ctx.random_permutation(), ctx.two_opt(tour, i, j). ctx.bounds is
#   None and ctx.random_point() raises for a permutation-typed context; see
#   algo.py's own module/class docstrings for the exact contract.
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
    problem: a native `Problem` handle (`sezgi.bbob(...)`, ...) OR a
        `sezgi.Problem` subclass instance -- both routed through
        `sezgi.as_native_problem` (final-review fix 5), so a native handle
        passes through unchanged and a `Problem` subclass is converted the
        same way `Algorithm.run`/the builtin wrapper classes already do;
        anything else raises `as_native_problem`'s own friendly `TypeError`
        instead of pyo3's raw conversion error.

    Returns a dict including `best_x`: the best EVALUATED point (paired with
    `best_f`). For most algorithms this always lies within `problem`'s
    declared bounds. It is not guaranteed to for every algorithm: some (e.g.
    HHO) charge raw, pre-boundary-repair trial points against the budget
    before boundary repair, and such a point can become the reported best if
    it happens to be the run's own minimum.
    """
    if isinstance(spec, dict):
        spec = json.dumps(spec)
    problem = as_native_problem(problem)
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
        Grouping is by algo only, not (algo, suite): a mixed-suite tree
        (BBOB and CEC 2022 runs for the same algo) pools both suites' runs
        into that one algo's curve, so read a per-suite tree or filter
        records by suite first for a curve that is suite-specific.

    Each curve is a dict `{"evals": [...], "proportion": [...]}`: `evals`
    ascending, `proportion` in [0, 1] and monotonically nondecreasing.
    """
    return _sezgi.ecdf(log_root, targets=targets, per_algo=per_algo)


def coco_export(log_root, out_dir):
    """Export an on-disk IOH archive at `log_root` as a COCO/BBOB "old
    format" archive rooted at `out_dir`, so it can be post-processed with
    `cocopp`. Returns the list of written file paths (as strings), sorted
    for determinism.

    BBOB-only: COCO's "old format" IS the BBOB archive format and has no
    CEC counterpart, so this raises `ValueError` if `log_root` holds any
    non-BBOB scenario (naming the offending suite), rather than silently
    merging it into a `bbob`-labeled archive. A mixed BBOB+CEC tree must be
    filtered to its BBOB records before exporting.
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
    # `_preset`'s wrapper is a transparent pass-through (same *args/**kwargs
    # shape as `fn` itself, only post-processing the return value), so
    # `functools.wraps` is a clean fit here -- it recovers `__doc__`,
    # `__name__` and `__qualname__` from the wrapped native `preset_*`
    # function (previously all destroyed: the bare closure above always
    # graded as MISSING, `__name__ == "wrapper"`) and sets `__wrapped__`,
    # so `inspect.signature(sezgi.presets.X)` also now reports the native
    # function's own `(pop_size, budget, ...)` signature instead of the
    # uninformative `(*args, **kwargs)`. Most native `preset_*` bodies are
    # themselves still undocumented (`py-sezgi/src/lib.rs`) -- authoring
    # those doc comments is separate follow-up work, out of this task's
    # scope; this fix only stops the wrapper itself from throwing away
    # whatever documentation already exists (today: `preset_ga_bin`/
    # `ga_int`/`ga_cat`).
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return json.loads(fn(*args, **kwargs))
    return wrapper


# All presets from crates/components/src/presets.rs are exposed here, one
# `sezgi.presets.X(pop_size, budget, ...)` builder per preset (see
# `_preset`'s own docstring for the wrapping mechanism).
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
    # M3-8 Task 9: typed-operator presets, pairing with
    # problems.onemax/int_quadratic/cat_match below.
    ga_bin=_preset(_sezgi.preset_ga_bin),
    ga_int=_preset(_sezgi.preset_ga_int),
    ga_cat=_preset(_sezgi.preset_ga_cat),
)

# `preset_es_mu_plus_lambda` (`py-sezgi/src/lib.rs`) itself carries no doc
# comment yet, so `_preset`'s `functools.wraps` rescue has nothing to
# recover here -- this docstring is assigned directly, converting what used
# to be a `#`-comment block above `presets = SimpleNamespace(...)` (now
# removed) into the real, autodoc-visible thing.
presets.es_mu_plus_lambda.__doc__ = """Builds an ES (mu+lambda) preset.

pop_size, budget: as in every other `sezgi.presets.*` builder.
dist: the mutation distribution, plus its own (flattened,
    all-distributions-superimposed) parameters -- parsed on the Rust side
    by `parse_distribution` (`py-sezgi/src/lib.rs`):

        dist="uniform"                        -- no extra params
        dist="gaussian", mean=0.0, sigma=0.5
        dist="cauchy",   loc=0.0,  scale=1.0
        dist="levy",     alpha=1.5
        dist="student_t", nu=3.0
        dist="laplace",  loc=0.0,  scale=1.0

    Raises `ValueError` ("unknown distribution ...") for an unrecognized
    `dist`."""


def _paper_package(algo_names, problem_names, results, rope=0.0, samples=20000, seed=1):
    """`stats.paper_package(algo_names, problem_names, results, rope=0.0,
    samples=20000, seed=1) -> dict`

    Runs the FULL statistical comparison suite (stats.friedman + Nemenyi CD
    + pairwise stats.wilcoxon with Holm correction + pairwise
    stats.cliffs_delta + pairwise stats.bayesian_signed_rank +
    stats.plackett_luce) over one results matrix in a single call, ready to
    drop into a paper. results: rows = problems (matching problem_names,
    one row each), columns = algorithms (matching algo_names).
    rope/samples/seed are forwarded to the Bayesian sub-tests. Returns a
    dict with keys friedman, nemenyi_cd, pairwise_wilcoxon_holm (list of
    [i, j, p_value] rows, Holm-corrected), cliffs (list of [i, j, delta]
    rows), bayes (list of [i, j, {p_left, p_rope, p_right}] rows),
    plackett_luce, latex_summary, latex_tests -- see each individual
    stats.* function's own docstring for its sub-result's shape."""
    return _sezgi.stats_paper_package(algo_names, problem_names, results,
                                      rope=rope, samples=samples, seed=seed)


def _bayesian_signed_rank(a, b, rope=0.0, samples=20000, seed=1):
    """`stats.bayesian_signed_rank(a, b, rope=0.0, samples=20000, seed=1)
    -> dict`

    A Bayesian alternative to stats.wilcoxon for two PAIRED samples, with a
    Region Of Practical Equivalence (rope: |a[i] - b[i]| <= rope counts as
    "practically equal"). samples draws from the posterior (Monte Carlo);
    seed makes the draw reproducible. Returns a dict (minimize convention,
    lower is better, values sum to 1.0): p_left (P(a practically better
    than b)), p_rope (P(practically equivalent)), p_right (P(b practically
    better than a))."""
    return _sezgi.stats_bayesian_signed_rank(a, b, rope=rope, samples=samples, seed=seed)


def _bayesian_plackett_luce(rankings, samples=2000, burn_in=500, seed=1):
    """`stats.bayesian_plackett_luce(rankings, samples=2000, burn_in=500,
    seed=1) -> dict`

    A Bayesian (posterior-sampling) counterpart to stats.plackett_luce:
    same rankings input (a list of lists, each a permutation of 0..k item
    indices, best-to-worst), but returns a posterior distribution over
    worths instead of a single point estimate. burn_in draws are discarded
    before samples are recorded; seed makes the draw reproducible. Returns
    a dict: mean_worths (posterior mean per item, sums to 1.0), ci_low/
    ci_high (per-item 95% credible interval), p_best (per-item posterior
    probability of the largest worth), samples (the recorded draw count)."""
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
    """Runs `crates/bias`'s structural-bias battery (M3-1 Task 8): `runs`
    independent runs of `spec` on the BIAS-toolbox null problem
    (`bias.f0`), then a KS/AD/Holm test of whether the runs' final
    positions cluster away from uniform.

    spec: dict (JSON-compatible algorithm spec) or a JSON string -- same
        convention as `solve`'s own `spec` argument.
    dim, budget: as in `solve`.
    runs: number of independent runs whose final positions feed the
        KS/AD/Holm battery (mirrors `crates/bias`'s own MIN_RUNS floor).
    seed: master seed for the `runs` independent runs.

    Returns a dict with keys `per_dim_ks` (list of `{d, p_value, n}`),
    `per_dim_ad` (list of `{a2, p_value, n}`), `holm_rejections_ks`,
    `holm_rejections_ad`, `verdict` ("no_evidence" or "evidence"), `detail`
    (`None` for "no_evidence", the evidence-not-accusation detail string
    otherwise), `final_positions` (runs x dim)."""
    return _sezgi.bias_structural(_spec_json(spec), dim, budget, runs=runs, seed=seed)


def _bias_structural_positions(final_positions):
    """The bias bridge for externally-authored algorithms (M3-4 Task 4):
    runs the SAME KS/AD/Holm battery as `bias.structural`, but over
    caller-supplied `final_positions` (each row one run's final best x,
    e.g. collected from repeated `Algorithm.solve(bias.f0(...))` calls)
    instead of an `AlgorithmSpec`-driven engine run.

    final_positions: a runs x dim matrix (list of equal-length rows); dim
        is inferred from row length.

    Returns the SAME shape `bias.structural` returns (see its own
    docstring). Raises `ValueError` for fewer than 5 rows or ragged rows
    (mirrors `crates/bias`'s own MIN_RUNS floor)."""
    return _sezgi.bias_structural_positions(final_positions)


def _bias_central(spec, dim, budget, fids=None, instances_shifted=None, runs_per=20, seed=0):
    """Runs `crates/bias`'s central-bias test (M3-1 Task 8): compares
    `spec`'s performance on a set of BBOB functions at their original
    optimum vs. an instance-shifted (off-center) variant.

    spec, dim, budget: as in `bias.structural`.
    fids: BBOB function IDs to test; defaults to `crates/bias`'s own
        `[1, 4, 13]` when omitted. Raises `ValueError` for a fid in
        `{5, 6, 20, 24}` (not translation-invariant).
    instances_shifted: BBOB instances for the shifted variant; defaults to
        `crates/bias`'s own `[1, 2]` when omitted.
    runs_per: independent runs per (fid, instance, variant).
    seed: master seed.

    Returns a dict with keys `gap_centered`, `gap_shifted`, `wilcoxon`
    (`{w_statistic, z, p_value, n_effective, method}`), `effect`,
    `verdict` ("no_evidence" or "evidence"), `detail` (`None` for
    "no_evidence", the evidence-not-accusation detail string otherwise)."""
    return _sezgi.bias_central(_spec_json(spec), dim, budget, fids=fids,
                               instances_shifted=instances_shifted, runs_per=runs_per, seed=seed)


def _bias_report(spec, dim, budget, seed=0, structural_runs=None, central_fids=None,
                  central_instances=None, central_runs_per=None):
    """Runs the full bias report (M3-1 Task 8): `bias.structural` and
    `bias.central` together, plus a LaTeX-ready summary.

    spec, dim, budget, seed: as in `bias.structural`/`bias.central`.
    structural_runs: forwarded to `bias.structural` as its own `runs`
        argument (that default, 30, applies when omitted).
    central_fids, central_instances, central_runs_per: forwarded to
        `bias.central` as `fids`, `instances_shifted`, `runs_per`
        respectively (their own defaults apply when omitted).

    Returns a dict with keys `structural` (`bias.structural`'s own
    shape), `central` (`bias.central`'s own shape), `signature` (always
    `None` today -- the T6 Rajwar-Deep signature-bias test is DEFERRED in
    `crates/bias` itself: no reachable source fully specifies its
    statistical procedure), `latex_summary` (str, never containing the
    literal "NaN"), `plot_data` (`{final_positions, gap_centered,
    gap_shifted}`)."""
    return _sezgi.bias_report(_spec_json(spec), dim, budget, seed=seed,
                              structural_runs=structural_runs, central_fids=central_fids,
                              central_instances=central_instances,
                              central_runs_per=central_runs_per)


# Bias-scanning namespace (M3-1 Task 8): mirrors crates/bias's public
# structs 1:1 by field name -- see each bound function's own docstring
# above for its exact contract. Every dict returned by structural/
# structural_positions/central/report that carries a verdict has two flat
# keys: `verdict` ("no_evidence" or "evidence") and `detail` (None for
# "no_evidence", the evidence-not-accusation detail string otherwise).
# `bias.f0` binds `_sezgi.bias_f0` directly (no wrapper needed -- it takes
# no spec/JSON argument), so its docstring already reaches Python unchanged.
bias = SimpleNamespace(
    structural=_bias_structural,
    structural_positions=_bias_structural_positions,
    f0=_sezgi.bias_f0,
    central=_bias_central,
    report=_bias_report,
)

def _mo_nsga2(problem, dim, pop_size, budget, m=None, seed=0,
              eta_c=20.0, eta_m=20.0, p_c=0.9, p_m=None,
              p_c_bin=0.9, p_m_bin=None, p_c_cat=0.9, p_m_cat=None,
              k=None, l=None, log_dir=None, label=None):
    """`mo.nsga2(problem, dim, pop_size, budget, m=None, seed=0, eta_c=20.0,
    eta_m=20.0, p_c=0.9, p_m=None, p_c_bin=0.9, p_m_bin=None, p_c_cat=0.9,
    p_m_cat=None, k=None, l=None, log_dir=None, label=None) -> dict`

    Runs NSGA-II (Deb et al. 2002) over one of this module's built-in
    multi-objective problem families -- see this file's own "Problem
    strings" note above for `problem`/`dim`/`m`/`k`/`l`'s per-family
    validation rules.

    Returns a dict with keys individuals (list of float-lists, one per
    final-population member -- a Binary block's bits are flattened to
    0.0/1.0), objectives (list of float-lists, parallel to individuals),
    front0 (list of ints: indices of the final population's non-dominated
    set), evals_used (int), and violations (list of floats, <= 0.0,
    0.0 = feasible; present ONLY when the problem is constrained -- dtlz8/
    dtlz9 today).

    eta_c/eta_m/p_c default to the NSGA-II paper's own pinned experimental
    settings (Deb et al. 2002, Sec. IV.A); p_m=None resolves on the Rust
    side to 1/n_variables. p_c_bin/p_m_bin are the binary-genotype
    counterparts (consulted only for zdt5's all-Binary space); p_c_bin
    defaults to 0.9 (mirroring p_c; the paper gives no verified
    binary-specific default), p_m_bin=None resolves to 1/l (the paper's own
    stated binary default). p_c_cat/p_m_cat are the Categorical-genotype
    counterparts (consulted only for a Mixed space containing a Categorical
    block -- no problem string in this module's own catalog builds one
    today, so they are currently validated but inert, exactly like
    p_c_bin/p_m_bin on every non-zdt5 problem); p_c_cat defaults to 0.9
    (mirroring p_c_bin's own reasoning), p_m_cat=None resolves to 1/n_cat
    (n_cat = the space's total flattened Categorical dimension).

    pop_size must be >= 4 AND a multiple of 4 (a KanGAL-faithful tightening
    of the naive "even, >= 4" rule) or this raises ValueError. When log_dir
    is given, the run is additionally streamed to
    <log_dir>/<label>-s<seed>.moa (sezgi-moa v1 format); label is then
    REQUIRED (ValueError otherwise)."""
    return _sezgi.mo_nsga2(problem, dim, pop_size, budget, m=m, seed=seed,
                           eta_c=eta_c, eta_m=eta_m, p_c=p_c, p_m=p_m,
                           p_c_bin=p_c_bin, p_m_bin=p_m_bin,
                           p_c_cat=p_c_cat, p_m_cat=p_m_cat, k=k, l=l,
                           log_dir=log_dir, label=label)


def _mo_pareto_front(problem, dim, n, m=None, k=None, l=None):
    """`mo.pareto_front(problem, dim, n, m=None, k=None, l=None) -> list of
    float-lists, or None`

    A deterministic `n`-point sample of the analytic Pareto front in
    OBJECTIVE space for one of this module's built-in problem families --
    same `problem`/`dim`/`m`/`k`/`l` mapping and validation as `mo.nsga2`.
    Returns None when the problem has no known analytic front sample at
    this `m` (e.g. DTLZ5/DTLZ6 with m > 3, or WFG1/WFG2 unconditionally)."""
    return _sezgi.mo_pareto_front(problem, dim, n, m=m, k=k, l=l)


# Multi-objective namespace (M3-2 Task 9, extended M3-7 Task 10): binds
# T6's NSGA-II runner, the ZDT/DTLZ/WFG benchmark suites, the exact
# 2-objective and general-M hypervolume, IGD, and sezgi-moa run logging.
# Every f64 is passed through EXACTLY as the Rust core computed it
# (bit-equality mandate: the R bindings assert against these same values),
# so nothing here rounds or reformats a number.
#
# Problem strings: "zdt1".."zdt6" (zdt5 is the binary-coded ZDT5, M3-7 Task
# 6), "dtlz1".."dtlz9" (dtlz8/dtlz9 are constrained, M3-7 Task 2), and
# "wfg1".."wfg9" (M3-7 Task 5/6). `dim`/`m`/`k`/`l` are each meaningful for
# only some families: `dim` is required for zdt1-4/6 and dtlz1-9, REJECTED
# (must be None) for zdt5 (fixed 80-bit layout) and wfg1-9 (dimension is
# derived from k+l); `m` is required for dtlz1-9/wfg1-9, REJECTED for every
# zdt problem (always 2-objective); `k`/`l` are wfg-only (REJECTED for zdt/
# dtlz), defaulting to the toolkit's own recommended values (k=4 for m=2,
# k=2*(m-1) for m>=3; l=20) when omitted. A parameter given where it does
# not apply, or omitted where required, raises ValueError.
#
# mo.nsga2(problem, dim, pop_size, budget, m=None, seed=0, eta_c=20.0,
#   eta_m=20.0, p_c=0.9, p_m=None, p_c_bin=0.9, p_m_bin=None, p_c_cat=0.9,
#   p_m_cat=None, k=None, l=None, log_dir=None, label=None) -> dict with keys individuals (list
#   of float-lists, one per final-population member -- a Binary block's
#   bits are flattened to 0.0/1.0), objectives (list of float-lists,
#   parallel to individuals), front0 (list of ints: indices of the final
#   population's non-dominated set), evals_used (int), and violations
#   (list of floats, <= 0.0, 0.0 = feasible; present ONLY when the problem
#   is constrained -- dtlz8/dtlz9 today). eta_c/eta_m/p_c default to the
#   NSGA-II paper's own pinned experimental settings (Deb et al. 2002,
#   Sec. IV.A); p_m=None resolves on the Rust side to 1/n_variables.
#   p_c_bin/p_m_bin are the binary-genotype counterparts (consulted only
#   for zdt5's all-Binary space); p_c_bin defaults to 0.9 (mirroring p_c;
#   the paper gives no verified binary-specific default), p_m_bin=None
#   resolves to 1/l (the paper's own stated binary default). p_c_cat/
#   p_m_cat are the Categorical-genotype counterparts (consulted only for a
#   Mixed space containing a Categorical block -- no problem string in this
#   module's own catalog builds one today, so they are currently validated
#   but inert, exactly like p_c_bin/p_m_bin on every non-zdt5 problem);
#   p_c_cat defaults to 0.9 (mirroring p_c_bin's own reasoning), p_m_cat=
#   None resolves to 1/n_cat (n_cat = the space's total flattened
#   Categorical dimension). pop_size must
#   be >= 4 AND a multiple of 4 (a KanGAL-faithful tightening of the naive
#   "even, >= 4" rule) or this raises ValueError. When log_dir is given,
#   the run is additionally streamed to
#   <log_dir>/<label>-s<seed>.moa (sezgi-moa v1 format); label is then
#   REQUIRED (ValueError otherwise).
#
# mo.hypervolume_2d(front, ref_point) -> float: the exact 2-objective
#   S-metric hypervolume (front: list of [f1, f2] rows; ref_point: a
#   2-element list).
#
# mo.hypervolume(front, ref_point) -> float: the exact general-M
#   hypervolume (While, Bradstreet & Barone 2012, the WFG algorithm; M=2
#   delegates to hypervolume_2d). ref_point is REQUIRED, with no default --
#   see that module's own "Choosing a reference point" doc for why no
#   value is silently picked.
#
# mo.igd(front, reference_front) -> float: Inverted Generational Distance
#   (Ishibuchi et al. 2015, eq. 12, p=1). Any equal, consistent number of
#   objectives.
#
# mo.pareto_front(problem, dim, n, m=None, k=None, l=None) -> list of
#   float-lists, or None when the problem has no known analytic front
#   sample (e.g. DTLZ5/DTLZ6 with m > 3, or WFG1/WFG2 unconditionally).
#
# mo.evaluate(problem, x, dim=None, m=None, k=None, l=None) -> list of
#   float: direct, one-shot objective evaluation of a decision vector x,
#   bypassing nsga2's population/budget machinery -- added (M3-7 Task 10)
#   so fixture-value tests can pin an exact x, mirroring
#   problems.cec2022_evaluate's convention.
#
# mo.evaluate_constraints(problem, x, dim=None, m=None, k=None, l=None) ->
#   list of float or None: the matching one-shot constraint-row evaluation
#   (None for an unconstrained problem).
#
# mo.read_moa(path, at=None) -> dict with keys algo, problem (the label
#   mo.nsga2 was called with -- kept as the on-disk header key name),
#   m, seed, budget, kind ("float"/"binary"), records (list of dicts:
#   eval_index, objectives, genotype), archive (the reconstructed
#   nondominated archive at evaluation budget `at`; at=None uses the
#   file's own logged budget, i.e. the full run's final archive).
mo = SimpleNamespace(
    nsga2=_mo_nsga2,
    hypervolume_2d=_sezgi.mo_hypervolume_2d,
    hypervolume=_sezgi.mo_hypervolume,
    igd=_sezgi.mo_igd,
    pareto_front=_mo_pareto_front,
    evaluate=_sezgi.mo_evaluate,
    evaluate_constraints=_sezgi.mo_evaluate_constraints,
    read_moa=_sezgi.mo_read_moa,
)

# Problems namespace (M3-3 Task 9): CEC 2022 + TSPLIB, mirroring
# `bias`/`mo`'s own SimpleNamespace-of-bound-functions convention. Every f64
# here is passed through EXACTLY as the Rust core computed it -- no
# rounding/formatting anywhere in this section (T10's R bindings assert
# bit-equality against these same values). M3-6 Task 9 adds CEC 2014 and
# CEC 2017, same trio-per-suite shape as CEC 2022.
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
# problems.cec2014(fid, dim) -> Problem: a solve()-eligible handle for a CEC
#   2014 function (crates/problems/src/cec2014/mod.rs Cec2014::new). fid in
#   1..=30 (the full suite); dim in {10, 30} (the two dims this crate
#   vendors). ValueError for any out-of-domain (fid, dim).
#
# problems.cec2014_evaluate(fid, dim, x) -> float: direct, one-shot
#   evaluation at x, same shape as cec2022_evaluate(...).
#
# problems.cec2014_f_star(fid) -> float: the pinned F_i* = 100*fid bias
#   (Cec2014::f_star). ValueError if fid is outside 1..=30.
#
# problems.cec2017(fid, dim) -> Problem: a solve()-eligible handle for a CEC
#   2017 function (crates/problems/src/cec2017/mod.rs Cec2017::new). fid in
#   {1} union {3..=30} (fid 2, "Sum of Different Powers", was officially
#   withdrawn from the suite -- ValueError with the Rust
#   Cec2017Error::Withdrawn message surfaced verbatim, distinct from an
#   ordinary out-of-range fid); dim in {10, 30}. ValueError for any other
#   out-of-domain (fid, dim).
#
# problems.cec2017_evaluate(fid, dim, x) -> float: direct, one-shot
#   evaluation at x, same shape as cec2022_evaluate(...) (withdrawn fid == 2
#   included).
#
# problems.cec2017_f_star(fid) -> float: the pinned F_i* = 100*fid bias
#   (Cec2017::f_star). ValueError if fid is outside {1} union {3..=30}.
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
# onemax/int_quadratic/cat_match (M3-8 Task 9): Problem handles for the
# diagnostic trio (crates/problems/src/diagnostics.rs, M3-8 Task 5) --
# "diagnostic, not benchmark" (that module's own honest-scope framing):
# minimal, hand-verifiable, single-global-optimum landscapes that exist to
# make ga_bin/ga_int/ga_cat reachable and testable from Python, not to serve
# as research-grade evaluation targets.
#
# problems.onemax(n_bits) -> Problem: Block::Binary{n_bits} (Goldberg 1989's
#   classic GA diagnostic, minimize-the-zero-bit-count re-expression). Known
#   optimum 0.0 (p.optimum()), at the all-true genotype. Pairs with
#   presets.ga_bin(pop_size, budget).
#
# problems.int_quadratic(lo, hi, n) -> Problem: Block::Int{lo,hi,n}, a
#   quadratic bowl around a target drawn once, deterministically, from
#   (lo, hi, n) (see diagnostics.rs's own doc -- no seed argument, per the
#   Task 5 brief's pinned constructor). Known optimum 0.0. ValueError if
#   lo >= hi. Pairs with presets.ga_int(pop_size, budget).
#
# problems.cat_match(k, n, seed) -> Problem: Block::Categorical{k,n}, a
#   Hamming-distance-to-target matching problem; seed IS caller-supplied
#   (unlike int_quadratic). Known optimum 0.0. Pairs with
#   presets.ga_cat(pop_size, budget).
#
# problems.mixed_diagnostic(n_float, n_int, k_cat, n_cat, n_bin) -> Problem
#   (M3-8 Task 9, NOT one of Task 5's brief-pinned diagnostics): a
#   Float+Int+Categorical+Binary mixed-space scaffold problem, added solely
#   so gen/compound (Task 5) is reachable end to end through a
#   Python-authored, mixed-space TOML AlgorithmSpec -- parse the TOML with
#   the stdlib's own tomllib into a dict and pass it straight to solve()
#   (no new solve-side API: solve()'s spec argument already accepts a dict).
#   Mirrors crates/components/src/compound.rs's own test-local MixedProblem
#   exactly: Block::Float{-5,5,n_float} + Block::Int{-5,5,n_int} +
#   Block::Categorical{k_cat,n_cat} + Block::Binary{n_bin}. Not a benchmark;
#   optimum() is always None (no verified reachable target is claimed).
problems = SimpleNamespace(
    cec2022=_sezgi.cec2022,
    cec2022_evaluate=_sezgi.cec2022_evaluate,
    cec2022_f_star=_sezgi.cec2022_f_star,
    cec2014=_sezgi.cec2014,
    cec2014_evaluate=_sezgi.cec2014_evaluate,
    cec2014_f_star=_sezgi.cec2014_f_star,
    cec2017=_sezgi.cec2017,
    cec2017_evaluate=_sezgi.cec2017_evaluate,
    cec2017_f_star=_sezgi.cec2017_f_star,
    tsp=_sezgi.tsp,
    tsp_load=_sezgi.tsp_load,
    tsp_tour_length=_sezgi.tsp_tour_length,
    onemax=_sezgi.onemax,
    int_quadratic=_sezgi.int_quadratic,
    cat_match=_sezgi.cat_match,
    mixed_diagnostic=_sezgi.mixed_diagnostic,
)

__all__ = ["Problem", "EvalSession", "bbob", "from_callable", "solve", "run_experiment", "presets",
           "stats", "results_matrix", "per_budget_packages", "read_ioh_records", "ecdf",
           "coco_export", "bias", "mo", "problems", "Algorithm", "AskTellAlgorithm", "algo",
           # M4-1 Task 4: the two family bases over sezgi.Algorithm.
           "PopulationAlgorithm", "LocalSearch",
           # M4-1 Task 1 (fix round 1): the new space builders / Problem ABC
           # surface. "as_space" is deliberately excluded -- it is a
           # problem.py-internal helper (used by Problem._to_native()), not
           # re-exported into the top-level sezgi namespace at all (only
           # reachable as sezgi.spaces.as_space), so it has no place in an
           # __all__ that only lists names this module actually binds.
           "Float", "Int", "Categorical", "Binary", "Permutation", "Space",
           "as_native_problem",
           # M4-1 Task 6: the two data recipes (sezgi.recipes) plus the
           # module itself.
           "recipes", "FeatureSelection", "MixedTuning"]


# Algorithm-authoring surfaces:
#
# - sezgi.AskTellAlgorithm (formerly the top-level sezgi.Algorithm, M3-4
#   Task 2, renamed M4-1 Task 3): the pure-Python ABC for subclassing an
#   algorithm's setup()/step() over the EvalSession ask/tell core
#   (py-sezgi/python/sezgi/algo.py). sezgi.algo also exposes AlgoContext,
#   SolveResult, BudgetExhausted for direct import, plus a module-level
#   `Algorithm = AskTellAlgorithm` compat alias (see algo.py's own
#   docstring note) -- `sezgi.algo.Algorithm` still resolves to this same
#   class.
#
# - sezgi.Algorithm (NEW, M4-1 Task 3, py-sezgi/python/sezgi/algorithm.py):
#   REPURPOSES the top-level `Algorithm` name for the engine-hosted
#   class-first base -- `generate(self, pop, ctx)` (required),
#   `initialize(self, n, ctx)` / `validate_space(self, space)` (optional),
#   `run(problem, budget, seed=0, pop_size=50, log_dir=None)`, running
#   INSIDE the Rust engine loop via `_sezgi.solve_with_py_generator` (Task
#   2's `PyGenerator` bridge, Task 3's `PyInitializer` bridge). Same
#   name-repurposing precedent as Task 1's `sezgi.Problem` (this
#   milestone) -- no existing test/example imported the OLD `sezgi.
#   Algorithm` binding after this task's own enumerated import updates
#   moved every one of them onto `sezgi.AskTellAlgorithm`.
#
# Imported last since both sezgi/algo.py and sezgi/algorithm.py themselves
# import sezgi-package submodules at their own top level (module-level
# attribute access on the FULL `sezgi` package only happens inside
# AskTellAlgorithm.solve()/Algorithm.run(), at call time, well after this
# module has finished initializing).
# - sezgi.PopulationAlgorithm / sezgi.LocalSearch (NEW, M4-1 Task 4,
#   py-sezgi/python/sezgi/algorithm.py): family bases over sezgi.Algorithm
#   with sensible defaults. PopulationAlgorithm: select(pop, k, ctx)
#   (default k-fold binary tournament, size 2) + vary(parents, ctx)
#   (REQUIRED) -- generate() is composed from both (does not override
#   initialize()). LocalSearch: neighbor(x, ctx) (REQUIRED) + accept(f_old,
#   f_new, ctx) (default greedy) -- pop_size=1 semantics, current_x/
#   current_f instance state; see algorithm.py's own class docstrings for
#   the exact select() draw pattern and the accept() design (what it does
#   and does not control given replace/mu-plus-lambda's own behavior at
#   pop_size=1).
from sezgi import algo
from sezgi.algo import AskTellAlgorithm
from sezgi.algorithm import Algorithm, PopulationAlgorithm, LocalSearch

# M4-1 Task 5: built-in algorithm wrapper classes (one per
# crates/components/src/presets.rs builder, table-driven -- see
# sezgi/builtins.py's own module doc for the full reconciliation) plus
# NSGA2 (a thin sezgi.mo.nsga2 skin, not a presets.rs builder). Imported
# last for the same reason as sezgi.algo/sezgi.algorithm above (builtins.py
# itself does `import sezgi` at module scope, resolved lazily via each
# class's own run() method, well after this module has finished
# initializing).
from sezgi import builtins
from sezgi.builtins import (
    GeneticAlgorithm, DifferentialEvolution, EvolutionStrategy,
    ParticleSwarm, SimulatedAnnealing, SHADE, LSHADE, CMAES, CMAESIpop,
    NelderMead, RandomSearch, GreyWolfOptimizer, WhaleOptimization,
    HarmonySearch, CuckooSearch, GrasshopperOptimization,
    SineCosineAlgorithm, JAYA, MothFlameOptimization, SalpSwarm,
    FireflyAlgorithm, BatAlgorithm, FlowerPollination, TLBO, HarrisHawks,
    AntLion, ArtificialBeeColony, GravitationalSearch, NSGA2,
)

# Final-review fix 1 (2026-09-02): all 29 builtin wrapper classes above
# (the milestone's headline deliverable) were bound at module scope but
# never listed in `__all__`, so `from sezgi import *` could not see any of
# them -- exactly the defect class Task 1's own fix round existed for
# (test_new_oop_symbols_are_in_dunder_all), which T5 nonetheless missed for
# all 29 names. Derived programmatically from `sezgi.builtins.__all__`
# (that module's own single source of truth, already used to
# table-generate the classes themselves) rather than re-enumerated by hand
# here, so this exact omission cannot regress silently again. "builtins"
# (the submodule name) is added alongside "algo"/"recipes" below, which are
# already listed for the same submodule-reachability reason.
__all__ += ["builtins"] + list(builtins.__all__)

# M4-1 Task 6: the two "optimize against data" recipes -- sezgi.Problem
# subclasses over sezgi/recipes.py, following the same
# "import the module last, re-export its public names" convention as
# sezgi.algo/sezgi.algorithm/sezgi.builtins above (recipes.py itself only
# touches numpy at import time -- no `import sezgi` there -- but kept in
# this same tail block for house-style consistency: every class-first
# surface built on top of sezgi.Problem/sezgi.Space is imported here, in
# the same place). sezgi.recipes remains reachable as its own submodule
# too (`import sezgi.recipes`), matching sezgi.builtins's own precedent.
from sezgi import recipes
from sezgi.recipes import FeatureSelection, MixedTuning
