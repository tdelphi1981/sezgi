"""sezgi — Rust-core, component-based metaheuristic optimization."""
import json
from types import SimpleNamespace

from sezgi._sezgi import Problem, EvalSession, bbob, from_callable
from sezgi import _sezgi


def solve(spec, problem, master_seed=0, run_id=0, log_dir=None, algo_name=None):
    """Run an algorithm spec against a problem.

    spec: dict (JSON-compatible algorithm spec) or a JSON string.
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
    sa=_preset(_sezgi.preset_sa),
    random_search=_preset(_sezgi.preset_random_search),
    nelder_mead=_preset(_sezgi.preset_nelder_mead),
    cmaes=_preset(_sezgi.preset_cmaes),
    cmaes_ipop=_preset(_sezgi.preset_cmaes_ipop),
    es_mu_plus_lambda=_preset(_sezgi.preset_es_mu_plus_lambda),
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

__all__ = ["Problem", "EvalSession", "bbob", "from_callable", "solve", "run_experiment", "presets",
           "stats", "results_matrix", "per_budget_packages", "read_ioh_records", "ecdf",
           "coco_export"]
