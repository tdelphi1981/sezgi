"""sezgi — Rust-core, component-based metaheuristic optimization."""
import json
from types import SimpleNamespace

from sezgi._sezgi import Problem, bbob, from_callable
from sezgi import _sezgi


def solve(spec, problem, master_seed=0, run_id=0, log_dir=None, algo_name=None):
    """Run an algorithm spec against a problem.

    spec: dict (JSON-compatible algorithm spec) or a JSON string.
    """
    if isinstance(spec, dict):
        spec = json.dumps(spec)
    return _sezgi.solve(spec, problem, master_seed=master_seed, run_id=run_id,
                        log_dir=log_dir, algo_name=algo_name)


def run_experiment(spec_toml, journal=None, parallel=True, threads=None):
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
    """
    return _sezgi.run_experiment(spec_toml, journal=journal, parallel=parallel,
                                 threads=threads)


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
    paper_package=_paper_package,
)

__all__ = ["Problem", "bbob", "from_callable", "solve", "run_experiment", "presets", "stats"]
