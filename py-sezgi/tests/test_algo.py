"""M3-4 Task 2: sezgi.Algorithm ABC, AlgoContext, and the setup/step driver.

Exercises the pure-Python algorithm-authoring surface over T1's generalized
EvalSession ask/tell core: subclass sezgi.Algorithm, implement setup()/step(),
call solve(). RandomSearch below is the reference toy subclass used
throughout (and by later M3-4 tasks porting real algorithms onto this ABC).
"""
import pytest
import sezgi
from sezgi.algo import AlgoContext, BudgetExhausted, SolveResult


class RandomSearch(sezgi.Algorithm):
    """Uniform random search: one batch of `batch` points per step."""
    def __init__(self, batch=10):
        self.batch = batch

    def setup(self, ctx):
        ctx.evaluate([ctx.random_point() for _ in range(self.batch)])

    def step(self, ctx):
        ctx.evaluate([ctx.random_point() for _ in range(self.batch)])


def test_solve_bbob_exhausts_budget_and_returns_result():
    res = RandomSearch(batch=10).solve(sezgi.bbob(1, 5, 1), budget=200, seed=42)
    assert isinstance(res, SolveResult)
    assert res.evals_used == 200          # 10-per-step fits 200 exactly
    assert res.budget == 200
    assert res.f_opt is not None
    assert res.gap == res.best_f - res.f_opt
    assert len(res.best_x) == 5
    assert res.algo == "randomsearch"     # default: cls.__name__.lower()

def test_solve_is_deterministic():
    a = RandomSearch().solve(sezgi.bbob(1, 5, 1), budget=100, seed=7)
    b = RandomSearch().solve(sezgi.bbob(1, 5, 1), budget=100, seed=7)
    assert (a.best_f, a.best_x, a.evals_used) == (b.best_f, b.best_x, b.evals_used)

def test_solve_cec2022_and_gap_vs_f_star():
    res = RandomSearch().solve(sezgi.problems.cec2022(3, 10), budget=100, seed=1)
    assert res.f_opt == 600.0
    assert res.gap >= 0.0

def test_solve_from_callable_gap_none():
    p = sezgi.from_callable(lambda x: sum(v * v for v in x), -1.0, 1.0, 2, vectorized=False)
    res = RandomSearch().solve(p, budget=50, seed=1)
    assert res.f_opt is None and res.gap is None

def test_partial_final_batch_stops_cleanly():
    # budget 95, batch 10: 9 full batches (90), 10th raises BudgetExhausted
    # inside step -> driver stops; the last 5 evals are never spent.
    res = RandomSearch(batch=10).solve(sezgi.bbob(1, 5, 1), budget=95, seed=3)
    assert res.evals_used == 90

def test_no_progress_step_raises():
    class Lazy(sezgi.Algorithm):
        def setup(self, ctx): ctx.evaluate([ctx.random_point()])
        def step(self, ctx): pass  # consumes nothing
    with pytest.raises(RuntimeError, match="consumed no budget"):
        Lazy().solve(sezgi.bbob(1, 2, 1), budget=10, seed=1)

def test_solve_error_path_still_finishes_ioh_archive(tmp_path):
    """Final-review fix (N3): session.finish() must run even when solve()
    raises, or a run that dies mid-flight silently loses its whole IOH
    archive. Reuses the no-progress RuntimeError guard as the error path,
    but with log_dir set -- the setup() eval is already logged before
    step() triggers the guard, so a *.dat file must exist afterward."""
    class Lazy(sezgi.Algorithm):
        def setup(self, ctx): ctx.evaluate([ctx.random_point()])
        def step(self, ctx): pass  # consumes nothing -> RuntimeError
    with pytest.raises(RuntimeError, match="consumed no budget"):
        Lazy().solve(sezgi.bbob(1, 2, 1), budget=10, seed=1,
                     log_dir=str(tmp_path))
    assert any(tmp_path.rglob("*.dat")), "IOH archive lost on the error path"

def test_setup_alone_never_evaluating_raises():
    class Never(sezgi.Algorithm):
        def setup(self, ctx): pass
        def step(self, ctx): pass
    with pytest.raises(RuntimeError):
        Never().solve(sezgi.bbob(1, 2, 1), budget=10, seed=1)

def test_ctx_surface():
    seen = {}
    class Probe(sezgi.Algorithm):
        def setup(self, ctx):
            seen["dim"], seen["bounds"] = ctx.dim, ctx.bounds
            seen["budget"] = ctx.budget
            ctx.evaluate([ctx.random_point()])
            seen["used"], seen["remaining"] = ctx.evals_used, ctx.remaining
            seen["best"] = ctx.best()
        def step(self, ctx):
            ctx.evaluate([ctx.random_point()])
    Probe().solve(sezgi.bbob(1, 3, 1), budget=5, seed=2)
    assert seen["dim"] == 3 and seen["bounds"] == (-5.0, 5.0)
    assert seen["budget"] == 5 and seen["used"] == 1 and seen["remaining"] == 4
    assert seen["best"] is not None and len(seen["best"][0]) == 3

def test_custom_name_and_log_dir(tmp_path):
    class Named(RandomSearch):
        name = "my-rs"
    res = Named().solve(sezgi.bbob(1, 2, 1), budget=20, seed=1,
                        log_dir=str(tmp_path))
    assert res.algo == "my-rs"
    assert any(tmp_path.rglob("*.dat"))   # IOH tree written

def test_abstract_methods_enforced():
    with pytest.raises(TypeError):
        sezgi.Algorithm()  # abstract


def test_bbob_records_shape_and_mixing():
    recs = sezgi.algo.bbob_records(RandomSearch, fids=[1, 2], dims=[2],
                                   instances=[1], seeds=[0, 1], budget=50)
    assert len(recs) == 4  # 2 fids x 1 dim x 1 instance x 2 seeds
    for r in recs:
        assert set(r) == {"algo", "fid", "dim", "instance", "seed", "budget",
                          "best_f", "f_opt", "gap", "evals_used", "wall_secs"}
        assert r["gap"] == r["best_f"] - r["f_opt"]
        assert r["wall_secs"] >= 0.0
    # mixes with run_experiment records in one stats call: same problem set,
    # a second "algorithm" from the same helper under a different name.
    # Two behaviorally identical algorithms produce all-zero per-problem differences,
    # which per_budget_packages rejects (n_effective = 0). The second algorithm
    # must therefore genuinely differ in search behavior.
    class HillClimber(sezgi.Algorithm):
        """Simple local-perturbation hill-climber: evaluate a random point,
        then iteratively perturb the best point found so far."""
        name = "hillclimber"
        def setup(self, ctx):
            ctx.evaluate([ctx.random_point()])
        def step(self, ctx):
            best_x, best_f = ctx.best()
            perturbed = [
                best_x[i] + ctx.rng.gauss(0, 0.1 * (ctx.bounds[1] - ctx.bounds[0]))
                for i in range(ctx.dim)
            ]
            # Clamp to bounds
            perturbed = [
                max(ctx.bounds[0], min(ctx.bounds[1], p)) for p in perturbed
            ]
            ctx.evaluate([perturbed])
    both = recs + sezgi.algo.bbob_records(HillClimber, fids=[1, 2], dims=[2],
                                          instances=[1], seeds=[0, 1], budget=50)
    algos, problems, matrix = sezgi.results_matrix(both, budget=50)
    assert sorted(algos) == ["hillclimber", "randomsearch"]
    assert len(problems) == 2 and len(matrix) == 2
    pkgs = sezgi.per_budget_packages(both)
    assert len(pkgs) == 1 and pkgs[0][0] == 50

def test_bbob_records_fresh_instance_per_run():
    created = []
    def factory():
        a = RandomSearch()
        created.append(a)
        return a
    sezgi.algo.bbob_records(factory, fids=[1], dims=[2], instances=[1],
                            seeds=[0, 1, 2], budget=20)
    assert len(created) == 3


# M3-4 Task 4: the bias bridge -- sezgi.bias.f0 (an f0 Problem handle usable
# with EvalSession.for_problem / Algorithm.solve) and
# sezgi.bias.structural_positions (the statistics-only structural-bias scan,
# runnable on final positions collected from ANY externally-authored
# Algorithm, not just a spec-driven engine run).

def test_bias_f0_handle_shape():
    p = sezgi.bias.f0(3, seed=0)
    assert p.dim() == 3
    assert p.bounds() == (0.0, 1.0)
    assert p.optimum() is None

def test_bias_f0_rejects_log_dir(tmp_path):
    with pytest.raises(ValueError):
        sezgi.EvalSession.for_problem(sezgi.bias.f0(2, seed=0), budget=10,
                                      log_dir=str(tmp_path))

def test_structural_positions_uniform_vs_clustered():
    import random
    rng = random.Random(0)
    uniform = [[rng.random() for _ in range(3)] for _ in range(30)]
    out = sezgi.bias.structural_positions(uniform)
    assert set(out) == {"per_dim_ks", "per_dim_ad", "holm_rejections_ks",
                        "holm_rejections_ad", "verdict", "detail", "final_positions"}
    # ANCHORED (per this project's convention): measured at seed=0. Raw
    # per-dim KS p-values = [0.0413, 0.2090, 0.2773], AD p-values =
    # [0.0384, 0.1123, 0.3726] -- the smallest raw p (0.0384) Holm-adjusts to
    # 3*0.0384 = 0.1152, well clear of alpha=0.01 (not a boundary case):
    # holm_rejections_ks=0, holm_rejections_ad=0.
    assert out["verdict"] == "no_evidence", out
    clustered = [[0.5 + rng.gauss(0.0, 1e-6) for _ in range(3)] for _ in range(30)]
    out2 = sezgi.bias.structural_positions(clustered)
    # ANCHORED: measured at seed=0 (continued stream). Raw per-dim KS/AD
    # p-values are all ~2.8e-07 / ~1.0e-07 -- tight clustering around 0.5
    # (sigma=1e-6) is unambiguously non-uniform, nowhere near a boundary.
    assert out2["verdict"] == "evidence", out2

def test_structural_positions_validation():
    with pytest.raises(ValueError):
        sezgi.bias.structural_positions([[0.5, 0.5]] * 4)   # < 5 runs
    with pytest.raises(ValueError):
        sezgi.bias.structural_positions([[0.5], [0.5, 0.5]] * 3)  # ragged

def test_custom_algorithm_bias_scan_end_to_end():
    positions = []
    for run in range(30):
        res = RandomSearch().solve(sezgi.bias.f0(3, seed=run), budget=60,
                                   seed=run)
        positions.append(res.best_x)
    out = sezgi.bias.structural_positions(positions)
    # ANCHORED: measured at this exact config (dim=3, budget=60, seeds 0..29).
    # Raw per-dim KS p-values = [0.8306, 0.4651, 0.9841], AD p-values =
    # [0.7795, 0.4678, 0.9660] -- nowhere near alpha=0.01, consistent with
    # RandomSearch's uniform per-batch resampling having no directional
    # operator to induce structural bias against f0's own random landscape.
    assert out["verdict"] == "no_evidence"  # uniform sampler must scan clean
