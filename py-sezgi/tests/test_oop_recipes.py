"""M4-1 Task 6: `sezgi.recipes.FeatureSelection`/`MixedTuning` -- the two
data-recipe `sezgi.Problem` subclasses.

Covers the brief's own pinned test list:
(a) FeatureSelection end-to-end recovery, anchored (fixed seed, exact mask
    + best_f) -- the SAME dataset/scorer/penalty/budget/seed as
    `examples/python/oop/feature_selection.py`.
(b) penalty term monotonicity: same scorer value, more features -> worse
    fitness (unit-level, no engine).
(c) empty-mask documented behavior (unit-level): `evaluate` on the all-
    False mask returns `float("inf")` and never calls `scorer`.
(d) MixedTuning smoke, in two genuinely different exercises (see the
    module's own docstring on why GeneticAlgorithm cannot run a Mixed
    space): (d1) a Float-only MixedTuning instance run via
    `sezgi.GeneticAlgorithm`'s auto-dispatch; (d2) a Float+Categorical
    Mixed-space MixedTuning instance run through a hand-built
    `gen/compound` AlgorithmSpec (the SAME pattern
    `test_oop_problem.py`'s own `MixedDiagnostic` test uses), asserting
    the objective receives the pinned per-block tuple types. (d2) goes
    through the real `_to_native()` -> engine -> Python callback round
    trip rather than a bare `evaluate(x)` unit call with a hand-built x,
    because the thing actually worth proving is that the ENGINE delivers
    the pinned per-block types to a Mixed-space MixedTuning instance, not
    merely that `evaluate` forwards whatever `x` it is given (which is
    true by construction from its one-line body).
(e) scorer exceptions propagate unchanged.
"""
import tomllib

import numpy as np
import pytest
import sezgi
from sezgi.recipes import FeatureSelection, MixedTuning


# ---------------------------------------------------------------------
# (a) FeatureSelection end-to-end recovery, anchored.
# ---------------------------------------------------------------------

_SEED = 98
_N_SAMPLES, _N_FEATURES = 20, 8
_INFORMATIVE = (1, 3, 6)
_COEFS = np.array([3.0, -2.0, 1.5])
_NOISE_STD = 0.2
_PENALTY = 0.3
_POP_SIZE = 20
_BUDGET = 200
_GA_SEED = 42


def _make_dataset():
    """Same construction as examples/python/oop/feature_selection.py's own
    make_dataset() -- kept independently here (not imported) so this test
    does not depend on the example script's own module layout, matching
    this suite's existing convention of defining anchored fixtures inline
    (test_oop_problem.py's Sphere/MiniOneMax)."""
    rng = np.random.default_rng(_SEED)
    X = rng.standard_normal((_N_SAMPLES, _N_FEATURES))
    noise = rng.normal(scale=_NOISE_STD, size=_N_SAMPLES)
    y = X[:, _INFORMATIVE] @ _COEFS + noise
    return X, y


def _ols_rss_scorer(X_sub, y):
    design = np.column_stack([np.ones(X_sub.shape[0]), X_sub])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coefs
    return float(np.sum(resid ** 2))


def test_feature_selection_recovers_informative_mask_anchored():
    """ANCHORED (per this project's convention): measured with the exact
    dataset/scorer/penalty/budget/seed above (mirroring
    examples/python/oop/feature_selection.py's own constants) --
    best_f=0.3607495482907973, best_x recovers EXACTLY the 3 informative
    columns (indices 1, 3, 6) and no others."""
    X, y = _make_dataset()
    problem = FeatureSelection(X, y, _ols_rss_scorer, penalty=_PENALTY)
    res = sezgi.GeneticAlgorithm(pop_size=_POP_SIZE).run(
        problem, budget=_BUDGET, seed=_GA_SEED)

    assert res.evals_used == _BUDGET
    assert res.best_f == pytest.approx(0.3607495482907973)
    assert len(res.best_x) == _N_FEATURES
    assert all(isinstance(v, bool) for v in res.best_x)
    recovered_mask = tuple(bool(v) for v in res.best_x)
    informative_mask = tuple(i in _INFORMATIVE for i in range(_N_FEATURES))
    assert recovered_mask == informative_mask


def test_feature_selection_recovery_is_deterministic():
    X, y = _make_dataset()
    problem = FeatureSelection(X, y, _ols_rss_scorer, penalty=_PENALTY)
    a = sezgi.GeneticAlgorithm(pop_size=_POP_SIZE).run(problem, budget=_BUDGET, seed=_GA_SEED)
    b = sezgi.GeneticAlgorithm(pop_size=_POP_SIZE).run(problem, budget=_BUDGET, seed=_GA_SEED)
    assert a.best_f == b.best_f
    assert a.best_x == b.best_x


def test_feature_selection_brute_force_confirms_unique_global_minimum():
    """Cross-check the anchor above against a full 2**8 = 256-mask brute
    force enumeration -- confirms the informative mask is the UNIQUE global
    minimum under this penalty (not merely the mask the GA happened to
    find), so the anchored recovery test is verifying something real."""
    import itertools

    X, y = _make_dataset()
    problem = FeatureSelection(X, y, _ols_rss_scorer, penalty=_PENALTY)
    informative_mask = tuple(i in _INFORMATIVE for i in range(_N_FEATURES))

    best_mask, best_f, second_f = None, None, None
    for bits in itertools.product([False, True], repeat=_N_FEATURES):
        f = problem.evaluate(list(bits))
        if best_f is None or f < best_f:
            second_f = best_f
            best_f, best_mask = f, bits
        elif second_f is None or f < second_f:
            second_f = f

    assert best_mask == informative_mask
    assert second_f > best_f, "the informative mask must be the UNIQUE minimum"


# ---------------------------------------------------------------------
# (b) Penalty term monotonicity (unit-level, no engine): with a scorer
# that returns the SAME value regardless of which/how-many columns are
# selected, fitness must strictly increase with popcount whenever
# penalty > 0.
# ---------------------------------------------------------------------

def _constant_scorer(X_sub, y):
    return 5.0


def test_penalty_term_is_monotonic_in_popcount():
    X = np.zeros((4, _N_FEATURES))
    y = np.zeros(4)
    problem = FeatureSelection(X, y, _constant_scorer, penalty=0.4)

    f_one = problem.evaluate([True] + [False] * (_N_FEATURES - 1))
    f_two = problem.evaluate([True, True] + [False] * (_N_FEATURES - 2))
    f_all = problem.evaluate([True] * _N_FEATURES)

    assert f_one < f_two < f_all
    # exact values: constant scorer (5.0) + penalty * popcount / n_features
    assert f_one == pytest.approx(5.0 + 0.4 * 1 / _N_FEATURES)
    assert f_two == pytest.approx(5.0 + 0.4 * 2 / _N_FEATURES)
    assert f_all == pytest.approx(5.0 + 0.4 * _N_FEATURES / _N_FEATURES)


def test_zero_penalty_ignores_popcount():
    X = np.zeros((4, _N_FEATURES))
    y = np.zeros(4)
    problem = FeatureSelection(X, y, _constant_scorer, penalty=0.0)

    f_one = problem.evaluate([True] + [False] * (_N_FEATURES - 1))
    f_all = problem.evaluate([True] * _N_FEATURES)
    assert f_one == f_all == 5.0


# ---------------------------------------------------------------------
# (c) Empty-mask documented behavior (unit-level): float("inf"), and
# `scorer` is NEVER called.
# ---------------------------------------------------------------------

def test_empty_mask_returns_inf_and_never_calls_scorer():
    calls = []

    def recording_scorer(X_sub, y):
        calls.append(X_sub.shape)
        return 0.0

    X = np.zeros((4, _N_FEATURES))
    y = np.zeros(4)
    problem = FeatureSelection(X, y, recording_scorer, penalty=1.0)

    f = problem.evaluate([False] * _N_FEATURES)
    assert f == float("inf")
    assert calls == [], "scorer must never be called for an empty mask"


def test_empty_mask_is_never_the_minimizer_among_nonempty_candidates():
    X = np.zeros((4, _N_FEATURES))
    y = np.zeros(4)
    problem = FeatureSelection(X, y, _constant_scorer, penalty=1.0)
    empty_f = problem.evaluate([False] * _N_FEATURES)
    nonempty_f = problem.evaluate([True] + [False] * (_N_FEATURES - 1))
    assert nonempty_f < empty_f


# ---------------------------------------------------------------------
# (e) Scorer exceptions propagate unchanged.
# ---------------------------------------------------------------------

class _CustomScorerError(RuntimeError):
    pass


def test_scorer_exception_propagates_unchanged():
    def failing_scorer(X_sub, y):
        raise _CustomScorerError("scorer blew up on purpose")

    X = np.zeros((4, _N_FEATURES))
    y = np.zeros(4)
    problem = FeatureSelection(X, y, failing_scorer, penalty=0.0)

    with pytest.raises(_CustomScorerError, match="scorer blew up on purpose"):
        problem.evaluate([True] + [False] * (_N_FEATURES - 1))


# ---------------------------------------------------------------------
# (d) MixedTuning smoke -- (d1) Float-only through GeneticAlgorithm
# auto-dispatch, (d2) a genuinely Mixed (Float+Categorical) space through
# a hand-built gen/compound AlgorithmSpec.
# ---------------------------------------------------------------------

def test_mixed_tuning_float_only_runs_via_ga_auto_dispatch():
    """GeneticAlgorithm auto-dispatches over a single-kind space (all-
    Float here) -- MixedTuning is a genuine drop-in Problem subclass for
    this path, no different from any other single-block Problem."""
    def objective(x):
        return sum((v - 1.5) ** 2 for v in x)

    problem = MixedTuning(sezgi.Float(-5.0, 5.0, 3), objective)
    res = sezgi.GeneticAlgorithm(pop_size=20).run(problem, budget=2000, seed=1)
    assert res.best_f < 1e-3
    assert all(v == pytest.approx(1.5, abs=0.05) for v in res.best_x)


_MIXED_TUNING_COMPOUND_TOML = """
name = "mixed-tuning-oop"
pop_size = 12

[init]
kind = "init/uniform"

[boundary]
kind = "boundary/clamp"

[[stages]]

[stages.generator]
kind = "gen/compound"
blocks = [
  { kind = "gen/ga-real", tournament_k = 2, pc = 0.9, eta_c = 15.0, eta_m = 20.0 },
  { kind = "gen/ga-cat", tournament_k = 2, p_c = 0.9 },
]

[stages.replacer]
kind = "replace/mu-plus-lambda"

[termination]
budget = 200
"""


def test_mixed_tuning_over_mixed_space_gets_pinned_per_block_types():
    """A genuinely Mixed (Float + Categorical) MixedTuning instance, run
    through a hand-built gen/compound AlgorithmSpec (GeneticAlgorithm
    itself rejects a Mixed space -- see sezgi.builtins.GeneticAlgorithm's
    own docstring) -- the objective asserts it receives EXACTLY the
    per-block tuple types Problem's own conversion table pins (Float ->
    list[float], Categorical -> list[int]), proving MixedTuning genuinely
    carries a Mixed space through _to_native() -> the engine -> the
    Python callback, not just through a bare evaluate(x) call."""
    calls = []

    def objective(x):
        calls.append(x)
        assert isinstance(x, tuple) and len(x) == 2
        float_block, cat_block = x
        assert isinstance(float_block, list)
        assert all(type(v) is float for v in float_block)
        assert isinstance(cat_block, list)
        assert all(type(v) is int and 0 <= v < 3 for v in cat_block)
        return sum(v * v for v in float_block) + sum(1 for c in cat_block if c != 0)

    space = sezgi.Space(sezgi.Float(-5.0, 5.0, 2), sezgi.Categorical(3, 2))
    problem = MixedTuning(space, objective)
    spec = tomllib.loads(_MIXED_TUNING_COMPOUND_TOML)
    r = sezgi.solve(spec, problem._to_native(), master_seed=42)

    assert isinstance(r["best_f"], float)
    assert len(calls) > 0, "objective must have been called at least once"


# ---------------------------------------------------------------------
# Export surface.
# ---------------------------------------------------------------------

def test_recipes_symbols_are_in_dunder_all():
    for name in ("recipes", "FeatureSelection", "MixedTuning"):
        assert name in sezgi.__all__


def test_recipes_are_problem_subclasses():
    assert issubclass(FeatureSelection, sezgi.Problem)
    assert issubclass(MixedTuning, sezgi.Problem)
