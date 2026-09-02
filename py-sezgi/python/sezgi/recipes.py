"""Data recipes (M4-1 Task 6) -- the "optimize against data" door: two
`sezgi.Problem` subclasses that bind a search space to a caller-supplied
dataset/objective, so a data-analysis workflow reaches the engine (and
every built-in wrapper class, `py-sezgi/python/sezgi/builtins.py`) through
the SAME `sezgi.Problem` ABC (`py-sezgi/python/sezgi/problem.py`) every
other Task 1-5 example uses -- no bespoke glue code per dataset.

`FeatureSelection`: wraps a `Binary(n_features)` search over a 2D dataset's
columns, scored by a caller-supplied `scorer(X_sub, y) -> float`.

`MixedTuning`: the general "tune anything" door -- binds a caller-supplied
`objective(x) -> float` over ANY declared `sezgi.Space` (a single block or a
`Mixed` multi-block space), with zero dataset-specific assumptions.

Pure Python; numpy only (no sklearn/scipy import anywhere in this module --
see `FeatureSelection`'s own docstring for the sklearn-shaped scorer note,
kept as a COMMENT, never an import).
"""
import numpy as np

from sezgi.problem import Problem
from sezgi.spaces import Binary


class FeatureSelection(Problem):
    """Binary feature-selection search over a 2D dataset's columns.

    `space()` is `sezgi.Binary(n_features)`: a genotype is a length-
    `n_features` boolean mask (`list[bool]`, per `sezgi.Problem`'s own
    genotype conversion table -- see `problem.py`'s module docstring),
    `True` selecting a column.

    MINIMIZE convention: `scorer(X_sub, y)` must return a LOWER-is-better
    number (e.g. an error/loss/residual metric, NOT accuracy/R^2/any
    higher-is-better score) -- `evaluate` never negates or inverts it. An
    sklearn cross-val ACCURACY scorer must be wrapped to flip its sign
    before it can be used here; a plain error metric (RMSE, log-loss, ...)
    already fits directly. Example (sklearn is NOT a dependency of this
    module or this project's Python package -- this is a comment only, no
    import anywhere in sezgi):

        # from sklearn.linear_model import LogisticRegression
        # from sklearn.model_selection import cross_val_score
        #
        # def scorer(X_sub, y):
        #     # cross_val_score's default scoring is HIGHER-is-better
        #     # (accuracy) -- negate it so lower is better, matching this
        #     # class's minimize convention.
        #     return -cross_val_score(LogisticRegression(), X_sub, y, cv=3).mean()
        #
        # FeatureSelection(X, y, scorer, penalty=0.01)

    `evaluate(mask)` = `scorer(X[:, mask], y) + penalty * (popcount(mask) /
    n_features)` -- the penalty term is a fraction of the FULL feature
    count (`popcount / n_features`, not a raw popcount), so it stays on a
    comparable scale to `scorer`'s own output regardless of `n_features`
    and its size is `penalty` at the all-features mask, `0` at the empty
    mask. Larger `penalty` biases the search toward smaller feature
    subsets; `penalty=0.0` (default) is a pure `scorer`-value search with
    no feature-count preference of its own.

    EMPTY MASK (popcount == 0): `scorer` is NEVER called -- `X[:, mask]`
    would be a `(n_samples, 0)` array, and most real scorers (a model fit,
    a distance-correlation-like statistic, ...) cannot meaningfully score
    zero columns; the popcount == 0 case is instead handled directly, as a
    DOCUMENTED SENTINEL: `evaluate([False, ..., False]) == float("inf")`.
    `+inf` was chosen over, say, a large-but-finite number because it is
    unambiguous (no dataset-dependent magnitude to pick or accidentally
    beat by a legitimately bad but non-empty selection) and it sorts
    correctly under every consumer's own comparison (`<`) with no special
    -casing needed -- the empty mask is simply never the minimizer of any
    run that has at least one non-empty candidate in its population, which
    every population-based search here always does. The bridge's fitness
    channel accepts a non-finite return here (see `problem.py`'s module
    docstring on `evaluate`'s return); this is a genotype-side FITNESS
    value, not a genotype coordinate, so the reverse `block_value_from_py`
    finiteness check (Task 1, Float coordinates only) does not apply to it.
    """

    def __init__(self, X, y, scorer, penalty=0.0):
        self.X = np.asarray(X)
        if self.X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {self.X.shape}")
        self.y = np.asarray(y)
        self.scorer = scorer
        self.penalty = float(penalty)
        self.n_features = self.X.shape[1]

    def space(self):
        return Binary(self.n_features)

    def evaluate(self, mask):
        mask_arr = np.asarray(mask, dtype=bool)
        k = int(mask_arr.sum())
        if k == 0:
            return float("inf")
        score = self.scorer(self.X[:, mask_arr], self.y)
        return score + self.penalty * (k / self.n_features)


class MixedTuning(Problem):
    """The general "tune anything" door: binds a caller-supplied
    `objective(x) -> float` over ANY declared `sezgi.Space` -- a single
    block (`sezgi.Float(...)`, `sezgi.Categorical(...)`, ...) or a
    multi-block `sezgi.Space(...)` (a "Mixed" space, per this crate's own
    `gen/compound` terminology). `evaluate` delegates to `objective`
    UNCHANGED -- `x`'s exact shape follows `sezgi.Problem`'s own genotype
    conversion table (bare value for a single-block space, a tuple of
    per-block values in `space()`'s own block order for a multi-block one).

    A hyperparameter-tuning example over a Float learning-rate block plus a
    Categorical optimizer-choice block (2 hyperparameters, minimizing a
    toy validation loss):

        space = sezgi.Space(sezgi.Float(1e-4, 1e-1, 1), sezgi.Categorical(3, 1))

        def objective(x):
            lr_block, opt_block = x  # per-block values, in space() order
            lr = lr_block[0]         # Float block -> list[float]
            opt_idx = opt_block[0]   # Categorical block -> list[int]
            # ... train/validate with these hyperparameters, return a loss ...
            return validation_loss

        MixedTuning(space, objective)

    Note `GeneticAlgorithm` (`sezgi.builtins`) auto-dispatches only over a
    SINGLE-kind space (all-Float, all-Binary, ...) -- it raises
    `NotImplementedError` on a genuinely mixed space like the one above
    (see `GeneticAlgorithm`'s own docstring); a mixed-space `MixedTuning`
    instance is run via a hand-built `gen/compound` `AlgorithmSpec` passed
    to `sezgi.solve(...)` directly (see `sezgi.problems.mixed_diagnostic`'s
    own doc for a worked mixed-space example), or `evaluate`d directly for
    a non-engine use (grid search, a notebook sanity check, ...). A
    single-kind `MixedTuning` space (e.g. Float-only, tuning several
    continuous hyperparameters at once) runs through `GeneticAlgorithm`
    exactly like any other single-kind `sezgi.Problem`.
    """

    def __init__(self, space, objective):
        self._space = space
        self.objective = objective

    def space(self):
        return self._space

    def evaluate(self, x):
        return self.objective(x)
