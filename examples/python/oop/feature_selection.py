"""OOP example: `sezgi.recipes.FeatureSelection` recovering a known
informative-column mask from a fixed synthetic dataset via
`sezgi.GeneticAlgorithm`'s space-driven auto-dispatch (M4-1 Task 5) --
`FeatureSelection.space()` is `Binary(n_features)`, so the wrapper
auto-dispatches to `sezgi.presets.ga_bin`.

Dataset (deterministic, pure numpy, no sklearn/scipy anywhere): a fixed
20x8 design matrix `X` drawn from a seeded `numpy.random.default_rng`, with
exactly 3 informative columns (indices 1, 3, 6) that linearly determine `y`
up to Gaussian noise; the other 5 columns are pure noise, uncorrelated with
`y` beyond sampling coincidence. Least-squares in-sample residual sum of
squares (RSS) is only ever weakly reduced by adding MORE columns (a
noise column can never make RSS worse), so a pure-RSS search over-selects
every column -- `FeatureSelection`'s own `penalty` term (a fraction of the
full feature count) is what makes the 3-column informative subset the
actual global minimum; see this script's own `PENALTY` constant and
`test_oop_recipes.py`'s brute-force cross-check (all 2**8 = 256 masks) of
that claim.

`scorer(X_sub, y)`: ordinary least squares (`numpy.linalg.lstsq`, with an
intercept column prepended) residual sum of squares -- a minimal,
dependency-free stand-in for the sklearn cross-val scorer
`FeatureSelection`'s own docstring sketches in a comment.
"""
import numpy as np

import sezgi
from sezgi.recipes import FeatureSelection

SEED = 98
N_SAMPLES, N_FEATURES = 20, 8
INFORMATIVE = (1, 3, 6)
COEFS = np.array([3.0, -2.0, 1.5])
NOISE_STD = 0.2
PENALTY = 0.3
POP_SIZE = 20
BUDGET = 200
GA_SEED = 42


def make_dataset():
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    noise = rng.normal(scale=NOISE_STD, size=N_SAMPLES)
    y = X[:, INFORMATIVE] @ COEFS + noise
    return X, y


def scorer(X_sub, y):
    """Ordinary-least-squares residual sum of squares (with an intercept
    column) -- lower is better, matching FeatureSelection's minimize
    convention."""
    design = np.column_stack([np.ones(X_sub.shape[0]), X_sub])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coefs
    return float(np.sum(resid ** 2))


def main():
    X, y = make_dataset()
    problem = FeatureSelection(X, y, scorer, penalty=PENALTY)
    res = sezgi.GeneticAlgorithm(pop_size=POP_SIZE).run(
        problem, budget=BUDGET, seed=GA_SEED)

    mask = tuple(bool(v) for v in res.best_x)
    informative_mask = tuple(i in INFORMATIVE for i in range(N_FEATURES))
    recovered = mask == informative_mask
    mask_str = "".join("1" if v else "0" for v in mask)

    print(f"feature_selection (oop): evals_used={res.evals_used} "
          f"best_f={res.best_f:.10g} popcount={sum(mask)} "
          f"mask={mask_str} recovered={recovered}")


if __name__ == "__main__":
    main()
