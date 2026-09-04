# Tutorial 2: Defining your own Problem

Every benchmark handle (`sezgi.bbob(...)`, `sezgi.problems.cec2022(...)`,
...) is really just a `Problem` — a search space plus an objective
function. This tutorial writes two of your own: a classic multimodal
benchmark function (Rastrigin), and a small data-driven objective built
from a fixed dataset, the shape most real optimization problems actually
take.

## The `Problem` ABC

Subclassing `sezgi.Problem` requires exactly two methods:

- `space()` — returns a `sezgi.Space(...)` or a bare block
  (`sezgi.Float(...)`, `sezgi.Int(...)`, ...) declaring the search space.
  Called once, when the instance is converted to a native handle — not
  re-evaluated per generation, so it must not depend on state that
  changes during a run.
- `evaluate(x)` — `x -> float`. For a single-block space (every example on
  this page), `x` is that block's own converted Python value: a `Float`
  block hands you `list[float]`, an `Int` block `list[int]`, and so on
  (see [Problem and spaces](../api/problem.md) for the full conversion
  table and the multi-block/tuple case Tutorial 6 uses).

Two optional overrides: `optimum()` (default `None`) reports a known
global minimum, which is what populates `SolveResult.gap`; `batch_evaluate`
(default: loops `evaluate`) lets a vectorized objective process a whole
generation's population in one call.

## Rastrigin: a known-optimum benchmark, pure stdlib

Rastrigin (`f(x) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))`, `A=10`) is a
classic highly-multimodal function — many regularly-spaced local minima
around one global minimum at the origin. Unlike a compiled-in BBOB
function, its math is fully visible here, using nothing beyond `math`:

```python exec="true" source="above"
import math

import sezgi


class Rastrigin(sezgi.Problem):
    A = 10.0

    def __init__(self, n=5, lo=-5.12, hi=5.12):
        self.n, self.lo, self.hi = n, lo, hi

    def space(self):
        return sezgi.Float(self.lo, self.hi, self.n)

    def evaluate(self, x):
        return self.A * len(x) + sum(
            v * v - self.A * math.cos(2.0 * math.pi * v) for v in x)

    def optimum(self):
        return 0.0  # the global minimum is known exactly


result = sezgi.GreyWolfOptimizer(pop_size=30).run(Rastrigin(n=5), budget=3000, seed=42)
print(f"evals_used={result.evals_used} best_f={result.best_f:.6g} gap={result.gap:.6g}")
```

`optimum()` returning `0.0` (rather than the base class's default `None`)
is what makes `result.gap` a real number instead of `None` — overriding it
is the whole difference between "a problem with no known answer" and "a
problem you can measure exact progress against". `[-5.12, 5.12]` is
Rastrigin's conventional per-coordinate domain, chosen so the
highly-multimodal region around the origin is fully inside the search box.

## From data: fitting a linear model as a `Problem`

A problem built from a fixed dataset looks the same as any other `Problem`
— `evaluate(x)` just closes over the data instead of a closed-form
formula. This one treats the coefficients of a linear model as the search
point and the mean squared error on a fixed dataset as the objective — the
same shape a hyperparameter search or a model-fitting task takes in
practice.

```python exec="true" source="above"
import numpy as np

import sezgi


class LinearFit(sezgi.Problem):
    """Search over coefficient vectors w minimizing MSE(X @ w, y) on a
    FIXED dataset (X, y) -- the search space's dimension is the number of
    coefficients, not the dataset size."""

    def __init__(self, X, y, lo=-5.0, hi=5.0):
        self.X, self.y = X, y
        self.n = X.shape[1]
        self.lo, self.hi = lo, hi

    def space(self):
        return sezgi.Float(self.lo, self.hi, self.n)

    def evaluate(self, x):
        pred = self.X @ np.asarray(x)
        return float(np.mean((pred - self.y) ** 2))

    def batch_evaluate(self, xs):
        """Vectorized: scores the WHOLE population in one matrix
        multiply instead of looping evaluate() once per individual --
        called once per generation with xs = the current population."""
        W = np.asarray(xs)              # (pop_size, n)
        preds = self.X @ W.T            # (n_samples, pop_size)
        errs = preds - self.y[:, None]  # broadcast y over the population
        return list(np.mean(errs ** 2, axis=0))


rng = np.random.default_rng(11)
X = rng.standard_normal((30, 3))
true_w = np.array([2.0, -1.0, 0.5])
y = X @ true_w + rng.normal(scale=0.1, size=30)  # noisy linear target

problem = LinearFit(X, y)
result = sezgi.GeneticAlgorithm(pop_size=30).run(problem, budget=2000, seed=7)

print(f"true_w = {true_w.tolist()}")
print(f"best_x = {[round(v, 4) for v in result.best_x]}")
print(f"best_f (MSE) = {result.best_f:.6g}")
```

`batch_evaluate` is called once per generation with the whole population
(`pop_size` points), not once per point — for `LinearFit` above this turns
`pop_size` separate `X @ w` matrix-vector products into one `X @ W.T`
matrix-matrix product, the same vectorization discipline a real
data-driven objective needs to stay fast. `LinearFit` has no `optimum()`
override, so `result.gap` is `None` here — the recovered `best_x` is
compared to `true_w` directly instead, since the noisy dataset's own true
minimum MSE is not exactly `0.0` and was never computed in closed form.

## Next

- [Writing an Algorithm subclass](03-algorithm-subclass.md) — author the
  SEARCH side instead of the problem side.
- [Feature selection recipe](05-feature-selection.md) — a specialized,
  ready-made `Problem` subclass (`sezgi.recipes.FeatureSelection`) for
  exactly this "optimize against data" shape, with a `Binary` mask space
  instead of `LinearFit`'s own `Float` one.
