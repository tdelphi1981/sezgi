# Quickstart (Python, class-first)

Every built-in algorithm is a class; every problem is a native handle
(`sezgi.bbob(...)`, `sezgi.problems.onemax(...)`, ...) or a
`sezgi.Problem` subclass you author yourself. `.run()` accepts either and
always returns the same `SolveResult` shape.

## The 10-line path

This block *actually runs* at build time (via `markdown-exec`) — the
output below it is real, not transcribed, so it cannot silently drift
from the code above it.

```python exec="true" source="above"
import sezgi

problem = sezgi.bbob(1, 5, 1)  # BBOB f1 (Sphere), dim=5, instance=1
result = sezgi.GeneticAlgorithm(pop_size=20).run(problem, budget=1000, seed=1)

print(f"evals_used={result.evals_used}")
print(f"best_f={result.best_f:.6g}")
print(f"f_opt={result.f_opt:.6g}")
print(f"gap={result.gap:.6g}")
```

`sezgi.bbob(fid, dim, instance)` is a `Problem` handle for a COCO/BBOB
noiseless function — `fid=1` is the Sphere function. `GeneticAlgorithm`
auto-dispatches on the problem's space kind (here, a continuous `Float`
space, so it delegates to `presets.ga_real` internally — see
[Built-in algorithm classes](api/builtins.md)). `budget` is the total
number of objective-function evaluations allowed; `seed` makes the run
byte-reproducible — the same `(problem, budget, seed)` always produces
the same `result` in this build and across the R frontend for a shared
algorithm.

## Reading the result

`SolveResult` (returned by every wrapper class's `.run()`) carries:

- `algo` — the preset name that actually ran (useful when a wrapper
  class auto-dispatches, like `GeneticAlgorithm` above).
- `seed`, `budget`, `evals_used` — the run's own bookkeeping;
  `evals_used` is `budget` unless the algorithm terminates early.
- `best_x`, `best_f` — the best point *evaluated* and its objective
  value. `best_f` is not guaranteed to lie within the problem's declared
  bounds for every algorithm (see `sezgi.solve`'s own docstring on the
  [Solve / compat internals](api/compat.md) page).
- `f_opt`, `gap` — the problem's known optimum (`None` if unknown) and
  `best_f - f_opt`.

## Your own problem

Subclass `sezgi.Problem` and implement `space()` and `evaluate(x)`:

```python exec="true" source="above"
import sezgi

class Sphere(sezgi.Problem):
    def __init__(self, n=3, lo=-5.0, hi=5.0):
        self.n, self.lo, self.hi = n, lo, hi

    def space(self):
        return sezgi.Float(self.lo, self.hi, self.n)

    def evaluate(self, x):
        return sum(v * v for v in x)

result = sezgi.GeneticAlgorithm(pop_size=20).run(Sphere(n=3), budget=500, seed=1)
print(f"evals_used={result.evals_used} best_f={result.best_f:.6g}")
```

See [Problem and spaces](api/problem.md) for the full space-builder
table (`Float`/`Int`/`Categorical`/`Binary`/`Permutation`, and mixed
spaces via `Space(*blocks)`).

## Next steps

- [API reference](api/spaces.md) for the full auto-generated Python API.
- [R surface](r.md) if you want the same guarantees from R.
