"""OOP example: authoring the PROBLEM side of the class-first surface --
a `sezgi.Problem` subclass for the Rastrigin function, PURE stdlib `math`
(no numpy, no `random` -- Rastrigin's own formula needs no randomness at
all), solved by a BUILT-IN algorithm wrapper class
(`sezgi.GreyWolfOptimizer`, one of the ~30 preset-backed classes in
`sezgi.builtins`) rather than a hand-authored `Algorithm`/
`PopulationAlgorithm`/`LocalSearch` subclass. Contrast `custom_de.py`/
`custom_pso_variant.py`/`local_search_2opt.py` in this directory, which
each author the ALGORITHM side against a built-in or self-authored
problem -- this file is the other half of the class-first surface:
"define a problem once, point any built-in class at it", demonstrating
`optimum()` and the `gap` it enables (Rastrigin's global minimum is
known exactly, unlike `local_search_2opt.py`'s own small TSP instance).
"""
import math

import sezgi


class Rastrigin(sezgi.Problem):
    """The Rastrigin function (A=10, the standard constant): `f(x) = A*n
    + sum(x_i^2 - A*cos(2*pi*x_i))`, a classic highly-multimodal
    benchmark (many regularly-spaced local minima around one global
    minimum) -- unlike a BBOB/CEC handle's own compiled-in objective,
    this problem's math is fully visible here, pure `math.cos`/`math.pi`.
    `space()` is `sezgi.Float(lo, hi, n)`, the conventional Rastrigin
    domain being `[-5.12, 5.12]` per coordinate. `optimum()` overrides
    the base's default `None`: the GLOBAL minimum is known exactly (`0.0`
    at the all-zero point), so a caller's `SolveResult.gap` is populated
    end to end -- see `sezgi.Problem.optimum`'s own docstring for the
    plumbing this override reaches."""

    A = 10.0

    def __init__(self, n=5, lo=-5.12, hi=5.12):
        self.n, self.lo, self.hi = n, lo, hi

    def space(self):
        return sezgi.Float(self.lo, self.hi, self.n)

    def evaluate(self, x):
        return self.A * len(x) + sum(
            v * v - self.A * math.cos(2.0 * math.pi * v) for v in x)

    def optimum(self):
        return 0.0


def main():
    res = sezgi.GreyWolfOptimizer(pop_size=30).run(Rastrigin(n=5), budget=3000, seed=42)
    print(f"custom_problem_rastrigin (oop engine): evals_used={res.evals_used} "
          f"best_f={res.best_f:.6g} gap={res.gap:.6g}")
    # ANCHORED (per this project's convention): measured once at this exact
    # config (Rastrigin(n=5), sezgi.GreyWolfOptimizer(pop_size=30),
    # budget=3000, seed=42), then pinned as literal values below.
    assert res.evals_used == 3000
    assert res.best_f == 1.4210854715202004e-14
    assert res.gap == 1.4210854715202004e-14


if __name__ == "__main__":
    main()
