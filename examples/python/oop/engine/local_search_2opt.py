"""OOP example: a `sezgi.LocalSearch` variant over a hand-authored,
permutation-typed `sezgi.Problem` subclass -- a small (10-city) TSP-like
instance defined IN THIS FILE (stdlib-only: plain Euclidean distance via
`math.hypot`, no vendored instance file), unlike `../tsp_two_opt.py`
(a `sezgi.AskTellAlgorithm` over the vendored `berlin52` instance,
`sezgi.problems.tsp("berlin52")`, with a fixed deterministic sweep over
EVERY `(i, j)` 2-opt pair in order) or `../custom_local_search.py` (a
`LocalSearch` variant, but over a Float-typed BBOB problem, perturbing
real coordinates). This file combines both of those: `LocalSearch`
(engine-hosted, `neighbor()`-only override, default greedy `accept()`,
`../custom_local_search.py`'s own family base) applied to a
PERMUTATION-typed, self-authored `Problem` (`../tsp_two_opt.py`'s own
problem TYPE, but not its problem instance or its algorithm base) -- and,
unlike `../tsp_two_opt.py`'s fixed sweep, proposes a RANDOM 2-opt segment
reversal each step, drawn entirely from `ctx.rng` (no fixed enumeration
order).

The engine's own default `init/uniform` initializer already knows how to
seed a Permutation block (a Fisher-Yates shuffle over `ctx.rng`,
`crates/components/src/init.rs`) -- this class does not need to (and does
not) override `initialize()`.
"""
import math

import sezgi

#: A small, fixed 10-city instance (arbitrary coordinates, hand-picked to
#: avoid ties/degenerate collinear layouts) -- deliberately tiny so a short,
#: single-restart local search still makes visible progress within a small
#: budget.
CITIES = [
    (0.0, 0.0), (2.0, 5.0), (5.0, 2.0), (6.0, 6.0), (1.0, 8.0),
    (8.0, 3.0), (3.0, 9.0), (9.0, 9.0), (4.0, 1.0), (7.0, 7.0),
]


class SmallTsp(sezgi.Problem):
    """A permutation-space `Problem`: `space()` is `sezgi.Permutation(n)`
    (one block, `n` cities); `evaluate(tour)` sums the closed-tour
    Euclidean length (`math.hypot`, pure stdlib) over consecutive cities
    in `tour` order, wrapping back to the start. No known optimum (this
    tiny instance's exact optimal tour was not independently computed),
    so `optimum()` stays at the base's default `None` -- unlike
    `custom_problem_rastrigin.py`, this example has no `gap` to report."""

    def __init__(self, cities=CITIES):
        self.cities = cities

    def space(self):
        return sezgi.Permutation(len(self.cities))

    def evaluate(self, tour):
        n = len(tour)
        total = 0.0
        for k in range(n):
            x1, y1 = self.cities[tour[k]]
            x2, y2 = self.cities[tour[(k + 1) % n]]
            total += math.hypot(x2 - x1, y2 - y1)
        return total


class RandomTwoOpt(sezgi.LocalSearch):
    """neighbor(): proposes ONE random 2-opt move -- reverses a randomly
    chosen contiguous segment `x[i:j+1]` of the current tour (`0 <= i <
    j < n`, both drawn from `ctx.rng.next_below`, no rejection loop:
    `i = ctx.rng.next_below(n - 1)` (range `[0, n-2]`), then
    `j = i + 1 + ctx.rng.next_below(n - 1 - i)` (range `[i+1, n-1]`) --
    keeps the base's default greedy `accept()` (see `sezgi.LocalSearch`'s
    own ACCEPT() DESIGN docstring for exactly what `accept()` does and
    does not control here). Unlike `../tsp_two_opt.py`'s fixed,
    exhaustive sweep over every `(i, j)` pair in a deterministic order,
    every step here proposes a fresh RANDOM segment -- a stochastic
    2-opt local search, not an exhaustive one."""

    def neighbor(self, x, ctx):
        n = len(x)
        i = ctx.rng.next_below(n - 1)
        j = i + 1 + ctx.rng.next_below(n - 1 - i)
        return x[:i] + list(reversed(x[i:j + 1])) + x[j + 1:]


def main():
    res = RandomTwoOpt().run(SmallTsp(), budget=1500, seed=42)
    print(f"local_search_2opt (oop engine): evals_used={res.evals_used} "
          f"best_f={res.best_f:.6g} tour_length={res.best_f:.6g}")
    # ANCHORED (per this project's convention): measured once at this exact
    # config (SmallTsp() over CITIES above, budget=1500, seed=42), then
    # pinned as literal values below.
    assert res.evals_used == 1500
    assert res.best_f == 33.33129925554508
    assert res.best_x == [1, 4, 6, 7, 9, 3, 5, 2, 8, 0]


if __name__ == "__main__":
    main()
