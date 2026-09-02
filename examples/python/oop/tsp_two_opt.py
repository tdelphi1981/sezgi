"""OOP example: ABC-subclass local search on a vendored TSP instance --
a seeded random start (AlgoContext.random_permutation()) plus a
first-improvement 2-opt sweep (AlgoContext.two_opt()), run until the budget
is exhausted.

M3-8 Task 7: the FIRST worked example over a permutation-typed problem, not
a port of an existing pure script -- unlike every other file in this
directory, it deliberately sits OUTSIDE
py-sezgi/tests/test_examples_oop_parity.py's 17-pair "OOP twin reproduces
its pure script bit-for-bit" gate (that gate's PAIRS list names only
Float-typed BBOB algorithms with a paired pure/OOP script under
examples/python/<algo>.py; this file has no such pure-script twin to
reproduce). Its own determinism/anchored-output gate is
py-sezgi/tests/test_tsp_two_opt_example.py instead, following the same
"run the script, assert the printed metrics line" convention as the twin
gate uses (see FIELDS there).

Task 8 (M3-8) will add examples/r/oop/tsp_two_opt.R, a 1-based R twin over
the SAME vendored instance/seed/budget. The printed numbers below
(evals_used, best_f, gap, tour_length) are the language-neutral values that
twin is meant to reproduce -- tour LENGTH is invariant to whether the tour
itself is printed 0-based (here) or 1-based (there), so anchoring on the
numbers rather than the raw tour lets both scripts match without needing
the same indexing convention.
"""
import sezgi

INSTANCE = "berlin52"
BUDGET = 2000
SEED = 42


class TspTwoOpt(sezgi.AskTellAlgorithm):
    """First-improvement 2-opt local search from one random start.

    setup(): draws one random tour (`ctx.random_permutation()`) and
    evaluates it -- the run's starting point.

    step(): evaluates the NEXT candidate in a fixed, deterministic sweep
    over every 2-opt segment reversal `(i, j)` with `0 <= i < j < n`
    (`ctx.two_opt`'s own inclusive `[i, j]` semantics -- see its docstring),
    enumerated in a fixed order so the sweep itself draws no randomness;
    accepts the candidate (replacing the current tour) only if it strictly
    improves the current tour's length (first-improvement, not
    best-improvement). The sweep wraps around and restarts once every pair
    has been tried, so a step always has a next candidate to evaluate
    regardless of the budget -- exactly ONE evaluation is charged per
    step() call, matching every other OOP example's "consume at least one
    eval per step" contract, and making `evals_used` land exactly on
    `budget` (one `setup()` eval plus `budget - 1` step evals).
    """
    name = "tsp_two_opt"

    def setup(self, ctx):
        n = ctx.n
        self.pairs = [(i, j) for i in range(n - 1) for j in range(i + 1, n)]
        self.k = 0
        self.tour = ctx.random_permutation()
        (self.best_f,) = ctx.evaluate([self.tour])

    def step(self, ctx):
        i, j = self.pairs[self.k % len(self.pairs)]
        self.k += 1
        candidate = ctx.two_opt(self.tour, i, j)
        (f,) = ctx.evaluate([candidate])
        if f < self.best_f:
            self.tour, self.best_f = candidate, f


def main():
    res = TspTwoOpt().solve(sezgi.problems.tsp(INSTANCE), budget=BUDGET, seed=SEED)
    print(f"tsp_two_opt (oop): evals_used={res.evals_used} "
          f"best_f={res.best_f:.6g} gap={res.gap:.6g} "
          f"tour_length={res.best_f:.6g}")


if __name__ == "__main__":
    main()
