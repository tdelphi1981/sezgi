"""OOP twin of examples/python/fa.py -- same math, same RNG draw order.

Port of the pure-Python Firefly Algorithm script onto sezgi.AskTellAlgorithm. The
pure script (and the Rust module doc it cites) remains the provenance for
the update equations and the hybrid live/frozen quirk; this file
re-derives NOTHING and must reproduce the pure script's evals_used/best_f/
gap output bit-for-bit at the same seed -- enforced by
py-sezgi/tests/test_examples_oop_parity.py.
"""
import math

import sezgi

BUDGET = 2000
SEED = 42


def fa_beta(r2, beta0, betamin, gamma):
    """Floored attractiveness: beta -> betamin (not 0) as r2/gamma grow."""
    return (beta0 - betamin) * math.exp(-gamma * r2) + betamin


def fa_alpha(progress, alpha0):
    """Closed-form substitute for alpha_new's per-generation geometric decay."""
    return alpha0 * (1e-4 / 0.9) ** progress


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Fa(sezgi.AskTellAlgorithm):
    """Firefly Algorithm (Yang, "Nature-Inspired Metaheuristic
    Algorithms", 2nd ed., Luniver Press, 2010), modeling the pairwise
    attractiveness-weighted double-loop update over
    `sezgi.AskTellAlgorithm`'s ask/tell loop. `dim`: problem
    dimensionality. `pop_size`: number of fireflies. `alpha0`: initial
    step-scale coefficient, geometrically decayed via a closed-form
    substitute for `alpha_new`'s per-generation decay. `beta0`: maximum
    attractiveness (at zero distance). `betamin`: floor attractiveness
    the pairwise weight decays toward (never 0) as distance grows.
    `gamma`: light-absorption coefficient controlling that decay rate."""

    name = "fa"

    def __init__(self, dim=5, pop_size=25, alpha0=0.5, beta0=1.0,
                 betamin=0.2, gamma=1.0):
        self.dim = dim
        self.pop_size = pop_size
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.betamin = betamin
        self.gamma = gamma

    def setup(self, ctx):
        lo, hi = ctx.bounds
        self.pop = [ctx.random_point() for _ in range(self.pop_size)]
        self.fitness = ctx.evaluate(self.pop)
        self.scale = [hi - lo for _ in range(self.dim)]

    def step(self, ctx):
        lo, hi = ctx.bounds
        progress = min(max(ctx.evals_used / ctx.budget, 0.0), 1.0)
        alpha = fa_alpha(progress, self.alpha0)

        # Rank order: stable sort by fitness ascending, ties -> original
        # index -- REQUIRED for the verified in-place loop semantics, not
        # merely cosmetic (mirrors ffa_mincon's own pre-sort).
        order = sorted(range(self.pop_size), key=lambda i: (self.fitness[i], i))
        lighto = [self.fitness[i] for i in order]
        nso = [list(self.pop[i]) for i in order]  # FROZEN rank-ordered snapshot
        ns = [list(row) for row in nso]           # LIVE working copy, mutated in place

        for i in range(self.pop_size):
            for j in range(self.pop_size):
                # r2 from the LIVE working copy on BOTH sides -- verified
                # quirk: not the frozen nso snapshot.
                r2 = sum((ns[i][d] - ns[j][d]) ** 2 for d in range(self.dim))
                if lighto[i] > lighto[j]:
                    beta = fa_beta(r2, self.beta0, self.betamin, self.gamma)
                    for d in range(self.dim):
                        u = ctx.rng.random()
                        step = alpha * (u - 0.5) * self.scale[d]
                        # self LIVE (accumulates across repeated j hits),
                        # target FROZEN.
                        ns[i][d] = ns[i][d] * (1.0 - beta) + nso[j][d] * beta + step

        # boundary/clamp applied AFTER the full double loop (same timing
        # as the preset's separate boundary stage, not mid-loop).
        offspring = [[clamp(v, lo, hi) for v in row] for row in ns]

        # replace/generational: fa_ndim.m overwrites the whole population
        # unconditionally every generation.
        self.fitness = ctx.evaluate(offspring)
        self.pop = offspring


def main():
    algo = Fa()
    res = algo.solve(sezgi.bbob(1, algo.dim, 1), budget=BUDGET, seed=SEED)
    print(f"fa (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
