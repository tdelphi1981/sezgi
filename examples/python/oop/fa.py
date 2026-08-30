"""OOP twin of examples/python/fa.py -- same math, same RNG draw order.

Port of the pure-Python Firefly Algorithm script onto sezgi.Algorithm. The
pure script (and the Rust module doc it cites) remains the provenance for
the update equations and the hybrid live/frozen quirk; this file
re-derives NOTHING and must reproduce the pure script's evals_used/best_f/
gap output bit-for-bit at the same seed -- enforced by
py-sezgi/tests/test_examples_oop_parity.py.
"""
import math

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 25  # number of fireflies
SEED = 42
ALPHA0 = 0.5
BETA0 = 1.0
BETAMIN = 0.2
GAMMA = 1.0


def fa_beta(r2):
    """Floored attractiveness: beta -> betamin (not 0) as r2/gamma grow."""
    return (BETA0 - BETAMIN) * math.exp(-GAMMA * r2) + BETAMIN


def fa_alpha(progress):
    """Closed-form substitute for alpha_new's per-generation geometric decay."""
    return ALPHA0 * (1e-4 / 0.9) ** progress


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Fa(sezgi.Algorithm):
    name = "fa"

    def setup(self, ctx):
        lo, hi = ctx.bounds
        self.pop = [ctx.random_point() for _ in range(POP_SIZE)]
        self.fitness = ctx.evaluate(self.pop)
        self.scale = [hi - lo for _ in range(DIM)]

    def step(self, ctx):
        lo, hi = ctx.bounds
        progress = min(max(ctx.evals_used / ctx.budget, 0.0), 1.0)
        alpha = fa_alpha(progress)

        # Rank order: stable sort by fitness ascending, ties -> original
        # index -- REQUIRED for the verified in-place loop semantics, not
        # merely cosmetic (mirrors ffa_mincon's own pre-sort).
        order = sorted(range(POP_SIZE), key=lambda i: (self.fitness[i], i))
        lighto = [self.fitness[i] for i in order]
        nso = [list(self.pop[i]) for i in order]  # FROZEN rank-ordered snapshot
        ns = [list(row) for row in nso]           # LIVE working copy, mutated in place

        for i in range(POP_SIZE):
            for j in range(POP_SIZE):
                # r2 from the LIVE working copy on BOTH sides -- verified
                # quirk: not the frozen nso snapshot.
                r2 = sum((ns[i][d] - ns[j][d]) ** 2 for d in range(DIM))
                if lighto[i] > lighto[j]:
                    beta = fa_beta(r2)
                    for d in range(DIM):
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
    res = Fa().solve(sezgi.bbob(1, DIM, 1), budget=BUDGET, seed=SEED)
    print(f"fa (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
