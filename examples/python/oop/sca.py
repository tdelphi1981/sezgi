"""OOP twin of examples/python/sca.py -- same math, same RNG draw order.

Port of the pure-Python Sine Cosine Algorithm script onto sezgi.AskTellAlgorithm.
The pure script (and the Rust module doc it cites) remains the provenance
for the update equations; this file re-derives NOTHING and must reproduce
the pure script's evals_used/best_f/gap output bit-for-bit at the same
seed -- enforced by py-sezgi/tests/test_examples_oop_parity.py.
"""
import math

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30  # number of search agents
SEED = 42


def sca_dim_step(x_d, x_best_d, r1, r2, r3, r4):
    """Eq. 3.1 (sin branch, r4 < 0.5) / Eq. 3.2 (cos branch)."""
    delta = abs(r3 * x_best_d - x_d)
    if r4 < 0.5:
        return x_d + r1 * math.sin(r2) * delta
    return x_d + r1 * math.cos(r2) * delta


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Sca(sezgi.AskTellAlgorithm):
    name = "sca"

    def setup(self, ctx):
        self.pop = [ctx.random_point() for _ in range(POP_SIZE)]
        self.fitness = ctx.evaluate(self.pop)

    def step(self, ctx):
        lo, hi = ctx.bounds
        progress = min(max(ctx.evals_used / ctx.budget, 0.0), 1.0)
        a = 2.0
        r1 = a * (1.0 - progress)  # = a - t*(a/T) = 2 - 2*progress

        # Destination_position: current population's fitness argmin,
        # ties -> lower index.
        best = min(range(POP_SIZE), key=lambda i: (self.fitness[i], i))
        x_best = self.pop[best]

        offspring = []
        for i in range(POP_SIZE):
            x = self.pop[i]
            new_x = []
            for d in range(DIM):
                # Pinned draw order: r2, then r3, then r4, fresh for every
                # (i, d) -- always all three, regardless of branch.
                r2 = 2.0 * math.pi * ctx.rng.random()
                r3 = 2.0 * ctx.rng.random()
                r4 = ctx.rng.random()
                new_x.append(clamp(sca_dim_step(x[d], x_best[d], r1, r2, r3, r4), lo, hi))
            offspring.append(new_x)

        # replace/generational: SCA.m overwrites every agent's position
        # every iteration unconditionally (no greedy comparison).
        self.fitness = ctx.evaluate(offspring)
        self.pop = offspring


def main():
    res = Sca().solve(sezgi.bbob(1, DIM, 1), budget=BUDGET, seed=SEED)
    print(f"sca (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
