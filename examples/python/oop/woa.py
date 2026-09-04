"""OOP twin of examples/python/woa.py -- same math, same RNG draw order.

Port of the pure-Python Whale Optimization Algorithm script onto
sezgi.AskTellAlgorithm. The pure script (and the Rust module doc it cites) remains
the provenance for the update equations; this file re-derives NOTHING and
must reproduce the pure script's evals_used/best_f/gap output bit-for-bit at
the same seed -- enforced by py-sezgi/tests/test_examples_oop_parity.py.
"""
import math

import sezgi

BUDGET = 2000
SEED = 42


def woa_encircle_step(target_d, x_d, big_a, big_c):
    return target_d - big_a * abs(big_c * target_d - x_d)


def woa_spiral_step(x_best_d, x_d, l, b=1.0):
    d = abs(x_best_d - x_d)
    return d * math.exp(b * l) * math.cos(2.0 * math.pi * l) + x_best_d


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Woa(sezgi.AskTellAlgorithm):
    """Whale Optimization Algorithm (Mirjalili & Lewis 2016, "The Whale
    Optimization Algorithm", Advances in Engineering Software), modeling
    the probability-gated shrinking-encirclement/spiral-bubble-net update
    explicitly over `sezgi.AskTellAlgorithm`'s ask/tell loop. `dim`:
    problem dimensionality. `pop_size`: number of whales."""

    name = "woa"

    def __init__(self, dim=5, pop_size=30):
        self.dim = dim
        self.pop_size = pop_size

    def setup(self, ctx):
        self.pop = [ctx.random_point() for _ in range(self.pop_size)]
        self.fitness = ctx.evaluate(self.pop)

    def step(self, ctx):
        lo, hi = ctx.bounds
        progress = min(max(ctx.evals_used / ctx.budget, 0.0), 1.0)
        a = 2.0 - 2.0 * progress
        a2 = -1.0 - progress

        best = min(range(self.pop_size), key=lambda i: (self.fitness[i], i))
        x_best = self.pop[best]

        offspring = []
        for i in range(self.pop_size):
            x = self.pop[i]
            # p, and then r1/r2 (search branch) or l_raw (spiral branch),
            # are all drawn ONCE per whale, before the dimension loop --
            # matching the reference MATLAB (WOA.m) exactly. Only the
            # random-leader index is drawn per dimension.
            p = ctx.rng.random()
            new_x = []
            if p < 0.5:
                r1, r2 = ctx.rng.random(), ctx.rng.random()
                big_a = 2.0 * a * r1 - a
                big_c = 2.0 * r2
                for d in range(self.dim):
                    if abs(big_a) < 1.0:
                        target_d = x_best[d]
                    else:
                        j = ctx.rng.randrange(self.pop_size)  # may equal i, matches reference MATLAB
                        target_d = self.pop[j][d]
                    val = woa_encircle_step(target_d, x[d], big_a, big_c)
                    new_x.append(clamp(val, lo, hi))
            else:
                l_raw = ctx.rng.random()
                l = (a2 - 1.0) * l_raw + 1.0
                for d in range(self.dim):
                    val = woa_spiral_step(x_best[d], x[d], l)
                    new_x.append(clamp(val, lo, hi))
            offspring.append(new_x)

        self.fitness = ctx.evaluate(offspring)
        self.pop = offspring


def main():
    algo = Woa()
    res = algo.solve(sezgi.bbob(1, algo.dim, 1), budget=BUDGET, seed=SEED)
    print(f"woa (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
