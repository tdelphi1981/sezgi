"""OOP twin of examples/python/goa.py -- same math, same RNG draw order.

Port of the pure-Python Grasshopper Optimisation Algorithm script onto
sezgi.Algorithm. The pure script (and the Rust module doc it cites) remains
the provenance for the update equations; this file re-derives NOTHING and
must reproduce the pure script's evals_used/best_f/gap output bit-for-bit at
the same seed -- enforced by py-sezgi/tests/test_examples_oop_parity.py.
"""
import math

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30
SEED = 42
GOA_F = 0.5
GOA_L = 1.5
GOA_C_MAX = 1.0
GOA_C_MIN = 1e-5
EPS = 1e-12


def goa_s(r):
    """Social force s(r) = f*e^(-r/l) - e^(-r) (Eq. 2.3)."""
    return GOA_F * math.exp(-r / GOA_L) - math.exp(-r)


def goa_pair_term(c, half_range_d, x_i_d, x_j_d, d_ij):
    # sezgi simplification: maps the raw Euclidean distance into [2, 4)
    # before applying s(.) -- the verified reference-implementation
    # semantics, see this file's module doc.
    dist_term = 2.0 + (d_ij % 2.0)
    s = goa_s(dist_term)
    # sezgi simplification: + EPS guards a zero-distance pair.
    return c * half_range_d * s * (x_j_d - x_i_d) / (d_ij + EPS)


def euclidean_dist(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Goa(sezgi.Algorithm):
    name = "goa"

    def setup(self, ctx):
        lo, hi = ctx.bounds
        self.half_range = (hi - lo) / 2.0
        self.pop = [ctx.random_point() for _ in range(POP_SIZE)]
        self.fitness = ctx.evaluate(self.pop)

    def step(self, ctx):
        lo, hi = ctx.bounds
        progress = min(max(ctx.evals_used / ctx.budget, 0.0), 1.0)
        c = GOA_C_MAX - progress * (GOA_C_MAX - GOA_C_MIN)
        best = min(range(POP_SIZE), key=lambda i: (self.fitness[i], i))
        x_best = self.pop[best]

        dists = [[euclidean_dist(self.pop[i], self.pop[j]) for j in range(POP_SIZE)]
                  for i in range(POP_SIZE)]

        offspring = []
        for i in range(POP_SIZE):
            x_i = self.pop[i]
            new_x = []
            for d in range(DIM):
                total = 0.0
                for j in range(POP_SIZE):
                    if j == i:
                        continue
                    total += goa_pair_term(c, self.half_range, x_i[d], self.pop[j][d], dists[i][j])
                new_x.append(clamp(c * total + x_best[d], lo, hi))
            offspring.append(new_x)

        self.fitness = ctx.evaluate(offspring)
        self.pop = offspring


def main():
    res = Goa().solve(sezgi.bbob(1, DIM, 1), budget=BUDGET, seed=SEED)
    print(f"goa (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
