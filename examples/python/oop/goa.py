"""OOP twin of examples/python/goa.py -- same math, same RNG draw order.

Port of the pure-Python Grasshopper Optimisation Algorithm script onto
sezgi.AskTellAlgorithm. The pure script (and the Rust module doc it cites) remains
the provenance for the update equations; this file re-derives NOTHING and
must reproduce the pure script's evals_used/best_f/gap output bit-for-bit at
the same seed -- enforced by py-sezgi/tests/test_examples_oop_parity.py.
"""
import math

import sezgi

BUDGET = 2000
SEED = 42


def goa_s(r, f, l):
    """Social force s(r) = f*e^(-r/l) - e^(-r) (Eq. 2.3)."""
    return f * math.exp(-r / l) - math.exp(-r)


def goa_pair_term(c, half_range_d, x_i_d, x_j_d, d_ij, f, l, eps):
    # sezgi simplification: maps the raw Euclidean distance into [2, 4)
    # before applying s(.) -- the verified reference-implementation
    # semantics, see this file's module doc.
    dist_term = 2.0 + (d_ij % 2.0)
    s = goa_s(dist_term, f, l)
    # sezgi simplification: + eps guards a zero-distance pair.
    return c * half_range_d * s * (x_j_d - x_i_d) / (d_ij + eps)


def euclidean_dist(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Goa(sezgi.AskTellAlgorithm):
    """Grasshopper Optimisation Algorithm (Saremi, Mirjalili & Lewis
    2017, Advances in Engineering Software 105, 30-47), modeling the
    pairwise social-force update explicitly over
    `sezgi.AskTellAlgorithm`'s ask/tell loop -- fully deterministic given
    the population and the shrinking coefficient `c` (RNG only seeds the
    initial population). `dim`: problem dimensionality. `pop_size`:
    number of grasshoppers. `goa_f`/`goa_l`: the social-force intensity/
    attractive-length-scale constants of `s(r) = f*e^(-r/l) - e^(-r)`
    (Eq. 2.3). `goa_c_max`/`goa_c_min`: the shrinking coefficient's
    linear decay bounds over the run. `eps`: guards a zero-distance
    pair's division."""

    name = "goa"

    def __init__(self, dim=5, pop_size=30, goa_f=0.5, goa_l=1.5,
                 goa_c_max=1.0, goa_c_min=1e-5, eps=1e-12):
        self.dim = dim
        self.pop_size = pop_size
        self.goa_f = goa_f
        self.goa_l = goa_l
        self.goa_c_max = goa_c_max
        self.goa_c_min = goa_c_min
        self.eps = eps

    def setup(self, ctx):
        lo, hi = ctx.bounds
        self.half_range = (hi - lo) / 2.0
        self.pop = [ctx.random_point() for _ in range(self.pop_size)]
        self.fitness = ctx.evaluate(self.pop)

    def step(self, ctx):
        lo, hi = ctx.bounds
        progress = min(max(ctx.evals_used / ctx.budget, 0.0), 1.0)
        c = self.goa_c_max - progress * (self.goa_c_max - self.goa_c_min)
        best = min(range(self.pop_size), key=lambda i: (self.fitness[i], i))
        x_best = self.pop[best]

        dists = [[euclidean_dist(self.pop[i], self.pop[j]) for j in range(self.pop_size)]
                  for i in range(self.pop_size)]

        offspring = []
        for i in range(self.pop_size):
            x_i = self.pop[i]
            new_x = []
            for d in range(self.dim):
                total = 0.0
                for j in range(self.pop_size):
                    if j == i:
                        continue
                    total += goa_pair_term(c, self.half_range, x_i[d], self.pop[j][d],
                                            dists[i][j], self.goa_f, self.goa_l, self.eps)
                new_x.append(clamp(c * total + x_best[d], lo, hi))
            offspring.append(new_x)

        self.fitness = ctx.evaluate(offspring)
        self.pop = offspring


def main():
    algo = Goa()
    res = algo.solve(sezgi.bbob(1, algo.dim, 1), budget=BUDGET, seed=SEED)
    print(f"goa (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
