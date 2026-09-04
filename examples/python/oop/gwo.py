"""OOP twin of examples/python/gwo.py -- same math, same RNG draw order.

Port of the pure-Python Grey Wolf Optimizer script onto sezgi.AskTellAlgorithm.
The pure script (and the Rust module doc it cites) remains the provenance
for the update equations; this file re-derives NOTHING and must reproduce
the pure script's evals_used/best_f/gap output bit-for-bit at the same
seed -- enforced by py-sezgi/tests/test_examples_oop_parity.py.
"""
import sezgi

BUDGET = 2000
SEED = 42


def gwo_dim_step(a, leaders_d, x_d, draws):
    total = 0.0
    for k in range(3):
        r1, r2 = draws[2 * k], draws[2 * k + 1]
        big_a = 2.0 * a * r1 - a
        c = 2.0 * r2
        total += leaders_d[k] - big_a * abs(c * leaders_d[k] - x_d)
    return total / 3.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Gwo(sezgi.AskTellAlgorithm):
    """Grey Wolf Optimizer (Mirjalili, Mirjalili & Lewis 2014, "Grey Wolf
    Optimizer", Advances in Engineering Software), modeling the
    three-leader (alpha/beta/delta) encircling update explicitly over
    `sezgi.AskTellAlgorithm`'s ask/tell loop. `dim`: problem
    dimensionality. `pop_size`: number of wolves."""

    name = "gwo"

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
        order = sorted(range(self.pop_size), key=lambda i: (self.fitness[i], i))
        leader_x = [self.pop[order[k]] for k in range(3)]
        offspring = []
        for i in range(self.pop_size):
            x = self.pop[i]
            new_x = []
            for d in range(self.dim):
                leaders_d = [leader_x[k][d] for k in range(3)]
                draws = [ctx.rng.random() for _ in range(6)]
                new_x.append(clamp(gwo_dim_step(a, leaders_d, x[d], draws), lo, hi))
            offspring.append(new_x)
        self.fitness = ctx.evaluate(offspring)
        self.pop = offspring


def main():
    algo = Gwo()
    res = algo.solve(sezgi.bbob(1, algo.dim, 1), budget=BUDGET, seed=SEED)
    print(f"gwo (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
