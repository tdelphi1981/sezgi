"""OOP twin of examples/python/jaya.py -- same math, same RNG draw order.

Port of the pure-Python JAYA script onto sezgi.AskTellAlgorithm. The pure script
(and the Rust module doc it cites) remains the provenance for the update
equations; this file re-derives NOTHING and must reproduce the pure
script's evals_used/best_f/gap output bit-for-bit at the same seed --
enforced by py-sezgi/tests/test_examples_oop_parity.py.
"""
import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30  # candidate count
SEED = 42


def jaya_dim_step(x_d, x_best_d, x_worst_d, r1, r2):
    """Eq. (1): X' = X + r1*(X_best - |X|) - r2*(X_worst - |X|)."""
    abs_x = abs(x_d)
    return x_d + r1 * (x_best_d - abs_x) - r2 * (x_worst_d - abs_x)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Jaya(sezgi.AskTellAlgorithm):
    name = "jaya"

    def setup(self, ctx):
        self.pop = [ctx.random_point() for _ in range(POP_SIZE)]
        self.fitness = ctx.evaluate(self.pop)

    def step(self, ctx):
        lo, hi = ctx.bounds
        # Current-population argmin/argmax, both computed before any draws.
        best = min(range(POP_SIZE), key=lambda i: (self.fitness[i], i))
        worst = max(range(POP_SIZE), key=lambda i: (self.fitness[i], i))
        x_best, x_worst = self.pop[best], self.pop[worst]

        # Pinned: r1[d], r2[d] drawn ONCE PER DIMENSION PER GENERATION,
        # shared across every agent (the paper's own worked example reuses
        # the same pair across all candidates) -- not fresh per (i, d).
        r1 = [ctx.rng.random() for _ in range(DIM)]
        r2 = [ctx.rng.random() for _ in range(DIM)]

        offspring = [
            [clamp(jaya_dim_step(self.pop[i][d], x_best[d], x_worst[d], r1[d], r2[d]), lo, hi)
             for d in range(DIM)]
            for i in range(POP_SIZE)
        ]
        new_fitness = ctx.evaluate(offspring)

        # replace/one-to-one-greedy: the paper's Fig. 1 flowchart accepts
        # X' only if strictly better, else keeps the previous solution.
        for i in range(POP_SIZE):
            if new_fitness[i] < self.fitness[i]:
                self.pop[i], self.fitness[i] = offspring[i], new_fitness[i]


def main():
    res = Jaya().solve(sezgi.bbob(1, DIM, 1), budget=BUDGET, seed=SEED)
    print(f"jaya (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
