"""OOP twin of examples/python/mfo.py -- same math, same RNG draw order.

Port of the pure-Python Moth-Flame Optimization script onto sezgi.Algorithm.
The pure script (and the Rust module doc it cites) remains the provenance
for the update equations and the flame-memory mechanism; this file
re-derives NOTHING and must reproduce the pure script's evals_used/best_f/
gap output bit-for-bit at the same seed -- enforced by
py-sezgi/tests/test_examples_oop_parity.py.
"""
import math

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30  # number of search agents (also the flame archive size)
SEED = 42


def mfo_dim_step(x_d, own_flame_d, target_flame_d, t):
    b = 1.0
    distance_to_flame = abs(own_flame_d - x_d)
    return distance_to_flame * math.exp(b * t) * math.cos(t * 2.0 * math.pi) + target_flame_d


def mfo_flame_count(n, progress):
    """Eq. (3.14): round(N - progress*(N-1)), clamped to [1, n]."""
    raw = n - progress * (n - 1)
    return max(1, min(n, round(raw)))


def merge_and_truncate(pop_fitness, pop_positions, flame_fitness, flame_positions):
    """Concatenate (pop first, flames second), stable-sort ascending by
    fitness, truncate to len(pop_fitness) -- MFO.m's own elitist
    double_population=[previous_population; best_flames] step."""
    combined = list(zip(pop_fitness, pop_positions)) + list(zip(flame_fitness, flame_positions))
    combined.sort(key=lambda fp: fp[0])
    combined = combined[:len(pop_fitness)]
    return [f for f, _ in combined], [p for _, p in combined]


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Mfo(sezgi.Algorithm):
    name = "mfo"

    def setup(self, ctx):
        self.pop = [ctx.random_point() for _ in range(POP_SIZE)]
        self.fitness = ctx.evaluate(self.pop)

        # Bootstrap: generation 0 seeds the flame archive from the sorted
        # initial population (a merge against an EMPTY old-flame set).
        self.flame_fitness, self.flames = merge_and_truncate(self.fitness, self.pop, [], [])

    def step(self, ctx):
        lo, hi = ctx.bounds
        progress = min(max(ctx.evals_used / ctx.budget, 0.0), 1.0)
        a = -1.0 - progress
        flame_count = mfo_flame_count(POP_SIZE, progress)

        offspring = []
        for i in range(POP_SIZE):
            x = self.pop[i]
            own_flame = self.flames[i]
            target_flame = self.flames[min(i, flame_count - 1)]
            new_x = []
            for d in range(DIM):
                t = (a - 1.0) * ctx.rng.random() + 1.0
                new_x.append(clamp(mfo_dim_step(x[d], own_flame[d], target_flame[d], t), lo, hi))
            offspring.append(new_x)

        # replace/generational: the flames, not the moth population,
        # carry the elitism, so the moths are overwritten unconditionally.
        new_fitness = ctx.evaluate(offspring)
        self.pop, self.fitness = offspring, new_fitness

        # adapter/mfo-flame-update: merge this generation's freshly moved
        # moths against the PREVIOUS flames, truncate back to POP_SIZE.
        self.flame_fitness, self.flames = merge_and_truncate(
            self.fitness, self.pop, self.flame_fitness, self.flames)


def main():
    res = Mfo().solve(sezgi.bbob(1, DIM, 1), budget=BUDGET, seed=SEED)
    print(f"mfo (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
