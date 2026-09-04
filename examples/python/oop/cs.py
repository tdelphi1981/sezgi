"""OOP twin of examples/python/cs.py -- same math, same RNG draw order.

Port of the pure-Python Cuckoo Search script onto sezgi.AskTellAlgorithm. The pure
script (and the Rust module doc it cites) remains the provenance for the
update equations; this file re-derives NOTHING and must reproduce the pure
script's evals_used/best_f/gap output bit-for-bit at the same seed --
enforced by py-sezgi/tests/test_examples_oop_parity.py.
"""
import math

import sezgi

BUDGET = 2000
SEED = 42


def gauss_polar(rng):
    """Polar Box-Muller (Marsaglia)."""
    while True:
        u = 2.0 * rng.random() - 1.0
        v = 2.0 * rng.random() - 1.0
        s = u * u + v * v
        if 0.0 < s < 1.0:
            return u * math.sqrt(-2.0 * math.log(s) / s)


def levy_mantegna(rng, alpha=1.5):
    num = math.gamma(1.0 + alpha) * math.sin(math.pi * alpha / 2.0)
    den = math.gamma((1.0 + alpha) / 2.0) * alpha * 2.0 ** ((alpha - 1.0) / 2.0)
    sigma_u = (num / den) ** (1.0 / alpha)
    u = sigma_u * gauss_polar(rng)
    v = abs(gauss_polar(rng))
    return u / (v ** (1.0 / alpha))


def cs_dim_step(x_i_d, x_best_d, levy, alpha_step=0.01):
    return x_i_d + alpha_step * levy * (x_i_d - x_best_d)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def abandon_order(fitness, pa):
    """Worst-first indices of the worst floor(pa*n) nests; ties -> higher
    index abandoned first."""
    n = len(fitness)
    k = int(pa * n)
    order = sorted(range(n), key=lambda i: (-fitness[i], -i))
    return order[:k]


class Cs(sezgi.AskTellAlgorithm):
    """Cuckoo Search via Levy flights (Yang & Deb 2009, 2009 World
    Congress on Nature & Biologically Inspired Computing (NaBIC), pp.
    210-214, IEEE), modeling the Levy-flight step plus a worst-fraction
    abandonment cycle explicitly over `sezgi.AskTellAlgorithm`'s ask/tell
    loop. `dim`: problem dimensionality. `pop_size`: nest count.
    `alpha_step`: step-size scale multiplying the Levy draw. `levy_alpha`:
    the Mantegna (1994) Levy-stability exponent. `pa`: fraction of the
    worst nests abandoned (re-randomized) each generation."""

    name = "cs"

    def __init__(self, dim=5, pop_size=25, alpha_step=0.01, levy_alpha=1.5,
                 pa=0.25):
        self.dim = dim
        self.pop_size = pop_size
        self.alpha_step = alpha_step
        self.levy_alpha = levy_alpha
        self.pa = pa

    def setup(self, ctx):
        self.pop = [ctx.random_point() for _ in range(self.pop_size)]
        self.fitness = ctx.evaluate(self.pop)

    def step(self, ctx):
        lo, hi = ctx.bounds
        best = min(range(self.pop_size), key=lambda i: (self.fitness[i], i))
        x_best = self.pop[best]

        offspring = [
            [clamp(cs_dim_step(self.pop[i][d], x_best[d],
                                levy_mantegna(ctx.rng, self.levy_alpha),
                                self.alpha_step), lo, hi)
             for d in range(self.dim)]
            for i in range(self.pop_size)
        ]
        new_fitness = ctx.evaluate(offspring)

        # replace/one-to-one-greedy
        for i in range(self.pop_size):
            if new_fitness[i] < self.fitness[i]:
                self.pop[i], self.fitness[i] = offspring[i], new_fitness[i]

        # adapter/abandon-worst-fraction: only if the WHOLE abandon batch
        # fits (mirrors the Rust adapter silently skipping on budget
        # exhaustion, via `if let Ok(new_fitness) = ctx.eval.evaluate(..)`).
        order = abandon_order(self.fitness, self.pa)
        if order and len(order) <= ctx.remaining:
            new_nests = [ctx.random_point() for _ in order]
            new_fit2 = ctx.evaluate(new_nests)
            for idx, gi, fi in zip(order, new_nests, new_fit2):
                self.pop[idx], self.fitness[idx] = gi, fi


def main():
    algo = Cs()
    res = algo.solve(sezgi.bbob(1, algo.dim, 1), budget=BUDGET, seed=SEED)
    print(f"cs (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
