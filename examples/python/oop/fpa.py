"""OOP twin of examples/python/fpa.py -- same math, same RNG draw order.

Port of the pure-Python Flower Pollination Algorithm script onto
sezgi.AskTellAlgorithm. The pure script (and the Rust module doc it cites) remains
the provenance for the update equations, including the two verified
deltas (the `u > p` GLOBAL-branch orientation and the local branch's j/k
distinct-from-each-other-only indices); this file re-derives NOTHING and
must reproduce the pure script's evals_used/best_f/gap output bit-for-bit
at the same seed -- enforced by py-sezgi/tests/test_examples_oop_parity.py.

Carries its own copy of the Mantegna Levy helper (cs_dim_step/gauss_polar/
levy_mantegna), mirroring the pure scripts' own deliberate duplication
between cs.py and fpa.py -- each example file stays self-contained.
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
    """Reused verbatim from cs.py -- FPA's global-pollination step is
    algebraically identical to CS's Levy-toward-best step (finding 2)."""
    return x_i_d + alpha_step * levy * (x_i_d - x_best_d)


def fpa_is_global_branch(u, p):
    """VERIFIED orientation: u > p selects GLOBAL, not u < p."""
    return u > p


def fpa_local_dim_step(x_d, x_j_d, x_k_d, epsilon):
    return x_d + epsilon * (x_j_d - x_k_d)


def pick_two_distinct(rng, n):
    """j, k distinct FROM EACH OTHER only -- self (i) is never excluded,
    matching fpa_demo.m's `JK=randperm(n)` first-two-entries behavior."""
    j = rng.randrange(n)
    while True:
        k = rng.randrange(n)
        if k != j:
            return j, k


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Fpa(sezgi.AskTellAlgorithm):
    """Flower Pollination Algorithm (Yang 2012, "Flower Pollination
    Algorithm for Global Optimization", UCNC 2012, LNCS vol. 7445,
    Springer, pp. 240-249), modeling the switch-probability-gated
    global/local pollination cycle explicitly over
    `sezgi.AskTellAlgorithm`'s ask/tell loop. `dim`: problem
    dimensionality. `pop_size`: flower count. `p_switch`: probability a
    flower takes the global (Levy-toward-best) branch rather than the
    local (neighbor-interpolation) branch. `alpha_step`: step-size scale
    multiplying the global branch's Levy draw. `levy_alpha`: the
    Mantegna (1994) Levy-stability exponent."""

    name = "fpa"

    def __init__(self, dim=5, pop_size=25, p_switch=0.8, alpha_step=0.01,
                 levy_alpha=1.5):
        self.dim = dim
        self.pop_size = pop_size
        self.p_switch = p_switch
        self.alpha_step = alpha_step
        self.levy_alpha = levy_alpha

    def setup(self, ctx):
        self.pop = [ctx.random_point() for _ in range(self.pop_size)]
        self.fitness = ctx.evaluate(self.pop)

    def step(self, ctx):
        lo, hi = ctx.bounds

        # Attractor: current-population fitness argmin (ties -> lower
        # index), computed once per generation before any draws -- a
        # documented simplification of the source's persisted global-best
        # (shared with ba.py/ssa.py, not FPA-specific).
        best = min(range(self.pop_size), key=lambda i: (self.fitness[i], i))
        x_best = self.pop[best]

        offspring = []
        for i in range(self.pop_size):
            x = self.pop[i]
            u = ctx.rng.random()
            if fpa_is_global_branch(u, self.p_switch):
                new_x = [clamp(cs_dim_step(x[d], x_best[d],
                                            levy_mantegna(ctx.rng, self.levy_alpha),
                                            self.alpha_step), lo, hi) for d in range(self.dim)]
            else:
                epsilon = ctx.rng.random()  # scalar, shared across every dimension
                j, k = pick_two_distinct(ctx.rng, self.pop_size)
                x_j, x_k = self.pop[j], self.pop[k]
                new_x = [clamp(fpa_local_dim_step(x[d], x_j[d], x_k[d], epsilon), lo, hi) for d in range(self.dim)]
            offspring.append(new_x)

        new_fitness = ctx.evaluate(offspring)

        # replace/one-to-one-greedy
        for i in range(self.pop_size):
            if new_fitness[i] < self.fitness[i]:
                self.pop[i], self.fitness[i] = offspring[i], new_fitness[i]


def main():
    algo = Fpa()
    res = algo.solve(sezgi.bbob(1, algo.dim, 1), budget=BUDGET, seed=SEED)
    print(f"fpa (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
