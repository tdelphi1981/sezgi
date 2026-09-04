"""OOP twin of examples/python/gsa.py -- same math, same RNG draw order.

Port of the pure-Python Gravitational Search Algorithm script onto
sezgi.AskTellAlgorithm. The pure script (and the Rust module doc it cites) remains
the provenance for the update equations; this file re-derives NOTHING and
must reproduce the pure script's evals_used/best_f/gap output bit-for-bit
at the same seed -- enforced by py-sezgi/tests/test_examples_oop_parity.py.

The per-agent velocity is a genuine persisted memory, carried in
self.velocity across generations. The Kbest elite set SHRINKS with progress
with NO floor (can reach 0 agents); the force loop nesting (target agent,
then Kbest members, then dimension, with one draw per (i,j,d)) and the
separate move.m-style velocity/position loop after it are copied verbatim.
"""
import math

import sezgi

BUDGET = 2000
SEED = 42


def gsa_mass(fit):
    fmax, fmin = max(fit), min(fit)
    if fmax == fmin:
        m = [1.0] * len(fit)
    else:
        best, worst = fmin, fmax
        m = [(f - worst) / (best - worst) for f in fit]
    total = sum(m)
    return [v / total for v in m]


def gsa_g(progress, g0, alpha):
    return g0 * math.exp(-alpha * progress)


def gsa_kbest_count(n, progress, final_percent):
    percent = final_percent + (1.0 - progress) * (100.0 - final_percent)
    return round(n * percent / 100.0)  # NOT clamped to a minimum of 1 -- can reach 0


def gsa_kbest_order(mass):
    """Indices sorted by mass DESCENDING -- a stable sort, ties keep the
    original agent order."""
    return sorted(range(len(mass)), key=lambda i: -mass[i])


def gsa_force_term(rand, mass_j, xj_d, xi_d, r_dist, eps):
    """NO M_i factor anywhere -- Gfield.m's own comment: Mp(i)/Mi(i)=1."""
    return rand * mass_j * (xj_d - xi_d) / (r_dist + eps)


def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gsa_velocity_step(v_d, accel_d, r1):
    """The random factor MULTIPLIES the OLD velocity, unlike ba.py's
    purely-additive term."""
    return r1 * v_d + accel_d


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Gsa(sezgi.AskTellAlgorithm):
    """Gravitational Search Algorithm (Rashedi, Nezamabadi-pour & Saryazdi
    2009, Information Sciences 179(13), 2232-2248), modeling the
    mass-weighted force field plus a velocity/position move step
    explicitly over `sezgi.AskTellAlgorithm`'s ask/tell loop, with a
    genuinely persisted per-agent velocity. `dim`: problem
    dimensionality. `pop_size`: number of agents. `g0`/`alpha`: the
    gravitational constant's initial value and exponential decay rate.
    `final_percent`: the Kbest elite set's final size as a percentage of
    `pop_size` (linearly interpolated from 100% at the start, with NO
    floor -- the elite set can shrink to 0 agents). `eps`: guards a
    zero-distance pair's division in the force term."""

    name = "gsa"

    def __init__(self, dim=5, pop_size=30, g0=100.0, alpha=20.0,
                 final_percent=2.0, eps=2.220446049250313e-16):
        self.dim = dim
        self.pop_size = pop_size
        self.g0 = g0
        self.alpha = alpha
        self.final_percent = final_percent
        self.eps = eps

    def setup(self, ctx):
        self.pop = [ctx.random_point() for _ in range(self.pop_size)]
        self.fitness = ctx.evaluate(self.pop)
        self.velocity = [[0.0] * self.dim for _ in range(self.pop_size)]

    def step(self, ctx):
        lo, hi = ctx.bounds
        progress = min(max(ctx.evals_used / ctx.budget, 0.0), 1.0)
        mass = gsa_mass(self.fitness)
        g_const = gsa_g(progress, self.g0, self.alpha)
        kbest = gsa_kbest_count(self.pop_size, progress, self.final_percent)
        order = gsa_kbest_order(mass)

        # Gfield.m: ALL per-(i,j,d) force draws happen first, completing the
        # WHOLE acceleration array, BEFORE move.m's per-(i,d) draws start.
        accel = [[0.0] * self.dim for _ in range(self.pop_size)]
        for i in range(self.pop_size):
            force_i = [0.0] * self.dim
            for ii in range(kbest):
                j = order[ii]
                if j == i:
                    continue
                r_dist = euclidean(self.pop[i], self.pop[j])
                for d in range(self.dim):
                    rnd = ctx.rng.random()
                    force_i[d] += gsa_force_term(rnd, mass[j], self.pop[j][d], self.pop[i][d], r_dist, self.eps)
            for d in range(self.dim):
                accel[i][d] = g_const * force_i[d]

        # move.m: velocity + position update, a SEPARATE loop after Gfield.
        offspring = []
        for i in range(self.pop_size):
            xs = []
            for d in range(self.dim):
                r1 = ctx.rng.random()
                v_new = gsa_velocity_step(self.velocity[i][d], accel[i][d], r1)
                self.velocity[i][d] = v_new
                xs.append(clamp(self.pop[i][d] + v_new, lo, hi))
            offspring.append(xs)

        new_fitness = ctx.evaluate(offspring)
        self.pop, self.fitness = offspring, new_fitness  # replace/generational: unconditional overwrite


def main():
    algo = Gsa()
    res = algo.solve(sezgi.bbob(1, algo.dim, 1), budget=BUDGET, seed=SEED)
    print(f"gsa (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
