"""Pure-Python Gravitational Search Algorithm (Rashedi, E.,
Nezamabadi-pour, H. & Saryazdi, S. 2009, "GSA: A Gravitational Search
Algorithm", Information Sciences 179(13), 2232-2248).

Teaches the SAME pinned update equations as sezgi's `gen/gsa` +
`replace/generational` Rust component (see crates/components/src/gsa.rs's
module doc, verified against Esmat Rashedi's own `GSA.m`/`Gconstant.m`/
`massCalculation.m`/`Gfield.m`/`move.m`) with nothing but the standard
library, driven through `sezgi.EvalSession`. Does NOT reproduce the Rust
component's RNG-stream (draw order) contract, only the same update rule;
see examples/specs/gsa.toml for the pinned-RNG preset.

Tier note: a **labeled metaphor preset** -- faithful to GSA.m's own
equations and loop structure, pinned draw order, but NOT validated against
the paper's reported benchmark numbers. No established equivalence
critique covers GSA -- primary-source only.

This is the wave's LAST stateful algorithm: the per-agent velocity is a
genuine persisted memory, carried in a plain Python list across
generations (the Rust component's `gsa/velocity` blackboard key plays the
same role there). Per generation: masses are normalized (`(fit-worst)/
(best-worst)`, EXCEPT a degenerate all-equal-fitness guard giving uniform
1/N masses, both cases normalized to sum to 1), G decays exponentially
with progress, the Kbest elite-set SHRINKS with progress (`2 + (1-progress)
*98` percent, NO floor -- it can legitimately reach 0 agents late in a
run), and for every target agent, every OTHER agent in the (mass-sorted)
Kbest set contributes a force with its OWN independent draw per dimension
(`rand*M_j*(x_j-x_i)/(R+eps)` -- notably NO `M_i` term anywhere, since
`Gfield.m`'s own comment states `Mp(i)/Mi(i)=1`). Velocity then updates as
`v' = rand*v + accel` (the random factor MULTIPLIES the OLD velocity,
unlike ba.py's purely-additive term) and position as `x' = x + v'`.

No cross-algorithm quality claims are made or implied here -- this is a
single seed on a single problem, reported as a gap, not a ranking.
"""
import math
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30  # number of agents
SEED = 42
LO, HI = -5.0, 5.0
G0 = 100.0
ALPHA = 20.0
FINAL_PERCENT = 2.0
EPS = 2.220446049250313e-16  # f64::EPSILON / numpy.finfo(float).eps


def gsa_mass(fit):
    fmax, fmin = max(fit), min(fit)
    if fmax == fmin:
        m = [1.0] * len(fit)
    else:
        best, worst = fmin, fmax
        m = [(f - worst) / (best - worst) for f in fit]
    total = sum(m)
    return [v / total for v in m]


def gsa_g(progress):
    return G0 * math.exp(-ALPHA * progress)


def gsa_kbest_count(n, progress):
    percent = FINAL_PERCENT + (1.0 - progress) * (100.0 - FINAL_PERCENT)
    return round(n * percent / 100.0)  # NOT clamped to a minimum of 1 -- can reach 0


def gsa_kbest_order(mass):
    """Indices sorted by mass DESCENDING -- a stable sort, ties keep the
    original agent order."""
    return sorted(range(len(mass)), key=lambda i: -mass[i])


def gsa_force_term(rand, mass_j, xj_d, xi_d, r_dist):
    """NO M_i factor anywhere -- Gfield.m's own comment: Mp(i)/Mi(i)=1."""
    return rand * mass_j * (xj_d - xi_d) / (r_dist + EPS)


def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def gsa_velocity_step(v_d, accel_d, r1):
    """The random factor MULTIPLIES the OLD velocity, unlike ba.py's
    purely-additive term."""
    return r1 * v_d + accel_d


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/gsa-py", algo_name="gsa-py", seed=SEED)

    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    used = POP_SIZE
    velocity = [[0.0] * DIM for _ in range(POP_SIZE)]

    while used + POP_SIZE <= BUDGET:
        progress = min(max(used / BUDGET, 0.0), 1.0)
        mass = gsa_mass(fitness)
        g_const = gsa_g(progress)
        kbest = gsa_kbest_count(POP_SIZE, progress)
        order = gsa_kbest_order(mass)

        # Gfield.m: ALL per-(i,j,d) force draws happen first, completing the
        # WHOLE acceleration array, BEFORE move.m's per-(i,d) draws start.
        accel = [[0.0] * DIM for _ in range(POP_SIZE)]
        for i in range(POP_SIZE):
            force_i = [0.0] * DIM
            for ii in range(kbest):
                j = order[ii]
                if j == i:
                    continue
                r_dist = euclidean(pop[i], pop[j])
                for d in range(DIM):
                    rnd = rng.random()
                    force_i[d] += gsa_force_term(rnd, mass[j], pop[j][d], pop[i][d], r_dist)
            for d in range(DIM):
                accel[i][d] = g_const * force_i[d]

        # move.m: velocity + position update, a SEPARATE loop after Gfield.
        offspring = []
        for i in range(POP_SIZE):
            xs = []
            for d in range(DIM):
                r1 = rng.random()
                v_new = gsa_velocity_step(velocity[i][d], accel[i][d], r1)
                velocity[i][d] = v_new
                xs.append(clamp(pop[i][d] + v_new, LO, HI))
            offspring.append(xs)

        new_fitness = session.evaluate(offspring)
        used += POP_SIZE
        pop, fitness = offspring, new_fitness  # replace/generational: unconditional overwrite

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"gsa: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
