"""Pure-Python Sine Cosine Algorithm (Mirjalili, S. 2016, "SCA: A Sine
Cosine Algorithm for Solving Optimization Problems", Knowledge-Based
Systems 96, 120-133).

Teaches the SAME pinned update equations as sezgi's `gen/sca` Rust
component (see crates/components/src/sca.rs's module doc) with nothing but
the standard library (`random`, `math` -- no numpy), driven through
`sezgi.EvalSession` so evaluation counting and budget enforcement come for
free. This script does NOT reproduce the Rust component's RNG-stream (draw
order) contract -- only the same mathematical update rule -- so results will
differ from the `sca` preset even at the same seed; see
`examples/specs/sca.toml` for the pinned-RNG preset itself.

Tier note: a **labeled metaphor preset** -- faithful to the primary
source's own reference MATLAB implementation (`SCA.m`), with a pinned
deterministic draw order, but NOT validated against the paper's reported
benchmark numbers. No established equivalence-critique reference covers
SCA -- cited here as primary-source only.

No cross-algorithm quality claims are made or implied here -- this is a
single seed on a single problem, reported as a gap, not a ranking.
"""
import math
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30  # number of search agents
SEED = 42
LO, HI = -5.0, 5.0


def sca_dim_step(x_d, x_best_d, r1, r2, r3, r4):
    """Eq. 3.1 (sin branch, r4 < 0.5) / Eq. 3.2 (cos branch)."""
    delta = abs(r3 * x_best_d - x_d)
    if r4 < 0.5:
        return x_d + r1 * math.sin(r2) * delta
    return x_d + r1 * math.cos(r2) * delta


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/sca-py", algo_name="sca-py", seed=SEED)

    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    used = POP_SIZE

    while used + POP_SIZE <= BUDGET:
        progress = min(max(used / BUDGET, 0.0), 1.0)
        a = 2.0
        r1 = a * (1.0 - progress)  # = a - t*(a/T) = 2 - 2*progress

        # Destination_position: current population's fitness argmin,
        # ties -> lower index.
        best = min(range(POP_SIZE), key=lambda i: (fitness[i], i))
        x_best = pop[best]

        offspring = []
        for i in range(POP_SIZE):
            x = pop[i]
            new_x = []
            for d in range(DIM):
                # Pinned draw order: r2, then r3, then r4, fresh for every
                # (i, d) -- always all three, regardless of branch.
                r2 = 2.0 * math.pi * rng.random()
                r3 = 2.0 * rng.random()
                r4 = rng.random()
                new_x.append(clamp(sca_dim_step(x[d], x_best[d], r1, r2, r3, r4), LO, HI))
            offspring.append(new_x)

        # replace/generational: SCA.m overwrites every agent's position
        # every iteration unconditionally (no greedy comparison).
        fitness = session.evaluate(offspring)
        pop = offspring
        used += POP_SIZE

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"sca: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
