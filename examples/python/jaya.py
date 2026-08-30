"""Pure-Python JAYA (Rao, R.V. 2016, "Jaya: A Simple and New Optimization
Algorithm for Solving Constrained and Unconstrained Optimization
Problems", International Journal of Industrial Engineering Computations
7(1), 19-34).

Teaches the SAME pinned update equations as sezgi's `gen/jaya` Rust
component (see crates/components/src/jaya.rs's module doc) with nothing
but the standard library (`random`, `math` -- no numpy), driven through
`sezgi.EvalSession` so evaluation counting and budget enforcement come for
free. This script does NOT reproduce the Rust component's RNG-stream (draw
order) contract -- only the same mathematical update rule -- so results
will differ from the `jaya` preset even at the same seed; see
`examples/specs/jaya.toml` for the pinned-RNG preset itself.

Tier note: a **labeled metaphor preset** -- faithful to the primary
paper's stated Eq. (1) AND its own worked numerical example (Section 2.1),
with a pinned deterministic draw order, but NOT validated against the
paper's reported benchmark numbers. No established equivalence-critique
reference covers JAYA -- cited here as primary-source only.

No cross-algorithm quality claims are made or implied here -- this is a
single seed on a single problem, reported as a gap, not a ranking.
"""
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30  # candidate count
SEED = 42
LO, HI = -5.0, 5.0


def jaya_dim_step(x_d, x_best_d, x_worst_d, r1, r2):
    """Eq. (1): X' = X + r1*(X_best - |X|) - r2*(X_worst - |X|)."""
    abs_x = abs(x_d)
    return x_d + r1 * (x_best_d - abs_x) - r2 * (x_worst_d - abs_x)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/jaya-py", algo_name="jaya-py", seed=SEED)

    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    used = POP_SIZE

    while used + POP_SIZE <= BUDGET:
        # Current-population argmin/argmax, both computed before any draws.
        best = min(range(POP_SIZE), key=lambda i: (fitness[i], i))
        worst = max(range(POP_SIZE), key=lambda i: (fitness[i], i))
        x_best, x_worst = pop[best], pop[worst]

        # Pinned: r1[d], r2[d] drawn ONCE PER DIMENSION PER GENERATION,
        # shared across every agent (the paper's own worked example reuses
        # the same pair across all candidates) -- not fresh per (i, d).
        r1 = [rng.random() for _ in range(DIM)]
        r2 = [rng.random() for _ in range(DIM)]

        offspring = [
            [clamp(jaya_dim_step(pop[i][d], x_best[d], x_worst[d], r1[d], r2[d]), LO, HI)
             for d in range(DIM)]
            for i in range(POP_SIZE)
        ]
        new_fitness = session.evaluate(offspring)
        used += POP_SIZE

        # replace/one-to-one-greedy: the paper's Fig. 1 flowchart accepts
        # X' only if strictly better, else keeps the previous solution.
        for i in range(POP_SIZE):
            if new_fitness[i] < fitness[i]:
                pop[i], fitness[i] = offspring[i], new_fitness[i]

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"jaya: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
