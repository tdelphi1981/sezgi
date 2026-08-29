"""Pure-Python Grey Wolf Optimizer (Mirjalili, Mirjalili & Lewis 2014,
"Grey Wolf Optimizer", Advances in Engineering Software).

Teaches the SAME pinned update equations as sezgi's `gen/gwo` Rust component
(see crates/components/src/gwo.rs's module doc) with nothing but the
standard library (`random`, `math` -- no numpy), driven through
`sezgi.EvalSession` so evaluation counting and budget enforcement come for
free. This script does NOT reproduce the Rust component's RNG-stream (draw
order) contract -- only the same mathematical update rule -- so results will
differ from the `gwo` preset even at the same seed; see `examples/specs/gwo.toml`
for the pinned-RNG preset itself.

Equivalence critique (why sezgi treats this as a "labeled metaphor", not a
novel mechanism): Camacho-Villalón, Dorigo & Stützle (ANTS 2020,
three-algorithm study); Camacho-Villalón, Dorigo & Stützle (International
Transactions in Operational Research, six-algorithm journal extension).

No cross-algorithm quality claims are made or implied here -- this is a
single seed on a single problem, reported as a gap, not a ranking.
"""
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30
SEED = 42
LO, HI = -5.0, 5.0


def gwo_dim_step(a, leaders_d, x_d, draws):
    """One dimension's update: mean of the three leaders' contributions."""
    total = 0.0
    for k in range(3):
        r1, r2 = draws[2 * k], draws[2 * k + 1]
        big_a = 2.0 * a * r1 - a
        c = 2.0 * r2
        total += leaders_d[k] - big_a * abs(c * leaders_d[k] - x_d)
    return total / 3.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/gwo-py", algo_name="gwo-py", seed=SEED)

    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    used = POP_SIZE

    while used + POP_SIZE <= BUDGET:
        progress = min(max(used / BUDGET, 0.0), 1.0)
        a = 2.0 - 2.0 * progress

        # Leaders: best 3 by fitness ascending, ties -> lower index.
        order = sorted(range(POP_SIZE), key=lambda i: (fitness[i], i))
        leader_x = [pop[order[k]] for k in range(3)]

        offspring = []
        for i in range(POP_SIZE):
            x = pop[i]
            new_x = []
            for d in range(DIM):
                leaders_d = [leader_x[k][d] for k in range(3)]
                draws = [rng.random() for _ in range(6)]
                new_x.append(clamp(gwo_dim_step(a, leaders_d, x[d], draws), LO, HI))
            offspring.append(new_x)

        # Unconditional generational replacement (replace/generational): a
        # partial final generation would leave a stale/offspring mix, so --
        # like the Rust engine's own budget-exhaustion break -- we only run a
        # generation if the WHOLE offspring batch fits the remaining budget.
        fitness = session.evaluate(offspring)
        pop = offspring
        used += POP_SIZE

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"gwo: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
