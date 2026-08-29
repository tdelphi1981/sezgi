"""Pure-Python Whale Optimization Algorithm (Mirjalili & Lewis 2016, "The
Whale Optimization Algorithm", Advances in Engineering Software).

Teaches the SAME pinned update equations as sezgi's `gen/woa` Rust component
(see crates/components/src/woa.rs's module doc) with nothing but the
standard library (`random`, `math` -- no numpy), driven through
`sezgi.EvalSession`. Does NOT reproduce the Rust component's RNG-stream
(draw order) contract, only the same update rule; see
`examples/specs/woa.toml` for the pinned-RNG preset.

Equivalence critique: Camacho-Villalón, Dorigo & Stützle (International
Transactions in Operational Research), which names WOA ("whale") explicitly
among six metaphor-based algorithms shown to be, component for component,
relabeled special cases of older operators.

No cross-algorithm quality claims are made or implied here -- single seed,
single problem, reported as a gap.
"""
import math
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30
SEED = 42
LO, HI = -5.0, 5.0


def woa_encircle_step(target_d, x_d, big_a, big_c):
    return target_d - big_a * abs(big_c * target_d - x_d)


def woa_spiral_step(x_best_d, x_d, l, b=1.0):
    d = abs(x_best_d - x_d)
    return d * math.exp(b * l) * math.cos(2.0 * math.pi * l) + x_best_d


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/woa-py", algo_name="woa-py", seed=SEED)

    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    used = POP_SIZE

    while used + POP_SIZE <= BUDGET:
        progress = min(max(used / BUDGET, 0.0), 1.0)
        a = 2.0 - 2.0 * progress
        a2 = -1.0 - progress

        best = min(range(POP_SIZE), key=lambda i: (fitness[i], i))
        x_best = pop[best]

        offspring = []
        for i in range(POP_SIZE):
            x = pop[i]
            # p, and then r1/r2 (search branch) or l_raw (spiral branch),
            # are all drawn ONCE per whale, before the dimension loop --
            # matching the reference MATLAB (WOA.m) exactly. Only the
            # random-leader index is drawn per dimension.
            p = rng.random()
            new_x = []
            if p < 0.5:
                r1, r2 = rng.random(), rng.random()
                big_a = 2.0 * a * r1 - a
                big_c = 2.0 * r2
                for d in range(DIM):
                    if abs(big_a) < 1.0:
                        target_d = x_best[d]
                    else:
                        j = rng.randrange(POP_SIZE)  # may equal i, matches reference MATLAB
                        target_d = pop[j][d]
                    val = woa_encircle_step(target_d, x[d], big_a, big_c)
                    new_x.append(clamp(val, LO, HI))
            else:
                l_raw = rng.random()
                l = (a2 - 1.0) * l_raw + 1.0
                for d in range(DIM):
                    val = woa_spiral_step(x_best[d], x[d], l)
                    new_x.append(clamp(val, LO, HI))
            offspring.append(new_x)

        # Unconditional generational replacement: only run a generation if
        # the whole offspring batch fits the remaining budget (mirrors the
        # Rust engine's own budget-exhaustion clean break).
        fitness = session.evaluate(offspring)
        pop = offspring
        used += POP_SIZE

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"woa: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
