"""Pure-Python Salp Swarm Algorithm (Mirjalili, S., Gandomi, A.H.,
Mirjalili, S.Z., Saremi, S., Faris, H. & Mirjalili, S.M. 2017, "Salp Swarm
Algorithm: A bio-inspired optimizer for engineering design problems",
Advances in Engineering Software 114, 163-191).

Teaches the SAME pinned update equations as sezgi's `gen/ssa` Rust
component (see crates/components/src/ssa.rs's module doc) with nothing
but the standard library (`random`, `math` -- no numpy), driven through
`sezgi.EvalSession` so evaluation counting and budget enforcement come for
free. This script does NOT reproduce the Rust component's RNG-stream (draw
order) contract -- only the same mathematical update rule -- so results
will differ from the `ssa` preset even at the same seed; see
`examples/specs/ssa.toml` for the pinned-RNG preset itself.

Tier note: a **labeled metaphor preset** -- faithful to the primary
source's own reference MATLAB implementation (`SSA.m`), with a pinned
deterministic draw order, but NOT validated against the paper's reported
benchmark numbers. No established equivalence-critique reference covers
SSA -- cited here as primary-source only.

sezgi simplification (shared with the Rust preset): `SSA.m`'s
`FoodPosition` is a persisted best-ever; this uses the current
population's fitness argmin instead (this wave's established
current-generation-best convention for single-scalar attractors).

No cross-algorithm quality claims are made or implied here -- this is a
single seed on a single problem, reported as a gap, not a ranking.
"""
import math
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30  # number of salps
SEED = 42
LO, HI = -5.0, 5.0


def ssa_c1(progress):
    return 2.0 * math.exp(-((4.0 * progress) ** 2))


def ssa_leader_dim_step(food_d, c1, c2, c3, lo, hi):
    term = c1 * ((hi - lo) * c2 + lo)
    return food_d + term if c3 < 0.5 else food_d - term


def ssa_follower_dim_step(x_old_d, prev_new_d):
    return (x_old_d + prev_new_d) / 2.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/ssa-py", algo_name="ssa-py", seed=SEED)

    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    used = POP_SIZE
    leader_count = POP_SIZE // 2  # fixed positional split, not fitness-based

    while used + POP_SIZE <= BUDGET:
        progress = min(max(used / BUDGET, 0.0), 1.0)
        c1 = ssa_c1(progress)

        # Food: current-population fitness argmin, ties -> lower index,
        # computed once before any draws.
        food_idx = min(range(POP_SIZE), key=lambda i: (fitness[i], i))
        food = pop[food_idx]

        # Built SEQUENTIALLY in ascending index order: followers chain off
        # the already-computed offspring entries from earlier this same
        # sweep (SSA.m's in-place transpose-and-reassign semantics).
        offspring = []
        for i in range(POP_SIZE):
            x_old = pop[i]
            if i < leader_count:
                new_x = []
                for d in range(DIM):
                    # Pinned draw order: c2, then c3, per (leader, d).
                    c2 = rng.random()
                    c3 = rng.random()
                    new_x.append(clamp(ssa_leader_dim_step(food[d], c1, c2, c3, LO, HI), LO, HI))
            else:
                # Follower: zero draws; chains off offspring[i-1], the
                # ALREADY-COMPUTED entry from this same sweep.
                prev = offspring[i - 1]
                new_x = [clamp(ssa_follower_dim_step(x_old[d], prev[d]), LO, HI) for d in range(DIM)]
            offspring.append(new_x)

        # replace/generational: SSA.m overwrites every salp unconditionally.
        fitness = session.evaluate(offspring)
        pop = offspring
        used += POP_SIZE

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"ssa: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
