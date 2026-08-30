"""Pure-Python Bat Algorithm (Yang, X.-S. 2010, "A new metaheuristic
bat-inspired algorithm", in: Nature Inspired Cooperative Strategies for
Optimization (NICSO 2010), Studies in Computational Intelligence vol.
284, Springer, pp. 65-74).

Teaches the SAME pinned update equations as sezgi's `gen/ba` +
`replace/bat-loudness-greedy` Rust components (see
crates/components/src/ba.rs's module doc) with nothing but the standard
library (`random`, `math` -- no numpy), driven through
`sezgi.EvalSession` so evaluation counting and budget enforcement come for
free. This script does NOT reproduce the Rust component's RNG-stream (draw
order) contract -- only the same mathematical update rule -- so results
will differ from the `bat` preset even at the same seed; see
`examples/specs/ba.toml` for the pinned-RNG preset itself.

Tier note: a **labeled metaphor preset** -- faithful to the primary
source's own reference MATLAB implementation (`bat_algorithm.m`), with a
pinned deterministic draw order, but NOT validated against the paper's
reported benchmark numbers. Equivalence critique: Camacho-Villalón,
Dorigo & Stützle (International Transactions in Operational Research,
six-algorithm critique: grey wolf, moth-flame, whale, firefly, bat,
antlion) -- cited here conservatively, as background for why this is
"labeled metaphor" rather than a mechanism sezgi treats as novel.

Reproduces two VERIFIED source quirks verbatim (not "fixed"): the
frequency draw is `Q = Qmin + (Qmin-Qmax)*u`, an inverted `[-2,0]` range
(not the paper-prose-intuitive `[0,2]`), and the velocity term uses
`(X - X_best)`, not `(X_best - X)` -- the two sign inversions compose into
a genuine pull TOWARD best, not away from it. `A=0.5`/`r=0.5` are FIXED
constants in the verified source (the demo explicitly does not implement
loudness/pulse-rate dynamics), so `velocity` is the only persisted state
here, carried in a plain Python list across generations (the Rust
component's `ba/velocity` blackboard key plays the same role there).

No cross-algorithm quality claims are made or implied here -- this is a
single seed on a single problem, reported as a gap, not a ranking.
"""
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30  # number of bats
SEED = 42
LO, HI = -5.0, 5.0
QMIN, QMAX = 0.0, 2.0
R0 = 0.5   # fixed pulse rate
A0 = 0.5   # fixed loudness
LOCAL_WALK_SCALE = 0.001


def ba_frequency(u):
    """VERIFIED VERBATIM against bat_algorithm.m: Qmin + (Qmin-Qmax)*u --
    note the sign, giving [-2, 0] for the pinned defaults, not [0, 2]."""
    return QMIN + (QMIN - QMAX) * u


def ba_velocity_step(v_d, x_d, best_d, freq):
    return v_d + (x_d - best_d) * freq


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/ba-py", algo_name="ba-py", seed=SEED)

    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    used = POP_SIZE
    velocity = [[0.0] * DIM for _ in range(POP_SIZE)]

    while used + POP_SIZE <= BUDGET:
        # best: current-population fitness argmin, ties -> lower index,
        # computed once before any draws (sezgi simplification, shared
        # with the Rust preset: bat_algorithm.m persists a best-ever).
        best_idx = min(range(POP_SIZE), key=lambda i: (fitness[i], i))
        best = pop[best_idx]

        offspring = []
        for i in range(POP_SIZE):
            x = pop[i]
            # ONE frequency draw per bat (not per dimension).
            freq = ba_frequency(rng.random())
            v_i = [ba_velocity_step(velocity[i][d], x[d], best[d], freq) for d in range(DIM)]
            s = [x[d] + v_i[d] for d in range(DIM)]

            # Local-walk trigger: ONE draw per bat.
            trigger = rng.random()
            if trigger > R0:
                # COMPLETE OVERWRITE of the candidate; the velocity move
                # above is discarded for `s` but v_i still commits below.
                s = [best[d] + LOCAL_WALK_SCALE * rng.gauss(0.0, 1.0) for d in range(DIM)]

            velocity[i] = v_i
            offspring.append(s)

        # boundary/clamp applied AFTER generate returns, same timing as
        # the Rust preset's separate boundary stage.
        offspring = [[clamp(v, LO, HI) for v in row] for row in offspring]

        new_fitness = session.evaluate(offspring)
        used += POP_SIZE

        # replace/bat-loudness-greedy: accept iff off_fit[i] <= own fit[i]
        # AND a FRESH, UNCONDITIONAL loudness draw < A0 -- the draw always
        # fires (MATLAB's non-short-circuiting `&`), whether or not the
        # fitness test holds.
        for i in range(POP_SIZE):
            loud_draw = rng.random()
            if new_fitness[i] <= fitness[i] and loud_draw < A0:
                pop[i], fitness[i] = offspring[i], new_fitness[i]

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"ba: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
