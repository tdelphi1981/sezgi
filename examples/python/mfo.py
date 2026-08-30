"""Pure-Python Moth-Flame Optimization (Mirjalili, S. 2015, "Moth-flame
optimization algorithm: A novel nature-inspired heuristic paradigm",
Knowledge-Based Systems 89, 228-249).

Teaches the SAME pinned update equations and flame-memory mechanism as
sezgi's `gen/mfo` + `adapter/mfo-flame-update` Rust components (see
crates/components/src/mfo.rs's module doc) with nothing but the standard
library (`random`, `math` -- no numpy), driven through `sezgi.EvalSession`
so evaluation counting and budget enforcement come for free. This script
does NOT reproduce the Rust component's RNG-stream (draw order) contract
-- only the same mathematical update rule -- so results will differ from
the `mfo` preset even at the same seed; see `examples/specs/mfo.toml` for
the pinned-RNG preset itself.

Tier note: a **labeled metaphor preset** -- faithful to the primary
source's own reference MATLAB implementation (`MFO.m`), with a pinned
deterministic draw order, but NOT validated against the paper's reported
benchmark numbers. Equivalence critique: Camacho-Villalón, Dorigo &
Stützle (International Transactions in Operational Research, six-algorithm
critique: grey wolf, moth-flame, whale, firefly, bat, antlion) -- cited
here conservatively, as background for why this is "labeled metaphor"
rather than a mechanism sezgi treats as novel.

The flame memory (`flames`/`flame_fitness`) is a genuine persisted archive
here too, held in plain Python variables across generations (this pure
script has no blackboard -- the Rust component's `adapter/mfo-flame-
update` is this state's canonical owner there; here the `main()` loop
plays that role directly).

No cross-algorithm quality claims are made or implied here -- this is a
single seed on a single problem, reported as a gap, not a ranking.
"""
import math
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30  # number of search agents (also the flame archive size)
SEED = 42
LO, HI = -5.0, 5.0


def mfo_dim_step(x_d, own_flame_d, target_flame_d, t):
    b = 1.0
    distance_to_flame = abs(own_flame_d - x_d)
    return distance_to_flame * math.exp(b * t) * math.cos(t * 2.0 * math.pi) + target_flame_d


def mfo_flame_count(n, progress):
    """Eq. (3.14): round(N - progress*(N-1)), clamped to [1, n]."""
    raw = n - progress * (n - 1)
    return max(1, min(n, round(raw)))


def merge_and_truncate(pop_fitness, pop_positions, flame_fitness, flame_positions):
    """Concatenate (pop first, flames second), stable-sort ascending by
    fitness, truncate to len(pop_fitness) -- MFO.m's own elitist
    double_population=[previous_population; best_flames] step."""
    combined = list(zip(pop_fitness, pop_positions)) + list(zip(flame_fitness, flame_positions))
    combined.sort(key=lambda fp: fp[0])
    combined = combined[:len(pop_fitness)]
    return [f for f, _ in combined], [p for _, p in combined]


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/mfo-py", algo_name="mfo-py", seed=SEED)

    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    used = POP_SIZE

    # Bootstrap: generation 0 seeds the flame archive from the sorted
    # initial population (a merge against an EMPTY old-flame set).
    flame_fitness, flames = merge_and_truncate(fitness, pop, [], [])

    while used + POP_SIZE <= BUDGET:
        progress = min(max(used / BUDGET, 0.0), 1.0)
        a = -1.0 - progress
        flame_count = mfo_flame_count(POP_SIZE, progress)

        offspring = []
        for i in range(POP_SIZE):
            x = pop[i]
            own_flame = flames[i]
            target_flame = flames[min(i, flame_count - 1)]
            new_x = []
            for d in range(DIM):
                t = (a - 1.0) * rng.random() + 1.0
                new_x.append(clamp(mfo_dim_step(x[d], own_flame[d], target_flame[d], t), LO, HI))
            offspring.append(new_x)

        # replace/generational: the flames, not the moth population,
        # carry the elitism, so the moths are overwritten unconditionally.
        new_fitness = session.evaluate(offspring)
        used += POP_SIZE
        pop, fitness = offspring, new_fitness

        # adapter/mfo-flame-update: merge this generation's freshly moved
        # moths against the PREVIOUS flames, truncate back to POP_SIZE.
        flame_fitness, flames = merge_and_truncate(fitness, pop, flame_fitness, flames)

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"mfo: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
