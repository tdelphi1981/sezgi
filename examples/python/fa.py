"""Pure-Python Firefly Algorithm (Yang, X.-S., Nature-Inspired Metaheuristic
Algorithms, 2nd ed., Luniver Press, 2010).

Teaches the SAME pinned update equations as sezgi's `gen/fa` Rust
component (see crates/components/src/fa.rs's module doc) with nothing but
the standard library (`random`, `math` -- no numpy), driven through
`sezgi.EvalSession` so evaluation counting and budget enforcement come for
free. This script does NOT reproduce the Rust component's RNG-stream (draw
order) contract -- only the same mathematical update rule -- so results
will differ from the `firefly` preset even at the same seed; see
`examples/specs/fa.toml` for the pinned-RNG preset itself.

Tier note: a **labeled metaphor preset** -- faithful to the primary
source's own reference MATLAB implementation (`fa_ndim.m`/`ffa_move.m`),
with a pinned deterministic draw order, but NOT validated against the
paper's reported benchmark numbers. Equivalence critique:
Camacho-Villalón, Dorigo & Stützle (International Transactions in
Operational Research, six-algorithm critique: grey wolf, moth-flame,
whale, firefly, bat, antlion) -- cited here conservatively, as background
for why this is "labeled metaphor" rather than a mechanism sezgi treats as
novel.

Reproduces the source's hybrid live/frozen quirk VERBATIM: the pairwise
distance and a firefly's own multiplicative self-term read the LIVE,
in-place-mutating position array, while the additive move target reads a
FROZEN, start-of-generation snapshot -- so a firefly with multiple
brighter attractors accumulates its move in place across the inner loop.
The floored attractiveness formula (`beta` never decays below `betamin`)
and the closed-form `alpha` decay are the same "progress substitutes a
literal generation counter" idiom already used by sca.py/mfo.py/ssa.py.

Cost note: O(pop_size^2 * dim) per generation, since every (i, j) pair is
visited regardless of whether it fires -- markedly pricier per generation
than sca.py/jaya.py/mfo.py/ssa.py.

No cross-algorithm quality claims are made or implied here -- this is a
single seed on a single problem, reported as a gap, not a ranking.
"""
import math
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 25  # number of fireflies
SEED = 42
LO, HI = -5.0, 5.0
ALPHA0 = 0.5
BETA0 = 1.0
BETAMIN = 0.2
GAMMA = 1.0


def fa_beta(r2):
    """Floored attractiveness: beta -> betamin (not 0) as r2/gamma grow."""
    return (BETA0 - BETAMIN) * math.exp(-GAMMA * r2) + BETAMIN


def fa_alpha(progress):
    """Closed-form substitute for alpha_new's per-generation geometric decay."""
    return ALPHA0 * (1e-4 / 0.9) ** progress


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/fa-py", algo_name="fa-py", seed=SEED)

    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    used = POP_SIZE
    scale = [HI - LO for _ in range(DIM)]

    while used + POP_SIZE <= BUDGET:
        progress = min(max(used / BUDGET, 0.0), 1.0)
        alpha = fa_alpha(progress)

        # Rank order: stable sort by fitness ascending, ties -> original
        # index -- REQUIRED for the verified in-place loop semantics, not
        # merely cosmetic (mirrors ffa_mincon's own pre-sort).
        order = sorted(range(POP_SIZE), key=lambda i: (fitness[i], i))
        lighto = [fitness[i] for i in order]
        nso = [list(pop[i]) for i in order]  # FROZEN rank-ordered snapshot
        ns = [list(row) for row in nso]      # LIVE working copy, mutated in place

        for i in range(POP_SIZE):
            for j in range(POP_SIZE):
                # r2 from the LIVE working copy on BOTH sides -- verified
                # quirk: not the frozen nso snapshot.
                r2 = sum((ns[i][d] - ns[j][d]) ** 2 for d in range(DIM))
                if lighto[i] > lighto[j]:
                    beta = fa_beta(r2)
                    for d in range(DIM):
                        u = rng.random()
                        step = alpha * (u - 0.5) * scale[d]
                        # self LIVE (accumulates across repeated j hits),
                        # target FROZEN.
                        ns[i][d] = ns[i][d] * (1.0 - beta) + nso[j][d] * beta + step

        # boundary/clamp applied AFTER the full double loop (same timing
        # as the preset's separate boundary stage, not mid-loop).
        offspring = [[clamp(v, LO, HI) for v in row] for row in ns]

        # replace/generational: fa_ndim.m overwrites the whole population
        # unconditionally every generation.
        fitness = session.evaluate(offspring)
        pop = offspring
        used += POP_SIZE

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"fa: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
