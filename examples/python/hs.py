"""Pure-Python Harmony Search (Geem, Kim & Loganathan 2001, "A new
heuristic optimization algorithm: harmony search", Simulation).

Teaches the SAME pinned update equations as sezgi's `gen/hs` Rust component
(see crates/components/src/hs.rs's module doc) with nothing but the standard
library (`random`, `math` -- no numpy), driven through `sezgi.EvalSession`.
Does NOT reproduce the Rust component's RNG-stream (draw order) contract,
only the same update rule; see `examples/specs/hs.toml` for the pinned-RNG
preset.

Equivalence critique: Weyland (2010), whose analysis shows HS's harmony-
memory-consideration/pitch-adjustment mechanism is, component for
component, a special case of evolution strategies.

No cross-algorithm quality claims are made or implied here -- single seed,
single problem, reported as a gap.
"""
import random

import sezgi

DIM = 5
BUDGET = 2000
HMS = 30  # harmony memory size (sezgi's "pop_size" for this preset)
SEED = 42
LO, HI = -5.0, 5.0
HMCR = 0.9
PAR = 0.3
BW_FRACTION = 0.01


def hs_dim_step(hmcr, par, bw_d, lo_d, hi_d, memory_d, rng):
    u1 = rng.random()
    if u1 < hmcr:
        j = rng.randrange(len(memory_d))
        value = memory_d[j]
        u2 = rng.random()
        if u2 < par:
            u3 = rng.random()
            value += bw_d * (2.0 * u3 - 1.0)
        return value
    u4 = rng.random()
    return lo_d + u4 * (hi_d - lo_d)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/hs-py", algo_name="hs-py", seed=SEED)

    bw = [BW_FRACTION * (HI - LO)] * DIM
    memory = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(HMS)]
    fitness = session.evaluate(memory)
    used = HMS

    # Exactly ONE new harmony per iteration (HMS need not match the offspring
    # count -- see hs.rs's module doc), so unlike the whole-population
    # generators (gwo/woa/goa) a single-eval iteration always fits until the
    # budget is fully exhausted.
    while used < BUDGET:
        new_harmony = [
            hs_dim_step(HMCR, PAR, bw[d], LO, HI, [row[d] for row in memory], rng)
            for d in range(DIM)
        ]
        new_harmony = [clamp(v, LO, HI) for v in new_harmony]
        new_f = session.evaluate([new_harmony])[0]
        used += 1

        # replace/worst-if-better: replace the current worst iff new_f beats it.
        worst = max(range(HMS), key=lambda k: fitness[k])
        if new_f < fitness[worst]:
            memory[worst] = new_harmony
            fitness[worst] = new_f

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"hs: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
