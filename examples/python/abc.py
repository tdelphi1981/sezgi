"""Pure-Python Artificial Bee Colony (Karaboga, D. 2005, "An Idea Based On
Honey Bee Swarm For Numerical Optimization", TR-06, Erciyes University; and
Karaboga, D., Basturk, B. 2007, "A Powerful and Efficient Algorithm for
Numerical Function Optimization: ABC Algorithm", Journal of Global
Optimization 39(3), 459-471).

Teaches the SAME pinned update equations as sezgi's `gen/abc-employed` +
`replace/abc-trial-greedy` + `adapter/abc-onlooker-scout` Rust components
(see crates/components/src/abc.rs's module doc, verified against
Karaboga & Basturk's OWN `ABCorig.m`) with nothing but the standard
library, driven through `sezgi.EvalSession`. Does NOT reproduce the Rust
components' RNG-stream (draw order) contract, only the same update rule;
see examples/specs/abc.toml for the pinned-RNG preset.

Tier note: a **labeled metaphor preset** -- faithful to the verified
primary artifacts' own update equations and loop structure, pinned draw
order, but NOT validated against any publication's reported benchmark
numbers. No established equivalence critique covers ABC -- primary
sources only.

sezgi's `pop_size` IS the food-source count `SN` directly (NOT Karaboga's
colony size `NP=2*SN`) -- there is only ever ONE array of solutions;
onlookers merely SELECT one of the SAME SN sources to perturb. Models the
THREE stages explicitly, one cycle at a time, matching `ABCorig.m`'s own
`employed -> probabilities -> onlooker -> scout` order term for term:

1. **Employed** -- exactly SN candidates (one dimension changed per
   source, `x_ij + phi*(x_ij - x_kj)`, `k` distinct from `i`), evaluated as
   ONE frozen batch, then a per-source greedy accept coupled with a
   `trial` counter (reset to 0 on accept, +1 on reject).
2. **Onlooker** -- a repeated LINEAR SCAN (not literal roulette-wheel
   sampling): `i=1; t=0; while t<SN: if rand<prob(i): t+=1; [[move]];
   i=(i+1) mod SN`. `prob` is computed ONCE (`0.9*transformed/max+0.1`,
   `transformed` via `calculateFitness`'s `f>=0 -> 1/(f+1)`, `f<0 -> 1+|f|`)
   and never recomputed mid-scan, even though positions/fitness DO mutate
   in place as visits succeed -- exactly SN accepted visits, each its own
   evaluation.
3. **Scout** -- at most ONE per cycle: the source with `max(trial)` (ties
   -> LAST index, `ABCorig.m`'s own `ind(end)`), re-randomized if its
   trial count exceeds `limit = SN*dim` (the dimension-scaling default from
   Karaboga & Akay 2009, since `ABCorig.m`'s own demo hardcodes an
   unrelated constant).

No cross-algorithm quality claims are made or implied here -- this is a
single seed on a single problem, reported as a gap, not a ranking.
"""
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 20  # food-source count SN (== sezgi's pop_size, NOT NP=2*SN)
SEED = 42
LO, HI = -5.0, 5.0
LIMIT = POP_SIZE * DIM  # abc_limit: SN*dim (Karaboga & Akay 2009's rule of thumb)


def abc_dim_step(x_i_j, x_k_j, phi):
    return x_i_j + phi * (x_i_j - x_k_j)


def abc_candidate(x_i, x_k, j, phi):
    c = list(x_i)
    c[j] = abc_dim_step(x_i[j], x_k[j], phi)
    return c


def abc_fitness_transform(f):
    return 1.0 / (f + 1.0) if f >= 0.0 else 1.0 + abs(f)


def abc_probabilities(transformed):
    max_fit = max(transformed)
    return [0.9 * (f / max_fit) + 0.1 for f in transformed]


def abc_scout_target(trials):
    """argmax, ties -> LAST index (ABCorig.m's ind(end), the opposite
    tie-break from this project's usual first-seen-wins convention)."""
    best = 0
    for k in range(1, len(trials)):
        if trials[k] >= trials[best]:
            best = k
    return best


def abc_draw_move(sn, dim, i, rng):
    j = rng.randrange(dim)
    while True:
        k = rng.randrange(sn)
        if k != i:
            break
    phi = (rng.random() - 0.5) * 2.0
    return j, k, phi


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/abc-py", algo_name="abc-py", seed=SEED)

    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    used = POP_SIZE
    trials = [0] * POP_SIZE

    while used + POP_SIZE <= BUDGET:
        # ---- Employed phase: frozen batch, one candidate per source ----
        candidates = []
        for i in range(POP_SIZE):
            j, k, phi = abc_draw_move(POP_SIZE, DIM, i, rng)
            candidates.append([clamp(v, LO, HI) for v in abc_candidate(pop[i], pop[k], j, phi)])
        new_fitness = session.evaluate(candidates)
        used += POP_SIZE
        for i in range(POP_SIZE):
            if new_fitness[i] < fitness[i]:
                pop[i], fitness[i], trials[i] = candidates[i], new_fitness[i], 0
            else:
                trials[i] += 1

        # ---- Onlooker phase: repeated linear scan, exactly SN accepted visits ----
        transformed = [abc_fitness_transform(f) for f in fitness]
        prob = abc_probabilities(transformed)
        i, t, budget_exhausted = 0, 0, False
        while t < POP_SIZE and not budget_exhausted:
            if rng.random() < prob[i]:
                t += 1
                j, k, phi = abc_draw_move(POP_SIZE, DIM, i, rng)
                candidate = abc_candidate(pop[i], pop[k], j, phi)
                candidate[j] = clamp(candidate[j], LO, HI)
                try:
                    f = session.evaluate([candidate])[0]
                    used += 1
                except ValueError:
                    budget_exhausted = True
                else:
                    if f < fitness[i]:
                        pop[i], fitness[i], trials[i] = candidate, f, 0
                    else:
                        trials[i] += 1
            i = (i + 1) % POP_SIZE

        # ---- Scout phase: at most one re-randomized source per cycle ----
        if not budget_exhausted:
            scout_idx = abc_scout_target(trials)
            if trials[scout_idx] > LIMIT:
                new_pos = [rng.uniform(LO, HI) for _ in range(DIM)]
                try:
                    f = session.evaluate([new_pos])[0]
                    used += 1
                except ValueError:
                    pass
                else:
                    pop[scout_idx], fitness[scout_idx], trials[scout_idx] = new_pos, f, 0

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"abc: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
