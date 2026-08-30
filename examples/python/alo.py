"""Pure-Python Ant Lion Optimizer (Mirjalili, S. 2015, "The Ant Lion
Optimizer", Advances in Engineering Software 83, 80-98).

Teaches the SAME pinned update equations as sezgi's `gen/alo` +
`replace/mu-plus-lambda` Rust component (see crates/components/src/alo.rs's
module doc, verified against the author's own `ALO.m`/
`Random_walk_around_antlion.m`/`RouletteWheelSelection.m`) with nothing but
the standard library, driven through `sezgi.EvalSession`. Does NOT
reproduce the Rust component's RNG-stream (draw order) contract, only the
same update rule; see examples/specs/alo.toml for the pinned-RNG preset.

Tier note: a **labeled metaphor preset** -- faithful to ALO.m's own
equations and loop structure, pinned draw order, but NOT validated against
the paper's reported benchmark numbers. Equivalence critique:
Camacho-Villalon, Dorigo & Stutzle (International Transactions in
Operational Research, six-algorithm critique: grey wolf, moth-flame,
whale, firefly, bat, antlion) -- cited conservatively as background.

Each ant's move is `(RA + RE) / 2`: a FULL-HORIZON random walk (cumsum of
+-1 steps, `t_max` steps long, where `t_max = budget // pop_size`) built
fresh EVERY generation, min-max normalized into a SHRINKING `[c, d]` bound
around the walk's target antlion (an `I`-ratio ladder that shrinks the
bounds as the run progresses), read at the current row -- once around a
roulette-selected antlion (1/fitness weights, WITH a floor-shift when any
fitness is <= 0, since sezgi's raw fitness can be negative), once around
the elite (current-population fitness argmin). The antlion population
itself carries elitism: after the ants move, pop and offspring are merged,
sorted ascending by fitness, and truncated back to pop_size (`replace/mu-
plus-lambda` -- ALO.m's own `double_population=[Sorted_antlions;
ant_position]; sort; truncate(N)` step).

No cross-algorithm quality claims are made or implied here -- this is a
single seed on a single problem, reported as a gap, not a ranking.
"""
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 25  # ant/antlion count
SEED = 42
LO, HI = -5.0, 5.0


def alo_i_ratio(progress):
    """Eq. (2.10)/(2.11): five SEQUENTIAL unconditional overwrites (not
    elif) -- reproduced literally; since each threshold implies the lower
    ones, the last that fires wins, matching a descending-elseif chain."""
    i = 1.0
    if progress > 0.1:
        i = 1.0 + 100.0 * progress
    if progress > 0.5:
        i = 1.0 + 1000.0 * progress
    if progress > 0.75:
        i = 1.0 + 10_000.0 * progress
    if progress > 0.9:
        i = 1.0 + 100_000.0 * progress
    if progress > 0.95:
        i = 1.0 + 1_000_000.0 * progress
    return i


def alo_shift_bound(scaled, antlion_d, use_plus):
    return scaled + antlion_d if use_plus else -scaled + antlion_d


def alo_cumsum_walk(steps):
    """Eq. (2.1): X[0]=0, X[k]=X[k-1]+(+1 if step else -1)."""
    x = [0.0]
    acc = 0.0
    for s in steps:
        acc += 1.0 if s else -1.0
        x.append(acc)
    return x


def alo_normalize_walk(x, row, c, d):
    """Eq. (2.7): min-max normalize the WHOLE walk into [c,d], read at row."""
    a, b = min(x), max(x)
    return (x[row] - a) * (d - c) / (b - a) + c


def alo_roulette_weights(fitness):
    """1/fitness (minimization), with a floor-shift (f - min(f) + 1) applied
    ONLY when some fitness is <= 0 -- keeps the reciprocal well-defined and
    monotonic (sezgi's raw fitness can be negative; ALO.m's own literal
    1./fitness formula assumes it never is)."""
    min_f = min(fitness)
    if min_f > 0.0:
        return [1.0 / f for f in fitness]
    shift = -min_f + 1.0
    return [1.0 / (f + shift) for f in fitness]


def alo_roulette_select(weights, u):
    total = sum(weights)
    target = u * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if acc > target:
            return i
    return 0  # fallback, matching ALO.m's chosen_index==-1 -> index 1 (1-based)


def alo_walk_around(dim, t_max, row, lo, hi, i_ratio, antlion, rng):
    lb_scaled = lo / i_ratio
    ub_scaled = hi / i_ratio
    use_plus_lo = rng.random() < 0.5
    use_plus_hi = rng.random() >= 0.5
    out = []
    for d in range(dim):
        c = alo_shift_bound(lb_scaled, antlion[d], use_plus_lo)
        dd = alo_shift_bound(ub_scaled, antlion[d], use_plus_hi)
        steps = [rng.random() > 0.5 for _ in range(t_max)]
        x = alo_cumsum_walk(steps)
        out.append(alo_normalize_walk(x, row, c, dd))
    return out


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/alo-py", algo_name="alo-py", seed=SEED)

    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    used = POP_SIZE

    # t_max = ALO.m's Max_iter, derived once from (budget, pop_size) -- both
    # constant across the run, so this is identical every call.
    t_max = max(1, BUDGET // POP_SIZE)
    generation = 0  # ALO.m's Current_iter = generation + 2 (1-based, iteration 1 = setup)

    while used + POP_SIZE <= BUDGET:
        current_iter = generation + 2
        progress = current_iter / t_max
        i_ratio = alo_i_ratio(progress)
        row = min(generation + 1, t_max)

        elite_idx = min(range(POP_SIZE), key=lambda i: (fitness[i], i))
        elite = pop[elite_idx]
        weights = alo_roulette_weights(fitness)

        offspring = []
        for _ in range(POP_SIZE):
            u_sel = rng.random()
            sel_idx = alo_roulette_select(weights, u_sel)
            selected = pop[sel_idx]

            ra = alo_walk_around(DIM, t_max, row, LO, HI, i_ratio, selected, rng)
            re = alo_walk_around(DIM, t_max, row, LO, HI, i_ratio, elite, rng)
            xs = [clamp((ra[d] + re[d]) / 2.0, LO, HI) for d in range(DIM)]
            offspring.append(xs)

        new_fitness = session.evaluate(offspring)
        used += POP_SIZE

        # replace/mu-plus-lambda: pop first, offspring second, stable sort
        # ascending by fitness, truncate to POP_SIZE.
        combined = list(zip(fitness, pop)) + list(zip(new_fitness, offspring))
        combined.sort(key=lambda item: item[0])
        combined = combined[:POP_SIZE]
        fitness = [f for f, _ in combined]
        pop = [p for _, p in combined]

        generation += 1

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"alo: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
