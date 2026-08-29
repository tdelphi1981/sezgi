"""Pure-Python Cuckoo Search (Yang, X.-S. & Deb, S. 2009, "Cuckoo Search via
Levy Flights", 2009 World Congress on Nature & Biologically Inspired
Computing (NaBIC), pp. 210-214, IEEE).

Teaches the SAME pinned update equations as sezgi's `gen/cuckoo_levy` +
`adapter/abandon-worst-fraction` (see crates/components/src/cs.rs's module
doc), stdlib only (`math.gamma` replaces the Rust component's Lanczos gamma
-- same closed-form Mantegna (1994) algorithm), via `sezgi.EvalSession`. Not
RNG-stream-identical to the Rust preset -- same equations only; see
`examples/specs/cs.toml` for the pinned-RNG preset. No equivalence-critique
citation -- not mandated for CS by the task brief.

sezgi simplification (shared with the Rust preset): the paper's Algorithm 1
compares a new egg against a RANDOM nest; this uses greedy same-index
replacement instead, matching `presets::cuckoo_search`'s divergence.

No cross-algorithm quality claims here -- single seed, single problem,
reported as a gap.
"""
import math
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 25  # nest count
SEED = 42
LO, HI = -5.0, 5.0
ALPHA_STEP = 0.01
LEVY_ALPHA = 1.5
PA = 0.25  # abandoned-fraction


def gauss_polar(rng):
    """Polar Box-Muller (Marsaglia)."""
    while True:
        u = 2.0 * rng.random() - 1.0
        v = 2.0 * rng.random() - 1.0
        s = u * u + v * v
        if 0.0 < s < 1.0:
            return u * math.sqrt(-2.0 * math.log(s) / s)


def levy_mantegna(rng, alpha=LEVY_ALPHA):
    num = math.gamma(1.0 + alpha) * math.sin(math.pi * alpha / 2.0)
    den = math.gamma((1.0 + alpha) / 2.0) * alpha * 2.0 ** ((alpha - 1.0) / 2.0)
    sigma_u = (num / den) ** (1.0 / alpha)
    u = sigma_u * gauss_polar(rng)
    v = abs(gauss_polar(rng))
    return u / (v ** (1.0 / alpha))


def cs_dim_step(x_i_d, x_best_d, levy, alpha_step=ALPHA_STEP):
    return x_i_d + alpha_step * levy * (x_i_d - x_best_d)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def abandon_order(fitness, pa):
    """Worst-first indices of the worst floor(pa*n) nests; ties -> higher
    index abandoned first."""
    n = len(fitness)
    k = int(pa * n)
    order = sorted(range(n), key=lambda i: (-fitness[i], -i))
    return order[:k]


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/cs-py", algo_name="cs-py", seed=SEED)

    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    used = POP_SIZE

    while used + POP_SIZE <= BUDGET:
        best = min(range(POP_SIZE), key=lambda i: (fitness[i], i))
        x_best = pop[best]

        offspring = [
            [clamp(cs_dim_step(pop[i][d], x_best[d], levy_mantegna(rng)), LO, HI)
             for d in range(DIM)]
            for i in range(POP_SIZE)
        ]
        new_fitness = session.evaluate(offspring)
        used += POP_SIZE

        # replace/one-to-one-greedy
        for i in range(POP_SIZE):
            if new_fitness[i] < fitness[i]:
                pop[i], fitness[i] = offspring[i], new_fitness[i]

        # adapter/abandon-worst-fraction: only if the WHOLE abandon batch
        # fits (mirrors the Rust adapter silently skipping on budget
        # exhaustion, via `if let Ok(new_fitness) = ctx.eval.evaluate(..)`).
        order = abandon_order(fitness, PA)
        if order and used + len(order) <= BUDGET:
            new_nests = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in order]
            new_fit2 = session.evaluate(new_nests)
            used += len(order)
            for idx, gi, fi in zip(order, new_nests, new_fit2):
                pop[idx], fitness[idx] = gi, fi

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"cs: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
