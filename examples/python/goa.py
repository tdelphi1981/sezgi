"""Pure-Python Grasshopper Optimisation Algorithm (Saremi, Mirjalili & Lewis
2017, "Grasshopper optimisation algorithm: theory and application",
Advances in Engineering Software, 105, 30-47).

Teaches the SAME pinned update equations as sezgi's `gen/goa` Rust component
(see crates/components/src/goa.rs's module doc), stdlib only. GOA's core
update draws NOTHING from RNG (fully deterministic given the population and
`c`); `random` here seeds only the initial population. Not RNG-stream-
identical to the Rust preset; see `examples/specs/goa.toml` for that.

Per the Rust module's IMPLEMENTER-VERIFY note, this follows the VERIFIED
reference-implementation semantics (confirmed against the paper's own
MATLAB File Exchange #61421 code, via `thieu1995/mealpy`'s `OriginalGOA`)
rather than the paper's literal prose: pairwise distance is mapped into the
bounded [2, 4) interval before applying the social force `s(.)`. No
equivalence-critique citation -- none is cited by the Rust module's doc.

No cross-algorithm quality claims here -- single seed, single problem,
reported as a gap.
"""
import math
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30
SEED = 42
LO, HI = -5.0, 5.0
GOA_F = 0.5
GOA_L = 1.5
GOA_C_MAX = 1.0
GOA_C_MIN = 1e-5
EPS = 1e-12


def goa_s(r):
    """Social force s(r) = f*e^(-r/l) - e^(-r) (Eq. 2.3)."""
    return GOA_F * math.exp(-r / GOA_L) - math.exp(-r)


def goa_pair_term(c, half_range_d, x_i_d, x_j_d, d_ij):
    # sezgi simplification: maps the raw Euclidean distance into [2, 4)
    # before applying s(.) -- the verified reference-implementation
    # semantics, see this file's module doc.
    dist_term = 2.0 + (d_ij % 2.0)
    s = goa_s(dist_term)
    # sezgi simplification: + EPS guards a zero-distance pair.
    return c * half_range_d * s * (x_j_d - x_i_d) / (d_ij + EPS)


def euclidean_dist(a, b):
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/goa-py", algo_name="goa-py", seed=SEED)

    half_range = (HI - LO) / 2.0
    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    used = POP_SIZE

    while used + POP_SIZE <= BUDGET:
        progress = min(max(used / BUDGET, 0.0), 1.0)
        c = GOA_C_MAX - progress * (GOA_C_MAX - GOA_C_MIN)
        best = min(range(POP_SIZE), key=lambda i: (fitness[i], i))
        x_best = pop[best]

        dists = [[euclidean_dist(pop[i], pop[j]) for j in range(POP_SIZE)]
                  for i in range(POP_SIZE)]

        offspring = []
        for i in range(POP_SIZE):
            x_i = pop[i]
            new_x = []
            for d in range(DIM):
                total = 0.0
                for j in range(POP_SIZE):
                    if j == i:
                        continue
                    total += goa_pair_term(c, half_range, x_i[d], pop[j][d], dists[i][j])
                new_x.append(clamp(c * total + x_best[d], LO, HI))
            offspring.append(new_x)

        # Unconditional generational replacement (replace/generational),
        # whole-batch-or-nothing per iteration (mirrors the Rust engine's
        # own budget-exhaustion clean break).
        fitness = session.evaluate(offspring)
        pop = offspring
        used += POP_SIZE

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"goa: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
