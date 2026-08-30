"""Pure-Python Flower Pollination Algorithm (Yang, X.-S. 2012, "Flower
Pollination Algorithm for Global Optimization", in: Unconventional
Computation and Natural Computation (UCNC 2012), Lecture Notes in Computer
Science vol. 7445, Springer, pp. 240-249).

Teaches the SAME pinned update equations as sezgi's `gen/fpa` +
`replace/one-to-one-greedy` Rust component (see
crates/components/src/fpa.rs's module doc) with nothing but the standard
library (`random`, `math` -- no numpy), driven through `sezgi.EvalSession`
so evaluation counting and budget enforcement come for free. This script
does NOT reproduce the Rust component's RNG-stream (draw order) contract --
only the same mathematical update rule -- so results will differ from the
`fpa` preset even at the same seed; see `examples/specs/fpa.toml` for the
pinned-RNG preset itself.

Tier note: a **labeled metaphor preset** -- faithful to the primary
source's own reference MATLAB demo (`fpa_demo.m`), with a pinned
deterministic draw order, but NOT validated against the paper's reported
benchmark numbers. No established equivalence critique covers FPA -- cited
here as primary-source only, per the task brief.

Reproduces two VERIFIED source deltas verbatim: the branch switch is
`u > p` selecting the GLOBAL (Levy) branch (NOT `u < p` -- with the demo's
own default `p=0.8` global fires only ~20% of the time, matching the
algorithm's "slight bias towards local pollination" prose), and the local
branch's `j`/`k` indices are drawn distinct FROM EACH OTHER ONLY, never
excluded from equaling `i` (the same self-selection-not-excluded idiom
`woa.rs` established) -- so a population of exactly 2 already suffices.
The global step's L*(X_i - X_best) is algebraically identical to `cs.py`'s
own Levy step (same Mantegna sampler, same 0.01 scale), reused here as
`cs_dim_step` rather than a separately-derived formula.

No cross-algorithm quality claims are made or implied here -- this is a
single seed on a single problem, reported as a gap, not a ranking.
"""
import math
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 25  # flower count
SEED = 42
LO, HI = -5.0, 5.0
P_SWITCH = 0.8
ALPHA_STEP = 0.01
LEVY_ALPHA = 1.5


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
    """Reused verbatim from cs.py -- FPA's global-pollination step is
    algebraically identical to CS's Levy-toward-best step (finding 2)."""
    return x_i_d + alpha_step * levy * (x_i_d - x_best_d)


def fpa_is_global_branch(u, p):
    """VERIFIED orientation: u > p selects GLOBAL, not u < p."""
    return u > p


def fpa_local_dim_step(x_d, x_j_d, x_k_d, epsilon):
    return x_d + epsilon * (x_j_d - x_k_d)


def pick_two_distinct(rng, n):
    """j, k distinct FROM EACH OTHER only -- self (i) is never excluded,
    matching fpa_demo.m's `JK=randperm(n)` first-two-entries behavior."""
    j = rng.randrange(n)
    while True:
        k = rng.randrange(n)
        if k != j:
            return j, k


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/fpa-py", algo_name="fpa-py", seed=SEED)

    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    used = POP_SIZE

    while used + POP_SIZE <= BUDGET:
        # Attractor: current-population fitness argmin (ties -> lower
        # index), computed once per generation before any draws -- a
        # documented simplification of the source's persisted global-best
        # (shared with ba.py/ssa.py, not FPA-specific).
        best = min(range(POP_SIZE), key=lambda i: (fitness[i], i))
        x_best = pop[best]

        offspring = []
        for i in range(POP_SIZE):
            x = pop[i]
            u = rng.random()
            if fpa_is_global_branch(u, P_SWITCH):
                new_x = [clamp(cs_dim_step(x[d], x_best[d], levy_mantegna(rng)), LO, HI) for d in range(DIM)]
            else:
                epsilon = rng.random()  # scalar, shared across every dimension
                j, k = pick_two_distinct(rng, POP_SIZE)
                x_j, x_k = pop[j], pop[k]
                new_x = [clamp(fpa_local_dim_step(x[d], x_j[d], x_k[d], epsilon), LO, HI) for d in range(DIM)]
            offspring.append(new_x)

        new_fitness = session.evaluate(offspring)
        used += POP_SIZE

        # replace/one-to-one-greedy
        for i in range(POP_SIZE):
            if new_fitness[i] < fitness[i]:
                pop[i], fitness[i] = offspring[i], new_fitness[i]

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"fpa: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
