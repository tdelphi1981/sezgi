"""Pure-Python Harris Hawks Optimization (Heidari, A.A., Mirjalili, S.,
Faris, H., Aljarah, I., Mafarja, M. & Chen, H. 2019, "Harris hawks
optimization: Algorithm and applications", Future Generation Computer
Systems 97, 849-872).

Teaches the SAME pinned update equations as sezgi's `gen/hho` +
`replace/generational` Rust component (see crates/components/src/hho.rs's
module doc, verified against the paper AUTHOR's own `HHO.m`) with nothing
but the standard library, driven through `sezgi.EvalSession`. Does NOT
reproduce the Rust component's RNG-stream (draw order) contract, only the
same update rule; see examples/specs/hho.toml for the pinned-RNG preset.

Tier note: a **labeled metaphor preset** -- faithful to `HHO.m`'s own
equations and loop structure, pinned draw order, but NOT validated against
the paper's reported benchmark numbers. No established equivalence
critique covers HHO -- primary-source only.

This is the wave's most structurally complex script: a per-hawk
escape-energy branch tree (explore vs. exploit, then a further split on
`r`), whose deepest branches (progressive rapid dives) EVALUATE mid-loop --
`Y` is tried first (accepted immediately if it improves, 1 eval), else `Z =
Y + S*Levy` is tried (2nd eval); neither improving leaves the hawk
UNCHANGED. Reproduces the CORRECTED soft-dive `Y` formula verbatim:
`Rabbit - E*|J*Rabbit - X_i|` (NOT `(Rabbit-X_i) - E*|J*Rabbit-X_i|`, a
genuinely different equation from the no-dive soft besiege). The rabbit
(attractor) uses the current-population fitness argmin (the wave's parked
simplification of `HHO.m`'s own persisted best-ever). `mean(X)` and the
random-hawk lookup both read the LIVE, in-place-updated working array (may
mix already-this-sweep-updated hawks with still-original ones).

sezgi simplification (script-specific, matching the Rust doc's own
budget-exhaustion note): if the FINAL per-generation batch evaluation would
overrun the budget (because dive trials already spent it), this script
simply stops at the generation boundary, rather than reproducing the
engine's own per-hawk graceful fallback -- an honest, disclosed departure
from the Rust engine's exact accounting at the very last generation only.

No cross-algorithm quality claims are made or implied here -- this is a
single seed on a single problem, reported as a gap, not a ranking.
"""
import math
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30  # number of hawks
SEED = 42
LO, HI = -5.0, 5.0
LEVY_ALPHA = 1.5


def gauss_polar(rng):
    while True:
        u = 2.0 * rng.random() - 1.0
        v = 2.0 * rng.random() - 1.0
        s = u * u + v * v
        if 0.0 < s < 1.0:
            return u * math.sqrt(-2.0 * math.log(s) / s)


def levy_mantegna(rng, alpha=LEVY_ALPHA):
    """HHO.m's own Levy(d) has NO 0.01 scale factor -- used directly."""
    num = math.gamma(1.0 + alpha) * math.sin(math.pi * alpha / 2.0)
    den = math.gamma((1.0 + alpha) / 2.0) * alpha * 2.0 ** ((alpha - 1.0) / 2.0)
    sigma_u = (num / den) ** (1.0 / alpha)
    u = sigma_u * gauss_polar(rng)
    v = abs(gauss_polar(rng))
    return u / (v ** (1.0 / alpha))


def hho_e1(progress):
    return 2.0 * (1.0 - progress)


def hho_explore_family_dim_step(x_rand_d, r_a, r_b, x_i_d):
    return x_rand_d - r_a * abs(x_rand_d - 2.0 * r_b * x_i_d)


def hho_explore_tree_dim_step(rabbit_d, mean_d, r_a, r_b, lo, hi):
    return (rabbit_d - mean_d) - r_a * ((hi - lo) * r_b + lo)


def hho_hard_besiege_dim_step(rabbit_d, e, x_i_d):
    return rabbit_d - e * abs(rabbit_d - x_i_d)


def hho_soft_besiege_dim_step(rabbit_d, e, jump, x_i_d):
    return (rabbit_d - x_i_d) - e * abs(jump * rabbit_d - x_i_d)


def hho_soft_dive_y_dim_step(rabbit_d, e, jump, x_i_d):
    """CORRECTED formula (HHO.m line 105's X1) -- no leading (rabbit-x_i)
    term, unlike hho_soft_besiege_dim_step's no-dive equation."""
    return rabbit_d - e * abs(jump * rabbit_d - x_i_d)


def hho_hard_dive_y_dim_step(rabbit_d, e, jump, mean_d):
    return rabbit_d - e * abs(jump * rabbit_d - mean_d)


def hho_dive_z_dim_step(y_d, s_d, levy_d):
    return y_d + s_d * levy_d


def hho_mean(work):
    n = len(work)
    dim = len(work[0])
    return [sum(row[d] for row in work) / n for d in range(dim)]


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def hho_run_dive(session, state, y, dim, current_fitness, rng):
    """Evaluate Y; accept immediately if it improves. Else Z=Y+S*Levy
    (interleaved per-dim draws); accept if IT improves. Neither -> None
    (caller keeps the un-dived candidate). Budget exhaustion at any point
    also falls back to None, spending nothing further."""
    try:
        fy = session.evaluate([y])[0]
        state["used"] += 1
    except ValueError:
        return None
    if fy < current_fitness:
        return y

    z = []
    for d in range(dim):
        s_d = rng.random()
        levy_d = levy_mantegna(rng)
        z.append(hho_dive_z_dim_step(y[d], s_d, levy_d))
    try:
        fz = session.evaluate([z])[0]
        state["used"] += 1
    except ValueError:
        return None
    return z if fz < current_fitness else None


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/hho-py", algo_name="hho-py", seed=SEED)

    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    state = {"used": POP_SIZE}

    while state["used"] + POP_SIZE <= BUDGET:
        progress = min(max(state["used"] / BUDGET, 0.0), 1.0)
        e1 = hho_e1(progress)

        # Rabbit: current-population fitness argmin -- pinned simplification
        # of HHO.m's own persisted best-ever, computed once per generation,
        # before any draws (shared wave-wide convention).
        rabbit_idx = min(range(POP_SIZE), key=lambda i: (fitness[i], i))
        rabbit = pop[rabbit_idx]

        # Live, in-place working copy: later hawks may observe earlier
        # hawks' already-this-generation-updated positions.
        work = [list(x) for x in pop]

        for i in range(POP_SIZE):
            e0 = 2.0 * rng.random() - 1.0
            e = e1 * e0

            if abs(e) >= 1.0:
                # Exploration
                q = rng.random()
                rand_idx = rng.randrange(POP_SIZE)  # always drawn, even if unused
                if q < 0.5:
                    x_rand = work[rand_idx]
                    r_a, r_b = rng.random(), rng.random()
                    new = [hho_explore_family_dim_step(x_rand[d], r_a, r_b, work[i][d]) for d in range(DIM)]
                else:
                    mean = hho_mean(work)
                    r_a, r_b = rng.random(), rng.random()
                    new = [hho_explore_tree_dim_step(rabbit[d], mean[d], r_a, r_b, LO, HI) for d in range(DIM)]
            else:
                # Exploitation
                r = rng.random()
                if r >= 0.5 and abs(e) < 0.5:
                    new = [hho_hard_besiege_dim_step(rabbit[d], e, work[i][d]) for d in range(DIM)]
                elif r >= 0.5:
                    jump = 2.0 * (1.0 - rng.random())
                    new = [hho_soft_besiege_dim_step(rabbit[d], e, jump, work[i][d]) for d in range(DIM)]
                elif abs(e) >= 0.5:
                    jump = 2.0 * (1.0 - rng.random())
                    y = [hho_soft_dive_y_dim_step(rabbit[d], e, jump, work[i][d]) for d in range(DIM)]
                    dived = hho_run_dive(session, state, y, DIM, fitness[i], rng)
                    new = dived if dived is not None else work[i]
                else:
                    jump = 2.0 * (1.0 - rng.random())
                    mean = hho_mean(work)
                    y = [hho_hard_dive_y_dim_step(rabbit[d], e, jump, mean[d]) for d in range(DIM)]
                    dived = hho_run_dive(session, state, y, DIM, fitness[i], rng)
                    new = dived if dived is not None else work[i]
            work[i] = new

        offspring = [[clamp(v, LO, HI) for v in row] for row in work]
        try:
            new_fitness = session.evaluate(offspring)
        except ValueError:
            break  # dive trials consumed the remaining budget; stop cleanly
        state["used"] += POP_SIZE
        pop, fitness = offspring, new_fitness  # replace/generational: unconditional overwrite

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"hho: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
