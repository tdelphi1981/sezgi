"""OOP twin of examples/python/hho.py -- same math, same RNG draw order.

Port of the pure-Python Harris Hawks Optimization script onto
sezgi.AskTellAlgorithm. The pure script (and the Rust module doc it cites) remains
the provenance for the update equations and the escape-energy branch tree;
this file re-derives NOTHING and must reproduce the pure script's
evals_used/best_f/gap output bit-for-bit at the same seed -- enforced by
py-sezgi/tests/test_examples_oop_parity.py.

The deepest branches (progressive rapid dives) evaluate MID-LOOP: Y is
tried first (accepted immediately if it improves, 1 eval), else
Z = Y + S*Levy is tried (2nd eval); neither improving leaves the hawk
UNCHANGED -- ctx.evaluate is called wherever the pure script's
session.evaluate is, including inside the per-hawk branch tree.
"""
import math

import sezgi
from sezgi.algo import BudgetExhausted

BUDGET = 2000
SEED = 42


def gauss_polar(rng):
    while True:
        u = 2.0 * rng.random() - 1.0
        v = 2.0 * rng.random() - 1.0
        s = u * u + v * v
        if 0.0 < s < 1.0:
            return u * math.sqrt(-2.0 * math.log(s) / s)


def levy_mantegna(rng, alpha=1.5):
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


def hho_run_dive(ctx, y, dim, current_fitness, levy_alpha):
    """Evaluate Y; accept immediately if it improves. Else Z=Y+S*Levy
    (interleaved per-dim draws); accept if IT improves. Neither -> None
    (caller keeps the un-dived candidate). Budget exhaustion at any point
    also falls back to None, spending nothing further."""
    try:
        fy = ctx.evaluate([y])[0]
    except BudgetExhausted:
        return None
    if fy < current_fitness:
        return y

    z = []
    for d in range(dim):
        s_d = ctx.rng.random()
        levy_d = levy_mantegna(ctx.rng, levy_alpha)
        z.append(hho_dive_z_dim_step(y[d], s_d, levy_d))
    try:
        fz = ctx.evaluate([z])[0]
    except BudgetExhausted:
        return None
    return z if fz < current_fitness else None


class Hho(sezgi.AskTellAlgorithm):
    """Harris Hawks Optimization (Heidari, Mirjalili, Faris, Aljarah,
    Mafarja & Chen 2019, Future Generation Computer Systems 97, 849-872),
    modeling the escape-energy-gated exploration/exploitation branch tree
    (including the progressive rapid dives) explicitly over
    `sezgi.AskTellAlgorithm`'s ask/tell loop. `dim`: problem
    dimensionality. `pop_size`: number of hawks. `levy_alpha`: the
    Mantegna (1994) Levy-stability exponent used by the dive branches'
    Levy-flight step."""

    name = "hho"

    def __init__(self, dim=5, pop_size=30, levy_alpha=1.5):
        self.dim = dim
        self.pop_size = pop_size
        self.levy_alpha = levy_alpha

    def setup(self, ctx):
        self.pop = [ctx.random_point() for _ in range(self.pop_size)]
        self.fitness = ctx.evaluate(self.pop)

    def step(self, ctx):
        # sezgi decision: the pure script's outer loop is guarded up front
        # by `while used + POP_SIZE <= BUDGET` -- it NEVER starts a
        # generation (never draws e0/e for hawk 0, never spends a mid-loop
        # dive eval) unless a full POP_SIZE final batch could in principle
        # still fit before any of THIS generation's dive spending. The
        # driver here only checks `ctx.remaining > 0` before calling step(),
        # which is weaker: without this explicit guard, a generation could
        # start with fewer than POP_SIZE remaining, drawing RNG and
        # possibly spending 0/1/2 extra evals per hawk on dives the pure
        # script's own outer guard would never have allowed it to attempt --
        # a different evals_used stopping point. Raise BudgetExhausted
        # ourselves, before any draws, whenever the guaranteed POP_SIZE
        # final batch doesn't fit -- an honest structural port of the pure
        # script's own up-front guard. (The pure script's OWN mid-generation
        # case -- dives spend enough that the FINAL batch then fails even
        # though the generation was allowed to start -- needs no extra
        # code: ctx.evaluate's natural BudgetExhausted on that last call
        # reproduces the pure script's `except ValueError: break` exactly,
        # since the exception simply propagates out of step() to the
        # driver.)
        if ctx.remaining < self.pop_size:
            raise BudgetExhausted(
                f"generation needs {self.pop_size} evals, {ctx.remaining} remaining")

        lo, hi = ctx.bounds
        progress = min(max(ctx.evals_used / ctx.budget, 0.0), 1.0)
        e1 = hho_e1(progress)

        # Rabbit: current-population fitness argmin -- pinned simplification
        # of HHO.m's own persisted best-ever, computed once per generation,
        # before any draws (shared wave-wide convention).
        rabbit_idx = min(range(self.pop_size), key=lambda i: (self.fitness[i], i))
        rabbit = self.pop[rabbit_idx]

        # Live, in-place working copy: later hawks may observe earlier
        # hawks' already-this-generation-updated positions.
        work = [list(x) for x in self.pop]

        for i in range(self.pop_size):
            e0 = 2.0 * ctx.rng.random() - 1.0
            e = e1 * e0

            if abs(e) >= 1.0:
                # Exploration
                q = ctx.rng.random()
                rand_idx = ctx.rng.randrange(self.pop_size)  # always drawn, even if unused
                if q < 0.5:
                    x_rand = work[rand_idx]
                    r_a, r_b = ctx.rng.random(), ctx.rng.random()
                    new = [hho_explore_family_dim_step(x_rand[d], r_a, r_b, work[i][d]) for d in range(self.dim)]
                else:
                    mean = hho_mean(work)
                    r_a, r_b = ctx.rng.random(), ctx.rng.random()
                    new = [hho_explore_tree_dim_step(rabbit[d], mean[d], r_a, r_b, lo, hi) for d in range(self.dim)]
            else:
                # Exploitation
                r = ctx.rng.random()
                if r >= 0.5 and abs(e) < 0.5:
                    new = [hho_hard_besiege_dim_step(rabbit[d], e, work[i][d]) for d in range(self.dim)]
                elif r >= 0.5:
                    jump = 2.0 * (1.0 - ctx.rng.random())
                    new = [hho_soft_besiege_dim_step(rabbit[d], e, jump, work[i][d]) for d in range(self.dim)]
                elif abs(e) >= 0.5:
                    jump = 2.0 * (1.0 - ctx.rng.random())
                    y = [hho_soft_dive_y_dim_step(rabbit[d], e, jump, work[i][d]) for d in range(self.dim)]
                    dived = hho_run_dive(ctx, y, self.dim, self.fitness[i], self.levy_alpha)
                    new = dived if dived is not None else work[i]
                else:
                    jump = 2.0 * (1.0 - ctx.rng.random())
                    mean = hho_mean(work)
                    y = [hho_hard_dive_y_dim_step(rabbit[d], e, jump, mean[d]) for d in range(self.dim)]
                    dived = hho_run_dive(ctx, y, self.dim, self.fitness[i], self.levy_alpha)
                    new = dived if dived is not None else work[i]
            work[i] = new

        offspring = [[clamp(v, lo, hi) for v in row] for row in work]
        new_fitness = ctx.evaluate(offspring)
        self.pop, self.fitness = offspring, new_fitness  # replace/generational: unconditional overwrite


def main():
    algo = Hho()
    res = algo.solve(sezgi.bbob(1, algo.dim, 1), budget=BUDGET, seed=SEED)
    print(f"hho (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
