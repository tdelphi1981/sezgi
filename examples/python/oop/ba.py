"""OOP twin of examples/python/ba.py -- same math, same RNG draw order.

Port of the pure-Python Bat Algorithm script onto sezgi.AskTellAlgorithm. The pure
script (and the Rust module doc it cites) remains the provenance for the
update equations and the two verified sign-inversion quirks (inverted
[-2, 0] frequency range, (X - X_best) velocity term); this file re-derives
NOTHING and must reproduce the pure script's evals_used/best_f/gap output
bit-for-bit at the same seed -- enforced by
py-sezgi/tests/test_examples_oop_parity.py.
"""
import sezgi

BUDGET = 2000
SEED = 42


def ba_frequency(u, qmin, qmax):
    """VERIFIED VERBATIM against bat_algorithm.m: Qmin + (Qmin-Qmax)*u --
    note the sign, giving [-2, 0] for the pinned defaults, not [0, 2]."""
    return qmin + (qmin - qmax) * u


def ba_velocity_step(v_d, x_d, best_d, freq):
    return v_d + (x_d - best_d) * freq


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Ba(sezgi.AskTellAlgorithm):
    """Bat Algorithm (Yang 2010, "A new metaheuristic bat-inspired
    algorithm", NICSO 2010, Studies in Computational Intelligence vol.
    284, Springer, pp. 65-74), modeling frequency-tuned velocity plus a
    loudness-gated local-walk step over `sezgi.AskTellAlgorithm`'s
    ask/tell loop. `dim`: problem dimensionality. `pop_size`: number of
    bats. `qmin`/`qmax`: frequency-draw range (note the sign inversion
    verified against `bat_algorithm.m`, giving `[-2, 0]` for the pinned
    defaults). `r0`: fixed pulse rate. `a0`: fixed loudness. `local_walk_scale`:
    step size of the loudness-triggered local-walk perturbation."""

    name = "ba"

    def __init__(self, dim=5, pop_size=30, qmin=0.0, qmax=2.0, r0=0.5,
                 a0=0.5, local_walk_scale=0.001):
        self.dim = dim
        self.pop_size = pop_size
        self.qmin = qmin
        self.qmax = qmax
        self.r0 = r0
        self.a0 = a0
        self.local_walk_scale = local_walk_scale

    def setup(self, ctx):
        self.pop = [ctx.random_point() for _ in range(self.pop_size)]
        self.fitness = ctx.evaluate(self.pop)
        self.velocity = [[0.0] * self.dim for _ in range(self.pop_size)]

    def step(self, ctx):
        lo, hi = ctx.bounds

        # best: current-population fitness argmin, ties -> lower index,
        # computed once before any draws (sezgi simplification, shared
        # with the Rust preset: bat_algorithm.m persists a best-ever).
        best_idx = min(range(self.pop_size), key=lambda i: (self.fitness[i], i))
        best = self.pop[best_idx]

        offspring = []
        for i in range(self.pop_size):
            x = self.pop[i]
            # ONE frequency draw per bat (not per dimension).
            freq = ba_frequency(ctx.rng.random(), self.qmin, self.qmax)
            v_i = [ba_velocity_step(self.velocity[i][d], x[d], best[d], freq) for d in range(self.dim)]
            s = [x[d] + v_i[d] for d in range(self.dim)]

            # Local-walk trigger: ONE draw per bat.
            trigger = ctx.rng.random()
            if trigger > self.r0:
                # COMPLETE OVERWRITE of the candidate; the velocity move
                # above is discarded for `s` but v_i still commits below.
                s = [best[d] + self.local_walk_scale * ctx.rng.gauss(0.0, 1.0) for d in range(self.dim)]

            self.velocity[i] = v_i
            offspring.append(s)

        # boundary/clamp applied AFTER generate returns, same timing as
        # the Rust preset's separate boundary stage.
        offspring = [[clamp(v, lo, hi) for v in row] for row in offspring]

        new_fitness = ctx.evaluate(offspring)

        # replace/bat-loudness-greedy: accept iff off_fit[i] <= own fit[i]
        # AND a FRESH, UNCONDITIONAL loudness draw < a0 -- the draw always
        # fires (MATLAB's non-short-circuiting `&`), whether or not the
        # fitness test holds.
        for i in range(self.pop_size):
            loud_draw = ctx.rng.random()
            if new_fitness[i] <= self.fitness[i] and loud_draw < self.a0:
                self.pop[i], self.fitness[i] = offspring[i], new_fitness[i]


def main():
    algo = Ba()
    res = algo.solve(sezgi.bbob(1, algo.dim, 1), budget=BUDGET, seed=SEED)
    print(f"ba (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
