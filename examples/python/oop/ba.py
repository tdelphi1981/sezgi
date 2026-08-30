"""OOP twin of examples/python/ba.py -- same math, same RNG draw order.

Port of the pure-Python Bat Algorithm script onto sezgi.Algorithm. The pure
script (and the Rust module doc it cites) remains the provenance for the
update equations and the two verified sign-inversion quirks (inverted
[-2, 0] frequency range, (X - X_best) velocity term); this file re-derives
NOTHING and must reproduce the pure script's evals_used/best_f/gap output
bit-for-bit at the same seed -- enforced by
py-sezgi/tests/test_examples_oop_parity.py.
"""
import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30  # number of bats
SEED = 42
QMIN, QMAX = 0.0, 2.0
R0 = 0.5   # fixed pulse rate
A0 = 0.5   # fixed loudness
LOCAL_WALK_SCALE = 0.001


def ba_frequency(u):
    """VERIFIED VERBATIM against bat_algorithm.m: Qmin + (Qmin-Qmax)*u --
    note the sign, giving [-2, 0] for the pinned defaults, not [0, 2]."""
    return QMIN + (QMIN - QMAX) * u


def ba_velocity_step(v_d, x_d, best_d, freq):
    return v_d + (x_d - best_d) * freq


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Ba(sezgi.Algorithm):
    name = "ba"

    def setup(self, ctx):
        self.pop = [ctx.random_point() for _ in range(POP_SIZE)]
        self.fitness = ctx.evaluate(self.pop)
        self.velocity = [[0.0] * DIM for _ in range(POP_SIZE)]

    def step(self, ctx):
        lo, hi = ctx.bounds

        # best: current-population fitness argmin, ties -> lower index,
        # computed once before any draws (sezgi simplification, shared
        # with the Rust preset: bat_algorithm.m persists a best-ever).
        best_idx = min(range(POP_SIZE), key=lambda i: (self.fitness[i], i))
        best = self.pop[best_idx]

        offspring = []
        for i in range(POP_SIZE):
            x = self.pop[i]
            # ONE frequency draw per bat (not per dimension).
            freq = ba_frequency(ctx.rng.random())
            v_i = [ba_velocity_step(self.velocity[i][d], x[d], best[d], freq) for d in range(DIM)]
            s = [x[d] + v_i[d] for d in range(DIM)]

            # Local-walk trigger: ONE draw per bat.
            trigger = ctx.rng.random()
            if trigger > R0:
                # COMPLETE OVERWRITE of the candidate; the velocity move
                # above is discarded for `s` but v_i still commits below.
                s = [best[d] + LOCAL_WALK_SCALE * ctx.rng.gauss(0.0, 1.0) for d in range(DIM)]

            self.velocity[i] = v_i
            offspring.append(s)

        # boundary/clamp applied AFTER generate returns, same timing as
        # the Rust preset's separate boundary stage.
        offspring = [[clamp(v, lo, hi) for v in row] for row in offspring]

        new_fitness = ctx.evaluate(offspring)

        # replace/bat-loudness-greedy: accept iff off_fit[i] <= own fit[i]
        # AND a FRESH, UNCONDITIONAL loudness draw < A0 -- the draw always
        # fires (MATLAB's non-short-circuiting `&`), whether or not the
        # fitness test holds.
        for i in range(POP_SIZE):
            loud_draw = ctx.rng.random()
            if new_fitness[i] <= self.fitness[i] and loud_draw < A0:
                self.pop[i], self.fitness[i] = offspring[i], new_fitness[i]


def main():
    res = Ba().solve(sezgi.bbob(1, DIM, 1), budget=BUDGET, seed=SEED)
    print(f"ba (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
