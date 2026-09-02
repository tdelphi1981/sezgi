"""OOP twin of examples/python/ssa.py -- same math, same RNG draw order.

Port of the pure-Python Salp Swarm Algorithm script onto sezgi.AskTellAlgorithm.
The pure script (and the Rust module doc it cites) remains the provenance
for the update equations; this file re-derives NOTHING and must reproduce
the pure script's evals_used/best_f/gap output bit-for-bit at the same
seed -- enforced by py-sezgi/tests/test_examples_oop_parity.py.
"""
import math

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30  # number of salps
SEED = 42


def ssa_c1(progress):
    return 2.0 * math.exp(-((4.0 * progress) ** 2))


def ssa_leader_dim_step(food_d, c1, c2, c3, lo, hi):
    term = c1 * ((hi - lo) * c2 + lo)
    return food_d + term if c3 < 0.5 else food_d - term


def ssa_follower_dim_step(x_old_d, prev_new_d):
    return (x_old_d + prev_new_d) / 2.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Ssa(sezgi.AskTellAlgorithm):
    name = "ssa"

    def setup(self, ctx):
        self.pop = [ctx.random_point() for _ in range(POP_SIZE)]
        self.fitness = ctx.evaluate(self.pop)
        self.leader_count = POP_SIZE // 2  # fixed positional split, not fitness-based

    def step(self, ctx):
        lo, hi = ctx.bounds
        progress = min(max(ctx.evals_used / ctx.budget, 0.0), 1.0)
        c1 = ssa_c1(progress)

        # Food: current-population fitness argmin, ties -> lower index,
        # computed once before any draws.
        food_idx = min(range(POP_SIZE), key=lambda i: (self.fitness[i], i))
        food = self.pop[food_idx]

        # Built SEQUENTIALLY in ascending index order: followers chain off
        # the already-computed offspring entries from earlier this same
        # sweep (SSA.m's in-place transpose-and-reassign semantics).
        offspring = []
        for i in range(POP_SIZE):
            x_old = self.pop[i]
            if i < self.leader_count:
                new_x = []
                for d in range(DIM):
                    # Pinned draw order: c2, then c3, per (leader, d).
                    c2 = ctx.rng.random()
                    c3 = ctx.rng.random()
                    new_x.append(clamp(ssa_leader_dim_step(food[d], c1, c2, c3, lo, hi), lo, hi))
            else:
                # Follower: zero draws; chains off offspring[i-1], the
                # ALREADY-COMPUTED entry from this same sweep.
                prev = offspring[i - 1]
                new_x = [clamp(ssa_follower_dim_step(x_old[d], prev[d]), lo, hi) for d in range(DIM)]
            offspring.append(new_x)

        # replace/generational: SSA.m overwrites every salp unconditionally.
        self.fitness = ctx.evaluate(offspring)
        self.pop = offspring


def main():
    res = Ssa().solve(sezgi.bbob(1, DIM, 1), budget=BUDGET, seed=SEED)
    print(f"ssa (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
