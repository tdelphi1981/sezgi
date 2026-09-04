"""OOP twin of examples/python/ssa.py -- same math, same RNG draw order.

Port of the pure-Python Salp Swarm Algorithm script onto sezgi.AskTellAlgorithm.
The pure script (and the Rust module doc it cites) remains the provenance
for the update equations; this file re-derives NOTHING and must reproduce
the pure script's evals_used/best_f/gap output bit-for-bit at the same
seed -- enforced by py-sezgi/tests/test_examples_oop_parity.py.
"""
import math

import sezgi

BUDGET = 2000
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
    """Salp Swarm Algorithm (Mirjalili, Gandomi, Mirjalili, Saremi,
    Faris & Mirjalili 2017, Advances in Engineering Software 114,
    163-191), modeling the leader-follower salp-chain update explicitly
    over `sezgi.AskTellAlgorithm`'s ask/tell loop -- leaders move toward
    food, followers chain off the immediately preceding, already-updated
    salp in the same sweep. `dim`: problem dimensionality. `pop_size`:
    number of salps (split into a fixed positional leader/follower
    half, not fitness-based)."""

    name = "ssa"

    def __init__(self, dim=5, pop_size=30):
        self.dim = dim
        self.pop_size = pop_size

    def setup(self, ctx):
        self.pop = [ctx.random_point() for _ in range(self.pop_size)]
        self.fitness = ctx.evaluate(self.pop)
        self.leader_count = self.pop_size // 2  # fixed positional split, not fitness-based

    def step(self, ctx):
        lo, hi = ctx.bounds
        progress = min(max(ctx.evals_used / ctx.budget, 0.0), 1.0)
        c1 = ssa_c1(progress)

        # Food: current-population fitness argmin, ties -> lower index,
        # computed once before any draws.
        food_idx = min(range(self.pop_size), key=lambda i: (self.fitness[i], i))
        food = self.pop[food_idx]

        # Built SEQUENTIALLY in ascending index order: followers chain off
        # the already-computed offspring entries from earlier this same
        # sweep (SSA.m's in-place transpose-and-reassign semantics).
        offspring = []
        for i in range(self.pop_size):
            x_old = self.pop[i]
            if i < self.leader_count:
                new_x = []
                for d in range(self.dim):
                    # Pinned draw order: c2, then c3, per (leader, d).
                    c2 = ctx.rng.random()
                    c3 = ctx.rng.random()
                    new_x.append(clamp(ssa_leader_dim_step(food[d], c1, c2, c3, lo, hi), lo, hi))
            else:
                # Follower: zero draws; chains off offspring[i-1], the
                # ALREADY-COMPUTED entry from this same sweep.
                prev = offspring[i - 1]
                new_x = [clamp(ssa_follower_dim_step(x_old[d], prev[d]), lo, hi) for d in range(self.dim)]
            offspring.append(new_x)

        # replace/generational: SSA.m overwrites every salp unconditionally.
        self.fitness = ctx.evaluate(offspring)
        self.pop = offspring


def main():
    algo = Ssa()
    res = algo.solve(sezgi.bbob(1, algo.dim, 1), budget=BUDGET, seed=SEED)
    print(f"ssa (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
