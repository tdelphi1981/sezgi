"""OOP twin of examples/python/tlbo.py -- same math, same RNG draw order.

Port of the pure-Python Teaching-Learning-Based Optimization script onto
sezgi.AskTellAlgorithm. The pure script (and the Rust module doc it cites) remains
the provenance for the update equations; this file re-derives NOTHING and
must reproduce the pure script's evals_used/best_f/gap output bit-for-bit
at the same seed -- enforced by py-sezgi/tests/test_examples_oop_parity.py.

Each generation is TWO passes (teacher then learner), EACH with its own
greedy accept and its own ctx.evaluate call -- 2*POP_SIZE evaluations per
step, matching the pure script exactly.
"""
import sezgi
from sezgi.algo import BudgetExhausted

BUDGET = 2000
SEED = 42


def tlbo_teacher_dim_step(x_d, teacher_d, mean_d, tf, r):
    return x_d + r * (teacher_d - tf * mean_d)


def tlbo_learner_dim_step(x_d, partner_d, r, partner_is_better):
    step = x_d - partner_d
    if partner_is_better:
        step = -step
    return x_d + r * step


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Tlbo(sezgi.AskTellAlgorithm):
    """Teaching-Learning-Based Optimization (Rao, Savsani & Vakharia
    2011, Computer-Aided Design 43(3), 303-315), modeling the two-phase
    teacher (move-toward-teacher-minus-mean) then learner
    (move-toward/away-from-a-random-partner) cycle explicitly over
    `sezgi.AskTellAlgorithm`'s ask/tell loop, each phase with its own
    greedy accept. `dim`: problem dimensionality. `pop_size`: number of
    learners."""

    name = "tlbo"

    def __init__(self, dim=5, pop_size=30):
        self.dim = dim
        self.pop_size = pop_size

    def setup(self, ctx):
        self.pop = [ctx.random_point() for _ in range(self.pop_size)]
        self.fitness = ctx.evaluate(self.pop)

    def step(self, ctx):
        # sezgi decision: the pure script guards the WHOLE generation (both
        # the teacher and the learner phase, 2*POP_SIZE evals total) up
        # front via `while used + 2*POP_SIZE <= BUDGET` -- it never starts
        # a generation it cannot finish. The driver here only calls step()
        # while ctx.remaining > 0, and ctx.evaluate only guards ONE call at
        # a time; without this explicit check the twin would run a bonus
        # teacher-only partial generation (committing its greedy accept)
        # whenever remaining fell strictly between POP_SIZE and
        # 2*POP_SIZE-1, a generation the pure script never runs and a
        # different evals_used stopping point. Raise BudgetExhausted
        # ourselves, before any evaluate call, whenever the full
        # 2*POP_SIZE doesn't fit -- an honest structural port of the pure
        # script's own up-front guard.
        if ctx.remaining < 2 * self.pop_size:
            raise BudgetExhausted(
                f"generation needs {2 * self.pop_size} evals, {ctx.remaining} remaining")

        lo, hi = ctx.bounds

        # ---- Teacher phase (Population mean first, then teacher -- no draw) ----
        mean = [0.0] * self.dim
        for x in self.pop:
            for d in range(self.dim):
                mean[d] += x[d]
        mean = [m / self.pop_size for m in mean]
        teacher_idx = min(range(self.pop_size), key=lambda i: (self.fitness[i], i))
        teacher = self.pop[teacher_idx]

        teacher_offspring = []
        for i in range(self.pop_size):
            x = self.pop[i]
            tf = 1.0 + ctx.rng.randrange(2)  # randi{1,2}, once per learner
            new_x = [clamp(tlbo_teacher_dim_step(x[d], teacher[d], mean[d], tf, ctx.rng.random()), lo, hi) for d in range(self.dim)]
            teacher_offspring.append(new_x)
        new_fitness = ctx.evaluate(teacher_offspring)
        for i in range(self.pop_size):
            if new_fitness[i] < self.fitness[i]:
                self.pop[i], self.fitness[i] = teacher_offspring[i], new_fitness[i]

        # ---- Learner phase (on the teacher phase's updated population) ----
        learner_offspring = []
        for i in range(self.pop_size):
            j = ctx.rng.randrange(self.pop_size)
            while j == i:
                j = ctx.rng.randrange(self.pop_size)
            partner_is_better = self.fitness[j] < self.fitness[i]
            x, partner = self.pop[i], self.pop[j]
            new_x = [clamp(tlbo_learner_dim_step(x[d], partner[d], ctx.rng.random(), partner_is_better), lo, hi) for d in range(self.dim)]
            learner_offspring.append(new_x)
        new_fitness = ctx.evaluate(learner_offspring)
        for i in range(self.pop_size):
            if new_fitness[i] < self.fitness[i]:
                self.pop[i], self.fitness[i] = learner_offspring[i], new_fitness[i]


def main():
    algo = Tlbo()
    res = algo.solve(sezgi.bbob(1, algo.dim, 1), budget=BUDGET, seed=SEED)
    print(f"tlbo (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
