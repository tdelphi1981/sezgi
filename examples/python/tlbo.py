"""Pure-Python Teaching-Learning-Based Optimization (Rao, R.V., Savsani,
V.J., Vakharia, D.P. 2011, "Teaching-learning-based optimization: A novel
method for constrained mechanical design optimization problems",
Computer-Aided Design 43(3), 303-315).

Teaches the SAME pinned update equations as sezgi's `gen/tlbo-teacher` +
`gen/tlbo-learner` (each paired with `replace/one-to-one-greedy`) Rust
components -- sezgi's FIRST multi-stage preset (see
crates/components/src/tlbo.rs's module doc) -- with nothing but the
standard library, driven through `sezgi.EvalSession`. Does NOT reproduce
the Rust component's RNG-stream (draw order) contract, only the same
update rule; see examples/specs/tlbo.toml for the pinned-RNG preset.

Tier note: a **labeled metaphor preset** -- faithful to a well-regarded
THIRD-PARTY reimplementation's own update equations (Yarpiz's `tlbo.m`,
Project Code YPEA111 -- Rao's own code was not independently locatable),
with a pinned deterministic draw order, but NOT validated against the
paper's reported benchmark numbers.

This script models BOTH generation stages EXPLICITLY, one after the other,
exactly as the two-stage Rust preset does: a full teacher pass over every
learner (with its own greedy accept, right there, per learner) followed by
a full learner pass (also with its own greedy accept) -- 2*pop_size
evaluations per generation, not 1*pop_size. The teaching factor `TF` is
`randi{1,2}`, drawn ONCE PER LEARNER (not once per generation); the
learner phase's partner `j` is drawn UNCONDITIONALLY DISTINCT from `i`
(never self-selecting, unlike fpa.py's/woa's "self not excluded" idiom).

sezgi does NOT implement any duplicate-removal/re-evaluation step some of
Rao's own published comparisons quietly relied on (see Crepinsek, Liu &
Mernik 2012, "A note on teaching-learning-based optimization algorithm",
Information Sciences 212, 79-93, for the documented critique of that
omission and TLBO's "parameter-free" framing) -- every evaluation this
script performs is counted, honestly, exactly 2*pop_size per generation.

No cross-algorithm quality claims are made or implied here -- this is a
single seed on a single problem, reported as a gap, not a ranking.
"""
import random

import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 30  # number of learners
SEED = 42
LO, HI = -5.0, 5.0


def tlbo_teacher_dim_step(x_d, teacher_d, mean_d, tf, r):
    return x_d + r * (teacher_d - tf * mean_d)


def tlbo_learner_dim_step(x_d, partner_d, r, partner_is_better):
    step = x_d - partner_d
    if partner_is_better:
        step = -step
    return x_d + r * step


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def main():
    rng = random.Random(SEED)
    session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET)
    # session = sezgi.EvalSession(fid=1, dim=DIM, instance=1, budget=BUDGET,
    #                              log_dir="logs/tlbo-py", algo_name="tlbo-py", seed=SEED)

    pop = [[rng.uniform(LO, HI) for _ in range(DIM)] for _ in range(POP_SIZE)]
    fitness = session.evaluate(pop)
    used = POP_SIZE

    while used + 2 * POP_SIZE <= BUDGET:
        # ---- Teacher phase (Population mean first, then teacher -- no draw) ----
        mean = [0.0] * DIM
        for x in pop:
            for d in range(DIM):
                mean[d] += x[d]
        mean = [m / POP_SIZE for m in mean]
        teacher_idx = min(range(POP_SIZE), key=lambda i: (fitness[i], i))
        teacher = pop[teacher_idx]

        teacher_offspring = []
        for i in range(POP_SIZE):
            x = pop[i]
            tf = 1.0 + rng.randrange(2)  # randi{1,2}, once per learner
            new_x = [clamp(tlbo_teacher_dim_step(x[d], teacher[d], mean[d], tf, rng.random()), LO, HI) for d in range(DIM)]
            teacher_offspring.append(new_x)
        new_fitness = session.evaluate(teacher_offspring)
        used += POP_SIZE
        for i in range(POP_SIZE):
            if new_fitness[i] < fitness[i]:
                pop[i], fitness[i] = teacher_offspring[i], new_fitness[i]

        # ---- Learner phase (on the teacher phase's updated population) ----
        learner_offspring = []
        for i in range(POP_SIZE):
            j = rng.randrange(POP_SIZE)
            while j == i:
                j = rng.randrange(POP_SIZE)
            partner_is_better = fitness[j] < fitness[i]
            x, partner = pop[i], pop[j]
            new_x = [clamp(tlbo_learner_dim_step(x[d], partner[d], rng.random(), partner_is_better), LO, HI) for d in range(DIM)]
            learner_offspring.append(new_x)
        new_fitness = session.evaluate(learner_offspring)
        used += POP_SIZE
        for i in range(POP_SIZE):
            if new_fitness[i] < fitness[i]:
                pop[i], fitness[i] = learner_offspring[i], new_fitness[i]

    best_x, best_f = session.best()
    gap = best_f - session.f_opt()
    print(f"tlbo: evals_used={session.evals_used()} best_f={best_f:.6g} gap={gap:.6g}")
    session.finish()


if __name__ == "__main__":
    main()
