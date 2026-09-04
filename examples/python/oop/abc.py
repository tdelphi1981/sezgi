"""OOP twin of examples/python/abc.py -- same math, same RNG draw order.

Port of the pure-Python Artificial Bee Colony script onto sezgi.AskTellAlgorithm.
The pure script (and the Rust module doc it cites) remains the provenance
for the update equations and the three-phase (employed/onlooker/scout)
structure; this file re-derives NOTHING and must reproduce the pure
script's evals_used/best_f/gap output bit-for-bit at the same seed --
enforced by py-sezgi/tests/test_examples_oop_parity.py.

Per-source trial counters persist in self.trials. The onlooker phase is a
repeated LINEAR SCAN (not roulette-wheel sampling); the scout phase abandons
at most one source per cycle, chosen by argmax-with-last-index-tiebreak.

No up-front whole-generation budget guard is needed here (contrast tlbo.py
and oop/hho.py): only the employed phase's batch is unconditional, and it
is always the FIRST budget-consuming action of the step, so ctx.evaluate's
own pre-check reproduces the pure script's `while used + POP_SIZE <= BUDGET`
guard exactly, with nothing spent beforehand that the pure script wouldn't
also have spent. The onlooker and scout phases are already opportunistic
in the pure script (per-call try/except), which ctx.evaluate's raise
reproduces directly.
"""
import sezgi
from sezgi.algo import BudgetExhausted

BUDGET = 2000
SEED = 42


def abc_dim_step(x_i_j, x_k_j, phi):
    return x_i_j + phi * (x_i_j - x_k_j)


def abc_candidate(x_i, x_k, j, phi):
    c = list(x_i)
    c[j] = abc_dim_step(x_i[j], x_k[j], phi)
    return c


def abc_fitness_transform(f):
    return 1.0 / (f + 1.0) if f >= 0.0 else 1.0 + abs(f)


def abc_probabilities(transformed):
    max_fit = max(transformed)
    return [0.9 * (f / max_fit) + 0.1 for f in transformed]


def abc_scout_target(trials):
    """argmax, ties -> LAST index (ABCorig.m's ind(end), the opposite
    tie-break from this project's usual first-seen-wins convention)."""
    best = 0
    for k in range(1, len(trials)):
        if trials[k] >= trials[best]:
            best = k
    return best


def abc_draw_move(sn, dim, i, rng):
    j = rng.randrange(dim)
    while True:
        k = rng.randrange(sn)
        if k != i:
            break
    phi = (rng.random() - 0.5) * 2.0
    return j, k, phi


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Abc(sezgi.AskTellAlgorithm):
    """Artificial Bee Colony (Karaboga 2005, TR-06, Erciyes University;
    Karaboga & Basturk 2007, Journal of Global Optimization 39(3),
    459-471), modeling the employed/onlooker/scout cycle explicitly, one
    stage at a time, over `sezgi.AskTellAlgorithm`'s ask/tell loop.
    `dim`: problem dimensionality. `pop_size`: food-source count SN (==
    sezgi's own `pop_size`, NOT Karaboga's colony size `NP=2*SN` -- there
    is only ever one array of solutions). `limit`: the scout phase's
    trial-abandonment threshold (default `pop_size * dim`, Karaboga &
    Akay 2009's dimension-scaling rule of thumb)."""

    name = "abc"

    def __init__(self, dim=5, pop_size=20, limit=None):
        self.dim = dim
        self.pop_size = pop_size
        self.limit = limit if limit is not None else pop_size * dim

    def setup(self, ctx):
        self.pop = [ctx.random_point() for _ in range(self.pop_size)]
        self.fitness = ctx.evaluate(self.pop)
        self.trials = [0] * self.pop_size

    def step(self, ctx):
        lo, hi = ctx.bounds

        # ---- Employed phase: frozen batch, one candidate per source ----
        candidates = []
        for i in range(self.pop_size):
            j, k, phi = abc_draw_move(self.pop_size, self.dim, i, ctx.rng)
            candidates.append([clamp(v, lo, hi) for v in abc_candidate(self.pop[i], self.pop[k], j, phi)])
        new_fitness = ctx.evaluate(candidates)
        for i in range(self.pop_size):
            if new_fitness[i] < self.fitness[i]:
                self.pop[i], self.fitness[i], self.trials[i] = candidates[i], new_fitness[i], 0
            else:
                self.trials[i] += 1

        # ---- Onlooker phase: repeated linear scan, exactly SN accepted visits ----
        transformed = [abc_fitness_transform(f) for f in self.fitness]
        prob = abc_probabilities(transformed)
        i, t, budget_exhausted = 0, 0, False
        while t < self.pop_size and not budget_exhausted:
            if ctx.rng.random() < prob[i]:
                t += 1
                j, k, phi = abc_draw_move(self.pop_size, self.dim, i, ctx.rng)
                candidate = abc_candidate(self.pop[i], self.pop[k], j, phi)
                candidate[j] = clamp(candidate[j], lo, hi)
                try:
                    f = ctx.evaluate([candidate])[0]
                except BudgetExhausted:
                    budget_exhausted = True
                else:
                    if f < self.fitness[i]:
                        self.pop[i], self.fitness[i], self.trials[i] = candidate, f, 0
                    else:
                        self.trials[i] += 1
            i = (i + 1) % self.pop_size

        # ---- Scout phase: at most one re-randomized source per cycle ----
        if not budget_exhausted:
            scout_idx = abc_scout_target(self.trials)
            if self.trials[scout_idx] > self.limit:
                new_pos = ctx.random_point()
                try:
                    f = ctx.evaluate([new_pos])[0]
                except BudgetExhausted:
                    pass
                else:
                    self.pop[scout_idx], self.fitness[scout_idx], self.trials[scout_idx] = new_pos, f, 0


def main():
    algo = Abc()
    res = algo.solve(sezgi.bbob(1, algo.dim, 1), budget=BUDGET, seed=SEED)
    print(f"abc (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
