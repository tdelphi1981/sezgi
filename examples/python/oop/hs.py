"""OOP twin of examples/python/hs.py -- same math, same RNG draw order.

Port of the pure-Python Harmony Search script onto sezgi.AskTellAlgorithm. The
pure script (and the Rust module doc it cites) remains the provenance for
the update equations; this file re-derives NOTHING and must reproduce the
pure script's evals_used/best_f/gap output bit-for-bit at the same seed --
enforced by py-sezgi/tests/test_examples_oop_parity.py.
"""
import sezgi

BUDGET = 2000
SEED = 42


def hs_dim_step(hmcr, par, bw_d, lo_d, hi_d, memory_d, rng):
    u1 = rng.random()
    if u1 < hmcr:
        j = rng.randrange(len(memory_d))
        value = memory_d[j]
        u2 = rng.random()
        if u2 < par:
            u3 = rng.random()
            value += bw_d * (2.0 * u3 - 1.0)
        return value
    u4 = rng.random()
    return lo_d + u4 * (hi_d - lo_d)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Hs(sezgi.AskTellAlgorithm):
    """Harmony Search (Geem, Kim & Loganathan 2001, "A new heuristic
    optimization algorithm: harmony search", Simulation), modeling the
    memory-consideration / pitch-adjustment / random-selection cycle
    explicitly over `sezgi.AskTellAlgorithm`'s ask/tell loop. `dim`:
    problem dimensionality. `hms`: harmony memory size (sezgi's
    `pop_size` for this preset). `hmcr`: harmony memory considering
    rate. `par`: pitch-adjustment rate. `bw_fraction`: pitch-adjustment
    bandwidth, as a fraction of the search-space width per dimension."""

    name = "hs"

    def __init__(self, dim=5, hms=30, hmcr=0.9, par=0.3, bw_fraction=0.01):
        self.dim = dim
        self.hms = hms
        self.hmcr = hmcr
        self.par = par
        self.bw_fraction = bw_fraction

    def setup(self, ctx):
        lo, hi = ctx.bounds
        self.bw = [self.bw_fraction * (hi - lo)] * self.dim
        self.memory = [ctx.random_point() for _ in range(self.hms)]
        self.fitness = ctx.evaluate(self.memory)

    def step(self, ctx):
        lo, hi = ctx.bounds
        new_harmony = [
            hs_dim_step(self.hmcr, self.par, self.bw[d], lo, hi,
                        [row[d] for row in self.memory], ctx.rng)
            for d in range(self.dim)
        ]
        new_harmony = [clamp(v, lo, hi) for v in new_harmony]
        new_f = ctx.evaluate([new_harmony])[0]

        # replace/worst-if-better: replace the current worst iff new_f beats it.
        worst = max(range(self.hms), key=lambda k: self.fitness[k])
        if new_f < self.fitness[worst]:
            self.memory[worst] = new_harmony
            self.fitness[worst] = new_f


def main():
    algo = Hs()
    res = algo.solve(sezgi.bbob(1, algo.dim, 1), budget=BUDGET, seed=SEED)
    print(f"hs (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
