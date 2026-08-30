"""OOP twin of examples/python/hs.py -- same math, same RNG draw order.

Port of the pure-Python Harmony Search script onto sezgi.Algorithm. The
pure script (and the Rust module doc it cites) remains the provenance for
the update equations; this file re-derives NOTHING and must reproduce the
pure script's evals_used/best_f/gap output bit-for-bit at the same seed --
enforced by py-sezgi/tests/test_examples_oop_parity.py.
"""
import sezgi

DIM = 5
BUDGET = 2000
HMS = 30  # harmony memory size (sezgi's "pop_size" for this preset)
SEED = 42
HMCR = 0.9
PAR = 0.3
BW_FRACTION = 0.01


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


class Hs(sezgi.Algorithm):
    name = "hs"

    def setup(self, ctx):
        lo, hi = ctx.bounds
        self.bw = [BW_FRACTION * (hi - lo)] * DIM
        self.memory = [ctx.random_point() for _ in range(HMS)]
        self.fitness = ctx.evaluate(self.memory)

    def step(self, ctx):
        lo, hi = ctx.bounds
        new_harmony = [
            hs_dim_step(HMCR, PAR, self.bw[d], lo, hi,
                        [row[d] for row in self.memory], ctx.rng)
            for d in range(DIM)
        ]
        new_harmony = [clamp(v, lo, hi) for v in new_harmony]
        new_f = ctx.evaluate([new_harmony])[0]

        # replace/worst-if-better: replace the current worst iff new_f beats it.
        worst = max(range(HMS), key=lambda k: self.fitness[k])
        if new_f < self.fitness[worst]:
            self.memory[worst] = new_harmony
            self.fitness[worst] = new_f


def main():
    res = Hs().solve(sezgi.bbob(1, DIM, 1), budget=BUDGET, seed=SEED)
    print(f"hs (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
