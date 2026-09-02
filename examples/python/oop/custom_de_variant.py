"""OOP example: a custom DE/rand/1-shaped `sezgi.PopulationAlgorithm`
variant (M4-1 Task 4) -- overrides ONLY `vary()`, keeping the base's
default tournament `select()`. Unlike every algo-named file in this
directory (a port of an existing pure script onto `sezgi.AskTellAlgorithm`,
enforced bit-for-bit by `test_examples_oop_parity.py`'s 17-pair gate), this
is a NEW-style example authored directly against the engine-hosted
`sezgi.PopulationAlgorithm` base with no pure-script twin to reproduce --
same precedent as `tsp_two_opt.py`. Its own determinism/anchored-output
gate is `py-sezgi/tests/test_oop_families.py`.
"""
import sezgi

F = 0.5  # DE differential weight


class CustomDERandOne(sezgi.PopulationAlgorithm):
    """DE/rand/1-shaped vary(): for each parent slot, draws THREE donor
    indices via ctx.rng.next_below(len(parents)) and combines them as
    r1 + F * (r2 - r3) -- the classic DE/rand/1 mutation, applied directly
    over the tournament-selected parent pool this base's default select()
    already built. No separate crossover step (kept deliberately small)."""

    def vary(self, parents, ctx):
        n = len(parents)
        offspring = []
        for _ in range(n):
            r1 = ctx.rng.next_below(n)
            r2 = ctx.rng.next_below(n)
            r3 = ctx.rng.next_below(n)
            a, b, c = parents[r1], parents[r2], parents[r3]
            offspring.append([ai + F * (bi - ci) for ai, bi, ci in zip(a, b, c)])
        return offspring


def main():
    res = CustomDERandOne().run(sezgi.bbob(1, 10, 1), budget=2000, seed=42, pop_size=20)
    print(f"custom_de_variant (oop): evals_used={res.evals_used} "
          f"best_f={res.best_f:.6g} gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
