"""OOP example: a DE/rand/2/bin-shaped `sezgi.PopulationAlgorithm` variant
-- overrides ONLY `vary()`, keeping the base's default tournament
`select()`. `../custom_de_variant.py` already teaches DE/rand/1 with NO
crossover step (a mutant replaces the whole individual); this file teaches
the OTHER classic DE shape: TWO donor difference vectors (DE/rand/2)
combined with BINOMIAL crossover against the tournament-selected target
(the "/bin" half of the "DE/rand/2/bin" name) -- read alongside
`../custom_de_variant.py`, the pair covers both ends of the mutation/
crossover split a DE `vary()` can express. Solved over a built-in
`sezgi.bbob(...)` problem, same shape as every other engine example in
this directory.
"""
import sezgi


class CustomDERandTwoBin(sezgi.PopulationAlgorithm):
    """DE/rand/2/bin vary(): for each target slot `t` (0-indexed over the
    `n = len(parents)` tournament-selected parents), independently draws
    FIVE donor indices via `ctx.rng.next_below(n)` (r1..r5, not required
    to be distinct from each other or from `t` -- same "independent
    draws, no rejection loop" convention `custom_de_variant.py`'s own
    vary() uses) and forms the donor vector
    `donor = parents[r1] + f * (parents[r2] - parents[r3])
                          + f * (parents[r4] - parents[r5])`
    (DE/rand/2's two difference terms). BINOMIAL crossover then mixes the
    donor with the target parent `parents[t]` gene-by-gene: gene `i` is
    taken from `donor` if `ctx.rng.next_f64() < cr` OR `i == jrand` (one
    dimension, drawn via `ctx.rng.next_below(dim)`, forced from the donor
    so a child can never be an exact copy of its own target -- the
    standard DE/bin "at least one donor gene" guarantee), else from the
    target unchanged.

    `f`: DE differential weight, scaling both donor differences.
    `cr`: crossover rate, the per-gene probability of taking the donor's
    own value instead of the target's.
    """

    def __init__(self, f=0.5, cr=0.9):
        self.f = f
        self.cr = cr

    def vary(self, parents, ctx):
        n = len(parents)
        offspring = []
        for t in range(n):
            r1 = ctx.rng.next_below(n)
            r2 = ctx.rng.next_below(n)
            r3 = ctx.rng.next_below(n)
            r4 = ctx.rng.next_below(n)
            r5 = ctx.rng.next_below(n)
            target = parents[t]
            dim = len(target)
            jrand = ctx.rng.next_below(dim)
            donor = [parents[r1][i]
                     + self.f * (parents[r2][i] - parents[r3][i])
                     + self.f * (parents[r4][i] - parents[r5][i])
                     for i in range(dim)]
            child = [donor[i] if (ctx.rng.next_f64() < self.cr or i == jrand)
                     else target[i] for i in range(dim)]
            offspring.append(child)
        return offspring


def main():
    res = CustomDERandTwoBin().run(sezgi.bbob(1, 10, 1), budget=2000, seed=42, pop_size=20)
    print(f"custom_de (oop engine): evals_used={res.evals_used} "
          f"best_f={res.best_f:.6g} gap={res.gap:.6g}")
    # ANCHORED (per this project's convention): measured once at this exact
    # config (sezgi.bbob(1, 10, 1), budget=2000, seed=42, pop_size=20), then
    # pinned as literal values below.
    assert res.evals_used == 2000
    assert res.best_f == -84.38545173431334
    assert res.gap == 0.018487196747699386


if __name__ == "__main__":
    main()
