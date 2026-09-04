"""OOP example: a mixed-space (Float + Categorical + Int) `sezgi.Problem`
subclass paired with a `sezgi.Algorithm` subclass whose `generate()`
handles the resulting per-block TUPLE payload directly -- unlike every
other file in this directory, whose problems are all single-block (Float
or Permutation), so `pop.individuals[i]`/`generate()`'s return values are
bare lists there. Here `x` (both `evaluate`'s argument and each
`pop.individuals[i]` entry) is a 3-tuple `(floats, cats, ints)`, in
`space()`'s own declared block order -- see `sezgi.Problem`'s own module
docstring for the general bare/tuple convention this follows. Also
demonstrates `sezgi.Algorithm.validate_space()`, the build-time veto hook
no other file in this directory exercises.
"""
import sezgi


class MixedTuningProblem(sezgi.Problem):
    """A toy hyperparameter-tuning-flavored objective: `Float(-5, 5, 2)`
    (two continuous knobs), `Categorical(3, 1)` (one discrete "mode"
    choice), `Int(1, 5, 1)` (one integer knob) -- three blocks, in that
    order. `evaluate(x)` unpacks `x` into its three per-block values
    (the multi-block convention: `x` is a tuple, one entry per block, in
    `space()`'s own declared order) and sums each sub-block's own
    contribution: the two floats squared (minimized at `0.0`), a
    fixed per-mode penalty (minimized by mode `0`, penalty `0.0`), and a
    scaled squared distance from the int knob's own target value `3`
    (minimized at `0.0`) -- so the THEORETICAL minimum across the whole
    space is exactly `0.0`, reachable by the int/categorical blocks
    exactly and approached (never landed on exactly by a stochastic
    search) by the two continuous ones, giving `optimum()` a real value
    to report a `gap` against."""

    _MODE_PENALTY = (0.0, 0.5, 1.5)

    def space(self):
        return sezgi.Space(
            sezgi.Float(-5.0, 5.0, 2),
            sezgi.Categorical(3, 1),
            sezgi.Int(1, 5, 1),
        )

    def evaluate(self, x):
        floats, cats, ints = x
        return (sum(v * v for v in floats)
                + self._MODE_PENALTY[cats[0]]
                + 0.1 * (ints[0] - 3) ** 2)

    def optimum(self):
        return 0.0


class MixedBlockVariation(sezgi.Algorithm):
    """generate(): a full override that must handle the THREE-BLOCK tuple
    convention `MixedTuningProblem`'s own `Space(Float, Categorical,
    Int)` produces. No `select()`/`vary()` split here (that is
    `PopulationAlgorithm`'s job): each offspring is bred from exactly ONE
    uniformly-random parent (`ctx.rng.next_below(n)`, `n =
    len(pop.individuals)`) with the three sub-blocks varied
    independently:

    - Float sub-block: each coordinate takes an independent
      `ctx.rng.next_f64()`-driven step of `+-step` (unbounded -- the
      engine's own `boundary/clamp` repairs it back into `[-5, 5]` after
      `generate()` returns, the same as any other block kind here).
    - Categorical sub-block: each gene is independently RESAMPLED
      uniformly over `0..k` (`ctx.rng.next_below(k)`, `k` read from
      `ctx.space[1]["k"]` -- the categorical block's own descriptor,
      `sezgi.Algorithm`'s own class docstring pins this dict shape) with
      probability `p_resample` (an independent `ctx.rng.next_f64()` draw
      per gene), else left unchanged.
    - Int sub-block: each gene independently random-walks by exactly
      `+1` or `-1` (`ctx.rng.next_below(2)`), boundary-clamped the same
      way as the Float sub-block.

    validate_space(): a build-time veto requiring EXACTLY this three-block
    layout, in order (Float, Categorical, Int) -- raises `ValueError`
    for any other space BEFORE any `generate()` call, e.g. a plain
    Float-only BBOB problem (see `main()`'s own demonstration below)."""

    def __init__(self, step=0.3, p_resample=0.2):
        self.step = step
        self.p_resample = p_resample

    def validate_space(self, space):
        kinds = [b["kind"] for b in space]
        if kinds != ["float", "categorical", "int"]:
            raise ValueError(
                "MixedBlockVariation requires a Space(Float, Categorical, "
                f"Int) block order, got {kinds}")

    def generate(self, pop, ctx):
        n = len(pop.individuals)
        k = ctx.space[1]["k"]
        offspring = []
        for _ in range(n):
            floats, cats, ints = pop.individuals[ctx.rng.next_below(n)]
            new_floats = [v + (ctx.rng.next_f64() - 0.5) * 2.0 * self.step
                          for v in floats]
            new_cats = [ctx.rng.next_below(k) if ctx.rng.next_f64() < self.p_resample
                        else c for c in cats]
            new_ints = [i + (1 if ctx.rng.next_below(2) else -1) for i in ints]
            offspring.append((new_floats, new_cats, new_ints))
        return offspring


def main():
    problem = MixedTuningProblem()
    algo = MixedBlockVariation()
    res = algo.run(problem, budget=3000, seed=42, pop_size=20)
    print(f"mixed_space_tuning (oop engine): evals_used={res.evals_used} "
          f"best_f={res.best_f:.6g} gap={res.gap:.6g}")

    # validate_space demo: this algorithm only accepts a Space(Float,
    # Categorical, Int) block order -- pointing it at a plain Float-only
    # BBOB problem is vetoed at build time, before any generate() call.
    veto_ok = False
    try:
        MixedBlockVariation().run(sezgi.bbob(1, 3, 1), budget=10, seed=1)
    except ValueError:
        veto_ok = True
    print(f"mixed_space_tuning (oop engine): validate_space_veto={veto_ok}")

    # ANCHORED (per this project's convention): measured once at this exact
    # config (MixedTuningProblem(), MixedBlockVariation(), budget=3000,
    # seed=42, pop_size=20), then pinned as literal values below.
    assert res.evals_used == 3000
    assert res.best_f == 0.0015112224326435369
    assert res.gap == 0.0015112224326435369
    assert veto_ok


if __name__ == "__main__":
    main()
