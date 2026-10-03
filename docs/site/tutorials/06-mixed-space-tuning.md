# Tutorial 6: Mixed-space tuning

Real hyperparameter tuning rarely lives in one block kind: a learning
rate is continuous, an optimizer choice is categorical, a layer count is
an integer. `sezgi.recipes.MixedTuning` is the general "tune anything"
door — it binds a caller-supplied `objective(x) -> float` over ANY
declared `sezgi.Space`, including a genuinely mixed one.

## `MixedTuning(space, objective)`

Unlike `FeatureSelection` (Tutorial 5), `MixedTuning` makes zero
dataset-specific assumptions — it is `evaluate = objective`, plus the
usual `space()` accessor:

```python exec="true" source="above"
import sezgi
from sezgi.recipes import MixedTuning

_MODE_PENALTY = (0.0, 0.5, 1.5)


def objective(x):
    """A toy tuning objective over THREE blocks: two continuous knobs,
    one discrete 'mode' choice, one integer knob -- x is a 3-tuple
    (floats, cats, ints), one entry per block, in space() order."""
    floats, cats, ints = x
    return (sum(v * v for v in floats)          # minimized at 0.0
            + _MODE_PENALTY[cats[0]]             # minimized by mode 0
            + 0.1 * (ints[0] - 3) ** 2)          # minimized at int=3

space = sezgi.Space(
    sezgi.Float(-5.0, 5.0, 2),    # two continuous knobs
    sezgi.Categorical(3, 1),      # one discrete mode choice
    sezgi.Int(1, 5, 1),           # one integer knob
)
problem = MixedTuning(space, objective)
print(f"space blocks (declared order): {[type(b).__name__ for b in space.blocks]}")
```

`x`'s shape follows `sezgi.Problem`'s own genotype conversion table: a
multi-block space hands `evaluate`/`objective` a tuple of per-block
values, in `space()`'s declared order — here, `(list[float], list[int],
list[int])` (a `Categorical` block converts to indices, same as `Int`).

## Running a mixed space: hybrid presets

`GeneticAlgorithm` auto-dispatches only over a SINGLE-kind space
(all-Float, all-Binary, ...) — pointing it at a genuinely mixed space
raises `NotImplementedError` before any run starts. Since 0.1.3, however,
17 presets (`DifferentialEvolution` rand1/best1/jde (jde since 0.1.6), `GreyWolfOptimizer`,
`WhaleOptimization`, `SineCosineAlgorithm`, `JAYA`,
`GrasshopperOptimization`, `SalpSwarm`, `FireflyAlgorithm`,
`FlowerPollination`, `TLBO`, `CuckooSearch`, plus `AntLion` and
`EvolutionStrategy` since 0.1.4, plus `GravitationalSearch` and
`BatAlgorithm` since 0.1.5, plus `ParticleSwarm` and
`MothFlameOptimization` since 0.1.6) run on a mixed space directly
as a **hybrid** through `gen/compound`: the preset's own generator varies
the float blocks, and GA variation (`gen/ga-bin` / `gen/ga-int` /
`gen/ga-cat` / `gen/ga-perm`, default parameters) varies every other block;
parent selection and replacement follow the preset's own pipeline. A space
with no float block gets zero preset-specific variation (all variation is
GA). Stateful presets that keep structural state (CMA-ES, SHADE, ...) still raise
`NotImplementedError` naming why.

```python
import sezgi
from sezgi.recipes import MixedTuning

# Float block + Binary block: DE on the floats, GA variation on the bits.
space = sezgi.Space(sezgi.Float(-5.0, 5.0, 3), sezgi.Binary(6))
problem = MixedTuning(
    space,
    lambda x: sum(v * v for v in x[0]) + sum(1 for b in x[1] if not b),
)
result = sezgi.DifferentialEvolution().run(problem, budget=2000, seed=7)
print(f"hybrid DE best_f={result.best_f:.4g}")
```

Alternatively, a hand-authored `sezgi.Algorithm` can vary all three
sub-blocks of the tuning problem above with fully custom logic. This
tutorial takes that class-first path:

```python exec="true" source="above"
import sezgi
from sezgi.recipes import MixedTuning

_MODE_PENALTY = (0.0, 0.5, 1.5)


def objective(x):
    floats, cats, ints = x
    return (sum(v * v for v in floats)
            + _MODE_PENALTY[cats[0]]
            + 0.1 * (ints[0] - 3) ** 2)


class MixedBlockVariation(sezgi.Algorithm):
    """generate(): breeds each offspring from ONE uniformly-random parent,
    varying the three sub-blocks independently -- Float steps by a random
    +-step, Categorical resamples with probability p_resample, Int
    random-walks by +-1. validate_space(): a build-time veto requiring
    EXACTLY a (Float, Categorical, Int) block order."""

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
        k = ctx.space[1]["k"]  # the Categorical block's own k
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


space = sezgi.Space(sezgi.Float(-5.0, 5.0, 2), sezgi.Categorical(3, 1), sezgi.Int(1, 5, 1))
problem = MixedTuning(space, objective)

result = MixedBlockVariation().run(problem, budget=3000, seed=42, pop_size=20)
print(f"evals_used={result.evals_used} best_f={result.best_f:.6g}")
print(f"distance from the theoretical minimum (0.0): {result.best_f - 0.0:.6g}")
```

`MixedTuning` has no `optimum` parameter — its `evaluate` delegates purely
to `objective`, with no place to plumb a known minimum through, so
`result.gap` stays `None` for a `MixedTuning`-wrapped run (unlike Tutorial
2's `Rastrigin`, which overrides `optimum()` directly on a hand-authored
`Problem`). This objective's theoretical minimum (`0.0`, at the origin,
mode 0, int=3) is known by construction here, so the distance above is
computed by hand instead.

## `validate_space()`: a build-time veto

`MixedBlockVariation.validate_space` rejects any space that is not
exactly `(Float, Categorical, Int)`, BEFORE any `generate()` call —
pointing it at a plain Float-only BBOB problem is vetoed at build time:

```python exec="true" source="above"
import sezgi


class MixedBlockVariation(sezgi.Algorithm):
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
        return pop.individuals  # unreachable in this demo


veto_ok = False
try:
    MixedBlockVariation().run(sezgi.bbob(1, 3, 1), budget=10, seed=1)
except ValueError:
    veto_ok = True
print(f"validate_space_veto={veto_ok}")
```

## Next

- [Multi-objective optimization with NSGA-II](07-nsga2-multi-objective.md)
  — the one built-in algorithm with its own result shape.
