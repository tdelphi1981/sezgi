# Tutorial 4: PopulationAlgorithm and LocalSearch bases

Tutorial 3 wrote a full `generate(self, pop, ctx)` from scratch. Most
real algorithm variants only want to change ONE piece of a well-known
shape — how offspring are bred, or how one point is perturbed — while
keeping everything else the base already provides. `PopulationAlgorithm`
and `LocalSearch` are two family bases that compose `generate()` for you
from smaller, more specific hooks.

## `PopulationAlgorithm`: override only `vary()`

`PopulationAlgorithm` (a subclass of `Algorithm`) implements `generate()`
by calling `self.select(pop, len(pop.individuals), ctx)` (default:
k-fold binary tournament, tournament size 2, minimization) and then
`self.vary(parents, ctx)` — a subclass need only override `vary`, the
turn-a-parent-pool-into-offspring step, which is REQUIRED:

```python exec="true" source="above"
import sezgi


class CustomDERandOne(sezgi.PopulationAlgorithm):
    """DE/rand/1-shaped vary(): for each parent slot, draws THREE donor
    indices and combines them as r1 + f * (r2 - r3) -- the classic
    DE/rand/1 mutation, applied over the tournament-selected parent pool
    this base's default select() already built. `f`: differential weight."""

    def __init__(self, f=0.5):
        self.f = f

    def vary(self, parents, ctx):
        n = len(parents)
        offspring = []
        for _ in range(n):
            r1 = ctx.rng.next_below(n)
            r2 = ctx.rng.next_below(n)
            r3 = ctx.rng.next_below(n)
            a, b, c = parents[r1], parents[r2], parents[r3]
            offspring.append([ai + self.f * (bi - ci) for ai, bi, ci in zip(a, b, c)])
        return offspring


result = CustomDERandOne().run(sezgi.bbob(1, 10, 1), budget=2000, seed=42, pop_size=20)
print(f"evals_used={result.evals_used} best_f={result.best_f:.6g} gap={result.gap:.6g}")
```

`select`'s default tournament is still available to override too, but
`vary` alone is enough to author a full DE/GA/ES-flavored variant — this
is the same pattern `examples/python/oop/custom_de_variant.py` and
`examples/python/oop/engine/custom_de.py` (a DE/rand/2/bin variant, one
step more elaborate) demonstrate as full example scripts.

## `LocalSearch`: override only `neighbor()`

`LocalSearch` is the single-trajectory counterpart: a subclass implements
`neighbor(self, x, ctx)` (REQUIRED — propose ONE perturbed candidate from
the current point) and may override `accept(self, f_old, f_new, ctx)`
(default: greedy, `f_new <= f_old`). It is designed for `pop_size=1` —
`run()` defaults to and enforces it:

```python exec="true" source="above"
import sezgi


class CustomPerturbationSearch(sezgi.LocalSearch):
    """neighbor(): perturbs every coordinate of x by an independent
    uniform draw in [-step, step]. The base's default greedy accept()
    tracks whichever of {current, neighbor} is better."""

    def __init__(self, step=0.3):
        self.step = step

    def neighbor(self, x, ctx):
        return [xi + (ctx.rng.next_f64() - 0.5) * 2 * self.step for xi in x]


result = CustomPerturbationSearch().run(sezgi.bbob(1, 10, 1), budget=2000, seed=42)
print(f"evals_used={result.evals_used} best_f={result.best_f:.6g} gap={result.gap:.6g}")
```

## `accept()`'s real job (read this before assuming SA-style semantics)

`LocalSearch` runs over the engine's default `replace/mu-plus-lambda`
replacer, which at `pop_size=1` with exactly one offspring per call
ALWAYS keeps the strictly-better (or, on an exact tie, the OLD) of
`{current, neighbor}` — this is a structural guarantee of the replacer
itself, not a decision `accept()` makes. Two consequences:

1. `accept()` can never actually be called with an `f_new` objectively
   worse than `f_old` — the replacer has already filtered that case out
   before Python sees the population again. This base's `accept()`
   therefore **cannot implement true simulated-annealing-style "sometimes
   accept a worse move" semantics** — that would require intercepting the
   replacer's own decision, which this design deliberately does not
   attempt.
2. The default `accept` (`f_new <= f_old`) is a tautology over every pair
   this base can ever present to it — always true, faithfully mirroring
   (not fighting) what the replacer already enforces.

What `accept()` DOES control: whether this base's own internally-tracked
anchor (the point the NEXT `neighbor()` call is proposed from) advances to
match the engine's already-decided outcome, or stays put — i.e. `accept()`
governs this base's own search trajectory, never population membership
(which the replacer alone decides at `pop_size=1`). A stricter-than-default
`accept` (e.g. rejecting an exact tie) makes the base re-propose from the
same previous anchor again on the next call, even though the reported
population has already moved.

## Next

- [Feature selection recipe](05-feature-selection.md) — a `Problem`
  subclass solved by a plain `GeneticAlgorithm`, no custom `Algorithm`
  needed for this one.
- [Mixed-space tuning](06-mixed-space-tuning.md) — a `Problem` AND an
  `Algorithm` authored together, when the search space mixes block kinds.
