# Tutorial 3: Writing an engine-hosted Algorithm subclass

Every algorithm class in `sezgi.builtins` (`GeneticAlgorithm`,
`DifferentialEvolution`, ...) is a thin skin over a Rust component preset.
`sezgi.Algorithm` is the other side of that same door: subclass it, write
`generate(self, pop, ctx)` in Python, and the Rust engine calls your
method once per generation — population initialization, boundary repair,
replacement, and termination all still happen Rust-side, exactly like a
built-in algorithm. This is different from `sezgi.AskTellAlgorithm`
(covered in the README's ask/tell section), which drives its own
`setup()`/`step()` loop entirely in Python: an `Algorithm` subclass's
hooks run *inside* the engine's own generation loop.

## `generate(self, pop, ctx)` — the one required hook

```python exec="true" source="above"
import sezgi


class SimpleMutation(sezgi.Algorithm):
    """Every generation: perturb every current individual by an
    independent uniform step in [-step, step] per coordinate."""

    def __init__(self, step=0.5):
        self.step = step

    def generate(self, pop, ctx):
        return [[xi + (ctx.rng.next_f64() - 0.5) * 2 * self.step for xi in x]
                for x in pop.individuals]


result = SimpleMutation(step=0.5).run(sezgi.bbob(1, 5, 1), budget=1500, seed=3, pop_size=20)
print(f"evals_used={result.evals_used} best_f={result.best_f:.6g} gap={result.gap:.6g}")
```

`pop.individuals` is a plain Python list, one entry per genotype (bare
`list[float]` here, since `bbob`'s space is a single `Float` block —
Tutorial 6 covers the multi-block/tuple case). `pop.fitness` is a numpy
1-D `float64` array, parallel to `pop.individuals` — unused above, but
essential for anything that needs to compare individuals (Tutorial 4's
`select`/`vary` split does this internally). `generate`'s return value is
the SAME bare/tuple convention: an iterable of x-values, one generation's
worth of offspring — the count is entirely this method's own choice, not
checked against `pop`'s size.

## `ctx.rng` — the determinism knob

`ctx` is a handle over the engine's own per-call context:
`ctx.iteration` (the current generation count), `ctx.space` (one
descriptor dict per space block — `{"kind": "float", "lo": ..., "hi": ...,
"n": ...}` for a `Float` block), and `ctx.rng` — a handle over the
engine's OWN seeded stream for this stage/call. `ctx.rng.next_f64()` draws
a uniform float in `[0, 1)`; `ctx.rng.next_below(n)` draws a uniform
integer in `[0, n)`. Drawing from `ctx.rng` (rather than, say, Python's own
`random` module) is what keeps a `run()` deterministic under a fixed
`seed` — the same seed always produces the same sequence of `ctx.rng`
draws, and therefore the same run, exactly like every built-in Rust
component drawing from its own stage stream:

```python exec="true" source="above"
import sezgi


class SimpleMutation(sezgi.Algorithm):
    def __init__(self, step=0.5):
        self.step = step

    def generate(self, pop, ctx):
        return [[xi + (ctx.rng.next_f64() - 0.5) * 2 * self.step for xi in x]
                for x in pop.individuals]


r1 = SimpleMutation(step=0.5).run(sezgi.bbob(1, 5, 1), budget=1500, seed=3, pop_size=20)
r2 = SimpleMutation(step=0.5).run(sezgi.bbob(1, 5, 1), budget=1500, seed=3, pop_size=20)
print(f"same seed twice: best_x equal = {r1.best_x == r2.best_x}, best_f equal = {r1.best_f == r2.best_f}")
```

## `initialize(self, n, ctx)` — optional population seeding

By default, an `Algorithm` subclass that does not override `initialize`
never has it called at all: the engine's own Rust `init/uniform`
initializer seeds the population instead, with no Python call involved.
Overriding `initialize` replaces that with a Python-authored seeding
strategy — same signature convention as `generate`'s return value (an
iterable of `n` x-values). Below, `CornerStart` seeds every individual
near one corner of the search box instead of uniformly across it,
demonstrating that this hook genuinely controls where the search begins:

```python exec="true" source="above"
import sezgi


class CornerStart(sezgi.Algorithm):
    def __init__(self, step=0.5):
        self.step = step

    def initialize(self, n, ctx):
        block = ctx.space[0]  # this problem's one Float block
        dim = block["n"]
        lo, hi = block["lo"], block["hi"]
        near = lo + 0.05 * (hi - lo)     # 5% in from the lower bound
        span = 0.1 * (hi - lo)           # a narrow band near that corner
        return [[near + ctx.rng.next_f64() * span for _ in range(dim)]
                for _ in range(n)]

    def generate(self, pop, ctx):
        return [[xi + (ctx.rng.next_f64() - 0.5) * 2 * self.step for xi in x]
                for x in pop.individuals]


result = CornerStart(step=0.5).run(sezgi.bbob(1, 5, 1), budget=1500, seed=3, pop_size=20)
print(f"evals_used={result.evals_used} best_f={result.best_f:.6g} gap={result.gap:.6g}")
print(f"best_x = {[round(v, 4) for v in result.best_x]}")
```

Whether `initialize` was actually overridden is decided by comparing the
resolved method against `Algorithm.initialize` itself (function identity,
not a sentinel return value) — so a subclass that legitimately wants
`initialize` to do something unusual (including returning an empty
sequence) is never mistaken for "did not override it".

## Spec persistence: the one honest gap

An engine-hosted Python algorithm is a live callback registered into the
Rust engine's per-call registry — it is process-local. Unlike a purely
Rust-component spec (`sezgi.presets.*`, used by `sezgi.solve()` and every
benchmarking/spec-file path), there is no `.toml`/`.json` wire format a
Python `generate`/`initialize` callback can be serialized into. A
`sezgi.Algorithm` subclass instance cannot be saved to and reloaded from a
spec file the way a preset-backed run can — an honest degrade, not a gap
this milestone silently papers over.

## Next

- [PopulationAlgorithm and LocalSearch](04-population-and-local-search.md)
  — two family bases that compose `generate()` for you from smaller,
  more specific hooks (`select`/`vary`, or `neighbor`/`accept`).
