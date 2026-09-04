# Examples gallery

Every entry below is a real, runnable script under `examples/python/` —
not a snippet reproduced for the docs. Run any of them directly:

```
py-sezgi/.venv/bin/python examples/python/oop/gwo.py
```

This gallery is a curated subset (12 of the full catalog's ~51 scripts):
six representative picks from the 19 parameterized OOP algorithm twins,
the five engine-hosted authoring examples (`examples/python/oop/engine/`),
and the feature-selection data recipe. See `examples/README.md` for the
full 17-algorithm catalog (pure-script / OOP-twin / spec-file triplets)
this gallery draws from.

## Algorithm twins (`sezgi.AskTellAlgorithm`, ask/tell style)

Each of these ports a pure-Python metaheuristic script onto
`sezgi.AskTellAlgorithm` — same update equations, same RNG draw order,
bit-for-bit matched against its pure-script twin
(`test_examples_oop_parity.py`'s 17-pair gate). All four below solve
`sezgi.bbob(1, dim, 1)` (Sphere) at budget 2000, seed 42.

| Example | File | Category | What it teaches |
|---|---|---|---|
| Grey Wolf Optimizer | `examples/python/oop/gwo.py` | Swarm intelligence | Leader-following update with a literal `a = 2 - 2*progress` explore-to-exploit decay schedule (see [Exploration vs exploitation](learn/03-exploration-vs-exploitation.md)) |
| Cuckoo Search | `examples/python/oop/cs.py` | Swarm intelligence (Lévy flight) | Lévy-flight step generation via a closed-form Mantegna (1994) algorithm, stdlib-only (`math.gamma` in place of the Rust component's Lanczos gamma) |
| Harris Hawks Optimization | `examples/python/oop/hho.py` | Swarm intelligence | A multi-phase (exploration/exploitation) update rule with a generational replacer |
| Teaching-Learning-Based Optimization | `examples/python/oop/tlbo.py` | Non-swarm metaheuristic | A population update driven by a "teacher" and pairwise "learner" interactions — no leader/pheromone/velocity metaphor at all |

## Engine-hosted authoring (`sezgi.Algorithm` family, class-first)

These seven (five under `examples/python/oop/engine/`, plus two more from
`examples/python/oop/` demonstrating the same engine-hosted surface) each
demonstrate ONE distinct way to author against the engine-hosted
class surface (`sezgi.Algorithm`/`PopulationAlgorithm`/`LocalSearch`/
`Problem`) — Tutorials 3, 4, and 6 walk through the same hooks these
scripts exercise.

| Example | File | Category | What it teaches |
|---|---|---|---|
| DE/rand/1, `vary()` only | `examples/python/oop/custom_de_variant.py` | Population algorithm | The smallest possible `PopulationAlgorithm` subclass: one method, inherits the base's tournament `select()` |
| DE/rand/2/bin, two donor vectors | `examples/python/oop/engine/custom_de.py` | Population algorithm | A second, more elaborate `vary()` — two difference terms plus binomial crossover — read alongside `custom_de_variant.py` for the mutation/crossover split's two ends |
| Simplified PSO, full `generate()` | `examples/python/oop/engine/custom_pso_variant.py` | Full `Algorithm` override | No `select()`/`vary()` split to lean on, plus per-instance state (velocities, personal/global bests) carried across `generate()` calls on `self` |
| Random 2-opt local search | `examples/python/oop/engine/local_search_2opt.py` | Local search | A `LocalSearch` subclass (`neighbor()` only) over a hand-authored PERMUTATION-typed `Problem` — a small in-file TSP instance, no vendored data file |
| Custom perturbation local search | `examples/python/oop/custom_local_search.py` | Local search | The Float-space counterpart to `local_search_2opt.py` — perturbs real coordinates instead of reversing a tour segment |
| Rastrigin problem authoring | `examples/python/oop/engine/custom_problem_rastrigin.py` | Problem authoring | Defines the PROBLEM side only (pure `math`, no numpy) and solves it with a stock `sezgi.GreyWolfOptimizer` — the other half of the class-first surface from every row above |
| Mixed-space `Problem` + `Algorithm` | `examples/python/oop/engine/mixed_space_tuning.py` | Mixed-space authoring | A `Space(Float, Categorical, Int)` problem paired with an `Algorithm` that varies all three sub-blocks and vetoes any other space via `validate_space()` — see [Tutorial 6](tutorials/06-mixed-space-tuning.md) |

## Data recipes

| Example | File | Category | What it teaches |
|---|---|---|---|
| Feature selection | `examples/python/oop/feature_selection.py` | Data recipe | `sezgi.recipes.FeatureSelection` recovering a known 3-of-8 informative-column mask from a synthetic dataset via `GeneticAlgorithm`'s `Binary` auto-dispatch — see [Tutorial 5](tutorials/05-feature-selection.md) |

## Running these yourself

Every script above is stdlib/numpy-only (no sklearn/scipy anywhere in
this project's Python surface), deterministic under a fixed seed, and
ends by printing an `evals_used=... best_f=... gap=...` line — the same
metrics-line convention this whole site's tutorials use. Each is gated by
its own anchored pytest (`py-sezgi/tests/test_examples_oop_parity.py`,
`test_oop_families.py`, `test_examples_engine.py`, or
`test_feature_selection_example.py`), so a script's printed numbers are
never allowed to silently drift from what is committed here.
