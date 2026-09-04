# Built-in algorithm classes (29)

One class per built-in algorithm — 26 uniformly generated (table-driven
from `crates/components/src/presets.rs`), plus `GeneticAlgorithm`
(auto-dispatch over 5 space-typed presets) and `DifferentialEvolution`
(dispatch over 3 mutation variants), both hand-written, plus `NSGA2` (a
thin skin over `sezgi.mo.nsga2`, not a preset). Every class but `NSGA2`
shares the same shape: `__init__(pop_size=..., **preset_kwargs)`,
`run(problem, budget, seed=0, run_id=0, log_dir=None) -> SolveResult`.
`NSGA2` is the one exception — its own docstring describes the
multi-objective result dictionary its `run()` returns instead.

These classes are created at runtime (`type()`), not via `class`
statements, so this page renders under mkdocstrings'
`force_inspection: true` handler option, which mkdocstrings requires for
classes it cannot discover through normal static inspection.

::: sezgi.builtins
    options:
      members:
        - EvolutionStrategy
        - ParticleSwarm
        - SimulatedAnnealing
        - SHADE
        - LSHADE
        - CMAES
        - CMAESIpop
        - NelderMead
        - RandomSearch
        - GreyWolfOptimizer
        - WhaleOptimization
        - HarmonySearch
        - CuckooSearch
        - GrasshopperOptimization
        - SineCosineAlgorithm
        - JAYA
        - MothFlameOptimization
        - SalpSwarm
        - FireflyAlgorithm
        - BatAlgorithm
        - FlowerPollination
        - TLBO
        - HarrisHawks
        - AntLion
        - ArtificialBeeColony
        - GravitationalSearch
        - GeneticAlgorithm
        - DifferentialEvolution
        - NSGA2
