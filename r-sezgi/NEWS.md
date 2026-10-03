# sezgi 0.1.3

* `gen/compound` now accepts any registered sub-generator that is eligible
  (stateless, pop-to-pop, no internal evaluation, no nesting) instead of a
  fixed GA-only table; ineligible kinds are rejected with the reason.
* Mixed-space auto-dispatch for 11 presets: `DifferentialEvolution`
  (rand_1/best_1), `GreyWolfOptimizer`, `WhaleOptimization`,
  `SineCosineAlgorithm`, `JAYA`, `GrasshopperOptimization`, `SalpSwarm`,
  `FireflyAlgorithm`, `FlowerPollination`, `TLBO`, `CuckooSearch`. The
  result is a hybrid: the preset's own generator on float blocks, GA
  variation on binary/int/categorical/permutation blocks. A space with no
  float block gets zero preset-specific variation (all variation is GA).
* New shape guards: the evaluator panics with "genotype shape mismatch:"
  on a mis-shaped genotype, and the engine re-checks genotype shape after
  the adapter step.
* The multi-block engine error hint now mentions eligible non-GA
  generators inside `gen/compound`.
* Presets that cannot ride inside `gen/compound` (stateful, one-offspring,
  internal-evaluation) raise an error naming why on a mixed space.

# sezgi 0.1.2

* `GeneticAlgorithm` now auto-dispatches multi-block single-kind spaces
  (e.g. several Float blocks) through `gen/compound`; mixed-kind spaces
  still raise. Other flat presets run on a multi-block space now fail fast
  with an engine genotype-shape error instead of silently corrupting.

# sezgi 0.1.1

* Provenance and licensing documentation: corrected TSPLIB status note,
  added inst/COPYRIGHTS and Copyright field, rescoped LICENSE.note, added
  THIRD-PARTY-NOTICES; no functional changes.

# sezgi 0.1.0

* Initial release. sezgi ships 29 built-in algorithm classes over one
  deterministic Rust engine, with Python and R frontends that are
  bit-exact against each other for shared algorithms.
* BBOB and CEC 2014/2017/2022 benchmark suites.
* NSGA-II multi-objective optimization over ZDT/DTLZ/WFG.
* IOH-format logging with ECDF/COCO export.
* A statistical-comparison toolkit.
* A structural-bias scanner.
