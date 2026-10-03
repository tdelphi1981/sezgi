# sezgi 0.1.6

* Mixed-space hybrid auto-dispatch widened to 17 presets: `ParticleSwarm` and
  `MothFlameOptimization` join the list (since 0.1.6), and
  `DifferentialEvolution(variant = "jde")` is now allowed on mixed spaces
  (float blocks keep the preset's own generator, GA variation on discrete
  blocks; a space with no float block is all GA).
* Engine: `gen/pso`, `gen/de-jde` and `gen/mfo` now own their state (PSO
  velocity/pbest, jDE per-individual F/CR, MFO flame memory) and are on the
  `gen/compound` self-contained allow-list. The companion components
  (`replace/pso-commit`, `adapter/jde-commit`, `adapter/mfo-flame-update`) are
  kept for spec compatibility; their state duties moved into the generators.
  Single-block preset runs are bit-identical to 0.1.5.
* Still excluded from hybrid dispatch (structural): `shade`/`lshade`, `cmaes`
  (+ ipop), `abc`, `sa`, `harmony_search`, `nelder_mead`, `hho`.

# sezgi 0.1.5

* Mixed-space hybrid auto-dispatch widened to 15 presets:
  `GravitationalSearch` and `BatAlgorithm` join the list (float blocks keep
  the preset's own generator, GA variation on discrete blocks; a space with
  no float block is all GA).
* Engine (`gen/compound`): every block now owns a nested blackboard stored in
  the parent blackboard under `cmp{i}/bb`, so self-contained stateful
  generators keep per-block state. An allow-list
  (`SELF_CONTAINED_COMPOUND_KINDS`: `gen/gsa`, `gen/ba`) admits them;
  other stateful generators (PSO, CMA-ES, SHADE, ...) stay rejected.
  Existing stateless compound specs are bit-identical.

# sezgi 0.1.4

* Mixed-space hybrid auto-dispatch widened to 13 presets: `AntLion` and
  `EvolutionStrategy` join the 0.1.3 list (float blocks keep the preset's
  own generator, GA variation on discrete blocks; a space with no float
  block is all GA). `SimulatedAnnealing` stays ineligible: its fixed
  population of 1 cannot host the GA discrete sub-generators.
* Engine: the post-adapter genotype shape sweep now runs only for stages
  that have an adapter.
* New R guard tests for `.sz_hybrid_wrap_stages` (malformed preset specs).
* The ineligible-preset error now lists the real preset names
  (`cuckoo_search`, `alo`, `es_mu_plus_lambda`) instead of "cuckoo-search".

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
