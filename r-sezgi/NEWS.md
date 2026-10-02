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
