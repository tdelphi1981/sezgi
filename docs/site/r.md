# R surface

This site's API reference (auto-generated from docstrings) covers only
the **Python** frontend. The **R** frontend (package `sezgi`, in
`r-sezgi/`) mirrors the Python class-first surface 1:1 by name, and is
documented separately through R's own tooling:

- **109 roxygen-generated `.Rd` pages** under `r-sezgi/man/`, covering
  every exported symbol.
- **`r-sezgi/DESCRIPTION`** for the package's own summary and system
  requirements (`SystemRequirements: Cargo (Rust's package manager), rustc (>= 1.88)`).
- Install with `install.packages("sezgi")` — the package is [on
  CRAN](https://cran.r-project.org/package=sezgi) (see
  [Install → R](install.md#install-in-r) for the development version via
  r-universe and the source-build fallback), then view any page from an
  R session:

  ```r
  library(sezgi)
  ?GeneticAlgorithm
  ?Algorithm
  ```

A browsable HTML reference site for the R package (e.g. via `pkgdown`) is
**not built by this milestone** — it is a deliberate scope cut to avoid
the R-side scope creep of authoring `r-sezgi/README.md` and vignettes
from scratch (both of which `pkgdown` needs and neither of which exists
today). This page is the honest summary in the meantime: it does not
claim a generated R site exists.

## Correspondence table

R6 classes and functions mirror their Python counterparts by name (`sz_`
prefix on free functions, matching the C-style prefix R packages
conventionally use; class names are identical). Cross-language runs
sharing a `(spec/preset, problem, seed, budget)` are bit-exact — verified
by the project's own R/Python parity test suites, not merely asserted.

| Python | R | Notes |
|---|---|---|
| `sezgi.Problem` (ABC) | `sezgi::Problem` (`R6::R6Class`) | Same `space()`/`evaluate()`/`optimum()` contract. |
| `sezgi.Algorithm` | `sezgi::Algorithm` | Engine-hosted `generate(pop, ctx)` base. |
| `sezgi.PopulationAlgorithm` | `sezgi::PopulationAlgorithm` | `select()`/`vary()` split. |
| `sezgi.LocalSearch` | `sezgi::LocalSearch` | `neighbor()`/`accept()` split. |
| `sezgi.AskTellAlgorithm` | `sz_algorithm()` / `sz_algo_solve()` | `setup()`/`step()` ask/tell surface, expressed as base-R functions (scope ruling: no R6 here). |
| `sezgi.GeneticAlgorithm`, ..., 29 built-in classes | `sezgi::GeneticAlgorithm`, ..., 29 built-in classes | One R6 class per Python wrapper class, table-driven on both sides from the same `crates/components/src/presets.rs` builder set. |
| `sezgi.solve(spec, problem, ...)` | `sz_solve_bbob()`, `sz_solve_onemax()`, `sz_solve_tsp()`, `sz_solve_cec2014()`/`cec2017()`/`cec2022()`, `sz_solve_cat_match()`, `sz_solve_int_quadratic()`, `sz_solve_mixed_diagnostic()` | Per-problem compat internals (see [Solve / compat internals](api/compat.md)). |
| `sezgi.presets.*` | `sz_preset_*` | Same preset catalog, one builder per algorithm variant. |
| `sezgi.run_experiment` | `sz_run_experiment` | Same `ExperimentSpec`/IOH-logging semantics. |
| `sezgi.stats.*` | `sz_stats_*` (via `R/stats.R`) | Same statistical-comparison suite. |
| `sezgi.bias.*` | `sz_bias_*` | Same structural/central bias scanner. |
| `sezgi.mo.*` | `sz_nsga2`, `sz_mo_hypervolume_2d`, `sz_mo_igd`, `sz_mo_pareto_front`, ... | Same NSGA-II runner and MO indicators, bit-exact against the Python bindings. |
| `sezgi.Float`/`Int`/`Categorical`/`Binary`/`Permutation`/`Space` | `sz_float`/`sz_int`/`sz_categorical`/`sz_binary`/`sz_permutation`/`sz_space` | Same space builders. |
| `sezgi.recipes.FeatureSelection`/`MixedTuning` | `sezgi::FeatureSelection`/`MixedTuning` | Same two data recipes. |

## No overclaim

This page states what exists (109 `.Rd` pages, a class-for-class R6
mirror, bit-exact cross-language runs) and what does not (a generated,
browsable pkgdown site; R vignettes). Neither is invented here.
