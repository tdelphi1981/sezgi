# Presets

The 34 preset builders backing every built-in algorithm class (`sezgi.presets.X(pop_size, budget, ...)`, one per `crates/components/src/presets.rs` builder). Every built-in class in [Built-in algorithm classes](builtins.md) delegates to exactly one (or, for `GeneticAlgorithm`/`DifferentialEvolution`, one of several) of these -- this page is the compat-internals view of the same catalog. Rendered from the native `sezgi._sezgi.preset_*` functions directly (the wrapping `_preset()` closure in `sezgi/__init__.py` only rescues `__doc__`/`__name__`/signature via `functools.wraps`; the docstring itself lives on the native function).

::: sezgi._sezgi.preset_de_rand_1
    options:
      heading: "presets.de_rand_1"

::: sezgi._sezgi.preset_de_best_1
    options:
      heading: "presets.de_best_1"

::: sezgi._sezgi.preset_jde
    options:
      heading: "presets.jde"

::: sezgi._sezgi.preset_shade
    options:
      heading: "presets.shade"

::: sezgi._sezgi.preset_lshade
    options:
      heading: "presets.lshade"

::: sezgi._sezgi.preset_ga_real
    options:
      heading: "presets.ga_real"

::: sezgi._sezgi.preset_pso
    options:
      heading: "presets.pso"

::: sezgi._sezgi.preset_gwo
    options:
      heading: "presets.gwo"

::: sezgi._sezgi.preset_woa
    options:
      heading: "presets.woa"

::: sezgi._sezgi.preset_harmony_search
    options:
      heading: "presets.harmony_search"

::: sezgi._sezgi.preset_cuckoo_search
    options:
      heading: "presets.cuckoo_search"

::: sezgi._sezgi.preset_goa
    options:
      heading: "presets.goa"

::: sezgi._sezgi.preset_sca
    options:
      heading: "presets.sca"

::: sezgi._sezgi.preset_jaya
    options:
      heading: "presets.jaya"

::: sezgi._sezgi.preset_mfo
    options:
      heading: "presets.mfo"

::: sezgi._sezgi.preset_ssa
    options:
      heading: "presets.ssa"

::: sezgi._sezgi.preset_firefly
    options:
      heading: "presets.firefly"

::: sezgi._sezgi.preset_bat
    options:
      heading: "presets.bat"

::: sezgi._sezgi.preset_fpa
    options:
      heading: "presets.fpa"

::: sezgi._sezgi.preset_tlbo
    options:
      heading: "presets.tlbo"

::: sezgi._sezgi.preset_hho
    options:
      heading: "presets.hho"

::: sezgi._sezgi.preset_alo
    options:
      heading: "presets.alo"

::: sezgi._sezgi.preset_abc
    options:
      heading: "presets.abc"

::: sezgi._sezgi.preset_gsa
    options:
      heading: "presets.gsa"

::: sezgi._sezgi.preset_sa
    options:
      heading: "presets.sa"

::: sezgi._sezgi.preset_random_search
    options:
      heading: "presets.random_search"

::: sezgi._sezgi.preset_nelder_mead
    options:
      heading: "presets.nelder_mead"

::: sezgi._sezgi.preset_cmaes
    options:
      heading: "presets.cmaes"

::: sezgi._sezgi.preset_cmaes_ipop
    options:
      heading: "presets.cmaes_ipop"

::: sezgi._sezgi.preset_es_mu_plus_lambda
    options:
      heading: "presets.es_mu_plus_lambda"

::: sezgi._sezgi.preset_ga_perm
    options:
      heading: "presets.ga_perm"

::: sezgi._sezgi.preset_ga_bin
    options:
      heading: "presets.ga_bin"

::: sezgi._sezgi.preset_ga_int
    options:
      heading: "presets.ga_int"

::: sezgi._sezgi.preset_ga_cat
    options:
      heading: "presets.ga_cat"
