# Solve / compat internals (compat)

!!! note "This page documents compat internals"
    `solve()`, `run_experiment()`, and the preset catalog (`sezgi.presets.*`,
    on the [Presets](presets.md) page) are the spec-file-driven internals
    every built-in class's `.run()` is itself implemented on top of. They
    remain fully working, public, and documented here — but the
    class-first surface ([Algorithm](algorithm.md),
    [built-in classes](builtins.md)) is the primary, recommended API for
    everyday use. Reach for this page when you need a raw `AlgorithmSpec`
    (e.g. TOML spec files, `run_experiment`, or R interop).

::: sezgi.solve

::: sezgi.run_experiment

::: sezgi.read_ioh_records

::: sezgi.ecdf

::: sezgi.coco_export

::: sezgi.results_matrix

::: sezgi.per_budget_packages
