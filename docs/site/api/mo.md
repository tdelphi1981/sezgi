# Multi-objective (mo)

`sezgi.mo.*`: NSGA-II over the built-in ZDT/DTLZ/WFG problem families,
plus hypervolume, IGD, an analytic Pareto-front sampler, and
`sezgi-moa`-format archive logging/reading. `sezgi.NSGA2`
([Built-in algorithm classes](builtins.md)) is a thin class-first skin
over `mo.nsga2` below.

::: sezgi._mo_nsga2
    options:
      heading: "mo.nsga2"

::: sezgi._sezgi.mo_hypervolume_2d
    options:
      heading: "mo.hypervolume_2d"

::: sezgi._sezgi.mo_hypervolume
    options:
      heading: "mo.hypervolume"

::: sezgi._sezgi.mo_igd
    options:
      heading: "mo.igd"

::: sezgi._mo_pareto_front
    options:
      heading: "mo.pareto_front"

::: sezgi._sezgi.mo_evaluate
    options:
      heading: "mo.evaluate"

::: sezgi._sezgi.mo_evaluate_constraints
    options:
      heading: "mo.evaluate_constraints"

::: sezgi._sezgi.mo_read_moa
    options:
      heading: "mo.read_moa"
