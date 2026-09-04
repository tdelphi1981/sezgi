# Algorithm, PopulationAlgorithm, LocalSearch

The engine-hosted, class-first authoring surface: subclass and override
`generate(pop, ctx)` (or, for the two family bases below, only the
narrower `vary()`/`neighbor()` hook) — the loop itself runs *inside* the
Rust engine, not in a Python-owned ask/tell loop.

::: sezgi.algorithm
    options:
      members:
        - Algorithm
        - PopulationAlgorithm
        - LocalSearch
