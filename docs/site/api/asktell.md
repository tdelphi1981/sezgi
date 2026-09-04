# AskTellAlgorithm and AlgoContext

The Python-owned ask/tell authoring surface, for algorithms more
naturally expressed as a loop over an `EvalSession` than as an
engine-hosted `generate()` hook. `sezgi.algo.Algorithm` is kept as a
compat alias for `AskTellAlgorithm` under its pre-rename name.

::: sezgi.algo
    options:
      members:
        - AskTellAlgorithm
        - AlgoContext
        - SolveResult
        - BudgetExhausted
        - bbob_records
