# Suppresses "no visible binding for global variable 'self'/'private'" R CMD
# check NOTEs -- the standard R6 false positive: `self`/`private` are bound
# by R6's own `$new()` machinery at instantiation time, not visible to
# static analysis of a `public = list(...)` method body (e.g.
# `.sz_make_preset_class()`'s `run_fn` in `R/builtins.R`, which reads
# `self$pop_size`/`self$preset_kwargs`).
utils::globalVariables(c("self", "private"))
