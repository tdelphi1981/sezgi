# Problem ABC and native conversion

`sezgi.Problem` is the subclassable ABC for authoring your own search
problem; `as_native_problem` converts any `Problem` subclass instance (or
an already-native handle, unchanged) into the native handle every entry
point (`solve`, `Algorithm.run`, `EvalSession.for_problem`, ...) accepts.

::: sezgi.problem
    options:
      members:
        - Problem
        - as_native_problem
