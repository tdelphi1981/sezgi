---
name: Bug report
about: Report unexpected behavior, a crash, or a parity/determinism break
title: "[BUG] "
labels: bug
---

## Summary

A clear, concise description of what went wrong.

## Environment

- Frontend: Python / R / Rust core directly
- sezgi version (`sezgi.__version__` / `sezgi_version()` / crate version):
- OS and architecture:
- Rust toolchain version (`rustc --version`), if built from source:
- Python or R version, if applicable:

## Steps to reproduce

A minimal, self-contained script or command that reproduces the issue.
For a determinism or cross-language parity issue, include the exact
`(spec, problem, seed, budget)` used.

```python
# minimal reproduction here
```

## Expected behavior

What you expected to happen (e.g. a specific value, or "no panic").

## Actual behavior

What actually happened — the exact output, error message, or stack
trace.

## Additional context

Anything else relevant: whether this reproduces on the main checkout as
well as your worktree, whether it is new in a specific version, related
issues, etc.
