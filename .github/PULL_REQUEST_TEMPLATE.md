## Summary

What does this change do, and why?

## Gates run

Check every gate this change actually requires, and report the result
count for each (see `CONTRIBUTING.md` for the exact commands):

- [ ] `cargo test --workspace --release` — result:
- [ ] `pytest` (py-sezgi) — result:
- [ ] R testthat gate (`R CMD INSTALL --preclean r-sezgi` + `testthat::test_dir(...)`) — result:
- [ ] `mkdocs build --strict` (only if `docs/site/`, `mkdocs.yml`, or a
      rendered docstring/example changed) — result:
- [ ] `cargo clippy` (both workspaces, if Rust source changed) —
      baseline unchanged / new warnings explained below

## Scope

- [ ] This PR does not modify any existing test file's assertions (only
      adds new test files), OR: it does, and the reason is explained
      below (existing-test edits are exceptional, not routine).
- [ ] If this touches `crates/` or `r-sezgi/src/rust`: cross-language
      parity is preserved (no behavior change to a shared algorithm
      without a corresponding parity-test update in the SAME PR).
- [ ] `docs/DECISIONS.md` updated, if this makes or changes a
      non-trivial design decision or closes/narrows a recorded
      deferral. Not applicable otherwise.

## Related issues

Closes #

## Additional context

Anything a reviewer needs to know that isn't obvious from the diff.
