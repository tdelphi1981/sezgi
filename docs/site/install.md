# Install

sezgi's core is Rust; the Python and R packages are thin, class-first
frontends over it. There is no published wheel or CRAN package yet, so
both frontends are built from source.

## Prerequisites

- A [Rust toolchain](https://rustup.rs/) (stable channel; the workspace
  is pinned to the edition declared in the root `Cargo.toml`).
- Python >= 3.9 (Python frontend) and/or R >= 4.0 (R frontend) — install
  whichever frontend(s) you need; neither depends on the other.

## Python (uv-first)

sezgi's own development uses [`uv`](https://docs.astral.sh/uv/) exclusively
for Python tooling — no bare `pip`/`venv`. From the repository root:

```bash
uv venv py-sezgi/.venv
uv pip install --python py-sezgi/.venv/bin/python maturin
cd py-sezgi
uv run --python .venv/bin/python maturin develop --release
```

`maturin develop --release` compiles the Rust extension
(`sezgi._sezgi`, via PyO3) and installs it into the venv in editable
mode. Verify:

```bash
py-sezgi/.venv/bin/python -c "import sezgi; print(sezgi.__version__)"
```

### From source, without `uv`

If you already manage a Python environment another way, the only
requirement is `maturin >= 1.7` (declared in `py-sezgi/pyproject.toml`)
available in that environment; `maturin develop` (or `maturin develop
--release` for an optimized build) builds and installs sezgi into
whichever environment `maturin` is running in. This project's own
development and CI use `uv` exclusively (see above), so that path is the
best-tested one.

## R

From the repository root, the Rust core builds automatically via `cargo`
as part of R's own install step:

```bash
R CMD INSTALL r-sezgi
```

Then:

```r
library(sezgi)
```

`r-sezgi/DESCRIPTION` declares the same Rust-toolchain prerequisite named
above (`SystemRequirements: Cargo (Rust package manager)`); no other
system dependency is required. See [R surface](r.md) for what the R
package exposes and how it corresponds to the Python API documented on
this site.

## Building from source (Rust toolchain, both frontends)

Both frontends compile the *same* Rust workspace (`crates/`) — there is
exactly one native implementation, not two. To build and test the Rust
core alone (no Python or R involved):

```bash
cargo test --workspace --release
```

To build everything and run every language's test suite:

```bash
cargo test --workspace --release                                  # Rust
cd py-sezgi && maturin develop --release && pytest                # Python
R CMD INSTALL --preclean r-sezgi \
  && Rscript -e 'testthat::test_dir("r-sezgi/tests/testthat", package = "sezgi")'  # R
```
