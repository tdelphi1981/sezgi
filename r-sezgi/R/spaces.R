# Search-space builders (M4-2 Task 1) -- the R mirror of py-sezgi's
# `sezgi.Float`/`Int`/`Categorical`/`Binary`/`Permutation`/`Space`
# (`py-sezgi/python/sezgi/spaces.py`, M4-1 Task 1). Signatures and
# validation semantics mirror that file EXACTLY, including what it does
# NOT do: `spaces.py`'s five block dataclasses are frozen, unvalidated
# value objects -- no `__post_init__`, no bounds/whole-number/`k >= 2`
# check anywhere in the constructor. The ONLY validation `spaces.py`
# performs is in `Space.__init__` (at least one block; every argument must
# be one of the five block types) -- reproduced below in `sz_space()`.
# Bound/arity checking for an actual space happens only once it reaches
# the Rust core (`SearchSpace::new`, `crates/core/src/space.rs`), which
# Task 1 does not call (no space-consuming Rust binding exists yet -- see
# Task 2). Field names mirror `Block` (`crates/core/src/space.rs:13-19`)
# 1:1: `Float { lo, hi, n }`, `Int { lo, hi, n }`, `Categorical { k, n }`,
# `Permutation { n }`, `Binary { n }`.
#
# `.sz_space_to_blocks(space)` is the R analogue of `Space._to_json()`'s
# per-block dict shape (`spaces.py`'s own `_to_dict()` methods), returning
# a plain R `list` (not a JSON string) -- one entry per block, each a
# named list with a `type` field (matching `Block`'s own
# `#[serde(tag = "type", rename_all = "snake_case")]`, the same key
# `_to_dict()` uses) plus that block's numeric fields. Numeric coercion
# (`as.double()`/`as.integer()`) happens HERE, at serialization time --
# exactly where `_to_dict()` coerces (`float(self.lo)`, `int(self.n)`) --
# not at construction, matching the frozen dataclasses' own lack of
# construction-time coercion.

# ---- block builders ---------------------------------------------------

#' A continuous block: `n` coordinates, each in `[lo, hi]`.
#'
#' Mirrors `sezgi.Float` (`py-sezgi/python/sezgi/spaces.py`) exactly: a
#' plain value object, fields stored verbatim with no construction-time
#' validation (bounds are enforced only once the space reaches the Rust
#' core, e.g. `SearchSpace::new`, `crates/core/src/space.rs`).
#'
#' @param lo Numeric scalar, the lower bound.
#' @param hi Numeric scalar, the upper bound.
#' @param n Numeric/integer scalar, the number of coordinates. Required --
#'   `spaces.py`'s `Float.n` has no default either.
#' @returns An object of class `c("sz_block_float", "sz_block")` with
#'   elements `lo`, `hi`, `n`.
#' @export
sz_float <- function(lo, hi, n) {
  structure(list(lo = lo, hi = hi, n = n), class = c("sz_block_float", "sz_block"))
}

#' An integer block: `n` coordinates, each in `[lo, hi]` (inclusive).
#'
#' Mirrors `sezgi.Int` (`py-sezgi/python/sezgi/spaces.py`) exactly -- see
#' [sz_float()]'s doc for the no-construction-time-validation note.
#'
#' @inheritParams sz_float
#' @returns An object of class `c("sz_block_int", "sz_block")` with
#'   elements `lo`, `hi`, `n`.
#' @export
sz_int <- function(lo, hi, n) {
  structure(list(lo = lo, hi = hi, n = n), class = c("sz_block_int", "sz_block"))
}

#' A categorical block: `n` genes, each a category index in `0..k`.
#'
#' Mirrors `sezgi.Categorical` (`py-sezgi/python/sezgi/spaces.py`) exactly
#' -- see [sz_float()]'s doc for the no-construction-time-validation note.
#'
#' @param k Numeric/integer scalar, the number of categories.
#' @param n Numeric/integer scalar, the number of genes. Required --
#'   `spaces.py`'s `Categorical.n` has no default either.
#' @returns An object of class `c("sz_block_categorical", "sz_block")` with
#'   elements `k`, `n`.
#' @export
sz_categorical <- function(k, n) {
  structure(list(k = k, n = n), class = c("sz_block_categorical", "sz_block"))
}

#' A binary block: `n` bits.
#'
#' Mirrors `sezgi.Binary` (`py-sezgi/python/sezgi/spaces.py`) exactly --
#' see [sz_float()]'s doc for the no-construction-time-validation note.
#'
#' @param n Numeric/integer scalar, the number of bits.
#' @returns An object of class `c("sz_block_binary", "sz_block")` with
#'   element `n`.
#' @export
sz_binary <- function(n) {
  structure(list(n = n), class = c("sz_block_binary", "sz_block"))
}

#' A permutation block: one permutation of `0..n`.
#'
#' Mirrors `sezgi.Permutation` (`py-sezgi/python/sezgi/spaces.py`) exactly
#' -- see [sz_float()]'s doc for the no-construction-time-validation note.
#'
#' @param n Numeric/integer scalar, the permutation length.
#' @returns An object of class `c("sz_block_permutation", "sz_block")` with
#'   element `n`.
#' @export
sz_permutation <- function(n) {
  structure(list(n = n), class = c("sz_block_permutation", "sz_block"))
}

# ---- sz_space -----------------------------------------------------------

#' Composes one or more blocks into a search space, in the given order.
#'
#' Mirrors `sezgi.Space` (`py-sezgi/python/sezgi/spaces.py`) exactly: at
#' least one block is required, and every argument must inherit
#' `"sz_block"` (i.e. be built by [sz_float()], [sz_int()],
#' [sz_categorical()], [sz_binary()], or [sz_permutation()]) -- these are
#' the only two validation rules `Space.__init__` itself performs.
#'
#' @param ... One or more `sz_block` objects, in space order.
#' @returns An object of class `"sz_space"` with element `blocks` (a
#'   `list` of the given blocks, in order).
#' @export
sz_space <- function(...) {
  blocks <- list(...)
  if (length(blocks) == 0) {
    stop("sz_space() requires at least one block")
  }
  ok <- vapply(blocks, inherits, logical(1), what = "sz_block")
  if (!all(ok)) {
    bad <- blocks[[which(!ok)[1]]]
    stop(sprintf(
      "sz_space() blocks must be sz_block objects (sz_float()/sz_int()/sz_categorical()/sz_binary()/sz_permutation()), got %s",
      paste(class(bad), collapse = "/")
    ))
  }
  structure(list(blocks = blocks), class = "sz_space")
}

# ---- print methods --------------------------------------------------------

#' Prints an `sz_block` object (any of the five block kinds).
#'
#' @param x An `sz_block` object (from [sz_float()], [sz_int()],
#'   [sz_categorical()], [sz_binary()], or [sz_permutation()]).
#' @param ... Ignored.
#' @returns `x`, invisibly.
#' @export
print.sz_block <- function(x, ...) {
  kind <- sub("^sz_block_", "", class(x)[1])
  fields <- names(x)
  vals <- vapply(fields, function(f) format(x[[f]]), character(1))
  cat(sprintf(
    "<sz_block_%s %s>\n", kind,
    paste(sprintf("%s=%s", fields, vals), collapse = ", ")
  ))
  invisible(x)
}

#' Prints an `sz_space` object.
#'
#' @param x An `sz_space` object (from [sz_space()]).
#' @param ... Ignored.
#' @returns `x`, invisibly.
#' @export
print.sz_space <- function(x, ...) {
  n <- length(x$blocks)
  cat(sprintf("<sz_space: %d block%s>\n", n, if (n == 1L) "" else "s"))
  for (b in x$blocks) {
    cat("  ")
    print(b)
  }
  invisible(x)
}

# ---- serialization to Task 2's Rust wire shape ---------------------------

#' Serializes a space to the flat block-list representation Task 2's Rust
#' entry point consumes -- the R analogue of `Space._to_json()`'s per-block
#' dict shape (`spaces.py`'s `_to_dict()` methods), returning a plain
#' `list` (not a JSON string): one entry per block, each a named `list`
#' with a `type` field (`"float"`/`"int"`/`"categorical"`/`"binary"`/
#' `"permutation"`, matching `Block`'s own
#' `#[serde(tag = "type", rename_all = "snake_case")]`,
#' `crates/core/src/space.rs:11-19` -- the same key `_to_dict()` uses)
#' plus that block's own numeric fields, coerced (`as.double()` for
#' `lo`/`hi`, `as.integer()` for `n`/`k`) the same way `_to_dict()`
#' coerces.
#'
#' @param space An `sz_space` object (from [sz_space()]).
#' @returns A `list`, one entry per block, in space order.
#' @noRd
.sz_space_to_blocks <- function(space) {
  if (!inherits(space, "sz_space")) {
    stop(sprintf(
      "expected an sz_space (see sz_space()), got %s",
      paste(class(space), collapse = "/")
    ))
  }
  lapply(space$blocks, .sz_block_to_list)
}

#' @param b An `sz_block` object.
#' @returns A named `list` with a `type` field plus that block's own
#'   numeric fields, coerced to the types `Block`'s `Deserialize` impl
#'   expects.
#'
#' M4-2 Task 2 fix: the Int branch's `lo`/`hi` used to go through
#' `as.integer()` (R's 32-bit integer type), silently narrowing
#' `Block::Int`'s `i64` bounds (`crates/core/src/space.rs`) for any value
#' outside `[-2147483647, 2147483647]` (`as.integer()` produces `NA` with a
#' warning past that range, which would then round-trip as a wrong/missing
#' bound). Widened to `as.double()` -- matching the Float branch's own
#' coercion, and matching how `sz_solve_int_quadratic()`'s `lo`/`hi`
#' params already cross the same FFI boundary (`f64` all the way, cast to
#' `i64` on the Rust side via `f64_to_i64`) -- so the full practical `i64`
#' range round-trips exactly (a `double` has 53 bits of exact integer
#' precision, comfortably more than this crate's own Int-typed
#' diagnostics/presets ever need). `n` stays `as.integer()`: it is a
#' `usize` dimension count, never expected to approach either limit.
#' @noRd
.sz_block_to_list <- function(b) {
  if (inherits(b, "sz_block_float")) {
    list(type = "float", lo = as.double(b$lo), hi = as.double(b$hi), n = as.integer(b$n))
  } else if (inherits(b, "sz_block_int")) {
    list(type = "int", lo = as.double(b$lo), hi = as.double(b$hi), n = as.integer(b$n))
  } else if (inherits(b, "sz_block_categorical")) {
    list(type = "categorical", k = as.integer(b$k), n = as.integer(b$n))
  } else if (inherits(b, "sz_block_binary")) {
    list(type = "binary", n = as.integer(b$n))
  } else if (inherits(b, "sz_block_permutation")) {
    list(type = "permutation", n = as.integer(b$n))
  } else {
    stop(sprintf("unknown block class: %s", paste(class(b), collapse = "/")))
  }
}
