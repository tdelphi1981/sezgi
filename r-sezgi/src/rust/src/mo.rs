//! Multi-objective bindings (`sz_nsga2` / `sz_mo_hypervolume_2d` /
//! `sz_mo_hypervolume` / `sz_mo_igd` / `sz_mo_pareto_front` / `sz_mo_evaluate`
//! / `sz_mo_evaluate_constraints` / `sz_mo_read_moa`) -- M3-2 Task 10,
//! extended M3-7 Task 11.
//!
//! Binds T6's NSGA-II runner (`sezgi_components::nsga2::{nsga2_run,
//! Nsga2Config}`, incl. the M3-7 constrained and binary paths), the
//! ZDT/DTLZ/WFG benchmark suites (`sezgi_problems::{Zdt, Zdt5, Dtlz, Wfg}`),
//! the exact 2-objective and general-M hypervolume / IGD indicators
//! (`sezgi_stats::{hypervolume_2d, hypervolume, igd}`), and the sezgi-moa v1
//! archive logging (`sezgi_bench::{nsga2_run_logged, read_moa}`) -- the exact
//! same Rust surface py-sezgi's `sezgi.mo` module binds (M3-2 Task 9 /
//! M3-7 Task 10, `py-sezgi/src/lib.rs`'s "Multi-objective bindings" section;
//! see also `.superpowers/sdd/2026-08-30-sezgi-m3-2/task-9-report.md` and
//! `.superpowers/sdd/2026-09-01-sezgi-m3-7/task-11-report.md` for the full
//! dict-shape provenance this file mirrors 1:1 by key name).
//!
//! Every scalar/vector f64 is passed through EXACTLY as savvy already
//! returns bit-for-bit for a Rust `f64` -- no rounding/formatting anywhere
//! in this file, so a same-seed call is bit-identical to both a repeat R
//! call and to py-sezgi's own binding (both call the identical seeded Rust
//! core; asserted directly in `tests/testthat/test-mo.R`'s cross-language
//! fixtures).
//!
//! ## Problem-string mapping (shared by `sz_nsga2`, `sz_mo_pareto_front`,
//! `sz_mo_evaluate`, and `sz_mo_evaluate_constraints`, via
//! `mo_problem_from_str`)
//! Identical to py-sezgi's own mapping (see that module's doc): `"zdt1"`..
//! `"zdt4"`, `"zdt6"` (real-coded); `"zdt5"` (the binary-coded T5, its own
//! `Zdt5` type -- fixed 80-bit layout, so `dim` is REJECTED for it);
//! `"dtlz1"`..`"dtlz9"` (`m` REQUIRED; 8/9 are the constrained pair and
//! surface `violations`); `"wfg1"`..`"wfg9"` (`m` required, optional `k`/`l`
//! with the toolkit-recommended defaults). **Passing `m` for a `zdt*`
//! problem is an error** (zdt problems are always 2-objective by
//! construction, so a caller-supplied `m` could never be honored -- rejecting
//! it surfaces the mistake instead of silently ignoring the argument); `k`/
//! `l` are likewise WFG-only. `dim`/`m`/`k`/`l` are each `Option<usize>` on
//! the Rust side (R's native nullable-argument convention): a parameter
//! given where it does not apply, or omitted where it is required, is an
//! honest error -- never silently ignored.
//!
//! ## Container-idiom decisions (key NAMES are the 1:1 mandate; containers
//! follow r-sezgi's own precedent, not py-sezgi's)
//! - `individuals`/`objectives` (`sz_nsga2` output) and `sz_mo_pareto_front`'s
//!   return: a `list` of numeric vectors, one per point/individual --
//!   mirrors `sz_bias_structural()`'s `final_positions` convention
//!   (`bias.rs`'s `structural_result_list`), NOT a matrix. A `Block::Binary`
//!   block's bits are flattened to `0.0`/`1.0` (zdt5's own path) -- the SAME
//!   convention py-sezgi's `genotype_to_flat_vec` uses, keeping
//!   `individuals` uniformly a "list of numeric vectors" across every
//!   problem family rather than a differently-typed field for zdt5 alone.
//! - `front`/`reference_front` (`sz_mo_hypervolume_2d`/`sz_mo_hypervolume`/
//!   `sz_mo_igd` INPUTS): an R-idiomatic numeric matrix, rows = points --
//!   mirrors `sz_stats_friedman()`'s `m` input convention (`matrix_to_rows`
//!   in `stats.rs`), NOT a list of rows (unlike py-sezgi's
//!   `list[list[float]]`). `matrix_to_rows` is duplicated here rather than
//!   imported (`stats.rs`'s copy is a private fn -- no shared private
//!   cross-module import, same rule this crate's other duplicated helpers
//!   already document).
//! - `front0` (`sz_nsga2` output): converted to 1-based indices (R's native
//!   convention) into `individuals`/`objectives` -- py-sezgi's `front0` is
//!   0-based; this binding converts so `result$individuals[[result$front0[i]]]`
//!   works directly for an R caller, rather than requiring `+ 1` at every
//!   call site.
//! - `violations` (`sz_nsga2` output, M3-7): present ONLY when the problem
//!   is constrained (dtlz8/dtlz9 today) -- mirrors py-sezgi's own
//!   present-only-when-meaningful convention (`MoRunResult::violations`'s
//!   own `Some` iff the problem is constrained), not a key every caller must
//!   always check for `NULL`.
//! - `sz_mo_read_moa`'s `records[[i]]$genotype`: for `kind = "binary"`
//!   records (zdt5), an R `logical` vector (`TRUE`/`FALSE`) rather than the
//!   `0.0`/`1.0` numeric flattening `individuals`/`sz_mo_evaluate`'s `x`
//!   use -- `# sezgi decision:` mirrors py-sezgi's own `MoArchiveGenotype::
//!   Binary(bits)` -> Python `bool` list mapping (`list of bool` in that
//!   binding's own doc) at the R-native equivalent (a `logical` vector,
//!   R's own boolean container), rather than reusing the `0.0`/`1.0`
//!   encoding used elsewhere in this file -- the raw archive record is a
//!   genuinely different value from a decision vector (it is never fed back
//!   into `sz_mo_evaluate`'s `x` slot), so there is no round-trip
//!   consistency requirement forcing the same numeric encoding here.
//!
//! ## `sz_nsga2`'s `log_dir`/`label` (M3-7 Task 9/11): sezgi-moa v1 run
//! logging
//! When `log_dir` is given, the run is executed via
//! [`sezgi_bench::nsga2_run_logged`] instead of the plain
//! [`sezgi_components::nsga2::nsga2_run`], streaming every feasible archive
//! insertion to `<log_dir>/<label>-s<seed>.moa` (sezgi-moa v1 format; see
//! `crates/bench/src/mo_archive.rs`'s own module doc for the exact format
//! grammar -- NO timestamp header line, so two runs with identical
//! `(problem, seed, budget, label)` inputs produce byte-identical files,
//! asserted directly in this module's own tests) -- `label` is REQUIRED
//! whenever `log_dir` is given (an honest error otherwise); omitting
//! `log_dir` runs exactly as before (byte-identical `nsga2_run` path), and
//! `label` is simply ignored if given without `log_dir`.

use savvy::{
    savvy, savvy_err, NullSexp, OwnedListSexp, OwnedLogicalSexp, OwnedRealSexp, OwnedStringSexp,
    RealSexp, Sexp,
};
use sezgi_bench::{nsga2_run_logged, read_moa as bench_read_moa, GenoKind, MoArchiveGenotype};
use sezgi_components::nsga2::{nsga2_run, MoRunResult, Nsga2Config};
use sezgi_core::mo::MoProblem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};
use sezgi_problems::zdt::Zdt5;
use sezgi_problems::{Dtlz, Wfg, Zdt};
use sezgi_stats::{hypervolume, hypervolume_2d, igd};
use std::path::Path;

/// Casts a non-negative-checked, WHOLE-NUMBER-checked `f64` (as passed
/// from R, which has no native unsigned integer type) to `u64`, rejecting
/// negative, non-finite, or fractional values. Duplicated from the
/// identically-named private helpers in `solve.rs` / `session.rs` /
/// `experiment.rs` / `stats.rs` / `bias.rs` -- no shared private
/// cross-module import, same "no shared private crate imports" rule those
/// files already document.
fn f64_to_u64(name: &str, x: f64) -> savvy::Result<u64> {
    if !x.is_finite() || x < 0.0 {
        return Err(savvy_err!(
            "{} must be a non-negative finite number, got {}",
            name,
            x
        ));
    }
    if x.fract() != 0.0 {
        return Err(savvy_err!("{} expected a whole number, got {}", name, x));
    }
    Ok(x as u64)
}

/// Same contract as [`f64_to_u64`], returning `usize` -- for `dim`/`m`/`n`/
/// `pop_size`/`k`/`l`-typed params, which are consumed as `usize` on the
/// Rust side.
fn f64_to_usize(name: &str, x: f64) -> savvy::Result<usize> {
    f64_to_u64(name, x).map(|v| v as usize)
}

/// Same contract as [`f64_to_usize`], threading an `Option` through --
/// every `dim`/`m`/`k`/`l` parameter in this file is independently optional
/// (see this module's own doc, "Problem-string mapping").
fn opt_f64_to_usize(name: &str, x: Option<f64>) -> savvy::Result<Option<usize>> {
    match x {
        Some(v) => Ok(Some(f64_to_usize(name, v)?)),
        None => Ok(None),
    }
}

/// Converts an R matrix (`RealSexp` with a `dim` attribute, R's column-major
/// storage) into `Vec<Vec<f64>>` with **rows = points** -- the "front"
/// convention this module's own doc documents. Duplicated from `stats.rs`'s
/// identically-named private `matrix_to_rows` (same "no shared private
/// cross-module import" rule).
///
/// R stores a matrix column-major: `data[col * nrow + row]`.
fn matrix_to_rows(x: &RealSexp) -> savvy::Result<Vec<Vec<f64>>> {
    let dim = x
        .get_dim()
        .ok_or_else(|| savvy_err!("expected a matrix (with a `dim` attribute), got a plain vector"))?;
    if dim.len() != 2 {
        return Err(savvy_err!(
            "expected a 2-D matrix, got {} dimensions",
            dim.len()
        ));
    }
    let nrow = dim[0] as usize;
    let ncol = dim[1] as usize;
    let data = x.as_slice();
    if data.len() != nrow * ncol {
        return Err(savvy_err!(
            "matrix data length {} does not match dim {}x{}",
            data.len(),
            nrow,
            ncol
        ));
    }

    let mut rows = vec![vec![0.0_f64; ncol]; nrow];
    for c in 0..ncol {
        for r in 0..nrow {
            rows[r][c] = data[c * nrow + r];
        }
    }
    Ok(rows)
}

/// WFG's own recommended `k` default (`sezgi_problems::wfg`'s own module
/// doc, "Recommended `k`/`l` defaults" section -- the widely-used
/// literature reading: `k=4` for `m=2`, `k=2*(m-1)` for `m>=3`). Guarded at
/// `m<=2` (not `m==2`) purely to avoid a `usize` underflow computing a
/// THROWAWAY value for `m<2`: an `m<2` call always fails [`Wfg::new`]'s own
/// `BadM` check first, so this default is never actually used in that case
/// -- identical guard to py-sezgi's own `wfg_default_k`.
fn wfg_default_k(m: usize) -> usize {
    if m <= 2 { 4 } else { 2 * (m - 1) }
}

/// WFG's own recommended `l` default (same module-doc section): `l=20`,
/// unconditional on `m`.
const WFG_DEFAULT_L: usize = 20;

/// Shared problem-string -> `Box<dyn MoProblem>` builder for `sz_nsga2_raw`,
/// `sz_mo_pareto_front_raw`, `sz_mo_evaluate_raw`, and
/// `sz_mo_evaluate_constraints_raw`. See this module's own doc for the full
/// mapping (identical to py-sezgi's `mo_problem_from_str`, including its
/// error wording, R-appropriately reworded for "= NULL" instead of
/// "=None").
fn mo_problem_from_str(
    problem: &str,
    dim: Option<usize>,
    m: Option<usize>,
    k: Option<usize>,
    l: Option<usize>,
) -> savvy::Result<Box<dyn MoProblem>> {
    let unknown = || {
        savvy_err!(
            "unknown problem `{}` (expected one of zdt1, zdt2, zdt3, zdt4, zdt5, zdt6, \
             dtlz1..dtlz9, or wfg1..wfg9)",
            problem
        )
    };

    // sezgi decision: ZDT5 (M3-7 Task 6) is checked BEFORE the generic
    // "zdt"-prefix branch below -- `Zdt::new(5, dim)` exists as a type but
    // deliberately REJECTS which=5 with its own "use Zdt5::new() instead"
    // error (zdt.rs's own doc), since ZDT5 is a separate, binary-coded type
    // with no free `dim`/real-coded space at all. Letting the generic
    // branch's `strip_prefix("zdt")` catch "zdt5" would just re-surface
    // that Rust-internal redirect error instead of actually constructing
    // it, so it is special-cased here first -- same ordering py-sezgi's own
    // `mo_problem_from_str` uses.
    if problem == "zdt5" {
        if m.is_some() {
            return Err(savvy_err!(
                "m is DTLZ-only (number of objectives); zdt5 is always 2-objective -- \
                 omit m (or pass m = NULL) for zdt5"
            ));
        }
        // sezgi decision: `dim` is REJECTED for zdt5 (mirroring how `m` is
        // rejected for every zdt problem), not merely ignored -- zdt5's
        // search space is a FIXED 80-bit layout (one 30-bit block plus ten
        // 5-bit blocks, Zitzler/Deb/Thiele 2000 Definition 4; see
        // `Zdt5::new`'s own doc), not a free-dimension real-coded space, so
        // a caller-supplied `dim` can never be honored and silently
        // dropping it would hide a caller's wrong assumption.
        if dim.is_some() {
            return Err(savvy_err!(
                "dim is not accepted for zdt5: its search space is a FIXED 80-bit layout \
                 (one 30-bit block plus ten 5-bit blocks) -- omit dim (or pass dim = NULL) for zdt5"
            ));
        }
        if k.is_some() || l.is_some() {
            return Err(savvy_err!(
                "k/l are wfg-only; omit them (or pass NULL) for zdt5"
            ));
        }
        return Ok(Box::new(Zdt5::new()));
    }

    if let Some(rest) = problem.strip_prefix("zdt") {
        let which: u32 = rest.parse().map_err(|_| unknown())?;
        if m.is_some() {
            return Err(savvy_err!(
                "m is DTLZ-only (number of objectives); zdt problems are always 2-objective -- \
                 omit m (or pass m = NULL) for a zdt problem"
            ));
        }
        if k.is_some() || l.is_some() {
            return Err(savvy_err!(
                "k/l are wfg-only; omit them (or pass NULL) for a zdt problem"
            ));
        }
        let dim = dim.ok_or_else(|| savvy_err!("dim is required for zdt problems"))?;
        let p = Zdt::new(which, dim).map_err(|e| savvy_err!("{e}"))?;
        Ok(Box::new(p))
    } else if let Some(rest) = problem.strip_prefix("wfg") {
        let which: u32 = rest.parse().map_err(|_| unknown())?;
        // sezgi decision: `dim` is REJECTED for wfg (same reasoning as
        // zdt5 above) -- a WFG instance's dimension `n = k + l` is DERIVED
        // from `k`/`l` (`Wfg::new`'s own signature takes no `dim` at all),
        // so there is no free `dim` slot to fill; a caller must use `k`/`l`
        // instead.
        if dim.is_some() {
            return Err(savvy_err!(
                "dim is not accepted for wfg problems: n = k + l is derived from k and l -- \
                 omit dim (or pass dim = NULL) and use k/l instead"
            ));
        }
        let m = m.ok_or_else(|| savvy_err!("m (number of objectives) is required for wfg problems"))?;
        // sezgi decision: `k`/`l` default to the toolkit's own recommended
        // values (`wfg_default_k`/`WFG_DEFAULT_L` above) when omitted,
        // mirroring `p_m`'s own `NULL`-resolves-to-a-formula precedent
        // rather than requiring every caller to spell them out.
        let k = k.unwrap_or_else(|| wfg_default_k(m));
        let l = l.unwrap_or(WFG_DEFAULT_L);
        let p = Wfg::new(which, m, k, l).map_err(|e| savvy_err!("{e}"))?;
        Ok(Box::new(p))
    } else if let Some(rest) = problem.strip_prefix("dtlz") {
        let which: u32 = rest.parse().map_err(|_| unknown())?;
        if k.is_some() || l.is_some() {
            return Err(savvy_err!(
                "k/l are wfg-only; omit them (or pass NULL) for a dtlz problem"
            ));
        }
        let m = m.ok_or_else(|| savvy_err!("m (number of objectives) is required for dtlz problems"))?;
        let dim = dim.ok_or_else(|| savvy_err!("dim is required for dtlz problems"))?;
        // dtlz8/dtlz9 (M3-7 Task 2) reuse `Dtlz::new` unchanged -- its own
        // `dim > m` constraint-surface check (`DtlzError::BadDimConstraintSurface`)
        // surfaces via the SAME `map_err` path as every other dtlz error.
        let p = Dtlz::new(which, m, dim).map_err(|e| savvy_err!("{e}"))?;
        Ok(Box::new(p))
    } else {
        Err(unknown())
    }
}

/// Flattens a [`Genotype`] into a single `Vec<f64>` (concatenating every
/// block in order) -- always either an all-`Block::Float` genotype (zdt1-4/
/// 6, dtlz1-9, wfg1-9) or an all-`Block::Binary` one (zdt5 only), since
/// `nsga2_run`/`nsga2_run_logged` validate exactly one of those two shapes
/// before ever constructing one. See this module's own doc, "Container-idiom
/// decisions", for the `Block::Binary` -> `0.0`/`1.0` flattening rationale
/// (identical to py-sezgi's own `genotype_to_flat_vec`).
fn genotype_to_flat_vec(g: &Genotype) -> Vec<f64> {
    let mut out = Vec::new();
    for b in &g.blocks {
        match b {
            BlockValues::Float(xs) => out.extend_from_slice(xs),
            BlockValues::Bin(bits) => out.extend(bits.iter().map(|&b| if b { 1.0 } else { 0.0 })),
            _ => {}
        }
    }
    out
}

/// Builds a [`Genotype`] matching `space`'s own block layout from a flat
/// `x` (the inverse of [`genotype_to_flat_vec`]'s flattening, used by
/// `sz_mo_evaluate_raw`/`sz_mo_evaluate_constraints_raw` to construct a
/// one-off individual from a caller-supplied decision vector). `Block::Float`
/// blocks take their slice of `x` verbatim; `Block::Binary` blocks read each
/// value as a bit via `!= 0.0` (matching `genotype_to_flat_vec`'s own
/// `0.0`/`1.0` encoding on the way back out) -- identical to py-sezgi's own
/// `genotype_from_flat`.
///
/// # Errors
/// A savvy error if `x.len() != space.dim()`, or if `space` has a block that
/// is neither `Block::Float` nor `Block::Binary` (unreachable through
/// `mo_problem_from_str`'s own zdt/dtlz/wfg constructors today, but checked
/// honestly rather than silently skipped).
fn genotype_from_flat(space: &SearchSpace, x: &[f64]) -> savvy::Result<Genotype> {
    if x.len() != space.dim() {
        return Err(savvy_err!(
            "x must have exactly {} coordinates (dim={}), got {}",
            space.dim(),
            space.dim(),
            x.len()
        ));
    }
    let mut blocks = Vec::with_capacity(space.blocks().len());
    let mut i = 0usize;
    for b in space.blocks() {
        match *b {
            Block::Float { n, .. } => {
                blocks.push(BlockValues::Float(x[i..i + n].to_vec()));
                i += n;
            }
            Block::Binary { n } => {
                blocks.push(BlockValues::Bin(x[i..i + n].iter().map(|&v| v != 0.0).collect()));
                i += n;
            }
            _ => {
                return Err(savvy_err!(
                    "sz_mo_evaluate only supports all-Float or all-Binary spaces"
                ))
            }
        }
    }
    Ok(Genotype { blocks })
}

/// Builds the named list mirroring `MoRunResult` 1:1 by field name --
/// `individuals`, `objectives`, `front0`, `evals_used`, and (M3-7)
/// `violations` -- same as py-sezgi's `mo_nsga2` dict shape. Container
/// idiom: see this module's own doc, "Container-idiom decisions".
fn nsga2_result_list(r: &MoRunResult) -> savvy::Result<OwnedListSexp> {
    let mut individuals = OwnedListSexp::new(r.individuals.len(), false)?;
    for (i, g) in r.individuals.iter().enumerate() {
        let flat = genotype_to_flat_vec(g);
        individuals.set_value(i, OwnedRealSexp::try_from_slice(flat.as_slice())?)?;
    }

    let mut objectives = OwnedListSexp::new(r.objectives.len(), false)?;
    for (i, row) in r.objectives.iter().enumerate() {
        objectives.set_value(i, OwnedRealSexp::try_from_slice(row.as_slice())?)?;
    }

    // 0-based (Rust/Python) -> 1-based (R) -- see this module's own doc.
    let front0: Vec<f64> = r.front0.iter().map(|&idx| (idx + 1) as f64).collect();

    // `violations` is present ONLY when the problem is constrained (see
    // this module's own doc, "Container-idiom decisions").
    let n_fields = if r.violations.is_some() { 5 } else { 4 };
    let mut out = OwnedListSexp::new(n_fields, true)?;
    out.set_name_and_value(0, "individuals", individuals)?;
    out.set_name_and_value(1, "objectives", objectives)?;
    out.set_name_and_value(2, "front0", OwnedRealSexp::try_from_slice(front0.as_slice())?)?;
    out.set_name_and_value(
        3,
        "evals_used",
        OwnedRealSexp::try_from_scalar(r.evals_used as f64)?,
    )?;
    if let Some(v) = &r.violations {
        out.set_name_and_value(4, "violations", OwnedRealSexp::try_from_slice(v.as_slice())?)?;
    }
    Ok(out)
}

/// NSGA-II run (Deb, Pratap, Agarwal & Meyarivan 2002) -- binds
/// [`sezgi_components::nsga2::nsga2_run`] (or, when `log_dir` is given,
/// [`sezgi_bench::nsga2_run_logged`] -- see this module's own doc). See this
/// module's own doc for the `problem`/`dim`/`m`/`k`/`l` mapping.
///
/// This is the raw savvy-generated binding: every `Option`-typed parameter
/// (savvy's own "optional args last" requirement) trails the required ones
/// in `dim, m, p_m, p_m_bin, p_m_cat, k, l, log_dir, label` order. The
/// public R entry point with R-native argument order and defaults is the
/// hand-written wrapper `sz_nsga2()` in `R/mo.R`, which calls this function
/// -- same raw/wrapper pattern as `sz_bias_structural()` /
/// `sz_bias_structural_raw()`.
///
/// @param problem One of `"zdt1"`..`"zdt6"`, `"dtlz1"`..`"dtlz9"`, or
///   `"wfg1"`..`"wfg9"`.
/// @param pop_size Population size (double, cast to `usize`). Must be
///   `>= 4` and a multiple of 4 (KanGAL's double-permutation tournament
///   pairing requires it -- NOT merely "even, >= 4").
/// @param budget Total evaluation budget (double, cast to `u64`).
/// @param seed Master RNG seed (double, cast to `u64`).
/// @param eta_c SBX distribution index. Paper default: `20.0`.
/// @param eta_m Polynomial-mutation distribution index. Paper default:
///   `20.0`.
/// @param p_c SBX crossover probability. Paper default: `0.9`.
/// @param p_c_bin Binary-genotype crossover probability. Default `0.9`
///   (mirrors `p_c`'s own default; only consulted for an all-Binary space,
///   i.e. zdt5).
/// @param p_c_cat Categorical-genotype crossover probability (post-M3-8
///   deferral cleanup). Default `0.9` (mirrors `p_c_bin`'s own default --
///   no paper/reference precedent exists for Categorical); only consulted
///   when `problem` builds a `Mixed` space containing a `Block::Categorical`
///   block. No problem string in this module's own catalog (zdt1-6,
///   dtlz1-9, wfg1-9) builds one today, so this is currently validated but
///   inert against every reachable problem -- exposed anyway for symmetry
///   with `p_c_bin`/`p_m_bin` and `Nsga2Config`'s own public field set.
/// @param dim Optional decision-space dimensionality (double, cast to
///   `usize`). REQUIRED (may be `NULL`, but the argument itself must be
///   supplied) for zdt1-4/6 and dtlz1-9; REJECTED (must be `NULL`) for
///   zdt5 and wfg1-9.
/// @param m Optional number of objectives (double, cast to `usize`).
///   REQUIRED for dtlz/wfg problems; must be `NULL` for zdt problems.
/// @param p_m Optional per-variable mutation probability. `NULL` resolves
///   on the Rust side to `1 / n_variables` (the paper's own default), never
///   re-derived here.
/// @param p_m_bin Optional per-bit binary mutation probability. `NULL`
///   resolves on the Rust side to `1 / l` (the paper's own binary-coded
///   default); only consulted for an all-Binary space.
/// @param p_m_cat Optional per-gene Categorical mutation probability
///   (post-M3-8 deferral cleanup). `NULL` resolves on the Rust side to
///   `1 / n_cat` (`n_cat` = the space's total flattened `Block::Categorical`
///   dimension), mirroring `p_m`/`p_m_bin`'s own `NULL`-resolves-to-a-formula
///   design; only consulted for a `Mixed` space containing a Categorical
///   block (currently unreachable through this catalog -- see `p_c_cat`
///   above).
/// @param k Optional WFG position-related-parameter count. `NULL` resolves
///   to the toolkit's own recommended default; wfg-only.
/// @param l Optional WFG distance-related-parameter count. `NULL` resolves
///   to `20`; wfg-only.
/// @param log_dir Optional sezgi-moa v1 log directory. When given, `label`
///   is required and the run additionally streams to
///   `<log_dir>/<label>-s<seed>.moa`.
/// @param label Optional sezgi-moa run label; required iff `log_dir` is
///   given.
/// @returns A named list with `individuals`, `objectives`, `front0`,
///   `evals_used`, and (present only for a constrained problem) `violations`
///   -- see this module's own doc.
///
/// # Errors
/// A savvy error for an unrecognized `problem` string, a `dim`/`m`/`k`/`l`
/// given where the problem does not accept it (or missing where required --
/// see `mo_problem_from_str`'s own doc), a `log_dir` given without `label`,
/// or any [`sezgi_components::nsga2::Nsga2Error`] / `MoRunLoggedError`
/// (including `pop_size` failing the `>= 4 && pop_size % 4 == 0` check).
/// @noRd
#[savvy]
#[allow(clippy::too_many_arguments)]
fn sz_nsga2_raw(
    problem: &str,
    pop_size: f64,
    budget: f64,
    seed: f64,
    eta_c: f64,
    eta_m: f64,
    p_c: f64,
    p_c_bin: f64,
    p_c_cat: f64,
    dim: Option<f64>,
    m: Option<f64>,
    p_m: Option<f64>,
    p_m_bin: Option<f64>,
    p_m_cat: Option<f64>,
    k: Option<f64>,
    l: Option<f64>,
    log_dir: Option<&str>,
    label: Option<&str>,
) -> savvy::Result<Sexp> {
    let dim_u = opt_f64_to_usize("dim", dim)?;
    let m_u = opt_f64_to_usize("m", m)?;
    let k_u = opt_f64_to_usize("k", k)?;
    let l_u = opt_f64_to_usize("l", l)?;
    let prob = mo_problem_from_str(problem, dim_u, m_u, k_u, l_u)?;

    // p_c_cat/p_m_cat (post-M3-8 deferral cleanup): real, user-tunable
    // parameters now, threaded through unconditionally -- see this
    // function's own doc, `p_c_cat`/`p_m_cat` sections, for why they are
    // currently inert against every problem in `mo_problem_from_str`'s
    // catalog (no reachable problem builds a `Mixed` space yet) while still
    // being exposed and validated, mirroring `p_c_bin`/`p_m_bin`'s own
    // already-shipped treatment exactly.
    let cfg = Nsga2Config {
        pop_size: f64_to_usize("pop_size", pop_size)?,
        budget: f64_to_u64("budget", budget)?,
        seed: f64_to_u64("seed", seed)?,
        eta_c,
        eta_m,
        p_c,
        p_m,
        p_c_bin,
        p_m_bin,
        p_c_cat,
        p_m_cat,
    };

    let result = if let Some(dir) = log_dir {
        let label = label.ok_or_else(|| {
            savvy_err!(
                "label is required when log_dir is given (sezgi-moa file naming: \
                 <log_dir>/<label>-s<seed>.moa)"
            )
        })?;
        nsga2_run_logged(prob.as_ref(), &cfg, Path::new(dir), label).map_err(|e| savvy_err!("{e}"))?
    } else {
        nsga2_run(prob.as_ref(), &cfg).map_err(|e| savvy_err!("{e}"))?
    };

    Ok(nsga2_result_list(&result)?.into())
}

/// Exact 2-objective hypervolume (Zitzler & Thiele 1999 S-metric,
/// reference-point variant; minimization) -- binds
/// [`sezgi_stats::hypervolume_2d`] exactly. See that function's doc for the
/// pinned definition.
///
/// @param front A numeric matrix, rows = points, 2 columns (`f1`, `f2`) --
///   R-idiomatic (unlike py-sezgi's `list[[f1,f2],...]`); see this module's
///   own doc, "Container-idiom decisions". Build with `rbind()`/`matrix()`.
/// @param ref_point A 2-element numeric vector.
/// @returns A numeric scalar.
///
/// # Errors
/// A savvy error if `ref_point` does not have exactly 2 values, if `front`
/// is not a matrix, or for any [`sezgi_stats::StatsError`] (empty front, a
/// non-2-objective row, or a non-finite value).
/// @export
#[savvy]
fn sz_mo_hypervolume_2d(front: RealSexp, ref_point: RealSexp) -> savvy::Result<Sexp> {
    let rows = matrix_to_rows(&front)?;
    let rp_slice = ref_point.as_slice();
    if rp_slice.len() != 2 {
        return Err(savvy_err!(
            "ref_point must have exactly 2 values, got {}",
            rp_slice.len()
        ));
    }
    let rp: [f64; 2] = [rp_slice[0], rp_slice[1]];
    let v = hypervolume_2d(&rows, &rp).map_err(|e| savvy_err!("{e}"))?;
    v.try_into()
}

/// General-M exact hypervolume (While, Bradstreet & Barone 2012, the WFG
/// algorithm; `M == 2` delegates internally to the SAME
/// [`sezgi_stats::hypervolume_2d`]) -- binds [`sezgi_stats::hypervolume`]
/// (M3-7 Task 8/11). Unlike `sz_mo_hypervolume_2d`, `front`/`ref_point` may
/// have any number `M >= 1` of objectives.
///
/// `ref_point` is REQUIRED, with no default: `sezgi_stats::moo_indicators`'s
/// own module doc, "Choosing a reference point: explicit-always, contested
/// in the literature" section, deliberately never picks one for the caller.
/// One common convention from that literature (also critiqued by Ishibuchi,
/// Imada, Setoguchi & Nojima 2018, "How to Specify a Reference Point in
/// Hypervolume Calculation for Fair Performance Comparison", GECCO
/// Companion) is the analytic front's nadir point (the componentwise worst
/// value across the front) scaled by `1.1` -- a caller-supplied choice,
/// never defaulted here.
///
/// @param front A numeric matrix, rows = points, any number of columns --
///   see this module's own doc, "Container-idiom decisions".
/// @param ref_point A numeric vector, same length as `front`'s column
///   count.
/// @returns A numeric scalar. An EMPTY `front` is NOT an error -- it
///   returns `0.0` (the algorithm's own base case).
///
/// # Errors
/// A savvy error for any [`sezgi_stats::StatsError`] (`ref_point` empty, a
/// `front` row with a different number of objectives than `ref_point`, or a
/// non-finite value), or if `front` is not a matrix.
/// @export
#[savvy]
fn sz_mo_hypervolume(front: RealSexp, ref_point: RealSexp) -> savvy::Result<Sexp> {
    let rows = matrix_to_rows(&front)?;
    let rp = ref_point.as_slice().to_vec();
    let v = hypervolume(&rows, &rp).map_err(|e| savvy_err!("{e}"))?;
    v.try_into()
}

/// Inverted Generational Distance (Ishibuchi et al. 2015, eq. 12, `p = 1`)
/// -- binds [`sezgi_stats::igd`] exactly. Any (equal, consistent) number of
/// objectives across both `front` and `reference_front`.
///
/// @param front A numeric matrix, rows = points -- see this module's own
///   doc, "Container-idiom decisions".
/// @param reference_front A numeric matrix, rows = points, same column
///   count as `front`.
/// @returns A numeric scalar.
///
/// # Errors
/// A savvy error if either argument is not a matrix, or for any
/// [`sezgi_stats::StatsError`] (an empty `front`/`reference_front`, a
/// dimension mismatch, or a non-finite value).
/// @export
#[savvy]
fn sz_mo_igd(front: RealSexp, reference_front: RealSexp) -> savvy::Result<Sexp> {
    let f = matrix_to_rows(&front)?;
    let rf = matrix_to_rows(&reference_front)?;
    let v = igd(&f, &rf).map_err(|e| savvy_err!("{e}"))?;
    v.try_into()
}

/// A deterministic `n`-point sample of the analytic Pareto front in
/// OBJECTIVE space, if known -- binds [`sezgi_core::mo::MoProblem::pareto_front`].
/// Same `problem`/`dim`/`m`/`k`/`l` mapping as `sz_nsga2_raw` (see this
/// module's own doc).
///
/// This is the raw savvy-generated binding (`problem`/`n` required,
/// `dim`/`m`/`k`/`l` trailing since they are all optional). The public R
/// entry point with R-native argument order is the hand-written wrapper
/// `sz_mo_pareto_front()` in `R/mo.R`, which calls this function -- same
/// raw/wrapper pattern as `sz_nsga2()` / `sz_nsga2_raw()`.
///
/// @param problem Same mapping as `sz_nsga2_raw`.
/// @param n Number of front points to sample (double, cast to `usize`).
/// @param dim Optional decision-space dimensionality (double, cast to
///   `usize`) -- same rule as `sz_nsga2_raw`.
/// @param m Optional number of objectives (double, cast to `usize`) --
///   same rule as `sz_nsga2_raw`.
/// @param k Optional WFG position-related-parameter count; wfg-only.
/// @param l Optional WFG distance-related-parameter count; wfg-only.
/// @returns A `list` of `n` numeric vectors (see this module's own doc,
///   "Container-idiom decisions"), or `NULL` when the problem has no known
///   analytic front sample (e.g. DTLZ5/DTLZ6 with `m > 3`, or WFG1/WFG2
///   unconditionally) -- mirrors py-sezgi's `sezgi.mo.pareto_front()`
///   return exactly.
///
/// # Errors
/// Same as `sz_nsga2_raw`'s problem-construction errors.
/// @noRd
#[savvy]
fn sz_mo_pareto_front_raw(
    problem: &str,
    n: f64,
    dim: Option<f64>,
    m: Option<f64>,
    k: Option<f64>,
    l: Option<f64>,
) -> savvy::Result<Sexp> {
    let n_u = f64_to_usize("n", n)?;
    let dim_u = opt_f64_to_usize("dim", dim)?;
    let m_u = opt_f64_to_usize("m", m)?;
    let k_u = opt_f64_to_usize("k", k)?;
    let l_u = opt_f64_to_usize("l", l)?;
    let prob = mo_problem_from_str(problem, dim_u, m_u, k_u, l_u)?;

    match prob.pareto_front(n_u) {
        Some(rows) => {
            let mut out = OwnedListSexp::new(rows.len(), false)?;
            for (i, row) in rows.iter().enumerate() {
                out.set_value(i, OwnedRealSexp::try_from_slice(row.as_slice())?)?;
            }
            Ok(out.into())
        }
        None => Ok(NullSexp.into()),
    }
}

/// Direct, one-shot objective evaluation of a decision vector `x` against
/// any `sz_mo_*` problem string, bypassing `sz_nsga2`'s population/budget
/// machinery entirely -- binds [`sezgi_core::mo::MoProblem::evaluate_batch`]
/// for a single individual. Same `problem`/`dim`/`m`/`k`/`l` mapping as
/// `sz_nsga2_raw`, via [`mo_problem_from_str`].
///
/// `# sezgi decision:` (M3-7 Task 11) added purely so this binding's OWN
/// test suite can pin exact fixture values (zdt5's all-ones/all-zeros hand
/// fixtures, dtlz8/9's hand fixtures, the committed
/// `wfg_reference_values.json` points) directly from R -- mirroring
/// py-sezgi's own `mo.evaluate` (added for the identical reason, M3-7 Task
/// 10) and the existing `sz_cec2022_evaluate`/`sz_cec2014_evaluate`/
/// `sz_cec2017_evaluate` one-shot-evaluation convention already established
/// in this crate.
///
/// `x`: for an all-Float problem, its raw decision values; for zdt5 (the
/// only all-Binary problem reachable here), each entry is read as a bit
/// (`!= 0.0` -> `TRUE`) -- see [`genotype_from_flat`]'s own doc.
///
/// This is the raw savvy-generated binding (`problem`/`x` required,
/// `dim`/`m`/`k`/`l` trailing since they are all optional). The public R
/// entry point with R-native argument order (`problem, dim, x, m, k, l`) is
/// the hand-written wrapper `sz_mo_evaluate()` in `R/mo.R`, which calls this
/// function -- same raw/wrapper pattern as `sz_nsga2()` / `sz_nsga2_raw()`.
///
/// @param problem Same mapping as `sz_nsga2_raw`.
/// @param x A numeric vector, the decision vector to evaluate.
/// @param dim Optional decision-space dimensionality -- same rule as
///   `sz_nsga2_raw`.
/// @param m Optional number of objectives -- same rule as `sz_nsga2_raw`.
/// @param k Optional WFG position-related-parameter count; wfg-only.
/// @param l Optional WFG distance-related-parameter count; wfg-only.
/// @returns A numeric vector, length equal to the problem's own objective
///   count.
///
/// # Errors
/// Same problem-construction errors as `sz_nsga2_raw`/`sz_mo_pareto_front`,
/// plus a savvy error if `length(x)` does not match the problem's own
/// dimension.
/// @noRd
#[savvy]
fn sz_mo_evaluate_raw(
    problem: &str,
    x: RealSexp,
    dim: Option<f64>,
    m: Option<f64>,
    k: Option<f64>,
    l: Option<f64>,
) -> savvy::Result<Sexp> {
    let dim_u = opt_f64_to_usize("dim", dim)?;
    let m_u = opt_f64_to_usize("m", m)?;
    let k_u = opt_f64_to_usize("k", k)?;
    let l_u = opt_f64_to_usize("l", l)?;
    let prob = mo_problem_from_str(problem, dim_u, m_u, k_u, l_u)?;
    let g = genotype_from_flat(prob.space(), x.as_slice())?;
    let row = prob
        .evaluate_batch(std::slice::from_ref(&g))
        .into_iter()
        .next()
        .expect("evaluate_batch returns one row per input genotype");
    Ok(OwnedRealSexp::try_from_slice(row.as_slice())?.into())
}

/// Direct, one-shot constraint-row evaluation, mirroring `sz_mo_evaluate`'s
/// calling convention exactly (same [`mo_problem_from_str`] mapping, same
/// [`genotype_from_flat`] decoding) -- binds
/// [`sezgi_core::mo::MoProblem::evaluate_constraints_batch`].
///
/// This is the raw savvy-generated binding, same raw/wrapper pattern as
/// `sz_mo_evaluate()` / `sz_mo_evaluate_raw()` above -- the public R entry
/// point is the hand-written wrapper `sz_mo_evaluate_constraints()` in
/// `R/mo.R`.
///
/// @param problem Same mapping as `sz_nsga2_raw`.
/// @param x A numeric vector, the decision vector to evaluate.
/// @param dim Optional decision-space dimensionality -- same rule as
///   `sz_nsga2_raw`.
/// @param m Optional number of objectives -- same rule as `sz_nsga2_raw`.
/// @param k Optional WFG position-related-parameter count; wfg-only.
/// @param l Optional WFG distance-related-parameter count; wfg-only.
/// @returns A numeric vector (`g_1..g_ncon`, `g_j >= 0` meaning SATISFIED --
///   see [`sezgi_core::mo::MoProblem::evaluate_constraints_batch`]'s own doc
///   for the pinned sign convention), or `NULL` for an unconstrained
///   problem (every zdt/wfg problem, and dtlz1-7).
///
/// # Errors
/// Same as `sz_mo_evaluate_raw`.
/// @noRd
#[savvy]
fn sz_mo_evaluate_constraints_raw(
    problem: &str,
    x: RealSexp,
    dim: Option<f64>,
    m: Option<f64>,
    k: Option<f64>,
    l: Option<f64>,
) -> savvy::Result<Sexp> {
    let dim_u = opt_f64_to_usize("dim", dim)?;
    let m_u = opt_f64_to_usize("m", m)?;
    let k_u = opt_f64_to_usize("k", k)?;
    let l_u = opt_f64_to_usize("l", l)?;
    let prob = mo_problem_from_str(problem, dim_u, m_u, k_u, l_u)?;
    let g = genotype_from_flat(prob.space(), x.as_slice())?;
    match prob.evaluate_constraints_batch(std::slice::from_ref(&g)) {
        Some(rows) => {
            let row = rows
                .into_iter()
                .next()
                .expect("evaluate_constraints_batch returns one row per input genotype");
            Ok(OwnedRealSexp::try_from_slice(row.as_slice())?.into())
        }
        None => Ok(NullSexp.into()),
    }
}

/// Reads a "sezgi-moa v1" archive file written by `sz_nsga2(..., log_dir =,
/// label =)` (M3-7 Task 9/11) -- binds [`sezgi_bench::read_moa`]. Returns a
/// named list:
/// - `algo` (character): the logging algorithm name -- always `"nsga2"`
///   today (`nsga2_run_logged`'s own fixed `NSGA2_ALGO_NAME`).
/// - `problem` (character): the `label` `sz_nsga2` was called with. **Kept
///   as the literal on-disk header key name** (`crates/bench/src/mo_archive.rs`'s
///   own format grammar: the header line is `problem <label>`, not
///   `label <label>`) rather than renamed here to `"label"` -- this binding
///   stays a thin, direct mirror of `MoArchiveRun`'s own field names, so a
///   reader cross-checking against the Rust struct (or the Python binding,
///   M3-7 Task 10) sees the SAME key everywhere.
/// - `m`, `seed`, `budget` (double, whole-number-valued -- R has no native
///   integer64).
/// - `kind` (character): `"float"` or `"binary"`.
/// - `records` (list, in file/eval order): each entry a named list with
///   `eval_index` (double), `objectives` (numeric vector), `genotype`
///   (numeric vector for `kind = "float"`, `logical` vector for
///   `kind = "binary"` -- see this module's own doc, "Container-idiom
///   decisions").
/// - `archive` (list of numeric vectors): the reconstructed nondominated
///   archive at evaluation budget `at`, via `MoArchiveRun::archive_at`.
///
/// `# sezgi decision:` `at = NULL` (the default) resolves to the file's own
/// logged `budget` header field (the full run's final archive) -- mirrors
/// py-sezgi's own `mo.read_moa(path, at=None)` default exactly.
///
/// This is directly `@export`ed (no hand-written wrapper needed): `at` is
/// the only optional parameter and trails `path`, so savvy already emits
/// `at = NULL` in the generated R signature -- same pattern as
/// `sz_mo_hypervolume_2d`/`sz_mo_igd`.
///
/// @param path Path to a sezgi-moa v1 file.
/// @param at Optional evaluation budget (double, cast to `u64`) at which to
///   reconstruct the archive; `NULL` (default) uses the file's own logged
///   `budget`.
/// @returns A named list -- see this function's own doc above.
///
/// # Errors
/// A savvy error for any [`sezgi_bench::MoArchiveError`] (missing file, a
/// malformed header, or a malformed record line).
/// @export
#[savvy]
fn sz_mo_read_moa(path: &str, at: Option<f64>) -> savvy::Result<Sexp> {
    let run = bench_read_moa(path).map_err(|e| savvy_err!("{e}"))?;
    let at_u64 = match at {
        Some(v) => Some(f64_to_u64("at", v)?),
        None => None,
    };
    let evals = at_u64.unwrap_or(run.budget);

    let mut out = OwnedListSexp::new(8, true)?;
    out.set_name_and_value(0, "algo", OwnedStringSexp::try_from(run.algo.as_str())?)?;
    out.set_name_and_value(1, "problem", OwnedStringSexp::try_from(run.problem.as_str())?)?;
    out.set_name_and_value(2, "m", OwnedRealSexp::try_from_scalar(run.m as f64)?)?;
    out.set_name_and_value(3, "seed", OwnedRealSexp::try_from_scalar(run.seed as f64)?)?;
    out.set_name_and_value(4, "budget", OwnedRealSexp::try_from_scalar(run.budget as f64)?)?;
    out.set_name_and_value(
        5,
        "kind",
        OwnedStringSexp::try_from(match run.kind {
            GenoKind::Float => "float",
            GenoKind::Binary => "binary",
        })?,
    )?;

    let mut records = OwnedListSexp::new(run.records.len(), false)?;
    for (i, r) in run.records.iter().enumerate() {
        let mut rd = OwnedListSexp::new(3, true)?;
        rd.set_name_and_value(0, "eval_index", OwnedRealSexp::try_from_scalar(r.eval_index as f64)?)?;
        rd.set_name_and_value(1, "objectives", OwnedRealSexp::try_from_slice(r.objectives.as_slice())?)?;
        let geno: Sexp = match &r.genotype {
            MoArchiveGenotype::Float(xs) => OwnedRealSexp::try_from_slice(xs.as_slice())?.into(),
            MoArchiveGenotype::Binary(bits) => OwnedLogicalSexp::try_from_slice(bits.as_slice())?.into(),
        };
        rd.set_name_and_value(2, "genotype", geno)?;
        records.set_value(i, rd)?;
    }
    out.set_name_and_value(6, "records", records)?;

    let archive = run.archive_at(evals);
    let mut archive_list = OwnedListSexp::new(archive.len(), false)?;
    for (i, row) in archive.iter().enumerate() {
        archive_list.set_value(i, OwnedRealSexp::try_from_slice(row.as_slice())?)?;
    }
    out.set_name_and_value(7, "archive", archive_list)?;

    Ok(out.into())
}
