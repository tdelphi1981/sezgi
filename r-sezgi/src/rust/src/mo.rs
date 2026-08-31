//! Multi-objective bindings (`sz_nsga2` / `sz_mo_hypervolume_2d` /
//! `sz_mo_igd` / `sz_mo_pareto_front`) -- M3-2 Task 10.
//!
//! Binds T6's NSGA-II runner (`sezgi_components::nsga2::nsga2_run`), the
//! ZDT/DTLZ benchmark suites (`sezgi_problems::{Zdt, Dtlz}`), and the exact
//! 2-objective hypervolume / IGD indicators (`sezgi_stats::{hypervolume_2d,
//! igd}`) -- the exact same Rust surface py-sezgi's `sezgi.mo` module binds
//! (M3-2 Task 9, `py-sezgi/src/lib.rs`'s "Multi-objective bindings"
//! section; see also `.superpowers/sdd/2026-08-30-sezgi-m3-2/task-9-report.md`
//! for the full dict-shape provenance this file mirrors 1:1 by key name).
//!
//! Every scalar/vector f64 is passed through EXACTLY as savvy already
//! returns bit-for-bit for a Rust `f64` -- no rounding/formatting anywhere
//! in this file, so a same-seed call is bit-identical to both a repeat R
//! call and to py-sezgi's own binding (both call the identical seeded Rust
//! core; asserted directly in `tests/testthat/test-mo.R`'s cross-language
//! fixture).
//!
//! ## Problem-string mapping (shared by `sz_nsga2` and `sz_mo_pareto_front`,
//! via `mo_problem_from_str`)
//! Identical to py-sezgi's own mapping (see that module's doc): `"zdt1"`,
//! `"zdt2"`, `"zdt3"`, `"zdt4"`, `"zdt6"` (ZDT5 is a binary-coded problem,
//! out of scope) and `"dtlz1"`..`"dtlz7"`. `m` (number of objectives) is
//! DTLZ-only and REQUIRED there; passing `m` for a zdt problem is an error
//! (zdt problems are always 2-objective by construction, so a
//! caller-supplied `m` could never be honored -- rejecting it surfaces the
//! mistake instead of silently ignoring the argument).
//!
//! ## Container-idiom decisions (key NAMES are the 1:1 mandate; containers
//! follow r-sezgi's own precedent, not py-sezgi's)
//! - `individuals`/`objectives` (`sz_nsga2` output) and `sz_mo_pareto_front`'s
//!   return: a `list` of numeric vectors, one per point/individual --
//!   mirrors `sz_bias_structural()`'s `final_positions` convention
//!   (`bias.rs`'s `structural_result_list`), NOT a matrix.
//! - `front`/`reference_front` (`sz_mo_hypervolume_2d`/`sz_mo_igd` INPUTS):
//!   an R-idiomatic numeric matrix, rows = points -- mirrors
//!   `sz_stats_friedman()`'s `m` input convention (`matrix_to_rows` in
//!   `stats.rs`), NOT a list of rows (unlike py-sezgi's `list[list[float]]`).
//!   `matrix_to_rows` is duplicated here rather than imported (`stats.rs`'s
//!   copy is a private fn -- no shared private cross-module import, same
//!   rule this crate's other duplicated helpers already document).
//! - `front0` (`sz_nsga2` output): converted to 1-based indices (R's native
//!   convention) into `individuals`/`objectives` -- py-sezgi's `front0` is
//!   0-based; this binding converts so `result$individuals[[result$front0[i]]]`
//!   works directly for an R caller, rather than requiring `+ 1` at every
//!   call site.

use savvy::{savvy, savvy_err, NullSexp, OwnedListSexp, OwnedRealSexp, RealSexp, Sexp};
use sezgi_components::nsga2::{nsga2_run, MoRunResult, Nsga2Config};
use sezgi_core::mo::MoProblem;
use sezgi_core::space::{BlockValues, Genotype};
use sezgi_problems::{Dtlz, Zdt};
use sezgi_stats::{hypervolume_2d, igd};

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
/// `pop_size`-typed params, which are consumed as `usize` on the Rust side.
fn f64_to_usize(name: &str, x: f64) -> savvy::Result<usize> {
    f64_to_u64(name, x).map(|v| v as usize)
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

/// Shared problem-string -> `Box<dyn MoProblem>` builder for `sz_nsga2_raw`
/// and `sz_mo_pareto_front_raw`. See this module's own doc for the full
/// mapping (identical to py-sezgi's `mo_problem_from_str`).
fn mo_problem_from_str(problem: &str, dim: usize, m: Option<usize>) -> savvy::Result<Box<dyn MoProblem>> {
    let unknown = || {
        savvy_err!(
            "unknown problem `{}` (expected one of zdt1, zdt2, zdt3, zdt4, zdt6, or dtlz1..dtlz7)",
            problem
        )
    };
    if let Some(rest) = problem.strip_prefix("zdt") {
        let which: u32 = rest.parse().map_err(|_| unknown())?;
        if m.is_some() {
            return Err(savvy_err!(
                "m is DTLZ-only (number of objectives); zdt problems are always 2-objective -- \
                 omit m (or pass m = NULL) for a zdt problem"
            ));
        }
        let p = Zdt::new(which, dim).map_err(|e| savvy_err!("{e}"))?;
        Ok(Box::new(p))
    } else if let Some(rest) = problem.strip_prefix("dtlz") {
        let which: u32 = rest.parse().map_err(|_| unknown())?;
        let m = m.ok_or_else(|| savvy_err!("m (number of objectives) is required for dtlz problems"))?;
        let p = Dtlz::new(which, m, dim).map_err(|e| savvy_err!("{e}"))?;
        Ok(Box::new(p))
    } else {
        Err(unknown())
    }
}

/// Flattens a [`Genotype`] into a single `Vec<f64>` (concatenating every
/// `Block::Float` block in order) -- identical to py-sezgi's
/// `genotype_to_flat_vec`. These are always single- or multi-Float-block
/// genotypes here, since `nsga2_run` validates an all-`Block::Float` space
/// before ever constructing one.
fn genotype_to_flat_vec(g: &Genotype) -> Vec<f64> {
    let mut out = Vec::new();
    for b in &g.blocks {
        if let BlockValues::Float(xs) = b {
            out.extend_from_slice(xs);
        }
    }
    out
}

/// Builds the named list mirroring `MoRunResult` 1:1 by field name --
/// `individuals`, `objectives`, `front0`, `evals_used` -- same as
/// py-sezgi's `mo_nsga2` dict shape. Container idiom: see this module's own
/// doc, "Container-idiom decisions".
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

    let mut out = OwnedListSexp::new(4, true)?;
    out.set_name_and_value(0, "individuals", individuals)?;
    out.set_name_and_value(1, "objectives", objectives)?;
    out.set_name_and_value(2, "front0", OwnedRealSexp::try_from_slice(front0.as_slice())?)?;
    out.set_name_and_value(
        3,
        "evals_used",
        OwnedRealSexp::try_from_scalar(r.evals_used as f64)?,
    )?;
    Ok(out)
}

/// NSGA-II run (Deb, Pratap, Agarwal & Meyarivan 2002) -- binds
/// [`sezgi_components::nsga2::nsga2_run`]. See this module's own doc for the
/// `problem`/`m` mapping.
///
/// This is the raw savvy-generated binding (required args only, optional
/// args -- `m`/`p_m` -- trailing; savvy has no way to express a non-`NULL`
/// default for a required argument in the generated signature, and requires
/// optional args last). The public R entry point with R-native defaults
/// (`m = NULL`, `seed = 0`, `eta_c = 20.0`, `eta_m = 20.0`, `p_c = 0.9`,
/// `p_m = NULL`) is the hand-written wrapper `sz_nsga2()` in `R/mo.R`, which
/// calls this function -- same raw/wrapper pattern as `sz_bias_structural()`
/// / `sz_bias_structural_raw()`.
///
/// @param problem One of `"zdt1"`, `"zdt2"`, `"zdt3"`, `"zdt4"`, `"zdt6"`,
///   or `"dtlz1"`..`"dtlz7"`.
/// @param dim Decision-space dimensionality (double, cast to `usize`).
/// @param pop_size Population size (double, cast to `usize`). Must be
///   `>= 4` and a multiple of 4 (KanGAL's double-permutation tournament
///   pairing requires it -- NOT merely "even, >= 4").
/// @param budget Total evaluation budget (double, cast to `u64`).
/// @param seed Master RNG seed (double, cast to `u64`).
/// @param eta_c SBX distribution index. Paper default: `20.0`.
/// @param eta_m Polynomial-mutation distribution index. Paper default:
///   `20.0`.
/// @param p_c SBX crossover probability. Paper default: `0.9`.
/// @param m Optional number of objectives (double, cast to `usize`).
///   REQUIRED for dtlz problems; must be `NULL` for zdt problems.
/// @param p_m Optional per-variable mutation probability. `NULL` resolves
///   on the Rust side to `1 / n_variables` (the paper's own default), never
///   re-derived here.
/// @returns A named list with `individuals`, `objectives`, `front0`,
///   `evals_used` -- see this module's own doc.
///
/// # Errors
/// A savvy error for an unrecognized `problem` string, an `m` given for a
/// zdt problem, a missing `m` for a dtlz problem, or any
/// [`sezgi_components::nsga2::Nsga2Error`] (including `pop_size` failing
/// the `>= 4 && pop_size % 4 == 0` check).
/// @noRd
#[savvy]
#[allow(clippy::too_many_arguments)]
fn sz_nsga2_raw(
    problem: &str,
    dim: f64,
    pop_size: f64,
    budget: f64,
    seed: f64,
    eta_c: f64,
    eta_m: f64,
    p_c: f64,
    m: Option<f64>,
    p_m: Option<f64>,
) -> savvy::Result<Sexp> {
    let dim_u = f64_to_usize("dim", dim)?;
    let m_u = match m {
        Some(v) => Some(f64_to_usize("m", v)?),
        None => None,
    };
    let prob = mo_problem_from_str(problem, dim_u, m_u)?;

    let cfg = Nsga2Config {
        pop_size: f64_to_usize("pop_size", pop_size)?,
        budget: f64_to_u64("budget", budget)?,
        seed: f64_to_u64("seed", seed)?,
        eta_c,
        eta_m,
        p_c,
        p_m,
        // p_c_bin/p_m_bin (M3-7 Task 3): not yet exposed as R args (every
        // problem this binding constructs is a real-coded ZDT/DTLZ space)
        // -- fixed inert placeholders, unused on the all-Float path these
        // bindings drive.
        p_c_bin: 0.9,
        p_m_bin: None,
    };

    let result = nsga2_run(prob.as_ref(), &cfg).map_err(|e| savvy_err!("{e}"))?;
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
/// Same `problem`/`m` mapping as `sz_nsga2_raw` (see this module's own doc).
///
/// This is the raw savvy-generated binding (required args only, `m`
/// trailing since it is optional). The public R entry point with R-native
/// defaults (`m = NULL`) is the hand-written wrapper `sz_mo_pareto_front()`
/// in `R/mo.R`, which calls this function -- same raw/wrapper pattern as
/// `sz_nsga2()` / `sz_nsga2_raw()`.
///
/// @param problem Same mapping as `sz_nsga2_raw`.
/// @param dim Decision-space dimensionality (double, cast to `usize`).
/// @param n Number of front points to sample (double, cast to `usize`).
/// @param m Optional number of objectives (double, cast to `usize`).
///   REQUIRED for dtlz, must be `NULL` for zdt -- same rule as
///   `sz_nsga2_raw`.
/// @returns A `list` of `n` numeric vectors (see this module's own doc,
///   "Container-idiom decisions"), or `NULL` when the problem has no known
///   analytic front sample at this `m` (verified case: DTLZ5/DTLZ6 with
///   `m > 3`) -- mirrors py-sezgi's `sezgi.mo.pareto_front()` return
///   exactly.
///
/// # Errors
/// Same as `sz_nsga2_raw`'s problem-construction errors.
/// @noRd
#[savvy]
fn sz_mo_pareto_front_raw(problem: &str, dim: f64, n: f64, m: Option<f64>) -> savvy::Result<Sexp> {
    let dim_u = f64_to_usize("dim", dim)?;
    let n_u = f64_to_usize("n", n)?;
    let m_u = match m {
        Some(v) => Some(f64_to_usize("m", v)?),
        None => None,
    };
    let prob = mo_problem_from_str(problem, dim_u, m_u)?;

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
