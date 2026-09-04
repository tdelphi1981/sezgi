//! CEC 2022 + CEC 2014 + CEC 2017 + TSP direct-evaluation bindings
//! (`sz_cec2022_evaluate` / `sz_cec2022_f_star` / `sz_cec2014_evaluate` /
//! `sz_cec2014_f_star` / `sz_cec2017_evaluate` / `sz_cec2017_f_star` /
//! `sz_tsp_load` / `sz_tsp_tour_length`) -- M3-3 Task 10 (CEC 2022 + TSP),
//! extended by M3-6 Task 10 (CEC 2014 + CEC 2017).
//!
//! Mirrors py-sezgi's `sezgi.problems` module (M3-3 Task 9 for CEC 2022 +
//! TSP, M3-6 Task 9 for CEC 2014 + CEC 2017; `py-sezgi/src/lib.rs`)
//! key-for-key for these direct-evaluation entry points.
//! `sz_preset_ga_perm` / `sz_solve_tsp` / `sz_solve_cec2022` /
//! `sz_solve_cec2014` / `sz_solve_cec2017` (the solve()-eligible half of
//! py-sezgi's `sezgi.problems`/`sezgi.presets`) live in `solve.rs` instead,
//! alongside every other `sz_preset_*` / `sz_solve_*` binding -- py-sezgi
//! keeps direct-eval and solve-integration in one Python namespace, but
//! r-sezgi's `sz_solve_bbob` precedent already puts every solve-machinery
//! binding in `solve.rs`, so this file only takes the direct-eval half.
//!
//! CEC 2014's `fid` domain is `1..=30`; CEC 2017's is `1` union `3..=30`
//! (`fid = 2`, "Sum of Different Powers", was officially withdrawn -- see
//! `sz_cec2017_evaluate`/`sz_cec2017_f_star`'s own docs, and
//! `sezgi_problems::Cec2017Error::Withdrawn`'s doc for the full ruling).
//! Both suites vendor only `dim` in `{10,30}` -- UNLIKE CEC 2022's `{2,10,20}`
//! -- so no hybrid/`dim=2` special case applies to either new suite.
//!
//! ## Index-convention decision (tour arguments/outputs; NOT x/coordinate
//! values, which are unchanged)
//! r-sezgi uses 1-based city numbering throughout -- UNLIKE py-sezgi's
//! 0-based `tour`/`coords` convention -- mirroring the established
//! `sz_stats_plackett_luce`/`sz_bayesian_plackett_luce` `rankings` 1-based
//! item-id precedent (`stats.rs`'s `rankings_from_list`), not py-sezgi's own
//! 0-based choice. Concretely: `sz_tsp_tour_length`'s `tour` argument is a
//! permutation of `1:n_cities`; `sz_tsp_load`'s `coords` matrix row `i` is
//! city `i`; `sz_solve_tsp`'s `best_x` (in `solve.rs`) is a 1-based
//! permutation of `1:n_cities` -- the same "city i" numbering everywhere,
//! consistent with row `i` of `coords`, TSPLIB node `i` (TSPLIB node numbers
//! are themselves 1-based -- see `tsp.rs`'s module doc -- so
//! `crates/problems/data/tsplib/berlin52.opt.tour`'s TOUR_SECTION can be
//! used verbatim as an R tour with no conversion, unlike Python's, which
//! needs `-1`), and the underlying Rust genotype index `i - 1`. Every
//! conversion point is `-1`/`+1` at the R binding boundary only --
//! `sezgi_problems::Tsp` itself is untouched (still 0-based
//! `BlockValues::Perm`).

use savvy::{
    savvy, savvy_err, NullSexp, OwnedListSexp, OwnedRealSexp, OwnedStringSexp, RealSexp, Sexp,
};
use sezgi_core::problem::Problem;
use sezgi_core::space::{BlockValues, Genotype};
use sezgi_problems::{Cec2014, Cec2017, Cec2022, Tsp, TspError};

/// Casts a non-negative-checked, WHOLE-NUMBER-checked `f64` (as passed
/// from R, which has no native unsigned integer type) to `u64`, rejecting
/// negative, non-finite, or fractional values. Duplicated from the
/// identically-named private helpers in `solve.rs` / `session.rs` /
/// `experiment.rs` / `stats.rs` / `bias.rs` / `mo.rs` -- no shared private
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

/// Same contract as [`f64_to_u64`], returning `usize` -- for `dim`/
/// `n_cities`-typed params, which are consumed as `usize` on the Rust side.
fn f64_to_usize(name: &str, x: f64) -> savvy::Result<usize> {
    f64_to_u64(name, x).map(|v| v as usize)
}

/// Same contract as [`f64_to_u64`], returning `u32` -- for `fid`-typed
/// params.
fn f64_to_u32(name: &str, x: f64) -> savvy::Result<u32> {
    f64_to_u64(name, x).map(|v| v as u32)
}

/// Shared `name_or_text` resolver for `sz_tsp_load`/`sz_tsp_tour_length`
/// (and `sz_solve_tsp` in `solve.rs`, which duplicates the vendored-only
/// half): tries `Tsp::vendored(name_or_text)` first (a vendored instance
/// name), and on [`TspError::UnknownVendored`] only, falls back to parsing
/// `name_or_text` as raw TSPLIB `.tsp` file text via `Tsp::from_tsplib`.
/// Any other [`TspError`] (from either path) is surfaced directly. Mirrors
/// py-sezgi's identically-named `load_tsp` helper exactly.
fn load_tsp(name_or_text: &str) -> savvy::Result<Tsp> {
    match Tsp::vendored(name_or_text) {
        Ok(t) => Ok(t),
        Err(TspError::UnknownVendored(_)) => {
            Tsp::from_tsplib(name_or_text).map_err(|e| savvy_err!("{e}"))
        }
        Err(e) => Err(savvy_err!("{e}")),
    }
}

/// Direct, one-shot evaluation of a CEC 2022 function at `x`, bypassing
/// `sz_solve_bbob`-style budget/engine machinery entirely -- binds
/// [`Cec2022::new`] + [`Cec2022::evaluate_batch`] exactly. See
/// `Cec2022::new`'s own doc for the exact `fid`/`dim` domain.
///
/// @param fid CEC 2022 function id, `1..=12` (double, cast to `u32`).
/// @param dim Problem dimension, one of `2`, `10`, `20` (double, cast to
///   `usize`); `dim = 2` is additionally rejected for a hybrid function
///   (`fid` 6-8).
/// @param x A numeric vector of exactly `dim` coordinates.
/// @returns A numeric scalar.
///
/// # Errors
/// A savvy error for `fid` outside `1..=12`, `dim` outside `{2,10,20}`,
/// `dim = 2` for a hybrid function, or `length(x) != dim`.
/// @export
#[savvy]
fn sz_cec2022_evaluate(fid: f64, dim: f64, x: RealSexp) -> savvy::Result<Sexp> {
    let fid_u = f64_to_u32("fid", fid)?;
    let dim_u = f64_to_usize("dim", dim)?;
    let p = Cec2022::new(fid_u, dim_u).map_err(|e| savvy_err!("{e}"))?;

    let xs = x.as_slice();
    if xs.len() != dim_u {
        return Err(savvy_err!(
            "x must have exactly {} coordinates (dim={}), got {}",
            dim_u,
            dim_u,
            xs.len()
        ));
    }
    let g = Genotype { blocks: vec![BlockValues::Float(xs.to_vec())] };
    let v = p.evaluate_batch(&[g])[0];
    v.try_into()
}

/// The report's pinned `F_i*` bias for a CEC 2022 function -- binds
/// [`Cec2022::f_star`] (module doc section 1.2's table). `f_star` does not
/// depend on `dim`, so an internal probe `dim = 10` is used purely to
/// validate `fid` (every `fid` in `1..=12` accepts `dim = 10`, hybrids
/// included).
///
/// @param fid CEC 2022 function id, `1..=12` (double, cast to `u32`).
/// @returns A numeric scalar.
///
/// # Errors
/// A savvy error if `fid` is outside `1..=12`.
/// @export
#[savvy]
fn sz_cec2022_f_star(fid: f64) -> savvy::Result<Sexp> {
    let fid_u = f64_to_u32("fid", fid)?;
    let p = Cec2022::new(fid_u, 10).map_err(|e| savvy_err!("{e}"))?;
    p.f_star().try_into()
}

/// Direct, one-shot evaluation of a CEC 2014 (Liang, Qu & Suganthan 2013)
/// function at `x`, bypassing `sz_solve_bbob`-style budget/engine machinery
/// entirely -- binds [`Cec2014::new`] + [`Cec2014::evaluate_batch`] exactly.
/// Mirrors `sz_cec2022_evaluate` (M3-6 Task 10 -- see py-sezgi's
/// `sezgi.problems.cec2014_evaluate` for the identical binding on the Python
/// side). See [`Cec2014::new`]'s own doc for the exact `fid`/`dim` domain.
///
/// @param fid CEC 2014 function id, `1..=30` (double, cast to `u32`).
/// @param dim Problem dimension, one of `10`, `30` (double, cast to
///   `usize`).
/// @param x A numeric vector of exactly `dim` coordinates.
/// @returns A numeric scalar.
///
/// # Errors
/// A savvy error for `fid` outside `1..=30`, `dim` outside `{10,30}`, or
/// `length(x) != dim`.
/// @export
#[savvy]
fn sz_cec2014_evaluate(fid: f64, dim: f64, x: RealSexp) -> savvy::Result<Sexp> {
    let fid_u = f64_to_u32("fid", fid)?;
    let dim_u = f64_to_usize("dim", dim)?;
    let p = Cec2014::new(fid_u, dim_u).map_err(|e| savvy_err!("{e}"))?;

    let xs = x.as_slice();
    if xs.len() != dim_u {
        return Err(savvy_err!(
            "x must have exactly {} coordinates (dim={}), got {}",
            dim_u,
            dim_u,
            xs.len()
        ));
    }
    let g = Genotype { blocks: vec![BlockValues::Float(xs.to_vec())] };
    let v = p.evaluate_batch(&[g])[0];
    v.try_into()
}

/// The report's pinned `F_i*` bias for a CEC 2014 function -- binds
/// [`Cec2014::f_star`] (`F_i* = 100*fid`). Does not depend on `dim`, so an
/// internal probe `dim = 10` is used purely to validate `fid`.
///
/// @param fid CEC 2014 function id, `1..=30` (double, cast to `u32`).
/// @returns A numeric scalar.
///
/// # Errors
/// A savvy error if `fid` is outside `1..=30`.
/// @export
#[savvy]
fn sz_cec2014_f_star(fid: f64) -> savvy::Result<Sexp> {
    let fid_u = f64_to_u32("fid", fid)?;
    let p = Cec2014::new(fid_u, 10).map_err(|e| savvy_err!("{e}"))?;
    p.f_star().try_into()
}

/// Direct, one-shot evaluation of a CEC 2017 (Awad, Ali, Liang, Qu &
/// Suganthan 2016) function at `x`, bypassing `sz_solve_bbob`-style
/// budget/engine machinery entirely -- binds [`Cec2017::new`] +
/// [`Cec2017::evaluate_batch`] exactly. Mirrors `sz_cec2014_evaluate` (M3-6
/// Task 10 -- see py-sezgi's `sezgi.problems.cec2017_evaluate` for the
/// identical binding on the Python side). See [`Cec2017::new`]'s own doc for
/// the exact `fid`/`dim` domain.
///
/// @param fid CEC 2017 function id, `1` or `3..=30` (double, cast to `u32`);
///   `fid = 2` ("Sum of Different Powers") was officially withdrawn from the
///   suite.
/// @param dim Problem dimension, one of `10`, `30` (double, cast to
///   `usize`).
/// @param x A numeric vector of exactly `dim` coordinates.
/// @returns A numeric scalar.
///
/// # Errors
/// A savvy error for `fid` outside `{1} union {3..=30}`, `dim` outside
/// `{10,30}`, or `length(x) != dim`. `fid = 2` raises a dedicated error --
/// the Rust [`sezgi_problems::Cec2017Error::Withdrawn`] message is surfaced
/// VERBATIM, distinct from an ordinary out-of-range `fid`.
/// @export
#[savvy]
fn sz_cec2017_evaluate(fid: f64, dim: f64, x: RealSexp) -> savvy::Result<Sexp> {
    let fid_u = f64_to_u32("fid", fid)?;
    let dim_u = f64_to_usize("dim", dim)?;
    let p = Cec2017::new(fid_u, dim_u).map_err(|e| savvy_err!("{e}"))?;

    let xs = x.as_slice();
    if xs.len() != dim_u {
        return Err(savvy_err!(
            "x must have exactly {} coordinates (dim={}), got {}",
            dim_u,
            dim_u,
            xs.len()
        ));
    }
    let g = Genotype { blocks: vec![BlockValues::Float(xs.to_vec())] };
    let v = p.evaluate_batch(&[g])[0];
    v.try_into()
}

/// The report's pinned `F_i*` bias for a CEC 2017 function -- binds
/// [`Cec2017::f_star`] (`F_i* = 100*fid`, the fid-gapped C dispatch bias, not
/// the report's contiguous renumbering -- see `Cec2017::new`'s own module
/// doc). Does not depend on `dim`, so an internal probe `dim = 10` is used
/// purely to validate `fid`.
///
/// @param fid CEC 2017 function id, `1` or `3..=30` (double, cast to `u32`).
/// @returns A numeric scalar.
///
/// # Errors
/// A savvy error if `fid` is outside `{1} union {3..=30}` (`fid = 2`
/// included, via the [`sezgi_problems::Cec2017Error::Withdrawn`] message).
/// @export
#[savvy]
fn sz_cec2017_f_star(fid: f64) -> savvy::Result<Sexp> {
    let fid_u = f64_to_u32("fid", fid)?;
    let p = Cec2017::new(fid_u, 10).map_err(|e| savvy_err!("{e}"))?;
    p.f_star().try_into()
}

/// Loads a TSPLIB `EUC_2D` instance, either a vendored instance name
/// (`"berlin52"`, `"eil51"`, `"st70"`) or raw TSPLIB file text (see
/// [`load_tsp`]).
///
/// @param name_or_text A vendored instance name, or raw TSPLIB `.tsp` file
///   text.
/// @returns A named list with `name` (character scalar), `n_cities` (double),
///   `coords` (an `n_cities` x 2 numeric matrix, row `i` is city `i`'s
///   `(x, y)` coordinate pair -- see this module's own doc, "Index-convention
///   decision"), `known_optimum` (double, or `NULL` for an instance parsed
///   from raw text rather than a vendored name).
///
/// # Errors
/// A savvy error for any [`TspError`] (unknown vendored name that also fails
/// to parse as TSPLIB text, malformed TSPLIB text, unsupported
/// `EDGE_WEIGHT_TYPE`, ...).
/// @export
#[savvy]
fn sz_tsp_load(name_or_text: &str) -> savvy::Result<Sexp> {
    let t = load_tsp(name_or_text)?;
    let n = t.n_cities();

    // R matrix storage is column-major: `data[col * nrow + row]`.
    let mut coords_flat = vec![0.0_f64; n * 2];
    for (i, &(x, y)) in t.coords().iter().enumerate() {
        coords_flat[i] = x;
        coords_flat[n + i] = y;
    }
    let mut coords = OwnedRealSexp::try_from_slice(coords_flat.as_slice())?;
    coords.set_dim(&[n, 2])?;

    let mut out = OwnedListSexp::new(4, true)?;
    out.set_name_and_value(0, "name", OwnedStringSexp::try_from(t.name())?)?;
    out.set_name_and_value(1, "n_cities", OwnedRealSexp::try_from_scalar(n as f64)?)?;
    out.set_name_and_value(2, "coords", coords)?;
    // Same fix as `sz_solve_r_problem`'s result tail (`solve.rs`; see its
    // comment for the full explanation): savvy 0.10.2's `set_name_and_value`
    // allocates `set_name`'s CHARSXP unconditionally on every call before
    // attaching `v` -- so `known_optimum`'s already-bare `Sexp` (both match
    // arms below drop their `Owned*Sexp` token, or hold `NullSexp`, via
    // `.into()`) would sit unprotected across that guaranteed allocation.
    // Fixed by attaching via `set_value` first (nothing allocates between
    // `known_optimum`'s construction and `SET_VECTOR_ELT`) and naming via
    // `set_name` second -- AND, unlike the single-field `solve.rs` sites,
    // `known_optimum` is constructed HERE, immediately before its own
    // `set_value` call, rather than earlier alongside `coords` (its
    // original position): constructing it earlier would leave it
    // unprotected across the `name`/`n_cities`/`coords` fields' own
    // `set_name_and_value` calls above, each of which is an equally
    // unconditional allocation.
    let known_optimum: Sexp = match t.known_optimum() {
        Some(v) => OwnedRealSexp::try_from_scalar(v)?.into(),
        None => NullSexp.into(),
    };
    out.set_value(3, known_optimum)?;
    out.set_name(3, "known_optimum")?;
    Ok(out.into())
}

/// Closed-tour length of a 1-based `tour` (a permutation of `1:n_cities` --
/// see this module's own doc, "Index-convention decision") on the instance
/// named/parsed by `name_or_text` (see [`load_tsp`]), via
/// [`Tsp::evaluate_batch`]'s `nint`-rounded `EUC_2D` sum (`tsp.rs`'s module
/// doc).
///
/// UNLIKE `Tsp::evaluate_batch` itself (which returns `f64::INFINITY` for a
/// malformed genotype, since genotype validity is normally the engine's
/// `SearchSpace::validate` job, not `Tsp`'s own) -- this binding validates
/// `tour` itself (exact length, every entry in `1..=n_cities`, no repeats)
/// and raises a precise error instead, since a `tour` coming directly from R
/// has no engine-side validation gate in front of it.
///
/// @param name_or_text A vendored instance name, or raw TSPLIB `.tsp` file
///   text (same as `sz_tsp_load`).
/// @param tour A numeric vector, a permutation of `1:n_cities` (1-based).
/// @returns A numeric scalar.
///
/// # Errors
/// A savvy error for any [`TspError`] resolving `name_or_text`, or if `tour`
/// is not a permutation of `1:n_cities` (wrong length, an out-of-range
/// entry, or a repeated entry).
/// @export
#[savvy]
fn sz_tsp_tour_length(name_or_text: &str, tour: RealSexp) -> savvy::Result<Sexp> {
    let t = load_tsp(name_or_text)?;
    let n = t.n_cities();

    let tour_slice = tour.as_slice();
    if tour_slice.len() != n {
        return Err(savvy_err!(
            "tour must have exactly {} entries (one per city), got {}",
            n,
            tour_slice.len()
        ));
    }

    let mut seen = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for &v in tour_slice {
        let city1 = f64_to_usize("tour entry", v)?;
        if city1 < 1 || city1 > n {
            return Err(savvy_err!(
                "tour entry {} out of range for a {}-city instance (expected 1..{})",
                city1,
                n,
                n
            ));
        }
        let city0 = city1 - 1;
        if seen[city0] {
            return Err(savvy_err!(
                "tour has a repeated city {}; not a valid permutation",
                city1
            ));
        }
        seen[city0] = true;
        order.push(city0 as u32);
    }

    let g = Genotype { blocks: vec![BlockValues::Perm(order)] };
    let v = t.evaluate_batch(&[g])[0];
    v.try_into()
}
