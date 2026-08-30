//! Bias-scanning bindings (`sz_bias_structural` / `sz_bias_central` /
//! `sz_bias_report`) -- M3-1 Task 9.
//!
//! Mirrors `crates/bias`'s public structs 1:1 by field name, matching
//! py-sezgi's own "Bias-scanning bindings" section
//! (`py-sezgi/src/lib.rs`, M3-1 Task 8) key-for-key: flat `verdict`
//! (`"no_evidence"`/`"evidence"`) + `detail` (`NULL`/character scalar)
//! siblings on every list that carries a [`BiasVerdict`], rather than a
//! nested verdict sub-list -- see that section's own doc for the rationale
//! (this module's own `verdict` field's decision was already made there;
//! this file only needs to mirror it, not re-decide it).
//!
//! T6 (the Rajwar-Deep signature-bias test) is DEFERRED in `sezgi-bias`
//! itself (no `signature_scan` exists -- see `crates/bias/src/report.rs`'s
//! module doc, "T6 (signature test): deferred"); `sz_bias_report()`'s
//! `signature` element is therefore always R `NULL`, mirroring
//! `BiasReport::signature`'s own always-`None` `Option<BiasVerdict>` today
//! and py-sezgi's `bias_report()["signature"] is None`.
//!
//! Every scalar statistic (`d`, `p_value`, `a2`, `w_statistic`, `z`,
//! `effect`, gap/position entries) is passed through as the exact `f64`
//! savvy already returns bit-for-bit for a Rust `f64` -- no
//! rounding/formatting anywhere in this file, so a same-seed call is
//! bit-identical to both a repeat R call and to py-sezgi's own binding
//! (both call the identical seeded Rust core).

use savvy::{
    savvy, savvy_err, NullSexp, OwnedListSexp, OwnedRealSexp, OwnedStringSexp, RealSexp, Sexp,
};
use sezgi_bias::{
    bias_report, central_bias_scan, structural_bias_scan, BiasReportConfig, BiasVerdict,
    CentralBiasConfig, CentralBiasResult, StructuralBiasConfig, StructuralBiasResult,
};
use sezgi_core::spec::AlgorithmSpec;
use sezgi_stats::uniformity::{AdResult, KsResult};
use sezgi_stats::{WilcoxonMethod, WilcoxonResult};

/// Casts a non-negative-checked, WHOLE-NUMBER-checked `f64` (as passed
/// from R, which has no native unsigned integer type) to `u64`, rejecting
/// negative, non-finite, or fractional values. Duplicated from the
/// identically-named private helpers in `solve.rs` / `session.rs` /
/// `experiment.rs` / `stats.rs` -- no shared private cross-module import,
/// same "no shared private crate imports" rule those files already
/// document.
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

/// Same contract as [`f64_to_u64`], returning `usize` -- for `dim`-typed
/// params, which are consumed as `usize` on the Rust side.
fn f64_to_usize(name: &str, x: f64) -> savvy::Result<usize> {
    f64_to_u64(name, x).map(|v| v as usize)
}

/// Same contract as [`f64_to_u64`], returning `u32` -- for `runs`/
/// `runs_per`-typed params.
fn f64_to_u32(name: &str, x: f64) -> savvy::Result<u32> {
    f64_to_u64(name, x).map(|v| v as u32)
}

/// Converts an R numeric vector (`fids`/`instances_shifted`) into
/// `Vec<u32>`, applying [`f64_to_u64`]'s strict whole-number check to every
/// element -- a fractional or negative fid/instance is rejected, not
/// silently truncated.
fn f64_slice_to_u32_vec(name: &str, xs: &[f64]) -> savvy::Result<Vec<u32>> {
    xs.iter().map(|&x| f64_to_u32(name, x)).collect()
}

/// Builds the named-list mirroring py-sezgi's `per_dim_ks` entry dict:
/// `d`, `p_value`, `n`.
fn ks_result_list(r: &KsResult) -> savvy::Result<OwnedListSexp> {
    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "d", OwnedRealSexp::try_from_scalar(r.d)?)?;
    out.set_name_and_value(1, "p_value", OwnedRealSexp::try_from_scalar(r.p_value)?)?;
    out.set_name_and_value(2, "n", OwnedRealSexp::try_from_scalar(r.n as f64)?)?;
    Ok(out)
}

/// Builds the named-list mirroring py-sezgi's `per_dim_ad` entry dict:
/// `a2`, `p_value`, `n`.
fn ad_result_list(r: &AdResult) -> savvy::Result<OwnedListSexp> {
    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "a2", OwnedRealSexp::try_from_scalar(r.a2)?)?;
    out.set_name_and_value(1, "p_value", OwnedRealSexp::try_from_scalar(r.p_value)?)?;
    out.set_name_and_value(2, "n", OwnedRealSexp::try_from_scalar(r.n as f64)?)?;
    Ok(out)
}

/// Builds the named-list mirroring py-sezgi's `wilcoxon` sub-dict:
/// `w_statistic`, `z`, `p_value`, `n_effective`, `method` -- same shape
/// `sz_stats_wilcoxon()` (`stats.rs`) already returns.
fn wilcoxon_result_list(r: &WilcoxonResult) -> savvy::Result<OwnedListSexp> {
    let mut out = OwnedListSexp::new(5, true)?;
    out.set_name_and_value(0, "w_statistic", OwnedRealSexp::try_from_scalar(r.w_statistic)?)?;
    out.set_name_and_value(1, "z", OwnedRealSexp::try_from_scalar(r.z)?)?;
    out.set_name_and_value(2, "p_value", OwnedRealSexp::try_from_scalar(r.p_value)?)?;
    out.set_name_and_value(
        3,
        "n_effective",
        OwnedRealSexp::try_from_scalar(r.n_effective as f64)?,
    )?;
    let method = match r.method {
        WilcoxonMethod::Exact => "exact",
        WilcoxonMethod::NormalApprox => "normal_approx",
    };
    out.set_name_and_value(4, "method", OwnedStringSexp::try_from(method)?)?;
    Ok(out)
}

/// Builds the `(verdict, detail)` `Sexp` pair shared by every list this
/// module builds from a [`BiasVerdict`]: `verdict` is the character scalar
/// `"no_evidence"`/`"evidence"`; `detail` is R `NULL` for `NoEvidence`, the
/// `Evidence::detail` character scalar otherwise -- mirrors py-sezgi's own
/// `set_verdict` helper exactly (flat siblings, not a nested sub-list).
fn verdict_detail_sexp(v: &BiasVerdict) -> savvy::Result<(Sexp, Sexp)> {
    match v {
        BiasVerdict::NoEvidence => {
            let verdict: Sexp = OwnedStringSexp::try_from("no_evidence")?.into();
            let detail: Sexp = NullSexp.into();
            Ok((verdict, detail))
        }
        BiasVerdict::Evidence { detail } => {
            let verdict: Sexp = OwnedStringSexp::try_from("evidence")?.into();
            let detail_sexp: Sexp = OwnedStringSexp::try_from(detail.as_str())?.into();
            Ok((verdict, detail_sexp))
        }
    }
}

/// Builds the named list mirroring py-sezgi's `structural_result_to_dict`:
/// `per_dim_ks`, `per_dim_ad`, `holm_rejections_ks`, `holm_rejections_ad`,
/// `verdict`, `detail`, `final_positions` (a list of `runs` numeric
/// vectors, each length `dim` -- the R-native analogue of Python's list of
/// lists).
fn structural_result_list(r: &StructuralBiasResult) -> savvy::Result<OwnedListSexp> {
    let mut ks_list = OwnedListSexp::new(r.per_dim_ks.len(), false)?;
    for (i, k) in r.per_dim_ks.iter().enumerate() {
        ks_list.set_value(i, ks_result_list(k)?)?;
    }
    let mut ad_list = OwnedListSexp::new(r.per_dim_ad.len(), false)?;
    for (i, a) in r.per_dim_ad.iter().enumerate() {
        ad_list.set_value(i, ad_result_list(a)?)?;
    }
    let mut positions = OwnedListSexp::new(r.final_positions.len(), false)?;
    for (i, row) in r.final_positions.iter().enumerate() {
        positions.set_value(i, OwnedRealSexp::try_from_slice(row.as_slice())?)?;
    }
    let (verdict, detail) = verdict_detail_sexp(&r.verdict)?;

    let mut out = OwnedListSexp::new(7, true)?;
    out.set_name_and_value(0, "per_dim_ks", ks_list)?;
    out.set_name_and_value(1, "per_dim_ad", ad_list)?;
    out.set_name_and_value(
        2,
        "holm_rejections_ks",
        OwnedRealSexp::try_from_scalar(r.holm_rejections_ks as f64)?,
    )?;
    out.set_name_and_value(
        3,
        "holm_rejections_ad",
        OwnedRealSexp::try_from_scalar(r.holm_rejections_ad as f64)?,
    )?;
    out.set_name_and_value(4, "verdict", verdict)?;
    out.set_name_and_value(5, "detail", detail)?;
    out.set_name_and_value(6, "final_positions", positions)?;
    Ok(out)
}

/// Builds the named list mirroring py-sezgi's `central_result_to_dict`:
/// `gap_centered`, `gap_shifted`, `wilcoxon`, `effect`, `verdict`, `detail`.
fn central_result_list(r: &CentralBiasResult) -> savvy::Result<OwnedListSexp> {
    let (verdict, detail) = verdict_detail_sexp(&r.verdict)?;
    let mut out = OwnedListSexp::new(6, true)?;
    out.set_name_and_value(
        0,
        "gap_centered",
        OwnedRealSexp::try_from_slice(r.gap_centered.as_slice())?,
    )?;
    out.set_name_and_value(
        1,
        "gap_shifted",
        OwnedRealSexp::try_from_slice(r.gap_shifted.as_slice())?,
    )?;
    out.set_name_and_value(2, "wilcoxon", wilcoxon_result_list(&r.wilcoxon)?)?;
    out.set_name_and_value(3, "effect", OwnedRealSexp::try_from_scalar(r.effect)?)?;
    out.set_name_and_value(4, "verdict", verdict)?;
    out.set_name_and_value(5, "detail", detail)?;
    Ok(out)
}

/// Structural-bias scan: run `spec_json` repeatedly on the f0
/// random-function null problem and test its final positions for departure
/// from uniformity -- see [`sezgi_bias::structural::structural_bias_scan`].
///
/// This is the raw savvy-generated binding (required args only; savvy has
/// no way to express a non-`NULL` default for a required argument in the
/// generated signature). The public R entry point with R-native defaults
/// (`runs = 30`, `seed = 0`) is the hand-written wrapper
/// `sz_bias_structural()` in `R/bias.R`, which calls this function -- same
/// raw/wrapper pattern as `sz_stats_bayesian_signed_rank()` /
/// `sz_stats_bayesian_signed_rank_raw()`.
///
/// @param spec_json Algorithm spec as JSON.
/// @param dim f0's domain dimensionality (>= 1, double, cast to `usize`).
/// @param budget Per-run evaluation budget (double, cast to `u64`).
/// @param runs Number of independent runs (double, cast to `u32`).
/// @param seed Master RNG seed (double, cast to `u64`).
/// @returns A named list with `per_dim_ks`, `per_dim_ad`,
///   `holm_rejections_ks`, `holm_rejections_ad`, `verdict`, `detail`,
///   `final_positions` (mirrors py-sezgi's `sezgi.bias.structural()` dict
///   keys exactly).
/// @noRd
#[savvy]
fn sz_bias_structural_raw(
    spec_json: &str,
    dim: f64,
    budget: f64,
    runs: f64,
    seed: f64,
) -> savvy::Result<Sexp> {
    let spec = AlgorithmSpec::from_json(spec_json).map_err(|e| savvy_err!("{e}"))?;
    let cfg = StructuralBiasConfig {
        runs: f64_to_u32("runs", runs)?,
        dim: f64_to_usize("dim", dim)?,
        budget: f64_to_u64("budget", budget)?,
        seed: f64_to_u64("seed", seed)?,
    };
    let r = structural_bias_scan(&spec, &cfg).map_err(|e| savvy_err!("{e}"))?;
    Ok(structural_result_list(&r)?.into())
}

/// Central-bias scan: run `spec_json` on paired centered/shifted BBOB
/// conditions and test whether its performance gap differs between them --
/// see [`sezgi_bias::central::central_bias_scan`].
///
/// This is the raw savvy-generated binding; the public R entry point with
/// R-native defaults (`fids = NULL`, `instances_shifted = NULL`,
/// `runs_per = 20`, `seed = 0`) is the hand-written wrapper
/// `sz_bias_central()` in `R/bias.R`, which calls this function. `fids`/
/// `instances_shifted` are the trailing `Option<RealSexp>` args (savvy
/// requires optional args to come last in a raw signature); `NULL` falls
/// back to `sezgi_bias::report`'s own `DEFAULT_CENTRAL_FIDS`/
/// `DEFAULT_CENTRAL_INSTANCES` (`[1, 4, 13]`/`[1, 2]`), same as py-sezgi's
/// `bias_central`.
///
/// @param spec_json Algorithm spec as JSON.
/// @param dim Problem dimension (BBOB requires >= 2, double, cast to `usize`).
/// @param budget Per-run evaluation budget (double, cast to `u64`).
/// @param runs_per Independent runs per `(fid, instance)` pair (double,
///   cast to `u32`).
/// @param seed Master RNG seed, shared by both conditions (double, cast to
///   `u64`).
/// @param fids Optional numeric vector of BBOB function ids; `NULL` uses
///   the verified defaults `c(1, 4, 13)`. Every entry must be
///   translation-invariant -- fids 5, 6, 20, 24 are rejected.
/// @param instances_shifted Optional numeric vector of BBOB instance
///   numbers; `NULL` uses the verified defaults `c(1, 2)`.
/// @returns A named list with `gap_centered`, `gap_shifted`, `wilcoxon`,
///   `effect`, `verdict`, `detail` (mirrors py-sezgi's
///   `sezgi.bias.central()` dict keys exactly).
///
/// # Errors
/// A savvy error for every [`sezgi_bias::BiasError`] case, including a fid
/// in `{5, 6, 20, 24}` (not translation-invariant).
/// @noRd
#[savvy]
fn sz_bias_central_raw(
    spec_json: &str,
    dim: f64,
    budget: f64,
    runs_per: f64,
    seed: f64,
    fids: Option<RealSexp>,
    instances_shifted: Option<RealSexp>,
) -> savvy::Result<Sexp> {
    let spec = AlgorithmSpec::from_json(spec_json).map_err(|e| savvy_err!("{e}"))?;
    let fids_v = match fids {
        Some(f) => f64_slice_to_u32_vec("fids", f.as_slice())?,
        None => sezgi_bias::report::DEFAULT_CENTRAL_FIDS.to_vec(),
    };
    let instances_v = match instances_shifted {
        Some(v) => f64_slice_to_u32_vec("instances_shifted", v.as_slice())?,
        None => sezgi_bias::report::DEFAULT_CENTRAL_INSTANCES.to_vec(),
    };
    let cfg = CentralBiasConfig {
        fids: fids_v,
        dim: f64_to_usize("dim", dim)?,
        instances_shifted: instances_v,
        runs_per: f64_to_u32("runs_per", runs_per)?,
        budget: f64_to_u64("budget", budget)?,
        seed: f64_to_u64("seed", seed)?,
    };
    let r = central_bias_scan(&spec, &cfg).map_err(|e| savvy_err!("{e}"))?;
    Ok(central_result_list(&r)?.into())
}

/// One-call bias report: runs both the structural and central bias scans
/// on `spec_json` and assembles a single report -- see
/// [`sezgi_bias::report::bias_report`].
///
/// This is the raw savvy-generated binding; the public R entry point with
/// R-native defaults (every optional knob `NULL`) is the hand-written
/// wrapper `sz_bias_report()` in `R/bias.R`, which calls this function.
/// Every `NULL` knob falls back to [`BiasReportConfig::new`]'s own
/// documented verified-method default (`structural_runs` -> 30,
/// `central_fids` -> `[1, 4, 13]`, `central_instances` -> `[1, 2]`,
/// `central_runs_per` -> 20), same as py-sezgi's `bias_report`.
///
/// @param spec_json Algorithm spec as JSON.
/// @param dim Shared dimensionality for both scans (double, cast to `usize`).
/// @param budget Shared per-run evaluation budget (double, cast to `u64`).
/// @param seed Shared master RNG seed (double, cast to `u64`).
/// @param structural_runs Optional double, cast to `u32`; `NULL` uses 30.
/// @param central_runs_per Optional double, cast to `u32`; `NULL` uses 20.
/// @param central_fids Optional numeric vector; `NULL` uses `c(1, 4, 13)`.
/// @param central_instances Optional numeric vector; `NULL` uses `c(1, 2)`.
/// @returns A named list with `structural`, `central`, `signature` (always
///   `NULL` -- T6 is deferred, see this module's own doc), `latex_summary`
///   (never contains the literal `"NaN"`), `plot_data` (mirrors py-sezgi's
///   `sezgi.bias.report()` dict keys exactly).
/// @noRd
#[savvy]
#[allow(clippy::too_many_arguments)]
fn sz_bias_report_raw(
    spec_json: &str,
    dim: f64,
    budget: f64,
    seed: f64,
    structural_runs: Option<f64>,
    central_runs_per: Option<f64>,
    central_fids: Option<RealSexp>,
    central_instances: Option<RealSexp>,
) -> savvy::Result<Sexp> {
    let spec = AlgorithmSpec::from_json(spec_json).map_err(|e| savvy_err!("{e}"))?;
    let mut cfg = BiasReportConfig::new(
        f64_to_usize("dim", dim)?,
        f64_to_u64("budget", budget)?,
        f64_to_u64("seed", seed)?,
    );
    if let Some(v) = structural_runs {
        cfg.structural_runs = f64_to_u32("structural_runs", v)?;
    }
    if let Some(v) = central_runs_per {
        cfg.central_runs_per = f64_to_u32("central_runs_per", v)?;
    }
    if let Some(f) = central_fids {
        cfg.central_fids = f64_slice_to_u32_vec("central_fids", f.as_slice())?;
    }
    if let Some(v) = central_instances {
        cfg.central_instances = f64_slice_to_u32_vec("central_instances", v.as_slice())?;
    }

    let r = bias_report(&spec, &cfg).map_err(|e| savvy_err!("{e}"))?;

    let mut out = OwnedListSexp::new(5, true)?;
    out.set_name_and_value(0, "structural", structural_result_list(&r.structural)?)?;
    out.set_name_and_value(1, "central", central_result_list(&r.central)?)?;
    // T6 is deferred in sezgi-bias itself -- always NULL, see this module's
    // own doc.
    let signature_sexp: Sexp = NullSexp.into();
    out.set_name_and_value(2, "signature", signature_sexp)?;
    out.set_name_and_value(
        3,
        "latex_summary",
        OwnedStringSexp::try_from(r.latex_summary.as_str())?,
    )?;

    let mut positions = OwnedListSexp::new(r.plot_data.final_positions.len(), false)?;
    for (i, row) in r.plot_data.final_positions.iter().enumerate() {
        positions.set_value(i, OwnedRealSexp::try_from_slice(row.as_slice())?)?;
    }
    let mut plot = OwnedListSexp::new(3, true)?;
    plot.set_name_and_value(0, "final_positions", positions)?;
    plot.set_name_and_value(
        1,
        "gap_centered",
        OwnedRealSexp::try_from_slice(r.plot_data.gap_centered.as_slice())?,
    )?;
    plot.set_name_and_value(
        2,
        "gap_shifted",
        OwnedRealSexp::try_from_slice(r.plot_data.gap_shifted.as_slice())?,
    )?;
    out.set_name_and_value(4, "plot_data", plot)?;

    Ok(out.into())
}
