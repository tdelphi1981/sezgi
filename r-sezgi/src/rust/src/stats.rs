use savvy::{
    savvy, savvy_err, ListSexp, NumericSexp, OwnedListSexp, OwnedRealSexp, OwnedStringSexp,
    RealSexp, Sexp, StringSexp,
};
use sezgi_stats::{
    bayesian_signed_rank, cliffs_delta, cliffs_magnitude, friedman, paper_package, plackett_luce,
    wilcoxon_signed_rank, BayesSignedRankResult, FriedmanResult, PlackettLuceResult,
    WilcoxonMethod,
};

/// Converts an R matrix (`RealSexp` with a `dim` attribute, R's column-major
/// storage) into `Vec<Vec<f64>>` with **rows = problems, columns =
/// algorithms**, matching the `sezgi_stats` results-matrix convention (see
/// `crates/stats/src/lib.rs`).
///
/// R stores a matrix column-major: `data[col * nrow + row]`. Getting this
/// transpose backwards is the classic bug here (see
/// `matrix_marshalling_catches_transposition` in `tests/testthat/test-stats.R`
/// for a regression fixture that would fail if rows/cols were swapped).
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

/// Converts an R `list` of integer (or numeric-but-integer-ish) vectors into
/// `Vec<Vec<usize>>` for [`plackett_luce`].
///
/// **1-based -> 0-based convention**: each R ranking vector is a permutation
/// of the 1-based item ids `1..=k` (R's natural indexing, matching
/// `algo_names` positions), best (rank 1) first. `sezgi_stats::plackett_luce`
/// expects 0-based item indices (a permutation of `0..k`), so every value is
/// decremented by 1 here. This choice is documented on the R-facing
/// `sz_stats_plackett_luce()` roxygen doc.
fn rankings_from_list(rankings: &ListSexp) -> savvy::Result<Vec<Vec<usize>>> {
    let mut out = Vec::with_capacity(rankings.len());
    for i in 0..rankings.len() {
        let elt = rankings
            .get_by_index(i)
            .ok_or_else(|| savvy_err!("rankings[[{}]] is missing", i + 1))?;
        let num: NumericSexp = elt.try_into()?;
        let ints = num.as_slice_i32()?;
        let mut row = Vec::with_capacity(ints.len());
        for &v in ints {
            if v < 1 {
                return Err(savvy_err!(
                    "rankings[[{}]]: item ids must be 1-based (>= 1), got {}",
                    i + 1,
                    v
                ));
            }
            row.push((v - 1) as usize);
        }
        out.push(row);
    }
    Ok(out)
}

/// Casts a non-negative-checked `f64` (as passed from R, which has no native
/// unsigned integer type) to `u64`, rejecting negative or non-finite values.
fn f64_to_u64(name: &str, x: f64) -> savvy::Result<u64> {
    if !x.is_finite() || x < 0.0 {
        return Err(savvy_err!(
            "{} must be a non-negative finite number, got {}",
            name,
            x
        ));
    }
    Ok(x as u64)
}

/// Builds the named list mirroring py-sezgi's `stats_friedman` dict:
/// `statistic`, `p_value`, `mean_ranks`.
fn friedman_result_list(r: &FriedmanResult) -> savvy::Result<OwnedListSexp> {
    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "statistic", OwnedRealSexp::try_from_scalar(r.statistic)?)?;
    out.set_name_and_value(1, "p_value", OwnedRealSexp::try_from_scalar(r.p_value)?)?;
    out.set_name_and_value(
        2,
        "mean_ranks",
        OwnedRealSexp::try_from_slice(r.mean_ranks.as_slice())?,
    )?;
    Ok(out)
}

/// Builds the named list mirroring py-sezgi's `stats_bayesian_signed_rank`
/// (and the nested `bayes` entries of `stats_paper_package`) dict:
/// `p_left`, `p_rope`, `p_right`.
fn bayes_result_list(r: &BayesSignedRankResult) -> savvy::Result<OwnedListSexp> {
    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "p_left", OwnedRealSexp::try_from_scalar(r.p_left)?)?;
    out.set_name_and_value(1, "p_rope", OwnedRealSexp::try_from_scalar(r.p_rope)?)?;
    out.set_name_and_value(2, "p_right", OwnedRealSexp::try_from_scalar(r.p_right)?)?;
    Ok(out)
}

/// Builds the named list mirroring py-sezgi's `stats_plackett_luce` (and the
/// nested `plackett_luce` entry of `stats_paper_package`) dict: `worths`,
/// `p_best`, `iterations`.
fn pl_result_list(r: &PlackettLuceResult) -> savvy::Result<OwnedListSexp> {
    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "worths", OwnedRealSexp::try_from_slice(r.worths.as_slice())?)?;
    out.set_name_and_value(1, "p_best", OwnedRealSexp::try_from_slice(r.p_best.as_slice())?)?;
    out.set_name_and_value(
        2,
        "iterations",
        OwnedRealSexp::try_from_scalar(r.iterations as f64)?,
    )?;
    Ok(out)
}

/// Runs the Friedman test on a results matrix.
///
/// @param m A numeric matrix, rows = problems, columns = algorithms, lower
///   is better (e.g. built with `rbind()` or `matrix()`).
/// @returns A named list with `statistic`, `p_value`, `mean_ranks` (mirrors
///   py-sezgi's `stats_friedman()` dict keys exactly).
/// @export
#[savvy]
fn sz_stats_friedman(m: RealSexp) -> savvy::Result<Sexp> {
    let matrix = matrix_to_rows(&m)?;
    let r = friedman(&matrix).map_err(|e| savvy_err!("{e}"))?;
    Ok(friedman_result_list(&r)?.into())
}

/// Wilcoxon signed-rank test for two paired samples (Pratt zero-handling and
/// tie correction; see `sezgi_stats::wilcoxon_signed_rank`).
///
/// @param a Numeric vector.
/// @param b Numeric vector, same length as `a`.
/// @returns A named list with `w_statistic`, `z`, `p_value`, `n_effective`,
///   `method` (`"exact"` or `"normal_approx"`; mirrors py-sezgi's
///   `stats_wilcoxon()` dict keys exactly). `"exact"` is used when
///   `n_effective <= 25` and there are no zero differences or tied `|d|`
///   ranks; see `sezgi_stats::wilcoxon_signed_rank`'s doc comment for the
///   full eligibility rule (the exact p-value formula is semver-pinned).
/// @export
#[savvy]
fn sz_stats_wilcoxon(a: RealSexp, b: RealSexp) -> savvy::Result<Sexp> {
    let av = a.as_slice();
    let bv = b.as_slice();
    let r = wilcoxon_signed_rank(av, bv).map_err(|e| savvy_err!("{e}"))?;
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
    Ok(out.into())
}

/// Cliff's delta effect size for two independent (unpaired) samples.
///
/// @param a Numeric vector.
/// @param b Numeric vector.
/// @returns A numeric scalar in `[-1, 1]`.
/// @export
#[savvy]
fn sz_stats_cliffs_delta(a: RealSexp, b: RealSexp) -> savvy::Result<Sexp> {
    let delta = cliffs_delta(a.as_slice(), b.as_slice()).map_err(|e| savvy_err!("{e}"))?;
    delta.try_into()
}

/// Qualitative magnitude label for a Cliff's delta value (Romano et al. 2006
/// thresholds).
///
/// @param delta A Cliff's delta value (as returned by
///   `sz_stats_cliffs_delta()`).
/// @returns A character scalar: one of `"negligible"`, `"small"`,
///   `"medium"`, `"large"`.
/// @export
#[savvy]
fn sz_stats_cliffs_magnitude(delta: f64) -> savvy::Result<Sexp> {
    cliffs_magnitude(delta).try_into()
}

/// Bayesian signed-rank test with a region of practical equivalence (ROPE)
/// (Dirichlet-weighted Monte Carlo; see `sezgi_stats::bayesian_signed_rank`).
///
/// This is the raw savvy-generated binding (required args only; savvy has no
/// way to express a non-`NULL` default for a required argument in the
/// generated signature). The public R entry point with R-native defaults is
/// the hand-written wrapper `sz_stats_bayesian_signed_rank()` in
/// `R/stats.R`, which calls this function.
///
/// @param a Numeric vector.
/// @param b Numeric vector, same length as `a`.
/// @param rope Region of practical equivalence half-width (>= 0).
/// @param samples Number of Monte Carlo samples. Passed to R as a double
///   (savvy has no unsigned integer scalar arg type) and cast to `u64`
///   after checking it is non-negative and finite.
/// @param seed Master RNG seed, passed through unchanged (`u64` via `f64`
///   cast) — the same `(a, b, rope, samples, seed)` always produces a
///   bit-identical result, since R calls the exact same seeded Rust core as
///   Python and Rust.
/// @returns A named list with `p_left`, `p_rope`, `p_right` (mirrors
///   py-sezgi's `stats_bayesian_signed_rank()` dict keys exactly).
#[savvy]
fn sz_stats_bayesian_signed_rank_raw(
    a: RealSexp,
    b: RealSexp,
    rope: f64,
    samples: f64,
    seed: f64,
) -> savvy::Result<Sexp> {
    let av = a.as_slice();
    let bv = b.as_slice();
    let samples_u = f64_to_u64("samples", samples)?;
    let seed_u = f64_to_u64("seed", seed)?;
    let r = bayesian_signed_rank(av, bv, rope, samples_u, seed_u).map_err(|e| savvy_err!("{e}"))?;
    Ok(bayes_result_list(&r)?.into())
}

/// Plackett-Luce maximum-likelihood ranking (Hunter 2004 MM algorithm; see
/// `sezgi_stats::plackett_luce`).
///
/// @param rankings A `list` of integer (or integer-valued numeric) vectors,
///   each a full ranking of the same `k` items as **1-based item ids**
///   (R's natural indexing; e.g. `list(c(1, 2, 3), c(2, 1, 3))` for k = 3
///   items, best first). Converted to 0-based indices before calling the
///   Rust core, which expects a permutation of `0..k`.
/// @returns A named list with `worths`, `p_best`, `iterations` (mirrors
///   py-sezgi's `stats_plackett_luce()` dict keys exactly).
/// @export
#[savvy]
fn sz_stats_plackett_luce(rankings: ListSexp) -> savvy::Result<Sexp> {
    let rv = rankings_from_list(&rankings)?;
    let r = plackett_luce(&rv).map_err(|e| savvy_err!("{e}"))?;
    Ok(pl_result_list(&r)?.into())
}

/// Computes a comprehensive statistical analysis package for algorithm
/// comparison (Friedman, Nemenyi CD, pairwise Wilcoxon+Holm, Cliff's delta,
/// Bayesian signed-rank, Plackett-Luce, and LaTeX tables; see
/// `sezgi_stats::paper_package`).
///
/// This is the raw savvy-generated binding (required args only; savvy has no
/// way to express a non-`NULL` default for a required argument in the
/// generated signature). The public R entry point with R-native defaults is
/// the hand-written wrapper `sz_stats_paper_package()` in `R/stats.R`, which
/// calls this function.
///
/// @param algo_names Character vector of algorithm names.
/// @param problem_names Character vector of problem names.
/// @param m A numeric matrix, rows = problems (`length(problem_names)`),
///   columns = algorithms (`length(algo_names)`), lower is better.
/// @param rope Region of practical equivalence half-width (>= 0) for the
///   Bayesian signed-rank test.
/// @param samples Number of Monte Carlo samples per pair for the Bayesian
///   signed-rank test (double, cast to `u64`; see
///   `sz_stats_bayesian_signed_rank_raw()`).
/// @param seed Master RNG seed (double, cast to `u64`); per-pair seeds are
///   `seed + pair_index` (wrapping), matching `sezgi_stats::paper_package`.
/// @returns A named list with `friedman` (nested list: `statistic`,
///   `p_value`, `mean_ranks`), `nemenyi_cd`, `pairwise_wilcoxon_holm` (list
///   of length-3 numeric vectors `c(i, j, adjusted_p)`, 0-based algorithm
///   indices, i < j), `cliffs` (list of length-3 numeric vectors `c(i, j,
///   delta)`), `bayes` (list of 3-element lists `list(i, j, bayes_result)`
///   where `bayes_result` has `p_left`/`p_rope`/`p_right`),
///   `plackett_luce` (nested list: `worths`, `p_best`, `iterations`),
///   `latex_summary`, `latex_tests` (mirrors py-sezgi's
///   `stats_paper_package()` dict keys exactly).
#[savvy]
fn sz_stats_paper_package_raw(
    algo_names: StringSexp,
    problem_names: StringSexp,
    m: RealSexp,
    rope: f64,
    samples: f64,
    seed: f64,
) -> savvy::Result<Sexp> {
    let algos: Vec<String> = algo_names.iter().map(|s| s.to_string()).collect();
    let problems: Vec<String> = problem_names.iter().map(|s| s.to_string()).collect();
    let matrix = matrix_to_rows(&m)?;
    let samples_u = f64_to_u64("samples", samples)?;
    let seed_u = f64_to_u64("seed", seed)?;

    let pkg = paper_package(&algos, &problems, &matrix, rope, samples_u, seed_u)
        .map_err(|e| savvy_err!("{e}"))?;

    let mut out = OwnedListSexp::new(8, true)?;
    out.set_name_and_value(0, "friedman", friedman_result_list(&pkg.friedman)?)?;
    out.set_name_and_value(1, "nemenyi_cd", OwnedRealSexp::try_from_scalar(pkg.nemenyi_cd)?)?;

    let mut pw = OwnedListSexp::new(pkg.pairwise_wilcoxon_holm.len(), false)?;
    for (idx, &(i, j, p)) in pkg.pairwise_wilcoxon_holm.iter().enumerate() {
        pw.set_value(idx, OwnedRealSexp::try_from_slice([i as f64, j as f64, p])?)?;
    }
    out.set_name_and_value(2, "pairwise_wilcoxon_holm", pw)?;

    let mut cl = OwnedListSexp::new(pkg.cliffs.len(), false)?;
    for (idx, &(i, j, delta)) in pkg.cliffs.iter().enumerate() {
        cl.set_value(idx, OwnedRealSexp::try_from_slice([i as f64, j as f64, delta])?)?;
    }
    out.set_name_and_value(3, "cliffs", cl)?;

    let mut bl = OwnedListSexp::new(pkg.bayes.len(), false)?;
    for (idx, (i, j, res)) in pkg.bayes.iter().enumerate() {
        let mut entry = OwnedListSexp::new(3, false)?;
        entry.set_value(0, OwnedRealSexp::try_from_scalar(*i as f64)?)?;
        entry.set_value(1, OwnedRealSexp::try_from_scalar(*j as f64)?)?;
        entry.set_value(2, bayes_result_list(res)?)?;
        bl.set_value(idx, entry)?;
    }
    out.set_name_and_value(4, "bayes", bl)?;

    out.set_name_and_value(5, "plackett_luce", pl_result_list(&pkg.plackett_luce)?)?;
    out.set_name_and_value(
        6,
        "latex_summary",
        OwnedStringSexp::try_from(pkg.latex_summary.as_str())?,
    )?;
    out.set_name_and_value(
        7,
        "latex_tests",
        OwnedStringSexp::try_from(pkg.latex_tests.as_str())?,
    )?;

    Ok(out.into())
}
