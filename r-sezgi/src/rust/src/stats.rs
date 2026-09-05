use savvy::{
    savvy, savvy_err, ListSexp, NumericSexp, OwnedListSexp, OwnedRealSexp, OwnedStringSexp,
    RealSexp, Sexp, StringSexp,
};
use sezgi_bench::{
    per_budget_packages as bench_per_budget_packages, results_matrix as bench_results_matrix,
    Aggregate, RunKey, RunRecord,
};
use sezgi_stats::{
    bayesian_plackett_luce, bayesian_signed_rank, cliffs_delta, cliffs_magnitude, friedman,
    paper_package, plackett_luce, wilcoxon_signed_rank, BayesPlackettLuceResult,
    BayesSignedRankResult, FriedmanResult, PaperPackage, PlackettLuceResult, WilcoxonMethod,
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

/// Casts a non-negative-checked, WHOLE-NUMBER-checked `f64` (as passed
/// from R, which has no native unsigned integer type) to `u64`, rejecting
/// negative, non-finite, or fractional values.
///
/// M2d-3: previously silently truncated a fractional `x` toward zero; now
/// rejects it (see `session.rs`'s copy of this doc comment for the
/// py-sezgi asymmetry this closes).
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

/// Builds the named list mirroring py-sezgi's `stats_bayesian_plackett_luce`
/// dict: `mean_worths`, `ci_low`, `ci_high`, `p_best`, `samples`.
fn bayes_pl_result_list(r: &BayesPlackettLuceResult) -> savvy::Result<OwnedListSexp> {
    let mut out = OwnedListSexp::new(5, true)?;
    out.set_name_and_value(
        0,
        "mean_worths",
        OwnedRealSexp::try_from_slice(r.mean_worths.as_slice())?,
    )?;
    out.set_name_and_value(1, "ci_low", OwnedRealSexp::try_from_slice(r.ci_low.as_slice())?)?;
    out.set_name_and_value(
        2,
        "ci_high",
        OwnedRealSexp::try_from_slice(r.ci_high.as_slice())?,
    )?;
    out.set_name_and_value(3, "p_best", OwnedRealSexp::try_from_slice(r.p_best.as_slice())?)?;
    out.set_name_and_value(4, "samples", OwnedRealSexp::try_from_scalar(r.samples as f64)?)?;
    Ok(out)
}

/// Bayesian Plackett-Luce posterior via Gibbs sampling (Caron & Doucet
/// 2012 latent exponential-race augmentation; see
/// `sezgi_stats::bayesian_plackett_luce`).
///
/// This is the raw savvy-generated binding (required args only; savvy has
/// no way to express a non-`NULL` default for a required argument in the
/// generated signature). The public R entry point with R-native defaults
/// is the hand-written wrapper `sz_bayesian_plackett_luce()` in
/// `R/stats.R`, which calls this function -- same raw/wrapper pattern as
/// `sz_stats_bayesian_signed_rank()` / `sz_stats_bayesian_signed_rank_raw()`.
///
/// @param rankings A `list` of integer (or integer-valued numeric)
///   vectors, each a full ranking of the same `k` items as **1-based item
///   ids** (same convention as `sz_stats_plackett_luce()`; converted via
///   the SAME `rankings_from_list()` helper). Best (rank 1) first.
/// @param samples Number of post-burn-in Gibbs iterations to record
///   (double, cast to `u64`).
/// @param burn_in Number of initial Gibbs iterations to discard (double,
///   cast to `u64`).
/// @param seed Master RNG seed, passed through unchanged (`u64` via `f64`
///   cast) -- the same `(rankings, samples, burn_in, seed)` always
///   produces a bit-identical result, since R calls the exact same seeded
///   Rust core as Python and Rust.
/// @returns A named list with `mean_worths`, `ci_low`, `ci_high`,
///   `p_best`, `samples` (mirrors py-sezgi's
///   `stats_bayesian_plackett_luce()` dict keys exactly).
/// @noRd
#[savvy]
fn sz_bayesian_plackett_luce_raw(
    rankings: ListSexp,
    samples: f64,
    burn_in: f64,
    seed: f64,
) -> savvy::Result<Sexp> {
    let rv = rankings_from_list(&rankings)?;
    let samples_u = f64_to_u64("samples", samples)?;
    let burn_in_u = f64_to_u64("burn_in", burn_in)?;
    let seed_u = f64_to_u64("seed", seed)?;
    let r = bayesian_plackett_luce(&rv, samples_u, burn_in_u, seed_u)
        .map_err(|e| savvy_err!("{e}"))?;
    Ok(bayes_pl_result_list(&r)?.into())
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
/// @examples
/// sz_stats_wilcoxon(c(1, 2, 3, 4, 5), c(2, 1, 4, 3, 6))
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
/// @noRd
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

/// Builds the named list mirroring `sz_stats_paper_package_raw`'s return
/// shape from an already-computed [`PaperPackage`]. Shared with
/// `sz_per_budget_packages_raw` so both produce identically-shaped package
/// lists (mirrors py-sezgi's `paper_package_to_dict` split).
fn paper_package_to_list(pkg: &PaperPackage) -> savvy::Result<OwnedListSexp> {
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

    Ok(out)
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
/// @noRd
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

    Ok(paper_package_to_list(&pkg)?.into())
}

// ---------------------------------------------------------------------
// Reporting bindings (sz_results_matrix / sz_per_budget_packages)
// ---------------------------------------------------------------------

/// Parses an aggregate string (`"mean"` | `"median"`) into
/// [`sezgi_bench::Aggregate`], raising a savvy error matching "unknown
/// aggregate" on anything else.
fn parse_aggregate(aggregate: &str) -> savvy::Result<Aggregate> {
    match aggregate {
        "mean" => Ok(Aggregate::Mean),
        "median" => Ok(Aggregate::Median),
        other => Err(savvy_err!(
            "unknown aggregate `{}` (expected \"mean\" or \"median\")",
            other
        )),
    }
}

/// Rebuilds `RunRecord`s from a `data.frame`'s columns, as returned by
/// `sz_run_experiment()`: `algo, fid, dim, instance, seed, budget, suite,
/// best_f, f_opt, evals`. `wall_secs` is not a column of that data.frame, so
/// it is always defaulted to `0.0` -- it plays no role in
/// `results_matrix`/`per_budget_packages`.
///
/// `suite` (M3-5 Task 1): the R-native wrapper (`R/experiment.R`) is
/// responsible for the backward-compat default -- a `df` lacking a `suite`
/// column gets an all-`"sezgi-bbob"` vector built there before this
/// function ever sees it, so `suite` here is always a full-length column.
#[allow(clippy::too_many_arguments)]
fn records_from_columns(
    algo: &StringSexp,
    fid: &RealSexp,
    dim: &RealSexp,
    instance: &RealSexp,
    seed: &RealSexp,
    budget: &RealSexp,
    suite: &StringSexp,
    best_f: &RealSexp,
    f_opt: &RealSexp,
    evals: &RealSexp,
) -> savvy::Result<Vec<RunRecord>> {
    let n = algo.len();
    for (name, len) in [
        ("fid", fid.len()),
        ("dim", dim.len()),
        ("instance", instance.len()),
        ("seed", seed.len()),
        ("budget", budget.len()),
        ("suite", suite.len()),
        ("best_f", best_f.len()),
        ("f_opt", f_opt.len()),
        ("evals", evals.len()),
    ] {
        if len != n {
            return Err(savvy_err!(
                "column `{}` has length {} but `algo` has length {} (all columns must have the same length)",
                name, len, n
            ));
        }
    }

    let algo_s = algo.to_vec();
    let fid_s = fid.as_slice();
    let dim_s = dim.as_slice();
    let instance_s = instance.as_slice();
    let seed_s = seed.as_slice();
    let budget_s = budget.as_slice();
    let suite_s = suite.to_vec();
    let best_f_s = best_f.as_slice();
    let f_opt_s = f_opt.as_slice();
    let evals_s = evals.as_slice();

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(RunRecord {
            key: RunKey {
                algo: algo_s[i].to_string(),
                fid: f64_to_u64("fid", fid_s[i])? as u32,
                dim: f64_to_u64("dim", dim_s[i])? as usize,
                instance: f64_to_u64("instance", instance_s[i])? as u32,
                seed: f64_to_u64("seed", seed_s[i])?,
                budget: f64_to_u64("budget", budget_s[i])?,
                suite: suite_s[i].to_string(),
            },
            best_f: best_f_s[i],
            f_opt: f_opt_s[i],
            evals_used: f64_to_u64("evals", evals_s[i])?,
            wall_secs: 0.0,
        });
    }
    Ok(out)
}

/// Converts a `Vec<Vec<f64>>` (rows = problems, columns = algorithms) into
/// an R matrix -- the inverse of [`matrix_to_rows`].
fn rows_to_matrix(rows: &[Vec<f64>]) -> savvy::Result<OwnedRealSexp> {
    let nrow = rows.len();
    let ncol = rows.first().map_or(0, |r| r.len());
    let mut data = vec![0.0_f64; nrow * ncol];
    for (r, row) in rows.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            data[c * nrow + r] = v;
        }
    }
    let mut out = OwnedRealSexp::try_from_slice(data.as_slice())?;
    out.set_dim(&[nrow, ncol])?;
    Ok(out)
}

/// Builds a `sezgi_stats`-shaped results matrix for one `budget` from a
/// `sz_run_experiment()` data.frame's columns -- see
/// `sezgi_bench::reporting::results_matrix`.
///
/// This is the raw savvy-generated binding; the public R entry point with
/// R-native defaults is the hand-written wrapper `sz_results_matrix()` in
/// `R/experiment.R`, which extracts these columns from a data.frame and
/// calls this function.
///
/// @param algo Character vector (the `algo` column).
/// @param fid,dim,instance,seed,budget_col,best_f,f_opt,evals Numeric
///   vectors (the correspondingly-named columns; `budget_col` avoids a name
///   clash with the scalar `budget` argument below).
/// @param budget Only rows with this budget are used.
/// @param aggregate `"mean"` or `"median"`.
/// @param suite Character vector (the `suite` column) -- see
///   `records_from_columns`'s doc comment for the backward-compat default
///   the R-native wrapper applies when `df` has no `suite` column.
/// @returns A named list with `algo_names` (character vector),
///   `problem_labels` (character vector, `f{fid}d{dim}i{instance}` for the
///   BBOB suite, `{short}-f{fid}d{dim}i{instance}` for any other suite),
///   and `matrix` (numeric matrix, rows = problems, columns = algorithms).
/// @noRd
#[allow(clippy::too_many_arguments)]
#[savvy]
fn sz_results_matrix_raw(
    algo: StringSexp,
    fid: RealSexp,
    dim: RealSexp,
    instance: RealSexp,
    seed: RealSexp,
    budget_col: RealSexp,
    suite: StringSexp,
    best_f: RealSexp,
    f_opt: RealSexp,
    evals: RealSexp,
    budget: f64,
    aggregate: &str,
) -> savvy::Result<Sexp> {
    let records = records_from_columns(
        &algo, &fid, &dim, &instance, &seed, &budget_col, &suite, &best_f, &f_opt, &evals,
    )?;
    let agg = parse_aggregate(aggregate)?;
    let budget_u = f64_to_u64("budget", budget)?;

    let (algo_names, problem_labels, matrix) = bench_results_matrix(&records, budget_u, agg)
        .map_err(|e| savvy_err!("{e}"))?;

    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "algo_names", OwnedStringSexp::try_from(algo_names.as_slice())?)?;
    out.set_name_and_value(
        1,
        "problem_labels",
        OwnedStringSexp::try_from(problem_labels.as_slice())?,
    )?;
    out.set_name_and_value(2, "matrix", rows_to_matrix(&matrix)?)?;
    Ok(out.into())
}

/// Builds one `sezgi_stats::PaperPackage` PER DISTINCT BUDGET present in a
/// `sz_run_experiment()` data.frame's columns, in ascending budget order --
/// see `sezgi_bench::reporting::per_budget_packages`. Per Piotrowski et al.
/// (2025), algorithm rankings can flip across budgets, so this makes
/// multi-budget reporting the default rather than a single, arbitrarily
/// chosen budget's report.
///
/// This is the raw savvy-generated binding; the public R entry point with
/// R-native defaults is the hand-written wrapper `sz_per_budget_packages()`
/// in `R/experiment.R`, which extracts these columns from a data.frame and
/// calls this function.
///
/// @param algo Character vector (the `algo` column).
/// @param fid,dim,instance,seed,budget_col,best_f,f_opt,evals Numeric
///   vectors (the correspondingly-named columns).
/// @param suite Character vector (the `suite` column) -- see
///   `records_from_columns`'s doc comment for the backward-compat default
///   the R-native wrapper applies when `df` has no `suite` column.
/// @param rope Region of practical equivalence half-width (>= 0) for the
///   Bayesian signed-rank test, forwarded to every budget's package.
/// @param samples Number of Monte Carlo samples per pair (double, cast to
///   `u64`).
/// @param seed Master RNG seed (double, cast to `u64`), forwarded to every
///   budget's package.
/// @param aggregate `"mean"` or `"median"`.
/// @returns A named list, one entry per distinct budget in ascending order,
///   named by the budget (as a string); each value has exactly the shape
///   `sz_stats_paper_package_raw()` returns.
/// @noRd
#[allow(clippy::too_many_arguments)]
#[savvy]
fn sz_per_budget_packages_raw(
    algo: StringSexp,
    fid: RealSexp,
    dim: RealSexp,
    instance: RealSexp,
    seed: RealSexp,
    budget_col: RealSexp,
    suite: StringSexp,
    best_f: RealSexp,
    f_opt: RealSexp,
    evals: RealSexp,
    rope: f64,
    samples: f64,
    master_seed: f64,
    aggregate: &str,
) -> savvy::Result<Sexp> {
    let records = records_from_columns(
        &algo, &fid, &dim, &instance, &seed, &budget_col, &suite, &best_f, &f_opt, &evals,
    )?;
    let agg = parse_aggregate(aggregate)?;
    let samples_u = f64_to_u64("samples", samples)?;
    let seed_u = f64_to_u64("seed", master_seed)?;

    let packages = bench_per_budget_packages(&records, rope, samples_u, seed_u, agg)
        .map_err(|e| savvy_err!("{e}"))?;

    let mut out = OwnedListSexp::new(packages.len(), true)?;
    for (idx, (budget, pkg)) in packages.iter().enumerate() {
        out.set_name_and_value(idx, budget.to_string().as_str(), paper_package_to_list(pkg)?)?;
    }
    Ok(out.into())
}
