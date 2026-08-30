//! Structural-bias scan: run an algorithm repeatedly on [`crate::F0Random`]
//! and test whether its final positions depart from uniformity, per the
//! BIAS-toolbox method.
//!
//! # Provenance
//!
//! ## The founding paper
//!
//! Kononova, A. V., Corne, D. W., De Wilde, P., Shneer, V., & Caraffini, F.
//! (2015). "Structural bias in population-based algorithms." *Information
//! Sciences*, 297, 468-490 (preprint: arXiv:1408.5350, verified PDF, 36
//! pages, fetched 2026-08-30). This paper defines `f0` (their eq. (1), p. 7,
//! quoted verbatim):
//!
//! ```text
//! f0 : D subset R^n -> [0,1], where x in Uniform(D), f0(x) in Uniform[0,1],
//! x and f0(x) are i.i.d.
//! ```
//!
//! immediately followed by: "Again, without loss of generality, we can
//! consider `D = [0,1]^n`." **This CONFIRMS `crate::f0`'s existing domain
//! (`F0_LO = 0.0`, `F0_HI = 1.0`) exactly — no change to T2's domain
//! constants is needed by this task.**
//!
//! Section 3.1 (p. 9) runs "50 independent runs" per parameter setting; the
//! quantities visualised and tested in Section 5 (p. 21, Fig. 3/5/8 captions)
//! are "positions of points with the best fitness values" in the final
//! population of each run — i.e. the run's own **final best position**, not
//! the final population as a whole. Section 5 (p. 23) proposes the
//! Kolmogorov-Smirnov test, applied independently per dimension, comparing
//! the empirical distribution of that dimension's coordinate across runs
//! against the U(0,1) CDF, to give an objective per-dimension p-value; no
//! multiple-testing correction or quantitative rejection-count decision rule
//! is specified in this paper (evidence is presented as raw per-dimension
//! p-values, interpreted qualitatively).
//!
//! ## The BIAS toolbox
//!
//! Vermetten, D., van Stein, B., Caraffini, F., Minku, L. L., & Kononova, A.
//! V. (2022). "BIAS: A Toolbox for Benchmarking Structural Bias in the
//! Continuous Domain." *IEEE Transactions on Evolutionary Computation*,
//! 26(6), 1380-1393. (Note: the brief's author list included "Doerr"; the
//! verified author list is Vermetten, van Stein, Caraffini, Minku, Kononova
//! -- Doerr is not an author of this paper.) Its reference implementation is
//! `github.com/nikivanstein/BIAS` (verified via `gh api`, 2026-08-30: files
//! `README.md`, `example.py`, `BIAS/SB_Toolbox.py`, `BIAS/SB_Test_runner.py`
//! read directly from the `master` branch):
//!
//! - **f0 implementation** (`BIAS/SB_Toolbox.py`): `def f0(x): return
//!   np.random.uniform()`, called in `example.py` with `bounds = [(0,1),
//!   ...]` -- confirms the `[0,1]^d` domain independently of the 2015 paper.
//! - **Number of runs**: `BIAS.__init__`'s docstring: "Use f0 as objective
//!   function for at least 30 independent optimization runs." `predict()`
//!   and `predict_deep()` both hard-require `n_samples in [30, 50, 100,
//!   600]` (their precomputed critical-value tables only cover these four
//!   sizes). **30 is therefore the toolbox's own verified minimum/default
//!   run count** -- used as [`DEFAULT_RUNS`] below.
//! - **Collected data**: `example.py`: `samples.append(result.x)` --
//!   `result.x` is the optimizer's own returned best point, confirming
//!   "final best position per run" (matching the 2015 paper's convention,
//!   not the final population) is what `BIAS.predict()` consumes; this
//!   matches [`StructuralBiasResult::final_positions`] here.
//! - **Statistical battery**: `BIAS/SB_Test_runner.py`'s `get_test_dict()`
//!   runs ~36 per-dimension goodness-of-fit tests against U(0,1) (named in
//!   `SB_Toolbox.py::_get_test_names_dict`: 1/2/3-spacing, range, min, max,
//!   AD, tAD, Shapiro, JB, LD-min/max, Kurt, MPD-max/min, Wasserstein, NS,
//!   KS, CvM, Durbin, Kuiper, HG1/2, Greenwood, QM, RC, Moran, Cressie1/2,
//!   Vasicek, Swartz, Morales, Pardo, Marhuenda, Zhang1/2). Both `KS` and
//!   `AD` are named members of this battery, and the R helper
//!   `SB_Test_runner.py::R_test_ad` computes `ad.test(x, "punif", max=1,
//!   min=0)` -- the same case-0 (fully specified U(0,1) reference)
//!   Anderson-Darling test `sezgi_stats::uniformity::ad_uniform` implements.
//!
//!   // sezgi simplification: this module ships only the KS+AD pair (already
//!   // built in T3) as a documented SUBSET of the toolbox's ~36-test
//!   // battery; the other ~34 tests (spacings, Shapiro/JB normality-of-
//!   // transform tests, CvM, Durbin, Kuiper, the PoweR-derived tests, etc.)
//!   // are not implemented here.
//!
//! - **Multiple-testing correction**: `BIAS.predict(..., corr_method=
//!   "fdr_bh", alpha=0.01, ...)` -- the toolbox's DEFAULT correction is
//!   Benjamini-Hochberg FDR (`statsmodels.stats.multitest.multipletests`),
//!   applied PER DIMENSION across the battery of ~36 tests (i.e. its
//!   correction family is "the tests", for one dimension at a time).
//!   `'holm'` is explicitly named as a supported alternative in the same
//!   docstring.
//!
//!   // sezgi simplification: this module uses Holm (`sezgi_stats::holm`,
//!   // per this task's brief and the toolbox's own supported-alternative
//!   // list) rather than the toolbox's BH-FDR default. Its correction FAMILY
//!   // is also reoriented: since sezgi's battery has only two members
//!   // (KS, AD) rather than ~36, correcting "across the battery, per
//!   // dimension" would be nearly powerless (only 2 comparisons). Instead,
//!   // each statistic's `dim` per-dimension p-values are Holm-corrected
//!   // ACROSS DIMENSIONS (family size = `dim`), independently for KS and
//!   // for AD -- this is the multiple-comparisons problem `dim > 1` actually
//!   // poses for a 2-test battery, and is what `holm_rejections_ks`/
//!   // `holm_rejections_ad` count.
//!
//! - **Decision rule**: `BIAS.predict_type`: a test in the battery counts as
//!   "activated" only if it rejects, after per-dimension correction, in MORE
//!   than 10% of dimensions; overall structural bias is claimed only if at
//!   least one test in the battery is activated by that threshold ("No clear
//!   evidence of bias detected" otherwise). This rule is shaped for the
//!   toolbox's typical `dim = 30` scenario (guarding a single-dimension
//!   fluke out of many).
//!
//!   // sezgi simplification: the closest faithful analogue for a 2-member
//!   // battery under sezgi's own dimension-corrected axis (see above) is:
//!   // structural bias is reported ([`crate::BiasVerdict::Evidence`]) if
//!   // ANY Holm-corrected per-dimension p-value, in EITHER family (KS or
//!   // AD), falls below `ALPHA` -- i.e. `holm_rejections_ks > 0 ||
//!   // holm_rejections_ad > 0`. `ALPHA = 0.01` is the toolbox's own default
//!   // significance level (also the threshold `sezgi_stats::uniformity`'s
//!   // own provenance notes were written against). This is a STRICTER
//!   // trigger than the toolbox's own >10%-of-dimensions threshold (a
//!   // single rejected dimension is sufficient here), a deliberate
//!   // trade-off given the much smaller battery: with only two tests, this
//!   // module cannot itself distinguish a genuine, broadly-shared departure
//!   // from a single-dimension fluke the way the toolbox's 36-test,
//!   // multi-dimension aggregation can -- `per_dim_ks`/`per_dim_ad` are
//!   // still returned in full so a caller (or a human) can make that
//!   // judgment from the raw per-dimension evidence. Note the combined
//!   // false-Evidence bound: each family's Holm step controls FWER at
//!   // `ALPHA` WITHIN that family, so OR-ing the two families bounds the
//!   // overall Type-I probability at ~`2 * ALPHA` (~0.02), not `ALPHA`.
//!
//! # f0-domain verdict
//!
//! Both sources confirm `[0,1]^d`; `crate::f0::F0_LO`/`F0_HI` are UNCHANGED
//! by this task.
//!
//! # Seed derivation
//!
//! [`structural_bias_scan`] passes `cfg.seed` as BOTH `F0Random::new`'s own
//! `seed` argument AND `RunConfig::master_seed`, run `i` (`0..cfg.runs`)
//! using `run_id = i`. This is precisely the scenario `crate::f0`'s own
//! module doc ("RNG stream path") analyses and declares safe: `F0Random`
//! mixes `BIAS_SEED_BASE` into its own master before deriving its stream, so
//! reusing the same numeric `cfg.seed` for both purposes does NOT collide
//! the two RNG namespaces. Each run gets its own independent engine stream
//! family via its own `run_id` (`crates/bench/src/experiment.rs` varies
//! both the seed and `run_id` per run; this module varies `run_id` only,
//! which the engine's stream derivation already keeps collision-free) --
//! never reusing `run_id = 0` for every run, which would collapse every
//! run's engine-side randomness to an identical sequence (the project-wide
//! ban on single-seed comparisons, adapted here to "single algorithm,
//! multiple runs").
//!
//! # Domain handling for out-of-domain / boundary final positions (fix wave)
//!
//! A whole-branch review (2026-08-30) measured that [`structural_bias_scan`]
//! errored on 21 of this project's own 25 presets (probe: 30 runs, dim=3,
//! budget=3000): boundary-clamping presets (pso, de, gwo, cmaes, and 17
//! others) routinely park a best coordinate at exactly `0.0`/`1.0`, which
//! `ad_uniform`'s domain contract hard-rejects (`ln(0)`/`ln(1)` undefined);
//! HHO's Evaluator-observed `best_x` can be a raw, un-clamped dive-trial
//! point OUTSIDE `[0,1]` entirely (measured: `-0.175`), which `ks_uniform`
//! rejects as out-of-domain. Both failure modes abort the WHOLE scan via `?`
//! -- i.e. the scan errored on exactly the boundary-clustering signature it
//! exists to detect.
//!
//! **Provenance check (per this project's discipline, before choosing a
//! design):** the BIAS toolbox reference implementation
//! (`github.com/nikivanstein/BIAS`, `master` branch, re-fetched 2026-08-30)
//! was checked for its own boundary/out-of-bounds handling. Findings:
//! - `BIAS/SB_Toolbox.py`'s `f0` returns `np.random.uniform()` un-clamped,
//!   and its `BIAS.predict()` passes each dimension's raw column straight
//!   into every battery test with NO preprocessing, clamping, or transform
//!   of any kind -- the docstring merely states data "should be scaled in
//!   `[0,1]`," enforced only by caller convention, never by code.
//! - `BIAS.predict()`'s per-test call IS defensively wrapped, but generically
//!   so, not with any boundary-specific rule (`SB_Toolbox.py`, `predict()`):
//!   `try: temp.append(tfunc(data[:, r], alpha=alpha)) except: next` -- a
//!   bare `except` that silently DROPS (skips) that one dimension's result
//!   for that one test on ANY exception, never aborting the whole battery,
//!   but also never treating the failure itself as evidence -- the dropped
//!   dimension simply does not vote in `predict_type`'s rejection count.
//! - The toolbox's own worked example (`example.py`) drives `scipy.optimize.
//!   differential_evolution` with `bounds=[(0,1), ...]`; scipy's DE always
//!   returns a bounds-respecting `result.x`, so the toolbox's own example
//!   never exercises an out-of-domain final position at all -- it has no
//!   principled rule for that case because it never needed one.
//!
//! **Verdict: the toolbox is silent on the specific question this fix
//! answers.** It has no domain-aware boundary/OOD preprocessing to follow
//! (its only defense is a blanket, reason-agnostic `except: skip`, which
//! would DISCARD the boundary-clustering evidence sezgi's scan exists to
//! surface -- the opposite of this project's purpose). Per the controller's
//! ruling for exactly this "toolbox silent" case, this module implements the
//! following instead, entirely at the SCAN level -- `sezgi_stats::
//! uniformity`'s pinned `ks_uniform`/`ad_uniform` general-purpose input
//! contracts are UNCHANGED:
//!
//! 1. **Out-of-domain positions are clamped into `[0,1]` before testing**
//!    (see [`scan_from_positions`]). An out-of-domain `best_x` means the
//!    algorithm's best CHARGED evaluation was a raw, pre-boundary-repair
//!    trial point (see `crates/components/src/hho.rs`'s module doc,
//!    "`best_f`/`best_x` and IOH visibility"); clamping it into the domain
//!    counts it as boundary mass for the uniformity test -- consistent with
//!    the scan's purpose, and this module's own doc already claims (see
//!    "The BIAS toolbox" section above) that the `[0,1]` domain mapping
//!    "lives in the bias crate," which this is what makes that claim true.
//! 2. **A per-dimension sample containing an exact `0.0`/`1.0` (after the
//!    clamp above) is NOT sent through `ad_uniform`** (which would error).
//!    Instead this module computes the LIMIT its own pinned formula takes at
//!    that boundary directly: `ad_uniform`'s statistic is `A_n = -n - (1/n)
//!    * sum_i (2i-1)[ln(x_i) + ln(1-x_{n+1-i})]` (see `sezgi_stats::
//!      uniformity`'s module doc, quoted from Marsaglia & Marsaglia 2004).
//!    Every summand is non-positive on `[0,1]` (`ln(x) <= 0`, `ln(1-x) <=
//!      0`), and a boundary value drives exactly one of its two `ln` terms to
//!    `-infinity` (never `+infinity` -- there is no way to reach `NaN` from
//!    `-infinity + -infinity` or `-infinity` plus any finite non-positive
//!    term). So the sum is deterministically `-infinity`, and `A_n = -n -
//!      (-infinity)/n = +infinity`: the mathematically exact limit, not an
//!    approximation, and NaN-free by construction. This module represents
//!    that dimension's AD result directly as `AdResult { a2: f64::INFINITY,
//!      p_value: 0.0, n }` -- the limiting MAXIMAL-evidence outcome (`adinf`'s
//!    own two-piece Horner polynomial is a finite-range fit and is not
//!    evaluated at `z = infinity`; `p_value = 0.0` is what `1 - adinf(z)`
//!    converges to as `z -> infinity`, since `adinf` is a CDF approximation
//!    and every genuine CDF tends to 1). `ks_uniform` needs no analogous
//!    special case: `BOUNDARY_TOL` already accepts exact `0.0`/`1.0` (see
//!      that function's own validation), so it runs unmodified on the same
//!      clamped column.
//!
//! Both rules flow deterministically (no RNG, no branching on float
//! comparison beyond exact `0.0`/`1.0` equality) through Holm correction
//! (`p_value = 0.0` is an ordinary finite input to `holm`), the LaTeX report
//! (`report.rs`'s `fmt_stat` already renders any non-finite value, `a2 =
//! f64::INFINITY` included, as the plain text `n/a` -- never the literal
//! `inf`/`NaN`; `fmt_p` never sees a non-finite `p_value` from this path,
//! since `0.0` is finite), the py bindings (`float('inf')` is a normal
//! Python float), and the R bindings (`Inf` is a normal R double) -- no
//! change was needed in `report.rs` or either binding for this specific
//! representation, only this module's own construction of the boundary
//! `AdResult`.

use sezgi_core::component::Registry;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_core::problem::Problem;
use sezgi_core::space::BlockValues;
use sezgi_core::spec::AlgorithmSpec;
use sezgi_stats::uniformity::{ad_uniform, ks_uniform, AdResult, KsResult};
use sezgi_stats::ranks::holm;

use crate::f0::F0Random;
use crate::{BiasError, BiasVerdict};

/// Verified default run count -- see this module's doc, "Number of runs".
pub const DEFAULT_RUNS: u32 = 30;

/// Significance threshold for the Holm-corrected per-dimension rejection
/// count -- the BIAS toolbox's own default `alpha` (`BIAS.predict`'s default
/// argument); see this module's doc, "Decision rule".
const ALPHA: f64 = 0.01;

/// Minimum number of runs accepted by [`structural_bias_scan`] and
/// [`scan_from_positions`]: mirrors `sezgi_stats::uniformity`'s own `MIN_N`
/// (5) -- `ks_uniform`/`ad_uniform` cannot be computed below it.
const MIN_RUNS: usize = 5;

#[derive(Debug, Clone, PartialEq)]
pub struct StructuralBiasConfig {
    /// Number of independent runs. Per this module's doc ("Number of
    /// runs"): the BIAS toolbox's own verified default/minimum is
    /// [`DEFAULT_RUNS`] (30).
    pub runs: u32,
    /// Dimensionality of f0's domain.
    pub dim: usize,
    /// Per-run evaluation budget. Overrides `spec.termination.budget`
    /// (mirroring `crates/bench/src/experiment.rs`'s own "budget is ALWAYS
    /// overridden" contract), so a caller need not bake the budget into the
    /// `AlgorithmSpec` it passes to [`structural_bias_scan`].
    pub budget: u64,
    /// Seed, mixed into both f0's own RNG master and the engine's
    /// `master_seed` -- see this module's doc, "Seed derivation".
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructuralBiasResult {
    /// Per-dimension KS test of `final_positions`' coordinates against
    /// U(0,1); `per_dim_ks[d]` is dimension `d`'s result. Length `dim`.
    pub per_dim_ks: Vec<KsResult>,
    /// Per-dimension AD test, same shape/order as `per_dim_ks`.
    pub per_dim_ad: Vec<AdResult>,
    /// Number of dimensions whose Holm-corrected (across dimensions) KS
    /// p-value falls below [`ALPHA`].
    pub holm_rejections_ks: usize,
    /// Number of dimensions whose Holm-corrected (across dimensions) AD
    /// p-value falls below [`ALPHA`].
    pub holm_rejections_ad: usize,
    /// Decision -- see this module's doc, "Decision rule".
    pub verdict: BiasVerdict,
    /// The raw evidence: `runs` x `dim`, row `i` is run `i`'s final best
    /// position on f0.
    pub final_positions: Vec<Vec<f64>>,
}

/// Runs `spec` on [`F0Random`] `cfg.runs` times (via `sezgi_core::engine`,
/// reusing its own RNG-derivation and budget-override conventions -- see
/// this module's doc) and tests the resulting final positions for
/// structural bias. See the module doc for the full method provenance.
///
/// # Errors
/// [`BiasError::InvalidConfig`] if `cfg.dim == 0` or `cfg.runs` is below
/// [`MIN_RUNS`]; [`BiasError::Spec`] if `spec` fails to validate against
/// f0's space; [`BiasError::Engine`] if a run fails (e.g. budget smaller
/// than population size); [`BiasError::Stats`] if `ks_uniform`/`ad_uniform`
/// reject a resulting column for a reason OTHER than the boundary/
/// out-of-domain cases [`scan_from_positions`] now handles itself (see the
/// module doc, "Domain handling for out-of-domain / boundary final
/// positions") -- e.g. a non-finite `best_x` coordinate, which would
/// indicate a genuine engine/problem defect upstream, not a domain-mapping
/// question this module can resolve.
pub fn structural_bias_scan(
    spec: &AlgorithmSpec,
    cfg: &StructuralBiasConfig,
) -> Result<StructuralBiasResult, BiasError> {
    if cfg.dim == 0 {
        return Err(BiasError::InvalidConfig("dim must be >= 1".into()));
    }
    if (cfg.runs as usize) < MIN_RUNS {
        return Err(BiasError::InvalidConfig(format!(
            "structural_bias_scan requires at least {MIN_RUNS} runs, got {}",
            cfg.runs
        )));
    }

    let mut reg = Registry::new();
    sezgi_components::register_builtins(&mut reg);

    let problem = F0Random::new(cfg.dim, cfg.seed);

    let mut spec = spec.clone();
    spec.termination.budget = cfg.budget; // always overridden, per this module's doc

    let engine = Engine::from_spec(&spec, &reg, problem.space())?;

    let mut final_positions = Vec::with_capacity(cfg.runs as usize);
    for run_id in 0..cfg.runs as u64 {
        let res = engine.run(&problem, RunConfig { master_seed: cfg.seed, run_id }, None)?;
        let BlockValues::Float(xs) = &res.best_x.blocks[0] else {
            unreachable!("F0Random's space is always a single Float block by construction")
        };
        final_positions.push(xs.clone());
    }

    scan_from_positions(final_positions, cfg.dim)
}

/// The statistics-only path, factored out so it is unit-testable without the
/// engine (synthetic `final_positions`, e.g. a hand-crafted biased/uniform
/// sampler) -- see this module's tests. `final_positions` must be `runs` x
/// `dim` (every row exactly `dim` entries long) with `runs >= `[`MIN_RUNS`].
pub(crate) fn scan_from_positions(
    final_positions: Vec<Vec<f64>>,
    dim: usize,
) -> Result<StructuralBiasResult, BiasError> {
    if dim == 0 {
        return Err(BiasError::InvalidConfig("dim must be >= 1".into()));
    }
    let runs = final_positions.len();
    if runs < MIN_RUNS {
        return Err(BiasError::InvalidConfig(format!(
            "scan_from_positions requires at least {MIN_RUNS} runs, got {runs}"
        )));
    }
    for (i, row) in final_positions.iter().enumerate() {
        if row.len() != dim {
            return Err(BiasError::InvalidConfig(format!(
                "scan_from_positions: row {i} has {} entries, expected dim={dim}",
                row.len()
            )));
        }
    }

    let mut per_dim_ks = Vec::with_capacity(dim);
    let mut per_dim_ad = Vec::with_capacity(dim);
    for d in 0..dim {
        // Domain mapping for this scan's own statistical tests -- see the
        // module doc, "Domain handling for out-of-domain / boundary final
        // positions": an out-of-domain coordinate (e.g. HHO's raw,
        // un-clamped dive-trial `best_x`) is clamped into `[0,1]` before
        // testing, counting it as boundary mass rather than erroring.
        // `final_positions` itself (returned in the result, and what
        // `report::BiasPlotData` plots) is left UNTOUCHED -- only the copy
        // used for KS/AD testing is clamped.
        let col: Vec<f64> =
            final_positions.iter().map(|row| row[d].clamp(0.0, 1.0)).collect();

        per_dim_ks.push(ks_uniform(&col)?);

        // A column containing an exact 0.0/1.0 (post-clamp) is the AD
        // statistic's own domain boundary: rather than calling `ad_uniform`
        // (which errors there by design -- ln(0)/ln(1) undefined), represent
        // the mathematically exact limit directly -- see the module doc's
        // "Domain handling..." section for the non-NaN derivation.
        if col.iter().any(|&v| v == 0.0 || v == 1.0) {
            per_dim_ad.push(AdResult { a2: f64::INFINITY, p_value: 0.0, n: col.len() });
        } else {
            per_dim_ad.push(ad_uniform(&col)?);
        }
    }

    let ks_pvals: Vec<f64> = per_dim_ks.iter().map(|r| r.p_value).collect();
    let ad_pvals: Vec<f64> = per_dim_ad.iter().map(|r| r.p_value).collect();
    let ks_holm = holm(&ks_pvals);
    let ad_holm = holm(&ad_pvals);

    let mut culprits = Vec::new();
    for (d, &p) in ks_holm.iter().enumerate() {
        if p < ALPHA {
            culprits.push(format!("dim {d}: KS holm-p={p:e}"));
        }
    }
    for (d, &p) in ad_holm.iter().enumerate() {
        if p < ALPHA {
            culprits.push(format!("dim {d}: AD holm-p={p:e}"));
        }
    }
    let holm_rejections_ks = ks_holm.iter().filter(|&&p| p < ALPHA).count();
    let holm_rejections_ad = ad_holm.iter().filter(|&&p| p < ALPHA).count();

    let verdict = if culprits.is_empty() {
        BiasVerdict::NoEvidence
    } else {
        BiasVerdict::Evidence {
            detail: format!(
                "evidence of structural bias toward non-uniform final positions on f0 \
                 (Holm-corrected p < {ALPHA} across dimensions): {}",
                culprits.join("; ")
            ),
        }
    };

    Ok(StructuralBiasResult {
        per_dim_ks,
        per_dim_ad,
        holm_rejections_ks,
        holm_rejections_ad,
        verdict,
        final_positions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::rng::RngStream;

    // ---- (a) statistics-only path, synthetic final_positions ----

    #[test]
    fn center_clustered_positions_yield_evidence() {
        // 30 runs, dim=2, every coordinate tightly clustered around 0.5 --
        // deterministic (no RNG), unambiguously non-uniform.
        let runs = 30;
        let final_positions: Vec<Vec<f64>> = (0..runs)
            .map(|i| {
                let jitter = (i as f64 - (runs as f64 - 1.0) / 2.0) * 0.001;
                vec![0.5 + jitter, 0.5 - jitter]
            })
            .collect();
        let r = scan_from_positions(final_positions, 2).expect("valid input");
        assert!(
            matches!(r.verdict, BiasVerdict::Evidence { .. }),
            "tightly center-clustered positions must be flagged as Evidence, got {:?}",
            r.verdict
        );
        assert!(r.holm_rejections_ks > 0 || r.holm_rejections_ad > 0);
        if let BiasVerdict::Evidence { detail } = &r.verdict {
            assert!(detail.contains("evidence of structural bias"));
            assert!(!detail.to_lowercase().contains("algorithm"),
                "evidence-not-accusation: detail must not name/blame a specific algorithm");
        }
    }

    #[test]
    fn seeded_uniform_positions_yield_no_evidence() {
        // dim=2, 500 draws per dim from this project's own RngStream --
        // deterministic given the seed, genuinely uniform.
        let mut rng = RngStream::from_master(20260830, &[4, 1]);
        let final_positions: Vec<Vec<f64>> =
            (0..500).map(|_| vec![rng.next_f64(), rng.next_f64()]).collect();
        let r = scan_from_positions(final_positions, 2).expect("valid input");
        assert_eq!(r.verdict, BiasVerdict::NoEvidence,
            "measured verdict at this seed: holm_rejections_ks={}, holm_rejections_ad={}",
            r.holm_rejections_ks, r.holm_rejections_ad);
        assert_eq!(r.holm_rejections_ks, 0);
        assert_eq!(r.holm_rejections_ad, 0);
    }

    #[test]
    fn rejects_dim_zero() {
        let err = scan_from_positions(vec![vec![]; 10], 0).unwrap_err();
        assert!(matches!(err, BiasError::InvalidConfig(_)), "got {err:?}");
    }

    #[test]
    fn rejects_too_few_runs() {
        let err = scan_from_positions(vec![vec![0.5]; 3], 1).unwrap_err();
        assert!(matches!(err, BiasError::InvalidConfig(_)), "got {err:?}");
    }

    #[test]
    fn rejects_ragged_rows() {
        let final_positions = vec![vec![0.1, 0.2], vec![0.3], vec![0.4, 0.5], vec![0.6, 0.7], vec![0.8, 0.9]];
        let err = scan_from_positions(final_positions, 2).unwrap_err();
        assert!(matches!(err, BiasError::InvalidConfig(_)), "got {err:?}");
    }

    // ---- (b) Holm-correction arithmetic, hand-checked 3-dim fixture ----
    //
    // Three columns (5 runs each), independently hand/pinned-algorithm
    // verified per-dimension:
    //
    //   col0 (dim 0) = [0.01, 0.02, 0.03, 0.04, 0.05]  (clustered, tiny p)
    //   col1 (dim 1) = [0.1, 0.3, 0.5, 0.7, 0.9]        (evenly spaced, huge p)
    //   col2 (dim 2) = [0.05, 0.15, 0.4, 0.6, 0.95]     (T3's KS hand-derived
    //                                                     fixture, mid p)
    //
    // col0 and col1 are `sezgi_stats::uniformity`'s own hand-derived/
    // pinned-algorithm KS+AD fixtures (see that module's tests --
    // `ks_uniform_clustered_sample_tiny_p`/`ad_uniform_clustered_sample_tiny_p`
    // and `ks_uniform_evenly_spaced_sample_large_p`/
    // `ad_uniform_evenly_spaced_sample_large_p`), so both KS and AD p-values
    // for those two columns are already independently pinned there. col2
    // reuses T3's KS-hand-derived array (`ks_uniform_hand_derived_d_both_
    // branches`, p = 0.8625362880828501); its AD p-value has no T3 fixture
    // of its own, so it is independently computed here (Python mirror of the
    // exact same pinned `ad_statistic`/`adinf` algorithm, per this project's
    // own "independently computed (pinned algorithm)" convention -- see e.g.
    // `sezgi_stats::uniformity`'s `ks_uniform_seeded_n500_large_p`):
    // a2 = 0.44079392948221763, p = 0.8075277279424962.
    //
    // KS p-values (dim order) = [5.833617325364261e-05, 0.9999999945629237,
    //                             0.8625362880828501]
    // Holm (m=3): sorted ascending is [dim0, dim2, dim1].
    //   rank1 (dim0): mult=3, adj = 3 * 5.833617325364261e-05
    //               = 1.7500851976092783e-04
    //   rank2 (dim2): mult=2, candidate = 2 * 0.8625362880828501
    //               = 1.7250725761657002 -> running_max stays at that
    //               candidate (> rank1's), clamped to 1.0
    //   rank3 (dim1): mult=1, candidate = 1 * 0.9999999945629237
    //               (< running_max 1.0) -> adj stays 1.0
    //   => holm-adjusted (original dim order) =
    //      [1.7500851976092783e-04, 1.0, 1.0]
    //   => only dim0 < ALPHA (0.01): holm_rejections_ks = 1
    //
    // AD p-values (dim order) = [4.8297186472368026e-08, 0.9995716562595469,
    //                             0.8075277279424962]
    // Holm (m=3): same sort order [dim0, dim2, dim1].
    //   rank1 (dim0): adj = 3 * 4.8297186472368026e-08
    //               = 1.4489155941710408e-07
    //   rank2 (dim2): candidate = 2 * 0.8075277279424962 = 1.6150554558849923,
    //               clamped to 1.0
    //   rank3 (dim1): candidate = 0.9995716562595469 < running_max (1.0)
    //               -> adj stays 1.0
    //   => holm-adjusted (original dim order) =
    //      [1.4489155941710408e-07, 1.0, 1.0]
    //   => only dim0 < ALPHA: holm_rejections_ad = 1
    //
    // Overall verdict: Evidence (dim0 rejects in both families).
    #[test]
    fn holm_correction_hand_checked_3dim_fixture() {
        let col0 = [0.01, 0.02, 0.03, 0.04, 0.05];
        let col1 = [0.1, 0.3, 0.5, 0.7, 0.9];
        let col2 = [0.05, 0.15, 0.4, 0.6, 0.95];
        let final_positions: Vec<Vec<f64>> =
            (0..5).map(|i| vec![col0[i], col1[i], col2[i]]).collect();

        let r = scan_from_positions(final_positions, 3).expect("valid fixture");

        assert_eq!(r.per_dim_ks.len(), 3);
        assert_eq!(r.per_dim_ad.len(), 3);

        let expected_ks_p = [5.833617325364261e-05, 0.9999999945629237, 0.8625362880828501];
        for (got, &want) in r.per_dim_ks.iter().zip(&expected_ks_p) {
            assert!((got.p_value - want).abs() < 1e-9, "got {}, want {}", got.p_value, want);
        }
        let expected_ad_p = [4.8297186472368026e-08, 0.9995716562595469, 0.8075277279424962];
        for (got, &want) in r.per_dim_ad.iter().zip(&expected_ad_p) {
            assert!((got.p_value - want).abs() < 1e-9, "got {}, want {}", got.p_value, want);
        }

        assert_eq!(r.holm_rejections_ks, 1, "only dim 0 should survive Holm at alpha=0.01");
        assert_eq!(r.holm_rejections_ad, 1, "only dim 0 should survive Holm at alpha=0.01");
        assert!(matches!(r.verdict, BiasVerdict::Evidence { .. }), "got {:?}", r.verdict);
    }

    // ---- engine-driven path ----

    fn f0_random_search_spec(pop_size: usize, budget: u64) -> AlgorithmSpec {
        sezgi_components::presets::random_search(pop_size, budget)
    }

    #[test]
    fn engine_driven_scan_is_deterministic() {
        let spec = f0_random_search_spec(10, 2000);
        let cfg = StructuralBiasConfig { runs: MIN_RUNS as u32 + 3, dim: 3, budget: 2000, seed: 20260830 };

        let r1 = structural_bias_scan(&spec, &cfg).expect("valid scan");
        let r2 = structural_bias_scan(&spec, &cfg).expect("valid scan");

        assert_eq!(r1, r2, "same config must yield a bit-identical StructuralBiasResult");
        // A few explicit bit-exact spot checks (per this task's brief),
        // beyond the whole-struct PartialEq above.
        assert_eq!(r1.per_dim_ks[0].d.to_bits(), r2.per_dim_ks[0].d.to_bits());
        assert_eq!(r1.per_dim_ks[0].p_value.to_bits(), r2.per_dim_ks[0].p_value.to_bits());
        assert_eq!(r1.per_dim_ad[0].a2.to_bits(), r2.per_dim_ad[0].a2.to_bits());
        assert_eq!(r1.final_positions[0][0].to_bits(), r2.final_positions[0][0].to_bits());
    }

    // Live smoke: random_search has no directional operator (uniform
    // resampling every generation, `replace/mu-plus-lambda` elitist
    // replacement) -- on f0 it SHOULD show no structural bias. ANCHORED (per
    // this project's convention): this is the actual measured verdict at
    // this seed/config, not a re-derivation -- see the p-values quoted
    // below, captured from this test's own first successful run.
    #[test]
    fn engine_driven_random_search_on_f0_yields_no_evidence_anchored() {
        let spec = f0_random_search_spec(20, 3000);
        let cfg = StructuralBiasConfig { runs: DEFAULT_RUNS, dim: 3, budget: 3000, seed: 20260830 };

        let r = structural_bias_scan(&spec, &cfg).expect("valid scan");

        // Measured at this exact config (captured from this test's own
        // first successful run, before the assertions below were added):
        //   per_dim_ks p-values = [0.20199785220840136, 0.9793597207603825,
        //                          0.7528879047657814]
        //   per_dim_ad p-values = [0.2684700046343683, 0.8938238884215298,
        //                          0.7760983926307092]
        // None below ALPHA (0.01), let alone after Holm correction --
        // consistent with random_search's uniform-resampling generator
        // having no directional operator to induce structural bias.
        assert_eq!(r.holm_rejections_ks, 0);
        assert_eq!(r.holm_rejections_ad, 0);
        assert_eq!(r.verdict, BiasVerdict::NoEvidence,
            "measured verdict for random_search(pop=20, budget=3000) on f0(dim=3), runs=30, seed=20260830: {:?}",
            r.verdict);
    }

    // ---- C1 regression: boundary-clamping preset (pso) must COMPLETE, not
    // error, and its own boundary clustering must surface as Evidence ----
    //
    // ANCHORED (per this project's convention): measured at this exact
    // config (pop=20, budget=3000, dim=3, runs=30, seed=20260830), captured
    // from this test's own first successful run after the fix. Before the
    // fix, this exact config returned `Err(BiasError::Stats(..))` from
    // `ad_uniform` ("sorted sample at index 0 = 0 touches the domain
    // boundary") -- see the module doc's "Domain handling..." section. PSO's
    // boundary-clamping repair routinely parks a coordinate at exactly
    // 0.0/1.0 on f0 (fitness is random, so a boundary-clamped point is as
    // likely as any other to be the run's "best"); this is precisely the
    // boundary-clustering signature the scan exists to detect, so `Evidence`
    // (driven by the AD family, via the new boundary-limit `AdResult`) is
    // the CORRECT measured verdict here, not a false positive.
    #[test]
    fn engine_driven_pso_on_f0_completes_and_flags_boundary_evidence_anchored() {
        let spec = sezgi_components::presets::pso(20, 3000);
        let cfg = StructuralBiasConfig { runs: 30, dim: 3, budget: 3000, seed: 20260830 };

        let r = structural_bias_scan(&spec, &cfg).expect(
            "pso must now COMPLETE the scan (pre-fix: errored on exact-0.0 boundary values)",
        );

        // Measured AD p-values: [0.0, 0.0, 0.0] (all three dimensions hit
        // the boundary-limit case), AD a2: [inf, inf, inf], holm_rejections_ad = 3.
        assert_eq!(r.per_dim_ad.len(), 3);
        for a in &r.per_dim_ad {
            assert_eq!(a.a2, f64::INFINITY, "measured: every dimension hit the boundary limit");
            assert_eq!(a.p_value, 0.0);
        }
        assert_eq!(r.holm_rejections_ad, 3);
        assert!(
            matches!(r.verdict, BiasVerdict::Evidence { .. }),
            "a boundary-clamping preset flagging Evidence of boundary clustering on f0 is the \
             expected CORRECT outcome (this is the exact signature the scan exists to detect), \
             got {:?}",
            r.verdict
        );
    }

    // ---- C1 regression: HHO (raw, un-clamped, sometimes out-of-domain
    // best_x) must also COMPLETE, not error ----
    //
    // ANCHORED: measured at this exact config, captured from this test's own
    // first successful run after the fix. Before the fix, this exact config
    // returned `Err(BiasError::Stats(..))` from `ks_uniform` ("value ... is
    // outside [0,1]") -- HHO's Evaluator-observed best_x can be a raw
    // pre-boundary-repair dive-trial point (see `crates/components/src/
    // hho.rs`'s module doc); measured here: final positions range as far as
    // -0.82 to 1.01, well outside [0,1], confirming the out-of-domain clamp
    // path is genuinely exercised by this fixture, not merely reachable in
    // theory.
    #[test]
    fn engine_driven_hho_on_f0_completes_and_flags_boundary_evidence_anchored() {
        let spec = sezgi_components::presets::hho(20, 3000);
        let cfg = StructuralBiasConfig { runs: 30, dim: 3, budget: 3000, seed: 20260830 };

        let r = structural_bias_scan(&spec, &cfg).expect(
            "hho must now COMPLETE the scan (pre-fix: errored on an out-of-[0,1] best_x)",
        );

        // Confirm the raw (unclamped) final_positions genuinely contain an
        // out-of-domain value -- proof this fixture exercises the
        // out-of-domain clamp path, not just the boundary-AD path.
        let raw_min = r.final_positions.iter().flatten().cloned().fold(f64::INFINITY, f64::min);
        let raw_max =
            r.final_positions.iter().flatten().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(raw_min < 0.0, "expected a measured out-of-domain (< 0.0) raw position, got min={raw_min}");
        assert!(raw_max > 1.0 || raw_min < 0.0,
            "expected at least one raw position outside [0,1], got min={raw_min} max={raw_max}");

        // Measured: all three AD dimensions hit the boundary limit too (an
        // out-of-domain value clamps to exactly 0.0 or 1.0).
        for a in &r.per_dim_ad {
            assert_eq!(a.a2, f64::INFINITY);
            assert_eq!(a.p_value, 0.0);
        }
        assert_eq!(r.holm_rejections_ad, 3);
        assert!(
            matches!(r.verdict, BiasVerdict::Evidence { .. }),
            "got {:?}",
            r.verdict
        );
    }
}
