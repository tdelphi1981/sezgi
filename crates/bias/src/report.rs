//! One-call bias report (M3-1 Task 7): consumes the structural (T4) and
//! central (T5) bias scans this crate hosts and assembles a single
//! [`BiasReport`] -- one `bias_report` call per algorithm, no separate
//! wiring of `structural_bias_scan`/`central_bias_scan` required.
//!
//! The LaTeX table this module emits mirrors `sezgi_stats::report`'s own
//! emitter style: booktabs (`\toprule`/`\midrule`/`\bottomrule`), numbers in
//! `{:.3e}` exponential notation, `\pm`/math-mode content always inside
//! `$...$`, and -- the M2c lesson that module's own doc/tests call out --
//! never the literal Rust-formatted text `NaN` (see [`fmt_stat`]/[`fmt_p`]).
//!
//! # "Evidence not accusation" in this module's own LaTeX phrasing
//!
//! Every row's verdict cell is one of exactly two phrase shapes (this
//! task's brief, verbatim): **"no evidence of {test}"** when the
//! underlying scan found none, or **"evidence consistent with {test}"**
//! when it did -- never "the algorithm is biased" and no ranking language
//! (no "better/worse than", no cross-algorithm comparison at all: this
//! report is about ONE algorithm's own scan results). See [`phrase`].
//!
//! # T6 (signature test): deferred
//!
//! Task 6 (the Rajwar-Deep signature-bias test) is DEFERRED by controller
//! ruling: the method could not be pinned from accessible sources (no
//! reachable source -- paper or reference implementation -- fully specifies
//! its statistical procedure). There is therefore no `signature.rs` and no
//! `signature_scan` in this crate yet.
//!
//! [`BiasReport::signature`] is `Option<BiasVerdict>`, always `None` today.
//! `BiasVerdict` (not a bespoke placeholder type) is deliberate: this
//! crate's own root doc already documents `BiasVerdict` as "shared by every
//! bias-scan flavour this crate hosts (structural bias, central bias here;
//! T6's own scan reuses it)" -- i.e. the future signature scan is already
//! expected to reduce to the same `NoEvidence`/`Evidence{detail}` shape
//! `structural_bias_scan`/`central_bias_scan` produce today. When T6 is
//! un-deferred, its scan need only produce a `BiasVerdict` and this field
//! becomes `Some(..)`; no shape change to [`BiasReport`] is anticipated.
//! Until then, [`bias_report_latex`] renders an honest "not run" row: it
//! states plainly that the test was not run because the method could not
//! be pinned from accessible sources -- a factual status, not a citation
//! (see [`bias_report_latex`]'s signature-row arm).

use sezgi_core::spec::AlgorithmSpec;
use sezgi_stats::ranks::holm;
use sezgi_stats::uniformity::{AdResult, KsResult};

use crate::central::{central_bias_scan, CentralBiasConfig, CentralBiasResult};
use crate::structural::{structural_bias_scan, StructuralBiasConfig, StructuralBiasResult};
use crate::{BiasError, BiasVerdict};

/// Default BBOB fids for [`BiasReportConfig::new`]'s `central_fids`: the
/// same three-fid subset already used as `central.rs`'s own anchored
/// end-to-end smoke test
/// (`engine_driven_random_search_on_bbob_yields_no_evidence_anchored`) --
/// fid 1 (Sphere, separable, no `f_pen`), fid 4 (an `f_pen`-bearing,
/// moderately ill-conditioned fid -- the exact fid the review-round-1
/// `f_pen`-displacement fix targeted), and fid 13 (a multimodal fid). A
/// small, cheap, representative default rather than the full 24-fid BBOB
/// suite. All three are translation-invariant (none is in
/// `central_bias_scan`'s excluded `{5, 6, 20, 24}` set), so this default is
/// always accepted.
pub const DEFAULT_CENTRAL_FIDS: [u32; 3] = [1, 4, 13];

/// Default BBOB instances for [`BiasReportConfig::new`]'s
/// `central_instances`: the same pair already used throughout `central.rs`'s
/// own tests (e.g. `engine_driven_scan_is_deterministic`,
/// `engine_driven_random_search_on_bbob_yields_no_evidence_anchored`).
pub const DEFAULT_CENTRAL_INSTANCES: [u32; 2] = [1, 2];

/// Default runs-per-pair for [`BiasReportConfig::new`]'s `central_runs_per`:
/// Kudela's own verified run count -- `central.rs`'s module doc,
/// "Performance measure", quoted verbatim: "mean error ... over **20**
/// independent runs."
pub const DEFAULT_CENTRAL_RUNS_PER: u32 = 20;

/// Configuration for [`bias_report`]: the natural union of
/// [`StructuralBiasConfig`] and [`CentralBiasConfig`] -- `dim`/`budget`/
/// `seed` are shared (both scans run the same algorithm under the same
/// evaluation budget and seed), while each scan's own knobs (run counts,
/// which BBOB fids/instances) stay per-scan.
#[derive(Debug, Clone, PartialEq)]
pub struct BiasReportConfig {
    /// Shared dimensionality: f0's domain size for the structural scan AND
    /// BBOB's dimensionality for the central scan. Must be `>= 2`
    /// (`central_bias_scan`'s own `dim >= 2` floor -- BBOB requires it --
    /// is the binding constraint here; `structural_bias_scan` alone would
    /// accept `dim >= 1`).
    pub dim: usize,
    /// Shared per-run evaluation budget for both scans. Overrides
    /// `spec.termination.budget` in both, per each scan module's own
    /// "budget is always overridden" convention.
    pub budget: u64,
    /// Shared seed, passed to both scans' own `seed` field. Each scan
    /// derives its own collision-free RNG streams from it independently --
    /// see `structural.rs`'s "Seed derivation" and `central.rs`'s "Pairing
    /// rationale" docs; reusing the same numeric seed for both scans does
    /// not collide their RNG namespaces (they are entirely separate scans,
    /// run one after the other, each owning its own `Engine`).
    pub seed: u64,
    /// Independent runs for the structural (f0) scan. Default origin:
    /// `crate::structural::DEFAULT_RUNS` (30) -- the BIAS toolbox's own
    /// verified minimum/default run count.
    pub structural_runs: u32,
    /// BBOB fids for the central scan. Default origin:
    /// [`DEFAULT_CENTRAL_FIDS`]. Every entry MUST be translation-invariant
    /// (`central_bias_scan` rejects fids 5, 6, 20, 24 with
    /// [`BiasError::InvalidConfig`] -- see `central.rs`'s "Construction
    /// fix" doc); this config's own default already respects that
    /// exclusion.
    pub central_fids: Vec<u32>,
    /// BBOB instances for the central scan. Default origin:
    /// [`DEFAULT_CENTRAL_INSTANCES`].
    pub central_instances: Vec<u32>,
    /// Independent runs per `(fid, instance)` pair for the central scan.
    /// Default origin: [`DEFAULT_CENTRAL_RUNS_PER`] (20) -- Kudela's own
    /// verified run count.
    pub central_runs_per: u32,
}

impl BiasReportConfig {
    /// Convenience constructor: `dim`/`budget`/`seed` are the only fields
    /// with no natural verified default (they must fit the caller's
    /// `AlgorithmSpec`/experiment); every other field is filled with this
    /// module's own documented verified-method default -- see each field's
    /// own doc for its origin. Every field is `pub`, so override any of
    /// them directly afterwards.
    pub fn new(dim: usize, budget: u64, seed: u64) -> Self {
        Self {
            dim,
            budget,
            seed,
            structural_runs: crate::structural::DEFAULT_RUNS,
            central_fids: DEFAULT_CENTRAL_FIDS.to_vec(),
            central_instances: DEFAULT_CENTRAL_INSTANCES.to_vec(),
            central_runs_per: DEFAULT_CENTRAL_RUNS_PER,
        }
    }
}

/// The raw evidence vectors behind [`BiasReport`]'s LaTeX summary -- no
/// recomputation, just the same vectors already produced by
/// [`structural_bias_scan`]/[`central_bias_scan`], surfaced for a caller
/// that wants to plot them directly.
#[derive(Debug, Clone, PartialEq)]
pub struct BiasPlotData {
    /// [`StructuralBiasResult::final_positions`], unchanged.
    pub final_positions: Vec<Vec<f64>>,
    /// [`CentralBiasResult::gap_centered`], unchanged.
    pub gap_centered: Vec<f64>,
    /// [`CentralBiasResult::gap_shifted`], unchanged.
    pub gap_shifted: Vec<f64>,
}

/// One algorithm's complete bias report: both scans this crate can run
/// today, T6's still-`None` placeholder, and a ready-to-paste LaTeX table.
#[derive(Debug, Clone, PartialEq)]
pub struct BiasReport {
    /// The structural-bias (f0) scan's full result.
    pub structural: StructuralBiasResult,
    /// The central-bias (BBOB centered-vs-shifted) scan's full result.
    pub central: CentralBiasResult,
    /// The T6 (signature) scan's result -- always `None` today; see this
    /// module's doc, "T6 (signature test): deferred".
    pub signature: Option<BiasVerdict>,
    /// LaTeX table: one row per test (structural KS, structural AD,
    /// central Wilcoxon, signature), each with a statistic, a $p$-value,
    /// and a verdict phrase -- see [`bias_report_latex`].
    pub latex_summary: String,
    /// The raw vectors behind the report -- see [`BiasPlotData`].
    pub plot_data: BiasPlotData,
}

/// Runs both the structural and central bias scans on `spec` and assembles
/// a single [`BiasReport`] -- the one-call entry point this task adds.
///
/// # Errors
/// Whatever [`structural_bias_scan`] or [`central_bias_scan`] themselves
/// return (both scans' own `Errors` sections apply unchanged); this
/// function adds no validation of its own beyond delegating `cfg`'s shared
/// fields into each scan's own config.
pub fn bias_report(spec: &AlgorithmSpec, cfg: &BiasReportConfig) -> Result<BiasReport, BiasError> {
    let structural = structural_bias_scan(
        spec,
        &StructuralBiasConfig {
            runs: cfg.structural_runs,
            dim: cfg.dim,
            budget: cfg.budget,
            seed: cfg.seed,
        },
    )?;

    let central = central_bias_scan(
        spec,
        &CentralBiasConfig {
            fids: cfg.central_fids.clone(),
            dim: cfg.dim,
            instances_shifted: cfg.central_instances.clone(),
            runs_per: cfg.central_runs_per,
            budget: cfg.budget,
            seed: cfg.seed,
        },
    )?;

    // T6 is deferred -- see this module's doc.
    Ok(assemble_report(structural, central, None))
}

/// Assembles a [`BiasReport`] (LaTeX table included) from already-computed
/// scan results, without invoking either scan -- factored out so report
/// ASSEMBLY (LaTeX rendering, phrasing) is unit-testable against hand-built
/// [`StructuralBiasResult`]/[`CentralBiasResult`] fixtures, mirroring
/// `structural.rs`'s/`central.rs`'s own `scan_from_positions`/
/// `scan_from_gaps` statistics-only split.
pub fn assemble_report(
    structural: StructuralBiasResult,
    central: CentralBiasResult,
    signature: Option<BiasVerdict>,
) -> BiasReport {
    let latex_summary = bias_report_latex(&structural, &central, signature.as_ref());
    let plot_data = BiasPlotData {
        final_positions: structural.final_positions.clone(),
        gap_centered: central.gap_centered.clone(),
        gap_shifted: central.gap_shifted.clone(),
    };
    BiasReport { structural, central, signature, latex_summary, plot_data }
}

/// Formats a test statistic for the LaTeX table. NaN-free (the M2c lesson
/// `sezgi_stats::report` already established): a non-finite value renders
/// as the plain text `n/a`, never the Rust-formatted literal `NaN`.
fn fmt_stat(x: f64) -> String {
    if x.is_finite() {
        format!("{:.3e}", x)
    } else {
        "n/a".to_string()
    }
}

/// Formats a $p$-value in math mode (`$0.1234$`), consistent with
/// `sezgi_stats::report::pairwise_tests_latex`'s own `$p=...$` convention
/// (the column header here already reads "$p$", so the cell itself is the
/// bare math-mode number). NaN-free: same `n/a` fallback as [`fmt_stat`],
/// deliberately kept OUTSIDE math mode -- `n/a` is not valid LaTeX math
/// content, so it must never be wrapped in `$...$`.
fn fmt_p(p: f64) -> String {
    if p.is_finite() {
        format!("${:.4}$", p)
    } else {
        "n/a".to_string()
    }
}

/// The single most significant (smallest finite $p$-value) entry of a
/// per-dimension KS battery -- NaN-safe: non-finite entries are never
/// preferred over a finite one; if every entry is non-finite, falls back to
/// the first entry (which then renders as `n/a` via [`fmt_stat`]/[`fmt_p`],
/// never a literal `NaN`). Returns `(NaN, NaN)` for an empty battery.
fn worst_ks(per_dim: &[KsResult]) -> (f64, f64) {
    per_dim
        .iter()
        .filter(|r| r.p_value.is_finite())
        .min_by(|a, b| a.p_value.partial_cmp(&b.p_value).expect("filtered to finite p_value"))
        .map(|r| (r.d, r.p_value))
        .or_else(|| per_dim.first().map(|r| (r.d, r.p_value)))
        .unwrap_or((f64::NAN, f64::NAN))
}

/// Same rule as [`worst_ks`], for a per-dimension AD battery.
fn worst_ad(per_dim: &[AdResult]) -> (f64, f64) {
    per_dim
        .iter()
        .filter(|r| r.p_value.is_finite())
        .min_by(|a, b| a.p_value.partial_cmp(&b.p_value).expect("filtered to finite p_value"))
        .map(|r| (r.a2, r.p_value))
        .or_else(|| per_dim.first().map(|r| (r.a2, r.p_value)))
        .unwrap_or((f64::NAN, f64::NAN))
}

/// Index of the smallest value in `pvals`, ignoring non-finite entries.
/// `None` for an empty or all-non-finite slice.
fn min_finite_idx(pvals: &[f64]) -> Option<usize> {
    pvals
        .iter()
        .enumerate()
        .filter(|(_, &p)| p.is_finite())
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("filtered to finite"))
        .map(|(i, _)| i)
}

/// The dimension whose HOLM-ADJUSTED $p$-value is smallest -- i.e. the same
/// (D, p) pair [`worst_ks`] would report, except `p` is the Holm-adjusted
/// value rather than the raw one, so the printed number is the one the
/// verdict actually compares against `ALPHA` (structural.rs's own
/// `holm_rejections_ks`/`holm_rejections_ad` decision). Fix Round 1: the
/// table previously printed the raw minimum $p$ next to a verdict driven by
/// the Holm-adjusted $p$, which could read as contradictory (a small raw
/// $p$ next to "no evidence" once Holm-corrected, or vice versa).
///
/// Reuses `sezgi_stats::ranks::holm` directly -- no Holm-logic duplication
/// (the same function `structural_bias_scan` itself calls). Holm's
/// adjusted values are non-decreasing in ascending-raw-p order (each step's
/// `running_max` only grows), so the dimension with the smallest RAW
/// $p$-value is always also the one with the smallest Holm-ADJUSTED
/// $p$-value -- selecting by raw $p$ and reporting that dimension's
/// adjusted $p$ is therefore exact, not an approximation.
///
/// NaN-safe: an empty battery, or one containing any non-finite raw
/// $p$-value (which `holm` cannot process -- it panics on `NaN` via
/// `partial_cmp`), falls back to [`worst_ks`]'s raw-$p$ selection instead
/// of calling `holm` at all; [`fmt_p`] still renders any resulting
/// non-finite value as `n/a`, never the literal `NaN`.
fn worst_ks_holm(per_dim: &[KsResult]) -> (f64, f64) {
    if per_dim.is_empty() || per_dim.iter().any(|r| !r.p_value.is_finite()) {
        return worst_ks(per_dim);
    }
    let pvals: Vec<f64> = per_dim.iter().map(|r| r.p_value).collect();
    let adjusted = holm(&pvals);
    let idx = min_finite_idx(&pvals).expect("non-empty and all-finite, checked above");
    (per_dim[idx].d, adjusted[idx])
}

/// Same rule as [`worst_ks_holm`], for a per-dimension AD battery.
fn worst_ad_holm(per_dim: &[AdResult]) -> (f64, f64) {
    if per_dim.is_empty() || per_dim.iter().any(|r| !r.p_value.is_finite()) {
        return worst_ad(per_dim);
    }
    let pvals: Vec<f64> = per_dim.iter().map(|r| r.p_value).collect();
    let adjusted = holm(&pvals);
    let idx = min_finite_idx(&pvals).expect("non-empty and all-finite, checked above");
    (per_dim[idx].a2, adjusted[idx])
}

/// The "evidence not accusation" verdict phrase, per this task's brief,
/// verbatim: `"no evidence of {test}"` when `no_evidence` is true, else
/// `"evidence consistent with {test}"`. Never claims an algorithm
/// categorically IS biased, and carries no ranking/comparison language --
/// matches [`crate::BiasVerdict`]'s own module-root doc.
fn phrase(no_evidence: bool, test: &str) -> String {
    if no_evidence {
        format!("no evidence of {test}")
    } else {
        format!("evidence consistent with {test}")
    }
}

/// Renders the one-call LaTeX summary: one row per test -- structural KS,
/// structural AD, central Wilcoxon, and the signature test (an honest "not
/// run" row when `signature` is `None`). Booktabs style, `{:.3e}`
/// statistics, math-mode `$p$` cells -- mirrors
/// `sezgi_stats::report::summary_table_latex`'s own conventions.
///
/// The printed `$p$` is always the SAME number the row's own verdict was
/// decided against (Fix Round 1): the two structural rows print the
/// Holm-adjusted minimum $p$ (see [`worst_ks_holm`]/[`worst_ad_holm`]) --
/// labelled "Holm-corrected" in the row's own Test cell, matching
/// `structural.rs`'s own "Holm-corrected p < ALPHA" decision-rule wording
/// -- since that IS the quantity `holm_rejections_ks`/`holm_rejections_ad`
/// compare against `ALPHA`; the central row prints the raw Wilcoxon $p$
/// unchanged, since `central_bias_scan` makes exactly one comparison and
/// applies no multiple-testing correction to it at all.
fn bias_report_latex(
    structural: &StructuralBiasResult,
    central: &CentralBiasResult,
    signature: Option<&BiasVerdict>,
) -> String {
    let mut latex = String::from("\\begin{tabular}{llll}\n\\toprule\n");
    latex.push_str("Test & Statistic & $p$ & Verdict \\\\\n\\midrule\n");

    let (ks_stat, ks_p) = worst_ks_holm(&structural.per_dim_ks);
    let ks_no_evidence = structural.holm_rejections_ks == 0;
    latex.push_str(&format!(
        "Structural bias (KS, Holm-corrected) & {} & {} & {} \\\\\n",
        fmt_stat(ks_stat),
        fmt_p(ks_p),
        phrase(ks_no_evidence, "structural bias")
    ));

    let (ad_stat, ad_p) = worst_ad_holm(&structural.per_dim_ad);
    let ad_no_evidence = structural.holm_rejections_ad == 0;
    latex.push_str(&format!(
        "Structural bias (AD, Holm-corrected) & {} & {} & {} \\\\\n",
        fmt_stat(ad_stat),
        fmt_p(ad_p),
        phrase(ad_no_evidence, "structural bias")
    ));

    let central_no_evidence = matches!(central.verdict, BiasVerdict::NoEvidence);
    latex.push_str(&format!(
        "Central bias (Wilcoxon) & {} & {} & {} \\\\\n",
        fmt_stat(central.wilcoxon.w_statistic),
        fmt_p(central.wilcoxon.p_value),
        phrase(central_no_evidence, "center-bias exploitation")
    ));

    match signature {
        Some(verdict) => {
            let no_evidence = matches!(verdict, BiasVerdict::NoEvidence);
            latex.push_str(&format!(
                "Signature (Rajwar-Deep) & -- & -- & {} \\\\\n",
                phrase(no_evidence, "signature bias")
            ));
        }
        None => {
            latex.push_str(
                "Signature (Rajwar-Deep) & -- & -- & not run: the Rajwar-Deep method could \
                 not be pinned from accessible sources \\\\\n",
            );
        }
    }

    latex.push_str("\\bottomrule\n\\end{tabular}\n");
    latex
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_stats::{WilcoxonMethod, WilcoxonResult};

    fn ks(d: f64, p: f64) -> KsResult {
        KsResult { d, p_value: p, n: 30 }
    }
    fn ad(a2: f64, p: f64) -> AdResult {
        AdResult { a2, p_value: p, n: 30 }
    }
    fn wilcoxon(w: f64, p: f64) -> WilcoxonResult {
        WilcoxonResult { w_statistic: w, z: -1.0, p_value: p, n_effective: 20, method: WilcoxonMethod::NormalApprox }
    }

    fn no_evidence_structural() -> StructuralBiasResult {
        StructuralBiasResult {
            per_dim_ks: vec![ks(0.05, 0.90), ks(0.06, 0.85)],
            per_dim_ad: vec![ad(0.3, 0.80), ad(0.35, 0.75)],
            holm_rejections_ks: 0,
            holm_rejections_ad: 0,
            verdict: BiasVerdict::NoEvidence,
            final_positions: vec![vec![0.4, 0.6]; 30],
        }
    }

    fn evidence_structural() -> StructuralBiasResult {
        StructuralBiasResult {
            per_dim_ks: vec![ks(0.9, 0.0001), ks(0.06, 0.85)],
            per_dim_ad: vec![ad(5.0, 0.00005), ad(0.35, 0.75)],
            holm_rejections_ks: 1,
            holm_rejections_ad: 1,
            verdict: BiasVerdict::Evidence {
                detail: "evidence of structural bias toward non-uniform final positions on f0 \
                          (Holm-corrected p < 0.01 across dimensions): dim 0: KS holm-p=2e-4"
                    .to_string(),
            },
            final_positions: vec![vec![0.5, 0.6]; 30],
        }
    }

    fn no_evidence_central() -> CentralBiasResult {
        CentralBiasResult {
            gap_centered: vec![1.0, 1.1, 0.9, 1.2, 1.0],
            gap_shifted: vec![1.0, 1.0, 1.1, 0.9, 1.05],
            wilcoxon: wilcoxon(6.0, 0.80),
            effect: 0.1,
            verdict: BiasVerdict::NoEvidence,
        }
    }

    fn evidence_central() -> CentralBiasResult {
        CentralBiasResult {
            gap_centered: vec![0.01, 0.02, 0.03, 0.04, 0.05],
            gap_shifted: vec![5.0, 6.0, 7.0, 8.0, 9.0],
            wilcoxon: wilcoxon(0.0, 0.0078125),
            effect: 1.0,
            verdict: BiasVerdict::Evidence {
                detail: "evidence of center-bias exploitation: the performance gap ... \
                          p=7.8e-3 < 0.05, Cliff's delta=1.0000 [large]"
                    .to_string(),
            },
        }
    }

    // ---- report assembly from hand-built results ----

    #[test]
    fn all_no_evidence_yields_exact_no_evidence_phrases() {
        let report = assemble_report(no_evidence_structural(), no_evidence_central(), None);

        assert_eq!(report.structural.verdict, BiasVerdict::NoEvidence);
        assert_eq!(report.central.verdict, BiasVerdict::NoEvidence);
        assert!(report.signature.is_none());

        let latex = &report.latex_summary;
        // Structural bias appears exactly twice (KS row, AD row), central once.
        assert_eq!(
            latex.matches("no evidence of structural bias").count(),
            2,
            "expected KS and AD rows both to read 'no evidence of structural bias', got:\n{latex}"
        );
        assert_eq!(
            latex.matches("no evidence of center-bias exploitation").count(),
            1,
            "expected the central row to read 'no evidence of center-bias exploitation', got:\n{latex}"
        );
        assert!(
            !latex.contains("evidence consistent with"),
            "no row should claim evidence when every underlying verdict is NoEvidence:\n{latex}"
        );
    }

    #[test]
    fn evidence_yields_exact_evidence_consistent_with_phrases() {
        let report = assemble_report(evidence_structural(), evidence_central(), None);

        let latex = &report.latex_summary;
        assert_eq!(
            latex.matches("evidence consistent with structural bias").count(),
            2,
            "expected both KS and AD rows to read 'evidence consistent with structural bias', got:\n{latex}"
        );
        assert!(
            latex.contains("evidence consistent with center-bias exploitation"),
            "expected the central row to read 'evidence consistent with center-bias exploitation', got:\n{latex}"
        );
        // "Evidence not accusation": the phrase must never claim the
        // algorithm itself IS biased, nor rank it against anything.
        assert!(!latex.to_lowercase().contains("is biased"));
        assert!(!latex.to_lowercase().contains("better than"));
        assert!(!latex.to_lowercase().contains("worse than"));
    }

    #[test]
    fn none_signature_renders_honest_not_run_row() {
        let report = assemble_report(no_evidence_structural(), no_evidence_central(), None);
        let latex = &report.latex_summary;

        assert!(
            latex.contains(
                "Signature (Rajwar-Deep) & -- & -- & not run: the Rajwar-Deep method could \
                 not be pinned from accessible sources"
            ),
            "signature row must state factually why it was not run, got:\n{latex}"
        );
        // Factual, not a citation: no bracketed/parenthetical reference
        // markers, no arXiv/DOI-shaped text.
        assert!(!latex.contains("arXiv"));
        assert!(!latex.contains("doi"));
        assert!(!latex.contains("[1]"));
    }

    #[test]
    fn latex_contains_no_nan_under_nan_prone_inputs() {
        // Hand-built result with an empty per-dimension battery (the edge
        // case that would otherwise force `worst_ks`/`worst_ad` to fall
        // back to a non-finite placeholder) and an explicit NaN p-value
        // elsewhere in the fixture.
        let structural = StructuralBiasResult {
            per_dim_ks: vec![],
            per_dim_ad: vec![ad(f64::NAN, f64::NAN)],
            holm_rejections_ks: 0,
            holm_rejections_ad: 0,
            verdict: BiasVerdict::NoEvidence,
            final_positions: vec![],
        };
        let central = CentralBiasResult {
            gap_centered: vec![],
            gap_shifted: vec![],
            wilcoxon: wilcoxon(f64::NAN, f64::NAN),
            effect: f64::NAN,
            verdict: BiasVerdict::NoEvidence,
        };

        let report = assemble_report(structural, central, None);
        let latex = &report.latex_summary;

        assert!(!latex.contains("NaN"), "LaTeX must never contain the literal 'NaN', got:\n{latex}");
        assert!(latex.contains("n/a"), "non-finite cells should render 'n/a', got:\n{latex}");
    }

    // ---- math-mode / escaping discipline (per the stats precedent) ----

    #[test]
    fn p_values_are_math_mode_and_balanced() {
        let report = assemble_report(evidence_structural(), evidence_central(), None);
        let latex = &report.latex_summary;

        for line in latex.lines() {
            if line.contains('$') {
                let dollars = line.matches('$').count();
                assert!(dollars % 2 == 0, "unbalanced math-mode delimiters in: {line}");
            }
        }
        // p-value cells are wrapped in bare math mode, e.g. "$0.0002$" --
        // the Holm-ADJUSTED KS p for this fixture (raw p=0.0001, family
        // size m=2, so adjusted = 2 * 0.0001 = 0.0002; see Fix Round 1's
        // `worst_ks_holm_reports_the_holm_adjusted_p_not_the_raw_one`
        // below for the arithmetic pinned independently of this string).
        assert!(latex.contains("$0.0002$"), "expected a math-mode p-value cell, got:\n{latex}");
    }

    // ---- Fix Round 1: structural rows print Holm-adjusted p, not raw ----

    #[test]
    fn worst_ks_holm_reports_the_holm_adjusted_p_not_the_raw_one() {
        // structural.rs's own hand-checked 3-dim Holm fixture (see that
        // module's `holm_correction_hand_checked_3dim_fixture` test
        // comment) -- reused verbatim, not re-derived, so this pins the
        // SAME arithmetic that module already hand-checked.
        let per_dim = vec![
            ks(0.0, 5.833617325364261e-05),
            ks(0.0, 0.9999999945629237),
            ks(0.0, 0.8625362880828501),
        ];
        let (_, p) = worst_ks_holm(&per_dim);
        assert!(
            (p - 1.7500851976092783e-04).abs() < 1e-12,
            "expected the Holm-adjusted p (not the raw 5.83e-05), got {p}"
        );
    }

    #[test]
    fn worst_ad_holm_reports_the_holm_adjusted_p_not_the_raw_one() {
        // Same fixture family, AD side -- see structural.rs's comment.
        let per_dim = vec![
            ad(0.0, 4.8297186472368026e-08),
            ad(0.0, 0.9995716562595469),
            ad(0.0, 0.8075277279424962),
        ];
        let (_, p) = worst_ad_holm(&per_dim);
        assert!(
            (p - 1.4489155941710408e-07).abs() < 1e-12,
            "expected the Holm-adjusted p (not the raw 4.83e-08), got {p}"
        );
    }

    #[test]
    fn structural_row_labels_say_holm_corrected_central_does_not() {
        let report = assemble_report(evidence_structural(), evidence_central(), None);
        let latex = &report.latex_summary;

        assert!(latex.contains("Structural bias (KS, Holm-corrected)"), "got:\n{latex}");
        assert!(latex.contains("Structural bias (AD, Holm-corrected)"), "got:\n{latex}");
        assert!(latex.contains("Central bias (Wilcoxon)"), "got:\n{latex}");

        let central_line = latex
            .lines()
            .find(|l| l.starts_with("Central bias"))
            .expect("central row present");
        assert!(
            !central_line.contains("Holm"),
            "central row's own p is raw (single-test, no correction applied) -- it must not \
             claim Holm-correction: {central_line}"
        );
    }

    #[test]
    fn printed_p_matches_the_p_the_verdict_was_decided_against() {
        // A fixture where the raw minimum p and the Holm-adjusted p differ
        // enough to be distinguishable at the table's own {:.4} precision
        // (this is exactly the scenario Fix Round 1 targets: the OLD table
        // printed the raw p here, which would have shown as "$0.0001$"
        // even though holm_rejections_ks/ad were driven by the larger,
        // adjusted 0.0002/0.0000001-scale numbers).
        let structural = StructuralBiasResult {
            per_dim_ks: vec![ks(0.9, 0.0001), ks(0.06, 0.85)],
            per_dim_ad: vec![ad(0.3, 0.75), ad(0.35, 0.8)],
            holm_rejections_ks: 1,
            holm_rejections_ad: 0,
            verdict: BiasVerdict::Evidence { detail: "evidence of structural bias".to_string() },
            final_positions: vec![vec![0.5, 0.5]; 30],
        };
        let report = assemble_report(structural, no_evidence_central(), None);
        let latex = &report.latex_summary;

        // The raw KS p (0.0001) must NOT appear as a bare math-mode cell;
        // the Holm-adjusted one (2 * 0.0001 = 0.0002) must.
        assert!(!latex.contains("$0.0001$"), "must not print the raw KS p, got:\n{latex}");
        assert!(latex.contains("$0.0002$"), "expected the Holm-adjusted KS p, got:\n{latex}");
    }

    #[test]
    fn no_stray_unescaped_underscores() {
        let report = assemble_report(evidence_structural(), evidence_central(), None);
        // This report's own fixed labels/phrases never contain an
        // underscore; regression guard against ever introducing one
        // unescaped (LaTeX would otherwise choke entering math mode).
        assert!(!report.latex_summary.contains('_'), "unescaped underscore in latex_summary");
    }

    #[test]
    fn plot_data_surfaces_raw_vectors_without_recomputation() {
        let structural = no_evidence_structural();
        let central = no_evidence_central();
        let expected_positions = structural.final_positions.clone();
        let expected_gap_centered = central.gap_centered.clone();
        let expected_gap_shifted = central.gap_shifted.clone();

        let report = assemble_report(structural, central, None);

        assert_eq!(report.plot_data.final_positions, expected_positions);
        assert_eq!(report.plot_data.gap_centered, expected_gap_centered);
        assert_eq!(report.plot_data.gap_shifted, expected_gap_shifted);
    }

    #[test]
    fn config_new_fills_documented_defaults() {
        let cfg = BiasReportConfig::new(5, 1000, 42);
        assert_eq!(cfg.dim, 5);
        assert_eq!(cfg.budget, 1000);
        assert_eq!(cfg.seed, 42);
        assert_eq!(cfg.structural_runs, crate::structural::DEFAULT_RUNS);
        assert_eq!(cfg.central_fids, DEFAULT_CENTRAL_FIDS.to_vec());
        assert_eq!(cfg.central_instances, DEFAULT_CENTRAL_INSTANCES.to_vec());
        assert_eq!(cfg.central_runs_per, DEFAULT_CENTRAL_RUNS_PER);
        // The default fid set must respect central_bias_scan's own
        // exclusion (fids 5, 6, 20, 24 are not translation-invariant).
        for fid in &cfg.central_fids {
            assert!(sezgi_problems::BbobProblem::is_translation_invariant(*fid));
        }
    }

    // ---- live end-to-end: random_search, tiny budgets, seeded ----

    #[test]
    fn live_bias_report_random_search_tiny_config_is_deterministic() {
        let spec = sezgi_components::presets::random_search(5, 50);
        let cfg = BiasReportConfig {
            dim: 2,
            budget: 50,
            seed: 20260830,
            structural_runs: 5,
            central_fids: vec![1],
            central_instances: vec![1],
            central_runs_per: 5,
        };

        let r1 = bias_report(&spec, &cfg).expect("valid tiny report config");
        let r2 = bias_report(&spec, &cfg).expect("valid tiny report config");

        assert_eq!(r1, r2, "same config must yield a bit-identical BiasReport");

        // Snapshot of the VERDICT FIELDS (not float formatting) -- ANCHORED
        // per this project's convention: these are the actual measured
        // verdicts from this test's own first successful run, not a
        // re-derivation. random_search has no directional/center-attracting
        // operator, so NoEvidence on both scans is the expected pattern
        // (matches structural.rs's/central.rs's own anchored smoke tests).
        assert_eq!(r1.structural.verdict, BiasVerdict::NoEvidence);
        assert_eq!(r1.central.verdict, BiasVerdict::NoEvidence);
        assert!(r1.signature.is_none());
        assert!(!r1.latex_summary.contains("NaN"));
        assert!(r1.latex_summary.contains(
            "Signature (Rajwar-Deep) & -- & -- & not run: the Rajwar-Deep method could not \
             be pinned from accessible sources"
        ));
    }
}
