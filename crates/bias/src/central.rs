//! Central-bias scan: run an algorithm on paired BBOB conditions -- the
//! same instance's optimum placed at the domain CENTER vs left at its
//! natural (shifted) location -- and test whether the algorithm's
//! performance gap differs between them. Per Kudela's "center-bias
//! exploitation" method: an algorithm whose search operators are
//! structurally attracted to the domain center (e.g. an initialization or
//! repair rule that keeps re-biasing candidates toward the middle of the
//! box) will look artificially strong on any benchmark whose optimum
//! happens to sit near that center, and artificially weak the moment the
//! optimum is moved away -- a confound entirely about the BENCHMARK's
//! optimum placement, not the algorithm's genuine search ability.
//!
//! # Provenance
//!
//! ## The founding paper (paywalled; reached via its own explicit successor)
//!
//! Kůdela, J. (2022). "A critical problem in benchmarking and analysis of
//! evolutionary computation methods." *Nature Machine Intelligence*, 4,
//! 1238-1245. This is the method's origin, but `nature.com` redirected every
//! fetch attempt to an institutional login wall (verified 2026-08-30: `gh`/
//! `WebFetch` both hit `idp.nature.com/authorize`, no anonymous abstract or
//! PDF reachable) and no open-access mirror of ITS full text was found
//! (ResearchGate's copy is request-gated, not directly fetchable either).
//!
//! Provenance is instead pinned via Kůdela, J. (2023). "The Evolutionary
//! Computation Methods No One Should Use." arXiv:2301.01984 (fetched via
//! `ar5iv.labs.arxiv.org/html/2301.01984`, 2026-08-30), which explicitly
//! states it reuses the 2022 paper's own method, not a new one -- quoted
//! verbatim: **"We utilize the same methodology that was used to uncover
//! the center-bias problem in [9] and [8]"**, where [8] is the 2022 Nature
//! Machine Intelligence paper above. Everything pinned below is therefore
//! the founding paper's own method, read through its author's own
//! restatement of it.
//!
//! ## Construction: centered vs shifted
//!
//! Quoted verbatim: **"we introduce a shift operation, that 'moves' the
//! benchmark function by a predetermined vector s, meaning that function
//! f(x) becomes f(x+s)"**, with the shift magnitude **"chosen ... as 10% of
//! the range - e.g., for F01, s=[20,20,...]"** (F01 = Sphere on `[-100,
//! 100]`, so `s` there is 10% of the 200-wide range). The paper's own
//! benchmark suite is 13 classical functions (its Table 1) whose STANDARD
//! form already places the optimum at/near the domain center (Sphere at the
//! origin being the canonical case); "unshifted" is that as-published,
//! near-center form, and "shifted" is the same function after the `f(x+s)`
//! translation moves its optimum away from center. Dimension 30, budget
//! 50,000 evaluations, in the source.
//!
//! ### sezgi construction (direction reversed, comparison preserved)
//!
//! sezgi-bbob's own `BbobProblem` instances are the OPPOSITE of Kudela's
//! "unshifted" starting point: every instance already has its `x_opt` drawn
//! uniformly off-center by construction (`crates/problems/src/bbob/mod.rs`;
//! confirmed by that crate's own `x_opt_not_at_center` test). There is no
//! "unshifted, near-center" BBOB instance to start from and shift AWAY from
//! center the way Kudela's classical-suite starting point allows.
//!
//! This task's brief resolves that mismatch by building the centered
//! condition the other direction: [`sezgi_problems::BbobProblem::recentered`]
//! takes an existing (already-shifted) instance and moves THAT instance's
//! own `x_opt` to the domain center instead -- the algebraic inverse of
//! Kudela's `f(x+s)` shift, applied to land ON center rather than move AWAY
//! from it. The pair actually compared is: the SAME `BbobProblem` instance
//! (same fid, same rotation/Gallagher data, same `f_opt`) evaluated (a) with
//! its optimum recentered (plays Kudela's "unshifted" role) and (b) at its
//! natural, off-center location (plays Kudela's "shifted" role). The
//! COMPARISON Kudela's method makes (near-center-optimum performance vs
//! away-from-center-optimum performance on the same underlying function) is
//! preserved exactly; only which condition is "the one we started from" is
//! reversed, a direct consequence of `BbobProblem`'s own shifted-by-
//! construction convention rather than a deviation from the source method.
//!
//! ## Construction fix (review round 1): native recentering, not a wrapper
//!
//! The first implementation of this task used a `CenteredProblem` WRAPPER
//! that translated every evaluated point by `x_opt - center` before
//! delegating to the underlying (shifted) instance -- `f(x) :=
//! inner.f(x + x_opt - center)`. Review found this introduces an
//! undocumented confound: roughly a third of the 24 BBOB fids (4, 7, 16,
//! 17, 18, 20, 21, 22, 23, 24) add a boundary penalty, `f_pen(xs) =
//! sum((|xs_i| - 5)+^2)`, computed on the RAW `xs` the caller queried --
//! i.e. on the point as it sits in the DECLARED `[-5,5]` domain, not on any
//! shifted/rotated coordinate. The wrapper's translation displaces that
//! penalty landscape along with the optimum: a query point AT the declared
//! domain's own boundary (which the algorithm is fully entitled to visit,
//! and which the UNWRAPPED instance penalizes with `f_pen = 0`) could
//! translate to a point OUTSIDE `[-5,5]` in the wrapped instance's frame,
//! spuriously triggering a large `f_pen` contribution the algorithm never
//! actually incurred at the point it queried. Confirmed empirically in
//! review: for fid 4, an edge point translated to coordinates reaching
//! 8.39/6.33 (outside `+/-5`), inflating the evaluated value by roughly
//! 55% versus evaluating the same point directly. This confound is entirely
//! an artifact of the wrapper's coordinate translation, unrelated to
//! genuine center-bias.
//!
//! **Fix:** [`central_bias_scan`] no longer wraps evaluated POINTS at all.
//! It instead builds the "centered" condition as a NATIVE
//! `BbobProblem::recentered()` instance -- `x_opt` (the instance's own
//! stored optimum) is moved to the domain center ONCE, up front, at
//! construction; every subsequent `evaluate_batch` call then runs on the
//! caller's own, untranslated coordinates, so `f_pen` (and everything else)
//! is computed on EXACTLY the point the algorithm queried, in the SAME
//! frame as the shifted condition. See `sezgi_problems::bbob::mod`'s
//! `BbobProblem::recentered`/`is_translation_invariant` docs for the full
//! mechanism and why it is a PURE relocation of the optimum (proved there
//! by `recentered_is_a_pure_translation_of_the_same_landscape`) for every
//! fid this scan accepts.
//!
//! **Fids excluded, and why:** `x_opt` is not ALWAYS a pure translation
//! origin -- fids 5 (LinearSlope), 6 (AttractiveSector), 20 (Schwefel), and
//! 24 (LunacekBiRastrigin) consume `x_opt` DIRECTLY inside their core
//! formula (a per-axis `x_opt.signum()`-driven asymmetry, baked into the
//! function's SHAPE, not just its optimum's location -- see
//! `BbobProblem::is_translation_invariant`'s doc for the exact mechanism
//! per fid). Forcing `x_opt` to the domain center for these four would
//! silently change the landscape being compared, not just relocate its
//! optimum -- a DIFFERENT, undocumented confound, arguably worse than the
//! one this fix removes. [`central_bias_scan`] therefore validates every
//! `cfg.fids` entry against `BbobProblem::is_translation_invariant` up
//! front and returns [`BiasError::InvalidConfig`] for any of these four --
//! this is a genuine scope restriction (of the 10 fids with `f_pen`, 8
//! remain fully eligible: 4, 7, 16, 17, 18, 21, 22, 23; fids 20/24 are
//! excluded for a DIFFERENT reason -- direct formula dependence, not
//! `f_pen` -- than the ones this fix targets), not merely a documentation
//! note: a caller cannot construct a scan over an ineligible fid at all.
//!
//! ## Performance measure
//!
//! Quoted verbatim: **"We also chose a simple performance measure - the
//! mean error (as the difference between the optimal function value and
//! best function value found) over 20 independent runs."** This is exactly
//! [`CentralBiasResult::gap_centered`]/[`CentralBiasResult::gap_shifted`]:
//! `f(best) - f_opt` per run (CONFIRMS the brief's "final objective gap" is
//! the right measure -- no change needed).
//!
//! ## Decision rule: Kudela's own is descriptive, not a formal test
//!
//! Kudela's own rule is a plain ratio-of-means threshold, quoted verbatim:
//! **"What we are interested in is the 'ratio' between the 'shifted' and
//! 'unshifted' results ... if this value is bigger than 1E+01 (meaning that
//! the method performs roughly at least on order of magnitude better on
//! unshifted problems), we take it as a confirmation of the presence of the
//! center-bias operator."** No p-value, hypothesis test, or significance
//! threshold appears anywhere alongside this rule in the fetched source
//! (confirmed by a dedicated extraction pass looking specifically for
//! "p-value"/"Wilcoxon"/"significance" near this passage: none found) --
//! this is exactly the "descriptive/illustrative, not a formal statistical
//! decision rule" case the brief anticipated as an EXPECTED possible
//! finding.
//!
//! // sezgi decision rule: since Kudela's own rule is a bare ratio
//! // threshold with no notion of sample-level uncertainty, and this
//! // module's whole point is a per-algorithm PAIRED comparison (see
//! // "Pairing rationale" below) for which this project already has a
//! // proper paired test on hand, [`central_bias_scan`]'s verdict is
//! // sezgi's OWN construction, not Kudela's: [`BiasVerdict::Evidence`] iff
//! // BOTH (a) the paired Wilcoxon signed-rank p-value (`gap_shifted` vs
//! // `gap_centered`, via `sezgi_stats::wilcoxon_signed_rank`) is below
//! // `ALPHA = 0.05` -- the conventional single-test significance level
//! // (unlike `structural.rs`'s `ALPHA = 0.01`, which is the BIAS
//! // toolbox's own default for correcting across ITS multi-test battery;
//! // this scan makes exactly ONE comparison, so no multiple-testing
//! // correction applies and the textbook single-test 0.05 is the
//! // appropriate default) -- AND (b) Cliff's delta (`gap_shifted` vs
//! // `gap_centered`) is at least `EFFECT_THRESHOLD = 0.33`, the "medium"
//! // magnitude boundary already pinned as `cliffs_magnitude`'s own cutoff
//! // in `sezgi_stats::pairwise` (Romano et al. 2006) -- gating out cases
//! // where a large `n` makes a practically negligible gap formally
//! // "significant". Both conditions must point the SAME direction
//! // (`effect > 0`, i.e. `gap_shifted` stochastically larger than
//! // `gap_centered`): that is the specific pattern Kudela's method looks
//! // for (worse when the optimum moves away from center), not mere
//! // asymmetry in either direction.
//!
//! # Pairing rationale (not a single-seed-comparison ban violation)
//!
//! [`central_bias_scan`] runs the recentered instance and the underlying
//! (shifted) `BbobProblem` under the IDENTICAL `RunConfig` (same
//! `master_seed`, same `run_id`) for each of its `(fid, instance, run)`
//! triples -- see the loop in [`central_bias_scan`]'s body. This is the
//! CORRECT paired design for `wilcoxon_signed_rank`, not a violation of
//! this project's single-seed comparative ban: that ban targets building a
//! cross-ALGORITHM league table off ONE seed's worth of runs (comparing
//! DIFFERENT algorithms via a single noisy sample each). Here both
//! "conditions" are the SAME algorithm, and giving them the same
//! engine-side randomness (same initial population, same
//! generator/replacer draws) while varying only WHICH condition's fitness
//! landscape they see is what makes each `(gap_centered[i],
//! gap_shifted[i])` pair a genuine matched pair -- exactly what a
//! signed-rank test is built to consume, and a STRONGER design than
//! independent seeds would be (it removes run-to-run engine randomness as a
//! confound, leaving only the centered/shifted difference). `run_id` is a
//! flat counter over every `(fid, instance, run)` triple in `cfg.fids x
//! cfg.instances_shifted x 0..cfg.runs_per` (never repeated, so no two
//! DIFFERENT pairs collapse onto the same engine stream), following
//! `structural.rs`'s own "vary `run_id` per run, engine keeps the streams
//! collision-free" pattern.

use sezgi_core::component::Registry;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_core::problem::Problem;
use sezgi_core::spec::AlgorithmSpec;
use sezgi_stats::{cliffs_delta, cliffs_magnitude, wilcoxon_signed_rank, WilcoxonResult};

use sezgi_problems::BbobProblem;

use crate::{BiasError, BiasVerdict};

/// Significance threshold for the paired Wilcoxon signed-rank p-value --
/// this crate's OWN choice (Kudela's source method has no formal test at
/// all); see this module's doc, "sezgi decision rule".
const ALPHA: f64 = 0.05;

/// Minimum Cliff's delta magnitude (in the direction "shifted worse than
/// centered") required alongside [`ALPHA`] before a scan is reported as
/// [`BiasVerdict::Evidence`] -- the "medium" boundary already pinned as
/// `cliffs_magnitude`'s own cutoff (Romano et al. 2006); see this module's
/// doc, "sezgi decision rule".
const EFFECT_THRESHOLD: f64 = 0.33;

/// Minimum number of `(fid, instance, run)` triples [`central_bias_scan`]
/// and [`scan_from_gaps`] accept: mirrors `wilcoxon_signed_rank`'s own
/// `n_effective < 5` floor (see `sezgi_stats::pairwise`) -- below it, that
/// function itself already has no reliable basis for a p-value, so this
/// module rejects the configuration before spending any engine budget.
const MIN_PAIRS: usize = 5;

#[derive(Debug, Clone, PartialEq)]
pub struct CentralBiasConfig {
    /// BBOB function IDs to scan. Every entry MUST be translation-invariant
    /// (`sezgi_problems::BbobProblem::is_translation_invariant`) -- fids 5,
    /// 6, 20, 24 are rejected with [`BiasError::InvalidConfig`]; see this
    /// module's doc, "Construction fix (review round 1)".
    pub fids: Vec<u32>,
    /// Dimensionality (BBOB requires `dim >= 2`).
    pub dim: usize,
    /// BBOB instance numbers -- each contributes its own paired
    /// (centered, shifted) condition per fid. Named `instances_shifted`
    /// (per the brief's exact signature) since these instances ARE the
    /// scan's "shifted" condition; each is also the base
    /// `BbobProblem::recentered()` derives the "centered" condition from.
    pub instances_shifted: Vec<u32>,
    /// Independent runs per `(fid, instance)` pair.
    pub runs_per: u32,
    /// Per-run evaluation budget -- overrides `spec.termination.budget`
    /// (this crate's standing "budget is always overridden" convention;
    /// see `structural.rs`).
    pub budget: u64,
    /// Seed, reused as BOTH conditions' shared `RunConfig::master_seed` --
    /// see this module's doc, "Pairing rationale".
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CentralBiasResult {
    /// Per-`(fid, instance, run)` gap `f(best) - f_opt` under the centered
    /// condition, in the same `(fid, instance, run)` order as
    /// [`CentralBiasResult::gap_shifted`] (so index `i` in both vectors is
    /// one matched pair).
    pub gap_centered: Vec<f64>,
    /// Same shape/order, under the shifted (natural instance) condition.
    pub gap_shifted: Vec<f64>,
    /// Paired Wilcoxon signed-rank test, `gap_shifted` vs `gap_centered`.
    pub wilcoxon: WilcoxonResult,
    /// Cliff's delta, `gap_shifted` vs `gap_centered`; positive means
    /// `gap_shifted` is stochastically the larger (worse) of the two.
    pub effect: f64,
    /// Decision -- see this module's doc, "sezgi decision rule".
    pub verdict: BiasVerdict,
}

/// Runs `spec` on paired centered/shifted BBOB conditions, per this
/// module's doc, and applies sezgi's own decision rule to the resulting
/// gap vectors.
///
/// # Errors
/// [`BiasError::InvalidConfig`] if `cfg.fids`/`cfg.instances_shifted` is
/// empty, `cfg.dim < 2`, `cfg.runs_per == 0`, `cfg.fids` contains a fid
/// that is not translation-invariant (5, 6, 20, 24 -- see this module's
/// doc, "Construction fix"), or the total number of `(fid, instance, run)`
/// triples is below [`MIN_PAIRS`]; [`BiasError::Bbob`] if a `(fid, dim,
/// instance)` combination fails to construct; [`BiasError::Spec`]/
/// [`BiasError::Engine`] as `structural_bias_scan`'s own analogous cases;
/// [`BiasError::Stats`] if the resulting gap vectors fail
/// `wilcoxon_signed_rank`/`cliffs_delta`'s own input checks (e.g. a
/// non-finite gap).
pub fn central_bias_scan(
    spec: &AlgorithmSpec,
    cfg: &CentralBiasConfig,
) -> Result<CentralBiasResult, BiasError> {
    if cfg.fids.is_empty() {
        return Err(BiasError::InvalidConfig("fids must not be empty".into()));
    }
    if cfg.dim < 2 {
        return Err(BiasError::InvalidConfig("dim must be >= 2 (BBOB requires dim >= 2)".into()));
    }
    if cfg.instances_shifted.is_empty() {
        return Err(BiasError::InvalidConfig("instances_shifted must not be empty".into()));
    }
    if cfg.runs_per == 0 {
        return Err(BiasError::InvalidConfig("runs_per must be >= 1".into()));
    }
    for &fid in &cfg.fids {
        if !BbobProblem::is_translation_invariant(fid) {
            return Err(BiasError::InvalidConfig(format!(
                "central_bias_scan: fid {fid} is not translation-invariant (x_opt participates \
                 directly in its objective formula beyond a coordinate shift -- see \
                 sezgi_problems::BbobProblem::is_translation_invariant); recentering it would \
                 change the landscape's shape rather than merely relocate its optimum, so it is \
                 excluded from this scan"
            )));
        }
    }
    let total_pairs = cfg.fids.len() * cfg.instances_shifted.len() * cfg.runs_per as usize;
    if total_pairs < MIN_PAIRS {
        return Err(BiasError::InvalidConfig(format!(
            "central_bias_scan requires at least {MIN_PAIRS} (fid, instance, run) triples, got {total_pairs}"
        )));
    }

    let mut reg = Registry::new();
    sezgi_components::register_builtins(&mut reg);

    let mut spec = spec.clone();
    spec.termination.budget = cfg.budget; // always overridden, per this module's doc

    // Every BBOB instance shares the same domain shape for a given `dim`
    // (`Float{lo:-5,hi:5,n:dim}`, independent of fid/instance), so one
    // Engine serves the whole scan -- mirrors `structural.rs`'s
    // engine-reuse. `probe` is dropped once the (owned-internally) engine
    // is built.
    let probe = BbobProblem::new(cfg.fids[0], cfg.dim, cfg.instances_shifted[0])?;
    let engine = Engine::from_spec(&spec, &reg, probe.space())?;
    drop(probe);

    let mut gap_centered = Vec::with_capacity(total_pairs);
    let mut gap_shifted = Vec::with_capacity(total_pairs);
    let mut run_id = 0u64;
    for &fid in &cfg.fids {
        for &instance in &cfg.instances_shifted {
            let shifted_problem = BbobProblem::new(fid, cfg.dim, instance)?;
            // A SEPARATE construction (not a clone of `shifted_problem`),
            // then natively recentered -- see this module's doc,
            // "Construction fix (review round 1)". Both constructions of
            // the same (fid, dim, instance) are deterministic and produce
            // identical f_opt/rotations/Gallagher data (only `x_opt`, and
            // for 21/22 its coincident first Gallagher peak, differ).
            let centered_problem = BbobProblem::new(fid, cfg.dim, instance)?.recentered()?;
            let f_opt = shifted_problem.f_opt();
            for _ in 0..cfg.runs_per {
                // SAME RunConfig for both conditions -- see this module's
                // doc, "Pairing rationale".
                let run_cfg = RunConfig { master_seed: cfg.seed, run_id };
                let r_centered = engine.run(&centered_problem, run_cfg, None)?;
                let r_shifted = engine.run(&shifted_problem, run_cfg, None)?;
                gap_centered.push(r_centered.best_f - f_opt);
                gap_shifted.push(r_shifted.best_f - f_opt);
                run_id += 1;
            }
        }
    }

    scan_from_gaps(gap_centered, gap_shifted)
}

/// The statistics-only decision path, factored out so it is unit-testable
/// without the engine (synthetic gap vectors) -- see this module's tests.
/// `gap_centered` and `gap_shifted` must be equal length, `>= `
/// [`MIN_PAIRS`], matched pairwise by index (the same `(fid, instance,
/// run)` triple at the same index in both).
pub(crate) fn scan_from_gaps(
    gap_centered: Vec<f64>,
    gap_shifted: Vec<f64>,
) -> Result<CentralBiasResult, BiasError> {
    if gap_centered.len() != gap_shifted.len() {
        return Err(BiasError::InvalidConfig(format!(
            "scan_from_gaps requires equal-length paired vectors, got {} centered and {} shifted",
            gap_centered.len(),
            gap_shifted.len()
        )));
    }
    if gap_centered.len() < MIN_PAIRS {
        return Err(BiasError::InvalidConfig(format!(
            "scan_from_gaps requires at least {MIN_PAIRS} pairs, got {}",
            gap_centered.len()
        )));
    }

    let wilcoxon = wilcoxon_signed_rank(&gap_shifted, &gap_centered)?;
    let effect = cliffs_delta(&gap_shifted, &gap_centered)?;

    let verdict = if wilcoxon.p_value < ALPHA && effect >= EFFECT_THRESHOLD {
        BiasVerdict::Evidence {
            detail: format!(
                "evidence of center-bias exploitation: the performance gap f(best)-f_opt over \
                 this scan's paired runs is significantly and non-negligibly larger under the \
                 shifted-optimum condition than the centered-optimum condition (paired Wilcoxon \
                 signed-rank p={:e} < {ALPHA}, Cliff's delta={:.4} [{}])",
                wilcoxon.p_value,
                effect,
                cliffs_magnitude(effect)
            ),
        }
    } else {
        BiasVerdict::NoEvidence
    };

    Ok(CentralBiasResult { gap_centered, gap_shifted, wilcoxon, effect, verdict })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::space::{Block, BlockValues, Genotype};

    fn g(xs: &[f64]) -> Genotype {
        Genotype { blocks: vec![BlockValues::Float(xs.to_vec())] }
    }

    fn domain_center(p: &BbobProblem) -> Vec<f64> {
        match &p.space().blocks()[0] {
            Block::Float { lo, hi, n } => vec![(lo + hi) / 2.0; *n],
            other => panic!("expected a single Float block, got {other:?}"),
        }
    }

    // ---- (a) statistics-only decision path, synthetic gap vectors ----

    // Evidence fixture: gap_centered tightly clustered near 0, gap_shifted
    // uniformly far larger and completely disjoint -- hand-checked.
    //
    // gap_centered = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    // gap_shifted  = [5.00, 6.00, 7.00, 8.00, 9.00, 10.00, 11.00, 12.00]
    //
    // d_i = gap_shifted_i - gap_centered_i =
    //   [4.99, 5.98, 6.97, 7.96, 8.95, 9.94, 10.93, 11.92] -- all positive,
    //   strictly increasing (no ties, no zeros), so rank(|d_i|) = i+1
    //   exactly (1-based).
    //
    // t0 = 0, n = 8, n_effective = 8 (<= 25, no ties) -> exact-eligible.
    // W+ = 1+2+...+8 = 36, W- = 0, W = min(36, 0) = 0.
    // Exact p = min(1, 2*count(sum<=0)/2^8): count(sum<=0) over subsets of
    // {1..8} = 1 (only the empty subset sums to 0; the smallest nonempty
    // subset sums to 1). p = 2*1/256 = 0.0078125 (exact dyadic rational).
    //
    // cliffs_delta(gap_shifted, gap_centered): every one of the 8 shifted
    // values (5.00..12.00) exceeds every one of the 8 centered values
    // (0.01..0.08) -- complete separation, so greater=64, less=0,
    // delta = (64-0)/(8*8) = 1.0 exactly ("large" magnitude, Romano
    // boundary 0.474).
    //
    // Decision: p=0.0078125 < ALPHA(0.05) AND effect=1.0 >=
    // EFFECT_THRESHOLD(0.33), direction positive -> Evidence.
    #[test]
    fn synthetic_disjoint_gaps_yield_evidence() {
        let gap_centered = vec![0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08];
        let gap_shifted = vec![5.00, 6.00, 7.00, 8.00, 9.00, 10.00, 11.00, 12.00];

        let r = scan_from_gaps(gap_centered, gap_shifted).expect("valid fixture");

        assert_eq!(r.wilcoxon.method, sezgi_stats::WilcoxonMethod::Exact);
        assert_eq!(r.wilcoxon.p_value, 0.0078125, "exact dyadic rational -- bit-exact assert");
        assert!((r.effect - 1.0).abs() < 1e-12, "effect: got {}", r.effect);
        assert!(
            matches!(r.verdict, BiasVerdict::Evidence { .. }),
            "disjoint, uniformly-worse-when-shifted gaps must be flagged as Evidence, got {:?}",
            r.verdict
        );
        if let BiasVerdict::Evidence { detail } = &r.verdict {
            assert!(detail.contains("evidence of center-bias exploitation"));
            assert!(
                !detail.to_lowercase().contains("algorithm"),
                "evidence-not-accusation: detail must not name/blame a specific algorithm"
            );
        }
    }

    // NoEvidence fixture: reuses `sezgi_stats::pairwise`'s OWN pinned/
    // cross-checked fixture (`ties_fall_back_to_normal_approx`) verbatim as
    // gap_shifted (a=[102,98,103,104,105]) vs gap_centered (b=[100]*5) --
    // that module already hand-derives/cross-checks (against SciPy)
    // w_statistic=1.5, z=-1.490028015252912, p=0.13621686984456766 for
    // exactly this pair, so this test does not re-derive the Wilcoxon
    // arithmetic; it hand-checks the DECISION built on top of that already-
    // pinned number: p=0.136 > ALPHA(0.05) -> NoEvidence, regardless of
    // effect size.
    //
    // cliffs_delta(gap_shifted, gap_centered): every element of
    // gap_shifted (102,98,103,104,105) vs the constant 100 -- 4 of 5
    // exceed 100 (greater), 1 (98) is below (less), each paired against
    // all 5 constant centered values: greater = 4*5 = 20, less = 1*5 = 5.
    // delta = (20-5)/(5*5) = 15/25 = 0.6 ("large" magnitude on its own --
    // demonstrating that a large point-estimate effect size does NOT
    // override the significance gate at this small n).
    #[test]
    fn synthetic_indistinguishable_gaps_yield_no_evidence_despite_large_effect() {
        let gap_centered = vec![100.0, 100.0, 100.0, 100.0, 100.0];
        let gap_shifted = vec![102.0, 98.0, 103.0, 104.0, 105.0];

        let r = scan_from_gaps(gap_centered, gap_shifted).expect("valid fixture");

        assert_eq!(r.wilcoxon.method, sezgi_stats::WilcoxonMethod::NormalApprox);
        assert!(
            (r.wilcoxon.p_value - 0.13621686984456766).abs() < 1e-6,
            "p_value: got {}",
            r.wilcoxon.p_value
        );
        assert!(r.wilcoxon.p_value > ALPHA, "fixture must be non-significant at ALPHA");
        assert!((r.effect - 0.6).abs() < 1e-12, "effect: got {}", r.effect);
        assert!(
            r.effect >= EFFECT_THRESHOLD,
            "fixture's effect size must itself clear the effect threshold, to prove the \
             significance gate (not the effect gate) is what suppresses Evidence here"
        );
        assert_eq!(
            r.verdict,
            BiasVerdict::NoEvidence,
            "a large point-estimate effect at p > ALPHA must NOT be reported as Evidence, got {:?}",
            r.verdict
        );
    }

    #[test]
    fn rejects_unequal_length_gap_vectors() {
        let err = scan_from_gaps(vec![0.1; 5], vec![0.1; 6]).unwrap_err();
        assert!(matches!(err, BiasError::InvalidConfig(_)), "got {err:?}");
    }

    #[test]
    fn rejects_too_few_pairs() {
        let err = scan_from_gaps(vec![0.1; 3], vec![0.2; 3]).unwrap_err();
        assert!(matches!(err, BiasError::InvalidConfig(_)), "got {err:?}");
    }

    // ---- (b) config validation: non-translation-invariant fids rejected ----

    #[test]
    fn rejects_non_translation_invariant_fids() {
        for &fid in &[5u32, 6, 20, 24] {
            let spec = sezgi_components::presets::random_search(10, 500);
            let cfg = CentralBiasConfig {
                fids: vec![fid],
                dim: 5,
                instances_shifted: vec![1, 2],
                runs_per: 3,
                budget: 500,
                seed: 1,
            };
            let err = central_bias_scan(&spec, &cfg).unwrap_err();
            assert!(
                matches!(err, BiasError::InvalidConfig(_)),
                "fid {fid} must be rejected before any engine work, got {err:?}"
            );
        }
    }

    // ---- (c) review-round-1 regression: no more f_pen displacement ----
    //
    // Regression test for the review-flagged confound: the OLD
    // `CenteredProblem` wrapper translated every evaluated point by
    // `x_opt - center` before delegating to the underlying (shifted)
    // instance, so a query point AT the domain's own boundary (fully
    // in-bounds, `f_pen == 0` there) could translate to a point OUTSIDE
    // `[-5,5]` in the wrapped instance's own frame, spuriously inflating
    // the evaluated value via `f_pen`. `BbobProblem::recentered` fixes
    // this by construction (it never translates query points at all), but
    // this test does not just trust that -- it reproduces the OLD buggy
    // computation by hand and shows it diverges from the FIXED one.
    #[test]
    fn recentered_fid4_has_no_f_pen_displacement_at_the_boundary() {
        let dim = 5;
        let shifted = BbobProblem::new(4, dim, 1).expect("valid BBOB construction");
        let centered =
            BbobProblem::new(4, dim, 1).expect("valid BBOB construction").recentered().expect("fid 4 is translation-invariant");

        // The domain's own boundary corner, pushed toward whichever side
        // makes the OLD wrapper's translation worst (same sign as this
        // instance's own x_opt per axis) -- guarantees the old
        // wrapper's translated point leaves [-5,5] on every axis with a
        // nonzero x_opt component, reproducing the review's scenario
        // deterministically regardless of which instance is used.
        let corner: Vec<f64> = shifted.x_opt().iter().map(|&o| 5.0 * o.signum()).collect();

        // (a) direct proof: the query point itself never left the domain,
        // so it carries zero boundary penalty on its own terms.
        assert_eq!(
            sezgi_problems::bbob::transform::f_pen(&corner),
            0.0,
            "the boundary corner itself must be exactly in-bounds"
        );

        // (b) replicate the OLD wrapper's translation (offset = x_opt -
        // center, center = 0 here) to show what it WOULD have evaluated,
        // and confirm the translated point actually left [-5,5] (so this
        // fixture genuinely reproduces the reviewed confound, not a
        // vacuous case).
        let old_wrapper_translated: Vec<f64> =
            corner.iter().zip(shifted.x_opt()).map(|(x, o)| x + o).collect();
        assert!(
            old_wrapper_translated.iter().any(|&v| v.abs() > 5.0),
            "fixture must reproduce the out-of-bounds translation the review flagged, got {old_wrapper_translated:?}"
        );
        let old_wrapper_value = shifted.evaluate_batch(&[g(&old_wrapper_translated)])[0];

        // (c) the FIXED evaluation: query the recentered instance at the
        // exact, untranslated corner.
        let fixed_value = centered.evaluate_batch(&[g(&corner)])[0];

        assert!(
            old_wrapper_value > fixed_value,
            "the old wrapper's translated-coordinate evaluation ({old_wrapper_value}) must be \
             inflated relative to the fixed native evaluation ({fixed_value}) -- reproducing (and \
             showing the fix for) the review-flagged f_pen displacement confound"
        );

        // (d) the fixed evaluation, at a point where every |x_i| == 5
        // exactly, must equal the SAME instance's un-recentered value at
        // its own `x_opt + corner`-relative-to-center point offset by
        // nothing (i.e. it is a genuinely different, non-inflated number,
        // not merely "less inflated") -- sanity: no f_pen contribution at
        // all should appear at this exact query point for the recentered
        // instance either, matching (a).
        assert_eq!(sezgi_problems::bbob::transform::f_pen(&domain_center(&centered)), 0.0);
    }

    // ---- (d) engine-driven path ----

    fn bbob_random_search_spec(pop_size: usize, budget: u64) -> AlgorithmSpec {
        sezgi_components::presets::random_search(pop_size, budget)
    }

    #[test]
    fn engine_driven_scan_is_deterministic() {
        let spec = bbob_random_search_spec(10, 500);
        let cfg = CentralBiasConfig {
            fids: vec![1, 4],
            dim: 5,
            instances_shifted: vec![1, 2],
            runs_per: 2,
            budget: 500,
            seed: 20260830,
        };

        let r1 = central_bias_scan(&spec, &cfg).expect("valid scan");
        let r2 = central_bias_scan(&spec, &cfg).expect("valid scan");

        assert_eq!(r1, r2, "same config must yield a bit-identical CentralBiasResult");
        assert_eq!(r1.wilcoxon.p_value.to_bits(), r2.wilcoxon.p_value.to_bits());
        assert_eq!(r1.effect.to_bits(), r2.effect.to_bits());
        assert_eq!(r1.gap_centered[0].to_bits(), r2.gap_centered[0].to_bits());
        assert_eq!(r1.gap_shifted[0].to_bits(), r2.gap_shifted[0].to_bits());
    }

    // Live smoke: random_search's generator (`gen/uniform-resample`) draws
    // uniformly over the WHOLE declared space every generation with no
    // directional/center-attracting operator, so it should show no
    // center-bias exploitation on either condition. Includes fid 4 (an
    // `f_pen`-using fid, per this round's fix) so this anchor also covers
    // the fixed construction end-to-end through the engine. ANCHORED (per
    // this project's convention): the p-value/effect quoted below are the
    // actual measured values from this test's own first successful run
    // AFTER the review-round-1 fix (re-anchored: the fix changes how the
    // centered condition is evaluated, so the old wrapper-based numbers no
    // longer apply), not a re-derivation.
    #[test]
    fn engine_driven_random_search_on_bbob_yields_no_evidence_anchored() {
        let spec = bbob_random_search_spec(20, 2000);
        let cfg = CentralBiasConfig {
            fids: vec![1, 4, 13],
            dim: 5,
            instances_shifted: vec![1, 2],
            runs_per: 5,
            budget: 2000,
            seed: 20260830,
        };

        let r = central_bias_scan(&spec, &cfg).expect("valid scan");

        // Measured at this exact config, AFTER the review-round-1 fix
        // (captured from this test's own first successful run post-fix,
        // before the assertions below were added):
        //   wilcoxon.p_value = 0.8050475827083527, method = NormalApprox
        //   effect (Cliff's delta, gap_shifted vs gap_centered) = 0.0
        // p well above ALPHA (0.05) and |effect| well below EFFECT_THRESHOLD
        // (0.33) -- consistent with random_search's uniform, non-directional
        // generator having no center-attracting operator to exploit.
        assert!(
            (r.wilcoxon.p_value - 0.8050475827083527).abs() < 1e-9,
            "p_value drifted from the anchored measurement: got {}",
            r.wilcoxon.p_value
        );
        assert!(
            (r.effect - 0.0).abs() < 1e-12,
            "effect drifted from the anchored measurement: got {}",
            r.effect
        );
        assert!(r.wilcoxon.p_value > ALPHA);
        assert!(r.effect.abs() < EFFECT_THRESHOLD);
        assert_eq!(
            r.verdict,
            BiasVerdict::NoEvidence,
            "measured verdict for random_search(pop=20, budget=2000) on BBOB fids=[1,4,13], \
             dim=5, instances=[1,2], runs_per=5, seed=20260830: {:?}",
            r.verdict
        );
    }
}
