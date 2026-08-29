//! ECDF (empirical cumulative distribution function) / anytime performance
//! curves over [`IohScenario`]/[`IohRun`] logs (M2d-2 Task 3).
//!
//! This module implements the COCO/IOH-profiler "anytime" analysis: for a
//! fixed set of precision `target`s, how large a fraction of (run, target)
//! pairs has reached its target by evaluation count `e`, as `e` grows. This
//! is the standard way to summarize a whole distribution of runs — across
//! many problems, targets and seeds — into a single monotone curve, rather
//! than reporting a single final value per run.
//!
//! ## PINNED definitions (semver events: changing any of these after this
//! module merges is a breaking change to every consumer of the emitted
//! curves — CSV/plots/paper packages included — and must be called out as
//! such, not adjusted silently)
//!
//! - [`default_targets`]: the 51-value COCO-convention target set
//!   `10^(2 - 0.2*k)` for `k = 0..=50` (`1e2` down to `1e-8`).
//! - [`hit_time`]: the first evaluation index at which
//!   `best_so_far - f_opt <= target`, found by scanning a run's rows IN
//!   FILE ORDER (ascending eval) and returning the first row that
//!   satisfies the inequality; `None` if no row ever does.
//! - The ECDF denominator: `(total runs) * (total targets)` across the
//!   scenarios given to [`ecdf`]/[`ecdf_per_algo`] — a (run, target) pair
//!   that never hits its target still counts in the denominator (it simply
//!   never contributes a step), so the curve need not reach `1.0`.

use crate::experiment::ExperimentError;
use crate::ioh_read::{IohRun, IohScenario, canonical_anytime};

/// COCO-convention default target precisions: `10^(2 - 0.2*k)` for
/// `k = 0..=50` — 51 values, descending from `1e2` to `1e-8`.
///
/// PINNED (see the module doc): this is the target set every ECDF curve
/// produced by this module is computed against unless a caller supplies
/// its own `targets`.
pub fn default_targets() -> Vec<f64> {
    (0..=50).map(|k| 10f64.powf(2.0 - 0.2 * k as f64)).collect()
}

/// First evaluation index at which `best_so_far - f_opt <= target`, `None`
/// if no row of `run` ever reaches it.
///
/// Scans `run.rows` in file order (ascending eval, per
/// [`crate::ioh_read`]'s format contract) and returns the FIRST row
/// meeting the target. `f_opt` is a plain `f64`: [`IohRun::f_opt`] is an
/// `Option` (absent on legacy logs), so the caller resolves it — this
/// function has no failure mode of its own.
///
/// ## Why "first row in file order" is correct even though the final row
/// can be a non-improvement duplicate
///
/// [`IohRun::rows`] holds the run's improvement rows plus an
/// always-written final row, which may or may not itself be an
/// improvement over the last improvement row (see the doc on
/// [`crate::ioh_read::IohRun::rows`]). Each improvement row's `raw_y` is,
/// by construction (`IohRunObserver::on_eval` only records a row when
/// `best_so_far` strictly decreases), the running best-so-far value, so
/// `raw_y` is non-increasing across ALL of `run.rows` in file order —
/// including the final row: if it is itself an improvement it continues
/// the strictly-decreasing sequence, and if it is not, it repeats the
/// last improvement row's value rather than regressing (best-so-far never
/// gets worse). Consequently, once some row's `raw_y - f_opt <= target`
/// holds, every later row's also holds, so there is exactly one
/// "first row that hits" (or none), and a plain forward scan finds it —
/// there is no scenario where a later, non-improvement final row could
/// hit a target that no earlier row already hit.
pub fn hit_time(run: &IohRun, f_opt: f64, target: f64) -> Option<u64> {
    run.rows
        .iter()
        .find(|&&(_, y)| y - f_opt <= target)
        .map(|&(eval, _)| eval)
}

/// A step-function ECDF/anytime curve: `evals` ascending, `proportion[i]`
/// is the fraction of (run, target) pairs hit by evaluation `evals[i]`
/// (monotonically nondecreasing, values in `[0, 1]`).
#[derive(Debug, Clone, PartialEq)]
pub struct EcdfCurve {
    pub evals: Vec<u64>,
    pub proportion: Vec<f64>,
}

fn missing_f_opt_err(run: &IohRun, sc: &IohScenario) -> ExperimentError {
    ExperimentError::IohRead(format!(
        "run instance {} (algo `{}`, fid {}, dim {}) has no f_opt \
         (legacy log?) — ECDF requires f_opt on every run",
        run.instance, sc.algo, sc.fid, sc.dim
    ))
}

/// ECDF over ALL (run, target) pairs across `scenarios`: every run in
/// every scenario, crossed with every target in `targets`.
///
/// PINNED (see the module doc): denominator = `(total runs) *
/// targets.len()`; a pair that never hits its target still counts in the
/// denominator. Errors (mentioning `"f_opt"`) if any run lacks `f_opt`.
///
/// Empty `scenarios` or empty `targets` yield an empty curve
/// (`evals: []`, `proportion: []`) by construction, not an error: with
/// `total_pairs == 0` the hit-collection loop never runs, so there is no
/// division to perform and nothing to report.
///
/// [`canonical_anytime`] is applied to (a private clone of) `scenarios`
/// FIRST (see `crate::ioh_read`'s module doc, "Two canonicalization
/// policies"), so a multi-budget archive counts each `(instance, seed)`
/// pair exactly once in the denominator — the largest-budget run's
/// trajectory, not one entry per budget it was logged at — since same-seed
/// runs at different budgets are correlated pseudo-replicates of ONE
/// sample here, not independent additional ones.
pub fn ecdf(scenarios: &[IohScenario], targets: &[f64]) -> Result<EcdfCurve, ExperimentError> {
    let mut scenarios = scenarios.to_vec();
    canonical_anytime(&mut scenarios)?;

    let mut total_pairs: u64 = 0;
    let mut hits: Vec<u64> = Vec::new();

    for sc in &scenarios {
        for run in &sc.runs {
            let f_opt = run.f_opt.ok_or_else(|| missing_f_opt_err(run, sc))?;
            for &target in targets {
                total_pairs += 1;
                if let Some(e) = hit_time(run, f_opt, target) {
                    hits.push(e);
                }
            }
        }
    }

    hits.sort_unstable();

    let mut evals = Vec::new();
    let mut proportion = Vec::new();
    let mut count: u64 = 0;
    let mut i = 0;
    while i < hits.len() {
        let e = hits[i];
        let mut j = i;
        while j < hits.len() && hits[j] == e {
            j += 1;
        }
        count += (j - i) as u64;
        evals.push(e);
        proportion.push(count as f64 / total_pairs as f64);
        i = j;
    }

    Ok(EcdfCurve { evals, proportion })
}

/// Per-algorithm ECDF curves: `scenarios` grouped by
/// [`IohScenario::algo`], each group's curve computed by [`ecdf`] over
/// just that group, groups returned in FIRST-APPEARANCE order in
/// `scenarios` (stable, pinned — not sorted; matches the convention used
/// elsewhere in this crate, e.g. `reporting::results_matrix`).
pub fn ecdf_per_algo(
    scenarios: &[IohScenario],
    targets: &[f64],
) -> Result<Vec<(String, EcdfCurve)>, ExperimentError> {
    let mut scenarios = scenarios.to_vec();
    canonical_anytime(&mut scenarios)?;

    let mut order: Vec<String> = Vec::new();
    for sc in &scenarios {
        if !order.contains(&sc.algo) {
            order.push(sc.algo.clone());
        }
    }

    order
        .into_iter()
        .map(|algo| {
            let group: Vec<IohScenario> =
                scenarios.iter().filter(|sc| sc.algo == algo).cloned().collect();
            // Already deduped above; `ecdf`'s own dedupe pass is a no-op here.
            let curve = ecdf(&group, targets)?;
            Ok((algo, curve))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::{AlgoEntry, AlgoSource, ExperimentSpec, ProblemEntry, run_experiment_logged};
    use crate::ioh_read::read_ioh_root;

    fn scenario_of(runs: Vec<IohRun>) -> IohScenario {
        IohScenario {
            algo: "algo".into(),
            suite: "bbob".into(),
            fid: 1,
            fname: "Sphere".into(),
            dim: 5,
            runs,
        }
    }

    #[test]
    fn default_targets_are_pinned() {
        let t = default_targets();
        assert_eq!(t.len(), 51);
        assert_eq!(t[0], 1e2);
        assert!((t[50] - 1e-8).abs() < 1e-20);
        assert!((t[25] - 10f64.powf(2.0 - 0.2 * 25.0)).abs() < 1e-16);
    }

    #[test]
    fn hit_time_requires_f_opt() {
        let run = IohRun { instance: 1, seed: None, f_opt: None, budget: None, rows: vec![(1, 5.0)], evals: 1 };
        let sc = scenario_of(vec![run]);

        let err = ecdf(&[sc], &[1.0]).unwrap_err();
        assert!(err.to_string().contains("f_opt"), "error must mention f_opt: {err}");
    }

    #[test]
    fn ecdf_hand_fixture() {
        // 2 runs, f_opt = 0, targets = [10.0, 1.0, 0.1]  (6 pairs total)
        // run A rows: (1, 50.0), (3, 5.0), (10, 0.05)
        //   hits: 10.0 at e=3; 1.0 at e=10 (5.0>1.0, 0.05<=1.0); 0.1 at e=10
        // run B rows: (2, 8.0), (7, 0.5)
        //   hits: 10.0 at e=2; 1.0 at e=7; 0.1 never
        // pooled hit times: [2,3,7,10,10]; ECDF steps:
        //   e=2 -> 1/6, e=3 -> 2/6, e=7 -> 3/6, e=10 -> 5/6
        let run_a = IohRun {
            instance: 1,
            seed: Some(11),
            f_opt: Some(0.0),
            budget: None,
            rows: vec![(1, 50.0), (3, 5.0), (10, 0.05)],
            evals: 10,
        };
        let run_b = IohRun {
            instance: 2,
            seed: Some(22),
            f_opt: Some(0.0),
            budget: None,
            rows: vec![(2, 8.0), (7, 0.5)],
            evals: 7,
        };
        let sc = scenario_of(vec![run_a, run_b]);
        let targets = vec![10.0, 1.0, 0.1];

        let curve = ecdf(&[sc], &targets).unwrap();

        assert_eq!(curve.evals, vec![2, 3, 7, 10]);
        // Not dyadic (denominator 6): assert against the same-expression
        // computation, not a literal decimal approximation.
        let expected_counts: [u64; 4] = [1, 2, 3, 5];
        let expected_proportion: Vec<f64> =
            expected_counts.iter().map(|&c| c as f64 / 6.0).collect();
        assert_eq!(curve.proportion, expected_proportion);
    }

    #[test]
    fn end_to_end_smoke_ecdf_per_algo() {
        let spec = ExperimentSpec {
            name: "ecdf-smoke".into(),
            seeds: vec![11, 22],
            budgets: vec![500],
            algorithms: vec![
                AlgoEntry {
                    name: "de".into(),
                    source: AlgoSource::Preset { kind: "de_rand_1".into(), pop_size: Some(6) },
                },
                AlgoEntry {
                    name: "pso".into(),
                    source: AlgoSource::Preset { kind: "pso".into(), pop_size: Some(6) },
                },
            ],
            problems: vec![ProblemEntry { suite: "bbob".into(), fid: 1, dim: 5, instances: vec![1] }],
        };
        let tmp = tempfile::tempdir().unwrap();
        run_experiment_logged(&spec, tmp.path(), false, None).unwrap();

        let scenarios = read_ioh_root(tmp.path()).unwrap();
        let curves = ecdf_per_algo(&scenarios, &default_targets()).unwrap();

        assert_eq!(curves.len(), 2);

        // First-appearance order in `scenarios`, not a cross-algorithm
        // performance comparison (single-seed comparative ban).
        let mut expected_order: Vec<String> = Vec::new();
        for sc in &scenarios {
            if !expected_order.contains(&sc.algo) {
                expected_order.push(sc.algo.clone());
            }
        }
        let got_order: Vec<String> = curves.iter().map(|(algo, _)| algo.clone()).collect();
        assert_eq!(got_order, expected_order);

        for (_, curve) in &curves {
            assert!(
                curve.proportion.iter().all(|&p| (0.0..=1.0).contains(&p)),
                "proportions must be in [0, 1]: {:?}",
                curve.proportion
            );
            assert!(
                curve.evals.windows(2).all(|w| w[0] <= w[1]),
                "evals must be ascending: {:?}",
                curve.evals
            );
            assert!(
                curve.proportion.windows(2).all(|w| w[0] <= w[1]),
                "proportion must be nondecreasing: {:?}",
                curve.proportion
            );
        }
    }

    /// Denominator check (M2d-3 Task 1, step 1c): a multi-budget archive
    /// with duplicate `(instance, seed)` runs (one logged at budget 200,
    /// one at budget 400, same seed/instance) must be counted ONCE in the
    /// ECDF denominator via `ecdf`'s internal `canonical_anytime` call —
    /// not once per budget it happens to have been logged at.
    #[test]
    fn ecdf_counts_each_instance_seed_once_on_a_multi_budget_archive() {
        let dup_short = IohRun {
            instance: 1, seed: Some(1), f_opt: Some(0.0), budget: Some(200),
            rows: vec![(1, 50.0)], evals: 200,
        };
        let dup_long = IohRun {
            instance: 1, seed: Some(1), f_opt: Some(0.0), budget: Some(400),
            rows: vec![(1, 50.0), (3, 5.0)], evals: 400,
        };
        let solo = IohRun {
            instance: 2, seed: Some(2), f_opt: Some(0.0), budget: None,
            rows: vec![(1, 8.0)], evals: 1,
        };
        let sc = scenario_of(vec![dup_short, dup_long, solo]);
        let targets = vec![10.0];

        // Denominator = (deduped run count) * targets.len() = 2 * 1 = 2, not
        // 3 * 1 = 3 (which is what an un-deduped denominator would give).
        // Both runs hit target 10.0 immediately (dup_long's first row is
        // still 50.0 > 10.0, but its second row 5.0 <= 10.0 at eval 3; solo's
        // 8.0 <= 10.0 at eval 1), so the final proportion must reach exactly
        // 1.0 with a deduped denominator of 2, never 2/3.
        let curve = ecdf(&[sc], &targets).unwrap();
        assert_eq!(
            *curve.proportion.last().unwrap(), 1.0,
            "final proportion must be 1.0 under a deduped denominator of 2 (not 2/3 under 3): {:?}",
            curve.proportion,
        );
    }
}
