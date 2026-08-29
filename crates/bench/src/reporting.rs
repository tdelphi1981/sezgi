//! Bridges [`crate::experiment::RunRecord`] outputs to [`sezgi_stats`]'s
//! paper-package statistics (M2c Task 10).
//!
//! [`results_matrix`] turns a flat `&[RunRecord]` into the `(algo_names,
//! problem_labels, matrix)` shape [`sezgi_stats`] expects, for one budget:
//! each cell aggregates the per-seed gaps (`best_f - f_opt`, lower is
//! better) for one (problem, algorithm) pair.
//!
//! ## Multi-budget reporting is first-class
//!
//! Piotrowski et al. (2025) show that algorithm rankings on benchmark
//! comparisons can flip depending on which evaluation budget is examined:
//! an algorithm that leads at a small budget need not lead at a large one,
//! and vice versa. Reporting statistics for a single, arbitrarily chosen
//! budget therefore risks presenting an artifact of that choice as a
//! general conclusion. [`per_budget_packages`] makes multi-budget reporting
//! the default rather than an afterthought: it produces one
//! [`sezgi_stats::PaperPackage`] PER DISTINCT BUDGET present in the
//! records, so a paper can report — and a reader can see — how (and
//! whether) conclusions hold across budgets, instead of only at one.

use crate::experiment::{ExperimentError, RunRecord};
use sezgi_stats::PaperPackage;
use std::collections::HashMap;

/// How to combine a (problem, algorithm) cell's per-seed gap values into a
/// single number for the results matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregate {
    Mean,
    Median,
}

impl Aggregate {
    /// `values` must be non-empty.
    fn apply(self, values: &[f64]) -> f64 {
        debug_assert!(!values.is_empty());
        match self {
            Aggregate::Mean => values.iter().sum::<f64>() / values.len() as f64,
            Aggregate::Median => {
                let mut sorted = values.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).expect("NaN in gap values"));
                let n = sorted.len();
                if n % 2 == 1 {
                    sorted[n / 2]
                } else {
                    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                }
            }
        }
    }
}

/// Builds a [`sezgi_stats`]-shaped results matrix for one `budget`.
///
/// - Records not matching `budget` are ignored.
/// - Problem label = `f{fid}d{dim}i{instance}`; problems and algorithms are
///   both ordered by FIRST APPEARANCE in `records` (stable, pinned — not
///   sorted). This makes the returned `algo_names`/`problem_labels` order
///   a deterministic function of `records`' order, not of hashing or of
///   numeric field values.
/// - A cell (problem, algorithm) is the `aggregate` of that pair's
///   per-seed gaps (`best_f - f_opt`; lower is better, matching
///   [`sezgi_stats`]'s convention).
/// - Every (problem, algorithm) pair that appears for ANY OTHER algorithm
///   (resp. problem) at this budget must also appear for this one — a
///   missing cell means the experiment is incomplete at this budget, and
///   is reported as [`ExperimentError::MissingCell`] rather than silently
///   skipped or filled in.
///
/// `(algo_names, problem_labels, matrix)` triple returned by
/// [`results_matrix`]: `matrix[i][j]` is the aggregated gap of
/// `algo_names[j]` on `problem_labels[i]`.
pub type ResultsMatrix = (Vec<String>, Vec<String>, Vec<Vec<f64>>);

/// Returns `(algo_names, problem_labels, matrix)` where `matrix[i][j]` is
/// the aggregated gap of `algo_names[j]` on `problem_labels[i]`.
pub fn results_matrix(
    records: &[RunRecord],
    budget: u64,
    aggregate: Aggregate,
) -> Result<ResultsMatrix, ExperimentError> {
    let mut algo_names: Vec<String> = Vec::new();
    let mut algo_index: HashMap<String, usize> = HashMap::new();
    let mut problem_labels: Vec<String> = Vec::new();
    let mut problem_index: HashMap<String, usize> = HashMap::new();
    let mut cells: HashMap<(usize, usize), Vec<f64>> = HashMap::new();

    for r in records {
        if r.key.budget != budget {
            continue;
        }

        let algo_idx = *algo_index.entry(r.key.algo.clone()).or_insert_with(|| {
            algo_names.push(r.key.algo.clone());
            algo_names.len() - 1
        });

        let label = format!("f{}d{}i{}", r.key.fid, r.key.dim, r.key.instance);
        let problem_idx = *problem_index.entry(label.clone()).or_insert_with(|| {
            problem_labels.push(label.clone());
            problem_labels.len() - 1
        });

        let gap = r.best_f - r.f_opt;
        cells.entry((problem_idx, algo_idx)).or_default().push(gap);
    }

    let mut matrix = Vec::with_capacity(problem_labels.len());
    for (p_idx, p_label) in problem_labels.iter().enumerate() {
        let mut row = Vec::with_capacity(algo_names.len());
        for (a_idx, a_name) in algo_names.iter().enumerate() {
            let gaps = cells.get(&(p_idx, a_idx)).ok_or_else(|| ExperimentError::MissingCell {
                algo: a_name.clone(),
                problem: p_label.clone(),
                budget,
            })?;
            row.push(aggregate.apply(gaps));
        }
        matrix.push(row);
    }

    Ok((algo_names, problem_labels, matrix))
}

/// Runs [`results_matrix`] + [`sezgi_stats::paper_package`] for every
/// distinct budget present in `records`, in ascending budget order.
///
/// See the module doc for why this is the multi-budget default rather
/// than a single-budget report: per Piotrowski et al. (2025), rankings can
/// flip across budgets, so each budget gets its own [`PaperPackage`]
/// instead of the caller having to pick one budget to report.
///
/// `rope`, `bayes_samples`, `master_seed` are forwarded unchanged to
/// [`sezgi_stats::paper_package`] for every budget (same Bayesian
/// signed-rank configuration applied per budget).
pub fn per_budget_packages(
    records: &[RunRecord],
    rope: f64,
    bayes_samples: u64,
    master_seed: u64,
    aggregate: Aggregate,
) -> Result<Vec<(u64, PaperPackage)>, ExperimentError> {
    let mut budgets: Vec<u64> = Vec::new();
    for r in records {
        if !budgets.contains(&r.key.budget) {
            budgets.push(r.key.budget);
        }
    }
    budgets.sort_unstable();

    let mut packages = Vec::with_capacity(budgets.len());
    for budget in budgets {
        let (algo_names, problem_labels, matrix) = results_matrix(records, budget, aggregate)?;
        let package = sezgi_stats::paper_package(
            &algo_names,
            &problem_labels,
            &matrix,
            rope,
            bayes_samples,
            master_seed,
        )
        .map_err(|source| ExperimentError::Stats { budget, source })?;
        packages.push((budget, package));
    }
    Ok(packages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::RunKey;

    fn record(algo: &str, fid: u32, dim: usize, instance: u32, seed: u64, budget: u64, gap: f64) -> RunRecord {
        let f_opt = 10.0;
        RunRecord {
            key: RunKey { algo: algo.into(), fid, dim, instance, seed, budget },
            best_f: f_opt + gap,
            f_opt,
            evals_used: budget,
            wall_secs: 0.0,
        }
    }

    /// 2 algos ("A", "B") x 2 problems x 2 seeds x 2 budgets. At budget
    /// 100 A is better (lower gaps); at budget 1000 B is better — the
    /// better algorithm FLIPS between budgets, by construction.
    fn flip_fixture_2problems() -> Vec<RunRecord> {
        let mut recs = Vec::new();
        // problem 1 = f1d2i1, problem 2 = f1d2i2
        for (instance, _) in [(1u32, ()), (2u32, ())] {
            // budget 100: A={1,2} B={3,4} -> A better (mean 1.5 < 3.5)
            recs.push(record("A", 1, 2, instance, 1, 100, 1.0));
            recs.push(record("A", 1, 2, instance, 2, 100, 2.0));
            recs.push(record("B", 1, 2, instance, 1, 100, 3.0));
            recs.push(record("B", 1, 2, instance, 2, 100, 4.0));
            // budget 1000: A={5,6} B={1,2} -> B better (mean 1.5 < 5.5)
            recs.push(record("A", 1, 2, instance, 1, 1000, 5.0));
            recs.push(record("A", 1, 2, instance, 2, 1000, 6.0));
            recs.push(record("B", 1, 2, instance, 1, 1000, 1.0));
            recs.push(record("B", 1, 2, instance, 2, 1000, 2.0));
        }
        recs
    }

    #[test]
    fn results_matrix_shape_labels_and_values() {
        let recs = flip_fixture_2problems();

        let (algos, problems, matrix) = results_matrix(&recs, 100, Aggregate::Mean).unwrap();
        assert_eq!(algos, vec!["A".to_string(), "B".to_string()], "first-appearance algo order");
        assert_eq!(problems, vec!["f1d2i1".to_string(), "f1d2i2".to_string()], "first-appearance problem order");
        assert_eq!(matrix.len(), 2, "one row per problem");
        assert_eq!(matrix[0].len(), 2, "one column per algorithm");

        // budget 100: A mean gap = (1+2)/2 = 1.5, B mean gap = (3+4)/2 = 3.5, both problems identical.
        for row in &matrix {
            assert!((row[0] - 1.5).abs() < 1e-12, "A cell: got {}", row[0]);
            assert!((row[1] - 3.5).abs() < 1e-12, "B cell: got {}", row[1]);
        }

        let (_, _, matrix_1000) = results_matrix(&recs, 1000, Aggregate::Mean).unwrap();
        // budget 1000: A mean gap = (5+6)/2 = 5.5, B mean gap = (1+2)/2 = 1.5.
        for row in &matrix_1000 {
            assert!((row[0] - 5.5).abs() < 1e-12, "A cell: got {}", row[0]);
            assert!((row[1] - 1.5).abs() < 1e-12, "B cell: got {}", row[1]);
        }
    }

    #[test]
    fn results_matrix_median_matches_mean_for_two_seeds() {
        // With exactly 2 seeds, median == mean (average of the two values);
        // this pins that Aggregate::Median takes the intended code path
        // (see `aggregate_median_vs_mean_distinct` below for a case where
        // they actually differ).
        let recs = flip_fixture_2problems();
        let (_, _, mean_matrix) = results_matrix(&recs, 100, Aggregate::Mean).unwrap();
        let (_, _, median_matrix) = results_matrix(&recs, 100, Aggregate::Median).unwrap();
        assert_eq!(mean_matrix, median_matrix);
    }

    #[test]
    fn friedman_mean_ranks_flip_across_budgets() {
        let recs = flip_fixture_2problems();

        let (_, _, matrix_100) = results_matrix(&recs, 100, Aggregate::Mean).unwrap();
        let (_, _, matrix_1000) = results_matrix(&recs, 1000, Aggregate::Mean).unwrap();

        let friedman_100 = sezgi_stats::friedman(&matrix_100).unwrap();
        let friedman_1000 = sezgi_stats::friedman(&matrix_1000).unwrap();

        // Column order is [A, B] at both budgets (first-appearance order is
        // fixture-stable). At budget 100 A is better -> A's mean rank is
        // lower; at budget 1000 B is better -> B's mean rank is lower.
        assert!(
            friedman_100.mean_ranks[0] < friedman_100.mean_ranks[1],
            "budget 100: expected A (idx 0) ranked better than B, got {:?}",
            friedman_100.mean_ranks
        );
        assert!(
            friedman_1000.mean_ranks[1] < friedman_1000.mean_ranks[0],
            "budget 1000: expected B (idx 1) ranked better than A, got {:?}",
            friedman_1000.mean_ranks
        );
    }

    /// 6 problems (paper_package requires >= 5 for Wilcoxon), 2 algos, 2
    /// seeds, 2 budgets, same flip property as `flip_fixture_2problems`.
    /// Exercises `per_budget_packages` end to end, separately from the
    /// smaller 2-problem fixture used above for exact matrix/friedman
    /// values (kept small and hand-checkable).
    fn flip_fixture_6problems() -> Vec<RunRecord> {
        let mut recs = Vec::new();
        for instance in 1u32..=6 {
            recs.push(record("A", 1, 2, instance, 1, 100, 1.0));
            recs.push(record("A", 1, 2, instance, 2, 100, 2.0));
            recs.push(record("B", 1, 2, instance, 1, 100, 3.0));
            recs.push(record("B", 1, 2, instance, 2, 100, 4.0));
            recs.push(record("A", 1, 2, instance, 1, 1000, 5.0));
            recs.push(record("A", 1, 2, instance, 2, 1000, 6.0));
            recs.push(record("B", 1, 2, instance, 1, 1000, 1.0));
            recs.push(record("B", 1, 2, instance, 2, 1000, 2.0));
        }
        recs
    }

    #[test]
    fn per_budget_packages_one_per_budget_ascending_and_ranks_flip() {
        let recs = flip_fixture_6problems();
        let packages = per_budget_packages(&recs, 0.1, 200, 42, Aggregate::Mean).unwrap();

        assert_eq!(packages.len(), 2, "one PaperPackage per distinct budget");
        assert_eq!(packages[0].0, 100, "ascending budget order");
        assert_eq!(packages[1].0, 1000, "ascending budget order");

        let (_, pkg_100) = &packages[0];
        let (_, pkg_1000) = &packages[1];
        assert_eq!(pkg_100.friedman.mean_ranks.len(), 2);
        assert!(
            pkg_100.friedman.mean_ranks[0] < pkg_100.friedman.mean_ranks[1],
            "budget 100: A should rank better, got {:?}",
            pkg_100.friedman.mean_ranks
        );
        assert!(
            pkg_1000.friedman.mean_ranks[1] < pkg_1000.friedman.mean_ranks[0],
            "budget 1000: B should rank better, got {:?}",
            pkg_1000.friedman.mean_ranks
        );
    }

    #[test]
    fn missing_cell_errors_naming_the_pair() {
        let mut recs = flip_fixture_2problems();
        // Drop every budget-100 record for algo "B" on problem f1d2i2 (instance 2).
        recs.retain(|r| {
            !(r.key.algo == "B" && r.key.instance == 2 && r.key.budget == 100)
        });

        let err = results_matrix(&recs, 100, Aggregate::Mean).unwrap_err();
        match err {
            ExperimentError::MissingCell { algo, problem, budget } => {
                assert_eq!(algo, "B");
                assert_eq!(problem, "f1d2i2");
                assert_eq!(budget, 100);
            }
            other => panic!("expected MissingCell, got {other:?}"),
        }
    }

    #[test]
    fn aggregate_median_vs_mean_distinct() {
        // Asymmetric (skewed) fixture: 3 seeds per (problem, algo) cell,
        // values chosen so mean != median.
        // gaps: 1.0, 2.0, 100.0 -> mean = 34.333..., median = 2.0
        let f_opt = 10.0;
        let recs = vec![
            RunRecord {
                key: RunKey { algo: "A".into(), fid: 1, dim: 2, instance: 1, seed: 1, budget: 100 },
                best_f: f_opt + 1.0, f_opt, evals_used: 100, wall_secs: 0.0,
            },
            RunRecord {
                key: RunKey { algo: "A".into(), fid: 1, dim: 2, instance: 1, seed: 2, budget: 100 },
                best_f: f_opt + 2.0, f_opt, evals_used: 100, wall_secs: 0.0,
            },
            RunRecord {
                key: RunKey { algo: "A".into(), fid: 1, dim: 2, instance: 1, seed: 3, budget: 100 },
                best_f: f_opt + 100.0, f_opt, evals_used: 100, wall_secs: 0.0,
            },
        ];

        let (_, _, mean_matrix) = results_matrix(&recs, 100, Aggregate::Mean).unwrap();
        let (_, _, median_matrix) = results_matrix(&recs, 100, Aggregate::Median).unwrap();

        assert!((mean_matrix[0][0] - (1.0 + 2.0 + 100.0) / 3.0).abs() < 1e-9);
        assert!((median_matrix[0][0] - 2.0).abs() < 1e-12);
        assert_ne!(mean_matrix[0][0], median_matrix[0][0], "mean and median must differ on a skewed fixture");
    }
}
