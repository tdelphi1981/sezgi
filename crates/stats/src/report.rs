//! LaTeX report generation for statistical comparisons of algorithms across
//! benchmark problems.
//!
//! Contains per-algorithm summary statistics and a comprehensive one-call
//! [`paper_package`] function that computes all necessary pairwise and
//! multi-algorithm comparisons, rendering both a summary table and a detailed
//! pairwise comparison matrix.

use crate::{
    bayesian_signed_rank, check_finite, cliffs_delta, cliffs_magnitude, friedman, holm,
    nemenyi_cd, plackett_luce, rank_matrix, wilcoxon_signed_rank, BayesSignedRankResult,
    FriedmanResult, PlackettLuceResult, StatsError,
};

/// Summary statistics for a single algorithm across multiple problems.
#[derive(Debug, Clone, PartialEq)]
pub struct AlgorithmSummary {
    /// Algorithm name.
    pub name: String,
    /// Mean performance across problems.
    pub mean: f64,
    /// Sample standard deviation (n-1 denominator).
    pub std: f64,
    /// Median performance across problems.
    pub median: f64,
    /// Mean rank across problems.
    pub mean_rank: f64,
}

/// Complete statistical analysis and LaTeX report for algorithm comparison.
#[derive(Debug, Clone)]
pub struct PaperPackage {
    /// Friedman test result.
    pub friedman: FriedmanResult,
    /// Nemenyi critical difference.
    pub nemenyi_cd: f64,
    /// Pairwise Wilcoxon signed-rank tests, Holm-adjusted p-values:
    /// `(algo_i, algo_j, adjusted_p)` for i < j.
    pub pairwise_wilcoxon_holm: Vec<(usize, usize, f64)>,
    /// Cliff's delta effect sizes: `(algo_i, algo_j, delta)` for i < j.
    pub cliffs: Vec<(usize, usize, f64)>,
    /// Bayesian signed-rank results: `(algo_i, algo_j, result)` for i < j.
    pub bayes: Vec<(usize, usize, BayesSignedRankResult)>,
    /// Plackett-Luce ranking result.
    pub plackett_luce: PlackettLuceResult,
    /// LaTeX summary table: per-problem mean±std for each algorithm.
    pub latex_summary: String,
    /// LaTeX pairwise comparison matrix: adjusted p-values and effect sizes.
    pub latex_tests: String,
}

/// Generates a LaTeX summary table with mean±std per algorithm per problem.
///
/// # Input
/// - `problem_names`: names of benchmark problems
/// - `algo_names`: names of algorithms
/// - `results_per_problem`: for each problem, a `Vec<Vec<f64>>` where each
///   inner vec contains per-seed performance values for one algorithm
///
/// # Output
/// LaTeX booktabs table with:
/// - `\toprule`, `\midrule`, `\bottomrule`
/// - Per-problem row: `$mean \pm std$` (math mode) per algorithm when the
///   cell has n > 1 samples; just the mean (no ±std, which is undefined for
///   a single sample with the n-1 denominator) when n = 1
/// - Best (lowest mean) per problem: bolded with `\textbf{...}`
/// - Underscores in names escaped as `\_`
/// - Numbers formatted as `{:.3e}` (3-digit exponential notation)
pub fn summary_table_latex(
    problem_names: &[String],
    algo_names: &[String],
    results_per_problem: &[Vec<Vec<f64>>],
) -> String {
    let mut latex = String::from("\\begin{tabular}{l");
    for _ in 0..algo_names.len() {
        latex.push('l');
    }
    latex.push_str("}\n\\toprule\n");

    // Header row
    latex.push_str("Problem");
    for name in algo_names {
        latex.push_str(" & ");
        latex.push_str(&escape_latex_name(name));
    }
    latex.push_str(" \\\\\n\\midrule\n");

    // Data rows
    for (problem_idx, problem_name) in problem_names.iter().enumerate() {
        latex.push_str(&escape_latex_name(problem_name));
        let mut best_mean = f64::INFINITY;
        let mut best_idx = 0;

        // First pass: find best mean
        for (algo_idx, algo_results) in results_per_problem[problem_idx].iter().enumerate() {
            if !algo_results.is_empty() {
                let mean = algo_results.iter().sum::<f64>() / algo_results.len() as f64;
                if mean < best_mean {
                    best_mean = mean;
                    best_idx = algo_idx;
                }
            }
        }

        // Second pass: output values
        for (algo_idx, algo_results) in results_per_problem[problem_idx].iter().enumerate() {
            latex.push_str(" & ");
            if !algo_results.is_empty() {
                let mean = algo_results.iter().sum::<f64>() / algo_results.len() as f64;

                // n = 1: sample std (n-1 denominator) is undefined — emit
                // just the mean. n > 1: mean ± std, wrapped in math mode
                // (\pm is a math-mode-only command).
                let cell = if algo_results.len() == 1 {
                    format!("{:.3e}", mean)
                } else {
                    let variance: f64 = algo_results
                        .iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f64>()
                        / (algo_results.len() as f64 - 1.0);
                    let std = variance.sqrt();
                    format!("${:.3e} \\pm {:.3e}$", mean, std)
                };

                if algo_idx == best_idx {
                    latex.push_str(&format!("\\textbf{{{}}}", cell));
                } else {
                    latex.push_str(&cell);
                }
            }
        }
        latex.push_str(" \\\\\n");
    }

    latex.push_str("\\bottomrule\n\\end{tabular}\n");
    latex
}

/// Escapes underscores in LaTeX text (for names).
fn escape_latex_name(name: &str) -> String {
    name.replace('_', "\\_")
}

/// Computes a comprehensive statistical analysis package for algorithm comparison.
///
/// # Input
/// - `algo_names`: names of algorithms (used for indexing and output)
/// - `problem_names`: names of benchmark problems
/// - `results`: matrix where `results[i][j]` is the performance of algorithm `j`
///   on problem `i`; lower is better. All rows must have exactly `k = algo_names.len()`
///   entries; `n = problem_names.len()` rows.
/// - `rope`: region of practical equivalence (half-width) for Bayesian signed-rank
/// - `bayes_samples`: number of Monte Carlo samples for Bayesian signed-rank
/// - `master_seed`: master RNG seed; per-pair seeds are `master_seed.wrapping_add(pair_index)`
///
/// # Output
/// A [`PaperPackage`] containing all results and LaTeX-formatted tables.
///
/// # Errors
/// Returns `StatsError::InvalidInput` if:
/// - The results matrix is ragged
/// - Friedman test fails (too few problems/algorithms)
/// - Nemenyi CD cannot be computed (k outside 2..=20 or n == 0)
/// - Any Wilcoxon test fails: `n_effective >= 5` paired samples per
///   algorithm pair are required *unless* the pair is exact-eligible (no
///   zero differences, no tied `|d|` ranks, `n_effective <= 25`), in which
///   case the exact small-n distribution applies down to `n_effective >=
///   1`; see [`wilcoxon_signed_rank`] docs for the full eligibility rule.
pub fn paper_package(
    algo_names: &[String],
    problem_names: &[String],
    results: &[Vec<f64>],
    rope: f64,
    bayes_samples: u64,
    master_seed: u64,
) -> Result<PaperPackage, StatsError> {
    check_finite("paper_package", results.iter().flatten().copied())?;

    let n = results.len();
    let k = algo_names.len();

    // Validate input: consistent dimensions
    if n != problem_names.len() {
        return Err(StatsError::InvalidInput(format!(
            "results has {} rows but problem_names has {} entries",
            n,
            problem_names.len()
        )));
    }
    for (i, row) in results.iter().enumerate() {
        if row.len() != k {
            return Err(StatsError::InvalidInput(format!(
                "results[{}] has {} columns, expected {} (ragged matrix)",
                i,
                row.len(),
                k
            )));
        }
    }

    // Friedman test
    let friedman_result = friedman(results)?;

    // Nemenyi critical difference
    let nemenyi_cd_val = nemenyi_cd(k, n)?;

    // All-pairs Wilcoxon with Holm adjustment
    let mut raw_pvals = Vec::new();
    let mut pair_indices = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            let a: Vec<f64> = results.iter().map(|row| row[i]).collect();
            let b: Vec<f64> = results.iter().map(|row| row[j]).collect();
            let wilcoxon_result = wilcoxon_signed_rank(&a, &b)?;
            raw_pvals.push(wilcoxon_result.p_value);
            pair_indices.push((i, j));
        }
    }
    let adjusted_pvals = holm(&raw_pvals);
    let pairwise_wilcoxon_holm: Vec<(usize, usize, f64)> = pair_indices
        .iter()
        .zip(adjusted_pvals.iter())
        .map(|(&(i, j), &p)| (i, j, p))
        .collect();

    // Cliff's delta per pair
    let mut cliffs_vec = Vec::new();
    for i in 0..k {
        for j in (i + 1)..k {
            let a: Vec<f64> = results.iter().map(|row| row[i]).collect();
            let b: Vec<f64> = results.iter().map(|row| row[j]).collect();
            let delta = cliffs_delta(&a, &b)?;
            cliffs_vec.push((i, j, delta));
        }
    }

    // Bayesian signed-rank per pair
    let mut bayes_vec = Vec::new();
    for (pair_idx, (i, j)) in pair_indices.iter().enumerate() {
        let a: Vec<f64> = results.iter().map(|row| row[*i]).collect();
        let b: Vec<f64> = results.iter().map(|row| row[*j]).collect();
        let seed = master_seed.wrapping_add(pair_idx as u64);
        let bayes_result = bayesian_signed_rank(&a, &b, rope, bayes_samples, seed)?;
        bayes_vec.push((*i, *j, bayes_result));
    }

    // Plackett-Luce on rank-matrix argsorted
    let ranks = rank_matrix(results);
    let mut rankings = Vec::new();
    for row in ranks {
        let mut argsorted: Vec<usize> = (0..k).collect();
        argsorted.sort_by(|&a, &b| {
            row[a]
                .partial_cmp(&row[b])
                .expect("NaN in rank matrix")
        });
        rankings.push(argsorted);
    }
    let pl_result = plackett_luce(&rankings)?;

    // LaTeX summary table
    // Build results_per_problem: for each problem, a vec of vecs of f64.
    // Each cell holds a single (already aggregated) value; the table
    // renderer emits just that value for n = 1 cells (no ±std, whose n-1
    // denominator would be zero).
    let results_per_problem: Vec<Vec<Vec<f64>>> = results
        .iter()
        .map(|row| row.iter().map(|&v| vec![v]).collect())
        .collect();
    let latex_summary = summary_table_latex(problem_names, algo_names, &results_per_problem);

    // LaTeX tests: pairwise matrix with Holm-adjusted p + Cliff magnitude
    let latex_tests = pairwise_tests_latex(
        algo_names,
        &pairwise_wilcoxon_holm,
        &cliffs_vec,
    );

    Ok(PaperPackage {
        friedman: friedman_result,
        nemenyi_cd: nemenyi_cd_val,
        pairwise_wilcoxon_holm,
        cliffs: cliffs_vec,
        bayes: bayes_vec,
        plackett_luce: pl_result,
        latex_summary,
        latex_tests,
    })
}

/// Generates a LaTeX pairwise comparison matrix.
///
/// Rows and columns are algorithms; each cell `(i, j)` with `i < j` shows
/// the Holm-adjusted Wilcoxon p-value and Cliff's delta magnitude.
fn pairwise_tests_latex(
    algo_names: &[String],
    pairwise_wilcoxon_holm: &[(usize, usize, f64)],
    cliffs: &[(usize, usize, f64)],
) -> String {
    let k = algo_names.len();
    let mut latex = String::from("\\begin{tabular}{l");
    for _ in 0..k {
        latex.push('l');
    }
    latex.push_str("}\n\\toprule\n");

    // Header
    latex.push_str("Pair");
    for name in algo_names {
        latex.push_str(" & ");
        latex.push_str(&escape_latex_name(name));
    }
    latex.push_str(" \\\\\n\\midrule\n");

    // Rows
    for (i, algo_name) in algo_names.iter().enumerate().take(k) {
        latex.push_str(&escape_latex_name(algo_name));
        for j in 0..k {
            latex.push_str(" & ");
            if i < j {
                // Upper triangle: show p-value and delta. A missing pair is
                // rendered as "--" (defensive: paper_package always supplies
                // every i < j pair). It must NOT default to 0.0 — p = 0.0000
                // would read as maximal significance.
                let p = pairwise_wilcoxon_holm
                    .iter()
                    .find(|&&(ai, aj, _)| ai == i && aj == j)
                    .map(|&(_, _, p)| p);
                let delta = cliffs
                    .iter()
                    .find(|&&(ai, aj, _)| ai == i && aj == j)
                    .map(|&(_, _, d)| d);
                match (p, delta) {
                    (Some(p), Some(delta)) => {
                        let mag = cliffs_magnitude(delta);
                        latex.push_str(&format!("$p={:.4}$, $\\Delta$ = {}", p, mag));
                    }
                    _ => latex.push_str("--"),
                }
            } else if i == j {
                latex.push_str("--");
            }
            // Lower triangle: empty (symmetric)
        }
        latex.push_str(" \\\\\n");
    }

    latex.push_str("\\bottomrule\n\\end{tabular}\n");
    latex
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_table_latex_2x2_fixture() {
        let problem_names = vec!["P_1".to_string(), "P_2".to_string()];
        let algo_names = vec!["A_1".to_string(), "B_2".to_string()];

        // P_1: A=1.0, B=2.0 (A better)
        // P_2: A=3.0, B=2.0 (B better)
        let results_per_problem = vec![
            vec![vec![1.0], vec![2.0]],
            vec![vec![3.0], vec![2.0]],
        ];

        let latex = summary_table_latex(&problem_names, &algo_names, &results_per_problem);

        // Check for required LaTeX elements
        assert!(latex.contains("\\toprule"), "missing \\toprule");
        assert!(latex.contains("\\midrule"), "missing \\midrule");
        assert!(latex.contains("\\bottomrule"), "missing \\bottomrule");

        // Check for escaped underscores in algorithm names
        assert!(latex.contains("A\\_1"), "underscore not escaped in A_1");
        assert!(latex.contains("B\\_2"), "underscore not escaped in B_2");

        // Check for escaped underscores in problem names (end-to-end escaping)
        assert!(latex.contains("P\\_1"), "underscore not escaped in P_1");
        assert!(latex.contains("P\\_2"), "underscore not escaped in P_2");

        // Verify exactly one bold per problem row
        let lines: Vec<&str> = latex.lines().collect();
        let mut matched_rows = 0;
        for line in &lines {
            if line.contains("P\\_1") || line.contains("P\\_2") {
                matched_rows += 1;
                let bold_count = line.matches("\\textbf{").count();
                assert_eq!(
                    bold_count, 1,
                    "expected exactly 1 bold per problem row, got {}",
                    bold_count
                );
            }
        }
        assert_eq!(
            matched_rows, 2,
            "expected to find 2 problem rows (P\\_1 and P\\_2), found {}",
            matched_rows
        );
    }

    #[test]
    fn paper_package_3x6_synthetic() {
        // 3 algorithms, 6 problems, synthetic data with some variance
        let algo_names = vec!["Algo_A".to_string(), "Algo_B".to_string(), "Algo_C".to_string()];
        let problem_names: Vec<String> = (0..6).map(|i| format!("P{}", i)).collect();

        let results = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.5, 2.2, 2.9],
            vec![1.2, 1.9, 3.1],
            vec![0.9, 2.3, 3.2],
            vec![1.3, 2.1, 2.8],
            vec![1.1, 2.0, 3.0],
        ];

        let package = paper_package(
            &algo_names,
            &problem_names,
            &results,
            0.1,
            1000,
            42,
        )
        .expect("valid fixture");

        // Check all fields are populated
        assert_eq!(package.friedman.mean_ranks.len(), 3);
        assert_eq!(package.pairwise_wilcoxon_holm.len(), 3); // C(3,2) = 3 pairs
        assert_eq!(package.cliffs.len(), 3);
        assert_eq!(package.bayes.len(), 3);
        assert_eq!(package.plackett_luce.worths.len(), 3);

        // Check Holm p-values are in [0, 1]
        for (_, _, p) in &package.pairwise_wilcoxon_holm {
            assert!(*p >= 0.0 && *p <= 1.0, "p-value out of range: {}", p);
        }

        // Check PL worths sum to 1 ± 1e-9
        let sum: f64 = package.plackett_luce.worths.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "PL worths sum to {}, expected 1.0",
            sum
        );

        // Check LaTeX tables are non-empty
        assert!(!package.latex_summary.is_empty());
        assert!(!package.latex_tests.is_empty());

        // paper_package cells are single aggregated values (n = 1): the
        // summary table must emit just the mean — never "NaN" (sample std's
        // n-1 denominator is zero for n = 1) and never a ±std.
        assert!(!package.latex_summary.contains("NaN"), "latex_summary must not contain NaN");
        assert!(!package.latex_summary.contains("\\pm"), "n=1 cells must not emit \\pm");
        assert!(!package.latex_tests.contains("NaN"), "latex_tests must not contain NaN");
    }

    #[test]
    fn summary_table_multi_sample_emits_math_mode_pm() {
        let problem_names = vec!["P1".to_string()];
        let algo_names = vec!["A".to_string(), "B".to_string()];

        // n = 3 samples per cell -> mean ± std in math mode.
        let results_per_problem = vec![vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]];

        let latex = summary_table_latex(&problem_names, &algo_names, &results_per_problem);

        assert!(!latex.contains("NaN"), "multi-sample table must not contain NaN");
        // Non-best cell: bare math-mode $mean \pm std$.
        assert!(
            latex.contains("$5.000e0 \\pm 1.000e0$"),
            "expected math-mode mean ± std cell, got:\n{}",
            latex
        );
        // Best cell: bolded math-mode cell.
        assert!(
            latex.contains("\\textbf{$2.000e0 \\pm 1.000e0$}"),
            "expected bolded math-mode best cell, got:\n{}",
            latex
        );
        // Every \pm must sit inside math mode: no " \pm " preceded by a
        // non-$ context. Check there is no \pm outside $...$ by ensuring
        // each line containing \pm has an even number of $ around it.
        for line in latex.lines() {
            if line.contains("\\pm") {
                let dollars = line.matches('$').count();
                assert!(dollars >= 2 && dollars % 2 == 0, "\\pm outside math mode in: {}", line);
            }
        }
    }

    #[test]
    fn summary_table_single_sample_emits_mean_only() {
        let problem_names = vec!["P1".to_string()];
        let algo_names = vec!["A".to_string(), "B".to_string()];

        let results_per_problem = vec![vec![vec![1.0], vec![2.0]]];

        let latex = summary_table_latex(&problem_names, &algo_names, &results_per_problem);

        assert!(!latex.contains("NaN"), "single-sample table must not contain NaN");
        assert!(!latex.contains("\\pm"), "single-sample cells must not emit \\pm");
        assert!(latex.contains("\\textbf{1.000e0}"), "best n=1 cell is the bolded bare mean");
        assert!(latex.contains("2.000e0"), "n=1 cell is the bare mean");
    }

    #[test]
    fn paper_package_determinism() {
        let algo_names = vec!["A".to_string(), "B".to_string()];
        let problem_names = vec![
            "P1".to_string(),
            "P2".to_string(),
            "P3".to_string(),
            "P4".to_string(),
            "P5".to_string(),
        ];

        let results = vec![
            vec![1.0, 2.0],
            vec![1.5, 1.8],
            vec![1.2, 2.1],
            vec![0.9, 2.3],
            vec![1.3, 1.9],
        ];

        let pkg1 = paper_package(&algo_names, &problem_names, &results, 0.1, 100, 777)
            .expect("first run");

        let pkg2 = paper_package(&algo_names, &problem_names, &results, 0.1, 100, 777)
            .expect("second run");

        // Check LaTeX outputs are identical
        assert_eq!(pkg1.latex_tests, pkg2.latex_tests, "latex_tests differs");
        assert_eq!(pkg1.latex_summary, pkg2.latex_summary, "latex_summary differs");

        // Check Bayes results are identical
        assert_eq!(pkg1.bayes, pkg2.bayes, "bayesian results differ");
    }

    #[test]
    fn paper_package_errors_on_nan() {
        let algo_names = vec!["A".to_string(), "B".to_string()];
        let problem_names = vec!["P1".to_string(), "P2".to_string()];
        let results = vec![vec![1.0, f64::NAN], vec![1.5, 1.8]];

        let err = paper_package(&algo_names, &problem_names, &results, 0.1, 100, 777)
            .expect_err("NaN must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("paper_package"), "message should name the function: {msg}");
        assert!(msg.contains("non-finite"), "message should say non-finite: {msg}");
    }

    #[test]
    fn paper_package_errors_on_too_few_problems() {
        // n = 1 problem: friedman() requires n >= 2 rows, so this still
        // errors regardless of Wilcoxon eligibility (unlike the fixture
        // below, which used to error here too before Task 6).
        let algo_names = vec!["A".to_string(), "B".to_string()];
        let problem_names = vec!["P1".to_string()];

        let results = vec![vec![1.0, 2.0]];

        let result = paper_package(&algo_names, &problem_names, &results, 0.1, 100, 777);

        // Should error because friedman needs >= 2 problems (rows).
        assert!(result.is_err(), "expected error for insufficient problems");
    }

    // *** TASK 6 BEHAVIOR CHANGE (M2d-1, flagged prominently per the
    // brief's disclosure rule): this is the SAME fixture that used to be
    // `paper_package_errors_on_too_few_problems` (3 problems, 2 algorithms,
    // A=[1.0,1.5,1.2], B=[2.0,1.8,2.1]). It used to error because the
    // per-pair `wilcoxon_signed_rank` call has n_effective = 3 < 5, and
    // pre-Task-6 that unconditionally errored (no exact small-n
    // distribution existed to fall back on). d = A - B = [-1.0, -0.3,
    // -0.9], |d| = [1.0, 0.3, 0.9] -- all distinct, no zeros -- so this
    // pair is now exact-eligible (t0=0, n_effective=3 <= 25, no ties), and
    // paper_package succeeds end-to-end (Nemenyi CD and
    // bayesian_signed_rank have no n_effective >= 5 floor, so nothing else
    // blocks it). This is a direct, intended consequence of Task 6 lifting
    // the n_effective < 5 floor for exact-eligible input inside
    // `wilcoxon_signed_rank` -- flagged here since it changes a
    // previously-pinned Err assertion to Ok.
    #[test]
    fn paper_package_succeeds_with_few_problems_via_exact_wilcoxon() {
        let algo_names = vec!["A".to_string(), "B".to_string()];
        let problem_names = vec!["P1".to_string(), "P2".to_string(), "P3".to_string()];

        let results = vec![
            vec![1.0, 2.0],
            vec![1.5, 1.8],
            vec![1.2, 2.1],
        ];

        let result = paper_package(&algo_names, &problem_names, &results, 0.1, 100, 777);

        assert!(
            result.is_ok(),
            "n_effective=3 is now exact-eligible (no zeros, no ties), not an error: {:?}",
            result.err()
        );
        let pkg = result.unwrap();
        assert_eq!(pkg.pairwise_wilcoxon_holm.len(), 1); // C(2,2)=1 pair
        let (_, _, p) = pkg.pairwise_wilcoxon_holm[0];
        assert!((0.0..=1.0).contains(&p), "p-value out of range: {}", p);
    }
}
