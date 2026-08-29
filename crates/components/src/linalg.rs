//! Linear algebra utilities for CMA-ES.
//! Dependency-free, deterministic algorithms.

/// Compute eigenvalues (ascending) and orthonormal eigenvectors (as columns) of a symmetric matrix
/// using the cyclic Jacobi method with pinned procedure.
///
/// Returns (eigenvalues, eigenvectors as columns) where column j pairs with eigenvalue j.
pub fn eigh_jacobi(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let d = a.len();
    if d == 0 {
        return (vec![], vec![]);
    }

    // Ensure input is square
    for row in a {
        assert_eq!(row.len(), d, "Matrix must be square");
    }

    // Create mutable copy of A
    let mut a = a.to_vec();

    // Initialize eigenvector matrix V to identity
    let mut v = (0..d)
        .map(|i| {
            let mut row = vec![0.0; d];
            row[i] = 1.0;
            row
        })
        .collect::<Vec<_>>();

    // Cyclic Jacobi with pinned procedure
    const MAX_SWEEPS: usize = 100;
    const THRESHOLD: f64 = 1e-14;
    const CONVERGENCE: f64 = 1e-12;

    for _sweep in 0..MAX_SWEEPS {
        let mut off_diag_norm_sq = 0.0;

        // Sweep upper triangle in row-major order
        for p in 0..d - 1 {
            for q in (p + 1)..d {
                let a_pq = a[p][q];

                if a_pq.abs() > THRESHOLD {
                    // Compute rotation angle
                    let theta = (a[q][q] - a[p][p]) / (2.0 * a_pq);

                    // Numerically stable t computation
                    let t = if theta >= 0.0 {
                        1.0 / (theta + (theta * theta + 1.0).sqrt())
                    } else {
                        -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                    };

                    let c = 1.0 / (t * t + 1.0).sqrt();
                    let s = t * c;

                    // Apply rotation to A: rows and columns p, q. Each iteration
                    // writes into row i (a[i][p], a[i][q]) *and* mirrors those
                    // values into rows p and q (a[p][i], a[q][i]) to keep A
                    // explicitly symmetric — three simultaneously-live mutable
                    // row borrows, which an iterator adaptor can't express, so
                    // the index form is the correct one here.
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..d {
                        if i != p && i != q {
                            let a_ip = a[i][p];
                            let a_iq = a[i][q];
                            a[i][p] = c * a_ip - s * a_iq;
                            a[i][q] = s * a_ip + c * a_iq;
                            a[p][i] = a[i][p];
                            a[q][i] = a[i][q];
                        }
                    }

                    // Update diagonal elements
                    let a_pp = a[p][p];
                    let a_qq = a[q][q];
                    a[p][p] = c * c * a_pp - 2.0 * s * c * a_pq + s * s * a_qq;
                    a[q][q] = s * s * a_pp + 2.0 * s * c * a_pq + c * c * a_qq;
                    a[p][q] = 0.0;
                    a[q][p] = 0.0;

                    // Update eigenvector matrix V: columns p, q
                    for row in &mut v {
                        let v_ip = row[p];
                        let v_iq = row[q];
                        row[p] = c * v_ip - s * v_iq;
                        row[q] = s * v_ip + c * v_iq;
                    }
                }

                off_diag_norm_sq += a_pq * a_pq;
            }
        }

        // Check convergence
        if off_diag_norm_sq.sqrt() < CONVERGENCE {
            break;
        }
    }

    // Extract eigenvalues from diagonal
    let eigenvalues: Vec<f64> = (0..d).map(|i| a[i][i]).collect();

    // Create index and sort by eigenvalue (stable sort)
    let mut indices: Vec<usize> = (0..d).collect();
    indices.sort_by(|&i, &j| eigenvalues[i].partial_cmp(&eigenvalues[j]).unwrap_or(std::cmp::Ordering::Equal));

    // Reorder eigenvalues and eigenvectors
    let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let mut sorted_eigenvectors = vec![vec![0.0; d]; d];
    for (new_col, &old_col) in indices.iter().enumerate() {
        for row in 0..d {
            sorted_eigenvectors[row][new_col] = v[row][old_col];
        }
    }

    (sorted_eigenvalues, sorted_eigenvectors)
}

/// Matrix-vector multiplication: y = A * x
pub fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), x.len(), "Dimension mismatch");
    let d = a.len();
    let mut y = vec![0.0; d];
    for i in 0..d {
        for j in 0..d {
            y[i] += a[i][j] * x[j];
        }
    }
    y
}

/// Outer product accumulation: C += s * y * y^T
pub fn vec_outer_add(c: &mut [Vec<f64>], s: f64, y: &[f64]) {
    let d = y.len();
    assert_eq!(c.len(), d, "Dimension mismatch");
    for i in 0..d {
        assert_eq!(c[i].len(), d, "Matrix must be square");
        for j in 0..d {
            c[i][j] += s * y[i] * y[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_almost_equal(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "Expected {} ≈ {}, diff = {}", a, b, (a - b).abs());
    }

    fn assert_vec_almost_equal(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "Length mismatch");
        for i in 0..a.len() {
            assert_almost_equal(a[i], b[i], tol);
        }
    }

    fn assert_mat_almost_equal(a: &[Vec<f64>], b: &[Vec<f64>], tol: f64) {
        assert_eq!(a.len(), b.len(), "Row count mismatch");
        for i in 0..a.len() {
            assert_vec_almost_equal(&a[i], &b[i], tol);
        }
    }

    #[test]
    fn identity_eigen() {
        let tol = 1e-9;
        let a = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let (eigenvalues, eigenvectors) = eigh_jacobi(&a);

        // All eigenvalues should be 1
        for &eval in &eigenvalues {
            assert_almost_equal(eval, 1.0, tol);
        }

        // V should be orthonormal
        let d = a.len();
        for j in 0..d {
            // Norm of column j should be 1
            let norm: f64 = eigenvectors.iter().map(|row| row[j] * row[j]).sum();
            assert_almost_equal(norm, 1.0, tol);
        }

        // V^T * V should be I
        for i in 0..d {
            for j in 0..d {
                let dot: f64 = eigenvectors.iter().map(|row| row[i] * row[j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_almost_equal(dot, expected, tol);
            }
        }
    }

    #[test]
    fn two_by_two_known() {
        let tol = 1e-9;
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];

        let (eigenvalues, eigenvectors) = eigh_jacobi(&a);

        // Expected eigenvalues: 1 and 3
        assert_almost_equal(eigenvalues[0], 1.0, tol);
        assert_almost_equal(eigenvalues[1], 3.0, tol);

        // Expected eigenvector directions (up to sign):
        // λ=1: (1,-1)/√2
        // λ=3: (1,1)/√2
        let v1_expected = [1.0 / 2.0_f64.sqrt(), -1.0 / 2.0_f64.sqrt()];
        let v3_expected = [1.0 / 2.0_f64.sqrt(), 1.0 / 2.0_f64.sqrt()];

        let v1_actual = [eigenvectors[0][0], eigenvectors[1][0]];
        let v3_actual = [eigenvectors[0][1], eigenvectors[1][1]];

        // Check dot product (up to sign)
        let mut dot1 = 0.0;
        for i in 0..2 {
            dot1 += v1_actual[i] * v1_expected[i];
        }
        assert!((dot1.abs() - 1.0).abs() < tol, "v1 dot with expected: {}", dot1.abs());

        let mut dot3 = 0.0;
        for i in 0..2 {
            dot3 += v3_actual[i] * v3_expected[i];
        }
        assert!((dot3.abs() - 1.0).abs() < tol, "v3 dot with expected: {}", dot3.abs());
    }

    #[test]
    fn fixed_5x5_reconstruction() {
        let tol = 1e-9;

        // Hard-coded symmetric 5x5 matrix with specific structure
        let a = vec![
            vec![5.5, 0.5, 0.333333, 0.25, 0.2],
            vec![0.5, 5.666667, 0.333333, 0.25, 0.2],
            vec![0.333333, 0.333333, 5.833333, 0.25, 0.2],
            vec![0.25, 0.25, 0.25, 6.0, 0.2],
            vec![0.2, 0.2, 0.2, 0.2, 6.2],
        ];

        let (eigenvalues, eigenvectors) = eigh_jacobi(&a);
        let d = a.len();

        // Test: A * v_j ≈ λ_j * v_j for all columns
        for j in 0..d {
            let v_col: Vec<f64> = (0..d).map(|i| eigenvectors[i][j]).collect();
            let av_col = mat_vec(&a, &v_col);

            for i in 0..d {
                let expected = eigenvalues[j] * v_col[i];
                assert_almost_equal(av_col[i], expected, tol);
            }
        }

        // Test: V^T * V ≈ I (orthonormality)
        for i in 0..d {
            for j in 0..d {
                let dot: f64 = eigenvectors.iter().map(|row| row[i] * row[j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_almost_equal(dot, expected, tol);
            }
        }

        // Test: sum(λ) ≈ trace(A)
        let trace_a: f64 = (0..d).map(|i| a[i][i]).sum();
        let sum_eval: f64 = eigenvalues.iter().sum();
        assert_almost_equal(sum_eval, trace_a, tol);
    }

    #[test]
    fn helpers_work() {
        let tol = 1e-9;

        // Test mat_vec
        let a = vec![vec![2.0, 1.0], vec![3.0, 4.0]];
        let x = vec![1.0, 2.0];
        let y = mat_vec(&a, &x);
        assert_vec_almost_equal(&y, &[4.0, 11.0], tol);

        // Test vec_outer_add
        let mut c = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
        let y = vec![1.0, 2.0];
        vec_outer_add(&mut c, 2.0, &y);
        let expected = vec![vec![2.0, 4.0], vec![4.0, 8.0]];
        assert_mat_almost_equal(&c, &expected, tol);
    }
}
