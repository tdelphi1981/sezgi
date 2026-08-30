//! The ZDT multi-objective test-problem suite.
//!
//! Source (PROVENANCE, fetched and read directly, not from memory): Eckart
//! Zitzler, Kalyan Deb, Lothar Thiele, "Comparison of Multiobjective
//! Evolutionary Algorithms: Empirical Results", *Evolutionary Computation*
//! 8(2):173-195, 2000 (MIT Press). PDF fetched from the authors' own ETH
//! Zurich TIK group publication list:
//! `https://sop.tik.ee.ethz.ch/publicationListFiles/zdt2000a.pdf`.
//!
//! The paper's Definition 4 (p.177-178) introduces six test functions
//! T1,...,T6 (T1-T4 and T6 here; T5 is excluded, see below) following the
//! common scheme of its Equation 6 (p.177):
//!
//! ```text
//! Minimize  T(x) = (f1(x1), f2(x))
//! subject to f2(x) = g(x2,...,xm) h(f1(x1), g(x2,...,xm))
//! where x = (x1,...,xm)
//! ```
//!
//! Quoted verbatim per function (T1=ZDT1, ..., T6=ZDT6; `m` is the paper's
//! name for the dimension, `Zdt::new`'s `dim`):
//!
//! - **T1** ("has a convex Pareto-optimal front"):
//!   `f1(x1) = x1`, `g(x2,...,xm) = 1 + 9 * sum_{i=2}^{m} xi/(m-1)`,
//!   `h(f1,g) = 1 - sqrt(f1/g)`, "where `m = 30`, and `xi in [0,1]`. The
//!   Pareto-optimal front is formed with `g(x) = 1`."
//! - **T2** ("the nonconvex counterpart to T1"): same `f1`, `g`; `h(f1,g) =
//!   1 - (f1/g)^2`, same `m = 30`, `xi in [0,1]` and `g(x) = 1` front.
//! - **T3** ("represents the discreteness feature; its Pareto-optimal front
//!   consists of several noncontiguous convex parts"): same `f1`, `g`;
//!   `h(f1,g) = 1 - sqrt(f1/g) - (f1/g) sin(10 pi f1)`, `m = 30`, `xi in
//!   [0,1]`, front at `g(x) = 1`. "The introduction of the sine function in
//!   h causes discontinuity in the Pareto-optimal front. However, there is
//!   no discontinuity in the parameter space."
//! - **T4** ("contains 21^9 local Pareto-optimal fronts and, therefore,
//!   tests for the EA's ability to deal with multimodality"): `f1(x1) =
//!   x1`, `g(x2,...,xm) = 1 + 10(m-1) + sum_{i=2}^{m} (xi^2 - 10 cos(4 pi
//!   xi))`, `h(f1,g) = 1 - sqrt(f1/g)`, "where `m = 10`, `x1 in [0,1]`, and
//!   `x2,...,xm in [-5,5]`. The global Pareto-optimal front is formed with
//!   `g(x) = 1`, the best local Pareto-optimal front with `g(x) = 1.25`."
//! - **T6** ("includes two difficulties caused by the nonuniformity of the
//!   search space: first, the Pareto-optimal solutions are nonuniformly
//!   distributed along the global Pareto front (the front is biased for
//!   solutions for which f1(x) is near one); second, the density of the
//!   solutions is lowest near the Pareto-optimal front and highest away
//!   from the front"): `f1(x1) = 1 - exp(-4 x1) sin^6(6 pi x1)`,
//!   `g(x2,...,xm) = 1 + 9 * ((sum_{i=2}^{m} xi)/(m-1))^0.25`, `h(f1,g) = 1
//!   - (f1/g)^2`, "where `m = 10`, `xi in [0,1]`. The Pareto-optimal front
//!   is formed with `g(x) = 1` and is nonconvex."
//!
//! The quoted `m = 30` (T1-T3) / `m = 10` (T4, T6) values are the paper's
//! published EXPERIMENTAL defaults, pinned here only as documentation: this
//! module's `Zdt::new(which, dim)` always takes `dim` explicitly and never
//! silently substitutes these numbers.
//!
//! **T5 excluded**: T5 is a deceptive, BINARY-coded problem (`xi` a bit
//! string, `f1(x1) = 1 + u(x1)` a unitation count) that does not fit this
//! module's real-coded `Block::Float` representation; it is a documented
//! deferral, not part of this suite.
//!
//! ## Analytic Pareto fronts (objective space)
//!
//! For T1, T2, T3, T4, all Pareto-optimal points have `g(x) = 1` (any
//! `g > 1` is strictly dominated, at the same `f1`, by the `g = 1` point
//! with identical `f1`), and `f1 = x1` directly, so the front is the graph
//! of `phi(f1) = h(f1, 1)` for `f1` ranging over `x1`'s domain -- EXCEPT for
//! T3, where `h`'s `sin(10 pi f1)` term makes `phi` non-monotonic: a point
//! `(f1, phi(f1))` is only non-dominated if no smaller `f1'` gives
//! `phi(f1') <= phi(f1)`, i.e. iff `phi(f1)` is a STRICT RUNNING MINIMUM as
//! `f1` sweeps up from `0`. That is exactly the paper's "several
//! noncontiguous convex parts": scanning `phi` for its running-minimum
//! segments (this module's `zdt3_segments`) recovers them directly from the
//! domination definition, without needing (or fabricating) a closed form
//! for the transcendental segment boundaries. `zdt3_segments` uses a dense
//! grid (1,000,000 cells); against an independent 2,000,000-cell reference
//! computed separately, it reproduces (to ~1e-6) the five segments widely
//! reported for T3: approximately `[0, 0.0830]`, `[0.1822, 0.2578]`,
//! `[0.4093, 0.4536]`, `[0.6184, 0.6528]`, `[0.8233, 0.8518]` -- a
//! numerically DERIVED, not literature-copied, result.
//!
//! T6 is different: `f1(x1) = 1 - exp(-4 x1) sin^6(6 pi x1)` is *not*
//! injective over `x1 in [0,1]` (e.g. `f1(0) = f1(1) = 1`), but since `h` is
//! strictly decreasing in `f1` on `g = 1`, every value `f1` attains is
//! non-dominated relative to every other -- so the (objective-space) front
//! is simply `{(f1, 1 - f1^2) : f1 in [f1_min, 1]}`, where `f1_min =
//! min_{x1 in [0,1]} f1(x1)`. Contrary to a naive guess, **the front does
//! NOT start at `f1 = 0`**: the paper's own text above (nonuniform density,
//! bias toward `f1` near one) already signals this, and `f1_min` is
//! strictly positive. `f1_min` has no closed elementary form (its
//! maximizing `x1` solves the transcendental `tan(6 pi x1) = 9 pi`); this
//! module derives it at run time via a deterministic golden/ternary-style
//! search over `x1 in [0, 1/6]` (`zdt6_f1_min`), where `exp(-4x)
//! sin^6(6*pi*x)` is unimodal (it rises from `0` at `x1=0` to its one
//! interior peak, then falls back toward `0` at `x1=1/6`, the first zero of
//! `sin(6*pi*x)` past the peak) -- an independent numeric check during this
//! task's development converged to `f1_min ~= 0.28077531881537`, matching
//! the commonly reported value for T6.

use std::f64::consts::PI;

use sezgi_core::mo::MoProblem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

#[derive(Debug, thiserror::Error)]
pub enum ZdtError {
    #[error(
        "which must be one of {{1,2,3,4,6}} (ZDT5 is a binary-coded deceptive problem, out of \
         scope for this real-coded suite -- see this module's doc), got {0}"
    )]
    UnknownWhich(u32),
    #[error("dim must be >= 2")]
    BadDim,
}

/// One instance of the ZDT multi-objective suite (`which` selects ZDT1,
/// ZDT2, ZDT3, ZDT4, or ZDT6 -- ZDT5 is excluded, see the module doc).
pub struct Zdt {
    which: u32,
    dim: usize,
    space: SearchSpace,
}

impl Zdt {
    /// `which` in `{1,2,3,4,6}`, `dim >= 2`. For `which == 4`, `dim` splits
    /// into `x1 in [0,1]` (1 variable) and `x2,...,x_dim in [-5,5]`
    /// (`dim - 1` variables) -- the paper's mixed-bound domain for T4 -- so
    /// `dim` must be at least 2 there too (1 for `x1` plus at least 1 more).
    pub fn new(which: u32, dim: usize) -> Result<Self, ZdtError> {
        if !matches!(which, 1 | 2 | 3 | 4 | 6) {
            return Err(ZdtError::UnknownWhich(which));
        }
        if dim < 2 {
            return Err(ZdtError::BadDim);
        }
        let space = if which == 4 {
            SearchSpace::new(vec![
                Block::Float { lo: 0.0, hi: 1.0, n: 1 },
                Block::Float { lo: -5.0, hi: 5.0, n: dim - 1 },
            ])
        } else {
            SearchSpace::new(vec![Block::Float { lo: 0.0, hi: 1.0, n: dim }])
        }
        .expect("ZDT bounds (0<1, -5<5) are always valid");
        Ok(Self { which, dim, space })
    }

    pub fn which(&self) -> u32 { self.which }
    pub fn dim(&self) -> usize { self.dim }

    /// Evaluate one already-flattened decision vector's two objectives
    /// directly from the verified `f1`/`g`/`h` formulas (module doc).
    fn eval_one(&self, xs: &[f64]) -> [f64; 2] {
        let m = xs.len();
        match self.which {
            1..=3 => {
                let f1 = xs[0];
                let g = 1.0 + 9.0 * xs[1..].iter().sum::<f64>() / (m - 1) as f64;
                let f2 = match self.which {
                    1 => g * (1.0 - (f1 / g).sqrt()),
                    2 => g * (1.0 - (f1 / g).powi(2)),
                    3 => g * (1.0 - (f1 / g).sqrt() - (f1 / g) * (10.0 * PI * f1).sin()),
                    _ => unreachable!(),
                };
                [f1, f2]
            }
            4 => {
                let f1 = xs[0];
                let g = 1.0
                    + 10.0 * (m - 1) as f64
                    + xs[1..].iter().map(|&xi| xi * xi - 10.0 * (4.0 * PI * xi).cos()).sum::<f64>();
                let f2 = g * (1.0 - (f1 / g).sqrt());
                [f1, f2]
            }
            6 => {
                let f1 = 1.0 - (-4.0 * xs[0]).exp() * (6.0 * PI * xs[0]).sin().powi(6);
                let g = 1.0 + 9.0 * (xs[1..].iter().sum::<f64>() / (m - 1) as f64).powf(0.25);
                let f2 = g * (1.0 - (f1 / g).powi(2));
                [f1, f2]
            }
            _ => unreachable!("Zdt::new rejects which outside {{1,2,3,4,6}}"),
        }
    }

    /// ZDT3's g=1 candidate curve `phi(f1) = h(f1, 1) = 1 - sqrt(f1) - f1
    /// sin(10 pi f1)` -- shared by [`Zdt::eval_one`] (at `g=1`),
    /// [`Zdt::zdt3_segments`], and tests.
    fn zdt3_phi(f1: f64) -> f64 { 1.0 - f1.sqrt() - f1 * (10.0 * PI * f1).sin() }

    /// ZDT3's disconnected Pareto-optimal segments in `f1`-space, found by
    /// scanning [`Zdt::zdt3_phi`] on a dense grid and tracking its running
    /// minimum -- see the module doc's derivation.
    fn zdt3_segments(grid_n: usize) -> Vec<(f64, f64)> {
        let mut segments = Vec::new();
        let mut running_min = f64::INFINITY;
        let mut seg_start: Option<f64> = None;
        let mut seg_end = 0.0_f64;
        for i in 0..=grid_n {
            let t = i as f64 / grid_n as f64;
            let v = Self::zdt3_phi(t);
            if v < running_min {
                running_min = v;
                if seg_start.is_none() {
                    seg_start = Some(t);
                }
                seg_end = t;
            } else if let Some(s) = seg_start.take() {
                segments.push((s, seg_end));
            }
        }
        if let Some(s) = seg_start {
            segments.push((s, seg_end));
        }
        segments
    }

    /// ZDT6's `f1_min = min_{x1 in [0,1]} f1(x1)`, via a deterministic
    /// ternary search over `x1 in [0, 1/6]` maximizing `q(x1) = exp(-4x1)
    /// sin^6(6 pi x1)` (unimodal there -- see module doc).
    fn zdt6_f1_min() -> f64 {
        let q = |x: f64| (-4.0 * x).exp() * (6.0 * PI * x).sin().powi(6);
        let (mut lo, mut hi) = (0.0_f64, 1.0 / 6.0);
        for _ in 0..100 {
            let m1 = lo + (hi - lo) / 3.0;
            let m2 = hi - (hi - lo) / 3.0;
            if q(m1) < q(m2) {
                lo = m1;
            } else {
                hi = m2;
            }
        }
        let xstar = (lo + hi) / 2.0;
        1.0 - q(xstar)
    }

    /// Split `remaining` extra points among segments proportionally to
    /// `weights` (which sum to 1) using the largest-remainder method, so the
    /// returned counts sum to EXACTLY `remaining`.
    fn distribute_extra(weights: &[f64], remaining: usize) -> Vec<usize> {
        let k = weights.len();
        let raw: Vec<f64> = weights.iter().map(|&w| w * remaining as f64).collect();
        let mut extra: Vec<usize> = raw.iter().map(|&r| r.floor() as usize).collect();
        let assigned: usize = extra.iter().sum();
        let mut leftover = remaining - assigned;
        let mut order: Vec<usize> = (0..k).collect();
        order.sort_by(|&i, &j| {
            let fi = raw[i] - raw[i].floor();
            let fj = raw[j] - raw[j].floor();
            fj.partial_cmp(&fi).unwrap()
        });
        let mut idx = 0;
        while leftover > 0 {
            extra[order[idx % k]] += 1;
            leftover -= 1;
            idx += 1;
        }
        extra
    }

    /// Sample `n` points from ZDT3's disconnected segments, spread
    /// proportionally to each segment's `f1`-length (largest-remainder
    /// allocation), evenly spaced in `f1` within each segment. If `n` is
    /// smaller than the number of segments, a CONSERVATIVE fallback takes
    /// just the start of the first `n` segments (documented, not every
    /// segment represented) rather than fabricating a different scheme.
    fn sample_zdt3(segments: &[(f64, f64)], n: usize) -> Vec<Vec<f64>> {
        let k = segments.len();
        if n < k {
            return segments.iter().take(n).map(|&(a, _)| vec![a, Self::zdt3_phi(a)]).collect();
        }
        let lens: Vec<f64> = segments.iter().map(|&(a, b)| b - a).collect();
        let total_len: f64 = lens.iter().sum();
        let weights: Vec<f64> = lens.iter().map(|&l| l / total_len).collect();
        let extra = Self::distribute_extra(&weights, n - k);
        let mut out = Vec::with_capacity(n);
        for (seg_idx, &(a, b)) in segments.iter().enumerate() {
            let c = 1 + extra[seg_idx];
            for j in 0..c {
                let f1 = if c == 1 { a } else { a + j as f64 * (b - a) / (c - 1) as f64 };
                out.push(vec![f1, Self::zdt3_phi(f1)]);
            }
        }
        out
    }
}

impl MoProblem for Zdt {
    fn space(&self) -> &SearchSpace { &self.space }
    fn n_objectives(&self) -> usize { 2 }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<Vec<f64>> {
        pop.iter()
            .map(|g| {
                let xs: Vec<f64> = if self.which == 4 {
                    match (g.blocks.first(), g.blocks.get(1)) {
                        (Some(BlockValues::Float(x1)), Some(BlockValues::Float(rest))) => {
                            let mut xs = x1.clone();
                            xs.extend_from_slice(rest);
                            xs
                        }
                        _ => return vec![f64::INFINITY; 2],
                    }
                } else {
                    match g.blocks.first() {
                        Some(BlockValues::Float(xs)) => xs.clone(),
                        _ => return vec![f64::INFINITY; 2],
                    }
                };
                let [f1, f2] = self.eval_one(&xs);
                vec![f1, f2]
            })
            .collect()
    }

    fn pareto_front(&self, n: usize) -> Option<Vec<Vec<f64>>> {
        if n == 0 {
            return Some(Vec::new());
        }
        let pts = match self.which {
            1 | 2 | 4 => (0..n)
                .map(|i| {
                    let f1 = if n == 1 { 0.0 } else { i as f64 / (n - 1) as f64 };
                    let f2 = match self.which {
                        1 | 4 => 1.0 - f1.sqrt(),
                        2 => 1.0 - f1 * f1,
                        _ => unreachable!(),
                    };
                    vec![f1, f2]
                })
                .collect(),
            3 => Self::sample_zdt3(&Self::zdt3_segments(1_000_000), n),
            6 => {
                let f1_min = Self::zdt6_f1_min();
                (0..n)
                    .map(|i| {
                        let f1 = if n == 1 { f1_min } else { f1_min + i as f64 * (1.0 - f1_min) / (n - 1) as f64 };
                        vec![f1, 1.0 - f1 * f1]
                    })
                    .collect()
            }
            _ => unreachable!("Zdt::new rejects which outside {{1,2,3,4,6}}"),
        };
        Some(pts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g1(xs: &[f64]) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs.to_vec())] } }
    fn g4(x1: f64, rest: &[f64]) -> Genotype {
        Genotype { blocks: vec![BlockValues::Float(vec![x1]), BlockValues::Float(rest.to_vec())] }
    }

    // ---- construction / errors ----

    #[test]
    fn unknown_which_is_error() {
        for which in [0u32, 5, 7, 100] {
            assert!(
                matches!(Zdt::new(which, 5), Err(ZdtError::UnknownWhich(w)) if w == which),
                "which={which}"
            );
        }
    }

    #[test]
    fn zdt5_excluded_as_unknown() {
        // ZDT5 is the binary-coded deceptive problem, deliberately excluded
        // (module doc): it must surface as UnknownWhich, not silently work.
        assert!(matches!(Zdt::new(5, 11), Err(ZdtError::UnknownWhich(5))));
    }

    #[test]
    fn dim_below_2_is_error() {
        for which in [1u32, 2, 3, 4, 6] {
            assert!(matches!(Zdt::new(which, 1), Err(ZdtError::BadDim)), "which={which}");
            assert!(matches!(Zdt::new(which, 0), Err(ZdtError::BadDim)), "which={which}");
        }
    }

    #[test]
    fn zdt1_2_3_6_space_is_single_unit_float_block() {
        for which in [1u32, 2, 3, 6] {
            let p = Zdt::new(which, 5).unwrap();
            assert_eq!(p.space().blocks(), &[Block::Float { lo: 0.0, hi: 1.0, n: 5 }], "which={which}");
            assert_eq!(p.space().dim(), 5);
        }
    }

    #[test]
    fn zdt4_space_has_mixed_bounds() {
        // The paper: x1 in [0,1], x2,...,xm in [-5,5] -- a genuinely mixed
        // domain, represented as two Float blocks.
        let p = Zdt::new(4, 5).unwrap();
        assert_eq!(
            p.space().blocks(),
            &[Block::Float { lo: 0.0, hi: 1.0, n: 1 }, Block::Float { lo: -5.0, hi: 5.0, n: 4 }]
        );
        assert_eq!(p.space().dim(), 5);
    }

    #[test]
    fn n_objectives_is_2_for_all() {
        for which in [1u32, 2, 3, 4, 6] {
            assert_eq!(Zdt::new(which, 5).unwrap().n_objectives(), 2, "which={which}");
        }
    }

    // ---- ZDT1 hand-computed fixtures ----
    // f1(x1)=x1, g=1+9*sum(x2..xm)/(m-1), h=1-sqrt(f1/g), f2=g*h.

    #[test]
    fn zdt1_fixtures() {
        let p = Zdt::new(1, 3).unwrap();
        // all-zeros: f1=0, g=1+9*0/2=1, f2=1*(1-sqrt(0/1))=1-0=1 -> (0,1)
        assert_eq!(p.evaluate_batch(&[g1(&[0.0, 0.0, 0.0])]), vec![vec![0.0, 1.0]]);
        // x1=1, rest=0: f1=1, g=1, f2=1*(1-sqrt(1/1))=1-1=0 -> (1,0)
        assert_eq!(p.evaluate_batch(&[g1(&[1.0, 0.0, 0.0])]), vec![vec![1.0, 0.0]]);
        // x=[0.4,1,1]: f1=0.4, g=1+9*(1+1)/2=1+9=10, f1/g=0.04, sqrt=0.2,
        // h=1-0.2=0.8, f2=10*0.8=8.0 -> (0.4,8.0)
        let out = p.evaluate_batch(&[g1(&[0.4, 1.0, 1.0])]);
        assert!((out[0][0] - 0.4).abs() < 1e-12 && (out[0][1] - 8.0).abs() < 1e-12, "{out:?}");
    }

    // ---- ZDT2 hand-computed fixtures ----
    // Same f1,g; h=1-(f1/g)^2.

    #[test]
    fn zdt2_fixtures() {
        let p = Zdt::new(2, 3).unwrap();
        // all-zeros: f1=0, g=1, h=1-0^2=1, f2=1 -> (0,1)
        assert_eq!(p.evaluate_batch(&[g1(&[0.0, 0.0, 0.0])]), vec![vec![0.0, 1.0]]);
        // x1=1, rest=0: f1=1, g=1, h=1-1^2=0 -> (1,0)
        assert_eq!(p.evaluate_batch(&[g1(&[1.0, 0.0, 0.0])]), vec![vec![1.0, 0.0]]);
        // x=[0.4,1,1]: g=10 (as ZDT1), f1/g=0.04, (f1/g)^2=0.0016,
        // h=1-0.0016=0.9984, f2=10*0.9984=9.984 -> (0.4,9.984)
        let out = p.evaluate_batch(&[g1(&[0.4, 1.0, 1.0])]);
        assert!((out[0][0] - 0.4).abs() < 1e-12 && (out[0][1] - 9.984).abs() < 1e-9, "{out:?}");
    }

    // ---- ZDT3 hand-computed fixtures ----
    // Same f1,g; h=1-sqrt(f1/g)-(f1/g)sin(10*pi*f1).

    #[test]
    fn zdt3_fixtures() {
        let p = Zdt::new(3, 2).unwrap();
        // all-zeros: f1=0,g=1, sqrt(0)=0, sin term coefficient 0*sin(0)=0,
        // h=1-0-0=1 -> (0,1)
        assert_eq!(p.evaluate_batch(&[g1(&[0.0, 0.0])]), vec![vec![0.0, 1.0]]);
        // x1=1, rest=0: f1=1,g=1, sqrt(1)=1, sin(10*pi*1)=sin(10*pi)=0
        // (10*pi is an exact multiple of pi), h=1-1-1*0=0 -> (1,0)
        let out = p.evaluate_batch(&[g1(&[1.0, 0.0])]);
        assert!((out[0][0] - 1.0).abs() < 1e-12 && out[0][1].abs() < 1e-9, "{out:?}");
        // x=[0.25,0]: f1=0.25,g=1, sqrt(0.25)=0.5, sin(10*pi*0.25)=sin(2.5*pi)
        // = sin(0.5*pi) = 1 (period 2*pi), h=1-0.5-0.25*1=0.25 -> (0.25,0.25)
        let out = p.evaluate_batch(&[g1(&[0.25, 0.0])]);
        assert!((out[0][0] - 0.25).abs() < 1e-12 && (out[0][1] - 0.25).abs() < 1e-9, "{out:?}");
    }

    // ---- ZDT4 hand-computed fixtures ----
    // f1=x1, g=1+10(m-1)+sum(xi^2-10cos(4*pi*xi)), h=1-sqrt(f1/g).

    #[test]
    fn zdt4_fixtures() {
        let p = Zdt::new(4, 3).unwrap();
        // all-zeros: xi=0 -> xi^2-10cos(0)=0-10=-10, sum over 2 terms=-20,
        // g=1+20-20=1, f1=0, h=1-sqrt(0)=1 -> (0,1)
        assert_eq!(p.evaluate_batch(&[g4(0.0, &[0.0, 0.0])]), vec![vec![0.0, 1.0]]);
        // x1=1, rest=0: g=1 (as above), f1=1, h=1-sqrt(1)=0 -> (1,0)
        let out = p.evaluate_batch(&[g4(1.0, &[0.0, 0.0])]);
        assert!((out[0][0] - 1.0).abs() < 1e-12 && out[0][1].abs() < 1e-9, "{out:?}");
        // x1=0.48, x2=x3=1: xi^2-10cos(4*pi*1)=1-10*1=-9 (4*pi is an exact
        // multiple of 2*pi so cos=1), sum over 2 terms=-18,
        // g=1+20-18=3, f1/g=0.48/3=0.16, sqrt=0.4, h=1-0.4=0.6, f2=3*0.6=1.8
        let out = p.evaluate_batch(&[g4(0.48, &[1.0, 1.0])]);
        assert!((out[0][0] - 0.48).abs() < 1e-12 && (out[0][1] - 1.8).abs() < 1e-9, "{out:?}");
    }

    // ---- ZDT6 hand-computed fixtures ----
    // f1=1-exp(-4x1)sin^6(6*pi*x1), g=1+9*(sum(x2..xm)/(m-1))^0.25, h=1-(f1/g)^2.

    #[test]
    fn zdt6_fixtures() {
        let p = Zdt::new(6, 2).unwrap();
        // all-zeros: x1=0 -> exp(0)=1, sin(0)=0 -> sin^6=0, f1=1-0=1;
        // rest=0 -> g=1+9*0^0.25=1; h=1-(1/1)^2=0 -> (1,0)
        let out = p.evaluate_batch(&[g1(&[0.0, 0.0])]);
        assert!((out[0][0] - 1.0).abs() < 1e-12 && out[0][1].abs() < 1e-12, "{out:?}");
        // x1=1, rest=0: sin(6*pi*1)=sin(6*pi)=0 (exact multiple of 2*pi) ->
        // f1=1-exp(-4)*0=1 too (same objective point as all-zeros: f1's not
        // injective), g=1, h=0 -> (1,0)
        let out = p.evaluate_batch(&[g1(&[1.0, 0.0])]);
        assert!((out[0][0] - 1.0).abs() < 1e-12 && out[0][1].abs() < 1e-12, "{out:?}");
        // x1=1/12, x2=0: 6*pi*x1=pi/2, sin=1, sin^6=1, f1=1-exp(-1/3)*1
        //   = 1 - 0.7165313105737893... = 0.28346868942621073...
        // g=1 (rest=0), h=1-f1^2 = 1 - 0.08035550... = 0.91964550211...
        let out = p.evaluate_batch(&[g1(&[1.0 / 12.0, 0.0])]);
        assert!((out[0][0] - 0.283_468_689_426_210_7).abs() < 1e-9, "{out:?}");
        assert!((out[0][1] - 0.919_645_502_114_986_5).abs() < 1e-9, "{out:?}");
    }

    // ---- pareto_front: shape / determinism, all which ----

    #[test]
    fn pareto_front_100_has_correct_shape() {
        for which in [1u32, 2, 3, 4, 6] {
            let p = Zdt::new(which, 6).unwrap();
            let front = p.pareto_front(100).unwrap();
            assert_eq!(front.len(), 100, "which={which}");
            assert!(front.iter().all(|row| row.len() == 2), "which={which}");
        }
    }

    #[test]
    fn pareto_front_is_deterministic() {
        for which in [1u32, 2, 3, 4, 6] {
            let p = Zdt::new(which, 6).unwrap();
            let a = p.pareto_front(100).unwrap();
            let b = p.pareto_front(100).unwrap();
            assert_eq!(a, b, "which={which}: pareto_front must be bit-identical across calls");
        }
    }

    #[test]
    fn pareto_front_monotonic_in_f1_all_functions() {
        for which in [1u32, 2, 3, 4, 6] {
            let p = Zdt::new(which, 6).unwrap();
            let front = p.pareto_front(100).unwrap();
            for w in front.windows(2) {
                assert!(w[1][0] >= w[0][0] - 1e-15, "which={which}: f1 not monotonic: {w:?}");
            }
        }
    }

    // ---- ZDT1: f2 = 1 - sqrt(f1), f1 in [0,1] ----

    #[test]
    fn zdt1_pareto_front_shape_and_spots() {
        let p = Zdt::new(1, 6).unwrap();
        let front = p.pareto_front(100).unwrap();
        // first: f1=0/(99)=0, f2=1-sqrt(0)=1
        assert!((front[0][0] - 0.0).abs() < 1e-12 && (front[0][1] - 1.0).abs() < 1e-12);
        // last: f1=99/99=1, f2=1-sqrt(1)=0
        assert!((front[99][0] - 1.0).abs() < 1e-12 && front[99][1].abs() < 1e-12);
        // interior i=33: f1=33/99=1/3, f2=1-sqrt(1/3)=1-0.5773502691896258
        //   =0.4226497308103742
        assert!((front[33][0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((front[33][1] - 0.422_649_730_810_374_2).abs() < 1e-9);
        for row in &front {
            assert!((row[1] - (1.0 - row[0].sqrt())).abs() < 1e-12, "{row:?}");
        }
    }

    // ---- ZDT2: f2 = 1 - f1^2, f1 in [0,1] ----

    #[test]
    fn zdt2_pareto_front_shape_and_spots() {
        let p = Zdt::new(2, 6).unwrap();
        let front = p.pareto_front(100).unwrap();
        assert!((front[0][0] - 0.0).abs() < 1e-12 && (front[0][1] - 1.0).abs() < 1e-12);
        assert!((front[99][0] - 1.0).abs() < 1e-12 && front[99][1].abs() < 1e-12);
        // interior i=33: f1=1/3, f2=1-(1/3)^2=1-1/9=8/9=0.8888888888888888
        assert!((front[33][0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((front[33][1] - 8.0 / 9.0).abs() < 1e-12);
        for row in &front {
            assert!((row[1] - (1.0 - row[0] * row[0])).abs() < 1e-12, "{row:?}");
        }
    }

    // ---- ZDT4: same front shape as ZDT1 (f2 = 1 - sqrt(f1)) ----

    #[test]
    fn zdt4_pareto_front_shape_matches_zdt1() {
        let p = Zdt::new(4, 6).unwrap();
        let front = p.pareto_front(100).unwrap();
        assert!((front[0][0] - 0.0).abs() < 1e-12 && (front[0][1] - 1.0).abs() < 1e-12);
        assert!((front[99][0] - 1.0).abs() < 1e-12 && front[99][1].abs() < 1e-12);
        assert!((front[33][0] - 1.0 / 3.0).abs() < 1e-12);
        assert!((front[33][1] - 0.422_649_730_810_374_2).abs() < 1e-9);
        for row in &front {
            assert!((row[1] - (1.0 - row[0].sqrt())).abs() < 1e-12, "{row:?}");
        }
    }

    // ---- ZDT3: disconnected front, phi(f1)=1-sqrt(f1)-f1*sin(10*pi*f1) ----

    #[test]
    fn zdt3_pareto_front_first_point_and_shape_relation() {
        let p = Zdt::new(3, 6).unwrap();
        let front = p.pareto_front(100).unwrap();
        // The very first candidate (t=0) is trivially non-dominated (no
        // smaller f1 exists), independent of grid resolution: exact (0,1).
        assert!((front[0][0] - 0.0).abs() < 1e-15 && (front[0][1] - 1.0).abs() < 1e-15);
        // Every returned point must lie exactly on the g=1 candidate curve.
        for row in &front {
            let expect = 1.0 - row[0].sqrt() - row[0] * (10.0 * PI * row[0]).sin();
            assert!((row[1] - expect).abs() < 1e-9, "{row:?}");
        }
    }

    #[test]
    fn zdt3_pareto_front_is_genuinely_disconnected() {
        // Proves the "several noncontiguous convex parts" property: a
        // sampled front spanning multiple segments must contain at least
        // one large jump in f1 between consecutive points (a segment gap),
        // not just fine-grained uniform coverage of [0,1].
        let p = Zdt::new(3, 6).unwrap();
        let front = p.pareto_front(100).unwrap();
        let max_gap = front.windows(2).map(|w| w[1][0] - w[0][0]).fold(0.0_f64, f64::max);
        assert!(max_gap > 0.05, "expected a segment gap > 0.05, got {max_gap}");
    }

    #[test]
    fn zdt3_segments_are_five_and_ordered() {
        let segments = Zdt::zdt3_segments(1_000_000);
        assert_eq!(segments.len(), 5, "{segments:?}");
        for &(a, b) in &segments {
            assert!(a < b, "{segments:?}");
        }
        for w in segments.windows(2) {
            assert!(w[0].1 < w[1].0, "segments must be ordered and disjoint: {segments:?}");
        }
        // First segment starts exactly at f1=0 (trivially non-dominated).
        assert_eq!(segments[0].0, 0.0);
    }

    // ---- ZDT6: f2 = 1 - f1^2 over f1 in [f1_min, 1], f1_min > 0 ----

    #[test]
    fn zdt6_pareto_front_does_not_start_at_zero() {
        let p = Zdt::new(6, 6).unwrap();
        let front = p.pareto_front(100).unwrap();
        // Independently verified (module doc): f1_min ~= 0.28077531881537,
        // strictly positive -- the front does NOT start at f1=0.
        assert!((front[0][0] - 0.280_775_318_815_37).abs() < 1e-6, "{:?}", front[0]);
        assert!(front[0][0] > 0.2, "front must not start at f1=0: {:?}", front[0]);
        // last: f1=1 exactly (fixed upper bound, independent of f1_min),
        // f2=1-1^2=0
        assert!((front[99][0] - 1.0).abs() < 1e-12 && front[99][1].abs() < 1e-12);
        for row in &front {
            assert!((row[1] - (1.0 - row[0] * row[0])).abs() < 1e-12, "{row:?}");
        }
    }

    #[test]
    fn zdt6_f1_min_matches_independent_derivation() {
        // tan(6*pi*x)=9*pi root and ternary search independently agree
        // (see module doc) at ~0.28077531881537.
        let f1_min = Zdt::zdt6_f1_min();
        assert!((f1_min - 0.280_775_318_815_37).abs() < 1e-9, "{f1_min}");
    }

    // ---- finite / general sanity across the whole suite ----

    #[test]
    fn all_which_evaluate_to_finite_values() {
        for which in [1u32, 2, 3, 4, 6] {
            let p = Zdt::new(which, 6).unwrap();
            let pop = if which == 4 {
                vec![g4(0.3, &[1.0, -2.0, 4.5, -5.0, 5.0]), g4(0.9, &[0.0, 0.0, 0.0, 0.0, 0.0])]
            } else {
                vec![g1(&[0.3, 0.1, 0.9, 0.5, 0.2, 0.7]), g1(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])]
            };
            let out = p.evaluate_batch(&pop);
            assert!(out.iter().all(|row| row.iter().all(|v| v.is_finite())), "which={which}: {out:?}");
        }
    }
}
