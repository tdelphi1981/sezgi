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
//! **T5 is `Zdt5`, a separate type**: T5 is a deceptive, BINARY-coded
//! problem (`xi` a bit string) that does not fit `Zdt`'s real-coded
//! `Block::Float` representation, so it is NOT a `Zdt::new(which, dim)`
//! variant; it is implemented below as its own type, [`Zdt5`], in this same
//! module. See [`Zdt5`]'s own doc for its provenance (Definition 4, quoted
//! verbatim) and design.
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
         scope for this real-coded suite -- use `Zdt5::new()` instead), got {0}"
    )]
    UnknownWhich(u32),
    #[error("dim must be >= 2")]
    BadDim,
}

/// One instance of the ZDT multi-objective suite (`which` selects ZDT1,
/// ZDT2, ZDT3, ZDT4, or ZDT6 -- ZDT5 is [`Zdt5`], a separate type; see the
/// module doc).
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

/// ZDT5, introduced as **T5** in Zitzler, Deb & Thiele (2000) -- Definition
/// 4, Equation 11, p.178 of the same PROVENANCE-verified source as this
/// module's other functions (sha256 `4b7cd99b01a4cc6143c5edce5f38d08d2ca65e\
/// 46ac3f4dc5f7424a297031fdc0`; see the module doc's header). Quoted
/// verbatim:
///
/// > The test function T5 describes a deceptive problem and distinguishes
/// > itself from the other test functions in that xi represents a binary
/// > string:
/// >
/// > f1(x1) = 1 + u(x1)
/// > g(x2,...,xm) = sum_{i=2}^{m} v(u(xi))
/// > h(f1,g) = 1/f1
/// >
/// > where u(xi) gives the number of ones in the bit vector xi
/// > (unitation),
/// >
/// > v(u(xi)) = { 2 + u(xi)  if u(xi) < 5 }
/// >            { 1          if u(xi) = 5 }
/// >
/// > and m = 11, x1 in {0,1}^30, and x2,...,xm in {0,1}^5. The true
/// > Pareto-optimal front is formed with g(x) = 10, while the best
/// > deceptive Pareto-optimal front is represented by the solutions for
/// > which g(x) = 11. The global Pareto-optimal front as well as the local
/// > ones are convex.
///
/// Composing per the shared scheme (module doc's Equation 6:
/// `f2(x) = g(...) h(f1, g(...))`): `h(f1,g) = 1/f1` (NOT `1/g`, and not a
/// function of `g` at all -- verified directly from the quoted Eq. 11), so
/// `f2 = g * h = g / f1`. Cross-checked bit-for-bit against pymoo 0.6.2's
/// `pymoo.problems.multi.zdt.ZDT5` (`normalize=False`) on pinned bitstrings
/// (all-zeros, all-ones, and several mixed patterns): exact agreement (see
/// this task's report for the transcript).
///
/// ## Layout
///
/// `m = 11`: one `Block::Binary { n: 30 }` for `x1`, then ten
/// `Block::Binary { n: 5 }` blocks for `x2,...,x11` -- 11 Binary blocks, 80
/// bits total, fixed by the paper (unlike `Zdt::new`'s explicit `dim`,
/// there is no free dimension parameter here, so `Zdt5::new()` takes none).
///
/// ## Analytic Pareto front (objective space)
///
/// The global front is `g(x) = 10` (every one of the ten 5-bit groups at
/// its OWN global optimum `u = 5`, `v = 1`), giving `f2 = 10/f1`. `f1 = 1 +
/// u(x1)` with `x1` an independent 30-bit block, so `u(x1)` ranges over the
/// 31 integers `0,...,30` and `f1` over `1,...,31` -- **the front is
/// GENUINELY DISCRETE**: exactly 31 distinct objective-space points, not a
/// continuous curve sampled at `n` locations like T1-T4/T6.
///
/// // sezgi decision: `pareto_front(n)` returns `min(n, 31)` points (never
/// // fabricating points beyond the 31 that exist), evenly spread across
/// // the 31 discrete `f1` values by rounding `i * 30 / (count - 1)` to the
/// // nearest integer index (`count >= 2`); `n == 1` returns the single
/// // point at `f1 = 1`; `n == 0` returns empty, matching `Zdt`'s
/// // convention. Because the spacing between consecutive exact index
/// // values is always `>= 1` for `count <= 31`, the rounded indices are
/// // provably strictly increasing (no duplicate/skipped points): if
/// // `x2 - x1 >= 1` then `round(x2) != round(x1)`, since two values
/// // rounding to the same integer must be within `1` of each other, with
/// // equality possible only at a `.5/.5` tie split by the two adjacent
/// // integers. For `n > 31`, the extra points cannot be manufactured
/// // without duplicating a discrete point, so they are simply not
/// // produced -- callers must not assume `pareto_front(n).len() == n` for
/// // this type (documented on [`Zdt5::pareto_front`] below), unlike every
/// // other function in this module.
///
/// The paper also names a DECEPTIVE local front at `g(x) = 11`: the
/// per-group landscape `v` is minimized at `u = 5` (`v = 1`, the true
/// optimum) but has a local optimum at `u = 0` (`v = 2`, better than
/// `v = 3,4,5,6` at `u = 1,2,3,4`, which is why it is "deceptive" --
/// bit-flip hill-climbing within a group tends to walk DOWN to `u = 0`
/// rather than climb all the way up to `u = 5`). The smallest `g` value
/// strictly above the true optimum `10` replaces exactly one group's `v=1`
/// with the next-smallest available value `v=2` (`u=0`), giving `g = 9*1 +
/// 2 = 11` -- matching the paper's stated `g(x) = 11` exactly. This module
/// does not special-case that front (it is not exposed via
/// `pareto_front`); it is exercised by this file's tests to prove `g=11`
/// points are dominated by `g=10` points at the same `f1`, and are
/// correctly excluded from `pareto_front`'s output.
#[derive(Debug, Clone)]
pub struct Zdt5 {
    space: SearchSpace,
}

impl Default for Zdt5 {
    fn default() -> Self { Self::new() }
}

impl Zdt5 {
    /// Fixed layout (module/type doc): no parameters, since the paper pins
    /// `m=11`, `x1` at 30 bits, and `x2,...,x11` at 5 bits each. Infallible:
    /// `SearchSpace::new` only ever rejects `Block::Float`/`Block::Int`
    /// blocks with `lo >= hi` (`crates/core/src/space.rs`), and `Binary`
    /// blocks carry no bounds to violate -- so, unlike `Zdt::new`, there is
    /// no error type here (module doc's "own error enum entry or struct --
    /// mirror the module's conventions" left this construction path with
    /// nothing left to fail on).
    pub fn new() -> Self {
        let mut blocks = vec![Block::Binary { n: 30 }];
        blocks.extend(std::iter::repeat_n(Block::Binary { n: 5 }, 10));
        let space =
            SearchSpace::new(blocks).expect("ZDT5's fixed Binary blocks carry no bounds to violate");
        Self { space }
    }

    /// `u(x)`: the unitation (count of `true`/one bits) of one bit vector,
    /// per the quoted Definition 4.
    fn unitation(bits: &[bool]) -> u32 { bits.iter().filter(|&&b| b).count() as u32 }

    /// `v(u)`, quoted verbatim above: `2 + u` if `u < 5`, else `1`. Each
    /// group block has exactly 5 bits, so `u` is always in `0..=5` here;
    /// the paper's `u(xi) = 5` branch and "otherwise" are the same thing on
    /// this fixed 5-bit domain, so `else` (not a redundant `u == 5` guard)
    /// is the faithful, total translation.
    fn v(u: u32) -> f64 { if u < 5 { 2.0 + u as f64 } else { 1.0 } }

    /// `f1 = 1 + u(x1)`, `g = sum_i v(u(xi))`, `h = 1/f1`, `f2 = g*h = g/f1`
    /// (module/type doc's Eq. 6 composition, verified against Eq. 11).
    fn eval_one(x1: &[bool], groups: &[Vec<bool>]) -> [f64; 2] {
        let f1 = 1.0 + Self::unitation(x1) as f64;
        let g: f64 = groups.iter().map(|grp| Self::v(Self::unitation(grp))).sum();
        [f1, g / f1]
    }
}

impl MoProblem for Zdt5 {
    fn space(&self) -> &SearchSpace { &self.space }
    fn n_objectives(&self) -> usize { 2 }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<Vec<f64>> {
        pop.iter()
            .map(|g| match g.blocks.split_first() {
                Some((BlockValues::Bin(x1), rest))
                    if rest.len() == 10 && rest.iter().all(|b| matches!(b, BlockValues::Bin(_))) =>
                {
                    let groups: Vec<Vec<bool>> = rest
                        .iter()
                        .map(|b| match b {
                            BlockValues::Bin(bits) => bits.clone(),
                            _ => unreachable!("checked by the outer guard"),
                        })
                        .collect();
                    let [f1, f2] = Self::eval_one(x1, &groups);
                    vec![f1, f2]
                }
                _ => vec![f64::INFINITY; 2],
            })
            .collect()
    }

    /// See the type doc's `// sezgi decision:` -- returns `min(n, 31)`
    /// points on the discrete global front (`g=10`, `f2 = 10/f1`,
    /// `f1 in {1,...,31}`); `pareto_front(n).len() == n` does NOT hold for
    /// `n > 31`, unlike every other function in this module.
    fn pareto_front(&self, n: usize) -> Option<Vec<Vec<f64>>> {
        if n == 0 {
            return Some(Vec::new());
        }
        const TOTAL: usize = 31; // u(x1) in 0..=30 -> f1 in 1..=31
        let count = n.min(TOTAL);
        let indices: Vec<usize> = if count == 1 {
            vec![0]
        } else {
            (0..count)
                .map(|i| ((i * (TOTAL - 1)) as f64 / (count - 1) as f64).round() as usize)
                .collect()
        };
        Some(
            indices
                .into_iter()
                .map(|u1| {
                    let f1 = 1.0 + u1 as f64;
                    vec![f1, 10.0 / f1]
                })
                .collect(),
        )
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
        // ZDT5 is the binary-coded deceptive problem: it is NOT a
        // Zdt::new(which, dim) variant (it is the separate `Zdt5` type,
        // below), so `which=5` must still surface as UnknownWhich, not
        // silently work.
        assert!(matches!(Zdt::new(5, 11), Err(ZdtError::UnknownWhich(5))));
    }

    #[test]
    fn unknown_which_error_text_points_at_zdt5() {
        // Before (pre-Zdt5): "...out of scope for this real-coded suite --
        // see this module's doc), got 5".
        // After (this task): "...out of scope for this real-coded suite --
        // use `Zdt5::new()` instead), got 5" -- the error now names the
        // concrete replacement type instead of pointing at prose.
        let Err(err) = Zdt::new(5, 11) else { panic!("Zdt::new(5, _) must be an error") };
        let msg = err.to_string();
        assert!(msg.contains("use `Zdt5::new()` instead"), "{msg}");
        assert!(!msg.contains("see this module's doc"), "{msg}");
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

    // ==================== Zdt5 (T5, binary-coded, deceptive) ====================
    //
    // Provenance: Zitzler/Deb/Thiele 2000, Definition 4, Eq. 11, p.178
    // (sha256 4b7cd99b01a4cc6143c5edce5f38d08d2ca65e46ac3f4dc5f7424a297031f\
    // dc0 -- see this task's report). Cross-checked bit-for-bit against
    // pymoo 0.6.2's `pymoo.problems.multi.zdt.ZDT5(normalize=False)`;
    // fixtures below are annotated with which pymoo case they reproduce.

    fn zdt5_bits(ones: usize, n: usize) -> Vec<bool> {
        let mut v = vec![false; n];
        v[..ones].fill(true);
        v
    }

    fn zdt5_geno(x1: Vec<bool>, groups: Vec<Vec<bool>>) -> Genotype {
        let mut blocks = vec![BlockValues::Bin(x1)];
        blocks.extend(groups.into_iter().map(BlockValues::Bin));
        Genotype { blocks }
    }

    // ---- construction / space shape ----

    #[test]
    fn zdt5_space_is_eleven_binary_blocks_eighty_bits() {
        let p = Zdt5::new();
        let mut expect = vec![Block::Binary { n: 30 }];
        expect.extend(std::iter::repeat_n(Block::Binary { n: 5 }, 10));
        assert_eq!(p.space().blocks(), expect.as_slice());
        assert_eq!(p.space().dim(), 80);
    }

    #[test]
    fn zdt5_n_objectives_is_2() {
        assert_eq!(Zdt5::new().n_objectives(), 2);
    }

    #[test]
    fn zdt5_default_matches_new() {
        // Default is infallible construction of the same fixed layout.
        assert_eq!(Zdt5::default().space().blocks(), Zdt5::new().space().blocks());
    }

    // ---- unitation / v() building blocks ----

    #[test]
    fn zdt5_unitation_counts_ones() {
        assert_eq!(Zdt5::unitation(&[false, false, false, false, false]), 0);
        assert_eq!(Zdt5::unitation(&[true, false, true, false, true]), 3);
        assert_eq!(Zdt5::unitation(&[true, true, true, true, true]), 5);
    }

    #[test]
    fn zdt5_v_boundary_at_u_equals_5() {
        // Definition 4: v(u) = 2+u for u<5, else 1 -- the jump from u=4 to
        // u=5 (6.0 -> 1.0) is the deceptive discontinuity itself.
        assert_eq!(Zdt5::v(0), 2.0);
        assert_eq!(Zdt5::v(1), 3.0);
        assert_eq!(Zdt5::v(2), 4.0);
        assert_eq!(Zdt5::v(3), 5.0);
        assert_eq!(Zdt5::v(4), 6.0, "last point before the deceptive drop");
        assert_eq!(Zdt5::v(5), 1.0, "the true group optimum: a sharp drop, not a continuation of 2+u");
    }

    // ---- whole-genotype hand fixtures (RE-DERIVED from Definition 4; each
    // ---- one also reproduced bit-for-bit by the pymoo 0.6.2 cross-check) ----

    #[test]
    fn zdt5_all_zeros_fixture() {
        // u(x1)=0 -> f1=1+0=1. Every one of the 10 groups has u=0<5 ->
        // v=2+0=2, so g=10*2=20. f2=g/f1=20/1=20. pymoo: [1, 20].
        let p = Zdt5::new();
        let g = zdt5_geno(zdt5_bits(0, 30), vec![zdt5_bits(0, 5); 10]);
        assert_eq!(p.evaluate_batch(&[g]), vec![vec![1.0, 20.0]]);
    }

    #[test]
    fn zdt5_all_ones_fixture() {
        // u(x1)=30 -> f1=1+30=31. Every group has u=5 -> v=1, g=10*1=10.
        // f2=g/f1=10/31. pymoo: [31, 0.32258064...].
        let p = Zdt5::new();
        let g = zdt5_geno(zdt5_bits(30, 30), vec![zdt5_bits(5, 5); 10]);
        let out = p.evaluate_batch(&[g]);
        assert_eq!(out[0][0], 31.0);
        assert!((out[0][1] - 10.0 / 31.0).abs() < 1e-12, "{out:?}");
    }

    #[test]
    fn zdt5_mixed_fixture() {
        // x1 has 7 leading ones (of 30) -> u(x1)=7, f1=1+7=8. Each of the
        // 10 groups is the 5-bit pattern [1,0,1,0,1] -> u=3<5, v=2+3=5, so
        // g=10*5=50. f2=g/f1=50/8=6.25. pymoo "mixed_1": [8, 6.25].
        let p = Zdt5::new();
        let group = vec![true, false, true, false, true];
        let g = zdt5_geno(zdt5_bits(7, 30), vec![group; 10]);
        assert_eq!(p.evaluate_batch(&[g]), vec![vec![8.0, 6.25]]);
    }

    #[test]
    fn zdt5_one_group_at_u4_fixture() {
        // x1 all-zero -> f1=1. One group is [1,1,1,1,0] -> u=4<5, v=6. The
        // other 9 groups are all-zero -> u=0, v=2 each. g=6+9*2=24.
        // f2=24/1=24. pymoo "x1_0_one_group_u4": [1, 24].
        let p = Zdt5::new();
        let mut groups = vec![zdt5_bits(0, 5); 10];
        groups[0] = vec![true, true, true, true, false];
        let g = zdt5_geno(zdt5_bits(0, 30), groups);
        assert_eq!(p.evaluate_batch(&[g]), vec![vec![1.0, 24.0]]);
    }

    #[test]
    fn zdt5_deceptive_g_equals_11_fixture() {
        // x1 all-zero -> f1=1. One group all-zero (u=0, v=2), the other 9
        // groups all-ones (u=5, v=1 each). g=2+9*1=11 -- exactly the
        // paper's stated "best deceptive Pareto-optimal front" value.
        // f2=11/1=11. pymoo "x1_0_9groups_ones_1group_zero": [1, 11].
        let p = Zdt5::new();
        let mut groups = vec![zdt5_bits(5, 5); 10];
        groups[0] = zdt5_bits(0, 5);
        let g = zdt5_geno(zdt5_bits(0, 30), groups);
        assert_eq!(p.evaluate_batch(&[g]), vec![vec![1.0, 11.0]]);
    }

    #[test]
    fn zdt5_malformed_genotype_yields_infinity() {
        // Wrong block shape (not 1 + 10 Bin blocks) must not panic.
        let p = Zdt5::new();
        let bad = Genotype { blocks: vec![BlockValues::Bin(vec![false; 30])] };
        assert_eq!(p.evaluate_batch(&[bad]), vec![vec![f64::INFINITY, f64::INFINITY]]);
    }

    // ---- front membership: g=10 on the front, g=11 dominated-but-excluded ----

    #[test]
    fn zdt5_g10_point_is_on_pareto_front() {
        // f1=8 (x1 has 7 ones), all 10 groups all-ones -> g=10, f2=10/8=1.25.
        let p = Zdt5::new();
        let front = p.pareto_front(31).unwrap();
        assert!(
            front.iter().any(|row| (row[0] - 8.0).abs() < 1e-12 && (row[1] - 1.25).abs() < 1e-12),
            "expected (8, 1.25) on the front: {front:?}"
        );
    }

    #[test]
    fn zdt5_g11_point_is_not_on_front_but_dominates_g12_at_same_f1() {
        use sezgi_components::nsga2::dominates;
        let p = Zdt5::new();

        // Same f1=8 as the g=10 fixture above, but g=11 (one group traded
        // down from u=5,v=1 to u=0,v=2): f2=11/8=1.375.
        let mut g11_groups = vec![zdt5_bits(5, 5); 10];
        g11_groups[0] = zdt5_bits(0, 5);
        let g11 = zdt5_geno(zdt5_bits(7, 30), g11_groups);
        let g11_obj = p.evaluate_batch(&[g11])[0].clone();
        assert_eq!(g11_obj[0], 8.0);
        assert!((g11_obj[1] - 11.0 / 8.0).abs() < 1e-12);

        // Not on the (g=10-only) front.
        let front = p.pareto_front(31).unwrap();
        assert!(
            !front.iter().any(|row| (row[0] - g11_obj[0]).abs() < 1e-9 && (row[1] - g11_obj[1]).abs() < 1e-9),
            "a g=11 point must not appear on the g=10 global front: {g11_obj:?} in {front:?}"
        );
        // But it IS dominated by the true front's same-f1 point (g=10 <
        // g=11 at equal f1).
        let g10_point = vec![8.0, 1.25];
        assert!(dominates(&g10_point, &g11_obj), "g=10 point must dominate the g=11 point at equal f1");

        // And the g=11 point in turn dominates a same-f1 g=12 point (one
        // group at u=1,v=3 instead of u=5,v=1 -> g=9*1+3=12, f2=12/8=1.5).
        let mut g12_groups = vec![zdt5_bits(5, 5); 10];
        g12_groups[0] = zdt5_bits(1, 5);
        let g12 = zdt5_geno(zdt5_bits(7, 30), g12_groups);
        let g12_obj = p.evaluate_batch(&[g12])[0].clone();
        assert_eq!(g12_obj[0], 8.0);
        assert!((g12_obj[1] - 1.5).abs() < 1e-12);
        assert!(dominates(&g11_obj, &g12_obj), "g=11 point must dominate a worse same-f1 g=12 point");
    }

    // ---- pareto_front: discrete-front sampling decision ----

    #[test]
    fn zdt5_pareto_front_31_covers_every_discrete_point_exactly() {
        let p = Zdt5::new();
        let front = p.pareto_front(31).unwrap();
        assert_eq!(front.len(), 31);
        for (i, row) in front.iter().enumerate() {
            let f1 = 1.0 + i as f64;
            assert_eq!(row[0], f1, "index {i}: {front:?}");
            assert_eq!(row[1], 10.0 / f1, "index {i}: {front:?}");
        }
    }

    #[test]
    fn zdt5_pareto_front_caps_at_31_never_fabricating_points() {
        // sezgi decision (type doc): requesting more than the 31 points
        // that genuinely exist returns exactly 31, not `n`.
        let p = Zdt5::new();
        for n in [32usize, 50, 100, 1000] {
            let front = p.pareto_front(n).unwrap();
            assert_eq!(front.len(), 31, "n={n}");
        }
    }

    #[test]
    fn zdt5_pareto_front_small_n_is_strictly_increasing_and_deterministic() {
        let p = Zdt5::new();
        for n in [1usize, 2, 5, 10, 17, 30] {
            let a = p.pareto_front(n).unwrap();
            let b = p.pareto_front(n).unwrap();
            assert_eq!(a, b, "n={n}: pareto_front must be deterministic");
            assert_eq!(a.len(), n.min(31), "n={n}");
            for w in a.windows(2) {
                assert!(w[1][0] > w[0][0], "n={n}: f1 must be strictly increasing: {a:?}");
                assert!((w[1][1] - 10.0 / w[1][0]).abs() < 1e-12);
            }
            // Endpoints always present.
            assert_eq!(a[0], vec![1.0, 10.0], "n={n}: first point must be f1=1");
            if n >= 2 {
                assert_eq!(a[n.min(31) - 1], vec![31.0, 10.0 / 31.0], "n={n}: last point must be f1=31");
            }
        }
    }

    #[test]
    fn zdt5_pareto_front_zero_is_empty() {
        assert_eq!(Zdt5::new().pareto_front(0), Some(Vec::new()));
    }

    // ---- nsga2_run integration: T3's binary path on a deceptive problem ----

    #[test]
    fn nsga2_run_binary_zdt5_front0_g_is_bounded() {
        // A deceptive problem legitimately may converge to the true front
        // (g=10), the paper's named deceptive local front (g=11), or --
        // with a modest budget on an 80-bit deceptive landscape -- some
        // intermediate mix of "escaped to the global per-group optimum
        // u=5" and "trapped at the LOCAL per-group optimum u=0" blocks
        // (recall: within one 5-bit group, v(u) strictly INCREASES from
        // u=0 to u=4 -- 2,3,4,5,6 -- before dropping sharply to 1 at u=5,
        // so ordinary hill-climbing selection pressure pushes every group
        // toward u=0 first; only crossover/mutation luck escapes further
        // up to the true optimum u=5). What must hold, honestly, is that
        // this selection pressure is doing its job at all: front0's g
        // must never regress into the genuinely-bad intermediate zone
        // (u in 1..=4 on a majority of groups), which the deceptive
        // structure makes strictly worse than the u=0 trap itself.
        //
        // Measured (this task's development run, pop=60, budget=9000,
        // p_m_bin defaulting to 1/80): a 10-seed sweep over
        // seed in {1,2,3,5,7,11,13,17,23,42} gave front0 g values (every
        // member of front0 had the SAME g -- only x1, which has no
        // deceptive structure, diversified across f1) of exactly
        // {19, 19, 18, 18, 18, 18, 18, 18, 16, 15} -- i.e. every run
        // pushed g down into [15, 19], comfortably below the "no group
        // stuck at an intermediate value" bound of 20 (10 groups all at
        // the LOCAL optimum u=0, v=2, the worst-case outcome consistent
        // with every group having escaped the genuinely bad u in 1..=4
        // zone). Anchored at g<=25 (headroom above the full observed
        // [15,19] range, well below the theoretical worst 60) with
        // seed=11 (a representative, non-cherry-picked point of that
        // sweep, g=18) for this test's fixed, deterministic assertion.
        use sezgi_components::nsga2::{Nsga2Config, nsga2_run};
        let p = Zdt5::new();
        let cfg = Nsga2Config {
            pop_size: 60,
            budget: 9000,
            seed: 11,
            eta_c: 20.0,
            eta_m: 20.0,
            p_c: 0.9,
            p_m: None,
            p_c_bin: 0.9,
            p_m_bin: None,
        };
        let result = nsga2_run(&p, &cfg).unwrap();
        assert!(!result.front0.is_empty());
        for &i in &result.front0 {
            let f1 = result.objectives[i][0];
            let f2 = result.objectives[i][1];
            let g = f2 * f1; // f2 = g/f1 -> g = f2*f1
            assert!((f1 - f1.round()).abs() < 1e-9, "f1 must be an integer (1+unitation): {f1}");
            assert!(
                g <= 25.0 + 1e-6,
                "front0 member {i} has g={g} (f1={f1}, f2={f2}): worse than the measured/anchored bound"
            );
        }
    }

    #[test]
    fn nsga2_run_binary_zdt5_same_seed_twice_bit_identical() {
        use sezgi_components::nsga2::{Nsga2Config, nsga2_run};
        let p = Zdt5::new();
        let cfg = Nsga2Config {
            pop_size: 20,
            budget: 2000,
            seed: 3,
            eta_c: 20.0,
            eta_m: 20.0,
            p_c: 0.9,
            p_m: None,
            p_c_bin: 0.9,
            p_m_bin: None,
        };
        let r1 = nsga2_run(&p, &cfg).unwrap();
        let r2 = nsga2_run(&p, &cfg).unwrap();
        assert_eq!(r1, r2, "same seed must reproduce a bit-identical MoRunResult");
    }
}
