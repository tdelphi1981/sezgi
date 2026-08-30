//! The DTLZ scalable multi-objective test-problem suite.
//!
//! Source (PROVENANCE, fetched and read directly, not from memory): Kalyanmoy
//! Deb, Lothar Thiele, Marco Laumanns, Eckart Zitzler, "Scalable Test
//! Problems for Evolutionary Multiobjective Optimization", chapter 6 (pp.
//! 105-145) of A. Abraham, R. Jain, R. Goldberg (eds.), *Evolutionary
//! Multiobjective Optimization: Theoretical Advances and Applications*,
//! Springer, 2005. PDF fetched from co-author Eckart Zitzler's own
//! publication page: `https://eckartzitzler.ch/img/publications/dtlz2005a.pdf`.
//!
//! **Numbering artifact pinned**: this chapter (section 6.7, "Test Problem
//! Suite") in fact defines NINE problems, DTLZ1-DTLZ9 (matching the count in
//! the earlier ETH TIK-Report 112 mirror of this same study, contra this
//! task's initial assumption that the report and the chapter differ in
//! problem *count* -- they do not; both define nine). What the chapter's
//! DTLZ1-DTLZ7 share, and DTLZ8/DTLZ9 do not, is the "bottom-up" construction
//! (section 6.4: an explicit `g`/cosine-or-linear-cascade objective
//! function, unconstrained, domain `[0,1]^n`) -- DTLZ8 and DTLZ9 instead use
//! the "constraint surface" approach (section 6.5: a hyper-box objective
//! space cut down by explicit inequality constraints `g_j(f) >= 0`, Eq.
//! 6.26-6.27) and so do not fit this module's `MoProblem` surface (no
//! constraint channel) without inventing one. DTLZ1-DTLZ7 is the set THIS
//! MODULE implements (`which: 1..=7`); DTLZ8/DTLZ9 are a documented
//! deferral, exactly mirroring [`crate::zdt`]'s ZDT5 exclusion.
//!
//! ## Shared construction (quoted, section 6.7)
//!
//! Every DTLZ1-DTLZ7 instance splits its `n` decision variables `x` into a
//! "position" group `x_pos = (x1,...,x_{M-1})` (`M` = number of objectives,
//! this module's `m`) and a "distance" group `xM` of the remaining
//! `k = n - M + 1` variables, domain `0 <= xi <= 1` for `i = 1,...,n`
//! throughout (quoted per-problem below). This module's `Dtlz::new(which, m,
//! dim)` takes `dim` (= `n`) explicitly, exactly as [`crate::zdt::Zdt::new`]
//! takes `dim`; it never silently substitutes a default `k`. The chapter's
//! own M=3 illustrative runs state `k` explicitly for DTLZ2 ("with k = 10"),
//! DTLZ3/DTLZ4/DTLZ5 ("using k = 10" / "with... k = 10"), DTLZ6 ("The size
//! of xM vector is chosen as 10"), and DTLZ7 ("For a problem with k = 20").
//! **DTLZ1's own M=3 run text states no explicit k** (unlike the others);
//! `k=5` for DTLZ1 is documented here only as the near-universal
//! reference-implementation convention (pymoo, jMetal, PlatEMO, and the
//! original DTLZ code released by the authors), NOT a chapter quote -- so
//! the widely-used standard-`k` set (`n = M + k - 1`) is: DTLZ1 `k=5`
//! (convention, not chapter-quoted), DTLZ2-DTLZ6 `k=10` (chapter-quoted),
//! DTLZ7 `k=20` (chapter-quoted).
//!
//! ## Per-problem formulas (quoted)
//!
//! - **DTLZ1** (Eq. 6.18-6.19, "an M-objective problem with a linear
//!   Pareto-optimal front"):
//!   ```text
//!   f1(x) = (1/2) x1 x2 ... x_{M-1} (1+g(xM)),
//!   f2(x) = (1/2) x1 x2 ... (1-x_{M-1}) (1+g(xM)),
//!   ...
//!   f_{M-1}(x) = (1/2) x1 (1-x2) (1+g(xM)),
//!   f_M(x) = (1/2) (1-x1) (1+g(xM)),
//!   subject to 0 <= xi <= 1, for i = 1,2,...,n.
//!
//!   g(xM) = 100 [ |xM| + sum_{xi in xM} ((xi-0.5)^2 - cos(20*pi*(xi-0.5))) ]
//!   ```
//!   "The Pareto-optimal solution corresponds to `xi = 0.5` (for all `xi in
//!   xM`) and the objective function values lie on the linear hyper-plane:
//!   `sum_{m=1}^{M} f_m* = 0.5`." The search space additionally contains
//!   `(11^k - 1)` local Pareto-optimal fronts.
//!
//! - **DTLZ2** (Eq. 6.20, "identical to the problem described in Section
//!   6.4.3" -- the generic sphere problem):
//!   ```text
//!   f1(x) = (1+g(xM)) cos(x1*pi/2) ... cos(x_{M-2}*pi/2) cos(x_{M-1}*pi/2),
//!   f2(x) = (1+g(xM)) cos(x1*pi/2) ... cos(x_{M-2}*pi/2) sin(x_{M-1}*pi/2),
//!   f3(x) = (1+g(xM)) cos(x1*pi/2) ... sin(x_{M-2}*pi/2),
//!   ...
//!   fM(x) = (1+g(xM)) sin(x1*pi/2),
//!   with g(xM) = sum_{xi in xM} (xi-0.5)^2, 0 <= xi <= 1, for i=1,...,n.
//!   ```
//!   "The Pareto-optimal solutions corresponds to `xi = 0.5` for all `xi in
//!   xM` and all objective function values must satisfy `sum_{m=1}^{M}
//!   (f_m*)^2 = 1`" (Eq. 6.9): the first-octant unit sphere.
//!
//! - **DTLZ3** (Eq. 6.21, "to investigate an MOEA's ability to converge to
//!   the global Pareto-optimal front"): the SAME cosine-cascade objectives
//!   as DTLZ2, but with DTLZ1's multimodal `g`:
//!   ```text
//!   g(xM) = 100 [ |xM| + sum_{xi in xM} ((xi-0.5)^2 - cos(20*pi*(xi-0.5))) ]
//!   ```
//!   "introduces `(3^k - 1)` local Pareto-optimal fronts and one global
//!   Pareto-optimal front... The global Pareto-optimal front corresponds to
//!   `xi = 0.5`" -- same sphere front as DTLZ2.
//!
//! - **DTLZ4** (Eq. 6.22, "to maintain a good distribution of solutions"):
//!   the SAME cosine-cascade objectives and the SAME `g` as DTLZ2, but with
//!   every `xi` in the trig arguments raised to a bias power `alpha`:
//!   ```text
//!   f1(x) = (1+g(xM)) cos(x1^alpha*pi/2) ... cos(x_{M-1}^alpha*pi/2), etc.
//!   ```
//!   "The parameter `alpha = 100` is suggested here." Front is IDENTICAL to
//!   DTLZ2's (same `g`, same cascade shape; `alpha` only re-biases the
//!   density with which uniform `x` maps onto the front, not the front's
//!   location).
//!
//! - **DTLZ5** (Eq. 6.23, "The mapping of `theta_i` in the test problem
//!   DTLZ2 can be replaced with that given in Equation 6.10"): DTLZ2's
//!   cascade with every `xi` (`i=1,...,M-1`) slot replaced by a `theta_i`:
//!   ```text
//!   f1(x) = (1+g(xM)) cos(theta1*pi/2) ... cos(theta_{M-1}*pi/2),
//!   ...
//!   fM(x) = (1+g(xM)) sin(theta1*pi/2),
//!   with theta_i = pi / (4*(1+g(xM))) * (1 + 2*g(xM)*xi), for i=2,3,...,(M-1),
//!   g(xM) = sum_{xi in xM} (xi-0.5)^2, 0 <= xi <= 1, for i=1,...,n.
//!   ```
//!   (`theta_1` is not redefined here -- section 6.7.5 only overrides
//!   `theta_i` for `i=2,...,(M-1)`, per Eq. 6.10's own index range, leaving
//!   `theta_1 = x1` playing DTLZ2's `x1` role directly, exactly as printed
//!   in the `cos(theta1*pi/2)` term above.) "The Pareto-optimal front
//!   corresponds to `xi = 0.5` for all `xi in xM` and function values
//!   satisfy `sum (f_m*)^2 = 1`." Section 6.4.4 (the construction this
//!   reuses) explains WHY this degenerates to a curve: "Since `g(r) = 0`
//!   corresponds to the Pareto-optimal front, `theta_i = pi/4` for all but
//!   the first variable" -- at `g=0`, `theta_i` (`i>=2`) collapses to the
//!   CONSTANT `pi/4` independent of `xi`, so only `theta_1` (hence `x1`)
//!   remains free: a one-dimensional curve, independent of `M`.
//!
//! - **DTLZ6** (section 6.7.6, "a similar modification to the g function in
//!   DTLZ5... as done in DTLZ3"): identical to DTLZ5 (Eq. 6.23's cascade and
//!   `theta_i`) except:
//!   ```text
//!   g(xM) = sum_{xi in xM} xi^0.1     (Eq. 6.24)
//!   ```
//!   "the Pareto-optimal front corresponds to `xi = 0` for all `xi in xM`."
//!
//! - **DTLZ7** (Eq. 6.25, "a disconnected set of Pareto-optimal regions"):
//!   ```text
//!   f1(x) = x1, f2(x) = x2, ..., f_{M-1}(x) = x_{M-1},
//!   fM(x) = (1+g(xM)) h(f1,...,f_{M-1},g),
//!   where g(xM) = 1 + (9/|xM|) * sum_{xi in xM} xi,
//!   h(f1,...,f_{M-1},g) = M - sum_{i=1}^{M-1} [ (fi/(1+g)) * (1+sin(3*pi*fi)) ],
//!   subject to 0 <= xi <= 1, for i=1,...,n.
//!   ```
//!   "This problem has `2^{M-1}` disconnected Pareto-optimal regions in the
//!   search space... The Pareto-optimal solutions corresponds to `xM = 0`."
//!   Note `g`'s minimum is `g* = 1` (at `xM = 0`), NOT `0` as in every other
//!   DTLZ problem here.
//!
//! ## DTLZ5/DTLZ6 front for `m > 3`: `pareto_front` returns `None`, documented
//!
//! The chapter's own degenerate-curve claim (quoted above) is contradicted
//! by later work for `m > 3`: Hisao Ishibuchi, Hiroyuki Masuda, Yusuke Nojima,
//! "Pareto Fronts of Many-Objective Degenerate Test Problems", *IEEE
//! Transactions on Evolutionary Computation*, 2016 (fetched PDF,
//! TEVC-00187-2015, p.2, section II.B) states verbatim: "As pointed out in
//! [5], [6], [8], some solutions with `g(xM) != 0` are Pareto optimal when
//! `M > 3`. That is, the true Pareto front of DTLZ5 is not a degenerate
//! curve in the case of four or more objectives," and further: "DTLZ6...is
//! the same as DTLZ5 in (2)-(5) except that `g(xM)`...DTLZ5 and DTLZ6 have
//! the same intended degenerate Pareto front..., which was derived from
//! `g(xM) = 0`" -- i.e. the same `m > 3` caveat applies to DTLZ6. This
//! module therefore samples the (correctly degenerate, chapter-verified) `m
//! <= 3` curve for `which in {5,6}` and returns `None` for `m > 3`, per this
//! task's brief: "do not improvise a front."
//!
//! ## Pareto-front sampling scheme (deterministic, `pareto_front(n)`)
//!
//! For DTLZ1 and DTLZ2/3/4 (shared sphere front), the analytic front is
//! parameterized by the SAME `(M-1)`-dimensional `x_pos in [0,1]^{M-1}` (or
//! angle-fraction) box used by the objective formulas themselves, evaluated
//! at `g=0`: a deterministic `r^{M-1}`-point lattice is built by taking `r =
//! ceil(n^{1/(M-1)})` evenly spaced values per axis (`0, 1/(r-1),
//! ..., 1`) and forming their full Cartesian product, so `r^{M-1}` is
//! "approximately `n`" (exact when `n` is a perfect `(M-1)`-th power, e.g.
//! always exact for `m=2`, where `M-1=1` and `r=n`). For DTLZ5/DTLZ6 (`m <=
//! 3` only), the front has exactly one free parameter (`x1`, `theta_1`), so
//! `n` evenly spaced `x1` values give exactly `n` points, with the other
//! `theta_i` fixed at their `g=0` value (computed, not hard-coded, by
//! reusing this module's own `theta_vals` at `g=0`). For DTLZ7, this module
//! DERIVES (does not copy) the disconnection numerically, mirroring
//! [`crate::zdt::Zdt`]'s ZDT3 `zdt3_segments`: since `f1,...,f_{M-1}` are
//! independently free (`g` depends only on `xM`, never on `x_pos`) and `fM`
//! is additively separable in them (`fM = (1+g*) M - sum_i fi(1+sin(3*pi*fi))`
//! at the front's `g* = 1`), coordinate `fi` is front-optimal
//! independently of every other coordinate, iff `T(fi) = fi*(1+sin(3*pi*fi))`
//! is a STRICT RUNNING MAXIMUM as `fi` increases from `0` (smaller `fi` is
//! never dominated by a point with equal-or-smaller `T`). `dtlz7_segments`
//! finds these running-maximum segments on a dense grid (1,000,000 cells);
//! it empirically finds exactly 2 segments per coordinate, so the FULL
//! `(M-1)`-dimensional front is the Cartesian product of these per-axis
//! segment-samples across all `M-1` coordinates -- `2^{M-1}` blocks,
//! independently confirming the chapter's stated count. `r =
//! ceil(n^{1/(M-1)})` points are drawn per axis (largest-remainder
//! allocation across segments, mirroring `Zdt::sample_zdt3`), so the
//! returned front again has "approximately `n`" points (exact for `m=2`).

use std::f64::consts::PI;

use sezgi_core::mo::MoProblem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

#[derive(Debug, thiserror::Error)]
pub enum DtlzError {
    #[error(
        "which must be one of 1..=7 (DTLZ1-DTLZ7; DTLZ8/DTLZ9 use the constraint-surface \
         construction and are out of scope for this suite -- see this module's doc), got {0}"
    )]
    UnknownWhich(u32),
    #[error("m (number of objectives) must be >= 2, got {0}")]
    BadM(usize),
    #[error("dim must be >= m so that k = dim - m + 1 >= 1 (dim={dim}, m={m})")]
    BadDim { dim: usize, m: usize },
}

/// One instance of the DTLZ scalable multi-objective suite (`which` selects
/// DTLZ1,...,DTLZ7 -- DTLZ8/DTLZ9 are excluded, see the module doc).
pub struct Dtlz {
    which: u32,
    m: usize,
    dim: usize,
    space: SearchSpace,
}

impl Dtlz {
    /// `which` in `1..=7`, `m >= 2` (number of objectives), `dim >= m`
    /// (so the distance-group size `k = dim - m + 1 >= 1`).
    pub fn new(which: u32, m: usize, dim: usize) -> Result<Self, DtlzError> {
        if !matches!(which, 1..=7) {
            return Err(DtlzError::UnknownWhich(which));
        }
        if m < 2 {
            return Err(DtlzError::BadM(m));
        }
        if dim < m {
            return Err(DtlzError::BadDim { dim, m });
        }
        let space = SearchSpace::new(vec![Block::Float { lo: 0.0, hi: 1.0, n: dim }])
            .expect("DTLZ bounds (0<1) are always valid");
        Ok(Self { which, m, dim, space })
    }

    pub fn which(&self) -> u32 { self.which }
    pub fn m(&self) -> usize { self.m }
    pub fn dim(&self) -> usize { self.dim }

    // ---- shared g-functions (module doc, per-problem formulas) ----

    fn g_value(which: u32, x_tail: &[f64]) -> f64 {
        let k = x_tail.len() as f64;
        match which {
            1 | 3 => {
                100.0
                    * (k + x_tail
                        .iter()
                        .map(|&xi| (xi - 0.5).powi(2) - (20.0 * PI * (xi - 0.5)).cos())
                        .sum::<f64>())
            }
            2 | 4 | 5 => x_tail.iter().map(|&xi| (xi - 0.5).powi(2)).sum(),
            6 => x_tail.iter().map(|&xi| xi.powf(0.1)).sum(),
            7 => 1.0 + 9.0 / k * x_tail.iter().sum::<f64>(),
            _ => unreachable!("Dtlz::new rejects which outside 1..=7"),
        }
    }

    /// DTLZ1's linear `f1,...,fM` cascade (Eq. 6.18), shared by
    /// [`Self::eval_one`] and [`Self::pareto_front`] (at `g=0`).
    fn dtlz1_f(x_pos: &[f64], g: f64) -> Vec<f64> {
        let n_vals = x_pos.len();
        let m = n_vals + 1;
        (0..m)
            .map(|j| {
                let mut val = 0.5 * (1.0 + g);
                if j + 2 <= m {
                    let upper = m - 2 - j;
                    for &xi in &x_pos[0..=upper] {
                        val *= xi;
                    }
                }
                if j > 0 {
                    val *= 1.0 - x_pos[m - 1 - j];
                }
                val
            })
            .collect()
    }

    /// The DTLZ2/3/4/5/6 cosine-cascade `f1,...,fM` (Eq. 6.20), shared by
    /// [`Self::eval_one`] and [`Self::pareto_front`]: `vals` plays the
    /// `x1,...,x_{M-1}` (DTLZ2/3), `x1^alpha,...` (DTLZ4), or
    /// `theta1,...,theta_{M-1}` (DTLZ5/6) role, each multiplied by `pi/2`
    /// inside `cos`/`sin`, exactly as printed in every one of those
    /// equations.
    fn cosine_cascade(vals: &[f64], g: f64) -> Vec<f64> {
        let n_vals = vals.len();
        let m = n_vals + 1;
        (0..m)
            .map(|j| {
                let mut val = 1.0 + g;
                if j + 2 <= m {
                    let upper = m - 2 - j;
                    for &v in &vals[0..=upper] {
                        val *= (v * PI / 2.0).cos();
                    }
                }
                if j > 0 {
                    val *= (vals[m - 1 - j] * PI / 2.0).sin();
                }
                val
            })
            .collect()
    }

    /// DTLZ5/DTLZ6's `theta_i` mapping (Eq. 6.23/6.10): `theta_1 = x1`
    /// unchanged; `theta_i` for `i=2,...,(M-1)` (0-indexed `1..len-1`)
    /// replaced per the quoted formula. At `g=0` every `theta_i` (`i>=2`)
    /// collapses to the constant `pi/4`, independent of `xi` -- the
    /// degenerate-curve property (module doc).
    fn theta_vals(x_pos: &[f64], g: f64) -> Vec<f64> {
        let mut theta = x_pos.to_vec();
        for idx in 1..theta.len() {
            theta[idx] = PI / (4.0 * (1.0 + g)) * (1.0 + 2.0 * g * x_pos[idx]);
        }
        theta
    }

    /// DTLZ7's `f1,...,fM` (Eq. 6.25): `x_pos` plays `f1,...,f_{M-1}`
    /// directly; shared by [`Self::eval_one`] and [`Self::pareto_front`].
    fn dtlz7_f(x_pos: &[f64], g: f64) -> Vec<f64> {
        let m = x_pos.len() + 1;
        let mut f = x_pos.to_vec();
        let h = m as f64
            - x_pos.iter().map(|&fi| fi / (1.0 + g) * (1.0 + (3.0 * PI * fi).sin())).sum::<f64>();
        f.push((1.0 + g) * h);
        f
    }

    fn eval_one(&self, xs: &[f64]) -> Vec<f64> {
        let x_pos = &xs[..self.m - 1];
        let x_tail = &xs[self.m - 1..];
        let g = Self::g_value(self.which, x_tail);
        match self.which {
            1 => Self::dtlz1_f(x_pos, g),
            2 | 3 => Self::cosine_cascade(x_pos, g),
            4 => {
                let alpha: Vec<f64> = x_pos.iter().map(|&x| x.powf(100.0)).collect();
                Self::cosine_cascade(&alpha, g)
            }
            5 | 6 => Self::cosine_cascade(&Self::theta_vals(x_pos, g), g),
            7 => Self::dtlz7_f(x_pos, g),
            _ => unreachable!("Dtlz::new rejects which outside 1..=7"),
        }
    }

    // ---- deterministic front-sampling lattice (module doc scheme) ----

    fn axis_grid(r: usize) -> Vec<f64> {
        if r <= 1 { vec![0.0] } else { (0..r).map(|i| i as f64 / (r - 1) as f64).collect() }
    }

    fn grid_r(n: usize, dim_free: usize) -> usize {
        ((n.max(1) as f64).powf(1.0 / dim_free as f64)).ceil().max(1.0) as usize
    }

    fn cartesian_product(axis: &[f64], dim: usize) -> Vec<Vec<f64>> {
        let mut out = vec![Vec::new()];
        for _ in 0..dim {
            let mut next = Vec::with_capacity(out.len() * axis.len());
            for combo in &out {
                for &v in axis {
                    let mut c = combo.clone();
                    c.push(v);
                    next.push(c);
                }
            }
            out = next;
        }
        out
    }

    /// DTLZ7's per-coordinate optimality curve `T(f) = f*(1+sin(3*pi*f))`,
    /// `f in [0,1]` (module doc derivation).
    fn dtlz7_t(f: f64) -> f64 { f * (1.0 + (3.0 * PI * f).sin()) }

    /// DTLZ7's disconnected per-coordinate segments, found by scanning
    /// [`Self::dtlz7_t`] on a dense grid and tracking its running maximum
    /// (module doc derivation, mirrors [`crate::zdt::Zdt::zdt3_segments`]).
    fn dtlz7_segments(grid_n: usize) -> Vec<(f64, f64)> {
        let mut segments = Vec::new();
        let mut running_max = f64::NEG_INFINITY;
        let mut seg_start: Option<f64> = None;
        let mut seg_end = 0.0_f64;
        for i in 0..=grid_n {
            let t = i as f64 / grid_n as f64;
            let v = Self::dtlz7_t(t);
            if v > running_max {
                running_max = v;
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

    /// Split `remaining` extra points among segments proportionally to
    /// `weights` (which sum to 1) using the largest-remainder method, so the
    /// returned counts sum to EXACTLY `remaining` (mirrors
    /// [`crate::zdt::Zdt`]'s internal `distribute_extra`).
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

    /// Sample `r` values from DTLZ7's disconnected segments (one coordinate
    /// axis), spread proportionally to each segment's length, evenly spaced
    /// within each segment (mirrors `Zdt::sample_zdt3`, but returns bare
    /// `f`-values since DTLZ7's per-coordinate curve is only used to build
    /// the Cartesian-product front, not plotted on its own).
    fn sample_dtlz7_axis(segments: &[(f64, f64)], r: usize) -> Vec<f64> {
        let k = segments.len();
        if r < k {
            return segments.iter().take(r).map(|&(a, _)| a).collect();
        }
        let lens: Vec<f64> = segments.iter().map(|&(a, b)| b - a).collect();
        let total_len: f64 = lens.iter().sum();
        let weights: Vec<f64> = lens.iter().map(|&l| l / total_len).collect();
        let extra = Self::distribute_extra(&weights, r - k);
        let mut out = Vec::with_capacity(r);
        for (seg_idx, &(a, b)) in segments.iter().enumerate() {
            let c = 1 + extra[seg_idx];
            for j in 0..c {
                out.push(if c == 1 { a } else { a + j as f64 * (b - a) / (c - 1) as f64 });
            }
        }
        out
    }
}

impl MoProblem for Dtlz {
    fn space(&self) -> &SearchSpace { &self.space }
    fn n_objectives(&self) -> usize { self.m }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<Vec<f64>> {
        pop.iter()
            .map(|g| match g.blocks.first() {
                Some(BlockValues::Float(xs)) => self.eval_one(xs),
                _ => vec![f64::INFINITY; self.m],
            })
            .collect()
    }

    fn pareto_front(&self, n: usize) -> Option<Vec<Vec<f64>>> {
        if n == 0 {
            return Some(Vec::new());
        }
        let dim_free = self.m - 1;
        match self.which {
            1 => {
                let r = Self::grid_r(n, dim_free);
                let combos = Self::cartesian_product(&Self::axis_grid(r), dim_free);
                Some(combos.iter().map(|c| Self::dtlz1_f(c, 0.0)).collect())
            }
            2..=4 => {
                let r = Self::grid_r(n, dim_free);
                let combos = Self::cartesian_product(&Self::axis_grid(r), dim_free);
                Some(combos.iter().map(|c| Self::cosine_cascade(c, 0.0)).collect())
            }
            5 | 6 => {
                // Degenerate curve only verified for m <= 3 (module doc).
                if self.m > 3 {
                    return None;
                }
                let axis = Self::axis_grid(n);
                Some(
                    axis.iter()
                        .map(|&x1| {
                            let mut x_pos = vec![0.0; dim_free];
                            x_pos[0] = x1;
                            let theta = Self::theta_vals(&x_pos, 0.0);
                            Self::cosine_cascade(&theta, 0.0)
                        })
                        .collect(),
                )
            }
            7 => {
                let segments = Self::dtlz7_segments(1_000_000);
                let r = Self::grid_r(n, dim_free);
                let axis = Self::sample_dtlz7_axis(&segments, r);
                let combos = Self::cartesian_product(&axis, dim_free);
                Some(combos.iter().map(|c| Self::dtlz7_f(c, 1.0)).collect())
            }
            _ => unreachable!("Dtlz::new rejects which outside 1..=7"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g1(xs: &[f64]) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs.to_vec())] } }

    // ---- construction / errors ----

    #[test]
    fn unknown_which_is_error() {
        for which in [0u32, 8, 9, 100] {
            assert!(
                matches!(Dtlz::new(which, 3, 10), Err(DtlzError::UnknownWhich(w)) if w == which),
                "which={which}"
            );
        }
    }

    #[test]
    fn m_below_2_is_error() {
        for which in 1u32..=7 {
            assert!(matches!(Dtlz::new(which, 1, 10), Err(DtlzError::BadM(1))), "which={which}");
            assert!(matches!(Dtlz::new(which, 0, 10), Err(DtlzError::BadM(0))), "which={which}");
        }
    }

    #[test]
    fn dim_below_m_is_error() {
        for which in 1u32..=7 {
            assert!(
                matches!(Dtlz::new(which, 5, 4), Err(DtlzError::BadDim { dim: 4, m: 5 })),
                "which={which}"
            );
        }
    }

    #[test]
    fn dim_equal_m_is_ok() {
        // k = dim - m + 1 = 1: minimal but valid.
        for which in 1u32..=7 {
            assert!(Dtlz::new(which, 4, 4).is_ok(), "which={which}");
        }
    }

    // ---- scalability: m=2 and m=3, standard k, all which ----

    #[test]
    fn space_and_dim_correct_for_m2_and_m3() {
        // Standard k per module doc: DTLZ1 k=5, DTLZ2-6 k=10, DTLZ7 k=20.
        for which in 1u32..=7 {
            let k = match which {
                1 => 5,
                7 => 20,
                _ => 10,
            };
            for m in [2usize, 3] {
                let dim = m + k - 1;
                let p = Dtlz::new(which, m, dim).unwrap();
                assert_eq!(p.space().blocks(), &[Block::Float { lo: 0.0, hi: 1.0, n: dim }]);
                assert_eq!(p.space().dim(), dim);
                assert_eq!(p.n_objectives(), m, "which={which} m={m}");
            }
        }
    }

    #[test]
    fn evaluate_batch_inner_lengths_match_m() {
        for which in 1u32..=7 {
            for m in [2usize, 3] {
                let dim = m + 9; // k=10, always >= any which's minimal k
                let p = Dtlz::new(which, m, dim).unwrap();
                let xs = vec![0.3; dim];
                let out = p.evaluate_batch(&[g1(&xs)]);
                assert_eq!(out.len(), 1, "which={which} m={m}");
                assert_eq!(out[0].len(), m, "which={which} m={m}");
                assert!(out[0].iter().all(|v| v.is_finite()), "which={which} m={m}: {out:?}");
            }
        }
    }

    // ---- DTLZ1 hand-computed fixtures (g=0 slice: all xM = 0.5) ----
    // f_j formula (Eq. 6.18): full product for f1, product-then-flip for
    // interior f_j, flip-only for fM.

    #[test]
    fn dtlz1_fixtures_g_zero() {
        let p = Dtlz::new(1, 3, 5).unwrap(); // m=3, k=2
        // x_pos=[0,0], xM=[0.5,0.5]: g=100*[2 + 2*(0-cos(0))] = 100*[2-2]=0.
        // f1=0.5*1*0*0=0, f2=0.5*1*0*(1-0)=0, f3=0.5*1*(1-0)=0.5 -> (0,0,0.5)
        let out = p.evaluate_batch(&[g1(&[0.0, 0.0, 0.5, 0.5])]);
        assert!(
            (out[0][0]).abs() < 1e-12 && (out[0][1]).abs() < 1e-12 && (out[0][2] - 0.5).abs() < 1e-12,
            "{out:?}"
        );
        // x_pos=[1,1], xM=[0.5,0.5]: g=0 (same as above).
        // f1=0.5*1*1*1=0.5, f2=0.5*1*1*(1-1)=0, f3=0.5*1*(1-1)=0 -> (0.5,0,0)
        let out = p.evaluate_batch(&[g1(&[1.0, 1.0, 0.5, 0.5])]);
        assert!(
            (out[0][0] - 0.5).abs() < 1e-12 && out[0][1].abs() < 1e-12 && out[0][2].abs() < 1e-12,
            "{out:?}"
        );
        // sum-of-objectives invariant holds at ANY x_pos when g=0: 0.5.
        let row = &p.evaluate_batch(&[g1(&[0.3, 0.7, 0.5, 0.5])])[0];
        let sum: f64 = row.iter().sum();
        assert!((sum - 0.5).abs() < 1e-12, "{row:?}");
    }

    #[test]
    fn dtlz1_fixture_nonzero_g() {
        let p = Dtlz::new(1, 2, 3).unwrap(); // m=2, k=2
        // x_pos=[0.5], xM=[0,0]: per xi=0, (0-0.5)^2 - cos(20*pi*(-0.5))
        //   = 0.25 - cos(-10*pi) = 0.25 - 1 = -0.75 (10*pi exact multiple of
        //   2*pi so cos=1); sum over 2 terms = -1.5; g=100*(2-1.5)=50.
        // f1=0.5*(1+50)*0.5 = 0.5*51*0.5 = 12.75
        // f2=0.5*(1+50)*(1-0.5) = 12.75
        // sum = 0.5*(1+g) = 25.5, matching the general sum-invariant.
        let out = p.evaluate_batch(&[g1(&[0.5, 0.0, 0.0])]);
        assert!((out[0][0] - 12.75).abs() < 1e-9, "{out:?}");
        assert!((out[0][1] - 12.75).abs() < 1e-9, "{out:?}");
        assert!(((out[0][0] + out[0][1]) - 25.5).abs() < 1e-9, "{out:?}");
    }

    // ---- DTLZ2 hand-computed fixtures (g=0 slice) ----

    #[test]
    fn dtlz2_fixtures_g_zero() {
        let p = Dtlz::new(2, 3, 5).unwrap(); // m=3, k=2
        // x=(0,0,0.5,0.5): g=0. angle=0 -> cos=1,sin=0.
        // f1=cos(0)*cos(0)=1, f2=cos(0)*sin(0)=0, f3=sin(0)=0 -> (1,0,0)
        let out = p.evaluate_batch(&[g1(&[0.0, 0.0, 0.5, 0.5])]);
        assert!(
            (out[0][0] - 1.0).abs() < 1e-12 && out[0][1].abs() < 1e-12 && out[0][2].abs() < 1e-12,
            "{out:?}"
        );
        // x=(0.5,0.5,0.5,0.5): g=0, angle=pi/4 for both -> cos=sin=sqrt(2)/2.
        // f1=cos^2(pi/4)=0.5, f2=cos(pi/4)*sin(pi/4)=0.5,
        // f3=sin(pi/4)=sqrt(2)/2=0.7071067811865476
        let out = p.evaluate_batch(&[g1(&[0.5, 0.5, 0.5, 0.5])]);
        assert!((out[0][0] - 0.5).abs() < 1e-12, "{out:?}");
        assert!((out[0][1] - 0.5).abs() < 1e-12, "{out:?}");
        assert!((out[0][2] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12, "{out:?}");
        // Sphere invariant.
        let s: f64 = out[0].iter().map(|v| v * v).sum();
        assert!((s - 1.0).abs() < 1e-12, "{out:?}");
    }

    // ---- DTLZ3: same cascade as DTLZ2, DTLZ1's multimodal g ----

    #[test]
    fn dtlz3_fixture_nonzero_g_matches_dtlz1_g_formula() {
        let p1 = Dtlz::new(1, 2, 3).unwrap();
        let p3 = Dtlz::new(3, 2, 3).unwrap();
        // Both use the SAME g at xM=[0,0] (g=50, from dtlz1_fixture_nonzero_g).
        // DTLZ3 at x1=0.5: angle=pi/4, f1=cos(pi/4)*(1+50), f2=sin(pi/4)*(1+50).
        let out1 = p1.evaluate_batch(&[g1(&[0.5, 0.0, 0.0])]);
        let out3 = p3.evaluate_batch(&[g1(&[0.5, 0.0, 0.0])]);
        let g = 50.0_f64;
        let expect_f1 = (1.0 + g) * (std::f64::consts::PI / 4.0).cos();
        let expect_f2 = (1.0 + g) * (std::f64::consts::PI / 4.0).sin();
        assert!((out3[0][0] - expect_f1).abs() < 1e-9, "{out3:?}");
        assert!((out3[0][1] - expect_f2).abs() < 1e-9, "{out3:?}");
        // Sanity: DTLZ1's own value at the same xM differs (linear, not cosine).
        assert!((out1[0][0] - 12.75).abs() < 1e-9);
    }

    // ---- DTLZ4: alpha=100 bias, x=0.5 vs x != 0.5 ----

    #[test]
    fn dtlz4_alpha_bias_at_half_vs_shifted() {
        let p2 = Dtlz::new(2, 2, 3).unwrap();
        let p4 = Dtlz::new(4, 2, 3).unwrap();
        // At x1=0.5, xM=[0.5,0.5] (g=0): DTLZ2 uses angle=0.5*pi/2=pi/4
        // directly, giving (sqrt(2)/2, sqrt(2)/2). DTLZ4 instead raises
        // x1 to alpha=100 first: 0.5^100 ~= 7.89e-31, so its angle is
        // ~1.24e-30 (indistinguishable from 0 in f64): f1 ~= cos(0) = 1,
        // f2 ~= sin(~0) ~= 0 -- density pulled hard away from the middle.
        let out2 = p2.evaluate_batch(&[g1(&[0.5, 0.5, 0.5])]);
        let out4 = p4.evaluate_batch(&[g1(&[0.5, 0.5, 0.5])]);
        assert!((out2[0][0] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12, "{out2:?}");
        assert!((out4[0][0] - 1.0).abs() < 1e-9, "{out4:?}");
        assert!(out4[0][1].abs() < 1e-9, "{out4:?}");
        // DTLZ4 recovers DTLZ2's x1=0.5 point when fed x1 = 0.5^(1/100)
        // (chosen so x1^100 = 0.5 exactly), demonstrating alpha only
        // reparameterizes the density, not the front's location.
        let x1_pre_bias = 0.5_f64.powf(1.0 / 100.0);
        let out4b = p4.evaluate_batch(&[g1(&[x1_pre_bias, 0.5, 0.5])]);
        assert!((out4b[0][0] - out2[0][0]).abs() < 1e-9, "{out4b:?} vs {out2:?}");
        assert!((out4b[0][1] - out2[0][1]).abs() < 1e-9, "{out4b:?} vs {out2:?}");
    }

    // ---- DTLZ5: theta transform hand point, m=3 ----

    #[test]
    fn dtlz5_theta_transform_fixture_m3() {
        let p = Dtlz::new(5, 3, 5).unwrap(); // m=3, k=2
        // x_pos=[0, x2] (x2 irrelevant: theta2 depends only on g, x2's own
        // coefficient vanishes when g=0), xM=[0.5,0.5] -> g=0.
        // theta1=x1=0 -> angle1=0 -> cos=1,sin=0.
        // theta2 = pi/(4*(1+0))*(1+0) = pi/4 -> angle2 = (pi/4)*(pi/2) = pi^2/8.
        let out = p.evaluate_batch(&[g1(&[0.0, 0.9, 0.5, 0.5])]);
        let angle2 = (PI / 4.0) * (PI / 2.0);
        assert!((out[0][0] - angle2.cos()).abs() < 1e-9, "{out:?}");
        assert!((out[0][1] - angle2.sin()).abs() < 1e-9, "{out:?}");
        assert!(out[0][2].abs() < 1e-12, "{out:?}");
        // x2 truly doesn't matter at g=0: a different x2 gives the same row.
        let out_b = p.evaluate_batch(&[g1(&[0.0, 0.1, 0.5, 0.5])]);
        assert_eq!(out, out_b);
        // Sphere invariant.
        let s: f64 = out[0].iter().map(|v| v * v).sum();
        assert!((s - 1.0).abs() < 1e-9, "{out:?}");
    }

    // ---- DTLZ6: same cascade as DTLZ5, g = sum xi^0.1 (front at xi=0) ----

    #[test]
    fn dtlz6_fixture_nonzero_g() {
        let p = Dtlz::new(6, 3, 5).unwrap(); // m=3, k=2
        // xM=[1,0]: g = 1^0.1 + 0^0.1 = 1 + 0 = 1 (front is at xi=0, so this
        // is a non-zero-g, off-front point).
        // x_pos=[0, 0.5]: theta1=0 -> angle1=0.
        // theta2 = pi/(4*(1+1))*(1+2*1*0.5) = (pi/8)*(2) = pi/4
        //   -> angle2 = (pi/4)*(pi/2) = pi^2/8.
        // f1=(1+1)*cos(0)*cos(angle2)=2*cos(pi^2/8)
        // f2=(1+1)*cos(0)*sin(angle2)=2*sin(pi^2/8)
        // f3=(1+1)*sin(0)=0
        let out = p.evaluate_batch(&[g1(&[0.0, 0.5, 1.0, 0.0])]);
        let angle2 = (PI / 4.0) * (PI / 2.0);
        assert!((out[0][0] - 2.0 * angle2.cos()).abs() < 1e-9, "{out:?}");
        assert!((out[0][1] - 2.0 * angle2.sin()).abs() < 1e-9, "{out:?}");
        assert!(out[0][2].abs() < 1e-9, "{out:?}");
    }

    // ---- DTLZ7: h/disconnection arithmetic at one point ----

    #[test]
    fn dtlz7_h_fixture() {
        let p = Dtlz::new(7, 3, 5).unwrap(); // m=3, k=3
        // x_pos=[1/6, 0] (f1=1/6, f2=0), xM=[0,0,0]: g=1+9/3*0=1.
        // h = 3 - [ (1/6)/(1+1) * (1+sin(3*pi/6)) + 0/(1+1)*(1+sin(0)) ]
        //   = 3 - [ (1/12)*(1+sin(pi/2)) + 0 ] = 3 - [ (1/12)*2 ] = 3 - 1/6
        //   = 17/6
        // f3 = (1+1)*h = 2*17/6 = 17/3 = 5.666666...
        let out = p.evaluate_batch(&[g1(&[1.0 / 6.0, 0.0, 0.0, 0.0, 0.0])]);
        assert!((out[0][0] - 1.0 / 6.0).abs() < 1e-12, "{out:?}");
        assert!(out[0][1].abs() < 1e-12, "{out:?}");
        assert!((out[0][2] - 17.0 / 3.0).abs() < 1e-9, "{out:?}");
    }

    // ---- front samples: shape, determinism, characterizing relations ----

    #[test]
    fn dtlz1_front_rows_sum_to_half() {
        for m in [2usize, 3] {
            let p = Dtlz::new(1, m, m + 4).unwrap();
            let front = p.pareto_front(100).unwrap();
            assert!(!front.is_empty(), "m={m}");
            for row in &front {
                assert_eq!(row.len(), m, "m={m}");
                let sum: f64 = row.iter().sum();
                assert!((sum - 0.5).abs() < 1e-9, "m={m}: {row:?}");
            }
        }
    }

    #[test]
    fn dtlz2_3_4_front_rows_are_unit_norm() {
        for which in [2u32, 3, 4] {
            for m in [2usize, 3] {
                let p = Dtlz::new(which, m, m + 9).unwrap();
                let front = p.pareto_front(100).unwrap();
                assert!(!front.is_empty(), "which={which} m={m}");
                for row in &front {
                    let s: f64 = row.iter().map(|v| v * v).sum();
                    assert!((s - 1.0).abs() < 1e-9, "which={which} m={m}: {row:?}");
                }
            }
        }
    }

    #[test]
    fn dtlz1_front_exact_n_for_m2() {
        let p = Dtlz::new(1, 2, 6).unwrap();
        let front = p.pareto_front(50).unwrap();
        assert_eq!(front.len(), 50);
    }

    #[test]
    fn dtlz2_front_exact_n_for_m2_and_m3() {
        let p2 = Dtlz::new(2, 2, 11).unwrap();
        assert_eq!(p2.pareto_front(100).unwrap().len(), 100);
        let p3 = Dtlz::new(2, 3, 12).unwrap();
        // r = ceil(sqrt(100)) = 10 -> r^2 = 100 exactly.
        assert_eq!(p3.pareto_front(100).unwrap().len(), 100);
    }

    #[test]
    fn dtlz5_6_front_none_for_m_gt_3() {
        for which in [5u32, 6] {
            let p = Dtlz::new(which, 4, 13).unwrap();
            assert!(p.pareto_front(50).is_none(), "which={which}");
        }
    }

    #[test]
    fn dtlz5_6_front_sampled_and_sphere_for_m_le_3() {
        for which in [5u32, 6] {
            for m in [2usize, 3] {
                let p = Dtlz::new(which, m, m + 9).unwrap();
                let front = p.pareto_front(50).unwrap();
                assert_eq!(front.len(), 50, "which={which} m={m}");
                for row in &front {
                    let s: f64 = row.iter().map(|v| v * v).sum();
                    assert!((s - 1.0).abs() < 1e-9, "which={which} m={m}: {row:?}");
                }
            }
        }
    }

    #[test]
    fn dtlz7_front_rows_satisfy_h_relation() {
        for m in [2usize, 3] {
            let p = Dtlz::new(7, m, m + 19).unwrap();
            let front = p.pareto_front(64).unwrap();
            assert!(!front.is_empty(), "m={m}");
            for row in &front {
                assert_eq!(row.len(), m, "m={m}");
                let (f_pos, f_m) = row.split_at(m - 1);
                let h = m as f64
                    - f_pos.iter().map(|&fi| fi / 2.0 * (1.0 + (3.0 * PI * fi).sin())).sum::<f64>();
                // g* = 1 at the front, so fM = (1+1)*h = 2*h.
                assert!((f_m[0] - 2.0 * h).abs() < 1e-9, "m={m}: {row:?}");
            }
        }
    }

    #[test]
    fn dtlz7_segments_per_coordinate_are_two() {
        // Independently confirms the chapter's "2^(M-1) disconnected
        // regions" claim (module doc): 2 running-maximum segments per axis.
        let segments = Dtlz::dtlz7_segments(1_000_000);
        assert_eq!(segments.len(), 2, "{segments:?}");
        for &(a, b) in &segments {
            assert!(a < b, "{segments:?}");
        }
        assert_eq!(segments[0].0, 0.0, "first segment starts at f=0 (trivially non-dominated)");
    }

    #[test]
    fn pareto_front_is_deterministic_all_which() {
        for which in 1u32..=7 {
            let p = Dtlz::new(which, 3, 12).unwrap();
            let a = p.pareto_front(50).unwrap();
            let b = p.pareto_front(50).unwrap();
            assert_eq!(a, b, "which={which}: pareto_front must be bit-identical across calls");
        }
    }

    #[test]
    fn pareto_front_zero_n_is_empty_for_all_which() {
        for which in 1u32..=7 {
            let p = Dtlz::new(which, 3, 12).unwrap();
            assert_eq!(p.pareto_front(0), Some(Vec::new()), "which={which}");
        }
    }
}
