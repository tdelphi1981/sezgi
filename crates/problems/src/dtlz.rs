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
//! 6.26-6.27). M3-7 Task 1 (`sezgi_core::mo::MoProblem::evaluate_constraints_batch`,
//! `docs/DECISIONS.md`'s M3-7 record) added the constraint channel this
//! surface needs; M3-7 Task 2 (this module's current form) wires DTLZ8 and
//! DTLZ9 into it, so this module now implements the FULL DTLZ1-DTLZ9 suite
//! (`which: 1..=9`) -- no DTLZ deferral remains (unlike [`crate::zdt`]'s
//! still-standing ZDT5 exclusion, a different suite with a different
//! reason).
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
//! - **DTLZ8** (Eq. 6.26, "the constraint surface approach"; quoted
//!   verbatim, PDF p.136):
//!   ```text
//!   Minimize  fj(x) = (1/floor(n/M)) sum_{i=floor((j-1)n/M)}^{floor(jn/M)} xi,
//!             j = 1, 2, ..., M,
//!   Subject to gj(x) = fM(x) + 4fj(x) - 1 >= 0, for j = 1, 2, ..., (M-1),
//!             gM(x) = 2fM(x) + min_{i,j=1,...,(M-1); i!=j} [fi(x)+fj(x)] - 1 >= 0,
//!             0 <= xi <= 1, for i = 1, 2, ..., n.
//!   ```
//!   "Here, the number of variables is considered to be larger than the
//!   number of objectives or n > M. We suggest n = 10M. In this problem,
//!   there are a total of M constraints. The Pareto-optimal front is a
//!   combination of a straight line and a hyper-plane. The straight line is
//!   the intersection of the first (M-1) constraints (with f1 = f2 = ... =
//!   f_{M-1}) and the hyper-plane is represented by the constraint gM."
//!
//!   **Block-partition convention** (`fj`'s sum bounds, verified against a
//!   second source): `x` (length `n`) splits into `M` contiguous blocks at
//!   the HALF-OPEN boundaries `[floor((j-1)n/M), floor(jn/M))` (0-indexed),
//!   block `j` (`j=1,...,M`) -- the printed
//!   `sum_{i=floor((j-1)n/M)}^{floor(jn/M)}` reads as inclusive of both ends
//!   only typographically; the half-open reading is the one that makes
//!   `floor(n/M)` (printed as a SINGLE constant, not re-derived per block)
//!   an actual, constant per-block count (exact whenever `M` divides `n`,
//!   which the chapter's own `n=10M` suggestion always satisfies -- an
//!   inclusive-both-ends reading would instead double-count every block
//!   boundary and could never match a constant `floor(n/M)` divisor except
//!   by coincidence). This is independently confirmed by ParMOO 0.5.1's own
//!   DTLZ8/DTLZ9 simulation code (`parmoo/simulations/dtlz.py`,
//!   `dtlz8_sim.__call__`/`dtlz9_sim.__call__`: `start = i * self.n //
//!   self.o; stop = (i + 1) * self.n // self.o`, i.e. the SAME half-open
//!   floor-division block boundaries, Python's `//` being floor division)
//!   -- see the task-2 report for the full cross-check (ParMOO does NOT
//!   ship the chapter's own normalization [`1/floor(n/M)`] or ANY
//!   constraint formulas for DTLZ8/DTLZ9 in this installed version -- its
//!   `dtlz8_sim`/`dtlz9_sim` compute a different, offset-parameterized
//!   quantity (`sum(|x-offset|)` / `sum(|x-offset|^0.1)`, no division) and
//!   its docstring-referenced `parmoo.constraints.dtlz` module does not
//!   exist in 0.5.1; only the block-BOUNDARY convention is corroborated,
//!   not the objective/constraint VALUES). The LAST block (`j=M`) absorbs
//!   any remainder (`floor(Mn/M) = n` exactly), so it may contain more than
//!   `floor(n/M)` variables when `M` does not divide `n` -- still divided
//!   by the SAME constant `floor(n/M)`, per the equation's single
//!   `1/floor(n/M)` factor printed OUTSIDE the sum.
//!
//!   `// sezgi decision:` **the `M=2` `gM` corner.** `gM`'s `min` ranges
//!   over PAIRS `i != j` drawn from `{1,...,(M-1)}` (quoted above) -- for
//!   `M=2` that index set is the SINGLETON `{1}`, which contains no `i !=
//!   j` pair, so the chapter's own formula is undefined at `M=2` (never
//!   discussed in the text; every worked example in the chapter uses
//!   `M=3`). This module follows the standard "min of an empty set is
//!   `+infinity`" convention: at `M=2`, `gM`'s min term is
//!   `f64::INFINITY`, so `gM = 2fM(x) + infinity - 1` is unconditionally
//!   `>= 0` (vacuously satisfied) -- `evaluate_constraints_batch` still
//!   returns exactly `M` columns (the dimension-count contract every
//!   constrained consumer relies on), `gM` simply never constrains
//!   anything when `M=2`. This could not be cross-checked against ParMOO
//!   (no constraint code ships for DTLZ8/DTLZ9 in the installed 0.5.1); it
//!   is this module's own reasoned extension of the chapter's stated
//!   formula to a case the chapter itself does not address.
//!
//! - **DTLZ9** (Eq. 6.27, "also created using the constraint surface
//!   approach"; quoted verbatim, PDF p.137):
//!   ```text
//!   Minimize  fj(x) = sum_{i=floor((j-1)n/M)}^{floor(jn/M)} xi^0.1,
//!             j = 1, 2, ..., M,
//!   Subject to gj(x) = fM(x)^2 + fj(x)^2 - 1 >= 0, for j = 1, 2, ..., (M-1),
//!             0 <= xi <= 1, for i = 1, 2, ..., n.
//!   ```
//!   "Here too, the number of variables is considered to be larger than the
//!   number of objectives. For this problem, we also suggest n = 10M. The
//!   Pareto-optimal front is a curve with f1 = f2 = ... = f_{M-1}, similar
//!   to that in DTLZ5. However, the density of solutions gets thinner
//!   towards the Pareto-optimal region. The Pareto-optimal curve lies on
//!   the intersection of all (M-1) constraints... A two-dimensional plot of
//!   the Pareto-optimal front with fM and any other objective function
//!   should represent a circular arc of radius one. A plot with any two
//!   objective functions except fM should show a 45 degree straight line."
//!
//!   Same block-partition convention as DTLZ8 above -- but `fj` here has NO
//!   `1/floor(n/M)` normalizing factor at all (printed as a bare sum of
//!   `xi^0.1`, unlike DTLZ8's averaged `fj`; quoted exactly as printed).
//!   `(M-1)` constraints total (no `gM` analogue -- DTLZ9's own constraint
//!   list stops at `j=(M-1)`, per the equation above; no `M=2` corner
//!   exists for DTLZ9 since it never defines a `gM`).
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
//!
//! For DTLZ9 (`m <= 3` only -- `// sezgi decision:` extending the SAME
//! caution as DTLZ5/DTLZ6's own documented `m > 3` caveat above, by
//! structural analogy: the chapter's own DTLZ9 text says its front is
//! "similar to that in DTLZ5", and this module already documents that
//! DTLZ5's degenerate-curve claim is contradicted for `m > 3`; no
//! DTLZ9-specific `m > 3` source was found either way, so this is a
//! conservative, unverified-for-DTLZ9-itself extension of an established
//! caution, not a chapter- or literature-confirmed claim about DTLZ9), the
//! front is the one-parameter curve `f1 = f2 = ... = f_{M-1} = c`, `fM =
//! sqrt(1 - c^2)` (from every `gj = fM^2 + fj^2 - 1 = 0` simultaneously:
//! `M-1` equal-radius-one circle equations force every `fj` for `j < M`
//! equal), `c` ranging over `n` evenly spaced points in `[0,1]` (mirrors
//! DTLZ5/DTLZ6's own single-free-parameter sampling immediately above) --
//! `m > 3` returns `None`.
//!
//! DTLZ8's front is a genuine union of two differently-shaped pieces (a
//! line AND a hyper-plane region, quoted above) -- the hyper-plane piece's
//! closed form is NOT a simple flat-plane equation in the unordered
//! objective coordinates (the `min` over unordered pairs makes it
//! piecewise: one flat piece per choice of "the two smallest of
//! `f1,...,f_{M-1}`", `(M-1) choose 2` such pieces glued into the
//! "triangular plane" the chapter describes) and the chapter gives no
//! closed form for enumerating or sampling that union. Per this task's
//! brief ("implement sampling ONLY where the chapter gives you a
//! defensible closed form; otherwise return `None`... do not force it"),
//! DTLZ8's `pareto_front` returns `None` unconditionally, with this note
//! standing in for a sampling scheme.

use std::f64::consts::PI;

use sezgi_core::mo::MoProblem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

#[derive(Debug, thiserror::Error)]
pub enum DtlzError {
    #[error(
        "which must be one of 1..=9 (DTLZ1-DTLZ9; DTLZ8/DTLZ9 use the constraint-surface \
         construction, see this module's doc), got {0}"
    )]
    UnknownWhich(u32),
    #[error("m (number of objectives) must be >= 2, got {0}")]
    BadM(usize),
    #[error("dim must be >= m so that k = dim - m + 1 >= 1 (dim={dim}, m={m})")]
    BadDim { dim: usize, m: usize },
    #[error(
        "dim must be > m for DTLZ8/DTLZ9 (the constraint-surface construction requires n > M \
         per the chapter's own Eq. 6.26/6.27; dim={dim}, m={m})"
    )]
    BadDimConstraintSurface { dim: usize, m: usize },
}

/// One instance of the DTLZ scalable multi-objective suite (`which` selects
/// DTLZ1,...,DTLZ9; DTLZ8/DTLZ9 use the constraint-surface construction, see
/// the module doc).
pub struct Dtlz {
    which: u32,
    m: usize,
    dim: usize,
    space: SearchSpace,
}

impl Dtlz {
    /// `which` in `1..=9`, `m >= 2` (number of objectives), `dim >= m` (so
    /// the distance-group size `k = dim - m + 1 >= 1`) -- except DTLZ8/DTLZ9
    /// (`which` 8 or 9), which require the STRICTER `dim > m` (chapter's
    /// own "n > M", Eq. 6.26/6.27; `dim == m` is rejected via
    /// [`DtlzError::BadDimConstraintSurface`], a check on top of, not
    /// instead of, the shared `dim >= m` check below).
    pub fn new(which: u32, m: usize, dim: usize) -> Result<Self, DtlzError> {
        if !matches!(which, 1..=9) {
            return Err(DtlzError::UnknownWhich(which));
        }
        if m < 2 {
            return Err(DtlzError::BadM(m));
        }
        if dim < m {
            return Err(DtlzError::BadDim { dim, m });
        }
        if matches!(which, 8 | 9) && dim <= m {
            return Err(DtlzError::BadDimConstraintSurface { dim, m });
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
            _ => unreachable!(
                "g_value is only called from eval_one's which-in-1..=7 branch \
                 (DTLZ8/DTLZ9 use dtlz89_blocks instead, no shared g)"
            ),
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

    // ---- DTLZ8/DTLZ9 constraint-surface functions (Eq. 6.26/6.27) ----

    /// Shared block-partition construction for DTLZ8 (`normalize = true`,
    /// dividing by the constant `floor(n/M)`, Eq. 6.26) and DTLZ9
    /// (`normalize = false`, a bare sum, Eq. 6.27) -- see the module doc's
    /// "Block-partition convention". `xs` (length `n`) splits into `m`
    /// contiguous HALF-OPEN blocks `[j*n/m, (j+1)*n/m)` (integer division,
    /// 0-indexed `j`), the convention independently confirmed against
    /// ParMOO 0.5.1's `dtlz8_sim`/`dtlz9_sim` (module doc). `transform` is
    /// the identity for DTLZ8's raw `xi` or `xi.powf(0.1)` for DTLZ9.
    fn dtlz89_blocks(xs: &[f64], m: usize, normalize: bool, transform: impl Fn(f64) -> f64) -> Vec<f64> {
        let n = xs.len();
        let divisor = if normalize { (n / m) as f64 } else { 1.0 };
        (0..m)
            .map(|j| {
                let lo = j * n / m;
                let hi = (j + 1) * n / m;
                xs[lo..hi].iter().map(|&x| transform(x)).sum::<f64>() / divisor
            })
            .collect()
    }

    /// DTLZ8's `M` constraints (Eq. 6.26), given this instance's own
    /// objective row `f` (length `M`): `g_j = fM + 4*fj - 1` for
    /// `j=1,...,(M-1)` (0-indexed `0..m-1`), then `gM = 2*fM +
    /// min_{i!=j;i,j<(M-1)}(fi+fj) - 1`. At `M=2` the inner double loop
    /// below never finds an `i != j` pair (the 0-indexed range `0..m-1` is
    /// the single-element `{0}`), so `min_pair` stays `f64::INFINITY` --
    /// this IS the module doc's `// sezgi decision:` "empty min = +infinity"
    /// ruling for the `M=2` `gM` corner, expressed as the natural fallout
    /// of the loop rather than a separate special case.
    fn dtlz8_constraints(f: &[f64]) -> Vec<f64> {
        let m = f.len();
        let f_m = f[m - 1];
        let mut g: Vec<f64> = (0..m - 1).map(|j| f_m + 4.0 * f[j] - 1.0).collect();
        let mut min_pair = f64::INFINITY;
        for i in 0..m - 1 {
            for j in 0..m - 1 {
                if i != j {
                    min_pair = min_pair.min(f[i] + f[j]);
                }
            }
        }
        g.push(2.0 * f_m + min_pair - 1.0);
        g
    }

    /// DTLZ9's `(M-1)` constraints (Eq. 6.27): `g_j = fM^2 + fj^2 - 1` for
    /// `j=1,...,(M-1)` (0-indexed `0..m-1`). No `gM` analogue (module doc).
    fn dtlz9_constraints(f: &[f64]) -> Vec<f64> {
        let m = f.len();
        let f_m = f[m - 1];
        (0..m - 1).map(|j| f_m.powi(2) + f[j].powi(2) - 1.0).collect()
    }

    fn eval_one(&self, xs: &[f64]) -> Vec<f64> {
        match self.which {
            8 => Self::dtlz89_blocks(xs, self.m, true, |x| x),
            9 => Self::dtlz89_blocks(xs, self.m, false, |x| x.powf(0.1)),
            _ => {
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
                    _ => unreachable!("Dtlz::new rejects which outside 1..=9"),
                }
            }
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

    /// `Some` only for DTLZ8/DTLZ9 (`which` 8 or 9) -- every other `which`
    /// inherits the trait default `None` (module doc, "no DTLZ deferral
    /// remains" -- but DTLZ1-7 remain UNCONSTRAINED problems, exactly as
    /// before this task; only DTLZ8/DTLZ9 use the constraint channel).
    fn evaluate_constraints_batch(&self, pop: &[Genotype]) -> Option<Vec<Vec<f64>>> {
        if !matches!(self.which, 8 | 9) {
            return None;
        }
        let n_con = if self.which == 8 { self.m } else { self.m - 1 };
        Some(
            pop.iter()
                .map(|g| match g.blocks.first() {
                    Some(BlockValues::Float(xs)) => {
                        let f = self.eval_one(xs);
                        if self.which == 8 {
                            Self::dtlz8_constraints(&f)
                        } else {
                            Self::dtlz9_constraints(&f)
                        }
                    }
                    _ => vec![f64::NEG_INFINITY; n_con],
                })
                .collect(),
        )
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
            // DTLZ8: no defensible closed form for the line+hyperplane
            // union (module doc's sampling-scheme section; brief: "do not
            // force it").
            8 => None,
            9 => {
                // Same m<=3 caution as DTLZ5/DTLZ6, by documented analogy
                // (module doc's sampling-scheme section).
                if self.m > 3 {
                    return None;
                }
                let axis = Self::axis_grid(n);
                Some(
                    axis.iter()
                        .map(|&c| {
                            let mut row = vec![c; dim_free];
                            row.push((1.0 - c * c).max(0.0).sqrt());
                            row
                        })
                        .collect(),
                )
            }
            _ => unreachable!("Dtlz::new rejects which outside 1..=9"),
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
        // which=8,9 are now valid (this task, M3-7 Task 2) -- replaced by
        // 10/11 in this mechanical update (module doc: DTLZ8/DTLZ9 no
        // longer "out of scope for this suite").
        for which in [0u32, 10, 11, 100] {
            assert!(
                matches!(Dtlz::new(which, 3, 10), Err(DtlzError::UnknownWhich(w)) if w == which),
                "which={which}"
            );
        }
    }

    #[test]
    fn unknown_which_error_text_mentions_1_to_9_not_out_of_scope() {
        // Mechanical error-text update (module doc, error enum doc): old
        // text "which must be one of 1..=7 (DTLZ1-DTLZ7; DTLZ8/DTLZ9 use
        // the constraint-surface construction and are out of scope for
        // this suite -- see this module's doc), got {0}" -> new text
        // "which must be one of 1..=9 (DTLZ1-DTLZ9; DTLZ8/DTLZ9 use the
        // constraint-surface construction, see this module's doc), got
        // {0}" (task-2 report quotes both verbatim).
        let msg = format!("{}", DtlzError::UnknownWhich(42));
        assert!(msg.contains("1..=9"), "{msg}");
        assert!(msg.contains("constraint-surface"), "{msg}");
        assert!(!msg.contains("out of scope"), "{msg}");
        assert!(msg.contains("42"), "{msg}");
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

    // ---- DTLZ8/DTLZ9-specific construction: dim > m required (n > M) ----

    #[test]
    fn dtlz8_9_dim_below_m_is_bad_dim_not_constraint_surface() {
        // dim < m still hits the SHARED BadDim check first (module doc's
        // `new` ordering note) -- BadDimConstraintSurface is a check ON TOP
        // of, not instead of, the shared dim >= m check.
        for which in [8u32, 9] {
            assert!(
                matches!(Dtlz::new(which, 5, 4), Err(DtlzError::BadDim { dim: 4, m: 5 })),
                "which={which}"
            );
        }
    }

    #[test]
    fn dtlz8_9_dim_equal_m_is_constraint_surface_error() {
        // dim == m passes the shared dim >= m check but fails DTLZ8/9's
        // OWN stricter dim > m (chapter's "n > M", Eq. 6.26/6.27).
        for which in [8u32, 9] {
            assert!(
                matches!(
                    Dtlz::new(which, 4, 4),
                    Err(DtlzError::BadDimConstraintSurface { dim: 4, m: 4 })
                ),
                "which={which}"
            );
        }
    }

    #[test]
    fn dtlz8_9_dim_above_m_is_ok() {
        for which in [8u32, 9] {
            assert!(Dtlz::new(which, 3, 4).is_ok(), "which={which}");
            assert!(Dtlz::new(which, 3, 30).is_ok(), "which={which} (chapter n=10M)");
        }
    }

    #[test]
    fn dtlz8_9_m_below_2_is_bad_m() {
        for which in [8u32, 9] {
            assert!(matches!(Dtlz::new(which, 1, 10), Err(DtlzError::BadM(1))), "which={which}");
            assert!(matches!(Dtlz::new(which, 0, 10), Err(DtlzError::BadM(0))), "which={which}");
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

    // ==================================================================
    // DTLZ8/DTLZ9 (M3-7 Task 2: constraint-surface construction)
    // ==================================================================

    #[test]
    fn dtlz1_7_evaluate_constraints_batch_stays_none() {
        // Frozen-gate sanity: DTLZ1-7 are UNCONSTRAINED, unaffected by this
        // task's constraint-channel wiring for DTLZ8/DTLZ9.
        for which in 1u32..=7 {
            let p = Dtlz::new(which, 3, 12).unwrap();
            assert_eq!(p.evaluate_constraints_batch(&[g1(&[0.3; 12])]), None, "which={which}");
        }
    }

    #[test]
    fn dtlz8_9_evaluate_batch_shape_and_finite() {
        for which in [8u32, 9] {
            for m in [2usize, 3] {
                let dim = 10 * m; // chapter-suggested n=10M
                let p = Dtlz::new(which, m, dim).unwrap();
                let xs = vec![0.4; dim];
                let out = p.evaluate_batch(&[g1(&xs)]);
                assert_eq!(out[0].len(), m, "which={which} m={m}");
                assert!(out[0].iter().all(|v| v.is_finite()), "which={which} m={m}: {out:?}");
            }
        }
    }

    #[test]
    fn dtlz8_9_evaluate_constraints_batch_row_lengths() {
        for m in [2usize, 3] {
            let p8 = Dtlz::new(8, m, 10 * m).unwrap();
            let g8 = p8.evaluate_constraints_batch(&[g1(&vec![0.3; 10 * m])]).unwrap();
            assert_eq!(g8[0].len(), m, "m={m}"); // DTLZ8: M constraints

            let p9 = Dtlz::new(9, m, 10 * m).unwrap();
            let g9 = p9.evaluate_constraints_batch(&[g1(&vec![0.3; 10 * m])]).unwrap();
            assert_eq!(g9[0].len(), m - 1, "m={m}"); // DTLZ9: M-1 constraints
        }
    }

    // ---- DTLZ8 hand-computed objective+constraint fixtures ----
    // Block boundaries [j*n/m, (j+1)*n/m), divisor floor(n/m) (module doc).

    #[test]
    fn dtlz8_fixtures_m2_dim4() {
        let p = Dtlz::new(8, 2, 4).unwrap(); // n=4, blocks of 2, divisor=2
        // xs=[0,0,0,0]: f=[0,0]. g0=fM+4f0-1=-1 (infeasible).
        // g1(=gM, M=2 corner)=+inf (vacuous, module doc decision).
        let f = p.evaluate_batch(&[g1(&[0.0, 0.0, 0.0, 0.0])]);
        assert_eq!(f[0], vec![0.0, 0.0]);
        let g = p.evaluate_constraints_batch(&[g1(&[0.0, 0.0, 0.0, 0.0])]).unwrap();
        assert_eq!(g[0], vec![-1.0, f64::INFINITY]);

        // xs=[1,1,1,1]: f=[1,1]. g0=1+4-1=4.
        let f = p.evaluate_batch(&[g1(&[1.0, 1.0, 1.0, 1.0])]);
        assert_eq!(f[0], vec![1.0, 1.0]);
        let g = p.evaluate_constraints_batch(&[g1(&[1.0, 1.0, 1.0, 1.0])]).unwrap();
        assert!((g[0][0] - 4.0).abs() < 1e-12, "{g:?}");
        assert_eq!(g[0][1], f64::INFINITY);

        // xs=[1,0,0,0]: block0=(1+0)/2=0.5, block1=(0+0)/2=0. g0=0+2-1=1.
        let f = p.evaluate_batch(&[g1(&[1.0, 0.0, 0.0, 0.0])]);
        assert_eq!(f[0], vec![0.5, 0.0]);
        let g = p.evaluate_constraints_batch(&[g1(&[1.0, 0.0, 0.0, 0.0])]).unwrap();
        assert!((g[0][0] - 1.0).abs() < 1e-12, "{g:?}");
    }

    #[test]
    fn dtlz8_fixtures_m3_dim6() {
        let p = Dtlz::new(8, 3, 6).unwrap(); // n=6, blocks of 2, divisor=2
        // xs all zero: f=[0,0,0]. g0=-1,g1=-1,g2=2*0+min(f0+f1=0)-1=-1.
        let g = p.evaluate_constraints_batch(&[g1(&[0.0; 6])]).unwrap();
        assert_eq!(g[0], vec![-1.0, -1.0, -1.0]);

        // xs all one: f=[1,1,1]. g0=4,g1=4,g2=2+2-1=3.
        let g = p.evaluate_constraints_batch(&[g1(&[1.0; 6])]).unwrap();
        assert!((g[0][0] - 4.0).abs() < 1e-12, "{g:?}");
        assert!((g[0][1] - 4.0).abs() < 1e-12, "{g:?}");
        assert!((g[0][2] - 3.0).abs() < 1e-12, "{g:?}");

        // mixed: xs=[1,0,0.5,0.5,0.2,0.2]. blocks: {1,0}->0.5, {.5,.5}->0.5,
        // {.2,.2}->0.2.
        let f = p.evaluate_batch(&[g1(&[1.0, 0.0, 0.5, 0.5, 0.2, 0.2])]);
        assert!((f[0][0] - 0.5).abs() < 1e-12, "{f:?}");
        assert!((f[0][1] - 0.5).abs() < 1e-12, "{f:?}");
        assert!((f[0][2] - 0.2).abs() < 1e-12, "{f:?}");
        let g = p.evaluate_constraints_batch(&[g1(&[1.0, 0.0, 0.5, 0.5, 0.2, 0.2])]).unwrap();
        assert!((g[0][0] - 1.2).abs() < 1e-9, "{g:?}");
        assert!((g[0][1] - 1.2).abs() < 1e-9, "{g:?}");
        assert!((g[0][2] - 0.4).abs() < 1e-9, "{g:?}");
    }

    // ---- DTLZ9 hand-computed objective+constraint fixtures ----
    // Same block boundaries as DTLZ8, but NO normalizing divisor (module
    // doc: "fj here has NO 1/floor(n/M) normalizing factor at all").

    #[test]
    fn dtlz9_fixtures_m2_dim4() {
        let p = Dtlz::new(9, 2, 4).unwrap(); // n=4, blocks of 2, no normalize
        // xs=[1,1,1,1]: f0=1^0.1+1^0.1=2, f1=2. g0=f1^2+f0^2-1=4+4-1=7.
        let f = p.evaluate_batch(&[g1(&[1.0, 1.0, 1.0, 1.0])]);
        assert_eq!(f[0], vec![2.0, 2.0]);
        let g = p.evaluate_constraints_batch(&[g1(&[1.0, 1.0, 1.0, 1.0])]).unwrap();
        assert_eq!(g[0].len(), 1); // M-1 = 1 constraint
        assert!((g[0][0] - 7.0).abs() < 1e-12, "{g:?}");

        // xs=[0,0,0,0]: f=[0,0]. g0=-1.
        let g = p.evaluate_constraints_batch(&[g1(&[0.0, 0.0, 0.0, 0.0])]).unwrap();
        assert!((g[0][0] - (-1.0)).abs() < 1e-12, "{g:?}");

        // xs=[1,0,1,0]: block sums = 1^0.1+0^0.1 = 1 each. f=[1,1]. g0=1.
        let f = p.evaluate_batch(&[g1(&[1.0, 0.0, 1.0, 0.0])]);
        assert_eq!(f[0], vec![1.0, 1.0]);
        let g = p.evaluate_constraints_batch(&[g1(&[1.0, 0.0, 1.0, 0.0])]).unwrap();
        assert!((g[0][0] - 1.0).abs() < 1e-12, "{g:?}");
    }

    #[test]
    fn dtlz9_fixtures_m3_dim6() {
        let p = Dtlz::new(9, 3, 6).unwrap(); // n=6, blocks of 2, no normalize
        // xs all one: f=[2,2,2]. g0=f2^2+f0^2-1=4+4-1=7. g1 same=7.
        let g = p.evaluate_constraints_batch(&[g1(&[1.0; 6])]).unwrap();
        assert_eq!(g[0].len(), 2); // M-1 = 2 constraints
        assert!((g[0][0] - 7.0).abs() < 1e-12, "{g:?}");
        assert!((g[0][1] - 7.0).abs() < 1e-12, "{g:?}");

        // mixed: xs=[1,1,0,0,1,0]: block0={1,1}sum=2, block1={0,0}sum=0,
        // block2={1,0}sum=1. f=[2,0,1]. g0=f2^2+f0^2-1=1+4-1=4.
        // g1=f2^2+f1^2-1=1+0-1=0 (exact boundary).
        let f = p.evaluate_batch(&[g1(&[1.0, 1.0, 0.0, 0.0, 1.0, 0.0])]);
        assert_eq!(f[0], vec![2.0, 0.0, 1.0]);
        let g = p.evaluate_constraints_batch(&[g1(&[1.0, 1.0, 0.0, 0.0, 1.0, 0.0])]).unwrap();
        assert!((g[0][0] - 4.0).abs() < 1e-12, "{g:?}");
        assert!(g[0][1].abs() < 1e-12, "{g:?}");
    }

    // ---- block-partition edge case: n not a multiple of m ----

    #[test]
    fn dtlz8_9_block_partition_edge_case_non_divisible_n() {
        // n=7, m=3: floor(n/m)=2, blocks [0,2),[2,4),[4,7) -- sizes 2,2,3
        // (the LAST block absorbs the remainder). DTLZ8 still divides
        // EVERY block (including the size-3 last one) by the SAME constant
        // divisor 2, per the module doc's block-partition convention.
        let p8 = Dtlz::new(8, 3, 7).unwrap();
        let f8 = p8.evaluate_batch(&[g1(&[1.0; 7])]);
        // block0 mean=(1+1)/2=1, block1 mean=1, block2=(1+1+1)/2=1.5 (NOT
        // 1 -- divided by the constant 2, not by its own size 3).
        assert!((f8[0][0] - 1.0).abs() < 1e-12, "{f8:?}");
        assert!((f8[0][1] - 1.0).abs() < 1e-12, "{f8:?}");
        assert!((f8[0][2] - 1.5).abs() < 1e-12, "{f8:?}");

        // DTLZ9 has no normalizing divisor at all, so its last block's sum
        // simply has one extra term (3 ones summed vs 2 for the other
        // blocks).
        let p9 = Dtlz::new(9, 3, 7).unwrap();
        let f9 = p9.evaluate_batch(&[g1(&[1.0; 7])]);
        assert!((f9[0][0] - 2.0).abs() < 1e-12, "{f9:?}");
        assert!((f9[0][1] - 2.0).abs() < 1e-12, "{f9:?}");
        assert!((f9[0][2] - 3.0).abs() < 1e-12, "{f9:?}");
    }

    // ---- DTLZ8's M=2 gM corner and its line-component feasibility ----

    #[test]
    fn dtlz8_m2_gm_is_vacuously_feasible_infinite() {
        // module doc's `// sezgi decision:` -- M=2's gM has an empty i!=j
        // pair set, so its min term is +infinity, making gM unconditionally
        // satisfied regardless of f0/fM.
        for f in [[0.0, 0.0], [1.0, 1.0], [0.5, -3.0], [-100.0, 100.0]] {
            let g = Dtlz::dtlz8_constraints(&f);
            assert_eq!(g.len(), 2, "f={f:?}");
            assert_eq!(g[1], f64::INFINITY, "f={f:?} g={g:?}");
        }
    }

    #[test]
    fn dtlz8_line_component_first_m_minus_1_constraints_are_boundary() {
        // Chapter: "the straight line is the intersection of the first
        // (M-1) constraints (with f1=f2=...=f_{M-1})" -- construct a point
        // on that line directly (f_j=t for j<M, fM=1-4t, from
        // g_j=fM+4fj-1=0 solved for fM) and check g_0..g_{M-2} are exactly
        // the boundary (0), and gM is feasible (>=0), for m=2 and m=3.
        for m in [2usize, 3] {
            let t = 0.1;
            let f_m = 1.0 - 4.0 * t;
            let mut f = vec![t; m - 1];
            f.push(f_m);
            let g = Dtlz::dtlz8_constraints(&f);
            assert_eq!(g.len(), m, "m={m}");
            for &gj in &g[..m - 1] {
                assert!(gj.abs() < 1e-12, "m={m} g={g:?}");
            }
            assert!(g[m - 1] >= 0.0, "m={m} g={g:?} (gM must be feasible on the line piece)");
        }
    }

    // ---- DTLZ9 feasibility of the chapter-described front (Eq. 6.27's
    // "circular arc of radius one") ----

    #[test]
    fn dtlz9_front_boundary_point_is_exactly_feasible() {
        // f1=f2=0.6, fM=0.8: 0.8^2+0.6^2=1 exactly (a Pythagorean triple).
        let g = Dtlz::dtlz9_constraints(&[0.6, 0.6, 0.8]);
        assert_eq!(g.len(), 2);
        assert!(g[0].abs() < 1e-12, "{g:?}");
        assert!(g[1].abs() < 1e-12, "{g:?}");
    }

    #[test]
    fn dtlz9_front_points_are_exactly_feasible_boundary() {
        for m in [2usize, 3] {
            let p = Dtlz::new(9, m, m + 5).unwrap();
            let front = p.pareto_front(20).unwrap();
            for row in &front {
                let g = Dtlz::dtlz9_constraints(row);
                for &gj in &g {
                    assert!(gj.abs() < 1e-9, "m={m} row={row:?} g={g:?}");
                }
            }
        }
    }

    // ---- pareto_front: DTLZ8 always None, DTLZ9 arc for m<=3 ----

    #[test]
    fn dtlz8_pareto_front_is_none_for_positive_n() {
        for m in [2usize, 3, 4] {
            let p = Dtlz::new(8, m, 10 * m).unwrap();
            assert!(p.pareto_front(50).is_none(), "m={m}");
            // n=0 short-circuits BEFORE the which-dispatch (same as every
            // which, existing behavior) -- an empty front makes no shape
            // claim, so it is Some(empty) even for DTLZ8.
            assert_eq!(p.pareto_front(0), Some(Vec::new()), "m={m}");
        }
    }

    #[test]
    fn dtlz9_pareto_front_arc_for_m_le_3_none_for_m_gt_3() {
        for m in [2usize, 3] {
            let p = Dtlz::new(9, m, 10 * m).unwrap();
            let front = p.pareto_front(20).unwrap();
            assert_eq!(front.len(), 20, "m={m}");
            for row in &front {
                assert_eq!(row.len(), m, "m={m}");
                // f1=...=f_{M-1}, and fM^2+f1^2=1 (unit-radius arc, the
                // chapter's own description).
                let (f_pos, f_m) = row.split_at(m - 1);
                for &fj in f_pos {
                    assert!((fj - f_pos[0]).abs() < 1e-12, "m={m}: {row:?}");
                }
                let s = f_m[0] * f_m[0] + f_pos[0] * f_pos[0];
                assert!((s - 1.0).abs() < 1e-9, "m={m}: {row:?}");
            }
        }
        let p4 = Dtlz::new(9, 4, 40).unwrap();
        assert!(p4.pareto_front(20).is_none());
    }

    #[test]
    fn dtlz9_pareto_front_deterministic_and_zero_n_empty() {
        for m in [2usize, 3] {
            let p = Dtlz::new(9, m, 10 * m).unwrap();
            let a = p.pareto_front(30).unwrap();
            let b = p.pareto_front(30).unwrap();
            assert_eq!(a, b, "m={m}");
            assert_eq!(p.pareto_front(0), Some(Vec::new()), "m={m}");
        }
    }

    // ---- nsga2_run integration: constrained fronts must be feasible ----

    #[test]
    fn nsga2_run_dtlz8_front0_is_fully_feasible() {
        use sezgi_components::nsga2::{Nsga2Config, nsga2_run};
        let p = Dtlz::new(8, 2, 20).unwrap(); // m=2, n=10M
        let cfg = Nsga2Config {
            pop_size: 40,
            budget: 4000,
            seed: 7,
            eta_c: 20.0,
            eta_m: 20.0,
            p_c: 0.9,
            p_m: None,
            // p_c_bin/p_m_bin (M3-7 Task 3): unused on this all-Float DTLZ8
            // space, present only because Nsga2Config now requires them.
            p_c_bin: 0.9,
            p_m_bin: None,
        };
        let result = nsga2_run(&p, &cfg).unwrap();
        let violations =
            result.violations.as_ref().expect("DTLZ8 is constrained: Some(violations) expected");
        assert_eq!(violations.len(), result.objectives.len());
        assert!(!result.front0.is_empty());
        for &i in &result.front0 {
            assert_eq!(
                violations[i], 0.0,
                "front0 member {i} has violation {} (expected fully feasible)",
                violations[i]
            );
        }
    }

    #[test]
    fn nsga2_run_dtlz9_front0_is_fully_feasible() {
        use sezgi_components::nsga2::{Nsga2Config, nsga2_run};
        let p = Dtlz::new(9, 2, 20).unwrap(); // m=2, n=10M
        let cfg = Nsga2Config {
            pop_size: 40,
            budget: 4000,
            seed: 7,
            eta_c: 20.0,
            eta_m: 20.0,
            p_c: 0.9,
            p_m: None,
            // p_c_bin/p_m_bin (M3-7 Task 3): unused on this all-Float DTLZ9
            // space, present only because Nsga2Config now requires them.
            p_c_bin: 0.9,
            p_m_bin: None,
        };
        let result = nsga2_run(&p, &cfg).unwrap();
        let violations =
            result.violations.as_ref().expect("DTLZ9 is constrained: Some(violations) expected");
        assert_eq!(violations.len(), result.objectives.len());
        assert!(!result.front0.is_empty());
        for &i in &result.front0 {
            assert_eq!(
                violations[i], 0.0,
                "front0 member {i} has violation {} (expected fully feasible)",
                violations[i]
            );
        }
    }
}
