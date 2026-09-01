//! Multi-objective quality indicators: exact hypervolume (a frozen
//! 2-objective specialization, and a general-M algorithm) and Inverted
//! Generational Distance (IGD).
//!
//! # Input convention
//!
//! Both functions take a "front" as `&[Vec<f64>]`: each row is one
//! objective vector (minimization -- lower is every-coordinate better).
//! `hypervolume_2d` requires every row to have exactly 2 objectives; `igd`
//! accepts any (consistent) number of objectives.
//!
//! # Provenance
//!
//! ## Hypervolume (`hypervolume_2d`): the S-metric definition
//!
//! Source: Zitzler, E., & Thiele, L. (1999). "Multiobjective Evolutionary
//! Algorithms: A Comparative Case Study and the Strength Pareto Approach."
//! *IEEE Transactions on Evolutionary Computation*, 3(4), 257-271. Section
//! III-B-1 (p. 260), "Size of the space covered," quoted verbatim
//! (converting the paper's typeset notation to prose):
//!
//! ```text
//! Let X' = (x1, x2, ..., xk) be a set of k decision vectors. The function
//! S(X') gives the volume enclosed by the union of the polytopes
//! p1, p2, ..., pk, where each pi is formed by the intersections of the
//! following hyperplanes arising out of xi, along with the axes: for each
//! axis in the objective space, there exists a hyperplane perpendicular to
//! the axis and passing through the point (f1(xi), f2(xi), ..., fn(xi)).
//! In the two-dimensional (2-D) case, each pi represents a rectangle
//! defined by the points (0,0) and (f1(xi), f2(xi)).
//! ```
//!
//! This module implements the now-standard reference-point variant of the
//! same S-metric: instead of anchoring every rectangle at the coordinate
//! origin `(0,0)` (as in the paper's maximization/profit setting), each
//! rectangle is anchored at a caller-supplied `ref_point` and spans between
//! it and the (minimization) objective vector, i.e. the corners
//! `(f1(xi), f2(xi))` and `ref_point`. `S` (the union-of-rectangles volume)
//! is then computed exactly, without double-counting overlaps, by an
//! elementary sort-and-sweep derived directly from the definition above:
//! discard dominated points (their rectangle is a subset of a surviving
//! rectangle's), sort the remainder ascending by `f1`, and sum
//! non-overlapping strips `(next_f1 - f1_i) * (ref2 - f2_i)` (the last
//! strip's `next_f1` is `ref_point[0]`). **Convention (this
//! implementation)**: a point that does not strictly dominate `ref_point`
//! (`f1 >= ref_point[0]` or `f2 >= ref_point[1]`) contributes an
//! empty/degenerate rectangle and is dropped before the sweep -- it
//! contributes nothing to `S`.
//!
//! ## Hypervolume (`hypervolume`): general-M exact algorithm (WFG)
//!
//! ### Provenance
//!
//! **(a) The paper (PRIMARY).** Lyndon While, Lucas Bradstreet, Luigi
//! Barone, "A Fast Way of Calculating Exact Hypervolumes," *IEEE
//! Transactions on Evolutionary Computation* 16(1):86-95, February 2012
//! (DOI `10.1109/TEVC.2010.2077298`; manuscript accepted 2010, "to appear"
//! at the time the toolkit's own README was written -- hence its citation
//! key `[WFG2010b]`). IEEE Xplore/ResearchGate/Semantic Scholar direct PDF
//! access all failed (as for the WFG *problem*-toolkit papers, T5); the
//! Walking Fish Group's own publications page
//! (`wfg.csse.uwa.edu.au/publications.html`, dead; recovered via the
//! Wayback Machine) lists this exact paper as `[WFG2012a]` with a `download`
//! link. Pinned:
//! `https://web.archive.org/web/20130410200500id_/http://www.wfg.csse.uwa.edu.au/publications/WFG2012a.pdf`,
//! sha256 `021b0e916fd633d4389f39b148234a8f866726b1cbeacf7a46b4559ee7df38b0`
//! (the `id_` Wayback modifier serves the raw archived bytes, not a
//! banner-wrapped replay page). Every formula and structural claim below is
//! transcribed from this PDF (`pdftotext -layout`); Fig. 5's own pseudo-code
//! is a raster image (not text-extractable), so the algorithm is
//! reconstructed instead from the paper's own prose derivation (Sections
//! II-D, III, IV-A), cross-checked against (b).
//!
//! **(b) The reference C implementation (ORACLE-ONLY; code-over-report
//! ruling for any prose/code divergence).** Cloned
//! `https://github.com/lbradstreet/WFG-hypervolume`, commit
//! `b19f35c5115cc0cae10a8f9b2cfdbb82a7c25806` (2014-05-12, "Remove 3D base
//! case code, which does not appear to actually improve our performance").
//! Its `README.md` cites this same `[WFG2010b]`/2012 TEC paper as the
//! algorithm's reference and states "Code currently performs minimisation
//! hypervolume calculations relative to the reference point" (i.e., the
//! repo's own default build already matches this crate's minimization
//! convention, unlike the paper's own maximization-relative-to-origin
//! exposition). **License finding** (decides nothing here; no line of this
//! C enters the repo): `wfg.c`/`wfg.h` carry a GNU GPL v2-or-later header
//! directly (`LICENSE.md` is the same GPLv2 text) -- copyleft, consistent
//! with an oracle-only, no-vendoring use.
//!
//! **(c) The 2022 LNCS "pitfalls" chapter -- abstract-only (paywalled).**
//! Hernandez Gomez, Falcon-Cardona, Coello Coello, "Considerations in the
//! Incremental Hypervolume Algorithm of the WFG," *Advances in
//! Computational Intelligence (MICAI 2022)*, LNCS 13612, Springer, 2022,
//! DOI `10.1007/978-3-031-19493-1_32`. Only the abstract is reachable
//! (`link.springer.com` redirects every full-text/PDF route, including the
//! direct `/content/pdf/...` URL, through an institutional-login wall;
//! ResearchGate's mirror returns HTTP 403). Abstract, quoted in full (the
//! only text recovered): "The hypervolume indicator (HV) has been subject
//! of a lot of research in the last few years ... Some years ago, the
//! Walking Fish Group (WFG) implemented a new version of the incremental
//! hypervolume algorithm, named IWFG 1.01. This implementation is the
//! fastest reported to date for determining the solution that contributes
//! the least to the HV of a non-dominated set. Nevertheless, this new
//! version has gone mostly unnoticed by the research community. We believe
//! that this is due to an error in the source code provided by the authors
//! of this algorithm, which appears when coupling it to a multi-objective
//! evolutionary algorithm. In this paper, we describe this error, and we
//! propose a solution to fix it." From the abstract and title alone: the
//! chapter's subject is **IWFG** (the *incremental*, least-contributor
//! variant used inside an EMOA's selection loop), not the plain
//! whole-front `Hyp(S)` this module computes -- so its specific bugfix is
//! very unlikely to bear on `hypervolume`'s scope directly, but since the
//! full text could not be verified, this implementation takes the brief's
//! own fallback: proceed with extra care on the delicate spots the brief
//! itself names (dominated-point filtering, duplicate points, points not
//! dominating the reference point) -- each is explicitly decided and
//! property-tested below rather than left to assumption.
//!
//! ### Algorithm (Sections II-D, III, IV-A of (a), reconstructed; (b) as
//! oracle for the reconstruction)
//!
//! The paper's own recursive identity (its Eq. (3)/(4), restated here for
//! minimization and a general reference point rather than the paper's
//! maximization-from-the-origin exposition):
//!
//! ```text
//! Hyp(p_1, ..., p_m) = sum_{i=1}^{m} ExcHyp(p_i, {p_{i+1}, ..., p_m})
//! ExcHyp(p, S)        = IncHyp(p) - Hyp(limit(S, p))       [paper Eq. (5)]
//! IncHyp(p)           = product_j (ref_point[j] - p[j])    [paper Eq. (2)]
//! limit(S, p)         = { limit(s, p) | s in S }           [paper Eq. (6)]
//! limit(s, p)         = componentwise worse(s, p)          [paper Eq. (7)]
//! ```
//!
//! where, for this module's minimization convention, `worse(x, y) =
//! max(x, y)` (the paper's own maximization exposition uses `worse = min`;
//! the reference C's `WORSE` macro is a compile-time flag selecting between
//! the two -- (b)'s own default build already targets minimization, per its
//! README quoted above). This identity holds for **any** processing order
//! of the points (it is a set identity, not order-dependent); `hv_recursive`
//! (private) implements it directly, with two performance-only additions
//! from the paper's own Section IV-B, neither of which changes the result:
//!
//! - **Sorting** (paper IV-B-1, "the simplest way to implement this
//!   optimization is to sort the points so that they are improving
//!   monotonically in one of the objectives"): the paper is explicit that
//!   *which* objective is arbitrary; this module fixes it as the last
//!   objective, at every recursion level, purely for deterministic,
//!   reproducible ordering (ties broken by Rust's stable sort preserving
//!   input order).
//! - **Pruning the limited set** (paper Section III, "any point in `S'`
//!   that is dominated by some other point in `S'` ... has no more
//!   relevance to the result, and it can be discarded before any further
//!   calculation is performed"; Fig. 4/6 show this typically removes
//!   50-80% of points after one `limit`): `prune_dominated` (private)
//!   removes every point weakly dominated by (or duplicating) another point
//!   in the same limited set, independently written from this description
//!   (not transcribed from (b)'s `makeDominatedBit`, though behaviorally
//!   equivalent -- corroborated by this task's oracle diff against the
//!   Fonseca/Lopez-Ibanez `hv` tool, an independent algorithm entirely;
//!   not itself an in-crate test, since that oracle is GPL/LGPL and is
//!   built only in the task's own scratchpad, never vendored here -- see
//!   the task report). This is a pure
//!   optimization: `Hyp` is provably invariant under removing a weakly
//!   dominated point (the identity above already proves this for the
//!   *duplicate/dominated invariance* convention, next section), so pruning
//!   changes only the cost of the recursive call, never its value. Without
//!   it, the recursion's cost is exponential in `front.len()` regardless of
//!   actual domination structure (paper Section IV-D, worst case `2^m - 1`);
//!   this is why pruning is load-bearing for the "n up to ~100" scale this
//!   module is validated at, not merely a nice-to-have.
//! - **2D shortcut** (a base case in the spirit of the paper's own IV-B-2,
//!   "a new base case when the points are reduced to two objectives ... a
//!   fast simple case"): the paper's own version of this requires an
//!   additional dimension-*reducing* slicing technique (HSO-style) layered
//!   on top of the basic recursion above, which is a materially different
//!   (and materially more complex) algorithm, explicitly NOT implemented
//!   here. Instead, `hypervolume` special-cases the *whole-problem* `M ==
//!   2` call by delegating directly to the frozen, independently-verified
//!   [`hypervolume_2d`] -- achieving the paper's cited benefit (an O(m log
//!   m) 2-objective fast path) for the one case this module can reach it
//!   without slicing, at the cost of that fast path only firing at the top
//!   level (an `M == 3` problem's internal recursion never drops to 2
//!   objectives, since `limit` changes point *values*, never the *count* of
//!   objectives). The paper's alternative 3-objective optimal base case
//!   (Beume et al.'s O(m log m) algorithm) is not implemented either
//!   (out of scope: a separate algorithm in its own right).
//!
//! ### Behavior conventions (`// sezgi decision:` tags, in the code)
//!
//! - Points that do not strictly dominate `ref_point` contribute nothing:
//!   filtered out before `hv_recursive` runs (extending [`hypervolume_2d`]'s
//!   own pinned convention, module doc above, to general M). Neither source
//!   does this: the absolute-value convention is baked into the paper's OWN
//!   Fig. 5 pseudocode (`inclhv(p)` takes `|p[j] - refPoint[j]|` -- review
//!   finding, transcribed from the rendered figure), and (b)'s reference C
//!   carries it through: `inclhv`/`exclhv` use `fabs` on every
//!   `ref[j] - p[j]` difference unconditionally, silently "reflecting" a
//!   would-be-negative excess back to positive if a point does not actually
//!   dominate `ref_point` in some coordinate -- i.e., both assume the
//!   precondition holds and give a wrong-but-plausible-looking number if it
//!   is violated, rather than erroring or clamping to zero. This module
//!   makes the stronger,
//!   well-defined choice instead (explicit pre-filter, matching
//!   `hypervolume_2d`'s own established convention exactly), rather than
//!   reproducing an unvalidated-precondition behavior.
//! - Duplicate points and points dominated by another front member
//!   contribute nothing extra and must not change the result: this is a
//!   direct consequence of the recursive identity above (not a separate
//!   filtering rule at the top level, unlike [`hypervolume_2d`]'s own
//!   pre-sweep) -- see `hypervolume_duplicate_and_dominated_points_do_not_change_result`
//!   for the property test. `prune_dominated` (the internal-recursion
//!   pruning, previous section) also depends on this same invariance.
//! - Empty `front` returns `Ok(0.0)`, **not** an error -- unlike
//!   [`hypervolume_2d`], which rejects an empty front. This is deliberate:
//!   `Hyp({}) = 0` is the recursion's own base case (the empty sum in the
//!   identity above), so treating it as an error would be an arbitrary
//!   inconsistency with the algorithm's own definition, not a safety
//!   guard -- `hypervolume_2d` predates this reasoning and is frozen, so it
//!   keeps its original (stricter) behavior rather than being changed to
//!   match.
//! - Dimension mismatches (`front` row length != `ref_point.len()`, or an
//!   empty `ref_point`) are rejected with [`StatsError::InvalidInput`],
//!   matching this module's existing conventions (`hypervolume_2d`, `igd`).
//!
//! ### Choosing a reference point: explicit-always, contested in the
//! literature
//!
//! Per this project's scope ruling 5, `hypervolume` (like `hypervolume_2d`)
//! never supplies a default `ref_point` -- callers always pass one
//! explicitly. This is not merely a defensive default-avoidance stance: Hisao
//! Ishibuchi, Ryo Imada, Yu Setoguchi, Yusuke Nojima, "How to Specify a
//! Reference Point in Hypervolume Calculation for Fair Performance
//! Comparison," *Evolutionary Computation* 26(3):411-440, 2018, shows that
//! the choice of reference point materially changes which algorithm looks
//! best on a given benchmark -- there is no single "correct" choice the
//! library could silently pick on the caller's behalf without embedding a
//! contested methodological judgment. Exposing `ref_point` as a required
//! parameter lets each caller make (and document) that judgment themselves.
//! This module's own examples and tests use one common convention from the
//! literature (also critiqued by Ishibuchi et al., but a defensible,
//! explicit default for illustration): the analytic front's nadir point
//! (the componentwise worst value across the front) scaled by `1.1` in
//! objective space.
//!
//! ## IGD (`igd`): pinned definition and variant
//!
//! Source: Ishibuchi, H., Masuda, H., Tanigaki, Y., & Nojima, Y. (2015).
//! "Modified Distance Calculation in Generational Distance and Inverted
//! Generational Distance." *Proceedings of the 8th International
//! Conference on Evolutionary Multi-Criterion Optimization (EMO 2015)*,
//! Part I, 110-125. Eq. (12), quoted verbatim (converting the paper's
//! typeset notation to prose):
//!
//! ```text
//! Inverted Generational Distance:
//!   IGD(A) = (1/|Z|) * ( sum_{j=1}^{|Z|} d^_j^p )^(1/p)
//! where d^_j is the Euclidean distance from z_j to its nearest objective
//! vector in A.
//! ```
//!
//! and, on `p` (same section): "In this paper, we always specify the value
//! of p as p = 1. ... the IGD with p = 1 is the average Euclidean distance
//! from each reference point to its nearest objective vector." **PINNED
//! variant**: this module implements exactly `p = 1` -- the plain
//! arithmetic mean of per-reference-point nearest-neighbor Euclidean
//! distances: `IGD(A) = (1/|Z|) * sum_{z in Z} min_{a in A} d(z,a)`. This is
//! NOT the general power-mean form, nor Schutze et al.'s `IGD_p` (which
//! moves the `1/|Z|` factor inside the outer power); per the same section,
//! that variant coincides with the plain-mean `IGD` only at `p = 1`. The
//! Ishibuchi et al. (2015) paper itself attributes the term "inverted
//! generational distance" to Coello & Sierra (2004) and Sierra & Coello
//! (2004).
//!
//! # Determinism
//!
//! Both functions are pure and deterministic (no RNG involved); repeat
//! calls on identical inputs are bit-identical, asserted directly in tests.

use crate::{check_finite, StatsError};

/// Exact 2-objective hypervolume (minimization) of `front` with respect to
/// `ref_point`. See the module doc for the pinned S-metric definition
/// (Zitzler & Thiele, 1999) and the reference-point sort-and-sweep
/// computation derived from it.
///
/// Points not strictly dominating `ref_point` contribute nothing (see
/// module doc, "Convention").
///
/// # Errors
/// [`StatsError::InvalidInput`] if any value (in `front` or `ref_point`) is
/// non-finite, if `front` is empty, or if any row of `front` does not have
/// exactly 2 objectives.
pub fn hypervolume_2d(front: &[Vec<f64>], ref_point: &[f64; 2]) -> Result<f64, StatsError> {
    check_finite("hypervolume_2d", front.iter().flatten().copied())?;
    check_finite("hypervolume_2d", ref_point.iter().copied())?;

    if front.is_empty() {
        return Err(StatsError::InvalidInput(
            "hypervolume_2d: front is empty".to_string(),
        ));
    }

    for (i, row) in front.iter().enumerate() {
        if row.len() != 2 {
            return Err(StatsError::InvalidInput(format!(
                "hypervolume_2d: front row {i} has {} objectives, expected exactly 2",
                row.len()
            )));
        }
    }

    let ref1 = ref_point[0];
    let ref2 = ref_point[1];

    // Keep only points that strictly dominate ref_point; everything else
    // contributes an empty/degenerate rectangle (module doc "Convention").
    let mut candidates: Vec<(f64, f64)> = front
        .iter()
        .filter_map(|p| {
            let (f1, f2) = (p[0], p[1]);
            if f1 < ref1 && f2 < ref2 {
                Some((f1, f2))
            } else {
                None
            }
        })
        .collect();

    candidates.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("non-finite already rejected above")
            .then(
                a.1.partial_cmp(&b.1)
                    .expect("non-finite already rejected above"),
            )
    });

    // Sweep ascending by f1, keeping only points that strictly improve f2
    // (non-dominated front); this also drops exact duplicates and any
    // point dominated by an earlier (smaller-f1) survivor.
    let mut kept: Vec<(f64, f64)> = Vec::with_capacity(candidates.len());
    let mut best_f2 = f64::INFINITY;
    for (f1, f2) in candidates {
        if f2 < best_f2 {
            kept.push((f1, f2));
            best_f2 = f2;
        }
    }

    let mut area = 0.0_f64;
    for (i, &(f1, f2)) in kept.iter().enumerate() {
        let next_f1 = if i + 1 < kept.len() {
            kept[i + 1].0
        } else {
            ref1
        };
        area += (next_f1 - f1) * (ref2 - f2);
    }

    Ok(area)
}

/// Exact general-M hypervolume (minimization) of `front` with respect to
/// `ref_point`, via the WFG algorithm (While, Bradstreet & Barone, 2012).
/// See the module doc, "Hypervolume (`hypervolume`): general-M exact
/// algorithm (WFG)", for the algorithm's provenance, structure, and the
/// pinned behavior conventions (ref-point boundary, duplicates/dominated
/// points, empty front).
///
/// # sezgi decision: empty front
/// Unlike [`hypervolume_2d`], an empty `front` is not an error: it returns
/// `Ok(0.0)` (module doc, "Behavior conventions").
///
/// # Errors
/// [`StatsError::InvalidInput`] if any value (in `front` or `ref_point`) is
/// non-finite, if `ref_point` is empty, or if any row of `front` does not
/// have the same number of objectives as `ref_point`.
pub fn hypervolume(front: &[Vec<f64>], ref_point: &[f64]) -> Result<f64, StatsError> {
    check_finite("hypervolume", front.iter().flatten().copied())?;
    check_finite("hypervolume", ref_point.iter().copied())?;

    if ref_point.is_empty() {
        return Err(StatsError::InvalidInput(
            "hypervolume: ref_point must have at least 1 objective, got 0".to_string(),
        ));
    }

    let m = ref_point.len();
    for (i, row) in front.iter().enumerate() {
        if row.len() != m {
            return Err(StatsError::InvalidInput(format!(
                "hypervolume: front row {i} has {} objectives, expected {m} (matching ref_point)",
                row.len()
            )));
        }
    }

    // sezgi decision: empty front -> 0.0, not an error (module doc,
    // "Behavior conventions"): Hyp({}) = 0 is the algorithm's own base case.
    if front.is_empty() {
        return Ok(0.0);
    }

    // sezgi decision: M == 2 delegates to the frozen hypervolume_2d exactly
    // (module doc, "2D shortcut") rather than running the general recursion.
    if m == 2 {
        return hypervolume_2d(front, &[ref_point[0], ref_point[1]]);
    }

    // sezgi decision: points that do not strictly dominate ref_point
    // contribute nothing (module doc, "Behavior conventions").
    let candidates: Vec<Vec<f64>> = front
        .iter()
        .filter(|p| p.iter().zip(ref_point).all(|(pi, ri)| pi < ri))
        .cloned()
        .collect();

    Ok(hv_recursive(&candidates, ref_point))
}

/// The WFG algorithm's core mutual recursion (module doc, "Algorithm"):
/// `Hyp(front) = sum_i ExcHyp(front[i], front[i+1..])`, `ExcHyp(p, S) =
/// IncHyp(p) - Hyp(limit(S, p))`. Assumes every row of `front` already
/// strictly dominates `ref_point` and has exactly `ref_point.len()`
/// objectives (both enforced by the public [`hypervolume`] caller; this
/// function performs no validation of its own, and is exercised directly
/// -- bypassing the public M==2 shortcut -- by this module's 2D property
/// test against [`hypervolume_2d`]).
fn hv_recursive(front: &[Vec<f64>], ref_point: &[f64]) -> f64 {
    if front.is_empty() {
        return 0.0;
    }

    // Sort optimization (module doc, "Algorithm"): ascending in one fixed
    // objective (the last); any objective is correct, a fixed choice keeps
    // results deterministic. Stable sort preserves input order among ties.
    let sort_key = ref_point.len() - 1;
    let mut sorted = front.to_vec();
    sorted.sort_by(|a, b| {
        a[sort_key]
            .partial_cmp(&b[sort_key])
            .expect("non-finite already rejected by the public hypervolume() caller")
    });

    let mut total = 0.0_f64;
    for i in 0..sorted.len() {
        let p = &sorted[i];
        // IncHyp(p): the box between p and ref_point (paper Eq. (2)).
        let incl: f64 = p
            .iter()
            .zip(ref_point.iter())
            .map(|(pi, ri)| ri - pi)
            .product();

        if i + 1 == sorted.len() {
            // Base case: nothing after p in this order, so its exclusive
            // hypervolume is its whole inclusive hypervolume.
            total += incl;
            continue;
        }

        // limit(q, p): componentwise "worse" of q and p; for minimization,
        // "worse" is the larger value (module doc, "Algorithm").
        let limited: Vec<Vec<f64>> = sorted[i + 1..]
            .iter()
            .map(|q| {
                q.iter()
                    .zip(p.iter())
                    .map(|(qi, pi)| qi.max(*pi))
                    .collect()
            })
            .collect();

        // Pruning (module doc, "Algorithm"): correctness-preserving, see
        // prune_dominated's own doc for why.
        let pruned = prune_dominated(limited);

        total += incl - hv_recursive(&pruned, ref_point);
    }

    total
}

/// Returns `points` filtered down to a mutually-non-dominated subset:
/// whenever `a` weakly dominates `b` (module doc's `weakly_dominates`),
/// `b` is discarded. Order of the input is not preserved.
///
/// Pure optimization, not a behavior change: `Hyp` is invariant under
/// removing a point weakly dominated by (or duplicating) another point in
/// the same set -- this is exactly the *duplicate/dominated invariance*
/// convention documented and property-tested for the public [`hypervolume`]
/// (module doc, "Behavior conventions"), applied here one recursion level
/// at a time so the recursive calls this function feeds stay small (module
/// doc, "Algorithm," on why this matters for the `n` this module targets).
fn prune_dominated(points: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut kept: Vec<Vec<f64>> = Vec::with_capacity(points.len());
    for p in points {
        if kept.iter().any(|q| weakly_dominates(q, &p)) {
            continue;
        }
        kept.retain(|q| !weakly_dominates(&p, q));
        kept.push(p);
    }
    kept
}

/// `a` weakly dominates `b`: `a[i] <= b[i]` for every objective `i`. This
/// includes the all-equal case (`a == b` weakly dominates itself), so an
/// exact duplicate of an already-kept point counts as dominated by
/// [`prune_dominated`] and is discarded.
fn weakly_dominates(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(ai, bi)| ai <= bi)
}

/// Inverted Generational Distance: the mean, over reference-front points,
/// of the Euclidean distance to the nearest point in `front`. Any (equal,
/// consistent) number of objectives. See the module doc for the pinned
/// definition (Ishibuchi et al., 2015, eq. 12, `p = 1`).
///
/// # Errors
/// [`StatsError::InvalidInput`] if any value is non-finite, if `front` or
/// `reference_front` is empty, or if the rows of `front` and
/// `reference_front` are not all of the same (nonzero) length.
pub fn igd(front: &[Vec<f64>], reference_front: &[Vec<f64>]) -> Result<f64, StatsError> {
    check_finite("igd", front.iter().flatten().copied())?;
    check_finite("igd", reference_front.iter().flatten().copied())?;

    if front.is_empty() {
        return Err(StatsError::InvalidInput("igd: front is empty".to_string()));
    }
    if reference_front.is_empty() {
        return Err(StatsError::InvalidInput(
            "igd: reference_front is empty".to_string(),
        ));
    }

    let d = front[0].len();
    if d == 0 {
        return Err(StatsError::InvalidInput(
            "igd: front rows must have at least 1 objective, got 0".to_string(),
        ));
    }
    for (i, row) in front.iter().enumerate() {
        if row.len() != d {
            return Err(StatsError::InvalidInput(format!(
                "igd: ragged front, row 0 has {d} objectives, row {i} has {}",
                row.len()
            )));
        }
    }
    for (i, row) in reference_front.iter().enumerate() {
        if row.len() != d {
            return Err(StatsError::InvalidInput(format!(
                "igd: reference_front row {i} has {} objectives, expected {d} (matching front)",
                row.len()
            )));
        }
    }

    let mut sum = 0.0_f64;
    for z in reference_front {
        let mut min_dist = f64::INFINITY;
        for a in front {
            let dist_sq: f64 = z.iter().zip(a.iter()).map(|(zi, ai)| (zi - ai).powi(2)).sum();
            let dist = dist_sq.sqrt();
            if dist < min_dist {
                min_dist = dist;
            }
        }
        sum += min_dist;
    }

    Ok(sum / reference_front.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand fixture: 3-point 2D non-dominated front, full rectangle
    // arithmetic.
    //
    // front = [(0.25, 0.75), (0.5, 0.5), (0.75, 0.25)], ref_point = (1, 1).
    // All three points strictly dominate ref_point and none dominates
    // another (each is better in one coordinate, worse in the other), so
    // all three survive the sweep, already sorted ascending by f1.
    //
    // i=1 (0.25, 0.75): width = 0.5  - 0.25 = 0.25, height = 1.0 - 0.75 = 0.25
    //                    area = 0.25 * 0.25 = 0.0625
    // i=2 (0.5, 0.5):    width = 0.75 - 0.5  = 0.25, height = 1.0 - 0.5  = 0.5
    //                    area = 0.25 * 0.5  = 0.125
    // i=3 (0.75, 0.25):  width = 1.0  - 0.75 = 0.25, height = 1.0 - 0.25 = 0.75
    //                    area = 0.25 * 0.75 = 0.1875
    //
    // total = 0.0625 + 0.125 + 0.1875 = 0.375
    //
    // Every value above (0.25, 0.5, 0.75, 0.0625, 0.125, 0.1875, 0.375) is
    // exactly representable in f64, so the assertion below is exact
    // equality, not a tolerance check.
    #[test]
    fn hypervolume_2d_hand_fixture_exact() {
        let front = vec![vec![0.25, 0.75], vec![0.5, 0.5], vec![0.75, 0.25]];
        let ref_point = [1.0, 1.0];

        let hv = hypervolume_2d(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 0.375);
    }

    // Dominated point contributes nothing: (0.6, 0.6) is dominated by
    // (0.5, 0.5) (0.5 <= 0.6 and 0.5 <= 0.6), so it is dropped by the sweep
    // and the hypervolume is unchanged from the base fixture (0.375).
    #[test]
    fn hypervolume_2d_dominated_point_contributes_nothing() {
        let front = vec![
            vec![0.25, 0.75],
            vec![0.5, 0.5],
            vec![0.75, 0.25],
            vec![0.6, 0.6],
        ];
        let ref_point = [1.0, 1.0];

        let hv = hypervolume_2d(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 0.375);
    }

    // Point outside the ref box contributes nothing: (1.5, 0.1) has
    // f1 = 1.5 > ref_point[0] = 1.0, so it does not strictly dominate
    // ref_point and is filtered out before the sweep; hypervolume is
    // unchanged from the base fixture (0.375).
    #[test]
    fn hypervolume_2d_point_outside_ref_box_contributes_nothing() {
        let front = vec![
            vec![0.25, 0.75],
            vec![0.5, 0.5],
            vec![0.75, 0.25],
            vec![1.5, 0.1],
        ];
        let ref_point = [1.0, 1.0];

        let hv = hypervolume_2d(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 0.375);
    }

    // Duplicate points don't double-count: (0.5, 0.5) appears twice; the
    // sweep's strict "f2 < best_f2" survivor test drops the second copy
    // (its f2 does not strictly improve on the first), so the hypervolume
    // is unchanged from the base fixture (0.375).
    #[test]
    fn hypervolume_2d_duplicate_points_do_not_double_count() {
        let front = vec![
            vec![0.25, 0.75],
            vec![0.5, 0.5],
            vec![0.75, 0.25],
            vec![0.5, 0.5],
        ];
        let ref_point = [1.0, 1.0];

        let hv = hypervolume_2d(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 0.375);
    }

    // Ref-point boundary-equality convention guard: a front point with
    // f1 == ref_point[0] exactly, and one with f2 == ref_point[1] exactly,
    // do NOT strictly dominate ref_point (the code's filter is `<`, not
    // `<=`), so both contribute zero area. Exact-representable values
    // (0.5, 1.0) so this is checked by exact equality, not tolerance.
    // Guards against a future "fix" of `<` to `<=` silently changing the
    // pinned convention.
    #[test]
    fn hypervolume_2d_boundary_equality_contributes_nothing() {
        let front = vec![
            vec![0.25, 0.75],
            vec![0.5, 0.5],
            vec![0.75, 0.25],
            vec![1.0, 0.5], // f1 == ref_point[0] exactly
            vec![0.5, 1.0], // f2 == ref_point[1] exactly
        ];
        let ref_point = [1.0, 1.0];

        let hv = hypervolume_2d(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 0.375);
    }

    // Same-f1 tie that is NOT an exact duplicate: front =
    // [(1,3), (2,2), (2,1), (3,0.5)], ref_point = (4,4). (2,2) and (2,1)
    // share f1 = 2 but (2,2) is weakly dominated by (2,1) (1 <= 2, 2 <= 2,
    // not equal), so (2,2) must contribute nothing; the running-min sweep
    // (sorted ascending by (f1, f2)) keeps (1,3), then (2,1) (f2=1 < 3,
    // survives), then skips (2,2) (f2=2 is not < best_f2=1), then keeps
    // (3,0.5) (f2=0.5 < 1).
    //
    // Kept, ascending f1: (1,3), (2,1), (3,0.5).
    //
    // i=1 (1,3):   width = 2 - 1 = 1,   height = 4 - 3   = 1   -> area = 1
    // i=2 (2,1):   width = 3 - 2 = 1,   height = 4 - 1   = 3   -> area = 3
    // i=3 (3,0.5): width = 4 - 3 = 1,   height = 4 - 0.5 = 3.5 -> area = 3.5
    //
    // total = 1 + 3 + 3.5 = 7.5
    #[test]
    fn hypervolume_2d_same_f1_tie_not_double_counted() {
        let front = vec![
            vec![1.0, 3.0],
            vec![2.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 0.5],
        ];
        let ref_point = [4.0, 4.0];

        let hv = hypervolume_2d(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 7.5);
    }

    #[test]
    fn hypervolume_2d_is_deterministic() {
        let front = vec![vec![0.25, 0.75], vec![0.5, 0.5], vec![0.75, 0.25]];
        let ref_point = [1.0, 1.0];

        let hv1 = hypervolume_2d(&front, &ref_point).unwrap();
        let hv2 = hypervolume_2d(&front, &ref_point).unwrap();
        assert_eq!(hv1.to_bits(), hv2.to_bits());
    }

    #[test]
    fn hypervolume_2d_errors_on_empty_front() {
        let front: Vec<Vec<f64>> = vec![];
        let ref_point = [1.0, 1.0];
        let err = hypervolume_2d(&front, &ref_point).expect_err("empty front must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("hypervolume_2d"), "message: {msg}");
        assert!(msg.contains("empty"), "message: {msg}");
    }

    #[test]
    fn hypervolume_2d_errors_on_non_2d_row() {
        let front = vec![vec![0.25, 0.75, 0.1], vec![0.5, 0.5]];
        let ref_point = [1.0, 1.0];
        let err = hypervolume_2d(&front, &ref_point).expect_err("non-2D row must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("hypervolume_2d"), "message: {msg}");
        assert!(msg.contains("expected exactly 2"), "message: {msg}");
    }

    #[test]
    fn hypervolume_2d_errors_on_non_finite_front_value() {
        let front = vec![vec![0.25, f64::NAN], vec![0.5, 0.5]];
        let ref_point = [1.0, 1.0];
        let err = hypervolume_2d(&front, &ref_point).expect_err("NaN must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("hypervolume_2d"), "message: {msg}");
        assert!(msg.contains("non-finite"), "message: {msg}");
    }

    #[test]
    fn hypervolume_2d_errors_on_non_finite_ref_point() {
        let front = vec![vec![0.25, 0.75]];
        let ref_point = [1.0, f64::INFINITY];
        let err = hypervolume_2d(&front, &ref_point).expect_err("non-finite ref_point rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("hypervolume_2d"), "message: {msg}");
        assert!(msg.contains("non-finite"), "message: {msg}");
    }

    // ---- hypervolume (general-M) -------------------------------------

    // Analytic fixture: single point, any M. IncHyp(p) = product_j(ref_j -
    // p_j) (module doc, paper Eq. (2)) -- exercised here for M=3 and M=4
    // with the "last point in the sorted order" base case of hv_recursive
    // (front.len() == 1, so no recursive limit/prune step is reached; for
    // M=2 this instead exercises the hypervolume_2d shortcut directly).
    //
    // M=3: p=(1,2,3), ref=(5,7,9,11) -- wait, ref must have 3 entries for
    // M=3: ref=(5,7,9). HV = (5-1)*(7-2)*(9-3) = 4*5*6 = 120.
    #[test]
    fn hypervolume_single_point_box_product_m3() {
        let front = vec![vec![1.0, 2.0, 3.0]];
        let ref_point = [5.0, 7.0, 9.0];

        let hv = hypervolume(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 120.0);
    }

    // M=4: p=(1,2,3,4), ref=(5,7,9,11). HV = (5-1)*(7-2)*(9-3)*(11-4)
    //                                       = 4*5*6*7 = 840.
    // All integers, exactly representable in f64 -- exact equality, not a
    // tolerance check.
    #[test]
    fn hypervolume_single_point_box_product_m4() {
        let front = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let ref_point = [5.0, 7.0, 9.0, 11.0];

        let hv = hypervolume(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 840.0);
    }

    // Analytic fixture: two-point inclusion-exclusion, M=3 (generalizes the
    // "two 2D boxes" case -- each front point anchors a box at ref_point;
    // Vol(A union B) = Vol(A) + Vol(B) - Vol(A intersect B), and since both
    // boxes share the ref_point corner, A intersect B is the box between
    // ref_point and limit(p,q) = componentwise max(p,q) (this module's
    // "worse" for minimization -- module doc, "Algorithm").
    //
    // ref = (10,10,10). p = (2,5,8), q = (5,8,2).
    // Mutual non-domination: p beats q in obj0 (2<5) and obj1 (5<8); q beats
    // p in obj2 (2<8) -- neither dominates.
    //
    // IncHyp(p) = (10-2)*(10-5)*(10-8) = 8*5*2 = 80
    // IncHyp(q) = (10-5)*(10-8)*(10-2) = 5*2*8 = 80
    // limit(p,q) = (max(2,5), max(5,8), max(8,2)) = (5,8,8)
    // IncHyp(limit) = (10-5)*(10-8)*(10-8) = 5*2*2 = 20
    // HV = 80 + 80 - 20 = 140
    #[test]
    fn hypervolume_two_point_inclusion_exclusion_m3() {
        let front = vec![vec![2.0, 5.0, 8.0], vec![5.0, 8.0, 2.0]];
        let ref_point = [10.0, 10.0, 10.0];

        let hv = hypervolume(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 140.0);
    }

    // Analytic fixture: three-point inclusion-exclusion, M=3 (exercises a
    // full two-level recursion: excl_hv(p0) limits+prunes {p1,p2}, and that
    // pruned pair itself needs one more limit+recurse step). Symmetric
    // (cyclic-permutation) construction so the by-hand inclusion-exclusion
    // is tractable:
    //
    // ref = (10,10,10).
    // p0 = (1,8,9), p1 = (9,1,8), p2 = (8,9,1).
    // Mutual non-domination (by the cyclic symmetry, same for every pair):
    // p0 beats p1 in obj0 (1<9); p1 beats p0 in obj1 (1<8) and obj2 (8<9) --
    // neither dominates, and likewise for (p0,p2) and (p1,p2).
    //
    // Inclusion-exclusion over 3 sets (each anchored at ref_point):
    //   HV = sum(singles) - sum(pairwise intersections) + (triple intersection)
    //
    // IncHyp(p0) = 9*2*1 = 18; IncHyp(p1) = 1*9*2 = 18; IncHyp(p2) = 2*1*9 = 18
    //   sum(singles) = 54
    // limit(p0,p1) = (9,8,9) -> IncHyp = 1*2*1 = 2
    // limit(p0,p2) = (8,9,9) -> IncHyp = 2*1*1 = 2
    // limit(p1,p2) = (9,9,8) -> IncHyp = 1*1*2 = 2
    //   sum(pairwise) = 6
    // limit(p0,p1,p2) = (9,9,9) -> IncHyp = 1*1*1 = 1
    //   triple = 1
    //
    // HV = 54 - 6 + 1 = 49
    #[test]
    fn hypervolume_three_point_inclusion_exclusion_m3() {
        let front = vec![vec![1.0, 8.0, 9.0], vec![9.0, 1.0, 8.0], vec![8.0, 9.0, 1.0]];
        let ref_point = [10.0, 10.0, 10.0];

        let hv = hypervolume(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 49.0);
    }

    // Analytic fixture: M=4, built by appending a 4th objective that is
    // IDENTICAL across all three points of the M=3 fixture above. Each
    // point's box then factors as (3D box) x [shared_4th_value,
    // ref_point[3]], and since that interval is the SAME for every point,
    // the union factors too: Union_i (A_i x I) = (Union_i A_i) x I, so
    // HV_4D = HV_3D * length(I) = 49 * (8 - 5) = 147.
    #[test]
    fn hypervolume_four_point_constant_dimension_factors_m4() {
        let front = vec![
            vec![1.0, 8.0, 9.0, 5.0],
            vec![9.0, 1.0, 8.0, 5.0],
            vec![8.0, 9.0, 1.0, 5.0],
        ];
        let ref_point = [10.0, 10.0, 10.0, 8.0];

        let hv = hypervolume(&front, &ref_point).expect("valid fixture");
        assert_eq!(hv, 147.0);
    }

    // "Reference point convention" fixture (module doc, "Choosing a
    // reference point"): this module's own examples/tests use the
    // analytic front's nadir (componentwise worst/max value across the
    // front, for minimization) scaled by 1.1 -- one common convention from
    // the literature (also the specific choice critiqued by Ishibuchi et
    // al. 2018, cited in the module doc -- this is the module's own
    // explicit, documented pick, not a silent library default). Reusing
    // the M=3 three-point symmetric front from the fixture above:
    // nadir = componentwise max = (9,9,9), ref_point = nadir * 1.1 =
    // (9.9,9.9,9.9).
    //
    // Same inclusion-exclusion structure as that fixture, just with
    // ref=(9.9,9.9,9.9) instead of (10,10,10):
    //   IncHyp(p_k) = 8.9*1.9*0.9 for every k (by the cyclic symmetry)
    //   every pairwise limit's IncHyp = 0.9*1.9*0.9 (same 3 factors in any
    //     order, so identical for all 3 pairs)
    //   triple limit's IncHyp = 0.9^3
    //   HV = 3*(8.9*1.9*0.9) - 3*(0.9*1.9*0.9) + 0.9^3 = 41.769
    // 0.1-scale decimals are not exactly representable in f64 (unlike the
    // integer fixtures above), so this is an epsilon check, not exact
    // equality.
    #[test]
    fn hypervolume_reference_point_convention_nadir_times_1_1() {
        let front = vec![vec![1.0, 8.0, 9.0], vec![9.0, 1.0, 8.0], vec![8.0, 9.0, 1.0]];
        let nadir: Vec<f64> = (0..3)
            .map(|j| front.iter().map(|p| p[j]).fold(f64::MIN, f64::max))
            .collect();
        let ref_point: Vec<f64> = nadir.iter().map(|&v| v * 1.1).collect();
        assert_eq!(ref_point, vec![9.9, 9.9, 9.9]);

        let expected = 3.0 * (8.9 * 1.9 * 0.9) - 3.0 * (0.9 * 1.9 * 0.9) + 0.9_f64.powi(3);

        let hv = hypervolume(&front, &ref_point).expect("valid fixture");
        assert!(
            (hv - expected).abs() < 1e-9,
            "hv={hv}, expected={expected}"
        );
    }

    // Analytic fixture: unit-hypercube extreme. A single point at the
    // origin with ref_point = (1,...,1) dominates the WHOLE unit hypercube,
    // so HV = 1 exactly, for any M. Checked for M=2 (hypervolume_2d
    // shortcut), M=3, and M=4 (general recursion).
    #[test]
    fn hypervolume_unit_hypercube_extreme() {
        for m in 2..=4 {
            let front = vec![vec![0.0; m]];
            let ref_point = vec![1.0; m];

            let hv = hypervolume(&front, &ref_point).expect("valid fixture");
            assert_eq!(hv, 1.0, "M={m}");
        }
    }

    // Duplicate/dominated invariance (module doc, "Behavior conventions"):
    // starting from the M=3 three-point fixture (HV=49), each of the
    // following additions must leave the result unchanged:
    // - an exact duplicate of an existing point;
    // - a point strictly dominated by an existing point (all coordinates
    //   worse, still inside the ref box);
    // - a point outside the ref box in one coordinate (does not strictly
    //   dominate ref_point);
    // - a point exactly ON the ref_point boundary in one coordinate
    //   (boundary-equality convention: `<` not `<=`, matching
    //   hypervolume_2d's own pinned convention).
    #[test]
    fn hypervolume_duplicate_and_dominated_points_do_not_change_result() {
        let base = vec![vec![1.0, 8.0, 9.0], vec![9.0, 1.0, 8.0], vec![8.0, 9.0, 1.0]];
        let ref_point = [10.0, 10.0, 10.0];
        let base_hv = hypervolume(&base, &ref_point).expect("valid fixture");
        assert_eq!(base_hv, 49.0);

        let mut with_duplicate = base.clone();
        with_duplicate.push(vec![1.0, 8.0, 9.0]); // exact duplicate of p0
        assert_eq!(
            hypervolume(&with_duplicate, &ref_point).expect("valid fixture"),
            49.0
        );

        let mut with_dominated = base.clone();
        with_dominated.push(vec![1.5, 8.5, 9.5]); // dominated by p0 = (1,8,9)
        assert_eq!(
            hypervolume(&with_dominated, &ref_point).expect("valid fixture"),
            49.0
        );

        let mut with_outside_ref_box = base.clone();
        with_outside_ref_box.push(vec![11.0, 0.1, 0.1]); // obj0 > ref_point[0]
        assert_eq!(
            hypervolume(&with_outside_ref_box, &ref_point).expect("valid fixture"),
            49.0
        );

        let mut with_boundary_equal = base.clone();
        with_boundary_equal.push(vec![10.0, 0.1, 0.1]); // obj0 == ref_point[0] exactly
        assert_eq!(
            hypervolume(&with_boundary_equal, &ref_point).expect("valid fixture"),
            49.0
        );
    }

    #[test]
    fn hypervolume_is_deterministic() {
        let front = vec![vec![1.0, 8.0, 9.0], vec![9.0, 1.0, 8.0], vec![8.0, 9.0, 1.0]];
        let ref_point = [10.0, 10.0, 10.0];

        let hv1 = hypervolume(&front, &ref_point).unwrap();
        let hv2 = hypervolume(&front, &ref_point).unwrap();
        assert_eq!(hv1.to_bits(), hv2.to_bits());
    }

    // sezgi decision: empty front -> Ok(0.0), NOT an error (module doc,
    // "Behavior conventions") -- deliberate divergence from hypervolume_2d.
    #[test]
    fn hypervolume_empty_front_is_zero_not_an_error() {
        let front: Vec<Vec<f64>> = vec![];
        let ref_point = [1.0, 1.0, 1.0];

        let hv = hypervolume(&front, &ref_point).expect("empty front must not error");
        assert_eq!(hv, 0.0);
    }

    #[test]
    fn hypervolume_errors_on_empty_ref_point() {
        let front = vec![vec![1.0, 2.0]];
        let ref_point: Vec<f64> = vec![];
        let err =
            hypervolume(&front, &ref_point).expect_err("empty ref_point must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("hypervolume"), "message: {msg}");
        assert!(msg.contains("ref_point"), "message: {msg}");
    }

    #[test]
    fn hypervolume_errors_on_dimension_mismatch() {
        let front = vec![vec![1.0, 2.0, 3.0], vec![1.0, 2.0]];
        let ref_point = [10.0, 10.0, 10.0];
        let err = hypervolume(&front, &ref_point).expect_err("ragged front rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("hypervolume"), "message: {msg}");
        assert!(msg.contains("expected 3"), "message: {msg}");
    }

    #[test]
    fn hypervolume_errors_on_non_finite_front_value() {
        let front = vec![vec![1.0, f64::NAN, 3.0]];
        let ref_point = [10.0, 10.0, 10.0];
        let err = hypervolume(&front, &ref_point).expect_err("NaN must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("hypervolume"), "message: {msg}");
        assert!(msg.contains("non-finite"), "message: {msg}");
    }

    #[test]
    fn hypervolume_errors_on_non_finite_ref_point() {
        let front = vec![vec![1.0, 2.0, 3.0]];
        let ref_point = [10.0, f64::INFINITY, 10.0];
        let err = hypervolume(&front, &ref_point).expect_err("non-finite ref_point rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("hypervolume"), "message: {msg}");
        assert!(msg.contains("non-finite"), "message: {msg}");
    }

    // M==2 shortcut: hypervolume(front, ref) with M=2 must equal
    // hypervolume_2d(front, ref) EXACTLY (bit-identical), since hypervolume
    // literally delegates to it (module doc, "2D shortcut") -- this is a
    // sanity check on the delegation itself, distinct from the property
    // test below (which instead exercises the general recursive engine,
    // hv_recursive, bypassing this shortcut).
    #[test]
    fn hypervolume_m2_shortcut_matches_hypervolume_2d_exactly() {
        let front = vec![vec![0.25, 0.75], vec![0.5, 0.5], vec![0.75, 0.25]];
        let ref_point = [1.0, 1.0];

        let via_general = hypervolume(&front, &ref_point).unwrap();
        let via_2d = hypervolume_2d(&front, &ref_point).unwrap();
        assert_eq!(via_general.to_bits(), via_2d.to_bits());
    }

    // Property test: the GENERAL recursive engine (hv_recursive, private),
    // exercised directly on random 2D mutually-nondominated fronts (i.e.
    // deliberately bypassing hypervolume()'s own M==2 shortcut, which would
    // make this check trivial-by-construction), must agree with the frozen,
    // independently-derived hypervolume_2d. Front generation and pre-filter
    // (points that strictly dominate ref_point) mirror hypervolume()'s own
    // logic, since hv_recursive itself performs no filtering.
    //
    // Deterministic PRNG (sezgi-core's RngStream, already a dependency of
    // this crate) for reproducibility; measures and asserts the maximum
    // observed relative difference across all trials, printed on failure so
    // a future tightening of the bound has a concrete number to start from.
    #[test]
    fn hypervolume_2d_property_agreement_with_hv_recursive() {
        use sezgi_core::rng::RngStream;

        let mut rng = RngStream::from_master(0xE2D_A9F1, &[8, 0]);
        let ref_point = [10.0, 10.0];
        let mut max_rel_diff = 0.0_f64;

        for trial in 0..500 {
            // Staircase construction: n distinct x-values sorted ascending,
            // paired with n distinct y-values sorted descending. This is
            // guaranteed mutually-non-dominated by construction (strictly
            // increasing x <-> strictly decreasing y), unlike rejection
            // sampling -- which was tried first and found to be
            // computationally intractable here: as a mutually-nondominated
            // front's own hypervolume approaches the full ref-box volume
            // (already >99% by n=20 for this ref_point, confirmed with a
            // scratchpad probe), the probability that one more uniformly
            // random point survives (is not already weakly dominated)
            // collapses, so naive rejection sampling stalls indefinitely
            // for exactly the larger n values this property test wants.
            let n = 2 + (rng.next_below(50) as usize);
            let mut xs: Vec<f64> = (0..n).map(|_| rng.next_f64() * 9.9).collect();
            let mut ys: Vec<f64> = (0..n).map(|_| rng.next_f64() * 9.9).collect();
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            ys.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let front: Vec<Vec<f64>> = xs.into_iter().zip(ys).map(|(x, y)| vec![x, y]).collect();

            let via_2d = hypervolume_2d(&front, &ref_point).expect("valid random front");

            let candidates: Vec<Vec<f64>> = front
                .iter()
                .filter(|p| p.iter().zip(ref_point).all(|(pi, ri)| *pi < ri))
                .cloned()
                .collect();
            let via_recursive = hv_recursive(&candidates, &ref_point);

            let diff = (via_2d - via_recursive).abs();
            let rel_diff = if via_2d.abs() > 0.0 {
                diff / via_2d.abs()
            } else {
                diff
            };
            max_rel_diff = max_rel_diff.max(rel_diff);
            assert!(
                rel_diff < 1e-9,
                "trial {trial}: hv_recursive={via_recursive}, hypervolume_2d={via_2d}, rel_diff={rel_diff}, front={front:?}"
            );
        }

        // Anchored measurement: observed max relative difference across
        // 500 random 2D fronts
        // (n in [2,51]) was 6.5e-16 -- i.e. a couple of ULPs, essentially
        // machine-epsilon level, NOT exact/bit-identical (two independently
        // derived algorithms -- a direct sort-sweep vs. a recursive
        // inclusion-exclusion -- take different floating-point operation
        // orders) but far tighter than the 1e-9 bound asserted below, which
        // is left with margin rather than pinned to the observed value so
        // the test does not become flaky on a different trial sequence.
        // This is a materially different (much tighter) agreement class
        // than the M==2 shortcut test above, which IS bit-identical by
        // construction (same function call, not independent algorithms).
        eprintln!("hypervolume_2d_property_agreement_with_hv_recursive: max_rel_diff={max_rel_diff:e} over 500 trials");
        assert!(
            max_rel_diff < 1e-9,
            "max relative difference across all trials: {max_rel_diff}"
        );
    }

    // Hand fixture: 2 reference points, 2 front points, distances computed
    // by hand.
    //
    // reference_front = [(0,0), (4,0)], front = [(1,0), (3,0)].
    //
    // z=(0,0): d to (1,0) = sqrt((0-1)^2 + (0-0)^2) = sqrt(1) = 1.0
    //          d to (3,0) = sqrt((0-3)^2 + (0-0)^2) = sqrt(9) = 3.0
    //          min = 1.0  (nearest front point is (1,0))
    // z=(4,0): d to (1,0) = sqrt((4-1)^2 + (0-0)^2) = sqrt(9) = 3.0
    //          d to (3,0) = sqrt((4-3)^2 + (0-0)^2) = sqrt(1) = 1.0
    //          min = 1.0  (nearest front point is (3,0), different from z=(0,0)'s)
    //
    // IGD = (1.0 + 1.0) / 2 = 1.0, exact (both nearest distances are
    // sqrt(1) = 1.0 exactly).
    #[test]
    fn igd_hand_fixture_exact() {
        let front = vec![vec![1.0, 0.0], vec![3.0, 0.0]];
        let reference_front = vec![vec![0.0, 0.0], vec![4.0, 0.0]];

        let value = igd(&front, &reference_front).expect("valid fixture");
        assert_eq!(value, 1.0);
    }

    #[test]
    fn igd_is_zero_when_front_covers_reference_front() {
        let reference_front = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 0.5]];
        // front is a superset of reference_front (order doesn't matter).
        let front = vec![
            vec![5.0, 5.0],
            vec![2.0, 0.5],
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        ];

        let value = igd(&front, &reference_front).expect("valid fixture");
        assert_eq!(value, 0.0);
    }

    #[test]
    fn igd_is_deterministic() {
        let front = vec![vec![1.0, 0.0], vec![3.0, 0.0]];
        let reference_front = vec![vec![0.0, 0.0], vec![4.0, 0.0]];

        let v1 = igd(&front, &reference_front).unwrap();
        let v2 = igd(&front, &reference_front).unwrap();
        assert_eq!(v1.to_bits(), v2.to_bits());
    }

    #[test]
    fn igd_errors_on_empty_front() {
        let front: Vec<Vec<f64>> = vec![];
        let reference_front = vec![vec![0.0, 0.0]];
        let err = igd(&front, &reference_front).expect_err("empty front must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("igd"), "message: {msg}");
        assert!(msg.contains("front is empty"), "message: {msg}");
    }

    #[test]
    fn igd_errors_on_empty_reference_front() {
        let front = vec![vec![0.0, 0.0]];
        let reference_front: Vec<Vec<f64>> = vec![];
        let err = igd(&front, &reference_front).expect_err("empty reference_front rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("igd"), "message: {msg}");
        assert!(msg.contains("reference_front is empty"), "message: {msg}");
    }

    #[test]
    fn igd_errors_on_dimension_mismatch() {
        let front = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let reference_front = vec![vec![0.0, 0.0, 0.0]];
        let err = igd(&front, &reference_front).expect_err("dimension mismatch rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("igd"), "message: {msg}");
        assert!(msg.contains("expected 2"), "message: {msg}");
    }

    #[test]
    fn igd_errors_on_ragged_front() {
        let front = vec![vec![0.0, 0.0], vec![1.0, 1.0, 1.0]];
        let reference_front = vec![vec![0.0, 0.0]];
        let err = igd(&front, &reference_front).expect_err("ragged front rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("igd"), "message: {msg}");
        assert!(msg.contains("ragged"), "message: {msg}");
    }

    #[test]
    fn igd_errors_on_non_finite_value() {
        let front = vec![vec![0.0, 0.0], vec![f64::NAN, 1.0]];
        let reference_front = vec![vec![0.0, 0.0]];
        let err = igd(&front, &reference_front).expect_err("NaN must be rejected");
        let StatsError::InvalidInput(msg) = err;
        assert!(msg.contains("igd"), "message: {msg}");
        assert!(msg.contains("non-finite"), "message: {msg}");
    }
}
