//! Pareto-dominance sorting and crowding-distance estimation for NSGA-II
//! (Deb, Pratap, Agarwal & Meyarivan, "A Fast and Elitist Multiobjective
//! Genetic Algorithm: NSGA-II", *IEEE Transactions on Evolutionary
//! Computation* 6(2):182-197, April 2002; PDF verified via
//! <https://www.cse.unr.edu/~sushil/class/gas/papers/nsga2.pdf>, a mirror of
//! the IEEE original, pp. 184-185).
//!
//! ## Layout
//! This module holds pure, engine-free, problem-free routines: Task 4's
//! Pareto dominance, fast non-dominated sorting, and crowding-distance
//! assignment, plus Task 5's variation operators (`sbx_pair`,
//! `polynomial_mutation`) below. They operate on plain `&[Vec<f64>]`
//! objective matrices / `&[f64]` decision vectors (minimization) -- no
//! `Ctx`, `Registry`, or `ComponentMeta` -- because NSGA-II is not a
//! scalar-graph component here; the recorded architecture plan runs
//! multi-objective search through a dedicated MO runner instead. The runner
//! itself -- including the tournament selection that consumes the
//! crowded-comparison operator pinned below -- arrives in a later task of
//! this milestone.
//!
//! ## Pareto dominance
//! Pinned to the standard minimization definition, consistent with the
//! paper's unglossed `p ≺ q` ("p dominates q") notation used throughout
//! Section III-A below: `a` dominates `b` iff `a[i] <= b[i]` for every
//! objective `i`, and `a[i] < b[i]` for at least one `i`.
//!
//! ## Fast non-dominated sort (paper, p. 184, `fast-non-dominated-sort(P)`)
//! Quoted verbatim from the algorithm box (comments after `if`/assignment
//! lines are the paper's own right-column annotations):
//! ```text
//! fast-non-dominated-sort(P)
//! for each p in P
//!   S_p = empty
//!   n_p = 0
//!   for each q in P
//!     if (p dominates q) then
//!       S_p = S_p union {q}              Add q to the set of solutions dominated by p
//!     else if (q dominates p) then
//!       n_p = n_p + 1                    Increment the domination counter of p
//!   if n_p = 0 then                      p belongs to the first front
//!     p_rank = 1
//!     F_1 = F_1 union {p}
//! i = 1                                  Initialize the front counter
//! while F_i != empty
//!   Q = empty                            Used to store the members of the next front
//!   for each p in F_i
//!     for each q in S_p
//!       n_q = n_q - 1
//!       if n_q = 0 then                  q belongs to the next front
//!         q_rank = i + 1
//!         Q = Q union {q}
//!   i = i + 1
//!   F_i = Q
//! ```
//! `fast_non_dominated_sort` below implements this exactly (0-indexed
//! fronts: the paper's `F_1` is `fronts[0]` here). Ordering is pinned for
//! reproducibility, since the paper does not specify a within-front output
//! order: front 0 lists indices in ascending original-index order (the
//! scan order of the outer `for each p in P` loop); every later front
//! lists indices in the order they are discovered while walking `F_i` in
//! its stored order and, for each `p`, each `S_p` member in the
//! ascending-index order `S_p` was built in during the initial scan.
//!
//! ## Crowding distance (paper, p. 185, `crowding-distance-assignment(I)`)
//! Quoted verbatim from the algorithm box:
//! ```text
//! crowding-distance-assignment(I)
//! l = |I|                                          number of solutions in I
//! for each i, set I[i]_distance = 0                initialize distance
//! for each objective m
//!   I = sort(I, m)                                 sort using each objective value
//!   I[1]_distance = I[l]_distance = infinity        so that boundary points are always selected
//!   for i = 2 to (l - 1)
//!     I[i]_distance = I[i]_distance
//!         + (I[i+1].m - I[i-1].m) / (f_max_m - f_min_m)   for all other points
//! ```
//! (`f_max_m`, `f_min_m` are the paper's `f_m^max`/`f_m^min`: the max/min
//! values of objective `m` over `I`. The paper also notes each objective is
//! normalized before this computation, which the `(f_max_m - f_min_m)`
//! division already achieves per-objective, so no separate pre-pass is
//! needed here.)
//!
//! **f_max_m == f_min_m guard.** The paper does not say what to do when an
//! objective is constant across the whole front (division by zero in the
//! formula above) -- it is silent on this edge case. Pinned instead to the
//! reference implementation: Dr. Kalyanmoy Deb's original NSGA-II C code
//! (KanGAL), as mirrored at
//! <https://raw.githubusercontent.com/darnir/nsga2/master/crowddist.c>,
//! function `assign_crowding_distance`:
//! ```text
//! if (pop->ind[obj_array[i][front_size-1]].obj[i] ==
//!     pop->ind[obj_array[i][0]].obj[i])
//! {
//!     pop->ind[obj_array[i][j]].crowd_dist += 0.0;
//! }
//! else
//! {
//!     pop->ind[obj_array[i][j]].crowd_dist +=
//!         (pop->ind[obj_array[i][j+1]].obj[i] - pop->ind[obj_array[i][j-1]].obj[i])
//!         / (pop->ind[obj_array[i][front_size-1]].obj[i]
//!            - pop->ind[obj_array[i][0]].obj[i]);
//! }
//! ```
//! i.e. when `f_max_m == f_min_m`, that objective's contribution for every
//! interior point is treated as `0.0` (skipped) rather than dividing by
//! zero. `crowding_distance` below pins the same convention -- see the
//! `// sezgi decision:` comment at the guard site. Boundary points (first
//! and last after sorting) still get `f64::INFINITY` unconditionally, per
//! the algorithm above, regardless of whether the range is zero.
//!
//! A front of size <= 2 needs no separate RESULT rule (the implementation
//! does take an explicit early-return branch for it, with an identical
//! outcome to the general path): with `l <= 2` the interior
//! loop `2..=(l-1)` never runs, and the boundary assignment
//! `I[1]_distance = I[l]_distance = infinity` covers every member (for
//! `l == 1`, position 1 and position `l` are the same element; for
//! `l == 2`, they are the only two elements). This is a natural fallout of
//! the general algorithm above, matching the reference C code's explicit
//! `front_size <= 2` shortcut without needing one here.
//!
//! Sort determinism: ties within an objective are broken by the tied
//! elements' original position in `front` (Rust's `slice::sort_by` is
//! stable, and the tie-break below makes this explicit rather than
//! implicit), so results are deterministic for a given `front` order --
//! required since the paper does not specify a tie-break and one must be
//! pinned for reproducible tests.
//!
//! ## Crowded-comparison operator (paper, p. 185)
//! Quoted:
//! ```text
//! i <_n j   if (i_rank < j_rank)
//!           or ((i_rank = j_rank) and (i_distance > j_distance))
//! ```
//! i.e. lower rank wins; ties broken by larger crowding distance. This
//! definition is pinned here for the record; it is *used* by the T6
//! tournament selector, implemented when the MO runner lands, not in this
//! module.
//!
//! ## SBX crossover and polynomial mutation (Task 5)
//! The β/δ polynomials themselves are standard (Deb & Agrawal 1995,
//! "Simulated Binary Crossover for Continuous Search Space", *Complex
//! Systems* 9:115-148; Deb & Goyal 1996 for the mutation operator), but
//! implementations differ in DRAW STRUCTURE (which gates exist, in what
//! order, and how many uniform draws each variable consumes). Per the M2d-4
//! WOA lesson noted in the task brief, that structure is pinned to the
//! reference C code rather than derived from the equations: the same
//! KanGAL mirror T4 pinned for crowding distance,
//! `darnir/nsga2` on GitHub, files `crossover.c` (`realcross`) and
//! `mutation.c` (`real_mutate_ind`), fetched and verified against the
//! quotes below (also `global.h` for the `EPS` constant).
//!
//! **`realcross` (`crossover.c`), quoted (types/braces elided where
//! obvious):**
//! ```text
//! if (randomperc() <= pcross_real)
//! {
//!     for (i=0; i<nreal; i++)
//!     {
//!         if (randomperc()<=0.5)
//!         {
//!             if (fabs(parent1->xreal[i]-parent2->xreal[i]) > EPS)
//!             {
//!                 if (parent1->xreal[i] < parent2->xreal[i])
//!                 { y1 = parent1->xreal[i]; y2 = parent2->xreal[i]; }
//!                 else
//!                 { y1 = parent2->xreal[i]; y2 = parent1->xreal[i]; }
//!                 yl = min_realvar[i];
//!                 yu = max_realvar[i];
//!                 rand = randomperc();
//!                 beta = 1.0 + (2.0*(y1-yl)/(y2-y1));
//!                 alpha = 2.0 - pow(beta,-(eta_c+1.0));
//!                 if (rand <= (1.0/alpha))
//!                     betaq = pow((rand*alpha),(1.0/(eta_c+1.0)));
//!                 else
//!                     betaq = pow((1.0/(2.0 - rand*alpha)),(1.0/(eta_c+1.0)));
//!                 c1 = 0.5*((y1+y2)-betaq*(y2-y1));
//!                 beta = 1.0 + (2.0*(yu-y2)/(y2-y1));
//!                 alpha = 2.0 - pow(beta,-(eta_c+1.0));
//!                 if (rand <= (1.0/alpha))
//!                     betaq = pow((rand*alpha),(1.0/(eta_c+1.0)));
//!                 else
//!                     betaq = pow((1.0/(2.0 - rand*alpha)),(1.0/(eta_c+1.0)));
//!                 c2 = 0.5*((y1+y2)+betaq*(y2-y1));
//!                 if (c1<yl) c1=yl;   if (c2<yl) c2=yl;
//!                 if (c1>yu) c1=yu;   if (c2>yu) c2=yu;
//!                 if (randomperc()<=0.5)
//!                 { child1->xreal[i] = c2; child2->xreal[i] = c1; }
//!                 else
//!                 { child1->xreal[i] = c1; child2->xreal[i] = c2; }
//!             }
//!             else
//!             { child1->xreal[i] = parent1->xreal[i]; child2->xreal[i] = parent2->xreal[i]; }
//!         }
//!         else
//!         { child1->xreal[i] = parent1->xreal[i]; child2->xreal[i] = parent2->xreal[i]; }
//!     }
//! }
//! else /* whole pair: */
//! {
//!     for (i=0; i<nreal; i++)
//!     { child1->xreal[i] = parent1->xreal[i]; child2->xreal[i] = parent2->xreal[i]; }
//! }
//! ```
//! `global.h`: `#define EPS 1.0e-14`.
//!
//! **Pinned draw structure, `sbx_pair`:**
//! - Step 1: ONE draw for the whole pair: `randomperc() <= pcross_real`
//!   (here, `next_f64() <= p_c`). On failure the pair is copied verbatim,
//!   both children, **zero further draws for the whole pair** -- the loop
//!   over variables never runs at all in the C code's `else` branch.
//! - Step 2, on success, per variable, in order:
//!   - Step 2a: ONE draw, the 0.5 exchange gate (`next_f64() <= 0.5`). On
//!     failure, copy this variable from both parents verbatim -- no
//!     further draws for this variable.
//!   - Step 2b, on success: the EPS guard `|p1[i]-p2[i]| > EPS` -- NOT a
//!     draw, a comparison. If it fails (parents effectively equal), copy
//!     this variable verbatim -- no further draws for this variable.
//!   - Step 2c, on success (parents differ): ONE draw, `rand = randomperc()`.
//!     Verified from the quote above: this single draw is **reused
//!     verbatim** for BOTH child computations (the c1-side beta/alpha/
//!     betaq block and the c2-side beta/alpha/betaq block each test and
//!     consume the SAME `rand` value -- `rand` is not redrawn between
//!     them). betaq for child "1" uses `beta = 1 + 2*(y1-yl)/(y2-y1)`
//!     (distance from the smaller parent y1 to the nearer bound yl); betaq
//!     for child "2" uses the symmetric `beta = 1 + 2*(yu-y2)/(y2-y1)`
//!     (distance from the larger parent y2 to the nearer bound yu); each
//!     branches on `rand <= 1/alpha` with `alpha = 2 - beta^-(eta_c+1)`,
//!     giving `betaq = (rand*alpha)^(1/(eta_c+1))` or
//!     `betaq = (1/(2-rand*alpha))^(1/(eta_c+1))`; children
//!     `c1 = 0.5*((y1+y2)-betaq1*(y2-y1))`, `c2 = 0.5*((y1+y2)+betaq2*(y2-y1))`;
//!     both clamped to `[yl,yu]`.
//!   - Step 2d: ONE more draw, the final per-variable swap gate
//!     (`randomperc() <= 0.5`). Verified: the C code SWAPS on success
//!     (`child1=c2, child2=c1`) and assigns in c1/c2 order on failure
//!     (`child1=c1, child2=c2`) -- it is a swap gate, not a plain
//!     assign-by-draw.
//! - **Draw count per variable** (pair-gate already passed): 1 (exchange
//!   gate fails) / 1 (exchange gate passes, EPS guard fails) / 3 (exchange
//!   gate passes, EPS guard passes: 1 exchange + 1 `rand` + 1 swap).
//! - **Draw count for the whole pair**: 1 (pair gate fails) or
//!   `1 + sum(per-variable draws above)` (pair gate passes).
//!
//! **`real_mutate_ind` (`mutation.c`), quoted:**
//! ```text
//! for (j=0; j<nreal; j++)
//! {
//!     if (randomperc() <= pmut_real)
//!     {
//!         y = ind->xreal[j];
//!         yl = min_realvar[j];
//!         yu = max_realvar[j];
//!         delta1 = (y-yl)/(yu-yl);
//!         delta2 = (yu-y)/(yu-yl);
//!         rnd = randomperc();
//!         mut_pow = 1.0/(eta_m+1.0);
//!         if (rnd <= 0.5)
//!         {
//!             xy = 1.0-delta1;
//!             val = 2.0*rnd+(1.0-2.0*rnd)*(pow(xy,(eta_m+1.0)));
//!             deltaq =  pow(val,mut_pow) - 1.0;
//!         }
//!         else
//!         {
//!             xy = 1.0-delta2;
//!             val = 2.0*(1.0-rnd)+2.0*(rnd-0.5)*(pow(xy,(eta_m+1.0)));
//!             deltaq = 1.0 - (pow(val,mut_pow));
//!         }
//!         y = y + deltaq*(yu-yl);
//!         if (y<yl) y = yl;
//!         if (y>yu) y = yu;
//!         ind->xreal[j] = y;
//!     }
//! }
//! ```
//! **Pinned draw structure, `polynomial_mutation`:** per variable, in
//! order: ONE draw, the gate `randomperc() <= pmut_real` (here,
//! `next_f64() <= p_m`); on failure, the variable is left unchanged, no
//! further draw for it. On success, ONE more draw, `rnd = randomperc()`,
//! which both branches (`rnd <= 0.5` vs else) the `deltaq` formula AND
//! supplies the numeric value used inside it (there is no separate
//! branch-selector draw distinct from the value draw). **Draw count per
//! variable**: 1 (gate fails) or 2 (gate passes: 1 gate + 1 `rnd`).
//!
//! `sbx_pair` and `polynomial_mutation` below implement exactly this
//! structure; any single-line deviation is marked with a
//! `// sezgi decision:` or `// sezgi simplification:` comment at the site.

use sezgi_core::rng::RngStream;

/// Pareto-dominance for minimization: `a` dominates `b` iff `a <= b` in
/// every objective and `a < b` in at least one. See the module doc for the
/// paper citation this is pinned to.
pub fn dominates(a: &[f64], b: &[f64]) -> bool {
    debug_assert_eq!(a.len(), b.len(), "dominates: objective vectors must have equal length");
    let mut strictly_better_somewhere = false;
    for i in 0..a.len() {
        if a[i] > b[i] {
            return false;
        }
        if a[i] < b[i] {
            strictly_better_somewhere = true;
        }
    }
    strictly_better_somewhere
}

/// Fronts as index lists, front 0 = non-dominated set, per Deb et al. 2002
/// (`fast-non-dominated-sort`, module doc). Ordering within and across
/// fronts is pinned; see the module doc for the exact rule.
pub fn fast_non_dominated_sort(objectives: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = objectives.len();
    let mut dominated_sets: Vec<Vec<usize>> = vec![Vec::new(); n]; // S_p
    let mut domination_count: Vec<usize> = vec![0; n]; // n_p
    let mut front0 = Vec::new();

    for p in 0..n {
        for q in 0..n {
            if p == q {
                continue;
            }
            if dominates(&objectives[p], &objectives[q]) {
                dominated_sets[p].push(q);
            } else if dominates(&objectives[q], &objectives[p]) {
                domination_count[p] += 1;
            }
        }
        if domination_count[p] == 0 {
            front0.push(p);
        }
    }

    let mut fronts = Vec::new();
    let mut current = front0;
    while !current.is_empty() {
        let mut next = Vec::new();
        for &p in &current {
            for &q in &dominated_sets[p] {
                domination_count[q] -= 1;
                if domination_count[q] == 0 {
                    next.push(q);
                }
            }
        }
        fronts.push(current);
        current = next;
    }
    fronts
}

/// Crowding distance for ONE front (indices into `objectives`); boundary
/// individuals get `f64::INFINITY`, per the verified source (module doc,
/// `crowding-distance-assignment`). Returns one distance per entry of
/// `front`, in `front`'s order (i.e. `result[i]` is the crowding distance
/// of `front[i]`).
pub fn crowding_distance(front: &[usize], objectives: &[Vec<f64>]) -> Vec<f64> {
    let l = front.len();
    let mut distance = vec![0.0f64; l];
    if l <= 2 {
        // Natural fallout of the general algorithm (see module doc): with
        // l <= 2, every position is a sort boundary.
        distance.fill(f64::INFINITY);
        return distance;
    }

    let m = objectives[front[0]].len();
    #[allow(clippy::needless_range_loop)] // `obj` indexes the objective column, not `objectives` itself
    for obj in 0..m {
        // Sort positions 0..l (into `front`) ascending by this objective;
        // ties broken by original position -- pinned, deterministic (see
        // module doc).
        let mut order: Vec<usize> = (0..l).collect();
        order.sort_by(|&a, &b| {
            objectives[front[a]][obj]
                .total_cmp(&objectives[front[b]][obj])
                .then(a.cmp(&b))
        });

        let f_min = objectives[front[order[0]]][obj];
        let f_max = objectives[front[order[l - 1]]][obj];
        distance[order[0]] = f64::INFINITY;
        distance[order[l - 1]] = f64::INFINITY;

        let range = f_max - f_min;
        for i in 1..l - 1 {
            let contrib = if range == 0.0 {
                // sezgi decision: the paper is silent on f_max == f_min;
                // pinned to the KanGAL reference C code's convention
                // (crowddist.c, `assign_crowding_distance`, quoted in the
                // module doc) -- skip this objective's contribution rather
                // than dividing by zero.
                0.0
            } else {
                let next = objectives[front[order[i + 1]]][obj];
                let prev = objectives[front[order[i - 1]]][obj];
                (next - prev) / range
            };
            distance[order[i]] += contrib;
        }
    }
    distance
}

/// Simulated Binary Crossover (SBX), bounded form, on one Float-vector
/// pair. Pinned to the KanGAL reference C code `crossover.c`, function
/// `realcross` (see the module doc's "SBX crossover and polynomial
/// mutation" section for the exact quoted structure and per-branch draw
/// counts). `lo`/`hi` are per-variable bounds (same length as `p1`/`p2`).
/// Returns `(child1, child2)`.
pub fn sbx_pair(
    p1: &[f64],
    p2: &[f64],
    lo: &[f64],
    hi: &[f64],
    eta_c: f64,
    p_c: f64,
    rng: &mut RngStream,
) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(p1.len(), p2.len(), "sbx_pair: parent vectors must have equal length");
    debug_assert_eq!(p1.len(), lo.len(), "sbx_pair: bounds must match parent length");
    debug_assert_eq!(p1.len(), hi.len(), "sbx_pair: bounds must match parent length");
    let n = p1.len();

    // Whole-pair gate: ONE draw for the whole pair (realcross's
    // `if (randomperc() <= pcross_real)`). On failure, copy both parents
    // verbatim -- zero further draws for the whole pair (the C code's
    // per-variable loop is in the `else` branch, never entered here).
    if rng.next_f64() > p_c {
        return (p1.to_vec(), p2.to_vec());
    }

    // sezgi decision: EPS pinned to the KanGAL reference's own constant
    // (global.h: `#define EPS 1.0e-14`), not re-derived.
    const EPS: f64 = 1.0e-14;

    let mut c1 = Vec::with_capacity(n);
    let mut c2 = Vec::with_capacity(n);
    for i in 0..n {
        // Per-variable exchange gate: ONE draw.
        if rng.next_f64() > 0.5 {
            c1.push(p1[i]);
            c2.push(p2[i]);
            continue;
        }

        // EPS guard: parents effectively equal in this variable -> copy
        // verbatim, no further draws for this variable.
        if (p1[i] - p2[i]).abs() <= EPS {
            c1.push(p1[i]);
            c2.push(p2[i]);
            continue;
        }

        let (y1, y2) = if p1[i] < p2[i] { (p1[i], p2[i]) } else { (p2[i], p1[i]) };
        let yl = lo[i];
        let yu = hi[i];

        // ONE draw, reused verbatim for BOTH children's betaq computation
        // (verified against realcross: `rand` is drawn once, read twice).
        let rand = rng.next_f64();
        // sezgi simplification: 1/(eta+1) hoisted out of the per-branch pow calls (the C recomputes it inline each time); numerically inert.
        let inv_eta1 = 1.0 / (eta_c + 1.0);

        // Child "1" side: distance from the smaller parent y1 to the
        // nearer bound yl.
        let beta1 = 1.0 + 2.0 * (y1 - yl) / (y2 - y1);
        let alpha1 = 2.0 - beta1.powf(-(eta_c + 1.0));
        let betaq1 = if rand <= 1.0 / alpha1 {
            (rand * alpha1).powf(inv_eta1)
        } else {
            (1.0 / (2.0 - rand * alpha1)).powf(inv_eta1)
        };
        let mut v1 = 0.5 * ((y1 + y2) - betaq1 * (y2 - y1));

        // Child "2" side: symmetric form, distance from the larger parent
        // y2 to the nearer bound yu.
        let beta2 = 1.0 + 2.0 * (yu - y2) / (y2 - y1);
        let alpha2 = 2.0 - beta2.powf(-(eta_c + 1.0));
        let betaq2 = if rand <= 1.0 / alpha2 {
            (rand * alpha2).powf(inv_eta1)
        } else {
            (1.0 / (2.0 - rand * alpha2)).powf(inv_eta1)
        };
        let mut v2 = 0.5 * ((y1 + y2) + betaq2 * (y2 - y1));

        v1 = v1.clamp(yl, yu);
        v2 = v2.clamp(yl, yu);

        // Final per-variable swap gate: ONE draw. Verified: the C code
        // SWAPS on success (child1=c2, child2=c1), assigns in order on
        // failure (child1=c1, child2=c2) -- a swap gate, not a plain
        // assign-by-draw.
        if rng.next_f64() <= 0.5 {
            c1.push(v2);
            c2.push(v1);
        } else {
            c1.push(v1);
            c2.push(v2);
        }
    }
    (c1, c2)
}

/// Bounded polynomial mutation, in place. Pinned to the KanGAL reference C
/// code `mutation.c`, function `real_mutate_ind` (see the module doc's
/// "SBX crossover and polynomial mutation" section for the exact quoted
/// structure and per-branch draw counts). `lo`/`hi` are per-variable bounds
/// (same length as `x`).
pub fn polynomial_mutation(
    x: &mut [f64],
    lo: &[f64],
    hi: &[f64],
    eta_m: f64,
    p_m: f64,
    rng: &mut RngStream,
) {
    debug_assert_eq!(x.len(), lo.len(), "polynomial_mutation: bounds must match x length");
    debug_assert_eq!(x.len(), hi.len(), "polynomial_mutation: bounds must match x length");
    let n = x.len();
    let mut_pow = 1.0 / (eta_m + 1.0);

    for j in 0..n {
        // Per-variable gate: ONE draw. On failure, leave the variable
        // unchanged -- no further draw for it.
        if rng.next_f64() > p_m {
            continue;
        }

        let y = x[j];
        let yl = lo[j];
        let yu = hi[j];
        let delta1 = (y - yl) / (yu - yl);
        let delta2 = (yu - y) / (yu - yl);

        // ONE more draw: `rnd` both branches the deltaq formula AND
        // supplies the numeric value used inside it -- no separate
        // branch-selector draw distinct from the value draw.
        let rnd = rng.next_f64();
        let deltaq = if rnd <= 0.5 {
            let xy = 1.0 - delta1;
            let val = 2.0 * rnd + (1.0 - 2.0 * rnd) * xy.powf(eta_m + 1.0);
            val.powf(mut_pow) - 1.0
        } else {
            let xy = 1.0 - delta2;
            let val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * xy.powf(eta_m + 1.0);
            1.0 - val.powf(mut_pow)
        };

        let y_new = (y + deltaq * (yu - yl)).clamp(yl, yu);
        x[j] = y_new;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- dominates: truth table ------------------------------------

    #[test]
    fn dominates_strict() {
        assert!(dominates(&[1.0, 2.0], &[2.0, 3.0]));
        assert!(!dominates(&[2.0, 3.0], &[1.0, 2.0]));
    }

    #[test]
    fn dominates_equal_vectors_neither() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0];
        assert!(!dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn dominates_mixed_better_worse_neither() {
        // a better in obj0 (1<2), worse in obj1 (3>2): incomparable.
        let a = [1.0, 3.0];
        let b = [2.0, 2.0];
        assert!(!dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    #[test]
    fn dominates_one_better_rest_equal() {
        // Equal in obj0, obj1; strictly better in obj2 -> dominates.
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 4.0];
        assert!(dominates(&a, &b));
        assert!(!dominates(&b, &a));
    }

    // ---- fast_non_dominated_sort: hand fixture ----------------------
    //
    // 6 points, 2 objectives (minimize both):
    //   0: (1, 6)   1: (2, 4)   2: (3, 3)   3: (4, 2)   4: (5, 1)   5: (3, 5)
    //
    // Points 0..4 form a strict Pareto staircase (obj0 strictly increases,
    // obj1 strictly decreases): every pair among them is incomparable
    // (mixed domination -- one better in obj0, worse in obj1, or vice
    // versa), so they are all mutually non-dominated -> front 0.
    //
    // Point 5 = (3, 5) ties point 2's obj0 (both 3) -- the "tie in one
    // objective" case -- and is dominated by point 2: obj0 3<=3 (tied),
    // obj1 3<=5 and 3<5, so point2 <= point5 everywhere and strictly less
    // in obj1 -> point 2 dominates point 5. Point 5 is also dominated by
    // point 1 = (2, 4): obj0 2<=3, obj1 4<=5, 4<5 -> dominates. Point 5 is
    // NOT dominated by point 0 = (1, 6): obj1 6<=5 is false. Once front 0
    // is removed, nothing remains to dominate point 5, so it is alone in
    // front 1.
    fn six_point_fixture() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 6.0],
            vec![2.0, 4.0],
            vec![3.0, 3.0],
            vec![4.0, 2.0],
            vec![5.0, 1.0],
            vec![3.0, 5.0],
        ]
    }

    #[test]
    fn sort_six_point_fixture_exact_fronts() {
        let objectives = six_point_fixture();
        let fronts = fast_non_dominated_sort(&objectives);
        assert_eq!(fronts, vec![vec![0, 1, 2, 3, 4], vec![5]]);
    }

    // ---- fast_non_dominated_sort: properties -------------------------

    fn assert_partition_and_ordering_invariant(objectives: &[Vec<f64>], fronts: &[Vec<usize>]) {
        let n = objectives.len();

        // Every index appears in exactly one front.
        let mut seen = vec![false; n];
        let mut total = 0;
        for front in fronts {
            for &idx in front {
                assert!(!seen[idx], "index {idx} appeared in more than one front");
                seen[idx] = true;
                total += 1;
            }
        }
        assert_eq!(total, n, "not every index was placed in a front");
        assert!(seen.iter().all(|&s| s), "some index was never placed in a front");

        // Front-ordering invariant: for every p in front k>0, at least one
        // member of front k-1 dominates p.
        for k in 1..fronts.len() {
            for &p in &fronts[k] {
                let dominated_by_prev = fronts[k - 1]
                    .iter()
                    .any(|&prev| dominates(&objectives[prev], &objectives[p]));
                assert!(
                    dominated_by_prev,
                    "index {p} in front {k} is not dominated by any member of front {}",
                    k - 1
                );
            }
        }
    }

    #[test]
    fn six_point_fixture_satisfies_properties() {
        let objectives = six_point_fixture();
        let fronts = fast_non_dominated_sort(&objectives);
        assert_partition_and_ordering_invariant(&objectives, &fronts);
    }

    #[test]
    fn seeded_random_matrix_satisfies_properties() {
        use sezgi_core::rng::RngStream;
        let mut rng = RngStream::from_master(42, &[]);
        let n = 25;
        let m = 3;
        let objectives: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..m).map(|_| rng.next_f64() * 10.0).collect())
            .collect();
        let fronts = fast_non_dominated_sort(&objectives);
        assert_partition_and_ordering_invariant(&objectives, &fronts);
    }

    // ---- crowding_distance: hand fixture -----------------------------
    //
    // 4-point front, 2 objectives:
    //   P0=(1,4)  P1=(2,3)  P2=(3,2)  P3=(4,1)
    //
    // This is a Pareto staircase, so P0 and P3 are the sort-boundary in
    // BOTH objectives -> f64::INFINITY.
    //
    // Objective 0 (values 1,2,3,4; sorted order already P0,P1,P2,P3):
    //   f_min=1, f_max=4, range=3.
    //   P1 (neighbors P0=1, P2=3): contributes (3-1)/3 = 2/3.
    //   P2 (neighbors P1=2, P3=4): contributes (4-2)/3 = 2/3.
    //
    // Objective 1 (values 4,3,2,1; ascending sort order is P3,P2,P1,P0):
    //   f_min=1, f_max=4, range=3.
    //   P2 (neighbors P3=1, P1=3): contributes (3-1)/3 = 2/3.
    //   P1 (neighbors P2=2, P0=4): contributes (4-2)/3 = 2/3.
    //
    // Totals: P0 = inf, P1 = 2/3 + 2/3 = 4/3, P2 = 2/3 + 2/3 = 4/3, P3 = inf.
    #[test]
    fn crowding_distance_four_point_hand_fixture() {
        let objectives = vec![
            vec![1.0, 4.0],
            vec![2.0, 3.0],
            vec![3.0, 2.0],
            vec![4.0, 1.0],
        ];
        let front = vec![0, 1, 2, 3];
        let d = crowding_distance(&front, &objectives);
        assert_eq!(d[0], f64::INFINITY);
        assert_eq!(d[3], f64::INFINITY);
        assert!((d[1] - 4.0 / 3.0).abs() < 1e-12);
        assert!((d[2] - 4.0 / 3.0).abs() < 1e-12);
    }

    // ---- crowding_distance: f_max == f_min guard ----------------------
    //
    // Synthetic (deliberately NOT Pareto-valid -- crowding_distance has no
    // domination precondition) 4-point front where objective 1 is constant
    // across the whole front, isolating the f_max==f_min guard:
    //   R0=(1,5)  R1=(2,5)  R2=(3,5)  R3=(4,5)
    //
    // Objective 0: f_min=1, f_max=4, range=3 -- normal formula.
    //   R1 (neighbors R0=1, R2=3): (3-1)/3 = 2/3.
    //   R2 (neighbors R1=2, R3=4): (4-2)/3 = 2/3.
    // Objective 1: f_min=f_max=5, range=0 -- guard triggers, contributes
    //   0.0 for every interior point (R1, R2); boundary points (R0, R3)
    //   still get f64::INFINITY unconditionally (assigned before the
    //   guarded loop runs).
    //
    // Totals: R0 = inf, R1 = 2/3 + 0 = 2/3, R2 = 2/3 + 0 = 2/3, R3 = inf.
    // Without the guard this would be 0/0 = NaN, corrupting the sum.
    #[test]
    fn crowding_distance_f_max_eq_f_min_guard() {
        let objectives = vec![
            vec![1.0, 5.0],
            vec![2.0, 5.0],
            vec![3.0, 5.0],
            vec![4.0, 5.0],
        ];
        let front = vec![0, 1, 2, 3];
        let d = crowding_distance(&front, &objectives);
        assert_eq!(d[0], f64::INFINITY);
        assert_eq!(d[3], f64::INFINITY);
        assert!(d[1].is_finite());
        assert!(d[2].is_finite());
        assert!((d[1] - 2.0 / 3.0).abs() < 1e-12);
        assert!((d[2] - 2.0 / 3.0).abs() < 1e-12);
    }

    // ---- crowding_distance: duplicate objective vectors ----------------
    //
    // Three fully-duplicate points (identical in both objectives) -> both
    // objectives have f_max==f_min across the whole front. Per the general
    // algorithm, boundary positions (first/last after sort, by original
    // position since all values tie) get infinity regardless of the guard;
    // the single interior position gets the guarded 0.0 contribution from
    // both objectives -> finite 0.0. This also matches
    // `fast_non_dominated_sort` placing exact duplicates in the same
    // front (checked separately below).
    #[test]
    fn crowding_distance_duplicate_vectors_middle_is_finite() {
        let objectives = vec![vec![2.0, 5.0], vec![2.0, 5.0], vec![2.0, 5.0]];
        let front = vec![0, 1, 2];
        let d = crowding_distance(&front, &objectives);
        // Stable tie-break by original position -> position 0 and 2 are
        // the sort boundary, position 1 is interior.
        assert_eq!(d[0], f64::INFINITY);
        assert_eq!(d[2], f64::INFINITY);
        assert_eq!(d[1], 0.0);
    }

    #[test]
    fn duplicate_objective_vectors_land_in_same_front() {
        let objectives = vec![vec![2.0, 5.0], vec![2.0, 5.0], vec![2.0, 5.0]];
        let fronts = fast_non_dominated_sort(&objectives);
        // No pair dominates another (all equal) -> all three in front 0.
        assert_eq!(fronts.len(), 1);
        assert_eq!(fronts[0], vec![0, 1, 2]);
    }

    // ---- crowding_distance: determinism -------------------------------

    #[test]
    fn crowding_distance_is_deterministic_across_runs() {
        let objectives = vec![
            vec![1.0, 4.0],
            vec![2.0, 3.0],
            vec![3.0, 2.0],
            vec![4.0, 1.0],
            vec![2.0, 3.0], // duplicate of index 1 -> exercises tie-break stability
        ];
        let front = vec![0, 1, 2, 3, 4];
        let d1 = crowding_distance(&front, &objectives);
        let d2 = crowding_distance(&front, &objectives);
        assert_eq!(d1, d2);
    }

    // ---- degenerate inputs ---------------------------------------------

    #[test]
    fn crowding_distance_empty_front() {
        let objectives: Vec<Vec<f64>> = vec![];
        let front: Vec<usize> = vec![];
        assert_eq!(crowding_distance(&front, &objectives), Vec::<f64>::new());
    }

    #[test]
    fn crowding_distance_single_point_front_is_infinite() {
        let objectives = vec![vec![1.0, 2.0]];
        let front = vec![0];
        assert_eq!(crowding_distance(&front, &objectives), vec![f64::INFINITY]);
    }

    #[test]
    fn crowding_distance_two_point_front_both_boundary() {
        let objectives = vec![vec![1.0, 5.0], vec![2.0, 3.0]];
        let front = vec![0, 1];
        assert_eq!(
            crowding_distance(&front, &objectives),
            vec![f64::INFINITY, f64::INFINITY]
        );
    }

    #[test]
    fn sort_empty_population() {
        let objectives: Vec<Vec<f64>> = vec![];
        assert_eq!(fast_non_dominated_sort(&objectives), Vec::<Vec<usize>>::new());
    }

    // ==================================================================
    // sbx_pair / polynomial_mutation (Task 5)
    // ==================================================================

    // ---- determinism ---------------------------------------------------

    #[test]
    fn sbx_pair_same_seed_bit_identical() {
        let p1 = vec![1.0, -2.5, 3.0];
        let p2 = vec![4.0, 2.5, -1.0];
        let lo = vec![-5.0, -5.0, -5.0];
        let hi = vec![5.0, 5.0, 5.0];
        let run = || {
            let mut rng = RngStream::from_master(42, &[]);
            sbx_pair(&p1, &p2, &lo, &hi, 15.0, 0.9, &mut rng)
        };
        let (a1, a2) = run();
        let (b1, b2) = run();
        assert_eq!(a1, b1, "same seed must produce bit-identical child1");
        assert_eq!(a2, b2, "same seed must produce bit-identical child2");
    }

    #[test]
    fn polynomial_mutation_same_seed_bit_identical() {
        let x0 = vec![1.0, -2.5, 3.0];
        let lo = vec![-5.0, -5.0, -5.0];
        let hi = vec![5.0, 5.0, 5.0];
        let run = || {
            let mut x = x0.clone();
            let mut rng = RngStream::from_master(7, &[]);
            polynomial_mutation(&mut x, &lo, &hi, 20.0, 0.5, &mut rng);
            x
        };
        assert_eq!(run(), run(), "same seed must produce bit-identical mutants");
    }

    // ---- bounds property -------------------------------------------------
    //
    // Seeded batch over mixed per-variable bounds (a ZDT4-style layout:
    // variable 0 in [0,1], variables 1..4 in [-5,5]) and randomly-drawn
    // parents/individuals within those bounds: crossover children and
    // mutants must never leave [lo,hi].

    #[test]
    fn sbx_and_mutation_stay_within_bounds() {
        let lo = vec![0.0, -5.0, -5.0, -5.0, -5.0];
        let hi = vec![1.0, 5.0, 5.0, 5.0, 5.0];
        let dim = lo.len();
        let mut rng = RngStream::from_master(123, &[]);

        let draw_point = |rng: &mut RngStream| -> Vec<f64> {
            (0..dim).map(|j| lo[j] + rng.next_f64() * (hi[j] - lo[j])).collect()
        };

        for _ in 0..300 {
            let p1 = draw_point(&mut rng);
            let p2 = draw_point(&mut rng);
            let (c1, c2) = sbx_pair(&p1, &p2, &lo, &hi, 15.0, 0.9, &mut rng);
            for j in 0..dim {
                assert!(c1[j] >= lo[j] && c1[j] <= hi[j],
                    "child1[{j}]={} outside [{}, {}]", c1[j], lo[j], hi[j]);
                assert!(c2[j] >= lo[j] && c2[j] <= hi[j],
                    "child2[{j}]={} outside [{}, {}]", c2[j], lo[j], hi[j]);
            }

            let mut m1 = c1.clone();
            let mut m2 = c2.clone();
            polynomial_mutation(&mut m1, &lo, &hi, 20.0, 0.4, &mut rng);
            polynomial_mutation(&mut m2, &lo, &hi, 20.0, 0.4, &mut rng);
            for j in 0..dim {
                assert!(m1[j] >= lo[j] && m1[j] <= hi[j],
                    "mutant1[{j}]={} outside [{}, {}]", m1[j], lo[j], hi[j]);
                assert!(m2[j] >= lo[j] && m2[j] <= hi[j],
                    "mutant2[{j}]={} outside [{}, {}]", m2[j], lo[j], hi[j]);
            }
        }
    }

    // ---- distribution sanity, anchored -----------------------------------
    //
    // p1=[0.0], p2=[10.0], eta_c=20, p_c=1.0 (pair gate always passes).
    // Bounds lo=-90, hi=100 are chosen so the bounds midpoint equals the
    // parents' midpoint (yl+yu = 10 = p1+p2): y1-yl = 0-(-90) = 90 and
    // yu-y2 = 100-10 = 90 are EQUAL, so beta1==beta2, alpha1==alpha2, and
    // (since `rand` is the SAME draw reused for both, per the module doc)
    // betaq1==betaq2 exactly on every trial where the compute branch is
    // taken -- giving c1+c2 == y1+y2 exactly (algebraic identity, not a
    // statistical one). On the branch where the exchange gate fails,
    // c1=p1, c2=p2, whose sum is also y1+y2. So (c1+c2)/2 == 5.0 on EVERY
    // trial, exactly (modulo clamping, which the wide bounds relative to
    // eta_c=20's tight spread never trigger here) -- measured over 2000
    // trials (seed 999): max_mid_dev == 0.0 exactly (see reconnaissance
    // probe used to derive this test).
    //
    // Separately, children differ from the parents whenever the 0.5
    // per-variable exchange gate passes (roughly half the time; measured
    // differs_fraction == 0.4625 over these same 2000 trials, well inside
    // a generous [0.35, 0.65] anchor band for a Bernoulli(0.5) proportion
    // at this sample size).
    #[test]
    fn sbx_pair_distribution_sanity_symmetric_pair() {
        let p1 = vec![0.0];
        let p2 = vec![10.0];
        let lo = vec![-90.0];
        let hi = vec![100.0];
        let mut rng = RngStream::from_master(999, &[]);
        let trials = 2000;
        let mut differs = 0usize;
        let mut max_mid_dev: f64 = 0.0;

        for _ in 0..trials {
            let (c1, c2) = sbx_pair(&p1, &p2, &lo, &hi, 20.0, 1.0, &mut rng);
            if c1[0] != p1[0] {
                differs += 1;
            }
            let mid_dev = ((c1[0] + c2[0]) / 2.0 - 5.0).abs();
            if mid_dev > max_mid_dev {
                max_mid_dev = mid_dev;
            }
        }

        assert!(max_mid_dev < 1e-9,
            "symmetric-pair mean must stay at the parents' midpoint (max dev {max_mid_dev})");
        let frac = differs as f64 / trials as f64;
        assert!((0.35..=0.65).contains(&frac),
            "differs_fraction {frac} should be near 0.5 (the per-variable exchange-gate rate)");
    }

    // ---- p_m=0 mutation is identity ---------------------------------------

    #[test]
    fn polynomial_mutation_zero_pm_is_identity() {
        let x0 = vec![1.0, -3.5, 4.9, 0.0];
        let mut x = x0.clone();
        let lo = vec![-5.0, -5.0, -5.0, -5.0];
        let hi = vec![5.0, 5.0, 5.0, 5.0];
        let mut rng = RngStream::from_master(2024, &[]);
        polynomial_mutation(&mut x, &lo, &hi, 20.0, 0.0, &mut rng);
        assert_eq!(x, x0, "p_m=0 must leave every variable unchanged");
    }

    // ---- p_c gate fails: children == parents verbatim ---------------------
    //
    // Seed 1's first raw draw is 0.8116121588818848 (verified via the
    // reconnaissance probe against RngStream::from_master(1, &[])), which
    // is > p_c=0.5, so realcross's whole-pair gate fails deterministically
    // and both children must equal the parents verbatim with zero further
    // draws consumed for the whole pair.

    #[test]
    fn sbx_pair_gate_fails_children_equal_parents_verbatim() {
        let p1 = vec![1.0, 3.0];
        let p2 = vec![4.0, 3.0];
        let lo = vec![-5.0, -5.0];
        let hi = vec![5.0, 5.0];
        let mut rng = RngStream::from_master(1, &[]);
        let (c1, c2) = sbx_pair(&p1, &p2, &lo, &hi, 2.0, 0.5, &mut rng);
        assert_eq!(c1, p1, "pair-gate failure must copy parent1 verbatim into child1");
        assert_eq!(c2, p2, "pair-gate failure must copy parent2 verbatim into child2");
    }

    // ---- EPS guard exercised: equal-in-one-variable parents -----------------
    //
    // p1[1] == p2[1] == 3.0 exactly (diff 0.0 <= EPS=1e-14): with the
    // per-variable exchange gate passing for variable 1 (seed 16, verified
    // via the reconnaissance probe below), the EPS guard must fire and
    // copy variable 1 verbatim into both children instead of running the
    // SBX math (which would otherwise divide by y2-y1 == 0).

    #[test]
    fn sbx_pair_eps_guard_copies_equal_variable() {
        let p1 = vec![1.0, 3.0];
        let p2 = vec![4.0, 3.0];
        let lo = vec![-5.0, -5.0];
        let hi = vec![5.0, 5.0];
        let mut rng = RngStream::from_master(16, &[]);
        let (c1, c2) = sbx_pair(&p1, &p2, &lo, &hi, 2.0, 1.0, &mut rng);
        assert_eq!(c1[1], 3.0, "EPS-guarded variable must be copied verbatim into child1");
        assert_eq!(c2[1], 3.0, "EPS-guarded variable must be copied verbatim into child2");
        // Variable 0 (distinct parents) DOES go through the SBX math and
        // differs from both parents.
        assert_ne!(c1[0], p1[0]);
        assert_ne!(c2[0], p2[0]);
    }

    // ---- draw-count / twin-stream raw-replay ---------------------------
    //
    // Two fixed cases replaying the raw uniform stream against a twin
    // RngStream clone of the pre-call state, asserting the NEXT draw after
    // the call matches the twin's next raw draw at the predicted index --
    // if sbx_pair/polynomial_mutation consumed a different number of draws
    // than pinned in the module doc, the two streams would (with
    // overwhelming probability) diverge from here on. Pattern follows the
    // existing twin-stream test in woa.rs.

    #[test]
    fn sbx_pair_draw_count_pass_branch() {
        // seed=16, p1=[1.0,3.0], p2=[4.0,3.0], p_c=1.0 (pair gate passes,
        // 1 draw): var0 -- exchange gate passes (1 draw), parents differ
        // (no EPS-guard draw), rand drawn once and reused for both
        // children (1 draw), final swap gate (1 draw) = 3 draws; var1 --
        // exchange gate passes (1 draw), EPS guard fires (parents equal,
        // 0 further draws) = 1 draw. Total = 1 (pair) + 3 (var0) + 1
        // (var1) = 5 draws.
        let p1 = vec![1.0, 3.0];
        let p2 = vec![4.0, 3.0];
        let lo = vec![-5.0, -5.0];
        let hi = vec![5.0, 5.0];

        let rng_before = RngStream::from_master(16, &[]);
        let mut rng = rng_before.clone();
        let (c1, c2) = sbx_pair(&p1, &p2, &lo, &hi, 2.0, 1.0, &mut rng);
        assert_eq!(c1.len(), 2);
        assert_eq!(c2.len(), 2);

        let mut twin = rng_before;
        for _ in 0..5 {
            let _ = twin.next_f64();
        }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "sbx_pair (pair gate pass) must consume exactly 5 draws for this fixed case");
    }

    #[test]
    fn sbx_pair_draw_count_pair_gate_fail_branch() {
        // seed=1: first draw 0.8116... > p_c=0.5, pair gate fails -> ONE
        // draw total for the whole pair, zero per-variable draws.
        let p1 = vec![1.0, 3.0];
        let p2 = vec![4.0, 3.0];
        let lo = vec![-5.0, -5.0];
        let hi = vec![5.0, 5.0];

        let rng_before = RngStream::from_master(1, &[]);
        let mut rng = rng_before.clone();
        let (c1, c2) = sbx_pair(&p1, &p2, &lo, &hi, 2.0, 0.5, &mut rng);
        assert_eq!(c1, p1);
        assert_eq!(c2, p2);

        let mut twin = rng_before;
        let _pair_gate = twin.next_f64();
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "sbx_pair (pair gate fail) must consume exactly 1 draw for the whole pair");
    }

    #[test]
    fn polynomial_mutation_draw_count_mixed_pass_fail() {
        // seed=9, p_m=0.5, x=[1.0,3.0]: var0 gate fails (draw 0.5990 >
        // 0.5, 1 draw, unchanged); var1 gate passes (draw 0.4297 <= 0.5,
        // 1 draw) then `rnd` is drawn once more (1 draw) = 2 draws.
        // Total = 1 + 2 = 3 draws.
        let lo = vec![-5.0, -5.0];
        let hi = vec![5.0, 5.0];

        let rng_before = RngStream::from_master(9, &[]);
        let mut rng = rng_before.clone();
        let mut x = vec![1.0, 3.0];
        polynomial_mutation(&mut x, &lo, &hi, 2.0, 0.5, &mut rng);
        assert_eq!(x[0], 1.0, "var0's gate must fail and leave it unchanged");
        assert_ne!(x[1], 3.0, "var1's gate must pass and mutate it");

        let mut twin = rng_before;
        for _ in 0..3 {
            let _ = twin.next_f64();
        }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "polynomial_mutation must consume exactly 3 draws for this mixed pass/fail case");
    }
}
