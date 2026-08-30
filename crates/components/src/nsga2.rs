//! Pareto-dominance sorting and crowding-distance estimation for NSGA-II
//! (Deb, Pratap, Agarwal & Meyarivan, "A Fast and Elitist Multiobjective
//! Genetic Algorithm: NSGA-II", *IEEE Transactions on Evolutionary
//! Computation* 6(2):182-197, April 2002; PDF verified via
//! <https://www.cse.unr.edu/~sushil/class/gas/papers/nsga2.pdf>, a mirror of
//! the IEEE original, pp. 184-185).
//!
//! ## Layout
//! This module currently holds ONLY the three pure, engine-free,
//! problem-free routines below (Task 4 of the M3-2 milestone): Pareto
//! dominance, fast non-dominated sorting, and crowding-distance assignment.
//! They operate on plain `&[Vec<f64>]` objective matrices (minimization) --
//! no `Ctx`, `Registry`, or `ComponentMeta` -- because NSGA-II is not a
//! scalar-graph component here; the recorded architecture plan runs
//! multi-objective search through a dedicated MO runner instead. Variation
//! operators (Task 5) and the runner itself -- including the tournament
//! selection that consumes the crowded-comparison operator pinned below --
//! arrive in later tasks of this milestone.
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
}
