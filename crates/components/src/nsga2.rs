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
//! (`nsga2_run`, Task 6) lives at the bottom of this module, below the
//! variation operators -- see its own "## NSGA-II main-loop runner" section
//! for the main-loop provenance (initialization, tournament pairing/
//! comparison, environmental selection, RNG derivation, budget-tail rule).
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
//! definition is pinned here for the record, as the paper's own STATED
//! operator -- **but Task 6's `tournament` does NOT use it as written**: the
//! verified KanGAL reference C (`tourselect.c`'s `tournament()`) recomputes
//! raw pairwise dominance directly instead of comparing the assigned `.rank`
//! field, only falling back to crowding on a dominance tie. See the
//! "Tournament comparison" part of the "## NSGA-II main-loop runner" section
//! below for the full verified quote and the discrepancy analysis.
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
//!
//! ## NSGA-II main-loop runner (Task 6)
//!
//! `nsga2_run` below wires the pieces above (`dominates`,
//! `fast_non_dominated_sort`, `crowding_distance`, `sbx_pair`,
//! `polynomial_mutation`) into the full generational NSGA-II algorithm,
//! against [`sezgi_core::mo::MoProblem`] / [`sezgi_core::mo::MoEvaluator`]
//! (T1's parallel MO surface to the scalar `Problem`/`Evaluator`/`Engine`).
//! Every structural piece below is fetched and verified from the SAME two
//! sources T4/T5 pinned: the paper itself (this module's header) and the
//! KanGAL reference C mirror, `darnir/nsga2` on GitHub
//! (`https://raw.githubusercontent.com/darnir/nsga2/master/*.c`) --
//! `initialize.c`, `tourselect.c`, `dominance.c`, `rand.c`, `rank.c`,
//! `fillnds.c`, `crossover.c`, `mutation.c`, `merge.c`, and `nsga2r.c` (the
//! `main()` driver, read specifically to establish CALL ORDER between
//! `selection`/`mutation_pop`/`fill_nondominated_sort` and to confirm the
//! `popsize % 4 == 0` validation).
//!
//! ### Initialization (`initialize.c`, `rand.c`)
//! Quoted verbatim (irrelevant binary-variable branch elided):
//! ```text
//! void initialize_pop (population *pop)
//! {
//!     for (i=0; i<popsize; i++)
//!         initialize_ind (&(pop->ind[i]));
//! }
//! void initialize_ind (individual *ind)
//! {
//!     if (nreal!=0)
//!         for (j=0; j<nreal; j++)
//!             ind->xreal[j] = rndreal (min_realvar[j], max_realvar[j]);
//! }
//! /* rand.c */
//! double rndreal (double low, double high)
//! {
//!     return (low + (high-low)*randomperc());
//! }
//! ```
//! i.e. one uniform draw per variable (`low + (high-low)*U(0,1)`),
//! individuals in order `0..popsize`, variables within an individual in
//! order `0..nreal` -- exactly the loop nesting `init_population` below
//! replicates (`lo[j] + (hi[j]-lo[j]) * rng.next_f64()`, individual-major,
//! variable-minor).
//!
//! ### Tournament pairing (`tourselect.c`, `selection`)
//! Quoted verbatim (the Fisher-Yates-style double-permutation shuffle and
//! the block-of-4 pairing loop):
//! ```text
//! void selection (population *old_pop, population *new_pop)
//! {
//!     for (i=0; i<popsize; i++) { a1[i] = a2[i] = i; }
//!     for (i=0; i<popsize; i++)
//!     {
//!         rand = rnd (i, popsize-1);
//!         temp = a1[rand]; a1[rand] = a1[i]; a1[i] = temp;
//!         rand = rnd (i, popsize-1);
//!         temp = a2[rand]; a2[rand] = a2[i]; a2[i] = temp;
//!     }
//!     for (i=0; i<popsize; i+=4)
//!     {
//!         parent1 = tournament (&old_pop->ind[a1[i]],   &old_pop->ind[a1[i+1]]);
//!         parent2 = tournament (&old_pop->ind[a1[i+2]], &old_pop->ind[a1[i+3]]);
//!         crossover (parent1, parent2, &new_pop->ind[i], &new_pop->ind[i+1]);
//!         parent1 = tournament (&old_pop->ind[a2[i]],   &old_pop->ind[a2[i+1]]);
//!         parent2 = tournament (&old_pop->ind[a2[i+2]], &old_pop->ind[a2[i+3]]);
//!         crossover (parent1, parent2, &new_pop->ind[i+2], &new_pop->ind[i+3]);
//!     }
//! }
//! ```
//! `rand.c`'s `rnd(low, high)`: `res = low + (int)(randomperc()*(high-low+1))`,
//! clamped down to `high`, but **with NO draw at all when `low >= high`**
//! (`if (low>=high) res=low;`, an early return before `randomperc()` is ever
//! called) -- this fires on exactly the LAST shuffle position (`i ==
//! popsize-1`, a self-swap). Verified structure: TWO independent
//! permutations `a1`, `a2` of `0..popsize` are built by walking `i =
//! 0..popsize` and, for EACH `i`, drawing the `a1` swap-partner THEN the
//! `a2` swap-partner (both draws share the SAME underlying stream,
//! interleaved per-`i` -- NOT `a1` shuffled fully, then `a2` shuffled
//! fully) -- `shuffle_two_interleaved` below replicates this draw
//! INTERLEAVING exactly, not just the resulting permutations' marginal
//! distribution. Each block of 4 consecutive `a1` (or `a2`) positions
//! produces ONE mating pair (2 tournament winners -> 1 `crossover` call ->
//! 2 children); `mutation_pop` (a SEPARATE full pass over the whole
//! `popsize`-sized child population, `mutation.c`, called from `nsga2r.c`'s
//! main loop AFTER `selection` returns, never interleaved into it) then
//! mutates every child in order `0..popsize` -- `generate_offspring` below
//! replicates this exact two-phase structure (all crossover first, then a
//! separate full mutation pass over the completed child population).
//!
//! **`popsize % 4 == 0` requirement.** The block-of-4 loop above requires
//! `popsize` to be an exact multiple of 4 for `a1`/`a2` to each divide into
//! whole blocks; `nsga2r.c`'s own `main()` enforces this directly (`if
//! (popsize<4 || (popsize%4)!=0) { ...; exit(1); }`). Per the task brief's
//! own stated preference ("if you keep KanGAL's double-permutation scheme,
//! implement its Fisher-Yates faithfully ... prefer faithfulness"), this
//! module keeps the scheme UNCHANGED rather than generalizing it to
//! arbitrary even sizes. **sezgi decision:** [`Nsga2Config::pop_size`] is
//! validated as `pop_size >= 4 && pop_size % 4 == 0` -- a strictly
//! NARROWER, backward-compatible subset of "even, >= 4" -- mirroring
//! `nsga2r.c`'s own hard validation rather than deviating the pairing
//! scheme itself.
//!
//! ### Tournament comparison (`tourselect.c`'s `tournament`, `dominance.c`)
//! Quoted verbatim:
//! ```text
//! individual* tournament (individual *ind1, individual *ind2)
//! {
//!     flag = check_dominance (ind1, ind2);
//!     if (flag==1)  return (ind1);
//!     if (flag==-1) return (ind2);
//!     if (ind1->crowd_dist > ind2->crowd_dist) return(ind1);
//!     if (ind2->crowd_dist > ind1->crowd_dist) return(ind2);
//!     if ((randomperc()) <= 0.5) return(ind1); else return(ind2);
//! }
//! ```
//! `check_dominance` (unconstrained branch, `dominance.c`): returns `1` if
//! `ind1` Pareto-dominates `ind2`, `-1` if `ind2` dominates `ind1`, `0`
//! otherwise -- EXACTLY this module's own [`dominates`] (minimization, same
//! truth table), recomputed directly on the two candidates' RAW objective
//! vectors.
//!
//! **Discovered discrepancy vs. the paper's stated `<_n` operator.** This
//! module's own "Crowded-comparison operator" section above (paper, p. 185)
//! pins `i <_n j` as `i_rank < j_rank OR (i_rank = j_rank AND i_distance >
//! j_distance)` -- comparing the ASSIGNED front-rank field first. The
//! reference C's `tournament()`, quoted above, does NOT read `.rank` at
//! all: it recomputes raw pairwise dominance between the two candidates
//! (`check_dominance`) and only falls back to `crowd_dist`, then a
//! coin-flip, on a dominance TIE (`flag==0`, neither directly dominates the
//! other). The two agree whenever one candidate directly dominates the
//! other (a direct dominator always has strictly lower rank than the point
//! it dominates, by construction of `fast-non-dominated-sort` -- so
//! `flag==1` implies `i_rank < j_rank` too), but they can DIVERGE when two
//! candidates from DIFFERENT ranks are pairwise mutually non-dominated
//! (possible: rank-`(k+1)` membership only requires SOME rank-`<=k` point
//! to dominate a candidate, not that this SPECIFIC opponent does) -- the
//! paper's operator would decide by rank alone there; the C code falls
//! through to `crowd_dist`/coin-flip instead. **Decision (faithful
//! preferred, per the task brief):** `tournament` below implements the
//! VERIFIED C reference exactly -- `dominates(a,b)` (reusing this module's
//! own function directly, since it already matches `check_dominance`'s
//! unconstrained truth table bit for bit) then `crowd_dist` then a single
//! `<= 0.5` coin-flip draw on a full tie -- NOT the paper's rank-comparison
//! paraphrase. This corrects (does not merely extend) the pre-T6 module
//! doc's speculative note that the rank-based operator "is used by the T6
//! tournament selector" -- that note predated fetching `tourselect.c` and
//! is superseded by this verified finding.
//!
//! ### Environmental selection (paper pseudocode, p. 186; `fillnds.c`)
//! Quoted verbatim from the paper's own boxed main-loop pseudocode (p. 186,
//! immediately below Fig. 2 "NSGA-II procedure"; right-column comments are
//! the paper's own):
//! ```text
//! R_t = P_t union Q_t                    combine parent and offspring population
//! F = fast-non-dominated-sort(R_t)       F = (F_1, F_2, ...), all nondominated fronts of R_t
//! P_{t+1} = empty and i = 1
//! until |P_{t+1}| + |F_i| <= N           until the parent population is filled
//!     crowding-distance-assignment(F_i)  calculate crowding-distance in F_i
//!     P_{t+1} = P_{t+1} union F_i        include ith nondominated front in the parent pop
//!     i = i + 1                          check the next front for inclusion
//! Sort(F_i, <_n)                          sort in descending order using <_n
//! P_{t+1} = P_{t+1} union F_i[1:(N - |P_{t+1}|)]   choose the first (N - |P_{t+1}|) elements of F_i
//! Q_{t+1} = make-new-pop(P_{t+1})        use selection, crossover and mutation to create Q_{t+1}
//! ```
//! `environmental_selection` below implements this exactly: combine the
//! parent and offspring populations (`2 * pop_size` total),
//! `fast_non_dominated_sort`, accumulate whole fronts while they still fit,
//! then for the first front that does NOT fit whole, compute
//! `crowding_distance` for JUST that front and take its highest-crowding
//! members until the population reaches exactly `pop_size`. `fillnds.c`'s
//! `fill_nondominated_sort` implements the identical
//! whole-fronts-then-split-front structure (confirmed by reading it) via a
//! linked-list dominance scan (`check_dominance` again) rather than this
//! crate's array-based `fast_non_dominated_sort` -- an implementation-detail
//! difference with no observable difference in the SET of fronts produced
//! (both are the standard fast-non-dominated-sort front partition).
//!
//! **Split-front truncation order (determinism deviation, pinned).** The
//! C's `crowding_fill` sorts the split front's members by `crowd_dist` via
//! `quicksort_dist` (a plain unstable quicksort -- its tie behavior is
//! whatever the platform's qsort-style implementation happens to do,
//! unspecified and not reproducible across builds). Per the task brief's
//! own instruction, this module instead uses Rust's stable `sort_by` with
//! an EXPLICIT documented tie-break: descending `crowd_dist`, ties broken
//! by ascending original front-position index. **sezgi decision:**
//! deterministic by construction, at the cost of not literally reproducing
//! the C's platform-dependent tie order (itself not a well-defined target
//! to reproduce).
//!
//! ### RNG stream derivation
//! [`NSGA2_SEED_BASE`] (`0x9531`, a distinct "NSGA"-flavored constant) is
//! mixed into the run's master exactly like `crates/bias/src/f0.rs`'s
//! `BIAS_SEED_BASE` (`0xB1A5`) and `sezgi_problems`' `BBOB_SEED_BASE`:
//! `master = NSGA2_SEED_BASE.wrapping_add(cfg.seed)`, never `cfg.seed`
//! passed to `RngStream::from_master` directly. Two documented sub-streams
//! are derived under that master via `RngStream::from_master(master,
//! path)`: `path = &[1]` for INITIALIZATION (`init_population`'s uniform
//! draws) and `path = &[2]` for all per-generation VARIATION (the
//! tournament shuffle, tournament coin-flip ties, `sbx_pair`,
//! `polynomial_mutation` -- one single shared stream across all of these
//! each generation, consumed in the fixed order documented above and in
//! each function's own doc). Non-collision reasoning mirrors `f0`'s own
//! (`crates/bias/src/f0.rs`'s doc): structurally, `master` here is never
//! `cfg.master_seed`/`cfg.seed` verbatim (always offset by
//! `NSGA2_SEED_BASE`, itself distinct from `BIAS_SEED_BASE`), so an
//! accidental collision with the engine's `[run_id, tag]` family or with
//! `f0`'s own stream would require an adversarially chosen `cfg.seed`;
//! path SHAPE (`&[1]`/`&[2]`, single-element, small integers) also differs
//! from both the engine's two-element `[run_id, tag]` paths and `f0`'s
//! `[3_000_000]` single-element path, a second independent
//! fold-count/value difference even in a hypothetical master collision.
//!
//! ### Budget-tail rule
//! Mirrors the scalar `Engine::run` exactly (`crates/core/src/engine.rs`):
//! offspring are generated SPECULATIVELY every loop attempt (consuming the
//! variation RNG stream even for a doomed final attempt, exactly as the
//! scalar engine's own generator stage runs before its budget check), then
//! `MoEvaluator::evaluate` is called; on `Err` (the batch would exceed the
//! remaining budget -- `MoEvaluator`'s all-or-nothing rule, T1), the loop
//! breaks immediately WITHOUT touching the population, and the run returns
//! the last successfully completed generation's population.
//! [`MoRunResult::evals_used`] is read directly from `MoEvaluator::used()`.
//! Exact accounting: the initial batch consumes `pop_size`; each completed
//! generation consumes exactly `pop_size` more (offspring count always
//! equals `pop_size`); a budget of the exact form `pop_size + k *
//! pop_size` is used up completely (verified by a dedicated test below); a
//! non-exact-multiple budget leaves the unspent tail unused (also
//! verified).
//!
//! ### Defaults (paper, Section IV.A "Test Problems", p. 187)
//! Quoted verbatim: "The crossover probability of p_c = 0.9 and a mutation
//! probability of p_m = 1/n or 1/l (where n is the number of decision
//! variables for real-coded GAs and l is the string length for
//! binary-coded GAs) are used. For real-coded NSGA-II, we use distribution
//! indices for crossover and mutation operators as eta_c = 20 and eta_m =
//! 20, respectively." (Converting the paper's typeset subscripts/Greek
//! letters to prose, per this file's existing convention.) `Nsga2Config`'s
//! `p_m: Option<f64>`, when `None`, resolves to `1/n` with `n` = the
//! problem's total flattened Float-block dimension (`1/l`, the
//! binary-string case, is out of scope -- this crate's NSGA-II is
//! real-coded only). **Correction to the task brief:** this defaults
//! paragraph is in Section **IV** ("Simulation Results"), subsection A
//! ("Test Problems") -- not "Section V" as the brief's own text guessed;
//! verified directly against the fetched PDF (the page carrying the boxed
//! main-loop pseudocode, p. 186, ends with the "IV. SIMULATION RESULTS"
//! heading, immediately followed on p. 187 by subsection "A. Test
//! Problems" and this defaults paragraph).
//!
//! ### Validation
//! [`Nsga2Error`]'s variants mirror `nsga2r.c`'s own input-time checks
//! (`main()`, quoted structurally above under "`popsize % 4 == 0`
//! requirement"): `popsize<4 || popsize%4!=0` -> exit; `pcross_real`/
//! `pmut_real` outside `[0,1]` -> exit; `eta_c<=0` / `eta_m<=0` -> exit.
//! `nsga2_run` additionally requires the problem's [`SearchSpace`] to be
//! ALL [`Block::Float`] (`Nsga2Error::NonFloatSpace`, per the task brief --
//! this crate's SBX/polynomial-mutation operators are real-coded only, T5).

use sezgi_core::mo::{MoEvaluator, MoProblem};
use sezgi_core::rng::RngStream;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

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

/// Crate-owned base constant mixed into every NSGA-II run's RNG master. See
/// the module doc's "RNG stream derivation" section for the full
/// non-collision reasoning against the scalar engine's `[run_id, tag]`
/// family and the bias crate's own `BIAS_SEED_BASE`.
pub const NSGA2_SEED_BASE: u64 = 0x9531;

/// NSGA-II run configuration. **PINNED once merged** (see the task-6
/// brief): field types/names are a contract other tasks (T8/T9) build on.
/// Defaults for `eta_c`/`eta_m`/`p_c` and `p_m`'s `None` resolution are
/// pinned to the paper's own experimental settings -- see the module doc's
/// "Defaults" section for the verified quote.
#[derive(Debug, Clone)]
pub struct Nsga2Config {
    /// Population size: validated `>= 4` and a multiple of 4 (see the
    /// module doc's "`popsize % 4 == 0` requirement").
    pub pop_size: usize,
    /// Total function-evaluation budget (init + every generation), enforced
    /// all-or-nothing via [`MoEvaluator`] -- see the module doc's
    /// "Budget-tail rule".
    pub budget: u64,
    pub seed: u64,
    /// SBX distribution index. Paper default: `20.0`. Validated `> 0`.
    pub eta_c: f64,
    /// Polynomial-mutation distribution index. Paper default: `20.0`.
    /// Validated `> 0`.
    pub eta_m: f64,
    /// SBX crossover probability. Paper default: `0.9`. Validated in
    /// `[0,1]`.
    pub p_c: f64,
    /// Per-variable mutation probability. `None` resolves to `1 /
    /// n_variables` (the paper's own default). `Some(p)` is validated in
    /// `[0,1]`.
    pub p_m: Option<f64>,
}

/// One completed NSGA-II run's outcome. **PINNED once merged.**
#[derive(Debug, Clone, PartialEq)]
pub struct MoRunResult {
    /// The final population's decision vectors.
    pub individuals: Vec<Genotype>,
    /// Parallel to `individuals`: each row is that individual's objective
    /// vector.
    pub objectives: Vec<Vec<f64>>,
    /// Indices (into `individuals`/`objectives`) of the final population's
    /// non-dominated set (front 0 of `fast_non_dominated_sort` on the final
    /// `objectives`).
    pub front0: Vec<usize>,
    /// Total evaluations charged, read directly from
    /// [`MoEvaluator::used`].
    pub evals_used: u64,
}

/// Errors from [`nsga2_run`]. See the module doc's "Validation" section for
/// the verified source of each check.
#[derive(Debug, thiserror::Error)]
pub enum Nsga2Error {
    #[error(
        "pop_size must be >= 4 and a multiple of 4 (KanGAL's double-permutation \
         tournament pairing requires it; see nsga2r.c's own `popsize % 4 == 0` check), got {pop_size}"
    )]
    InvalidPopSize { pop_size: usize },
    #[error("{name} must lie in [0, 1], got {value}")]
    InvalidProbability { name: &'static str, value: f64 },
    #[error("{name} must be > 0, got {value}")]
    InvalidEta { name: &'static str, value: f64 },
    #[error("nsga2_run requires an all-Float search space; block {index} is not Block::Float")]
    NonFloatSpace { index: usize },
    #[error("initial population (pop_size={pop_size}) exceeds the evaluation budget ({budget})")]
    BudgetTooSmallForInit { pop_size: usize, budget: u64 },
}

fn validate_config(cfg: &Nsga2Config) -> Result<(), Nsga2Error> {
    if cfg.pop_size < 4 || !cfg.pop_size.is_multiple_of(4) {
        return Err(Nsga2Error::InvalidPopSize { pop_size: cfg.pop_size });
    }
    if !(0.0..=1.0).contains(&cfg.p_c) {
        return Err(Nsga2Error::InvalidProbability { name: "p_c", value: cfg.p_c });
    }
    if let Some(p_m) = cfg.p_m {
        if !(0.0..=1.0).contains(&p_m) {
            return Err(Nsga2Error::InvalidProbability { name: "p_m", value: p_m });
        }
    }
    // `is_nan() ||` first, rather than the equivalent `!(x > 0.0)`, so
    // clippy's `neg_cmp_op_on_partial_ord` doesn't flag a negated
    // partial-order comparison -- both reject NaN and every `<= 0.0` value.
    if cfg.eta_c.is_nan() || cfg.eta_c <= 0.0 {
        return Err(Nsga2Error::InvalidEta { name: "eta_c", value: cfg.eta_c });
    }
    if cfg.eta_m.is_nan() || cfg.eta_m <= 0.0 {
        return Err(Nsga2Error::InvalidEta { name: "eta_m", value: cfg.eta_m });
    }
    Ok(())
}

/// Flattened per-variable lower/upper bounds, plus each block's length (for
/// reassembling a flat decision vector back into a [`Genotype`]).
type FloatBounds = (Vec<f64>, Vec<f64>, Vec<usize>);

/// Validates an all-`Block::Float` search space (module doc, "Validation")
/// and extracts flattened per-variable bounds plus each block's length (for
/// reassembling flat decision vectors back into a [`Genotype`]).
fn float_bounds(space: &SearchSpace) -> Result<FloatBounds, Nsga2Error> {
    let mut lo = Vec::new();
    let mut hi = Vec::new();
    let mut block_lens = Vec::new();
    for (index, b) in space.blocks().iter().enumerate() {
        match *b {
            Block::Float { lo: l, hi: h, n } => {
                for _ in 0..n {
                    lo.push(l);
                    hi.push(h);
                }
                block_lens.push(n);
            }
            _ => return Err(Nsga2Error::NonFloatSpace { index }),
        }
    }
    Ok((lo, hi, block_lens))
}

/// Reassembles a flat decision vector into a [`Genotype`] matching the
/// original space's block lengths (each block becomes `BlockValues::Float`).
fn to_genotype(flat: &[f64], block_lens: &[usize]) -> Genotype {
    let mut blocks = Vec::with_capacity(block_lens.len());
    let mut offset = 0;
    for &len in block_lens {
        blocks.push(BlockValues::Float(flat[offset..offset + len].to_vec()));
        offset += len;
    }
    Genotype { blocks }
}

/// Flattens a [`Genotype`] (already validated all-Float by
/// [`float_bounds`]/[`nsga2_run`]) into a single `Vec<f64>` for the
/// variation operators.
fn flatten_genotype(g: &Genotype) -> Vec<f64> {
    let mut out = Vec::with_capacity(g.blocks.iter().map(|b| match b {
        BlockValues::Float(xs) => xs.len(),
        _ => 0,
    }).sum());
    for b in &g.blocks {
        match b {
            BlockValues::Float(xs) => out.extend_from_slice(xs),
            _ => unreachable!("nsga2_run validated an all-Float space before ever constructing a Genotype"),
        }
    }
    out
}

/// Uniform-in-bounds initial population. Draw order pinned to
/// `initialize.c`/`rand.c` (module doc): individual-major, variable-minor,
/// one `next_f64()` draw per variable.
fn init_population(n: usize, lo: &[f64], hi: &[f64], rng: &mut RngStream) -> Vec<Vec<f64>> {
    (0..n)
        .map(|_| (0..lo.len()).map(|j| lo[j] + (hi[j] - lo[j]) * rng.next_f64()).collect())
        .collect()
}

/// `rand.c`'s `rnd(low, high)`: a uniform integer in `[low, high]`
/// inclusive, via `low + floor(U(0,1) * (high-low+1))` clamped down to
/// `high`. Pinned quirk (module doc, "Tournament pairing"): **no draw at
/// all** when `low >= high`.
fn rnd(low: usize, high: usize, rng: &mut RngStream) -> usize {
    if low >= high {
        return low;
    }
    let width = (high - low + 1) as f64;
    let mut res = low + (rng.next_f64() * width).floor() as usize;
    if res > high {
        res = high;
    }
    res
}

/// The `tourselect.c` `selection()` double-permutation shuffle, draws
/// INTERLEAVED per-index between `a1` and `a2` (module doc, "Tournament
/// pairing"): for each `i`, one `a1` swap-partner draw, then one `a2`
/// swap-partner draw, before moving to `i+1`.
fn shuffle_two_interleaved(n: usize, rng: &mut RngStream) -> (Vec<usize>, Vec<usize>) {
    let mut a1: Vec<usize> = (0..n).collect();
    let mut a2: Vec<usize> = (0..n).collect();
    for i in 0..n {
        let j1 = rnd(i, n - 1, rng);
        a1.swap(i, j1);
        let j2 = rnd(i, n - 1, rng);
        a2.swap(i, j2);
    }
    (a1, a2)
}

/// `tourselect.c`'s `tournament()`, verified faithful (module doc,
/// "Tournament comparison"): raw pairwise dominance first (reusing
/// [`dominates`] directly), then crowding distance, then a single coin-flip
/// draw on a full tie. Returns the winner's index (`i` or `j`).
fn tournament(objectives: &[Vec<f64>], crowd: &[f64], i: usize, j: usize, rng: &mut RngStream) -> usize {
    if dominates(&objectives[i], &objectives[j]) {
        return i;
    }
    if dominates(&objectives[j], &objectives[i]) {
        return j;
    }
    if crowd[i] > crowd[j] {
        return i;
    }
    if crowd[j] > crowd[i] {
        return j;
    }
    if rng.next_f64() <= 0.5 { i } else { j }
}

/// Crowding distance for an ALREADY-fully-included population (no
/// truncation): every front is complete, so every individual gets a
/// `crowd_dist` from [`crowding_distance`] directly. Mirrors `rank.c`'s
/// `assign_rank_and_crowding_distance`, called once on the initial
/// population before the generation loop (module doc).
fn crowd_dist_full(objectives: &[Vec<f64>]) -> Vec<f64> {
    let fronts = fast_non_dominated_sort(objectives);
    let mut out = vec![0.0f64; objectives.len()];
    for front in &fronts {
        let d = crowding_distance(front, objectives);
        for (k, &idx) in front.iter().enumerate() {
            out[idx] = d[k];
        }
    }
    out
}

/// One generation's offspring: `tourselect.c`'s double-permutation
/// tournament pairing + `sbx_pair` crossover (block-of-4 loop), followed by
/// a SEPARATE full `polynomial_mutation` pass over every child in order
/// (module doc, "Tournament pairing" -- the two-phase structure is pinned,
/// not interleaved).
#[allow(clippy::too_many_arguments)]
fn generate_offspring(
    genos: &[Genotype],
    objectives: &[Vec<f64>],
    crowd: &[f64],
    lo: &[f64],
    hi: &[f64],
    block_lens: &[usize],
    cfg: &Nsga2Config,
    p_m: f64,
    rng: &mut RngStream,
) -> Vec<Genotype> {
    let n = genos.len();
    let flat: Vec<Vec<f64>> = genos.iter().map(flatten_genotype).collect();
    let (a1, a2) = shuffle_two_interleaved(n, rng);
    let mut children: Vec<Vec<f64>> = vec![Vec::new(); n];

    let mut i = 0;
    while i < n {
        let p1 = tournament(objectives, crowd, a1[i], a1[i + 1], rng);
        let p2 = tournament(objectives, crowd, a1[i + 2], a1[i + 3], rng);
        let (c1, c2) = sbx_pair(&flat[p1], &flat[p2], lo, hi, cfg.eta_c, cfg.p_c, rng);
        children[i] = c1;
        children[i + 1] = c2;

        let p3 = tournament(objectives, crowd, a2[i], a2[i + 1], rng);
        let p4 = tournament(objectives, crowd, a2[i + 2], a2[i + 3], rng);
        let (c3, c4) = sbx_pair(&flat[p3], &flat[p4], lo, hi, cfg.eta_c, cfg.p_c, rng);
        children[i + 2] = c3;
        children[i + 3] = c4;

        i += 4;
    }

    for child in &mut children {
        polynomial_mutation(child, lo, hi, cfg.eta_m, p_m, rng);
    }

    children.iter().map(|f| to_genotype(f, block_lens)).collect()
}

/// `(μ+λ)` environmental selection: combine parent + offspring, sort into
/// fronts, fill whole fronts, and truncate the first non-fitting front by
/// descending crowding distance (ascending-index tie-break, deterministic
/// -- module doc, "Environmental selection"). Returns the new population's
/// `(individuals, objectives, crowd_dist)`, all parallel and of length `n`.
fn environmental_selection(
    mut genos: Vec<Genotype>,
    mut objectives: Vec<Vec<f64>>,
    off_genos: Vec<Genotype>,
    off_objectives: Vec<Vec<f64>>,
    n: usize,
) -> (Vec<Genotype>, Vec<Vec<f64>>, Vec<f64>) {
    genos.extend(off_genos);
    objectives.extend(off_objectives);
    let fronts = fast_non_dominated_sort(&objectives);

    let mut new_genos = Vec::with_capacity(n);
    let mut new_objectives = Vec::with_capacity(n);
    let mut new_crowd = Vec::with_capacity(n);

    for front in &fronts {
        if new_genos.len() >= n {
            break;
        }
        let d = crowding_distance(front, &objectives);
        if new_genos.len() + front.len() <= n {
            for (k, &idx) in front.iter().enumerate() {
                new_genos.push(genos[idx].clone());
                new_objectives.push(objectives[idx].clone());
                new_crowd.push(d[k]);
            }
        } else {
            let need = n - new_genos.len();
            let mut order: Vec<usize> = (0..front.len()).collect();
            // sezgi decision: stable, explicit tie-break (descending
            // crowd_dist, then ascending front-position index) -- see the
            // module doc's "Split-front truncation order" paragraph.
            order.sort_by(|&a, &b| d[b].total_cmp(&d[a]).then(a.cmp(&b)));
            for &k in order.iter().take(need) {
                let idx = front[k];
                new_genos.push(genos[idx].clone());
                new_objectives.push(objectives[idx].clone());
                new_crowd.push(d[k]);
            }
            break;
        }
    }
    (new_genos, new_objectives, new_crowd)
}

/// NSGA-II reference runner. See the module doc's "## NSGA-II main-loop
/// runner" section for the full provenance (initialization, tournament
/// pairing/comparison, environmental selection, RNG derivation, budget-tail
/// rule, defaults).
pub fn nsga2_run(problem: &dyn MoProblem, cfg: &Nsga2Config) -> Result<MoRunResult, Nsga2Error> {
    validate_config(cfg)?;
    let (lo, hi, block_lens) = float_bounds(problem.space())?;
    let dim = lo.len();
    let p_m = cfg.p_m.unwrap_or(1.0 / dim as f64);

    let master = NSGA2_SEED_BASE.wrapping_add(cfg.seed);
    let mut init_rng = RngStream::from_master(master, &[1]);
    let mut var_rng = RngStream::from_master(master, &[2]);

    let mut eval = MoEvaluator::new(problem, cfg.budget);

    let init_flat = init_population(cfg.pop_size, &lo, &hi, &mut init_rng);
    let mut genos: Vec<Genotype> = init_flat.iter().map(|f| to_genotype(f, &block_lens)).collect();
    let mut objectives = eval.evaluate(&genos).map_err(|_| Nsga2Error::BudgetTooSmallForInit {
        pop_size: cfg.pop_size,
        budget: cfg.budget,
    })?;
    let mut crowd = crowd_dist_full(&objectives);

    loop {
        // Budget-tail rule (module doc): generate speculatively, evaluate,
        // and break WITHOUT mutating the population on Err -- mirrors the
        // scalar engine's `Err(_) => break 'outer` exactly.
        let offspring =
            generate_offspring(&genos, &objectives, &crowd, &lo, &hi, &block_lens, cfg, p_m, &mut var_rng);
        let off_objectives = match eval.evaluate(&offspring) {
            Ok(o) => o,
            Err(_) => break,
        };
        let (new_genos, new_objectives, new_crowd) =
            environmental_selection(genos, objectives, offspring, off_objectives, cfg.pop_size);
        genos = new_genos;
        objectives = new_objectives;
        crowd = new_crowd;
    }

    let front0 = fast_non_dominated_sort(&objectives).into_iter().next().unwrap_or_default();

    Ok(MoRunResult { individuals: genos, objectives, front0, evals_used: eval.used() })
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

    // ==================================================================
    // nsga2_run (Task 6)
    // ==================================================================

    use sezgi_problems::Zdt;
    use sezgi_stats::igd;

    fn base_cfg() -> Nsga2Config {
        Nsga2Config { pop_size: 8, budget: 200, seed: 7, eta_c: 20.0, eta_m: 20.0, p_c: 0.9, p_m: None }
    }

    // ---- determinism golden ---------------------------------------------
    //
    // ZDT1, dim=6, pop=8, budget=200, seed=7. The exact objective values
    // below were measured ONCE from this implementation (see the task-6
    // report for the probe) and hardcoded as a bit-exact golden -- this is
    // a NEW golden (this task's own runner), not a reproduction of any
    // prior pin.
    #[test]
    fn nsga2_run_determinism_golden_zdt1() {
        let problem = Zdt::new(1, 6).unwrap();
        let cfg = base_cfg();
        let result = nsga2_run(&problem, &cfg).unwrap();
        assert_eq!(result.objectives.len(), 8);
        let got: Vec<[f64; 2]> =
            result.objectives[..3].iter().map(|row| [row[0], row[1]]).collect();
        let expect = [
            [0.943_376_728_949_411, 0.176_336_549_868_822_02],
            [9.526_467_756_149_07e-6, 1.812_548_914_192_057_7],
            [0.096_195_882_459_852_06, 1.342_229_206_945_092_7],
        ];
        for (i, (g, e)) in got.iter().zip(expect.iter()).enumerate() {
            assert!((g[0] - e[0]).abs() < 1e-12 && (g[1] - e[1]).abs() < 1e-12,
                "row {i}: got {g:?}, expected {e:?}");
        }
    }

    // ---- same-seed run-twice: bit-identical full result ------------------

    #[test]
    fn nsga2_run_same_seed_twice_bit_identical() {
        let problem = Zdt::new(1, 6).unwrap();
        let cfg = base_cfg();
        let r1 = nsga2_run(&problem, &cfg).unwrap();
        let r2 = nsga2_run(&problem, &cfg).unwrap();
        assert_eq!(r1, r2, "same seed must reproduce a bit-identical MoRunResult");
    }

    // ---- budget accounting: exact multiple ---------------------------
    //
    // pop=8, budget=200: init consumes 8; each generation consumes exactly
    // 8 more (offspring count == pop_size); 200 = 8 + 24*8 exactly, so 24
    // generations complete and the 25th attempt's evaluate fails cleanly
    // (200 + 8 > 200) -- evals_used must land EXACTLY on 200, not merely
    // <= 200.

    #[test]
    fn nsga2_run_budget_accounting_exact_multiple() {
        let problem = Zdt::new(1, 6).unwrap();
        let cfg = base_cfg(); // budget=200, pop_size=8: 8 + 24*8 == 200 exactly
        let result = nsga2_run(&problem, &cfg).unwrap();
        assert_eq!(result.evals_used, 200,
            "an exactly-divisible budget must be used up exactly: init + generations*pop_size");
    }

    // ---- budget accounting: non-multiple leaves the tail unspent ---------

    #[test]
    fn nsga2_run_budget_accounting_non_multiple_leaves_tail() {
        let problem = Zdt::new(1, 6).unwrap();
        let mut cfg = base_cfg();
        cfg.budget = 205; // 8 + 24*8 = 200, remaining 5 < pop_size=8: unspendable tail
        let result = nsga2_run(&problem, &cfg).unwrap();
        assert_eq!(result.evals_used, 200,
            "a non-exact-multiple budget must leave the unspendable tail (5 evals) unused");
    }

    // ---- budget smaller than the initial population -----------------

    #[test]
    fn nsga2_run_budget_smaller_than_init_is_clear_error() {
        let problem = Zdt::new(1, 6).unwrap();
        let mut cfg = base_cfg();
        cfg.budget = 4; // < pop_size=8
        let err = nsga2_run(&problem, &cfg).unwrap_err();
        assert!(matches!(err, Nsga2Error::BudgetTooSmallForInit { pop_size: 8, budget: 4 }),
            "got {err:?}");
    }

    // ---- pop_size validation ------------------------------------------

    #[test]
    fn nsga2_run_odd_pop_size_is_error() {
        let problem = Zdt::new(1, 6).unwrap();
        let mut cfg = base_cfg();
        cfg.pop_size = 7;
        let err = nsga2_run(&problem, &cfg).unwrap_err();
        assert!(matches!(err, Nsga2Error::InvalidPopSize { pop_size: 7 }), "got {err:?}");
    }

    #[test]
    fn nsga2_run_pop_size_below_4_is_error() {
        let problem = Zdt::new(1, 6).unwrap();
        let mut cfg = base_cfg();
        cfg.pop_size = 2;
        let err = nsga2_run(&problem, &cfg).unwrap_err();
        assert!(matches!(err, Nsga2Error::InvalidPopSize { pop_size: 2 }), "got {err:?}");
    }

    #[test]
    fn nsga2_run_even_but_not_multiple_of_4_pop_size_is_error() {
        // sezgi decision (module doc): pop_size must be a multiple of 4,
        // not merely even -- KanGAL's double-permutation pairing scheme
        // requires it. 6 is even but not a multiple of 4.
        let problem = Zdt::new(1, 6).unwrap();
        let mut cfg = base_cfg();
        cfg.pop_size = 6;
        let err = nsga2_run(&problem, &cfg).unwrap_err();
        assert!(matches!(err, Nsga2Error::InvalidPopSize { pop_size: 6 }), "got {err:?}");
    }

    // ---- non-Float space is rejected -----------------------------------

    struct TinyIntProblem { space: SearchSpace }
    impl TinyIntProblem {
        fn new() -> Self {
            let space = SearchSpace::new(vec![Block::Int { lo: 0, hi: 10, n: 2 }]).unwrap();
            Self { space }
        }
    }
    impl MoProblem for TinyIntProblem {
        fn space(&self) -> &SearchSpace { &self.space }
        fn n_objectives(&self) -> usize { 2 }
        fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<Vec<f64>> {
            pop.iter().map(|_| vec![0.0, 0.0]).collect()
        }
    }

    #[test]
    fn nsga2_run_non_float_space_is_error() {
        let problem = TinyIntProblem::new();
        let cfg = base_cfg();
        let err = nsga2_run(&problem, &cfg).unwrap_err();
        assert!(matches!(err, Nsga2Error::NonFloatSpace { index: 0 }), "got {err:?}");
    }

    // ---- p_c / p_m / eta validation ------------------------------------

    #[test]
    fn nsga2_run_p_c_out_of_range_is_error() {
        let problem = Zdt::new(1, 6).unwrap();
        for bad in [-0.1, 1.1] {
            let mut cfg = base_cfg();
            cfg.p_c = bad;
            let err = nsga2_run(&problem, &cfg).unwrap_err();
            assert!(matches!(err, Nsga2Error::InvalidProbability { name: "p_c", .. }), "bad={bad}, got {err:?}");
        }
    }

    #[test]
    fn nsga2_run_p_m_out_of_range_is_error() {
        let problem = Zdt::new(1, 6).unwrap();
        for bad in [-0.1, 1.1] {
            let mut cfg = base_cfg();
            cfg.p_m = Some(bad);
            let err = nsga2_run(&problem, &cfg).unwrap_err();
            assert!(matches!(err, Nsga2Error::InvalidProbability { name: "p_m", .. }), "bad={bad}, got {err:?}");
        }
    }

    #[test]
    fn nsga2_run_eta_c_non_positive_is_error() {
        let problem = Zdt::new(1, 6).unwrap();
        for bad in [0.0, -1.0] {
            let mut cfg = base_cfg();
            cfg.eta_c = bad;
            let err = nsga2_run(&problem, &cfg).unwrap_err();
            assert!(matches!(err, Nsga2Error::InvalidEta { name: "eta_c", .. }), "bad={bad}, got {err:?}");
        }
    }

    #[test]
    fn nsga2_run_eta_m_non_positive_is_error() {
        let problem = Zdt::new(1, 6).unwrap();
        for bad in [0.0, -1.0] {
            let mut cfg = base_cfg();
            cfg.eta_m = bad;
            let err = nsga2_run(&problem, &cfg).unwrap_err();
            assert!(matches!(err, Nsga2Error::InvalidEta { name: "eta_m", .. }), "bad={bad}, got {err:?}");
        }
    }

    // ---- front0 sanity: every front0 member is mutually non-dominated -----

    #[test]
    fn nsga2_run_front0_is_mutually_non_dominated() {
        let problem = Zdt::new(1, 6).unwrap();
        let cfg = base_cfg();
        let result = nsga2_run(&problem, &cfg).unwrap();
        assert!(!result.front0.is_empty());
        for &i in &result.front0 {
            for &j in &result.front0 {
                if i != j {
                    assert!(!dominates(&result.objectives[i], &result.objectives[j]),
                        "front0 member {i} dominates front0 member {j}: not mutually non-dominated");
                }
            }
        }
    }

    // ---- convergence smoke -----------------------------------------------
    //
    // ZDT1, dim=10, pop=40, budget=8000, seed=1. IGD (T7, `sezgi_stats::igd`)
    // of the final front0's objectives against `zdt1.pareto_front(200)`.
    // Measured value at this seed: 0.012945771007555612. Anchored threshold
    // below (0.05) is rounded up with ~4x real headroom over the measured
    // value (single-seed smoke test, not a comparative claim -- multi-seed
    // statistics are T8's job).
    #[test]
    fn nsga2_run_zdt1_convergence_smoke() {
        let problem = Zdt::new(1, 10).unwrap();
        let cfg = Nsga2Config {
            pop_size: 40, budget: 8000, seed: 1,
            eta_c: 20.0, eta_m: 20.0, p_c: 0.9, p_m: None,
        };
        let result = nsga2_run(&problem, &cfg).unwrap();
        let front0_objectives: Vec<Vec<f64>> =
            result.front0.iter().map(|&i| result.objectives[i].clone()).collect();
        let reference = problem.pareto_front(200).unwrap();
        let value = igd(&front0_objectives, &reference).unwrap();
        assert!(value < 0.05, "measured IGD {value} exceeds the anchored threshold 0.05");
    }
}
