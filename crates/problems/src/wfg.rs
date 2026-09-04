//! The WFG (Walking Fish Group) scalable multi-objective test-problem
//! toolkit: WFG1-WFG9, the full suite (WFG1-WFG3 landed first, commit
//! `a180ed1`; this task extends `Wfg::new`'s `which` range to WFG4-WFG9,
//! the final range, in place).
//!
//! ## Provenance (PROVENANCE-FIRST battle plan)
//!
//! **(a) The paper.** The IEEE TEC 2006 journal article (Huband, Hingston,
//! Barone, While, "A Review of Multiobjective Test Problems and a Scalable
//! Test Problem Toolkit," *IEEE Transactions on Evolutionary Computation*
//! 10(5):477-506, 2006) could not be fetched directly (IEEE Xplore returns
//! HTTP 418 to automated clients; ResearchGate 403; no open-access mirror
//! found for the journal version itself). Per this task's own battle plan,
//! the EMO2005 companion paper is pinned INSTEAD, and it is the "corrected
//! version: 22 June 2005" the plan explicitly accepts: Simon Huband, Luigi
//! Barone, Lyndon While, Phil Hingston, "A Scalable Multi-objective Test
//! Problem Toolkit," *Evolutionary Multi-Criterion Optimization: Third
//! International Conference, EMO 2005*, LNCS 3410, pp. 280-295,
//! Springer-Verlag, 2005 -- the toolkit's OWN `README.txt` (quoted in (b)
//! below) names this exact paper as the toolkit's primary reference. PDF
//! recovered from the Wayback Machine's archive of the toolkit's original
//! host (`wfg.csse.uwa.edu.au`, dead since; see (b)): a first capture
//! (2005-06-16) is the file titled "corrected version: 25 May 2005"; the
//! file was updated in place and a SECOND, different-digest capture
//! (2005-12-14 onward, unchanged through 2006-08-19) is titled "corrected
//! version: 22 June 2005" -- the exact date this task's brief requires.
//! Pinned: `https://web.archive.org/web/20051214183800/http://www.wfg.csse.uwa.edu.au:80/publications/WFG2005a_corrected.pdf`,
//! sha256 `6ef19d2563b2f9b70871e495509f09a41d5c66430b45bbe038424110d2e7b54a`.
//! Every formula below is transcribed from this PDF (`pdftotext -layout`),
//! cross-checked against (b).
//!
//! **(b) The official toolkit C++ (senior authority; toolkit-over-paper
//! ruling family).** Cloned `https://github.com/richardeverson/wfg`
//! (canonical host `wfg.csse.uwa.edu.au` is dead), commit
//! `a156c75ea4d3cd460ca95d66dc05dc1c065200f4` (2016-11-08). Its
//! `WFG_v2006.03.28/README.txt` states: "This archive contains a C++
//! implementation of the WFG Toolkit... as reported in: [WFG2005a] ... and
//! later updated in: ... A Review of Multiobjective Test Problems and a
//! Scalable Test Problem Toolkit. To appear in: IEEE Transactions on
//! Evolutionary Computation. ... This canonical form of this archive is
//! available at http://www.wfg.csse.uwa.edu.au/. ... Copyright 2006,
//! Walking Fish Group." -- confirming the provenance chain the plan
//! documents (this mirror's `WFG_v2006.03.28` dir IS the code as downloaded
//! from the official, now-dead site; independently corroborated by the
//! Wayback Machine, which separately archived the identical
//! `WFG_v2006.03.28.zip` from the live site's own `publications/data/`
//! directory in 2006). **License finding** (decides nothing here --
//! ORACLE-ONLY use, no line of the C++ enters this repo): the repo-root
//! `LICENSE` file is GPLv2, but it covers `richardeverson`'s OWN Python
//! wrapper code, not the vendored toolkit -- every file under
//! `WFG_v2006.03.28/` instead carries the WFG authors' own header: "Copyright
//! (c) 2005 The Walking Fish Group (WFG). This material is provided \"as
//! is\", with no warranty expressed or implied. Any use is at your own
//! risk. Permission to use or copy this software for any purpose is hereby
//! granted without fee, provided this notice is retained on all copies.
//! Permission to modify the code and to distribute modified code is
//! granted, provided a notice that the code was modified is included with
//! the above copyright notice." -- a permissive academic notice, distinct
//! from (and not superseded by) the wrapper's own GPLv2 file. **Compile**:
//! `g++ -std=c++11 -O2 -c` on every `Toolkit/*.cpp` file compiled CLEAN,
//! zero errors, zero warnings, ZERO portability edits needed (no
//! `WINDOWS.H`/`malloc.h`-style includes present at all in this version).
//! A small out-of-tree CLI probe driver (`probe.cpp`, calling the toolkit's
//! own public headers unmodified) was written in the scratchpad to extract
//! reference values; it never enters this repo. This task re-verified the
//! pinned PDF sha256 and the toolkit clone's commit hash (both matched
//! exactly) and extended `probe.cpp`'s own `wfg1|wfg2|wfg3` dispatch to also
//! cover `wfg4`..`wfg9` (same unmodified `Problems::WFG4`..`Problems::WFG9`
//! calls as the existing three), recompiling with the same
//! `g++ -std=c++11 -O2` flags -- again zero errors, zero warnings.
//!
//! **Toolkit-vs-paper divergence found (D scaling constant; toolkit
//! governs, no behavioral change here).** The compiled `CHANGE_LOG.txt`:
//! "2006-03-28: ... Updated ExampleShapes.cpp, FrameworkFunctions.h, and
//! FrameworkFunctions.cpp to reflect the introduction of the distance
//! scaling constant D by the IEEE TEC review paper. WFG1--WFG9, and I1--I5
//! have D values of 1.0, leaving their behaviour unchanged." I.e. the 2006
//! journal version generalizes the EMO2005 formula `fm = xM + Sm*hm` to
//! `fm = D*xM + Sm*hm`, but every WFG1-WFG9 instance fixes `D = 1.0`, so
//! this module's formulas (below, `D` omitted, matching the EMO2005 text
//! exactly) are bit-identical to the toolkit's own `D=1.0` evaluation --
//! recorded as a divergence for completeness, not because it changes any
//! computed value.
//!
//! ## The WFG formalism (section 4, quoted)
//!
//! ```text
//! Given                   z = {z1, ..., zk, zk+1, ..., zn}
//! Minimise      fm=1:M(x) = xM + Sm hm(x1, ..., xM-1)
//! where                   x = {x1, ..., xM} = {max(tpM, A1)(tp1 - 0.5) + 0.5, ...,
//!                                                   max(tpM, AM-1)(tpM-1 - 0.5) + 0.5, tpM}
//!                        tp = {tp1, ..., tpM} <- tp-1 <- ... <- t1 <- z[0,1]
//!                     z[0,1] = {z1,[0,1], ..., zn,[0,1]} = {z1/z1,max, ..., zn/zn,max}
//! ```
//! "M is the number of objectives, x is a set of M underlying parameters
//! (where xM is an underlying distance parameter, and x1:M-1 are underlying
//! position parameters), z is a set of k+l=n>=M working parameters (the
//! first k and the last l working parameters are position- and
//! distance-related parameters respectively), A1:M-1 in {0,1} are
//! degeneracy constants (for each Ai=0, the dimensionality of the Pareto
//! optimal front is reduced by one), h1:M are shape functions, S1:M>0 are
//! scaling constants... The domain of all zi in z is [0, zi,max]." For the
//! WFG1-WFG9 suite (Table 6, quoted below), `zi=1:n,max = 2i` -- this
//! module's `Wfg::new`'s `x_i in [0, 2i]` domain (the paper's `z`, the raw
//! decision vector this crate's `Genotype`/`SearchSpace` carry; not to be
//! confused with the paper's OWN `x`, the internal `[0,1]`-domain
//! transformed vector computed inside `evaluate_batch`).
//!
//! ## Shape functions (Table 1, quoted; `x1,...,xM-1 in [0,1]`)
//!
//! ```text
//! Linear:    linear1 = prod_{i=1}^{M-1} xi
//!            linear_{m=2:M-1} = (prod_{i=1}^{M-m} xi)(1 - x_{M-m+1})
//!            linearM = 1 - x1
//! Convex:    convex1 = prod_{i=1}^{M-1} (1 - cos(xi*pi/2))
//!            convex_{m=2:M-1} = (prod_{i=1}^{M-m} (1-cos(xi*pi/2)))(1-sin(x_{M-m+1}*pi/2))
//!            convexM = 1 - sin(x1*pi/2)
//! Concave:   concave1 = prod_{i=1}^{M-1} sin(xi*pi/2)
//!            concave_{m=2:M-1} = (prod_{i=1}^{M-m} sin(xi*pi/2))(cos(x_{M-m+1}*pi/2))
//!            concaveM = cos(x1*pi/2)
//! Mixed (alpha>0, A in {1,2,...}):
//!            mixedM = 1 - x1 - (cos(2*A*pi*x1 + pi/2)/(2*A*pi))^alpha
//! Disconnected (alpha,beta>0, A in {1,2,...}):
//!            discM = 1 - x1^alpha * cos^2(A*x1^beta*pi)
//! ```
//!
//! ## Transformation functions (Table 2, quoted; primary parameter(s)
//! `y`/`y1,...,y|y|` always have domain `[0,1]`)
//!
//! ```text
//! Bias: Polynomial (alpha>0, alpha!=1)
//!   b_poly(y, alpha) = y^alpha
//! Bias: Flat Region (A,B,C in [0,1], B<C, ...)
//!   b_flat(y,A,B,C) = A + min(0,floor(y-B))*A*(B-y)/B - min(0,floor(C-y))*(1-A)*(y-C)/(1-C)
//! Bias: Parameter Dependent (A in (0,1), 0<B<C)
//!   b_param(y,y',A,B,C) = y^(B+(C-B)*v(u(y')))
//!   v(u) = A - (1-2u)*|floor(0.5-u)+A|
//! Shift: Linear (A in (0,1))
//!   s_linear(y,A) = |y-A| / |floor(A-y)+A|
//! Shift: Deceptive (A in (0,1), 0<B<<1, 0<C<<1, A-B>0, A+B<1)
//!   s_decept(y,A,B,C) = 1 + (|y-A|-B) *
//!     ( floor(y-A+B)*(1-C+(A-B)/B)/(A-B) + floor(A+B-y)*(1-C+(1-A-B)/B)/(1-A-B) + 1/B )
//! Shift: Multi-modal (A in {1,2,...}, B>=0, (4A+2)pi>=4B, C in (0,1))
//!   s_multi(y,A,B,C) = (1 + cos((4A+2)*pi*(0.5 - t)) + 4*B*t^2) / (B+2),
//!     where t = |y-C| / (2*(floor(C-y)+C))
//! Reduction: Weighted Sum (|w|=|y|, w1,...,w|y|>0)
//!   r_sum(y,w) = (sum wi*yi) / (sum wi)
//! Reduction: Non-separable (A in {1,...,|y|}, |y| mod A == 0)
//!   r_nonsep(y,A) = ( sum_j [ yj + sum_{k=0}^{A-2} |yj - y_{1+(j+k) mod |y|}| ] )
//!                   / ( |y| * ceil(A/2) * (1+2A-2*ceil(A/2)) / A )
//! ```
//!
//! ## WFG1-WFG3 (Table 6, quoted), and the shared setup
//!
//! All problems: `Sm=1:M = 2m`; `A1 = 1`; `A2:M-1 = 0 for WFG3, 1 otherwise`
//! ("ensures the Pareto optimal fronts are not degenerate, except ... WFG3,
//! which has a one dimensional Pareto optimal front"); `zi=1:n,max = 2i`.
//! "The number of position-related parameters, k, must be divisible by the
//! number of underlying position parameters, M-1... The number of
//! distance-related parameters, l, can be set to any positive integer,
//! except for WFG2 and WFG3, for which l must be a multiple of two (due to
//! the nature of their non-separable reductions)." -- independently
//! confirmed by the compiled toolkit's own `ExampleProblems.cpp` `ArgsOK`
//! (`return k>=1 && k<n && M>=2 && k%(M-1)==0;`) and `WFG2_t2`'s own
//! `assert(l%2==0)`. For any transition vector `ti`, `y = t_{i-1}` (`y =
//! z[0,1]` for `t1`).
//!
//! ```text
//! WFG1  Shape:  hm=1:M-1 = convexm;  hM = mixedM (alpha=1, A=5)
//!   t1  t1_i=1:k = yi;  t1_i=k+1:n = s_linear(yi, 0.35)
//!   t2  t2_i=1:k = yi;  t2_i=k+1:n = b_flat(yi, 0.8, 0.75, 0.85)
//!   t3  t3_i=1:n = b_poly(yi, 0.02)
//!   t4  t4_i=1:M-1 = r_sum({y_{(i-1)k/(M-1)+1},...,y_{ik/(M-1)}},
//!                          {2((i-1)k/(M-1)+1),...,2ik/(M-1)})
//!       t4_M = r_sum({y_{k+1},...,y_n}, {2(k+1),...,2n})
//!
//! WFG2  Shape:  hm=1:M-1 = convexm;  hM = discM (alpha=beta=1, A=5)
//!   t1  As t1 from WFG1. (Linear shift.)
//!   t2  t2_i=1:k = yi
//!       t2_i=k+1:k+l/2 = r_nonsep({y_{k+2(i-k)-1}, y_{k+2(i-k)}}, 2)
//!   t3  t3_i=1:M-1 = r_sum({y_{(i-1)k/(M-1)+1},...,y_{ik/(M-1)}}, {1,...,1})
//!       t3_M = r_sum({y_{k+1},...,y_{k+l/2}}, {1,...,1})
//!
//! WFG3  Shape:  hm=1:M = linearm (degenerate)
//!   t1:3  As t1:3 from WFG2. (Linear shift, non-separable reduction, and
//!         weighted sum reduction.)
//!
//! WFG4  Shape:  hm=1:M = concavem
//!   t1  t1_i=1:n = s_multi(yi, 30, 10, 0.35)
//!   t2  t2_i=1:M-1 = r_sum({y_{(i-1)k/(M-1)+1},...,y_{ik/(M-1)}}, {1,...,1})
//!       t2_M = r_sum({y_{k+1},...,y_n}, {1,...,1})
//!
//! WFG5  Shape:  hm=1:M = concavem
//!   t1  t1_i=1:n = s_decept(yi, 0.35, 0.001, 0.05)
//!   t2  As t2 from WFG4. (Weighted sum reduction.)
//!
//! WFG6  Shape:  hm=1:M = concavem
//!   t1  As t1 from WFG1. (Linear shift.)
//!   t2  t2_i=1:M-1 = r_nonsep({y_{(i-1)k/(M-1)+1},...,y_{ik/(M-1)}}, k/(M-1))
//!       t2_M = r_nonsep({y_{k+1},...,y_n}, l)
//!
//! WFG7  Shape:  hm=1:M = concavem
//!   t1  t1_i=1:k = b_param(yi, r_sum({y_{i+1},...,y_n}, {1,...,1}), 0.98/49.98, 0.02, 50)
//!       t1_i=k+1:n = yi
//!   t2  As t1 from WFG1. (Linear shift.)
//!   t3  As t2 from WFG4. (Weighted sum reduction.)
//!
//! WFG8  Shape:  hm=1:M = concavem
//!   t1  t1_i=1:k = yi
//!       t1_i=k+1:n = b_param(yi, r_sum({y_1,...,y_{i-1}}, {1,...,1}), 0.98/49.98, 0.02, 50)
//!   t2  As t1 from WFG1. (Linear shift.)
//!   t3  As t2 from WFG4. (Weighted sum reduction.)
//!
//! WFG9  "As the example in Section 5" (Table 6's own text; Section 5's
//! worked example IS WFG9, independently confirmed by the compiled
//! toolkit's `ExampleProblems.cpp` `Problems::WFG9`, which calls exactly
//! `WFG9_t1`/`WFG9_t2`/`WFG6_t2` below -- toolkit-over-paper, quoted from
//! `ExampleTransitions.cpp` since Table 6 declines to spell WFG9 out
//! directly):
//!   Shape:  hm=1:M = concavem
//!   t1  t1_i=1:n-1 = b_param(yi, r_sum({y_{i+1},...,y_n}, {1,...,1}), 0.98/49.98, 0.02, 50)
//!       t1_n = yn
//!   t2  t2_i=1:k = s_decept(yi, 0.35, 0.001, 0.05)
//!       t2_i=k+1:n = s_multi(yi, 30, 95, 0.35)
//!   t3  As t2 from WFG6. (Non-separable reduction.)
//! ```
//!
//! "For WFG1-WFG7, a solution is Pareto optimal iff all `zi=k+1:n = 2i *
//! 0.35`, noting WFG2 is disconnected." -- the pinned-x boundary fixtures
//! below use this exact value to construct known-Pareto-optimal test
//! points (cross-checked against the compiled toolkit, not merely against
//! this module's own formulas). WFG8 and WFG9 have their own, more
//! elaborate optimality conditions (quoted in the `pareto_front` decisions
//! section below) that this module's fixtures do not need, since
//! `pareto_front`'s closed form (below) is independent of `which` beyond
//! `m`.
//!
//! ## Recommended `k`/`l` defaults (NOT a paper quote -- the EMO2005 paper
//! pinned above states no explicit default; per this task's own provenance
//! ruling ("if (a) fails, the toolkit alone MAY carry the task... every
//! formula must be transcribed from the C++"), the compiled toolkit's own
//! `WFG_v2006.03.28/README.txt` is quoted here as the reference-implementation
//! convention -- the SAME "not chapter-quoted, but the widely-used
//! reference default" status `crate::dtlz`'s module doc already gives
//! DTLZ1's `k=5`)
//!
//! "Recommended values for k and l: Whilst it is largely a matter of
//! choice, letting l=20 should be sufficient for the number of distance
//! related parameters. Whilst suggesting a value for k, the number of
//! position related parameters, is more difficult, we suggest the
//! following: If M == 2, then let l == 4. Otherwise, if M >= 3, let l ==
//! 2*(M-1)." -- `// sezgi decision:` the toolkit's own text says "let l"
//! twice in a paragraph explicitly about choosing "a value for k" (the
//! surrounding sentence is about k, not l, and l was already fixed at 20
//! one sentence earlier); read as the authors' own typo for "let k", which
//! is also the reading consistent with the widely-used literature
//! convention (k=4 for M=2, k=2(M-1) for M>=3, l=20) this brief itself
//! names as the value to VERIFY. Documented here as a TRANSCRIPTION of the
//! toolkit's literal text (typo included) plus the corrected reading this
//! module's constructor accepts as any valid `(k,l)`, never silently
//! substituted.
//!
//! ## `pareto_front` decisions (`// sezgi decision:` per problem)
//!
//! **WFG1: `None`.** The formalism's own note (section 4, quoted above):
//! "substituting in xM=0 and disregarding all transition vectors provides a
//! parametric equation that covers and is covered by the Pareto optimal
//! front of the actual problem" would, read alone, license sampling
//! `f = Sm*hm(x1,...,xM-1)` over the free box `x1,...,xM-1 in [0,1]^{M-1}`
//! directly. But WFG1's `hM = mixedM` is NOT monotonic in `x1` (Table 1:
//! "the Pareto optimal front [contains] both convex and concave segments");
//! this task's brief explicitly flags WFG1's front as "nontrivial" (citing
//! pymoo issue #157) precisely because a non-monotonic shape function can
//! make a naively-sampled point on the raw manifold DOMINATED by another
//! point on the same manifold, and neither the EMO2005 paper nor the
//! compiled toolkit gives a closed form (or a runnable front-generation
//! mode) for the dominance-filtered subset. Rather than assert an
//! unverified non-domination claim, this module returns `None` (with this
//! note standing in for a sampling scheme), per the brief's own
//! "`None` with a doc note is acceptable" allowance.
//!
//! **WFG2: `None`.** Section 7 explicitly flags WFG2 alone ("noting WFG2 is
//! disconnected") among WFG1-WFG7 when stating the Pareto-optimality
//! condition, and Fig. 1's own caption for its `disc` shape illustration
//! reads "(dominated regions not removed, ...)" -- i.e. the raw `discM`
//! manifold, as plotted by the SAME shape function WFG2 uses for `hM`,
//! contains points that are not actually part of the Pareto-optimal front.
//! No closed form for the dominance-filtered/disconnected subset is given.
//! `None`, per the same brief allowance as WFG1.
//!
//! **WFG3: `Some`, closed form.** WFG3 is the ONE case this task's brief
//! calls "degenerate" rather than merely "nontrivial", and it genuinely
//! IS trivial: `A2:M-1 = 0` (quoted above) collapses `x2,...,xM-1` to the
//! CONSTANT `0.5` at every Pareto-optimal point (`x_i = max(tpM,Ai)*(tpi -
//! 0.5)+0.5`; at the optimum `tpM=0` and `Ai=0` gives `x_i =
//! 0*(...)+0.5=0.5` regardless of `tpi`), leaving only `x1` free (`A1=1`
//! always, so `x1 = tp1` directly) -- a one-dimensional curve, independent
//! of `M`, exactly mirroring `crate::dtlz`'s own DTLZ5/DTLZ6 degenerate-curve
//! precedent (same construction: one free coordinate, the rest pinned by a
//! degeneracy constant, sampled via `n` evenly spaced values of the free
//! coordinate). `linearm`'s formula involves no non-monotonic term (Table
//! 1), so every point of this 1-parameter curve is genuinely
//! Pareto-optimal (no dominance-removal ambiguity, unlike WFG1/WFG2). This
//! module's `wfg3_front_point` (used by both `pareto_front` and its own
//! round-trip regression tests) was independently cross-checked against
//! the compiled toolkit: a `z` was engineered so that every position
//! parameter shares one fraction `x1` (so `r_sum`/`r_nonsep`'s reductions,
//! being either an unweighted average of equal inputs or an average-family
//! reduction of equal inputs, all reduce to that same `x1` exactly) and the
//! distance parameters are set to the Pareto-optimal `2i*0.35`; the
//! compiled toolkit's `WFG3(z,k,M)` output matches this module's closed
//! form to the toolkit's own double precision at every probed `x1` (module
//! tests, `wfg3_pareto_front_matches_toolkit_engineered_z`).
//!
//! **WFG4-WFG9: `Some`, closed form (the scaled unit hypersphere).** All six
//! share `hm=1:M = concavem` (quoted above) and the SAME non-degenerate
//! `A1:M-1 = 1` (module doc's "All Constants" quote), so at any
//! Pareto-optimal point `tpM = 0` (the distance parameter reduces to zero;
//! see below), giving `x_i = max(0,1)*(tpi-0.5)+0.5 = tpi` for `i<M`
//! directly (identical algebra to WFG3's `x_i=tpi` step above, just without
//! the degeneracy). `concave`'s own formula (Table 1) is the textbook
//! hyperspherical parametrization (`concave1 = prod sin(xi*pi/2)`,
//! `concaveM = cos(x1*pi/2)`, etc.) -- `sin^2+cos^2=1` telescopes across the
//! product exactly as it does for `crate::dtlz`'s own DTLZ2/3/4
//! `cosine_cascade` sphere front (same identity, sin/cos roles swapped), so
//! `sum_{m=1}^{M} concave_m(x)^2 = 1` for ANY `x1,...,xM-1 in [0,1]^{M-1}`.
//! With `Sm=2m` and (at the optimum) the additive distance term `xM=0`
//! dropping out of `calculate_f`'s `fm = xM + Sm*hm`, this gives `fm =
//! 2m*concave_m(x)` exactly, so `sum_m (fm/(2m))^2 = 1` -- the scaled unit
//! hypersphere the brief names, VERIFIED (not assumed) by this derivation
//! from Table 1's own quoted formula plus the shared `A/S` constants, not
//! merely asserted.
//!
//! `// sezgi decision:` **whether `x1,...,xM-1` genuinely range over the
//! FULL box `[0,1]^{M-1}` (not merely a sub-manifold) for all six problems,
//! not only whether the shape formula is a sphere.** For WFG1-WFG7 the
//! Pareto-optimality condition (quoted above, `zi=k+1:n=2i*0.35`) fixes
//! ONLY the distance-related `z`; every position-related `zi=1:k` is
//! unconstrained. The EMO2005 paper's OWN worked example (Section 5, which
//! Table 6 says WFG9 in particular follows verbatim) states this
//! explicitly: "Once the optimal values for `zk+1:n` are determined, the
//! position-related parameters can be varied arbitrarily to obtain
//! different Pareto optimal solutions" -- i.e. the position parameters are
//! a genuine FREE `(M-1)`-dimensional degree of freedom for the whole
//! suite, by the formalism's own construction (section 4: "x1,...,xM-1 are
//! underlying position parameters"), not a claim specific to WFG9. Table 7
//! lists plain "concave" geometry for WFG4-WFG9 (no "disconnected" flag
//! like WFG2, no "mixed" flag like WFG1), so -- unlike WFG1/WFG2 above --
//! there is no non-monotonic or disconnected-manifold caveat to rule out
//! any region of the sphere. `pareto_front` therefore returns `Some` for
//! ALL SIX (`which` 4..=9), sharing one closed-form point function
//! (`wfg_concave_front_point`) with `which` NOT even an input to it (the
//! front depends only on `m` and the swept `x_pos`, never on which of the
//! six transition stacks produced `tpM=0` -- module code).
//!
//! **Toolkit-anchoring scope** (`// sezgi decision:`, honestly bounded).
//! `wfg_concave_front_point` was cross-checked against the compiled toolkit
//! via engineered `z` for WFG4, WFG5, and WFG6 specifically (module tests
//! `wfg{4,5,6}_pareto_front_matches_toolkit_engineered_z`): each problem's
//! position-related transition stack (WFG4/WFG5: elementwise `s_multi`/
//! `s_decept` then an UNWEIGHTED `r_sum` group reduction; WFG6: `r_nonsep`
//! group reduction) preserves a per-group CONSTANT input exactly through
//! the reduction (an unweighted average, or a non-separable reduction, of
//! `c` repeated `c` times is `c`-derived by a closed, already
//! toolkit-probed formula -- `r_sum`/`r_nonsep`'s OWN fixtures above), so a
//! `z` with each position GROUP set to one shared fraction lets each
//! `x_pos` coordinate be swept independently and compared against the
//! toolkit's real `WFG4`/`WFG5`/`WFG6` output at that engineered `z`.
//! WFG7/WFG8/WFG9 were NOT independently swept this same way: their
//! position transforms (`b_param`, quoted above) make a position slot's
//! output depend on OTHER slots' raw values (Table 7: "position-related
//! parameters ... dependent on ... distance-related parameters (and other
//! position-related parameters)"), so a per-group-constant `z` does not
//! collapse to a simple, independently-verifiable closed form the way it
//! does for WFG4-WFG6. Since `wfg_concave_front_point` is the EXACT SAME
//! function for all six (previous paragraph), the WFG4-WFG6 toolkit
//! cross-check already exercises the shared front-point code; what is NOT
//! independently confirmed for WFG7-WFG9 specifically is that their OWN
//! transition stacks actually reach `tpM=0` with `tp1,...,tpM-1` covering
//! the full free box (rather than some narrower reachable subset) -- this
//! module relies on the paper's own general "varied arbitrarily" text
//! (quoted above) for that claim, for WFG7-WFG9, rather than an
//! independent per-problem toolkit sweep. Recorded as a known scope
//! boundary, not silently assumed.

use std::f64::consts::PI;

use sezgi_core::mo::MoProblem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

#[derive(Debug, thiserror::Error)]
pub enum WfgError {
    #[error("which must be one of 1..=9 (WFG1-WFG9, the full suite), got {0}")]
    UnknownWhich(u32),
    #[error("m (number of objectives) must be >= 2, got {0}")]
    BadM(usize),
    #[error(
        "k (number of position-related parameters) must be > 0 and divisible by m-1 \
         (k % (m-1) == 0, Table 6), got k={k} m={m}"
    )]
    BadK { k: usize, m: usize },
    #[error("l (number of distance-related parameters) must be > 0, got {0}")]
    BadL(usize),
    #[error(
        "WFG{which}'s non-separable reduction pairs consecutive distance parameters (Table 6 \
         / ExampleTransitions.cpp's WFG2_t2, `assert(l % 2 == 0)`), so l must be even; got l={l}"
    )]
    OddL { which: u32, l: usize },
}

/// One instance of the WFG scalable multi-objective suite, `which` in
/// `1..=9` (WFG1-WFG9, the full suite; see the module doc for provenance).
pub struct Wfg {
    which: u32,
    m: usize,
    k: usize,
    l: usize,
    n: usize,
    space: SearchSpace,
}

impl Wfg {
    /// `which` in `1..=9`; `m >= 2`; `k > 0` and `k % (m-1) == 0` (Table 6);
    /// `l > 0`, additionally EVEN for WFG2/WFG3 (Table 6's own non-separable
    /// reduction requirement, independently confirmed by the toolkit's own
    /// `assert`, module doc; WFG6/WFG9 also use a non-separable reduction,
    /// but their own `A` parameter always equals the FULL group/tail length
    /// -- `r_nonsep`'s own `|y| mod A == 0` restriction is then trivially
    /// satisfied for ANY `l`, module doc's `r_nonsep` quote -- so no extra
    /// parity restriction applies to them). Domain: `zi in [0, 2i]` for `i =
    /// 1,...,n` (`n = k+l`), the paper's own `zi,max = 2i` (Table 6).
    pub fn new(which: u32, m: usize, k: usize, l: usize) -> Result<Self, WfgError> {
        if !matches!(which, 1..=9) {
            return Err(WfgError::UnknownWhich(which));
        }
        if m < 2 {
            return Err(WfgError::BadM(m));
        }
        if k == 0 || !k.is_multiple_of(m - 1) {
            return Err(WfgError::BadK { k, m });
        }
        if l == 0 {
            return Err(WfgError::BadL(l));
        }
        if matches!(which, 2 | 3) && !l.is_multiple_of(2) {
            return Err(WfgError::OddL { which, l });
        }
        let n = k + l;
        let blocks =
            (1..=n).map(|i| Block::Float { lo: 0.0, hi: 2.0 * i as f64, n: 1 }).collect();
        let space = SearchSpace::new(blocks).expect("WFG bounds (0 < 2i) are always valid");
        Ok(Self { which, m, k, l, n, space })
    }

    pub fn which(&self) -> u32 { self.which }
    pub fn m(&self) -> usize { self.m }
    pub fn k(&self) -> usize { self.k }
    pub fn l(&self) -> usize { self.l }
    pub fn n(&self) -> usize { self.n }

    // ---- transformation functions (Table 2, TransFunctions.cpp) ----

    /// `correct_to_01`, `Misc.cpp`: clamps values within `epsilon=1e-10` of
    /// the `[0,1]` boundary exactly onto that boundary; leaves everything
    /// else (including small positive-side or negative-side FP noise NOT
    /// snapped by this asymmetric rule) untouched, matching the C exactly.
    fn correct_to_01(a: f64) -> f64 {
        const EPSILON: f64 = 1.0e-10;
        if (-EPSILON..=0.0).contains(&a) {
            0.0
        } else if (1.0..=1.0 + EPSILON).contains(&a) {
            1.0
        } else {
            a
        }
    }

    fn b_poly(y: f64, alpha: f64) -> f64 { Self::correct_to_01(y.powf(alpha)) }

    fn b_flat(y: f64, a: f64, b: f64, c: f64) -> f64 {
        let tmp1 = f64::min(0.0, (y - b).floor()) * a * (b - y) / b;
        let tmp2 = f64::min(0.0, (c - y).floor()) * (1.0 - a) * (y - c) / (1.0 - c);
        Self::correct_to_01(a + tmp1 - tmp2)
    }

    /// Used by WFG7/WFG8/WFG9 (this task).
    fn b_param(y: f64, u: f64, a: f64, b: f64, c: f64) -> f64 {
        let v = a - (1.0 - 2.0 * u) * ((0.5 - u).floor() + a).abs();
        Self::correct_to_01(y.powf(b + (c - b) * v))
    }

    fn s_linear(y: f64, a: f64) -> f64 {
        Self::correct_to_01((y - a).abs() / ((a - y).floor() + a).abs())
    }

    /// Used by WFG5/WFG9 (this task).
    fn s_decept(y: f64, a: f64, b: f64, c: f64) -> f64 {
        let tmp1 = (y - a + b).floor() * (1.0 - c + (a - b) / b) / (a - b);
        let tmp2 = (a + b - y).floor() * (1.0 - c + (1.0 - a - b) / b) / (1.0 - a - b);
        Self::correct_to_01(1.0 + ((y - a).abs() - b) * (tmp1 + tmp2 + 1.0 / b))
    }

    /// Used by WFG4/WFG9 (this task).
    fn s_multi(y: f64, a: i64, b: f64, c: f64) -> f64 {
        let tmp1 = (y - c).abs() / (2.0 * ((c - y).floor() + c));
        let tmp2 = (4.0 * a as f64 + 2.0) * PI * (0.5 - tmp1);
        Self::correct_to_01((1.0 + tmp2.cos() + 4.0 * b * tmp1.powi(2)) / (b + 2.0))
    }

    fn r_sum(y: &[f64], w: &[f64]) -> f64 {
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (&yi, &wi) in y.iter().zip(w.iter()) {
            numerator += wi * yi;
            denominator += wi;
        }
        Self::correct_to_01(numerator / denominator)
    }

    fn r_nonsep(y: &[f64], a: usize) -> f64 {
        let y_len = y.len();
        let mut numerator = 0.0;
        for j in 0..y_len {
            numerator += y[j];
            for kk in 0..a.saturating_sub(1) {
                numerator += (y[j] - y[(j + kk + 1) % y_len]).abs();
            }
        }
        let tmp = (a as f64 / 2.0).ceil();
        let denominator = y_len as f64 * tmp * (1.0 + 2.0 * a as f64 - 2.0 * tmp) / a as f64;
        Self::correct_to_01(numerator / denominator)
    }

    // ---- shape functions (Table 1, ShapeFunctions.cpp) ----
    //
    // `x` is always the FULL `M`-length `calculate_x` output (position
    // parameters `x[0..M-1]` plus the distance parameter `x[M-1]`, per
    // FrameworkFunctions.cpp), matching the toolkit exactly -- `x[M-1]` is
    // never read by any of these (the loop bound `M-m` and the extra-term
    // index `M-m` both stay `< M-1` for every `m` in `1..=M`), but keeping
    // `x.len() == M` (not `M-1`) matters for the internal `M-m` index math.

    fn shape_linear(x: &[f64], m: usize) -> f64 {
        let big_m = x.len();
        let mut result = 1.0;
        for &xi in &x[0..big_m - m] {
            result *= xi;
        }
        if m != 1 {
            result *= 1.0 - x[big_m - m];
        }
        Self::correct_to_01(result)
    }

    fn shape_convex(x: &[f64], m: usize) -> f64 {
        let big_m = x.len();
        let mut result = 1.0;
        for &xi in &x[0..big_m - m] {
            result *= 1.0 - (xi * PI / 2.0).cos();
        }
        if m != 1 {
            result *= 1.0 - (x[big_m - m] * PI / 2.0).sin();
        }
        Self::correct_to_01(result)
    }

    /// Used by WFG4-WFG9 (this task).
    fn shape_concave(x: &[f64], m: usize) -> f64 {
        let big_m = x.len();
        let mut result = 1.0;
        for &xi in &x[0..big_m - m] {
            result *= (xi * PI / 2.0).sin();
        }
        if m != 1 {
            result *= (x[big_m - m] * PI / 2.0).cos();
        }
        Self::correct_to_01(result)
    }

    fn shape_mixed(x: &[f64], a: i64, alpha: f64) -> f64 {
        let tmp = 2.0 * a as f64 * PI;
        Self::correct_to_01((1.0 - x[0] - (tmp * x[0] + PI / 2.0).cos() / tmp).powf(alpha))
    }

    fn shape_disc(x: &[f64], a: i64, alpha: f64, beta: f64) -> f64 {
        let tmp1 = a as f64 * x[0].powf(beta) * PI;
        Self::correct_to_01(1.0 - x[0].powf(alpha) * tmp1.cos().powi(2))
    }

    // ---- framework functions (FrameworkFunctions.cpp) ----

    /// `WFG_normalise_z`, `ExampleProblems.cpp`: divides each `zi` by its
    /// own domain bound `2i` (1-based `i`).
    fn wfg_normalise_z(z: &[f64]) -> Vec<f64> {
        z.iter().enumerate().map(|(i0, &zi)| zi / (2.0 * (i0 + 1) as f64)).collect()
    }

    /// `WFG_create_A`, `ExampleShapes.cpp`: `[1,0,0,...]` (length `M-1`) if
    /// `degenerate`, else `[1,1,...]`.
    fn wfg_create_a(m: usize, degenerate: bool) -> Vec<u8> {
        let mut a = vec![1u8; m - 1];
        if degenerate {
            for ai in a.iter_mut().skip(1) {
                *ai = 0;
            }
        }
        a
    }

    /// `calculate_x`, `FrameworkFunctions.cpp`.
    fn calculate_x(t_p: &[f64], a: &[u8]) -> Vec<f64> {
        let big_m = t_p.len();
        let mut result = Vec::with_capacity(big_m);
        for i in 0..big_m - 1 {
            let tmp1 = f64::max(t_p[big_m - 1], a[i] as f64);
            result.push(tmp1 * (t_p[i] - 0.5) + 0.5);
        }
        result.push(t_p[big_m - 1]);
        result
    }

    /// `calculate_f`, `FrameworkFunctions.cpp`, with `D=1.0` (module doc's
    /// divergence note: every WFG1-WFG9 instance fixes `D=1.0`).
    fn calculate_f(x: &[f64], h: &[f64], s: &[f64]) -> Vec<f64> {
        let x_m = *x.last().expect("x is never empty (M >= 2)");
        h.iter().zip(s.iter()).map(|(&hi, &si)| x_m + si * hi).collect()
    }

    fn s_vec(m: usize) -> Vec<f64> { (1..=m).map(|mm| 2.0 * mm as f64).collect() }

    // ---- per-problem transition stacks (ExampleTransitions.cpp) ----

    fn wfg1_t1(y: &[f64], k: usize) -> Vec<f64> {
        let mut t = y[..k].to_vec();
        t.extend(y[k..].iter().map(|&yi| Self::s_linear(yi, 0.35)));
        t
    }

    fn wfg1_t2(y: &[f64], k: usize) -> Vec<f64> {
        let mut t = y[..k].to_vec();
        t.extend(y[k..].iter().map(|&yi| Self::b_flat(yi, 0.8, 0.75, 0.85)));
        t
    }

    fn wfg1_t3(y: &[f64]) -> Vec<f64> { y.iter().map(|&yi| Self::b_poly(yi, 0.02)).collect() }

    fn wfg1_t4(y: &[f64], k: usize, m: usize) -> Vec<f64> {
        let n = y.len();
        let w: Vec<f64> = (1..=n).map(|i| 2.0 * i as f64).collect();
        let mut t = Vec::with_capacity(m);
        for i in 1..=(m - 1) {
            let head = (i - 1) * k / (m - 1);
            let tail = i * k / (m - 1);
            t.push(Self::r_sum(&y[head..tail], &w[head..tail]));
        }
        t.push(Self::r_sum(&y[k..n], &w[k..n]));
        t
    }

    /// `WFG2_t2`: `y[0..k]` unchanged, then `l/2` `r_nonsep`-of-consecutive-pairs.
    fn wfg2_t2(y: &[f64], k: usize) -> Vec<f64> {
        let n = y.len();
        let l = n - k;
        let mut t = y[..k].to_vec();
        for pair in 0..l / 2 {
            let head = k + 2 * pair;
            t.push(Self::r_nonsep(&y[head..head + 2], 2));
        }
        t
    }

    /// `WFG2_t3`: unweighted `r_sum` groups (position groups of `k/(M-1)`,
    /// one distance group over the whole reduced tail) -- shared by WFG2 and
    /// WFG3 (module doc's Table 6 quote: "As t1:3 from WFG2").
    fn wfg2_t3(y: &[f64], k: usize, m: usize) -> Vec<f64> {
        let n = y.len();
        let w = vec![1.0; n];
        let mut t = Vec::with_capacity(m);
        for i in 1..=(m - 1) {
            let head = (i - 1) * k / (m - 1);
            let tail = i * k / (m - 1);
            t.push(Self::r_sum(&y[head..tail], &w[head..tail]));
        }
        t.push(Self::r_sum(&y[k..n], &w[k..n]));
        t
    }

    /// `WFG4_t1`/`WFG5_t1`-style elementwise transform, applied to EVERY
    /// `y_i` (both position and distance, module doc's Table 6 quote: `t1
    /// i=1:n`) -- shared code, parameterized by the closure so WFG4's
    /// `s_multi(yi,30,10,0.35)` and WFG5's `s_decept(yi,0.35,0.001,0.05)`
    /// reuse the same loop.
    fn elementwise(y: &[f64], f: impl Fn(f64) -> f64) -> Vec<f64> { y.iter().map(|&yi| f(yi)).collect() }

    fn wfg4_t1(y: &[f64]) -> Vec<f64> { Self::elementwise(y, |yi| Self::s_multi(yi, 30, 10.0, 0.35)) }

    fn wfg5_t1(y: &[f64]) -> Vec<f64> { Self::elementwise(y, |yi| Self::s_decept(yi, 0.35, 0.001, 0.05)) }

    /// `WFG6_t2`: `r_nonsep` groups (position groups of `k/(M-1)`, one
    /// distance group over the whole tail `l`) -- shared by WFG6 and WFG9
    /// (module doc's Table 6 quote: "As t2 from WFG6").
    fn wfg6_t2(y: &[f64], k: usize, m: usize) -> Vec<f64> {
        let n = y.len();
        let mut t = Vec::with_capacity(m);
        for i in 1..=(m - 1) {
            let head = (i - 1) * k / (m - 1);
            let tail = i * k / (m - 1);
            t.push(Self::r_nonsep(&y[head..tail], k / (m - 1)));
        }
        t.push(Self::r_nonsep(&y[k..n], n - k));
        t
    }

    /// `WFG7_t1`: position slots (`i<k`) get `b_param` biased by the
    /// weighted sum of every LATER `y` (position and distance); distance
    /// slots (`i>=k`) pass through unchanged.
    fn wfg7_t1(y: &[f64], k: usize) -> Vec<f64> {
        let n = y.len();
        let w = vec![1.0; n];
        let mut t = Vec::with_capacity(n);
        for i in 0..k {
            let u = Self::r_sum(&y[i + 1..n], &w[i + 1..n]);
            t.push(Self::b_param(y[i], u, 0.98 / 49.98, 0.02, 50.0));
        }
        t.extend_from_slice(&y[k..n]);
        t
    }

    /// `WFG8_t1`: position slots (`i<k`) pass through unchanged; distance
    /// slots (`i>=k`) get `b_param` biased by the weighted sum of every
    /// EARLIER `y` (position and any earlier distance).
    fn wfg8_t1(y: &[f64], k: usize) -> Vec<f64> {
        let n = y.len();
        let w = vec![1.0; n];
        let mut t = y[..k].to_vec();
        for i in k..n {
            let u = Self::r_sum(&y[..i], &w[..i]);
            t.push(Self::b_param(y[i], u, 0.98 / 49.98, 0.02, 50.0));
        }
        t
    }

    /// `WFG9_t1` (== `I2_t1`): every `y_i` except the last gets `b_param`
    /// biased by the weighted sum of every LATER `y`; the last `y` passes
    /// through unchanged.
    fn wfg9_t1(y: &[f64]) -> Vec<f64> {
        let n = y.len();
        let w = vec![1.0; n];
        let mut t = Vec::with_capacity(n);
        for i in 0..n - 1 {
            let u = Self::r_sum(&y[i + 1..n], &w[i + 1..n]);
            t.push(Self::b_param(y[i], u, 0.98 / 49.98, 0.02, 50.0));
        }
        t.push(y[n - 1]);
        t
    }

    /// `WFG9_t2`: `s_decept` on position slots (`i<k`), `s_multi` on
    /// distance slots (`i>=k`), NOTE the different `B` constant (95, not
    /// 10) from WFG4's own `s_multi` call (module doc's Table 6 quote).
    fn wfg9_t2(y: &[f64], k: usize) -> Vec<f64> {
        let n = y.len();
        let mut t = Vec::with_capacity(n);
        for &yi in &y[..k] {
            t.push(Self::s_decept(yi, 0.35, 0.001, 0.05));
        }
        for &yi in &y[k..n] {
            t.push(Self::s_multi(yi, 30, 95.0, 0.35));
        }
        t
    }

    // ---- per-problem evaluation (ExampleProblems.cpp / ExampleShapes.cpp) ----

    fn eval_wfg1(&self, z: &[f64]) -> Vec<f64> {
        let (k, m) = (self.k, self.m);
        let y = Self::wfg_normalise_z(z);
        let y = Self::wfg1_t1(&y, k);
        let y = Self::wfg1_t2(&y, k);
        let y = Self::wfg1_t3(&y);
        let t_p = Self::wfg1_t4(&y, k, m);
        let a = Self::wfg_create_a(m, false);
        let x = Self::calculate_x(&t_p, &a);
        let mut h = Vec::with_capacity(m);
        for mm in 1..=(m - 1) {
            h.push(Self::shape_convex(&x, mm));
        }
        h.push(Self::shape_mixed(&x, 5, 1.0));
        Self::calculate_f(&x, &h, &Self::s_vec(m))
    }

    fn eval_wfg2(&self, z: &[f64]) -> Vec<f64> {
        let (k, m) = (self.k, self.m);
        let y = Self::wfg_normalise_z(z);
        let y = Self::wfg1_t1(&y, k);
        let y = Self::wfg2_t2(&y, k);
        let t_p = Self::wfg2_t3(&y, k, m);
        let a = Self::wfg_create_a(m, false);
        let x = Self::calculate_x(&t_p, &a);
        let mut h = Vec::with_capacity(m);
        for mm in 1..=(m - 1) {
            h.push(Self::shape_convex(&x, mm));
        }
        h.push(Self::shape_disc(&x, 5, 1.0, 1.0));
        Self::calculate_f(&x, &h, &Self::s_vec(m))
    }

    fn eval_wfg3(&self, z: &[f64]) -> Vec<f64> {
        let (k, m) = (self.k, self.m);
        let y = Self::wfg_normalise_z(z);
        let y = Self::wfg1_t1(&y, k);
        let y = Self::wfg2_t2(&y, k);
        let t_p = Self::wfg2_t3(&y, k, m);
        let a = Self::wfg_create_a(m, true);
        let x = Self::calculate_x(&t_p, &a);
        let h: Vec<f64> = (1..=m).map(|mm| Self::shape_linear(&x, mm)).collect();
        Self::calculate_f(&x, &h, &Self::s_vec(m))
    }

    /// Shared by WFG4-WFG9: all six use `hm=1:M = concavem` (module doc's
    /// Table 6 quote) over the non-degenerate `A1:M-1=1` framework, so once
    /// `t_p` (`tp`) is computed only the shape function and its non-`Wfg1`
    /// degeneracy vector differ from `eval_wfg1`'s own pattern.
    fn eval_concave(t_p: &[f64], m: usize) -> Vec<f64> {
        let a = Self::wfg_create_a(m, false);
        let x = Self::calculate_x(t_p, &a);
        let h: Vec<f64> = (1..=m).map(|mm| Self::shape_concave(&x, mm)).collect();
        Self::calculate_f(&x, &h, &Self::s_vec(m))
    }

    fn eval_wfg4(&self, z: &[f64]) -> Vec<f64> {
        let (k, m) = (self.k, self.m);
        let y = Self::wfg_normalise_z(z);
        let y = Self::wfg4_t1(&y);
        let t_p = Self::wfg2_t3(&y, k, m);
        Self::eval_concave(&t_p, m)
    }

    fn eval_wfg5(&self, z: &[f64]) -> Vec<f64> {
        let (k, m) = (self.k, self.m);
        let y = Self::wfg_normalise_z(z);
        let y = Self::wfg5_t1(&y);
        let t_p = Self::wfg2_t3(&y, k, m);
        Self::eval_concave(&t_p, m)
    }

    fn eval_wfg6(&self, z: &[f64]) -> Vec<f64> {
        let (k, m) = (self.k, self.m);
        let y = Self::wfg_normalise_z(z);
        let y = Self::wfg1_t1(&y, k);
        let t_p = Self::wfg6_t2(&y, k, m);
        Self::eval_concave(&t_p, m)
    }

    fn eval_wfg7(&self, z: &[f64]) -> Vec<f64> {
        let (k, m) = (self.k, self.m);
        let y = Self::wfg_normalise_z(z);
        let y = Self::wfg7_t1(&y, k);
        let y = Self::wfg1_t1(&y, k);
        let t_p = Self::wfg2_t3(&y, k, m);
        Self::eval_concave(&t_p, m)
    }

    fn eval_wfg8(&self, z: &[f64]) -> Vec<f64> {
        let (k, m) = (self.k, self.m);
        let y = Self::wfg_normalise_z(z);
        let y = Self::wfg8_t1(&y, k);
        let y = Self::wfg1_t1(&y, k);
        let t_p = Self::wfg2_t3(&y, k, m);
        Self::eval_concave(&t_p, m)
    }

    fn eval_wfg9(&self, z: &[f64]) -> Vec<f64> {
        let (k, m) = (self.k, self.m);
        let y = Self::wfg_normalise_z(z);
        let y = Self::wfg9_t1(&y);
        let y = Self::wfg9_t2(&y, k);
        let t_p = Self::wfg6_t2(&y, k, m);
        Self::eval_concave(&t_p, m)
    }

    /// WFG3's closed-form Pareto-front point at free parameter `x1 in
    /// [0,1]` (module doc's `pareto_front` decision): `x = [x1, 0.5,
    /// ..., 0.5, 0.0]` (length `m`; the trailing `0.0` is the `xM` slot,
    /// never read by `shape_linear`, module doc), `fm = Sm *
    /// shape_linear(x, m)`.
    fn wfg3_front_point(m: usize, x1: f64) -> Vec<f64> {
        let mut x = vec![0.5; m - 1];
        x[0] = x1;
        x.push(0.0); // xM slot; unread by shape_linear, kept for x.len() == M
        let s = Self::s_vec(m);
        (1..=m).map(|mm| s[mm - 1] * Self::shape_linear(&x, mm)).collect()
    }

    /// WFG4-WFG9's shared closed-form Pareto-front point (module doc's
    /// `pareto_front` decision, "the scaled unit hypersphere"): `x_pos`
    /// (length `m-1`, each in `[0,1]`) plays the free position parameters
    /// directly (`x_i = tp_i` at the optimum, module doc derivation); `x =
    /// [x_pos, 0.0]` (the trailing `0.0` is the `xM` slot, unread by
    /// `shape_concave`, kept for `x.len() == M` as `wfg3_front_point` does);
    /// `fm = Sm * shape_concave(x, m)`. Deliberately takes NO `which`
    /// parameter -- the same function serves all six problems (module doc's
    /// `// sezgi decision:`).
    fn wfg_concave_front_point(m: usize, x_pos: &[f64]) -> Vec<f64> {
        debug_assert_eq!(x_pos.len(), m - 1);
        let mut x = x_pos.to_vec();
        x.push(0.0); // xM slot; unread by shape_concave, kept for x.len() == M
        let s = Self::s_vec(m);
        (1..=m).map(|mm| s[mm - 1] * Self::shape_concave(&x, mm)).collect()
    }

    // ---- pareto_front lattice sampling (WFG4-WFG9) ----
    //
    // `axis_grid`/`grid_r`/`cartesian_product` moved to
    // `crate::front_lattice` in M3-8 T1 (this crate's `dtlz::Dtlz` had a
    // byte-identical copy of this same trio) -- see that module's own doc
    // for the extraction note.
}

impl MoProblem for Wfg {
    fn space(&self) -> &SearchSpace { &self.space }
    fn n_objectives(&self) -> usize { self.m }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<Vec<f64>> {
        pop.iter()
            .map(|g| {
                if g.blocks.len() != self.n {
                    return vec![f64::INFINITY; self.m];
                }
                let mut z = Vec::with_capacity(self.n);
                for b in &g.blocks {
                    match b {
                        BlockValues::Float(v) if v.len() == 1 => z.push(v[0]),
                        _ => return vec![f64::INFINITY; self.m],
                    }
                }
                match self.which {
                    1 => self.eval_wfg1(&z),
                    2 => self.eval_wfg2(&z),
                    3 => self.eval_wfg3(&z),
                    4 => self.eval_wfg4(&z),
                    5 => self.eval_wfg5(&z),
                    6 => self.eval_wfg6(&z),
                    7 => self.eval_wfg7(&z),
                    8 => self.eval_wfg8(&z),
                    9 => self.eval_wfg9(&z),
                    _ => unreachable!("Wfg::new rejects which outside 1..=9"),
                }
            })
            .collect()
    }

    /// See the module doc's `pareto_front` decisions section: `None` for
    /// WFG1/WFG2 (non-monotonic/disconnected shape, no defensible
    /// dominance-filtered closed form); WFG3's one-dimensional degenerate
    /// curve for `which == 3`; WFG4-WFG9's shared scaled-unit-hypersphere
    /// closed form (an `(m-1)`-dimensional lattice over the free position
    /// parameters, `r = ceil(n^(1/(m-1)))` evenly spaced values per axis,
    /// mirroring `crate::dtlz`'s own DTLZ2/3/4 sphere-front lattice
    /// sampling -- `// sezgi decision:` same "approximately `n`" tradeoff
    /// DTLZ2 already documents: exact when `n` is a perfect `(m-1)`-th
    /// power, always exact for `m=2`).
    fn pareto_front(&self, n: usize) -> Option<Vec<Vec<f64>>> {
        if n == 0 {
            return Some(Vec::new());
        }
        match self.which {
            3 => {
                let axis: Vec<f64> = if n == 1 {
                    vec![0.0]
                } else {
                    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
                };
                Some(axis.iter().map(|&x1| Self::wfg3_front_point(self.m, x1)).collect())
            }
            4..=9 => {
                let dim_free = self.m - 1;
                let r = crate::front_lattice::grid_r(n, dim_free);
                let axis = crate::front_lattice::axis_grid(r);
                let combos = crate::front_lattice::cartesian_product(&axis, dim_free);
                Some(combos.iter().map(|c| Self::wfg_concave_front_point(self.m, c)).collect())
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==== constructor validation ====

    #[test]
    fn new_rejects_unknown_which() {
        assert!(matches!(Wfg::new(0, 2, 4, 4), Err(WfgError::UnknownWhich(0))));
        assert!(matches!(Wfg::new(10, 2, 4, 4), Err(WfgError::UnknownWhich(10))));
        assert!(matches!(Wfg::new(100, 2, 4, 4), Err(WfgError::UnknownWhich(100))));
        // which=4..=9 (this task's own extension) are now valid, not errors.
        for which in 4u32..=9 {
            assert!(Wfg::new(which, 2, 4, 4).is_ok(), "which={which}");
        }
    }

    #[test]
    fn new_rejects_bad_m() {
        assert!(matches!(Wfg::new(1, 0, 4, 4), Err(WfgError::BadM(0))));
        assert!(matches!(Wfg::new(1, 1, 4, 4), Err(WfgError::BadM(1))));
    }

    #[test]
    fn new_rejects_bad_k() {
        // k == 0
        assert!(matches!(Wfg::new(1, 2, 0, 4), Err(WfgError::BadK { k: 0, m: 2 })));
        // k not divisible by m-1
        assert!(matches!(Wfg::new(1, 3, 5, 4), Err(WfgError::BadK { k: 5, m: 3 })));
        // k divisible by m-1 is fine (sanity, not an error case)
        assert!(Wfg::new(1, 3, 6, 4).is_ok());
    }

    #[test]
    fn new_rejects_bad_l() {
        assert!(matches!(Wfg::new(1, 2, 4, 0), Err(WfgError::BadL(0))));
    }

    #[test]
    fn new_rejects_odd_l_for_wfg2_wfg3_not_wfg1() {
        assert!(matches!(Wfg::new(2, 2, 4, 5), Err(WfgError::OddL { which: 2, l: 5 })));
        assert!(matches!(Wfg::new(3, 2, 4, 5), Err(WfgError::OddL { which: 3, l: 5 })));
        // WFG1 has no such restriction.
        assert!(Wfg::new(1, 2, 4, 5).is_ok());
        // Even l is fine for WFG2/WFG3.
        assert!(Wfg::new(2, 2, 4, 4).is_ok());
        assert!(Wfg::new(3, 2, 4, 4).is_ok());
        // WFG4-WFG9 have NO such restriction either -- their own
        // non-separable reductions (WFG6/WFG9's `r_nonsep`) always use
        // `A == |y|` (module doc's `Wfg::new` comment), trivially satisfying
        // `r_nonsep`'s `|y| mod A == 0`, unlike WFG2/WFG3's fixed pair size
        // `A=2`.
        for which in 4u32..=9 {
            assert!(Wfg::new(which, 2, 4, 5).is_ok(), "which={which}");
        }
    }

    #[test]
    fn new_ok_accessors_and_domain() {
        let p = Wfg::new(1, 3, 6, 20).unwrap();
        assert_eq!((p.which(), p.m(), p.k(), p.l(), p.n()), (1, 3, 6, 20, 26));
        for (i0, b) in p.space().blocks().iter().enumerate() {
            let i = i0 + 1;
            assert_eq!(*b, Block::Float { lo: 0.0, hi: 2.0 * i as f64, n: 1 });
        }
    }

    // ==== transformation function hand fixtures, toolkit probe cross-checked
    // (compiled WFG_v2006.03.28, `probe` CLI, scratchpad-only; values below
    // are the probe's own stdout, `%.17g`) ====

    fn close(a: f64, b: f64) {
        let rel = if b.abs() > 1e-12 { (a - b).abs() / b.abs() } else { (a - b).abs() };
        assert!(rel < 1e-13, "a={a} b={b} rel={rel}");
    }

    #[test]
    fn b_poly_matches_toolkit() {
        // probe b_poly y 20
        close(Wfg::b_poly(0.0, 20.0), 0.0);
        close(Wfg::b_poly(0.25, 20.0), 9.094_947_017_729_282e-13);
        close(Wfg::b_poly(0.5, 20.0), 9.5367431640625e-07);
        close(Wfg::b_poly(0.75, 20.0), 0.0031712119389339932);
        close(Wfg::b_poly(1.0, 20.0), 1.0);
    }

    #[test]
    fn b_flat_matches_toolkit() {
        // probe b_flat y 0.7 0.4 0.5
        close(Wfg::b_flat(0.0, 0.7, 0.4, 0.5), 1.1102230246251565e-16);
        close(Wfg::b_flat(0.2, 0.7, 0.4, 0.5), 0.35000000000000003);
        close(Wfg::b_flat(0.4, 0.7, 0.4, 0.5), 0.7);
        close(Wfg::b_flat(0.45, 0.7, 0.4, 0.5), 0.7);
        close(Wfg::b_flat(0.5, 0.7, 0.4, 0.5), 0.7);
        close(Wfg::b_flat(0.7, 0.7, 0.4, 0.5), 0.82);
        close(Wfg::b_flat(1.0, 0.7, 0.4, 0.5), 1.0);
    }

    #[test]
    fn b_param_matches_toolkit() {
        // probe b_param y u 0.5 2 10
        close(Wfg::b_param(0.0, 0.0, 0.5, 2.0, 10.0), 0.0);
        close(Wfg::b_param(0.0, 1.0, 0.5, 2.0, 10.0), 0.0);
        close(Wfg::b_param(0.5, 0.0, 0.5, 2.0, 10.0), 0.25);
        close(Wfg::b_param(0.5, 0.5, 0.5, 2.0, 10.0), 0.015625);
        close(Wfg::b_param(0.5, 1.0, 0.5, 2.0, 10.0), 0.0009765625);
        close(Wfg::b_param(1.0, 0.5, 0.5, 2.0, 10.0), 1.0);
        close(Wfg::b_param(0.3, 0.7, 0.5, 2.0, 10.0), 0.0001061992710753656);
    }

    #[test]
    fn s_linear_matches_toolkit() {
        // probe s_linear y 0.35
        close(Wfg::s_linear(0.0, 0.35), 1.0);
        close(Wfg::s_linear(0.1, 0.35), 0.7142857142857143);
        close(Wfg::s_linear(0.35, 0.35), 0.0);
        close(Wfg::s_linear(0.6, 0.35), 0.384_615_384_615_384_6);
        close(Wfg::s_linear(1.0, 0.35), 1.0);
    }

    #[test]
    fn s_decept_matches_toolkit() {
        // probe s_decept y 0.35 0.005 0.05
        close(Wfg::s_decept(0.0, 0.35, 0.005, 0.05), 0.049_999_999_999_995_24);
        close(Wfg::s_decept(0.2, 0.35, 0.005, 0.05), 0.600_724_637_681_157_5);
        close(Wfg::s_decept(0.345, 0.35, 0.005, 0.05), 1.0);
        close(Wfg::s_decept(0.35, 0.35, 0.005, 0.05), 0.0);
        close(Wfg::s_decept(0.355, 0.35, 0.005, 0.05), 1.0);
        close(Wfg::s_decept(0.6, 0.35, 0.005, 0.05), 0.639_147_286_821_710_8);
        close(Wfg::s_decept(1.0, 0.35, 0.005, 0.05), 0.050000000000014325);
    }

    #[test]
    fn s_multi_matches_toolkit() {
        // probe s_multi y 5 10 0.35
        close(Wfg::s_multi(0.0, 5, 10.0, 0.35), 1.0);
        close(Wfg::s_multi(0.1, 5, 10.0, 0.35), 0.433_422_662_368_675_9);
        close(Wfg::s_multi(0.35, 5, 10.0, 0.35), 0.0);
        close(Wfg::s_multi(0.6, 5, 10.0, 0.35), 0.14423159938810814);
        close(Wfg::s_multi(1.0, 5, 10.0, 0.35), 1.0);
    }

    #[test]
    fn r_sum_matches_toolkit() {
        // probe r_sum 2 y1 y2 1 5
        close(Wfg::r_sum(&[0.0, 0.0], &[1.0, 5.0]), 0.0);
        close(Wfg::r_sum(&[1.0, 0.0], &[1.0, 5.0]), 0.16666666666666666);
        close(Wfg::r_sum(&[0.0, 1.0], &[1.0, 5.0]), 0.833_333_333_333_333_4);
        close(Wfg::r_sum(&[0.5, 0.5], &[1.0, 5.0]), 0.5);
        close(Wfg::r_sum(&[0.2, 0.8], &[1.0, 5.0]), 0.700_000_000_000_000_1);
    }

    #[test]
    fn r_nonsep_matches_toolkit() {
        // probe r_nonsep 4 2 y1 y2 y3 y4
        close(Wfg::r_nonsep(&[0.0, 0.0, 0.0, 0.0], 2), 0.0);
        close(Wfg::r_nonsep(&[1.0, 1.0, 1.0, 1.0], 2), 0.666_666_666_666_666_6);
        close(Wfg::r_nonsep(&[0.1, 0.2, 0.3, 0.4], 2), 0.266_666_666_666_666_7);
        close(Wfg::r_nonsep(&[0.9, 0.1, 0.9, 0.1], 2), 0.8666666666666667);
    }

    // ==== shape function hand fixtures, toolkit probe cross-checked ====

    #[test]
    fn shape_linear_matches_toolkit() {
        // probe shape_linear m 3 x1 x2 x3
        close(Wfg::shape_linear(&[0.0, 0.0, 0.0], 1), 0.0);
        close(Wfg::shape_linear(&[1.0, 1.0, 1.0], 1), 1.0);
        close(Wfg::shape_linear(&[0.5, 0.5, 0.5], 1), 0.25);
        close(Wfg::shape_linear(&[0.2, 0.7, 0.9], 1), 0.13999999999999999);
        close(Wfg::shape_linear(&[0.0, 0.0, 0.0], 2), 0.0);
        close(Wfg::shape_linear(&[1.0, 1.0, 1.0], 2), 0.0);
        close(Wfg::shape_linear(&[0.5, 0.5, 0.5], 2), 0.25);
        close(Wfg::shape_linear(&[0.2, 0.7, 0.9], 2), 0.060_000_000_000_000_01);
        close(Wfg::shape_linear(&[0.0, 0.0, 0.0], 3), 1.0);
        close(Wfg::shape_linear(&[1.0, 1.0, 1.0], 3), 0.0);
        close(Wfg::shape_linear(&[0.5, 0.5, 0.5], 3), 0.5);
        close(Wfg::shape_linear(&[0.2, 0.7, 0.9], 3), 0.8);
    }

    #[test]
    fn shape_convex_matches_toolkit() {
        // probe shape_convex m 3 x1 x2 x3
        close(Wfg::shape_convex(&[0.0, 0.0, 0.0], 1), 0.0);
        close(Wfg::shape_convex(&[1.0, 1.0, 1.0], 1), 0.999_999_999_999_999_8);
        close(Wfg::shape_convex(&[0.5, 0.5, 0.5], 1), 0.085_786_437_626_904_92);
        close(Wfg::shape_convex(&[0.2, 0.7, 0.9], 1), 0.02672360707868885);
        close(Wfg::shape_convex(&[0.0, 0.0, 0.0], 2), 0.0);
        close(Wfg::shape_convex(&[1.0, 1.0, 1.0], 2), 0.0);
        close(Wfg::shape_convex(&[0.5, 0.5, 0.5], 2), 0.085_786_437_626_904_95);
        close(Wfg::shape_convex(&[0.2, 0.7, 0.9], 2), 0.005_334_520_407_321_194);
        close(Wfg::shape_convex(&[0.0, 0.0, 0.0], 3), 1.0);
        close(Wfg::shape_convex(&[1.0, 1.0, 1.0], 3), 0.0);
        close(Wfg::shape_convex(&[0.5, 0.5, 0.5], 3), 0.29289321881345254);
        close(Wfg::shape_convex(&[0.2, 0.7, 0.9], 3), 0.690_983_005_625_052_5);
    }

    #[test]
    fn shape_concave_matches_toolkit() {
        // probe shape_concave m 3 x1 x2 x3
        close(Wfg::shape_concave(&[0.0, 0.0, 0.0], 1), 0.0);
        close(Wfg::shape_concave(&[1.0, 1.0, 1.0], 1), 1.0);
        close(Wfg::shape_concave(&[0.5, 0.5, 0.5], 1), 0.499_999_999_999_999_9);
        close(Wfg::shape_concave(&[0.2, 0.7, 0.9], 1), 0.275_336_158_073_158_3);
        close(Wfg::shape_concave(&[0.0, 0.0, 0.0], 2), 0.0);
        close(Wfg::shape_concave(&[1.0, 1.0, 1.0], 2), 6.123233995736766e-17);
        close(Wfg::shape_concave(&[0.5, 0.5, 0.5], 2), 0.5);
        close(Wfg::shape_concave(&[0.2, 0.7, 0.9], 2), 0.1402907797042951);
        close(Wfg::shape_concave(&[0.0, 0.0, 0.0], 3), 1.0);
        close(Wfg::shape_concave(&[1.0, 1.0, 1.0], 3), 6.123233995736766e-17);
        close(Wfg::shape_concave(&[0.5, 0.5, 0.5], 3), std::f64::consts::FRAC_1_SQRT_2);
        close(Wfg::shape_concave(&[0.2, 0.7, 0.9], 3), 0.951_056_516_295_153_5);
    }

    #[test]
    fn shape_mixed_matches_toolkit() {
        // probe shape_mixed 5 1.0 2 x1 x2 (x2 unused by mixed, held at 0.5)
        close(Wfg::shape_mixed(&[0.0, 0.5], 5, 1.0), 1.0);
        close(Wfg::shape_mixed(&[0.5, 0.5], 5, 1.0), 0.500_000_000_000_000_1);
        close(Wfg::shape_mixed(&[1.0, 0.5], 5, 1.0), 1.5612390095675744e-17);
        close(Wfg::shape_mixed(&[0.2, 0.5], 5, 1.0), 0.8);
        close(Wfg::shape_mixed(&[0.7, 0.5], 5, 1.0), 0.30000000000000016);
    }

    #[test]
    fn shape_disc_matches_toolkit() {
        // probe shape_disc 5 1.0 1.0 2 x1 x2 (x2 unused by disc, held at 0.5)
        close(Wfg::shape_disc(&[0.0, 0.5], 5, 1.0, 1.0), 1.0);
        close(Wfg::shape_disc(&[0.5, 0.5], 5, 1.0, 1.0), 1.0);
        close(Wfg::shape_disc(&[1.0, 0.5], 5, 1.0, 1.0), 0.0);
        close(Wfg::shape_disc(&[0.2, 0.5], 5, 1.0, 1.0), 0.8);
        close(Wfg::shape_disc(&[0.7, 0.5], 5, 1.0, 1.0), 1.0);
    }

    // ==== whole-problem evaluation fixtures vs the compiled toolkit,
    // ~1e-15 relative (module doc's provenance section; probe stdout,
    // `%.17g`), across (m,k,l) configs incl. m=2 and m=3, boundary (z=0,
    // z=2i) and a generic interior point ====

    fn wfg_close(got: &[f64], want: &[f64]) { wfg_close_tol(got, want, 1e-13) }

    /// `wfg_close` with an explicit relative tolerance -- used by WFG5's
    /// own engineered-z pareto-front test (module doc note below) where the
    /// default `1e-13` is too tight for an INTRINSIC (not a bug) reason.
    fn wfg_close_tol(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len());
        let mut max_rel = 0.0f64;
        for (g, w) in got.iter().zip(want.iter()) {
            let rel = if w.abs() > 1e-12 { (g - w).abs() / w.abs() } else { (g - w).abs() };
            max_rel = max_rel.max(rel);
        }
        assert!(max_rel < tol, "got={got:?} want={want:?} max_rel={max_rel} tol={tol}");
    }

    fn eval_z(which: u32, m: usize, k: usize, l: usize, z: &[f64]) -> Vec<f64> {
        let p = Wfg::new(which, m, k, l).unwrap();
        match which {
            1 => p.eval_wfg1(z),
            2 => p.eval_wfg2(z),
            3 => p.eval_wfg3(z),
            4 => p.eval_wfg4(z),
            5 => p.eval_wfg5(z),
            6 => p.eval_wfg6(z),
            7 => p.eval_wfg7(z),
            8 => p.eval_wfg8(z),
            9 => p.eval_wfg9(z),
            _ => unreachable!(),
        }
    }

    /// Deterministic "generic interior point" `z`, period-8 fraction
    /// pattern times each `zi`'s own `2i` bound -- COMPUTED here (not
    /// hand-transcribed as decimal literals) so the exact same `f64`
    /// arithmetic (hence bit pattern) reaches the probe-derived expected
    /// values below; a rounded decimal transcription of the scratchpad's
    /// own `gen_fixture.py` output was tried first and found to disagree at
    /// the ~1e-2 relative level for the `l=20` configs (traced to
    /// `b_poly`'s `alpha=0.02` exponent amplifying sub-ULP input
    /// differences -- a genuine fixture-transcription hazard, not an
    /// algorithm bug; this helper removes the hazard by construction).
    const GENERIC_PATTERN: [f64; 8] = [0.1, 0.9, 0.35, 0.6, 0.25, 0.75, 0.5, 0.15];

    fn generic_z(n: usize) -> Vec<f64> {
        (0..n)
            .map(|idx0| {
                let i = idx0 + 1;
                GENERIC_PATTERN[idx0 % GENERIC_PATTERN.len()] * 2.0 * i as f64
            })
            .collect()
    }

    #[test]
    fn wfg1_matches_toolkit_all_configs() {
        wfg_close(&eval_z(1, 2, 4, 4, &[0.0; 8]), &[1.0, 5.0]);
        wfg_close(
            &eval_z(1, 2, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[3.0, 1.0],
        );
        wfg_close(
            &eval_z(1, 2, 4, 4, &generic_z(8)),
            &[2.9352610723460653, 0.985_367_885_275_968_1],
        );
        wfg_close(
            &eval_z(1, 2, 4, 20, &generic_z(24)),
            &[2.8355803592139246, 0.885_687_172_143_827_2],
        );
        wfg_close(&eval_z(1, 3, 4, 4, &[0.0; 8]), &[1.0, 1.0, 7.0]);
        wfg_close(
            &eval_z(1, 3, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[2.9999999999999996, 1.0, 1.0],
        );
        wfg_close(
            &eval_z(1, 3, 4, 4, &generic_z(8)),
            &[2.8865123826729717, 0.984_114_757_636_518_8, 0.987_374_616_477_829_6],
        );
        wfg_close(
            &eval_z(1, 3, 6, 20, &generic_z(26)),
            &[2.795_813_160_599_584, 0.898_229_192_372_312_1, 0.903_509_148_664_455_1],
        );
    }

    #[test]
    fn wfg2_matches_toolkit_all_configs() {
        wfg_close(&eval_z(2, 2, 4, 4, &[0.0; 8]), &[0.666_666_666_666_666_6, 4.666666666666667]);
        wfg_close(
            &eval_z(2, 2, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[2.6666666666666665, 0.666_666_666_666_666_6],
        );
        wfg_close(
            &eval_z(2, 2, 4, 4, &generic_z(8)),
            &[1.0656188139201694, 4.433_108_551_524_51],
        );
        wfg_close(
            &eval_z(2, 2, 4, 20, &generic_z(24)),
            &[1.061_223_209_524_565, 4.428712947128906],
        );
        wfg_close(
            &eval_z(2, 3, 4, 4, &[0.0; 8]),
            &[0.666_666_666_666_666_6, 0.666_666_666_666_666_6, 6.666666666666667],
        );
        wfg_close(
            &eval_z(2, 3, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[2.666_666_666_666_666, 0.666_666_666_666_666_6, 0.666_666_666_666_666_6],
        );
        wfg_close(
            &eval_z(2, 3, 4, 4, &generic_z(8)),
            &[0.662_956_278_081_332_7, 0.883_634_341_411_358_8, 6.507_326_007_326_007],
        );
        wfg_close(
            &eval_z(2, 3, 6, 20, &generic_z(26)),
            &[0.670_270_277_485_516_9, 0.757_885_481_803_128_8, 5.161_721_611_721_611],
        );
    }

    #[test]
    fn wfg3_matches_toolkit_all_configs() {
        wfg_close(&eval_z(3, 2, 4, 4, &[0.0; 8]), &[0.666_666_666_666_666_6, 4.666666666666667]);
        wfg_close(
            &eval_z(3, 2, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[2.6666666666666665, 0.666_666_666_666_666_6],
        );
        wfg_close(
            &eval_z(3, 2, 4, 4, &generic_z(8)),
            &[1.4823260073260072, 2.5573260073260076],
        );
        wfg_close(
            &eval_z(3, 2, 4, 20, &generic_z(24)),
            &[1.4779304029304028, 2.552_930_402_930_403],
        );
        wfg_close(
            &eval_z(3, 3, 4, 4, &[0.0; 8]),
            &[0.666_666_666_666_666_6, 0.666_666_666_666_666_6, 6.666666666666667],
        );
        wfg_close(
            &eval_z(3, 3, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[2.333333333333333, 1.3333333333333335, 0.666_666_666_666_666_6],
        );
        wfg_close(
            &eval_z(3, 3, 4, 4, &generic_z(8)),
            &[0.994_642_857_142_857_2, 1.5326923076923076, 3.5073260073260073],
        );
        wfg_close(
            &eval_z(3, 3, 6, 20, &generic_z(26)),
            &[0.977_073_260_073_260_1, 1.381_018_315_018_315, 3.811_721_611_721_612],
        );
    }

    #[test]
    fn wfg4_matches_toolkit_all_configs() {
        wfg_close(&eval_z(4, 2, 4, 4, &[0.0; 8]), &[3.0, 1.0000000000000002]);
        wfg_close(
            &eval_z(4, 2, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[3.0, 1.0000000000000002],
        );
        wfg_close(&eval_z(4, 2, 4, 4, &generic_z(8)), &[1.275106902131418, 3.7043553279145627]);
        wfg_close(
            &eval_z(4, 2, 4, 20, &generic_z(24)),
            &[1.307201355011113, 3.7364497807942576],
        );
        wfg_close(
            &eval_z(4, 3, 4, 4, &[0.0; 8]),
            &[3.0, 1.0000000000000002, 1.0000000000000004],
        );
        wfg_close(
            &eval_z(4, 3, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[3.0, 1.0000000000000002, 1.0000000000000004],
        );
        wfg_close(
            &eval_z(4, 3, 4, 4, &generic_z(8)),
            &[0.5234027613452319, 3.336416632547231, 4.009046367876935],
        );
        wfg_close(
            &eval_z(4, 3, 6, 20, &generic_z(26)),
            &[0.7643216027529967, 2.386370750591389, 5.28199452671207],
        );
    }

    #[test]
    fn wfg5_matches_toolkit_all_configs() {
        wfg_close(&eval_z(5, 2, 4, 4, &[0.0; 8]), &[0.20691819145581142, 4.037669334932526]);
        wfg_close(
            &eval_z(5, 2, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[0.2069181914555187, 4.037669334932491],
        );
        wfg_close(&eval_z(5, 2, 4, 4, &generic_z(8)), &[1.472382748286957, 4.192843843592891]);
        wfg_close(
            &eval_z(5, 2, 4, 20, &generic_z(24)),
            &[1.3491260959984441, 4.069587191304379],
        );
        wfg_close(
            &eval_z(5, 3, 4, 4, &[0.0; 8]),
            &[0.06231165940490613, 0.36286893008067367, 6.0315040023987745],
        );
        wfg_close(
            &eval_z(5, 3, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[0.0623116594048005, 0.3628689300801632, 6.031504002398758],
        );
        wfg_close(
            &eval_z(5, 3, 4, 4, &generic_z(8)),
            &[0.9758919594206503, 1.9878648923615367, 6.105838909417616],
        );
        wfg_close(
            &eval_z(5, 3, 6, 20, &generic_z(26)),
            &[0.8729891665309172, 1.0807685300864873, 6.222185423789944],
        );
    }

    #[test]
    fn wfg6_matches_toolkit_all_configs() {
        wfg_close(&eval_z(6, 2, 4, 4, &[0.0; 8]), &[0.4, 4.4]);
        wfg_close(
            &eval_z(6, 2, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[1.5755705045849462, 3.6360679774997897],
        );
        wfg_close(&eval_z(6, 2, 4, 4, &generic_z(8)), &[2.2745281058919207, 2.132880708391472]);
        wfg_close(
            &eval_z(6, 2, 4, 20, &generic_z(24)),
            &[2.389441763662721, 2.2477943661622724],
        );
        wfg_close(&eval_z(6, 3, 4, 4, &[0.0; 8]), &[0.4, 0.4, 6.4]);
        wfg_close(
            &eval_z(6, 3, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[1.9, 2.1320508075688775, 3.400000000000001],
        );
        wfg_close(
            &eval_z(6, 3, 4, 4, &generic_z(8)),
            &[1.8048665115797227, 3.296334560799609, 1.7057119031483137],
        );
        wfg_close(
            &eval_z(6, 3, 6, 20, &generic_z(26)),
            &[2.1124215637699324, 2.7933383676134986, 2.8329207632065847],
        );
    }

    #[test]
    fn wfg7_matches_toolkit_all_configs() {
        wfg_close(&eval_z(7, 2, 4, 4, &[0.0; 8]), &[1.0, 5.0]);
        wfg_close(
            &eval_z(7, 2, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[3.0, 1.0000000000000002],
        );
        wfg_close(&eval_z(7, 2, 4, 4, &generic_z(8)), &[1.8704337648781033, 3.1921236066258687]);
        wfg_close(
            &eval_z(7, 2, 4, 20, &generic_z(24)),
            &[1.8920739277801475, 3.2215863954181456],
        );
        wfg_close(&eval_z(7, 3, 4, 4, &[0.0; 8]), &[1.0, 1.0, 7.0]);
        wfg_close(
            &eval_z(7, 3, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[3.0, 1.0000000000000002, 1.0000000000000004],
        );
        wfg_close(
            &eval_z(7, 3, 4, 4, &generic_z(8)),
            &[1.468987858956224, 2.376210579221865, 4.626689532627775],
        );
        wfg_close(
            &eval_z(7, 3, 6, 20, &generic_z(26)),
            &[1.523215178535634, 2.189788550131792, 4.9221296628998905],
        );
    }

    #[test]
    fn wfg8_matches_toolkit_all_configs() {
        wfg_close(&eval_z(8, 2, 4, 4, &[0.0; 8]), &[1.0, 5.0]);
        wfg_close(
            &eval_z(8, 2, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[3.0, 1.0000000000000002],
        );
        wfg_close(&eval_z(8, 2, 4, 4, &generic_z(8)), &[1.8150614671080652, 3.3123011288284694]);
        wfg_close(
            &eval_z(8, 2, 4, 20, &generic_z(24)),
            &[1.8348896485538708, 3.3321293102742753],
        );
        wfg_close(&eval_z(8, 3, 4, 4, &[0.0; 8]), &[1.0, 1.0, 7.0]);
        wfg_close(
            &eval_z(8, 3, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[3.0, 1.0000000000000002, 1.0000000000000004],
        );
        wfg_close(
            &eval_z(8, 3, 4, 4, &generic_z(8)),
            &[1.388855962498448, 2.5058644460163766, 4.671527429136079],
        );
        wfg_close(
            &eval_z(8, 3, 6, 20, &generic_z(26)),
            &[1.4435032614621215, 2.2164976139071957, 5.040671141996371],
        );
    }

    #[test]
    fn wfg9_matches_toolkit_all_configs() {
        wfg_close(&eval_z(9, 2, 4, 4, &[0.0; 8]), &[0.46282151815629363, 4.398026241462924]);
        wfg_close(
            &eval_z(9, 2, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[0.46282151815620454, 4.398026241462929],
        );
        wfg_close(&eval_z(9, 2, 4, 4, &generic_z(8)), &[2.2590142706731453, 2.240874474702232]);
        wfg_close(
            &eval_z(9, 2, 4, 20, &generic_z(24)),
            &[2.3180548204442535, 2.284362054788964],
        );
        wfg_close(
            &eval_z(9, 3, 4, 4, &[0.0; 8]),
            &[0.4054781046317331, 0.6090569265354293, 6.391777208527434],
        );
        wfg_close(
            &eval_z(9, 3, 4, 4, &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]),
            &[0.40547810463171763, 0.6090569265351345, 6.391777208527458],
        );
        wfg_close(
            &eval_z(9, 3, 4, 4, &generic_z(8)),
            &[1.2130772647553276, 1.0552152617621613, 5.9624118310784535],
        );
        wfg_close(
            &eval_z(9, 3, 6, 20, &generic_z(26)),
            &[2.0962690522820666, 2.8626099745147995, 2.8048817550270218],
        );
    }

    // ==== WFG3 Pareto-front closed form, cross-checked against the
    // compiled toolkit via an engineered z (module doc's `pareto_front`
    // decision section) ====

    fn pareto3_engineered_z(x1: f64, k: usize, l: usize) -> Vec<f64> {
        let n = k + l;
        (1..=n)
            .map(|i| {
                let frac = if i <= k { x1 } else { 0.35 };
                frac * 2.0 * i as f64
            })
            .collect()
    }

    #[test]
    fn wfg3_pareto_front_matches_toolkit_engineered_z() {
        // m=2: f = (2*x1, 4*(1-x1))
        for &x1 in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let z = pareto3_engineered_z(x1, 4, 4);
            let got = eval_z(3, 2, 4, 4, &z);
            let want = Wfg::wfg3_front_point(2, x1);
            wfg_close(&got, &want);
        }
        // m=3: f = (x1, 2*x1, 6*(1-x1))
        for &x1 in &[0.0, 0.3, 0.6, 1.0] {
            let z = pareto3_engineered_z(x1, 4, 4);
            let got = eval_z(3, 3, 4, 4, &z);
            let want = Wfg::wfg3_front_point(3, x1);
            wfg_close(&got, &want);
        }
    }

    #[test]
    fn wfg3_pareto_front_matches_toolkit_probe_values() {
        // probe wfg3 4 2 <engineered z>; f = (2*x1, 4*(1-x1))
        wfg_close(
            &eval_z(3, 2, 4, 4, &pareto3_engineered_z(0.25, 4, 4)),
            &[0.500_000_000_000_000_1, 3.0],
        );
        wfg_close(&eval_z(3, 2, 4, 4, &pareto3_engineered_z(0.5, 4, 4)), &[1.0, 2.0]);
        wfg_close(&eval_z(3, 2, 4, 4, &pareto3_engineered_z(0.75, 4, 4)), &[1.5, 1.0]);
        // probe wfg3 4 3 <engineered z>; f = (x1, 2*x1, 6*(1-x1))
        wfg_close(
            &eval_z(3, 3, 4, 4, &pareto3_engineered_z(0.3, 4, 4)),
            &[0.30000000000000004, 0.600_000_000_000_000_1, 4.199_999_999_999_999],
        );
        wfg_close(
            &eval_z(3, 3, 4, 4, &pareto3_engineered_z(0.6, 4, 4)),
            &[0.600_000_000_000_000_1, 1.2, 2.4000000000000004],
        );
    }

    // ==== WFG4-WFG9 Pareto-front closed form (the scaled unit hypersphere),
    // cross-checked against the compiled toolkit via an engineered z for
    // WFG4/WFG5/WFG6 (module doc's "toolkit-anchoring scope" decision) ====

    /// Engineers `z` so every position GROUP `g` shares one fraction
    /// `group_ps[g]` (so a group's post-transform values are uniform --
    /// `r_sum`/`r_nonsep` of a constant reduce to a closed, already
    /// toolkit-probed formula, module doc) and every distance parameter
    /// sits at the Pareto-optimal `2i*0.35` (module doc's Table 6 quote,
    /// "For WFG1-WFG7..."). Shared by WFG4/WFG5/WFG6's own engineered-z
    /// tests below (only the position TRANSFORM differs between them, not
    /// this z-construction scheme).
    fn wfg469_engineered_z(m: usize, k: usize, l: usize, group_ps: &[f64]) -> Vec<f64> {
        assert_eq!(group_ps.len(), m - 1);
        let n = k + l;
        (1..=n)
            .map(|i| {
                if i <= k {
                    let g = (i - 1) * (m - 1) / k;
                    group_ps[g] * 2.0 * i as f64
                } else {
                    0.35 * 2.0 * i as f64
                }
            })
            .collect()
    }

    #[test]
    fn wfg4_pareto_front_matches_toolkit_engineered_z() {
        for &p in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let z = wfg469_engineered_z(2, 4, 4, &[p]);
            let x1 = Wfg::s_multi(p, 30, 10.0, 0.35);
            let want = Wfg::wfg_concave_front_point(2, &[x1]);
            wfg_close(&eval_z(4, 2, 4, 4, &z), &want);
        }
        for &(p1, p2) in &[(0.0, 0.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1.0, 1.0)] {
            let z = wfg469_engineered_z(3, 4, 4, &[p1, p2]);
            let x1 = Wfg::s_multi(p1, 30, 10.0, 0.35);
            let x2 = Wfg::s_multi(p2, 30, 10.0, 0.35);
            let want = Wfg::wfg_concave_front_point(3, &[x1, x2]);
            wfg_close(&eval_z(4, 3, 4, 4, &z), &want);
        }
    }

    #[test]
    fn wfg4_pareto_front_matches_toolkit_probe_values() {
        // probe wfg4 4 2 <engineered z, p=0.25>
        wfg_close(
            &eval_z(4, 2, 4, 4, &wfg469_engineered_z(2, 4, 4, &[0.25])),
            &[0.5274550550545022, 3.8583888683736647],
        );
        // probe wfg4 4 3 <engineered z, (p1,p2)=(0.25,0.75)>
        wfg_close(
            &eval_z(4, 3, 4, 4, &wfg469_engineered_z(3, 4, 4, &[0.25, 0.75])),
            &[0.30252328595989214, 0.8641492846830975, 5.787583302560497],
        );
    }

    #[test]
    fn wfg5_pareto_front_matches_toolkit_engineered_z() {
        // `// sezgi decision:` looser tolerance here, not elsewhere, for a
        // documented reason specific to WFG5: `wfg469_engineered_z`'s
        // distance slots are `0.35*2i` normalized back by `/(2i)` inside
        // `eval_wfg5` -- for `i` where `2i` is not a power of two (e.g.
        // `i=6`), that round trip does not recover the exact same `f64` as
        // the literal `0.35` this test's own `want` uses, leaving a ~1 ULP
        // input difference at ONE distance slot. WFG5's `t1` applies
        // `s_decept(...,B=0.001,...)` to that slot directly (module doc's
        // Table 6 quote: "t1_i=1:n", both position AND distance) --
        // `s_decept`'s `1/B` term (already visible in its own quoted
        // formula) amplifies that 1-ULP difference by ~1000x into a ~5e-14
        // absolute deviation in `tp_M`, which -- since `correct_to_01` only
        // snaps NEGATIVE near-zero values to `0.0` (module doc's own
        // asymmetric-clamp comment on `correct_to_01`), not small POSITIVE
        // ones -- survives as a small uniform ADDITIVE offset on every `fm`
        // (`calculate_f`'s `fm = xM + Sm*hm`), large only in RELATIVE terms
        // for the smallest-magnitude `f` coordinate. Measured: ~1.13e-12
        // relative at worst, so `1e-11` here (vs. the module's usual
        // `1e-13`) is a comfortably-bounded, understood tolerance, not an
        // unexplained loosening -- an intrinsic property of WFG5's own
        // small-`B` bias transform, not an implementation bug (mirrors T5's
        // documented `b_poly` small-exponent sensitivity, module doc's
        // `generic_z` comment).
        for &p in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let z = wfg469_engineered_z(2, 4, 4, &[p]);
            let x1 = Wfg::s_decept(p, 0.35, 0.001, 0.05);
            let want = Wfg::wfg_concave_front_point(2, &[x1]);
            wfg_close_tol(&eval_z(5, 2, 4, 4, &z), &want, 1e-11);
        }
        for &(p1, p2) in &[(0.0, 0.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1.0, 1.0)] {
            let z = wfg469_engineered_z(3, 4, 4, &[p1, p2]);
            let x1 = Wfg::s_decept(p1, 0.35, 0.001, 0.05);
            let x2 = Wfg::s_decept(p2, 0.35, 0.001, 0.05);
            let want = Wfg::wfg_concave_front_point(3, &[x1, x2]);
            wfg_close_tol(&eval_z(5, 3, 4, 4, &z), &want, 1e-11);
        }
    }

    #[test]
    fn wfg5_pareto_front_matches_toolkit_probe_values() {
        // probe wfg5 4 2 <engineered z, p=0.25>
        wfg_close(
            &eval_z(5, 2, 4, 4, &wfg469_engineered_z(2, 4, 4, &[0.25])),
            &[1.823472734047781, 1.6431033907631893],
        );
        // probe wfg5 4 3 <engineered z, (p1,p2)=(0.25,0.75)>
        wfg_close(
            &eval_z(5, 3, 4, 4, &wfg469_engineered_z(3, 4, 4, &[0.25, 0.75])),
            &[1.1084251752694765, 2.8958221234354298, 2.464655086144777],
        );
    }

    #[test]
    fn wfg6_pareto_front_matches_toolkit_engineered_z() {
        for &p in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let z = wfg469_engineered_z(2, 4, 4, &[p]);
            let x1 = Wfg::r_nonsep(&[p; 4], 4);
            let want = Wfg::wfg_concave_front_point(2, &[x1]);
            wfg_close(&eval_z(6, 2, 4, 4, &z), &want);
        }
        for &(p1, p2) in &[(0.0, 0.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.25), (1.0, 1.0)] {
            let z = wfg469_engineered_z(3, 4, 4, &[p1, p2]);
            let x1 = Wfg::r_nonsep(&[p1; 2], 2);
            let x2 = Wfg::r_nonsep(&[p2; 2], 2);
            let want = Wfg::wfg_concave_front_point(3, &[x1, x2]);
            wfg_close(&eval_z(6, 3, 4, 4, &z), &want);
        }
    }

    #[test]
    fn wfg6_pareto_front_matches_toolkit_probe_values() {
        // probe wfg6 4 2 <engineered z, p=1.0>
        wfg_close(
            &eval_z(6, 2, 4, 4, &wfg469_engineered_z(2, 4, 4, &[1.0])),
            &[1.1755705045849463, 3.23606797749979],
        );
        // probe wfg6 4 3 <engineered z, (p1,p2)=(0.5,0.5)>
        wfg_close(
            &eval_z(6, 3, 4, 4, &wfg469_engineered_z(3, 4, 4, &[0.5, 0.5])),
            &[0.5, 1.7320508075688772, 5.196152422706632],
        );
    }

    #[test]
    fn wfg49_pareto_front_rows_on_scaled_unit_hypersphere() {
        for which in 4u32..=9 {
            for &m in &[2usize, 3] {
                let p = Wfg::new(which, m, 2 * (m - 1), 4).unwrap();
                let front = p.pareto_front(64).unwrap();
                assert!(!front.is_empty(), "which={which} m={m}");
                for row in &front {
                    assert_eq!(row.len(), m);
                    let sum_sq: f64 =
                        row.iter().enumerate().map(|(i0, &f)| (f / (2.0 * (i0 + 1) as f64)).powi(2)).sum();
                    assert!(
                        (sum_sq - 1.0).abs() < 1e-9,
                        "which={which} m={m} row={row:?} sum_sq={sum_sq}"
                    );
                }
            }
        }
    }

    #[test]
    fn wfg49_pareto_front_deterministic_and_exact_n_for_m2() {
        for which in 4u32..=9 {
            let p = Wfg::new(which, 2, 4, 4).unwrap();
            let a = p.pareto_front(30).unwrap();
            let b = p.pareto_front(30).unwrap();
            assert_eq!(a, b, "which={which}: pareto_front must be bit-identical across calls");
            assert_eq!(a.len(), 30, "which={which}: m=2 lattice is exact (r=n)");
        }
    }

    // ==== pareto_front API: None for WFG1/WFG2, Some for WFG3/WFG4-WFG9,
    // n=0 always Some(empty) ====

    #[test]
    fn pareto_front_none_for_wfg1_wfg2() {
        assert!(Wfg::new(1, 3, 6, 4).unwrap().pareto_front(50).is_none());
        assert!(Wfg::new(2, 3, 6, 4).unwrap().pareto_front(50).is_none());
    }

    #[test]
    fn pareto_front_zero_n_always_some_empty() {
        for which in 1u32..=9 {
            let p = Wfg::new(which, 3, 6, 4).unwrap();
            assert_eq!(p.pareto_front(0), Some(Vec::new()), "which={which}");
        }
    }

    #[test]
    fn wfg3_pareto_front_deterministic_and_shape() {
        let p = Wfg::new(3, 3, 6, 4).unwrap();
        let a = p.pareto_front(30).unwrap();
        let b = p.pareto_front(30).unwrap();
        assert_eq!(a, b, "pareto_front must be bit-identical across calls");
        assert_eq!(a.len(), 30);
        for row in &a {
            assert_eq!(row.len(), 3);
            // f2 == 2*f1 exactly (module doc's derived closed form).
            assert!((row[1] - 2.0 * row[0]).abs() < 1e-12, "{row:?}");
        }
    }

    // ==== degenerate/boundary inputs (x at 0 and at 2i) via evaluate_batch,
    // and wrong-genotype-shape defensive handling ====

    fn genotype_1d_blocks(z: &[f64]) -> Genotype {
        Genotype { blocks: z.iter().map(|&v| BlockValues::Float(vec![v])).collect() }
    }

    #[test]
    fn evaluate_batch_boundary_zero_and_max_all_which() {
        for which in 1u32..=9 {
            let p = Wfg::new(which, 2, 4, 4).unwrap();
            let z0 = genotype_1d_blocks(&[0.0; 8]);
            let zmax = genotype_1d_blocks(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
            let fs = p.evaluate_batch(std::slice::from_ref(&z0));
            assert_eq!(fs.len(), 1);
            assert_eq!(fs[0].len(), 2);
            assert!(fs[0].iter().all(|v| v.is_finite()), "which={which} z=0: {:?}", fs[0]);
            let fs = p.evaluate_batch(std::slice::from_ref(&zmax));
            assert_eq!(fs[0].len(), 2);
            assert!(fs[0].iter().all(|v| v.is_finite()), "which={which} z=2i: {:?}", fs[0]);
        }
    }

    #[test]
    fn evaluate_batch_matches_eval_helpers() {
        let p1 = Wfg::new(1, 3, 6, 4).unwrap();
        let z = pareto3_engineered_z(0.42, 6, 4);
        let g = genotype_1d_blocks(&z);
        let via_batch = p1.evaluate_batch(std::slice::from_ref(&g));
        let direct = p1.eval_wfg1(&z);
        assert_eq!(via_batch[0], direct);
    }

    #[test]
    fn evaluate_batch_wrong_block_count_returns_infinity_row() {
        let p = Wfg::new(1, 3, 6, 4).unwrap(); // n = 10
        let g = genotype_1d_blocks(&[0.0; 5]); // wrong length
        let fs = p.evaluate_batch(&[g]);
        assert_eq!(fs[0], vec![f64::INFINITY; 3]);
    }

    #[test]
    fn evaluate_batch_wrong_block_type_returns_infinity_row() {
        let p = Wfg::new(1, 2, 4, 4).unwrap(); // n = 8
        let mut blocks: Vec<BlockValues> = (0..7).map(|_| BlockValues::Float(vec![0.0])).collect();
        blocks.push(BlockValues::Int(vec![1]));
        let g = Genotype { blocks };
        let fs = p.evaluate_batch(&[g]);
        assert_eq!(fs[0], vec![f64::INFINITY; 2]);
    }

    #[test]
    fn n_objectives_matches_m() {
        for m in [2usize, 3, 4] {
            let p = Wfg::new(1, m, 2 * (m - 1), 4).unwrap();
            assert_eq!(p.n_objectives(), m);
        }
    }
}
