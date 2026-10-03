use sezgi_core::component::*;
use sezgi_core::dist::Distribution;
use sezgi_core::problem::Population;
use sezgi_core::space::{Block, BlockValues, Genotype};

/// Harris Hawks Optimization (Heidari, A.A., Mirjalili, S., Faris, H.,
/// Aljarah, I., Mafarja, M. & Chen, H. 2019, "Harris hawks optimization:
/// Algorithm and applications", *Future Generation Computer Systems* 97,
/// 849-872, DOI 10.1016/j.future.2019.02.028) -- a **labeled metaphor
/// preset**: faithful to the primary source's own reference MATLAB
/// implementation's update equations and loop structure, with a pinned
/// deterministic draw order and property tests, but NOT validated against
/// the paper's (or any other publication's) reported benchmark numbers. No
/// established equivalence critique covers HHO (the Camacho-Villalón/
/// Dorigo/Stützle *ITOR* six-algorithm critique covers GWO/MFO/WOA/FA/BA/
/// ALO, not HHO) -- cited here as primary-source only, per the task brief.
/// This is the wave's most structurally complex preset: a multi-branch
/// escape-energy tree whose deepest branches (progressive rapid dives)
/// EVALUATE mid-`generate()` -- see "In-generator evaluation" below, the
/// project's first generator to do so (as opposed to an *adapter*, which
/// `cs.rs`'s `AbandonWorstFraction` already established).
///
/// ## Provenance (PROVENANCE-FIRST protocol)
///
/// **Primary artifact used (governs) -- the paper AUTHOR's own code:**
/// `HHO.m`, Ali Asghar Heidari's own reference MATLAB implementation,
/// fetched byte-for-byte from his own official GitHub repository
/// `aliasgharheidaricom/Harris-Hawks-Optimization-Algorithm-and-Applications`
/// (raw file `HHO.m` at the `master` branch -- the file's own header cites
/// this exact paper/DOI and lists Heidari as "Author, inventor and
/// programmer"). This is a strictly stronger provenance tier than most of
/// this wave's other tasks (which had to fall back to third-party/mealpy
/// reimplementations) -- the author's own code, not a reimplementation.
///
/// **Secondary source consulted (does NOT govern; several confirmed
/// deltas):** mealpy's `OriginalHHO` (`mealpy/swarm_based/HHO.py`). It
/// independently confirms the overall shape (per-hawk `E0`/`E`, an
/// explore/exploit split on `|E|`, a further split on a second uniform
/// draw, and a dive sub-branch that evaluates `Y` then `Z = Y + rand*Levy`
/// with greedy accept-or-keep) -- but it has THREE material divergences
/// from `HHO.m` that the primary source overrules: (a) mealpy's hard/soft
/// besiege *labels and formulas are swapped* relative to `HHO.m` (mealpy's
/// `|E|>=0.5` branch is labeled "Hard besiege" and uses the jump-strength
/// term `J`; `HHO.m`'s `|E|>=0.5` branch, with `r>=0.5`, is the *Soft*
/// besiege, and its true Hard besiege (`r>=0.5 && |E|<0.5`) uses NO jump
/// term at all -- see finding 3 below); (b) mealpy applies a FINAL
/// population-wide greedy `greedy_selection_population` after every
/// branch, including exploration and besiege-without-dive, which `HHO.m`
/// updates UNCONDITIONALLY (see finding 10); (c) mealpy's dive `Z` step
/// scales the Lévy term by `uniform(lb, ub)` (the search-space box), not
/// `HHO.m`'s own `rand(1,dim)` (unit interval) -- see finding 6. sezgi
/// follows `HHO.m` on all three points; mealpy is noted here only to record
/// that it was checked and found to diverge, not as a governing source.
///
/// **Extracted equations / loop structure (from `HHO.m`, quoted verbatim
/// from the fetched source; `t`/`T` = iteration/max-iterations, `N` = pop
/// size):**
///
/// ```text
/// E1 = 2*(1-(t/T));
/// for i = 1:N
///     E0 = 2*rand()-1;                          % -1 < E0 < 1
///     Escaping_Energy = E1*E0;
///
///     if abs(Escaping_Energy) >= 1
///         % Exploration
///         q = rand();
///         rand_Hawk_index = floor(N*rand()+1);
///         X_rand = X(rand_Hawk_index, :);
///         if q < 0.5
///             X(i,:) = X_rand - rand()*abs(X_rand - 2*rand()*X(i,:));
///         elseif q >= 0.5
///             X(i,:) = (Rabbit_Location(1,:) - mean(X)) - rand()*((ub-lb)*rand+lb);
///         end
///     elseif abs(Escaping_Energy) < 1
///         % Exploitation
///         r = rand();
///         if r>=0.5 && abs(Escaping_Energy)<0.5             % Hard besiege
///             X(i,:) = (Rabbit_Location) - Escaping_Energy*abs(Rabbit_Location - X(i,:));
///         end
///         if r>=0.5 && abs(Escaping_Energy)>=0.5            % Soft besiege
///             Jump_strength = 2*(1-rand());
///             X(i,:) = (Rabbit_Location-X(i,:)) - Escaping_Energy*abs(Jump_strength*Rabbit_Location - X(i,:));
///         end
///         if r<0.5 && abs(Escaping_Energy)>=0.5             % Soft besiege + rapid dives
///             Jump_strength = 2*(1-rand());
///             X1 = Rabbit_Location - Escaping_Energy*abs(Jump_strength*Rabbit_Location - X(i,:));
///             if fobj(X1) < fobj(X(i,:))
///                 X(i,:) = X1;
///             else
///                 X2 = Rabbit_Location - Escaping_Energy*abs(Jump_strength*Rabbit_Location - X(i,:)) + rand(1,dim).*Levy(dim);
///                 if fobj(X2) < fobj(X(i,:))
///                     X(i,:) = X2;
///                 end
///             end
///         end
///         if r<0.5 && abs(Escaping_Energy)<0.5              % Hard besiege + rapid dives
///             Jump_strength = 2*(1-rand());
///             X1 = Rabbit_Location - Escaping_Energy*abs(Jump_strength*Rabbit_Location - mean(X));
///             if fobj(X1) < fobj(X(i,:))
///                 X(i,:) = X1;
///             else
///                 X2 = Rabbit_Location - Escaping_Energy*abs(Jump_strength*Rabbit_Location - mean(X)) + rand(1,dim).*Levy(dim);
///                 if fobj(X2) < fobj(X(i,:))
///                     X(i,:) = X2;
///                 end
///             end
///         end
///     end
/// end
/// ```
///
/// and the source's own `Levy(d)` helper:
///
/// ```text
/// function o=Levy(d)
/// beta=1.5;
/// sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
/// u=randn(1,d)*sigma; v=randn(1,d); step=u./abs(v).^(1/beta);
/// o=step;
/// end
/// ```
///
/// `HHO.m` also runs a first per-iteration loop, BEFORE the branch tree
/// above, that clips each `X(i,:)` into `[lb,ub]`, re-evaluates `fobj(X(i,:))`
/// fresh, and updates `Rabbit_Location`/`Rabbit_Energy` if improved. The
/// FITNESS RE-SCAN half of this is functionally identical to what this
/// crate's engine already does for every preset (boundary-repair the
/// previous generation's offspring, then evaluate it, populating
/// `pop.fitness` for the next `generate()` call), merely one iteration
/// earlier in sezgi's synchronous model vs `HHO.m`'s "re-evaluate at the top
/// of next iteration" timing. The `Rabbit_Location`/`Rabbit_Energy` UPDATE
/// itself is a RATCHET against a value persisted across the WHOLE run
/// (initialized once, before the loop, never reset) -- NOT reproduced
/// literally; see finding 7's `// sezgi simplification:` for the verified
/// semantics and the parked-convention rationale.
///
/// ## Verified findings (each checked against `HHO.m`; SOURCE GOVERNS on
/// every delta from the plan's sketch)
///
/// 1. **`E0`/`E` -- confirmed, matches the sketch:** `E0 = 2*rand()-1`,
///    ONE draw per hawk per generation; `E = E1*E0` with `E1 = 2*(1-t/T)`
///    computed ONCE per generation (before the hawk loop), mapped onto this
///    project's established `progress = ctx.eval.used()/ctx.eval.budget()`
///    convention (`mfo.rs`/`ssa.rs` precedent): `E1 = hho_e1(progress)`.
/// 2. **Exploration -- confirmed two sub-branches on `q`, but the EXACT
///    draw granularity is a genuine, load-bearing delta from a naive
///    per-dimension reading (the wave's repeated "WOA lesson"):** MATLAB's
///    bare `rand()` (no size arguments) is a SCALAR, broadcast via `.*`/
///    implicit expansion across the whole `dim`-length row -- so BOTH
///    exploration formulas use exactly TWO PER-HAWK SCALAR draws (not
///    per-dimension). `rand_Hawk_index = floor(N*rand()+1)` is drawn
///    UNCONDITIONALLY, before the `if q<0.5` test, even though it is only
///    USED by the `q<0.5` sub-branch -- the `q>=0.5` branch still consumes
///    (and discards) this draw. Self-selection (`rand_Hawk_index == i`) is
///    NOT excluded, the same idiom `woa.rs`'s random-whale index already
///    established. Pinned draw order per hawk: `q`, then `rand_idx`
///    (always), then EITHER `r_a` then `r_b` (family-perch, finding 2a) OR
///    `r_a` then `r_b` (tall-tree, finding 2b) -- see [`hho_explore_family_
///    dim_step`]/[`hho_explore_tree_dim_step`].
/// 3. **Exploitation's hard/soft besiege mapping -- a genuine, easy-to-get-
///    backwards delta, VERIFIED against `HHO.m` line-for-line (and
///    confirmed diverging from mealpy, see the provenance note above):**
///    `r>=0.5 && |E|<0.5` is **Hard** besiege and uses **NO** `Jump_strength`
///    term at all (zero draws): `X' = Rabbit - E·|Rabbit - X_i|`.
///    `r>=0.5 && |E|>=0.5` is **Soft** besiege and DOES draw one
///    `Jump_strength = 2*(1-rand())` (one per-hawk scalar): `X' = (Rabbit -
///    X_i) - E·|J·Rabbit - X_i|`. See [`hho_hard_besiege_dim_step`]/
///    [`hho_soft_besiege_dim_step`].
/// 4. **Rapid dives -- confirmed as the sketch's four-way branch's `r<0.5`
///    half, with an important asymmetry between the two dive sub-branches
///    AND a distinct easy-to-miss trap corrected during review (fix round
///    1): the `r<0.5 && |E|>=0.5` dive's `Y` formula (`X1`) is `Rabbit -
///    E·|J·Rabbit - X_i|` -- it does NOT carry finding 3's no-dive soft
///    besiege's leading `(Rabbit - X_i)` term at all (compare `HHO.m` line
///    98's no-dive `X(i,:)=(Rabbit_Location-X(i,:))-Escaping_Energy*abs(...)`
///    against line 105's dive `X1=Rabbit_Location-Escaping_Energy*abs(...)`
///    -- Eq. 4 vs Eq. 10 in the paper, genuinely different equations, NOT
///    the same formula reused). Its own formula is [`hho_soft_dive_y_
///    dim_step`], structurally IDENTICAL in SHAPE to the `r<0.5 && |E|<0.5`
///    dive's `Y` ([`hho_hard_dive_y_dim_step`]) -- both are `Rabbit -
///    E·|J·Rabbit - ref|`, differing only in whether `ref` is `X_i` (soft
///    dive) or `mean(X)` (hard dive, which ALSO -- unlike the hard-besiege-
///    NO-dive branch, finding 3 -- DOES draw a `Jump_strength`). Kept as two
///    separately-named pure functions (mirroring the source's own two
///    separate `if` blocks) rather than one parameterized helper, so each
///    branch's formula identity stays independently pinned and testable.
/// 5. **The accept-chain -- confirmed, exactly TWO extra evaluations
///    maximum per diving hawk:** evaluate `Y`; if strictly better than
///    `X_i`'s CURRENT fitness, accept `Y` and stop (one evaluation total,
///    zero further RNG draws); else compute `Z = Y + rand(1,dim).*Levy(dim)`
///    and evaluate it; if strictly better than `X_i`'s current fitness,
///    accept `Z`; else `X_i` is left COMPLETELY UNCHANGED (two evaluations
///    total). See "In-generator evaluation" below for how this maps onto
///    `ctx.eval.evaluate(..)` and the engine's budget semantics.
/// 6. **Lévy application -- a genuine, notable delta from `cs.rs`'s/
///    `fpa.rs`'s own Lévy usage, NOT a reuse of [`cs_dim_step`]
///    (`crate::cs::cs_dim_step`):** `HHO.m`'s own `Levy(d)` has **NO** `0.01`
///    scale factor (unlike Yang's `cs`/`fpa` demos' `L=0.01*step`) and the
///    result is added DIRECTLY to `Y` (`Y + rand(1,dim).*Levy(dim)`), not
///    multiplied into a difference term. The call-boundary pin (`cs.rs`'s
///    precedent: pin at `Distribution::Levy{alpha}.sample()`, not at the
///    data-dependent raw-draw count underneath it) still applies: ONE
///    `Distribution::Levy{alpha:1.5}.sample(ctx.rng)` call per dimension.
///    `rand(1,dim)` (the per-dimension scalar multiplier `S`, `S[d] =
///    ctx.rng.next_f64()`) and the Lévy sample are drawn INTERLEAVED per
///    dimension (`S[d]` then `levy[d]`, ascending `d`) -- MATLAB's
///    vectorized `rand(1,dim).*Levy(dim)` has no canonical per-dimension
///    draw order to preserve (both operands are whole-vector calls, and
///    this project's Gaussian/Lévy sampler is its own self-contained
///    Mantegna implementation using the project's own polar Box-Muller
///    Gaussian -- not MATLAB's `randn` -- so exact raw-draw
///    correspondence is not meaningful regardless of ordering choice);
///    this interleaved per-`(i,d)` order is the implementer's choice,
///    following every other per-dimension formula in this crate. See
///    [`hho_dive_z_dim_step`].
/// 7. **Rabbit (attractor) -- a genuine, deliberate delta from the verified
///    source, corrected during review (fix round 1): `HHO.m`'s
///    `Rabbit_Location`/`Rabbit_Energy` are initialized ONCE, OUTSIDE the
///    `while t<T` loop (`Rabbit_Location=zeros(1,dim); Rabbit_Energy=inf;`),
///    and then only RATCHET forward across the ENTIRE run: Loop 1 does
///    re-scan `fobj(X(i,:))` fresh every iteration, but only overwrites
///    `Rabbit_Location`/`Rabbit_Energy` when a STRICTLY BETTER fitness is
///    found (`if fitness<Rabbit_Energy`) -- it never regresses, and (since
///    exploration/besiege-without-dive branches overwrite unconditionally,
///    finding 10, with no per-agent greedy guard) the current population's
///    own best CAN be worse than the historical best, in which case
///    `Rabbit_Location` correctly stays at that older, better point. This
///    is a genuinely PERSISTED best-ever, the SAME shape as `ssa.rs`'s
///    `FoodPosition`/`ba.rs`'s `best` (both independently verified as real
///    persisted-best-ever mechanisms in their own sources) -- NOT the
///    "fresh per-iteration re-scan equals current-pop argmin" claim an
///    earlier version of this module doc made, which was a genuine
///    provenance error (Loop 1's fresh re-evaluation feeds a ratcheting
///    comparison, not a stateless recomputation).
// sezgi simplification: this module uses the current population's fitness
// argmin ([`Population::best_index`]) as the rabbit, computed once per
// `generate()` call before any RNG draws, INSTEAD OF `HHO.m`'s verified
// persisted-best-ever `Rabbit_Location`/`Rabbit_Energy`. This is the SAME
// wave-wide parked current-pop-attractor convention already applied to
// `ssa.rs`'s `FoodPosition` and `ba.rs`'s `best` (first parked during
// SCA's review) -- not an HHO-specific
// deviation. Blackboard-backed persistence (matching the source exactly)
// is a recorded family-wide enhancement candidate, not implemented here.
/// 8. **`mean(X)` -- the SAME in-place, sequentially-updating semantics
///    `ssa.rs`'s follower chain established (Task 4), for a different
///    construct (a whole-population mean, not a single-predecessor
///    chain):** used in TWO places (finding 2b's tall-tree exploration,
///    finding 4's hard-besiege-based dive `Y`). `HHO.m` computes `mean(X)`
///    from the LIVE `X` matrix, which by the time hawk `i` is processed may
///    already contain earlier hawks' (`i' < i`) THIS-GENERATION updated
///    positions, mixed with later hawks' (`i' >= i`) still-original ones --
///    recomputed FRESH (an O(n·dim) scan) every single time it is
///    referenced, never cached across the sweep or across hawks. See
///    [`hho_mean`] and the in-place hand-derived property test.
/// 9. **Exploration's `X_rand` lookup ALSO reads the live in-place array**
///    (same mechanism as finding 8) -- may return an already-this-
///    generation-updated hawk for `rand_idx < i`.
/// 10. **Replacement -- a genuine, notable delta from most of this wave
///     (which reuse `replace/one-to-one-greedy`):** exploration and
///     besiege-WITHOUT-dive branches overwrite `X(i,:)` UNCONDITIONALLY (no
///     fitness comparison in the source at all) -- `replace/generational`
///     is the correct pin, the same reuse `sca.rs`/`mfo.rs`/`ssa.rs` already
///     established for a source with no per-agent greedy compare. The dive
///     branches' accept/reject is resolved ENTIRELY INSIDE `generate()`
///     itself (see "In-generator evaluation" below), so their already-
///     decided final position is likewise installed unconditionally by the
///     SAME `replace/generational` -- no dive-specific replacer is needed
///     or correct.
///
/// `// sezgi simplification:` `HHO.m`'s own demo code literally RE-CALLS
/// `fobj(X(i,:))` (a fresh, redundant evaluation) inside EVERY dive-branch
/// comparison (once for `fobj(X1)<fobj(X(i,:))`, and again for
/// `fobj(X2)<fobj(X(i,:))` if reached) instead of reusing the `fitness`
/// value its own Loop 1 already computed this same iteration for the
/// UNCHANGED `X(i,:)` (Loop 2 has not yet written `X(i,:)` while a dive is
/// still undecided). sezgi does NOT reproduce this redundant re-evaluation:
/// it reuses the already-known `pop.fitness[i]` (zero additional evaluation
/// cost) for both comparisons. This is a deliberate, honest-accounting
/// choice, not an algorithmic behavior change -- `fobj` is a pure function
/// and `X(i,:)` is unchanged between the two hypothetical re-evaluations,
/// so the accept/reject DECISION is bit-for-bit identical either way; only
/// the source's own redundant budget spend is avoided. See "In-generator
/// evaluation" below for why this is the honest design, not a shortcut.
///
/// ## In-generator evaluation and the eval-accounting design decision
/// (HHO's genuinely new engine ground)
///
/// `gen/hho`'s two dive sub-branches (finding 4/5) call `ctx.eval.evaluate(..)`
/// **inside** `generate()` -- the project's first *generator* (as opposed to
/// `cs.rs`'s `AbandonWorstFraction` *adapter*) to do so. This raised a real
/// design question, spelled out here prominently per the task brief (for
/// the reviewer to adjudicate):
///
/// **The problem:** `Engine::run`'s per-stage loop ALWAYS re-evaluates
/// whatever `generate()` returns (`eval.evaluate(&offspring)`, right after
/// boundary-repair) -- there is no way to opt a stage out of this. So
/// whatever `generate()` decides internally for a diving hawk (via its own
/// `ctx.eval.evaluate` calls) gets evaluated AGAIN by the engine once it is
/// returned as part of `offspring`. Three alternatives were on the table:
/// (i) accept this "double-eval" for dived hawks as a tagged budget-
/// accounting simplification; (ii) restructure so ALL evaluation happens
/// internally and the generator's returned positions are re-evaluated by
/// the engine as the ONLY official count (dives' trial evals become "extra"
/// spend on top of a still-mandatory final engine eval); (iii) something
/// else.
///
/// **Decision: (i), and -- on close reading of `HHO.m` itself -- this is
/// NOT actually a simplification being accepted, it is the FAITHFUL
/// behavior.** `HHO.m` itself evaluates a dive-accepted (or dive-rejected-
/// but-unchanged) position TWICE across two points in its OWN control flow:
/// once during THIS iteration's dive-trial (`fobj(X1)`/`fobj(X2)`, deciding
/// acceptance), and once again during NEXT iteration's Loop 1
/// (`fitness=fobj(X(i,:))`, refreshing `Rabbit_Location` and setting up
/// iteration `t+1`'s exploitation branch) -- and this happens EVEN WHEN the
/// dive was rejected and `X(i,:)` is left completely unchanged: Loop 1 makes
/// no exception for "unchanged since last iteration", it re-evaluates every
/// hawk's position unconditionally, every iteration. This crate's engine
/// computes that "next iteration's Loop 1 re-evaluation" SYNCHRONOUSLY,
/// immediately after `generate()` returns (via the engine's own per-
/// generation stage-evaluate call -- the SAME mechanism every other preset
/// in this crate already relies on to populate `pop.fitness` for the next
/// `generate()` call), rather than `HHO.m`'s one-iteration-deferred timing.
/// So `gen/hho`'s design is: the internal `ctx.eval.evaluate` calls are pure
/// TRIAL evaluations, used ONLY to decide the accept/reject outcome (the
/// same decision `HHO.m`'s own `if fobj(X1)<fobj(X(i,:))` performs, with the
/// one documented delta above: reusing `pop.fitness[i]` instead of the
/// source's own redundant re-call) -- they consume 0, 1, or 2 REAL
/// evaluations per diving hawk, exactly mirroring the source's own dive-
/// trial cost. `generate()` then returns, for EVERY hawk, the FINAL decided
/// position (whatever `HHO.m`'s own branch would leave `X(i,:)` as at the
/// end of this iteration's Loop 2); `presets::hho` pairs this generator with
/// `replace/generational` (finding 10), so the engine's own subsequent
/// stage-evaluate call evaluates that SAME final position again -- this is
/// exactly `HHO.m`'s own genuine (if mildly wasteful) two-tier eval
/// structure, merely computed at an equivalent but different point in the
/// control flow, not a sezgi-introduced accounting artifact.
///
/// Alternative (ii) was REJECTED: it would require EITHER skipping the
/// engine's own per-generation stage-evaluate for non-diving hawks too
/// (breaking every other preset's established "the engine always evaluates
/// the generator's returned offspring" contract, and losing IOH/
/// `global_best` visibility parity with the rest of the crate), OR
/// inventing a new mechanism for `Generator` to suppress the engine's
/// subsequent evaluate call for JUST the dive-accepted hawks (no such
/// mechanism exists, and per the analysis above, none is needed: the
/// "double" eval IS what the source does).
///
/// **`best_f`/`best_x` and IOH visibility -- CURRENT INVARIANT (M3-1 Task 1
/// engine unification):** `RunResult::best_f`/`best_x` are sourced directly
/// from `Evaluator`'s own best-tracking (`Evaluator::best_so_far`/
/// `best_x_so_far`), updated inside `Evaluator::evaluate` itself -- the sole
/// gateway every charged evaluation in this crate passes through, including
/// `gen/hho`'s internal `ctx.eval.evaluate(..)` dive-trial calls on the RAW,
/// un-clamped `Y`/`Z` candidate. So `best_f`/`best_x` equal the
/// Evaluator-observed minimum for HHO exactly as for every other preset, and
/// match the IOH `.dat` archive's own best-so-far stream -- there is no
/// remaining divergence for archive consumers to guard against.
///
/// Prior to this unification, `RunResult::best_f` was tracked separately by
/// the engine (`global_best`, fed only from the engine's own call sites) and
/// could diverge from the Evaluator-observed minimum, via a boundary-repair
/// gap (the engine only ever re-evaluated a boundary-repaired copy of a
/// dive-accepted position, never the raw point `gen/hho` itself evaluated
/// and accepted on) and an independent budget-tail gap (a generation's dive
/// evals could be charged and IOH-logged, then the whole generation
/// discarded on budget exhaustion before reaching `global_best`) -- see git
/// history for the old caveat and its
/// measured gap (BBOB f3/seed-19: `best_f` understated by 5.4).
///
/// [`hho_captured_best_f_matches_evaluator_observed_minimum`] (in this
/// module's test suite) is the regression test for this invariant, on the
/// same BBOB f3/seed-19 fixture the M2d-4 review used to pin the old gap.
///
/// **Budget-exhaustion semantics (pinned):** if an internal
/// `ctx.eval.evaluate(..)` call returns `BudgetExhausted` -- whether on the
/// `Y` trial or the `Z` trial -- `generate()` falls back to the UN-DIVED
/// candidate (the hawk's own pre-dive position, i.e. `X(i,:)` as it stood at
/// the start of this hawk's own branch) WITHOUT attempting any further
/// internal evaluation for that hawk. This is correct in every case: if `Y`'s
/// own eval fails, nothing is known about `Y`, so keeping the un-dived
/// position is the only sound choice; if `Y`'s eval succeeded but was not an
/// improvement and `Z`'s eval then fails, the un-dived position is EXACTLY
/// what the source would have kept anyway (since `Y` was already rejected)
/// -- the only difference is `Z`'s own trial was never attempted, which
/// costs nothing (a batch of 1 exceeding the remaining budget consumes
/// NOTHING, per `Evaluator::evaluate`'s all-or-nothing contract). Once the
/// engine's budget is fully exhausted, EVERY subsequent hawk's dive attempt
/// (this generation or any later, including the doomed speculative
/// `generate()` call the engine's outer loop issues before its OWN
/// budget check can fail -- see `engine.rs`'s
/// `stage_rng_streams_use_the_documented_per_stage_indices` test comment for
/// this documented speculative-generate behavior) likewise falls back at
/// zero additional cost, so `evals_used` never overshoots `budget`. See the
/// budget-accounting test below, which independently re-derives an EXACT
/// expected `evals_used` for a crafted run where dives occur.
///
/// The per-dimension math is factored into pure helper functions (below) so
/// each branch is unit-tested directly without needing to fake `Ctx`/
/// `RngStream`; [`hho_run_dive`] factors the shared `Y`-then-`Z` accept-chain
/// (identical for both dive sub-branches, differing only in how `Y` itself
/// is computed upstream) so it is exercised once. [`HhoGenerator::generate`]
/// builds the offspring SEQUENTIALLY in ascending index order (required for
/// findings 8/9's in-place semantics), maintaining its own mutable working
/// copy of the population's positions.
///
/// ## min_pop
///
/// `min_pop = 2`, per the task brief. `HHO.m`'s own math is well-defined
/// even at `n = 1` (rabbit = self, `rand_idx` = self, `mean(X)` = self,
/// none of which divide by zero or loop indefinitely) -- unlike several
/// other presets in this wave, this is NOT a hard mathematical requirement.
/// `min_pop = 2` is set for consistency with the wave's established
/// "needs a best-so-far distinct from `i`" convention and because a
/// population of exactly 1 makes "perch on another family member"/"rabbit"
/// trivially self-referential and not meaningful as a demonstration.
///
/// Boundary handling is NOT part of this generator: `presets::hho` reuses
/// `boundary/clamp` (same as every other preset in this crate), applied
/// AFTER `generate()` returns -- matching `HHO.m`'s own Loop-1-only
/// clipping (finding 7's note): the internal dive-trial evaluations
/// (finding 5) run on the RAW, un-clamped `Y`/`Z` candidates, exactly as
/// `HHO.m`'s own `fobj(X1)`/`fobj(X2)` calls do (no clip inside its Loop 2
/// either).
pub fn hho_e1(progress: f64) -> f64 {
    2.0 * (1.0 - progress)
}

/// Exploration, `q < 0.5` ("perch based on other family members"), per
/// dimension: `X_rand_d - r_a·|X_rand_d - 2·r_b·X_i_d|`. `r_a`/`r_b` are
/// PER-HAWK scalars (finding 2), reused across every dimension by the
/// caller -- not redrawn per `d`.
pub fn hho_explore_family_dim_step(x_rand_d: f64, r_a: f64, r_b: f64, x_i_d: f64) -> f64 {
    x_rand_d - r_a * (x_rand_d - 2.0 * r_b * x_i_d).abs()
}

/// Exploration, `q >= 0.5` ("perch on a random tall tree"), per dimension:
/// `(rabbit_d - mean_d) - r_a·((hi-lo)·r_b + lo)`. `r_a`/`r_b` are PER-HAWK
/// scalars (finding 2), `mean_d` is [`hho_mean`]'s live in-place value
/// (finding 8).
pub fn hho_explore_tree_dim_step(rabbit_d: f64, mean_d: f64, r_a: f64, r_b: f64, lo: f64, hi: f64) -> f64 {
    (rabbit_d - mean_d) - r_a * ((hi - lo) * r_b + lo)
}

/// Hard besiege, NO dive (`r>=0.5 && |E|<0.5`), per dimension: `rabbit_d -
/// E·|rabbit_d - X_i_d|`. Zero RNG draws (finding 3).
pub fn hho_hard_besiege_dim_step(rabbit_d: f64, e: f64, x_i_d: f64) -> f64 {
    rabbit_d - e * (rabbit_d - x_i_d).abs()
}

/// Soft besiege (`r>=0.5 && |E|>=0.5`, NO dive), per dimension: `(rabbit_d -
/// X_i_d) - E·|J·rabbit_d - X_i_d|`. NOT reused by the dive branch (fix
/// round 1: an earlier version of this module incorrectly reused this for
/// the soft-dive `Y` too -- `HHO.m`'s dive `X1` has no leading `(rabbit_d -
/// X_i_d)` term at all, a genuinely different equation; see
/// [`hho_soft_dive_y_dim_step`] and finding 4).
pub fn hho_soft_besiege_dim_step(rabbit_d: f64, e: f64, jump: f64, x_i_d: f64) -> f64 {
    (rabbit_d - x_i_d) - e * (jump * rabbit_d - x_i_d).abs()
}

/// Soft-besiege-based dive `Y` (`r<0.5 && |E|>=0.5`), per dimension:
/// `rabbit_d - E·|J·rabbit_d - X_i_d|` -- a DIFFERENT formula from
/// [`hho_soft_besiege_dim_step`] (no leading `(rabbit_d - X_i_d)` term;
/// `HHO.m` line 105's `X1`, not line 98's no-dive `X(i,:)` -- finding 4,
/// corrected during review). Structurally identical in shape to
/// [`hho_hard_dive_y_dim_step`], differing only in which reference point
/// (`X_i` here, `mean(X)` there) is used.
pub fn hho_soft_dive_y_dim_step(rabbit_d: f64, e: f64, jump: f64, x_i_d: f64) -> f64 {
    rabbit_d - e * (jump * rabbit_d - x_i_d).abs()
}

/// Hard-besiege-based dive `Y` (`r<0.5 && |E|<0.5`), per dimension:
/// `rabbit_d - E·|J·rabbit_d - mean_d|` -- a DIFFERENT formula from
/// [`hho_hard_besiege_dim_step`] (uses `mean(X)` instead of `X_i` AND draws
/// a `Jump_strength`, finding 4).
pub fn hho_hard_dive_y_dim_step(rabbit_d: f64, e: f64, jump: f64, mean_d: f64) -> f64 {
    rabbit_d - e * (jump * rabbit_d - mean_d).abs()
}

/// Dive's `Z` step, per dimension: `Y_d + S_d·levy_d` -- NO `0.01` scale
/// factor (finding 6, a genuine delta from `cs_dim_step`/`fpa`'s Lévy use).
pub fn hho_dive_z_dim_step(y_d: f64, s_d: f64, levy_d: f64) -> f64 {
    y_d + s_d * levy_d
}

/// Live, in-place population mean (finding 8): componentwise mean over
/// EVERY row of `work` as it currently stands (may mix already-this-
/// generation-updated rows with still-original ones). Recomputed fresh by
/// the caller every time it is needed -- never cached.
pub fn hho_mean(work: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let n = work.len() as f64;
    let mut mean = vec![0.0; dim];
    for row in work {
        for (m, &v) in mean.iter_mut().zip(row) { *m += v; }
    }
    for m in &mut mean { *m /= n; }
    mean
}

const LEVY_ALPHA: f64 = 1.5;

/// The shared rapid-dive accept-chain (finding 5): evaluate `y`; if
/// strictly better than `current_fitness`, accept it (`Some(y)`, no further
/// draws/evals). Else compute `z = y + S.*Levy` (per-dimension draws,
/// finding 6) and evaluate it; accept if strictly better (`Some(z)`), else
/// `None` -- meaning "leave the un-dived candidate unchanged" (finding 5's
/// "neither improved" outcome). `None` is ALSO returned, per the pinned
/// exhaustion semantics documented in the module doc, if either internal
/// evaluate call hits `BudgetExhausted` -- with no further evaluation
/// attempted beyond the one that failed.
pub fn hho_run_dive(y: Vec<f64>, dim: usize, current_fitness: f64, ctx: &mut Ctx) -> Option<Vec<f64>> {
    let y_g = Genotype { blocks: vec![BlockValues::Float(y.clone())] };
    let fy = match ctx.eval.evaluate(std::slice::from_ref(&y_g)) {
        Ok(f) => f[0],
        Err(_) => return None, // budget exhausted mid-dive: fall back to the un-dived candidate
    };
    if fy < current_fitness {
        return Some(y);
    }

    let z: Vec<f64> = (0..dim).map(|d| {
        let s_d = ctx.rng.next_f64();
        let levy_d = Distribution::Levy { alpha: LEVY_ALPHA }.sample(ctx.rng);
        hho_dive_z_dim_step(y[d], s_d, levy_d)
    }).collect();
    let z_g = Genotype { blocks: vec![BlockValues::Float(z.clone())] };
    let fz = match ctx.eval.evaluate(std::slice::from_ref(&z_g)) {
        Ok(f) => f[0],
        Err(_) => return None, // budget exhausted mid-dive: fall back to the un-dived candidate
    };
    if fz < current_fitness { Some(z) } else { None }
}

fn floats(g: &Genotype) -> &Vec<f64> {
    match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
}

pub struct HhoGenerator;

impl HhoGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        Ok(Self) // no tunable parameters -- every constant is fixed, per the pinned spec
    }
}

impl Generator for HhoGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2, "gen/hho requires a population of at least 2 (pop_size={})", pop.len());
        let n = pop.len();
        let dim = floats(&pop.individuals[0]).len();
        let (lo, hi) = match &ctx.space.blocks()[0] {
            Block::Float { lo, hi, .. } => (*lo, *hi),
            _ => unreachable!("gen/hho only supports float blocks"),
        };

        let used = ctx.eval.used() as f64;
        let budget = ctx.eval.budget() as f64;
        let progress = if budget > 0.0 { (used / budget).clamp(0.0, 1.0) } else { 0.0 };
        let e1 = hho_e1(progress);

        // Rabbit: current-population fitness argmin, ties -> lower index --
        // pinned (finding 7), computed ONCE per generate() call, before any
        // RNG draws, held fixed for the whole sweep.
        let rabbit_idx = pop.best_index().unwrap_or(0);
        let rabbit = floats(&pop.individuals[rabbit_idx]).clone();

        // Live, in-place working copy (findings 8/9): updated sequentially
        // as each hawk is processed, in ascending index order, so later
        // hawks' mean(X)/random-hawk-lookup can observe earlier hawks'
        // already-updated-this-generation positions.
        let mut work: Vec<Vec<f64>> = pop.individuals.iter().map(|g| floats(g).clone()).collect();

        for i in 0..n {
            let e0 = 2.0 * ctx.rng.next_f64() - 1.0;
            let e = e1 * e0;

            let new: Vec<f64> = if e.abs() >= 1.0 {
                // Exploration (finding 2).
                let q = ctx.rng.next_f64();
                // ALWAYS drawn, even when unused by the q>=0.5 sub-branch.
                let rand_idx = ctx.rng.next_below(n as u64) as usize;
                if q < 0.5 {
                    let x_rand = work[rand_idx].clone();
                    let r_a = ctx.rng.next_f64();
                    let r_b = ctx.rng.next_f64();
                    (0..dim).map(|d| hho_explore_family_dim_step(x_rand[d], r_a, r_b, work[i][d])).collect()
                } else {
                    let mean = hho_mean(&work, dim);
                    let r_a = ctx.rng.next_f64();
                    let r_b = ctx.rng.next_f64();
                    (0..dim).map(|d| hho_explore_tree_dim_step(rabbit[d], mean[d], r_a, r_b, lo, hi)).collect()
                }
            } else {
                // Exploitation (findings 3/4).
                let r = ctx.rng.next_f64();
                if r >= 0.5 && e.abs() < 0.5 {
                    // Hard besiege, no dive: zero draws.
                    (0..dim).map(|d| hho_hard_besiege_dim_step(rabbit[d], e, work[i][d])).collect()
                } else if r >= 0.5 {
                    // Soft besiege, no dive (|e| >= 0.5 implied).
                    let jump = 2.0 * (1.0 - ctx.rng.next_f64());
                    (0..dim).map(|d| hho_soft_besiege_dim_step(rabbit[d], e, jump, work[i][d])).collect()
                } else if e.abs() >= 0.5 {
                    // Soft-besiege-based rapid dive.
                    let jump = 2.0 * (1.0 - ctx.rng.next_f64());
                    let y: Vec<f64> = (0..dim)
                        .map(|d| hho_soft_dive_y_dim_step(rabbit[d], e, jump, work[i][d]))
                        .collect();
                    match hho_run_dive(y, dim, pop.fitness[i], ctx) {
                        Some(final_pos) => final_pos,
                        None => work[i].clone(), // fallback: un-dived candidate, unchanged
                    }
                } else {
                    // Hard-besiege-based rapid dive.
                    let jump = 2.0 * (1.0 - ctx.rng.next_f64());
                    let mean = hho_mean(&work, dim);
                    let y: Vec<f64> = (0..dim)
                        .map(|d| hho_hard_dive_y_dim_step(rabbit[d], e, jump, mean[d]))
                        .collect();
                    match hho_run_dive(y, dim, pop.fitness[i], ctx) {
                        Some(final_pos) => final_pos,
                        None => work[i].clone(), // fallback: un-dived candidate, unchanged
                    }
                }
            };
            work[i] = new;
        }

        work.into_iter().map(|xs| Genotype { blocks: vec![BlockValues::Float(xs)] }).collect()
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/hho", SupportedBlocks::Only(vec!["float"]))
            .with_min_pop(2)
            .with_internal_eval()
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/hho", |p| Ok(Box::new(HhoGenerator::from_params(p)?)));
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hho_meta_reports_internal_eval() {
        let m = HhoGenerator::from_params(&serde_json::json!({})).unwrap().meta();
        assert_eq!(m.kind, "gen/hho");
        assert!(m.internal_eval); assert_eq!(m.offspring, OffspringCount::PopLen);
    }
    use sezgi_core::engine::{Engine, RunConfig};
    use sezgi_core::problem::{Problem, SphereShifted, Evaluator};
    use sezgi_core::rng::RngStream;
    use sezgi_core::space::SearchSpace;
    use sezgi_core::state::Blackboard;

    fn g(xs: Vec<f64>) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs)] } }

    fn pop_nd(n: usize, dim: usize) -> Population {
        // Distinct fitness so best-index (rabbit) selection is unambiguous.
        Population {
            individuals: (0..n).map(|i| g((0..dim).map(|d| (i * dim + d) as f64 * 0.37 - 1.0).collect())).collect(),
            fitness: (0..n).map(|i| (n - i) as f64).collect(), // higher index = better (lower fitness)
        }
    }

    // A test-only Problem whose evaluate_batch returns pre-scripted values in
    // order, regardless of the genotype passed in -- used to give full,
    // deterministic control over the accept/reject outcome of `hho_run_dive`
    // and full-generator dive-path tests, without needing to predict what an
    // arbitrary RNG-derived Y/Z position's REAL fitness would be.
    struct ScriptedProblem {
        space: SearchSpace,
        values: std::sync::Mutex<std::collections::VecDeque<f64>>,
    }
    impl ScriptedProblem {
        fn new(dim: usize, lo: f64, hi: f64, values: Vec<f64>) -> Self {
            let space = SearchSpace::new(vec![sezgi_core::space::Block::Float { lo, hi, n: dim }]).unwrap();
            Self { space, values: std::sync::Mutex::new(values.into()) }
        }
    }
    impl Problem for ScriptedProblem {
        fn space(&self) -> &SearchSpace { &self.space }
        fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
            pop.iter().map(|_| self.values.lock().unwrap().pop_front().expect("scripted values exhausted")).collect()
        }
    }

    // ---- hho_e1 ----

    #[test]
    fn e1_schedule_endpoints() {
        assert_eq!(hho_e1(0.0), 2.0);
        assert_eq!(hho_e1(1.0), 0.0);
        assert_eq!(hho_e1(0.5), 1.0);
    }

    // ---- per-dimension step helpers: hand-computed values ----

    #[test]
    fn explore_family_dim_step_matches_hand_computed_value() {
        // x_rand=4.0, r_a=0.5, r_b=0.25, x_i=2.0:
        // 4.0 - 0.5*|4.0 - 2*0.25*2.0| = 4.0 - 0.5*|4.0-1.0| = 4.0 - 1.5 = 2.5
        assert_eq!(hho_explore_family_dim_step(4.0, 0.5, 0.25, 2.0), 2.5);
    }

    #[test]
    fn explore_family_dim_step_zero_when_x_rand_equals_x_i_and_r_b_half() {
        // x_rand == x_i == v, r_b=0.5: |v - 2*0.5*v| = |v-v| = 0 -> result = v exactly.
        for v in [3.0, -7.5, 0.0] {
            assert_eq!(hho_explore_family_dim_step(v, 0.9, 0.5, v), v);
        }
    }

    #[test]
    fn explore_tree_dim_step_matches_hand_computed_value() {
        // rabbit=10.0, mean=4.0, r_a=0.5, r_b=0.5, lo=-5.0, hi=5.0:
        // (10.0-4.0) - 0.5*((5.0-(-5.0))*0.5 + (-5.0)) = 6.0 - 0.5*(5.0-5.0) = 6.0
        assert_eq!(hho_explore_tree_dim_step(10.0, 4.0, 0.5, 0.5, -5.0, 5.0), 6.0);
    }

    #[test]
    fn hard_besiege_dim_step_matches_hand_computed_value_and_zero_e_limit() {
        // rabbit=5.0, e=0.5, x_i=1.0: 5.0 - 0.5*|5.0-1.0| = 5.0-2.0 = 3.0
        assert_eq!(hho_hard_besiege_dim_step(5.0, 0.5, 1.0), 3.0);
        // e=0 limit: result == rabbit_d exactly, independent of x_i_d.
        assert_eq!(hho_hard_besiege_dim_step(5.0, 0.0, 999.0), 5.0);
    }

    #[test]
    fn soft_besiege_dim_step_matches_hand_computed_value_and_zero_e_limit() {
        // rabbit=5.0, e=0.5, jump=2.0, x_i=1.0: (5.0-1.0) - 0.5*|2.0*5.0-1.0| = 4.0-0.5*9.0 = -0.5
        assert_eq!(hho_soft_besiege_dim_step(5.0, 0.5, 2.0, 1.0), -0.5);
        // e=0 limit: result == rabbit_d - x_i_d exactly, independent of jump.
        assert_eq!(hho_soft_besiege_dim_step(5.0, 0.0, 999.0, 1.0), 4.0);
    }

    #[test]
    fn hard_dive_y_dim_step_matches_hand_computed_value() {
        // rabbit=5.0, e=0.5, jump=2.0, mean=1.0: 5.0 - 0.5*|2.0*5.0-1.0| = 5.0-4.5 = 0.5
        assert_eq!(hho_hard_dive_y_dim_step(5.0, 0.5, 2.0, 1.0), 0.5);
    }

    #[test]
    fn soft_dive_y_dim_step_matches_hand_computed_value() {
        // Fix round 1: rabbit=5.0, e=0.5, jump=2.0, x_i=1.0: 5.0 - 0.5*|2.0*5.0-1.0| = 5.0-4.5 = 0.5
        // (same shape/arithmetic as hho_hard_dive_y_dim_step's fixture, with
        // x_i in place of mean -- see finding 4).
        assert_eq!(hho_soft_dive_y_dim_step(5.0, 0.5, 2.0, 1.0), 0.5);
    }

    #[test]
    fn soft_dive_y_dim_step_differs_from_soft_besiege_no_dive_dim_step() {
        // Fix round 1 regression guard: HHO.m's dive X1 (line 105) has NO
        // leading (rabbit-x_i) term that the no-dive soft besiege (line 98)
        // has -- so the two formulas must NOT agree in general, even though
        // both take the same (rabbit, e, jump, x_i) shape.
        let (rabbit, e, jump, x_i) = (5.0, 0.5, 2.0, 1.0);
        let dive_y = hho_soft_dive_y_dim_step(rabbit, e, jump, x_i);
        let no_dive = hho_soft_besiege_dim_step(rabbit, e, jump, x_i);
        assert_ne!(dive_y, no_dive,
            "the dive Y formula must differ from the no-dive soft besiege formula (Eq. 10 vs Eq. 4)");
        // no_dive = (rabbit - x_i) - e*T and dive_y = rabbit - e*T (same T),
        // so no_dive == dive_y - x_i exactly -- the no-dive formula's extra
        // leading term subtracts x_i relative to the dive Y formula.
        assert_eq!(no_dive, dive_y - x_i);
    }

    #[test]
    fn dive_z_dim_step_matches_hand_computed_value_and_has_no_scale_factor() {
        // y=2.0, s=3.0, levy=0.5: 2.0 + 3.0*0.5 = 3.5 -- NO 0.01 factor (finding 6).
        assert_eq!(hho_dive_z_dim_step(2.0, 3.0, 0.5), 3.5);
        assert_eq!(hho_dive_z_dim_step(2.0, 0.0, 999.0), 2.0, "s=0 leaves y unchanged regardless of levy");
    }

    // ---- hho_mean ----

    #[test]
    fn mean_computes_columnwise_average_over_all_rows() {
        let work = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 0.0]];
        assert_eq!(hho_mean(&work, 2), vec![3.0, 2.0]);
    }

    #[test]
    fn mean_single_row_equals_that_row() {
        let work = vec![vec![7.0, -3.0]];
        assert_eq!(hho_mean(&work, 2), vec![7.0, -3.0]);
    }

    // ---- hho_run_dive: the accept-chain and budget-exhaustion semantics ----

    #[test]
    fn run_dive_accepts_y_immediately_when_it_improves() {
        let dim = 2;
        let p = ScriptedProblem::new(dim, -5.0, 5.0, vec![3.0]); // fy = 3.0
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let y = vec![1.0, 2.0];
        let result = hho_run_dive(y.clone(), dim, 5.0, &mut ctx); // current_fitness = 5.0 > fy
        assert_eq!(result, Some(y), "Y must be accepted when it strictly improves");
        assert_eq!(ctx.eval.used(), 1, "accepting Y must consume exactly one evaluation, no Z trial");
    }

    #[test]
    fn run_dive_rejects_y_accepts_z_when_z_improves() {
        let dim = 2;
        let p = ScriptedProblem::new(dim, -5.0, 5.0, vec![10.0, 3.0]); // fy=10.0 (rejected), fz=3.0 (accepted)
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let y = vec![1.0, 2.0];
        let result = hho_run_dive(y.clone(), dim, 5.0, &mut ctx); // current_fitness = 5.0
        assert!(result.is_some(), "Z must be accepted when it strictly improves and Y did not");
        assert_ne!(result.as_ref().unwrap(), &y, "the accepted Z must differ from Y (S/Levy step is nonzero with overwhelming probability)");
        assert_eq!(ctx.eval.used(), 2, "rejecting Y then accepting Z must consume exactly two evaluations");
    }

    #[test]
    fn run_dive_rejects_both_leaves_none_meaning_unchanged() {
        let dim = 2;
        let p = ScriptedProblem::new(dim, -5.0, 5.0, vec![10.0, 20.0]); // neither improves current=5.0
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let y = vec![1.0, 2.0];
        let result = hho_run_dive(y, dim, 5.0, &mut ctx);
        assert_eq!(result, None, "neither Y nor Z improving must return None (caller keeps the un-dived candidate)");
        assert_eq!(ctx.eval.used(), 2, "both trials must still be attempted (and counted) before giving up");
    }

    #[test]
    fn run_dive_budget_exhausted_on_y_falls_back_without_spending() {
        let dim = 2;
        let p = ScriptedProblem::new(dim, -5.0, 5.0, vec![3.0]);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 0); // zero budget: Y's own evaluate must fail immediately
        let mut rng = RngStream::from_master(1, &[]);
        let rng_before = rng.clone();
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let y = vec![1.0, 2.0];
        let result = hho_run_dive(y, dim, 5.0, &mut ctx);
        assert_eq!(result, None, "budget exhaustion on Y's own trial must fall back to the un-dived candidate");
        assert_eq!(ctx.eval.used(), 0, "no evaluation may be charged on exhaustion");
        assert_eq!(ctx.rng.next_f64(), { let mut t = rng_before; t.next_f64() },
            "no RNG draws may be consumed either -- Z's S/Levy draws must never be attempted once Y's own eval fails");
    }

    #[test]
    fn run_dive_budget_exhausted_on_z_falls_back_without_further_spending() {
        let dim = 2;
        let p = ScriptedProblem::new(dim, -5.0, 5.0, vec![10.0]); // Y's own trial succeeds and is rejected
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1); // exactly enough budget for Y's trial only
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let y = vec![1.0, 2.0];
        let result = hho_run_dive(y, dim, 5.0, &mut ctx); // current=5.0, fy=10.0 not better
        assert_eq!(result, None, "budget exhaustion on Z's trial must fall back to the un-dived candidate");
        assert_eq!(ctx.eval.used(), 1, "only Y's own (successful) trial may be charged");
    }

    // ---- HhoGenerator: branch-tree twin-stream tests (hand-traced against
    // seeds discovered via a standalone RNG-replay search, at progress=0 so
    // e1=2.0 -- see the module's provenance-verified branch conditions) ----

    #[test]
    fn twin_stream_exploration_family_branch() {
        // seed=7, n=2, dim=2, progress=0: hawk 0 lands in exploration
        // (|E|>=1), q<0.5 (family-perch) sub-branch.
        let dim = 2; let n = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(7, &[]);
        let rng_before = rng.clone();
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = HhoGenerator.generate(&pop, &mut ctx);

        let mut twin = rng_before;
        let e0 = 2.0 * twin.next_f64() - 1.0;
        let e = hho_e1(0.0) * e0;
        assert!(e.abs() >= 1.0, "fixture must land in exploration");
        let q = twin.next_f64();
        assert!(q < 0.5, "fixture must land in the family-perch sub-branch");
        let rand_idx = twin.next_below(n as u64) as usize;
        let x_rand = floats(&pop.individuals[rand_idx]).clone();
        let r_a = twin.next_f64();
        let r_b = twin.next_f64();
        let x0 = floats(&pop.individuals[0]);
        let expected: Vec<f64> = (0..dim)
            .map(|d| hho_explore_family_dim_step(x_rand[d], r_a, r_b, x0[d]))
            .collect();
        assert_eq!(floats(&off[0]), &expected, "family-perch offspring must match the pinned formula and draw order");
    }

    #[test]
    fn twin_stream_exploration_tree_branch() {
        // seed=1, n=2, dim=2, progress=0: hawk 0 lands in exploration
        // (|E|>=1), q>=0.5 (tall-tree) sub-branch. Hawk 0 is the FIRST
        // processed, so mean(X) still equals the ORIGINAL population's mean
        // (no in-place effect yet for this test -- see the dedicated
        // in-place test below for that).
        let dim = 2; let n = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(1, &[]);
        let rng_before = rng.clone();
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = HhoGenerator.generate(&pop, &mut ctx);

        let mut twin = rng_before;
        let e0 = 2.0 * twin.next_f64() - 1.0;
        let e = hho_e1(0.0) * e0;
        assert!(e.abs() >= 1.0, "fixture must land in exploration");
        let q = twin.next_f64();
        assert!(q >= 0.5, "fixture must land in the tall-tree sub-branch");
        let _rand_idx = twin.next_below(n as u64); // drawn unconditionally, unused here (finding 2)
        let r_a = twin.next_f64();
        let r_b = twin.next_f64();
        let rabbit_idx = pop.best_index().unwrap();
        let rabbit = floats(&pop.individuals[rabbit_idx]);
        let mean: Vec<f64> = (0..dim).map(|d| {
            floats(&pop.individuals[0])[d] + floats(&pop.individuals[1])[d]
        }).map(|s| s / n as f64).collect();
        let expected: Vec<f64> = (0..dim)
            .map(|d| hho_explore_tree_dim_step(rabbit[d], mean[d], r_a, r_b, -5.0, 5.0))
            .collect();
        assert_eq!(floats(&off[0]), &expected, "tall-tree offspring must match the pinned formula and draw order");
    }

    #[test]
    fn twin_stream_soft_besiege_no_dive_branch() {
        // seed=4, n=2, dim=2, progress=0: hawk 0 lands in exploitation
        // (|E|<1), r>=0.5 && |E|>=0.5 (soft besiege, no dive).
        let dim = 2; let n = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(4, &[]);
        let rng_before = rng.clone();
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = HhoGenerator.generate(&pop, &mut ctx);

        let mut twin = rng_before;
        let e0 = 2.0 * twin.next_f64() - 1.0;
        let e = hho_e1(0.0) * e0;
        assert!(e.abs() < 1.0, "fixture must land in exploitation");
        let r = twin.next_f64();
        assert!(r >= 0.5 && e.abs() >= 0.5, "fixture must land in the soft-besiege-no-dive sub-branch");
        let jump = 2.0 * (1.0 - twin.next_f64());
        let rabbit_idx = pop.best_index().unwrap();
        let rabbit = floats(&pop.individuals[rabbit_idx]);
        let x0 = floats(&pop.individuals[0]);
        let expected: Vec<f64> = (0..dim)
            .map(|d| hho_soft_besiege_dim_step(rabbit[d], e, jump, x0[d]))
            .collect();
        assert_eq!(floats(&off[0]), &expected, "soft-besiege offspring must match the pinned formula and draw order");
    }

    #[test]
    fn twin_stream_hard_besiege_no_dive_consumes_zero_extra_draws_both_hawks() {
        // seed=45, n=2, dim=2, progress=0: BOTH hawks land in hard-besiege-
        // no-dive (r>=0.5 && |E|<0.5) -- proves the ENTIRE generation draws
        // exactly 2 values per hawk (e0, r), zero more, via a full-stream
        // post-call comparison (finding 3: no Jump_strength draw at all).
        let dim = 2; let n = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(45, &[]);
        let rng_before = rng.clone();
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = HhoGenerator.generate(&pop, &mut ctx);
        assert_eq!(off.len(), n);

        let mut twin = rng_before;
        let rabbit_idx = pop.best_index().unwrap();
        let rabbit = floats(&pop.individuals[rabbit_idx]).clone();
        for (i, (individual, off_i)) in pop.individuals.iter().zip(off.iter()).enumerate() {
            let e0 = 2.0 * twin.next_f64() - 1.0;
            let e = hho_e1(0.0) * e0;
            assert!(e.abs() < 1.0, "fixture hawk {i} must land in exploitation");
            let r = twin.next_f64();
            assert!(r >= 0.5 && e.abs() < 0.5, "fixture hawk {i} must land in hard-besiege-no-dive");
            let x_i = floats(individual);
            let expected: Vec<f64> = (0..dim).map(|d| hho_hard_besiege_dim_step(rabbit[d], e, x_i[d])).collect();
            assert_eq!(floats(off_i), &expected, "hawk {i}'s hard-besiege offspring must match the pinned formula");
        }
        // Exactly 2*n draws total for the whole generation -- zero more.
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/hho must consume EXACTLY e0 then r per hawk in this branch, zero Jump_strength draw");
    }

    #[test]
    fn in_place_mean_reflects_already_updated_earlier_hawk() {
        // seed=32, n=2, dim=1, progress=0: hawk 0 -> exploration/family
        // (no mean use, unconditional overwrite), hawk 1 -> exploration/tree
        // (uses mean(X)) -- proves mean(X) is computed from the LIVE,
        // in-place-updated working array (hawk 0's NEW position), not the
        // original pre-generation population (finding 8, ssa.rs's Task 4
        // in-place-chain precedent applied to a whole-population mean).
        let dim = 1; let n = 2;
        let p = SphereShifted::new(vec![0.0; dim], -10.0, 10.0);
        let space = p.space();
        let pop = Population {
            individuals: vec![g(vec![1.0]), g(vec![50.0])],
            fitness: vec![1.0, 2.0], // index 0 is rabbit
        };
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(32, &[]);
        let rng_before = rng.clone();
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = HhoGenerator.generate(&pop, &mut ctx);

        let mut twin = rng_before;
        // Hawk 0: exploration/family.
        let e0_0 = 2.0 * twin.next_f64() - 1.0;
        let e_0 = hho_e1(0.0) * e0_0;
        assert!(e_0.abs() >= 1.0, "fixture hawk 0 must land in exploration");
        let q0 = twin.next_f64();
        assert!(q0 < 0.5, "fixture hawk 0 must land in family-perch");
        let rand_idx0 = twin.next_below(n as u64) as usize;
        let x_rand0 = floats(&pop.individuals[rand_idx0])[0];
        let r_a0 = twin.next_f64();
        let r_b0 = twin.next_f64();
        let hawk0_new = hho_explore_family_dim_step(x_rand0, r_a0, r_b0, floats(&pop.individuals[0])[0]);
        assert_eq!(floats(&off[0])[0], hawk0_new);

        // Hawk 1: exploration/tree -- mean(X) must use hawk0_new, NOT the
        // original pop.individuals[0]=1.0.
        let e0_1 = 2.0 * twin.next_f64() - 1.0;
        let e_1 = hho_e1(0.0) * e0_1;
        assert!(e_1.abs() >= 1.0, "fixture hawk 1 must land in exploration");
        let q1 = twin.next_f64();
        assert!(q1 >= 0.5, "fixture hawk 1 must land in tall-tree");
        let _rand_idx1 = twin.next_below(n as u64);
        let r_a1 = twin.next_f64();
        let r_b1 = twin.next_f64();
        let rabbit = floats(&pop.individuals[0])[0]; // index 0 is rabbit (fitness 1.0 < 2.0)
        let mean_in_place = (hawk0_new + 50.0) / 2.0; // hawk0's NEW value + hawk1's still-original value
        let expected1 = hho_explore_tree_dim_step(rabbit, mean_in_place, r_a1, r_b1, -10.0, 10.0);
        assert_eq!(floats(&off[1])[0], expected1,
            "hawk 1's mean(X) must reflect hawk 0's ALREADY-UPDATED (this-generation) position");

        // Distinguish from the (wrong) frozen-snapshot alternative.
        let mean_frozen = (1.0 + 50.0) / 2.0;
        let wrong = hho_explore_tree_dim_step(rabbit, mean_frozen, r_a1, r_b1, -10.0, 10.0);
        assert_ne!(hawk0_new, 1.0, "sanity: hawk 0 must have actually moved from its old position");
        assert_ne!(floats(&off[1])[0], wrong,
            "hawk 1's offspring must NOT match the frozen-snapshot (pre-generation) mean alternative");
    }

    #[test]
    fn dive_path_soft_besiege_accepts_y() {
        // seed=0, n=2, dim=2, progress=0: hawk 0 lands in the soft-besiege-
        // based rapid dive (r<0.5 && |E|>=0.5). Y's fitness is scripted to
        // strictly improve, so it must be accepted immediately (one internal
        // evaluation, no Z trial, no S/Levy draws) -- the module's dive-path
        // hand-trace.
        let dim = 2;
        let pop = Population {
            individuals: vec![g(vec![1.0, -2.0]), g(vec![3.0, 4.0])],
            fitness: vec![1000.0, 1.0], // hawk 0's OWN fitness is deliberately bad; index 1 is rabbit
        };
        let rabbit = floats(&pop.individuals[1]).clone();
        let x0 = floats(&pop.individuals[0]).clone();

        // Hand-derive hawk 0's e/jump/Y from a twin clone of the SAME seed,
        // BEFORE running the real generator, so the scripted fitness can be
        // set to a value that strictly improves over pop.fitness[0]=1000.0
        // regardless of Y's exact numeric position.
        let mut twin = RngStream::from_master(0, &[]);
        let e0 = 2.0 * twin.next_f64() - 1.0;
        let e = hho_e1(0.0) * e0;
        assert!(e.abs() < 1.0, "fixture must land in exploitation");
        let r = twin.next_f64();
        assert!(r < 0.5 && e.abs() >= 0.5, "fixture must land in the soft-besiege rapid-dive sub-branch");
        let jump = 2.0 * (1.0 - twin.next_f64());
        // Fix round 1: this is hho_soft_dive_y_dim_step (no leading
        // (rabbit-x_i) term), NOT hho_soft_besiege_dim_step -- see finding 4.
        let y: Vec<f64> = (0..dim).map(|d| hho_soft_dive_y_dim_step(rabbit[d], e, jump, x0[d])).collect();

        let p = ScriptedProblem::new(dim, -5.0, 5.0, vec![-999.0]); // fy: strictly improves 1000.0
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(0, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = HhoGenerator.generate(&pop, &mut ctx);

        assert_eq!(floats(&off[0]), &y, "an improving Y must be accepted as hawk 0's final position");
        // Exactly ONE internal evaluation for hawk 0's dive (Y accepted, no Z trial).
        assert_eq!(ctx.eval.used(), 1, "accepting Y must consume exactly one internal evaluation");
    }

    #[test]
    fn dive_path_soft_besiege_rejects_both_leaves_hawk_unchanged() {
        // Same branch as above (seed=0), but Y/Z are BOTH scripted to be
        // worse than hawk 0's current fitness -- hawk 0's final position
        // must be its own UN-DIVED (pre-dive) value, unchanged, and exactly
        // two internal evaluations must be consumed (Y then Z). Also
        // hand-derives Y with the CORRECT formula (fix round 1:
        // hho_soft_dive_y_dim_step, not hho_soft_besiege_dim_step) so this
        // test documents the same pinned Y as dive_path_soft_besiege_
        // accepts_y, even though Y's exact value is irrelevant to a
        // scripted-rejection outcome.
        let dim = 2; let n = 2;
        let pop = Population {
            individuals: vec![g(vec![1.0, -2.0]), g(vec![3.0, 4.0])],
            fitness: vec![-1000.0, 1.0], // hawk 0 already excellent; nothing can improve on it
        };
        let rabbit = floats(&pop.individuals[1]).clone();
        let x0 = floats(&pop.individuals[0]).clone();

        let mut twin = RngStream::from_master(0, &[]);
        let e0 = 2.0 * twin.next_f64() - 1.0;
        let e = hho_e1(0.0) * e0;
        let r = twin.next_f64();
        assert!(r < 0.5 && e.abs() >= 0.5, "fixture must land in the soft-besiege rapid-dive sub-branch");
        let jump = 2.0 * (1.0 - twin.next_f64());
        let _y: Vec<f64> = (0..dim).map(|d| hho_soft_dive_y_dim_step(rabbit[d], e, jump, x0[d])).collect();

        let p = ScriptedProblem::new(dim, -5.0, 5.0, vec![999.0, 999.0]); // fy, fz: both far worse
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(0, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = HhoGenerator.generate(&pop, &mut ctx);

        let _ = n;
        assert_eq!(floats(&off[0]), &x0, "when neither Y nor Z improves, hawk 0 must be left COMPLETELY unchanged");
        assert_eq!(ctx.eval.used(), 2, "both Y and Z trials must be attempted (and counted)");
    }

    #[test]
    fn dive_path_hard_besiege_y_formula_uses_mean_not_x_i() {
        // seed=8, n=2, dim=2, progress=0: hawk 0 lands in the hard-besiege-
        // based rapid dive (r<0.5 && |E|<0.5) -- Y's formula uses mean(X),
        // not X_i (finding 4's asymmetry vs the soft-dive branch).
        let dim = 2; let n = 2;
        let pop = pop_nd(n, dim);
        let rabbit_idx = pop.best_index().unwrap();
        let rabbit = floats(&pop.individuals[rabbit_idx]).clone();
        let mean: Vec<f64> = (0..dim).map(|d| {
            (floats(&pop.individuals[0])[d] + floats(&pop.individuals[1])[d]) / n as f64
        }).collect();

        let mut twin = RngStream::from_master(8, &[]);
        let e0 = 2.0 * twin.next_f64() - 1.0;
        let e = hho_e1(0.0) * e0;
        assert!(e.abs() < 1.0, "fixture must land in exploitation");
        let r = twin.next_f64();
        assert!(r < 0.5 && e.abs() < 0.5, "fixture must land in the hard-besiege rapid-dive sub-branch");
        let jump = 2.0 * (1.0 - twin.next_f64());
        let y: Vec<f64> = (0..dim).map(|d| hho_hard_dive_y_dim_step(rabbit[d], e, jump, mean[d])).collect();

        // -999.0 universally improves over any fitness this fixture uses;
        // padded to several copies since hawk 1 (processed after hawk 0,
        // from the same continuing RNG stream) may ALSO land in a dive
        // sub-branch consuming further scripted values -- this test only
        // asserts on hawk 0's own outcome.
        let p = ScriptedProblem::new(dim, -5.0, 5.0, vec![-999.0; 4]);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(8, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = HhoGenerator.generate(&pop, &mut ctx);

        assert_eq!(floats(&off[0]), &y, "hard-dive Y (mean-based) must match the pinned formula and be accepted");
    }

    // ---- Determinism / min_pop ----

    #[test]
    fn determinism_same_seed_bit_identical() {
        let n = 6; let dim = 4;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let run = || {
            let mut evaluator = Evaluator::new(&p, 1000);
            let mut rng = RngStream::from_master(42, &[]);
            let mut bb = Blackboard::new();
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            HhoGenerator.generate(&pop, &mut ctx)
        };

        let a = run();
        let b = run();
        assert_eq!(a.len(), n);
        for (ga, gb) in a.iter().zip(b.iter()) {
            assert_eq!(floats(ga), floats(gb), "same seed must produce bit-identical offspring");
        }
    }

    #[test]
    fn min_pop_below_2_panics() {
        let n = 1; let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            HhoGenerator.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/hho must reject pop_size < 2 at runtime as a backstop");
    }

    // ---- Budget accounting (the plan's mandated in-generator-evaluation
    // test): an EXACT, mechanically-derived expected evals_used for a run
    // where dives genuinely occur, cross-checked against a real Engine::run
    // -- no overshoot beyond budget. ----

    #[test]
    fn budget_accounting_exact_count_across_one_generation_with_dives() {
        // The plan's mandated in-generator-evaluation budget-accounting
        // test: derive an EXACT expected evals_used for a run where dives
        // genuinely occur, independent of a real Engine::run, then verify a
        // real Engine::run consumes EXACTLY that many evaluations -- no
        // overshoot beyond budget.
        //
        // Method: reconstruct the population Engine::run's init phase would
        // produce (same problem, same RngStream path [run_id,0] per
        // engine.rs's documented convention), then call HhoGenerator::
        // generate() STANDALONE with the matching stage-0 generator stream
        // ([run_id,1]) and a SHARED Evaluator already at used=pop_size
        // (matching progress at the top of generation 0) -- the DELTA in
        // eval.used() across that standalone call is EXACTLY the internal
        // dive-trial evaluation count for generation 0 (measured directly,
        // not hand-guessed). The total budget = init + generation 0's
        // engine-own stage-evaluate (always pop_size, unconditional
        // replace/generational) + that measured internal delta is then fed
        // to a real Engine::run, asserting evals_used == budget exactly and
        // iterations == 1 (the run stops cleanly at the generation
        // boundary; any doomed speculative generation-1 attempt costs
        // nothing once the budget is fully exhausted -- see the module
        // doc's exhaustion-semantics note).
        // IMPORTANT: `gen/hho`'s branch tree depends on `progress =
        // ctx.eval.used()/ctx.eval.budget()` (finding 1), so the standalone
        // replay's `Evaluator` must be constructed with the SAME `budget`
        // value the eventual real `Engine::run` will use -- otherwise
        // `progress` differs between the replay and the real run and the
        // measured `internal_dive_evals` would not describe the real run at
        // all. Since the target budget is ITSELF `pop_size*2 +
        // internal_dive_evals` (a function of the measurement), resolve this
        // by fixed-point iteration: start from a budget guess, measure
        // `internal_dive_evals` under that guess's `progress`, recompute the
        // budget, and repeat until it stabilizes (branch decisions only
        // change at specific `progress` boundaries, so this converges in
        // practice within a couple of iterations for a fixed seed).
        let seed = 42u64; let run_id = 0u64;
        let pop_size = 10usize; let dim = 3usize;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();

        let mut init_rng = RngStream::from_master(seed, &[run_id, 0]);
        let init_individuals: Vec<Genotype> =
            (0..pop_size).map(|_| crate::init::sample_uniform(space, &mut init_rng)).collect();

        let mut budget = pop_size as u64 * 100; // initial guess
        let mut internal_dive_evals = 0u64;
        let mut converged = false;
        for _ in 0..20 {
            let mut evaluator = Evaluator::new(&p, budget);
            let init_fitness = evaluator.evaluate(&init_individuals).unwrap();
            assert_eq!(evaluator.used(), pop_size as u64);
            let pop = Population { individuals: init_individuals.clone(), fitness: init_fitness };

            let mut gen_rng = RngStream::from_master(seed, &[run_id, 1]); // stage 0 generator stream
            let mut bb = Blackboard::new();
            let used_before = evaluator.used();
            {
                let mut ctx = Ctx { space, rng: &mut gen_rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
                let off = HhoGenerator.generate(&pop, &mut ctx);
                assert_eq!(off.len(), pop_size);
            }
            internal_dive_evals = evaluator.used() - used_before;
            let new_budget = pop_size as u64 + pop_size as u64 + internal_dive_evals;
            if new_budget == budget { converged = true; break; }
            budget = new_budget;
        }
        assert!(converged, "budget fixed-point iteration did not converge");
        assert!(internal_dive_evals > 0,
            "test fixture (seed={seed}, pop_size={pop_size}, dim={dim}) must exercise at least one rapid dive -- got 0 internal evaluations, pick a different seed/pop_size");

        let reg = { let mut r = Registry::new(); crate::register_builtins(&mut r); r };
        let spec = crate::presets::hho(pop_size, budget);
        let e = Engine::from_spec(&spec, &reg, space).unwrap();
        let r = e.run(&p, RunConfig { master_seed: seed, run_id }, None).unwrap();

        assert_eq!(r.evals_used, budget,
            "evals_used must equal the exact hand-computed count (init + one generation's engine-own stage-evaluate + measured internal dive evals), no overshoot beyond budget");
        assert_eq!(r.iterations, 1, "exactly one generation should have completed with this exact budget");
    }

    // ---- `best_f`/Evaluator-observed-minimum unification (M3-1 Task 1;
    // was a documented divergence through M2d-4 -- see the module doc's
    // "global_best/IOH visibility" section above for the full mechanism
    // this test used to document as a known gap). ----

    /// **Regression test for the M3-1 Task 1 engine fix.** Through M2d-4,
    /// `gen/hho`'s internal dive-trial evaluations ran on the RAW,
    /// un-clamped `Y`/`Z` candidate, but `Engine::run`'s stage loop
    /// boundary-repaired the accepted offspring BEFORE its own capturing
    /// `eval.evaluate(&offspring)` call, so `RunResult::best_f` (sourced
    /// from the engine's own `global_best`, itself fed only by that
    /// capturing call and a couple of other engine-side sites) could miss a
    /// charged, IOH-observable raw dive-trial value entirely -- while an
    /// `EvalObserver` attached to the SAME `Evaluator` (the mechanism
    /// backing the IOH `.dat` best-so-far stream) sees every charged
    /// evaluation, raw dive trials included. M3-1 Task 1 unifies
    /// `RunResult::best_f`/`best_x` with `Evaluator`'s own best-tracking
    /// (now genotype-aware, not just scalar), which is updated inside
    /// `Evaluator::evaluate` itself -- the single gateway every charged
    /// evaluation in this crate passes through, including `gen/hho`'s
    /// internal `ctx.eval.evaluate(..)` dive-trial calls. So `best_f`/
    /// `best_x` can no longer diverge from the Evaluator-observed minimum,
    /// for HHO or any other preset. This fixture (BBOB f3/dim5/instance1,
    /// seed 19, pop_size=20, budget=5000) is the exact one the M2d-4 review
    /// used to find and pin the pre-fix divergence; kept as this crate's
    /// permanent regression coverage that the gap stays closed.
    #[test]
    fn hho_captured_best_f_matches_evaluator_observed_minimum() {
        use sezgi_problems::BbobProblem;

        struct MinObserver(std::sync::Arc<std::sync::Mutex<f64>>);
        impl sezgi_core::problem::EvalObserver for MinObserver {
            fn on_eval(&mut self, _eval_index: u64, _f: f64, best_so_far: f64) {
                let mut m = self.0.lock().unwrap();
                if best_so_far < *m { *m = best_so_far; }
            }
        }

        let reg = { let mut r = Registry::new(); crate::register_builtins(&mut r); r };
        let p = BbobProblem::new(3, 5, 1).unwrap(); // BBOB f3 (Rastrigin), dim 5, instance 1
        let spec = crate::presets::hho(20, 5000);
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();

        let observed_min = std::sync::Arc::new(std::sync::Mutex::new(f64::INFINITY));
        let obs = MinObserver(observed_min.clone());
        let r = e.run(&p, RunConfig { master_seed: 19, run_id: 0 }, Some(Box::new(obs))).unwrap();
        let observed_min = *observed_min.lock().unwrap();

        assert_eq!(r.best_f, observed_min,
            "RunResult::best_f ({}) must equal the Evaluator-observed minimum ({}) on this \
             fixture -- post-M3-1-Task-1, best_f/best_x are sourced from Evaluator's own \
             best-tracking, which sees every charged evaluation (including gen/hho's internal \
             raw dive-trial evaluate calls), so this cannot diverge for HHO or any other preset.",
            r.best_f, observed_min);
    }
}
