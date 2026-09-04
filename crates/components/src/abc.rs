use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::rng::RngStream;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};
use sezgi_core::state::StateReq;

/// Artificial Bee Colony (Karaboga, D. 2005, "An Idea Based On Honey Bee
/// Swarm For Numerical Optimization", Technical Report TR-06, Erciyes
/// University, Engineering Faculty, Computer Engineering Department; and
/// Karaboga, D., Basturk, B. 2007, "A Powerful and Efficient Algorithm for
/// Numerical Function Optimization: Artificial Bee Colony (ABC) Algorithm",
/// *Journal of Global Optimization* 39(3), 459-471, DOI
/// 10.1007/s10898-007-9149-x) -- a **labeled metaphor preset**: faithful to
/// the primary artifacts' own update equations and loop structure, with a
/// pinned deterministic draw order and property tests, but NOT validated
/// against the paper's (or any other publication's) reported benchmark
/// numbers. No established equivalence critique (the Camacho-Villalón/
/// Dorigo/Stützle *ITOR* six-algorithm critique does not cover ABC) --
/// primary sources only.
///
/// ## Provenance (PROVENANCE-FIRST protocol)
///
/// The official ABC homepage (`abc.erciyes.edu.tr/software.htm`, the
/// current home of the site the task brief pointed at,
/// `mf.erciyes.edu.tr/abc`) gates its C and MATLAB downloads behind a
/// registration form (`form.aspx`, not fetchable by this project's tooling
/// -- the same class of limitation earlier tasks' File Exchange
/// metadata-only pages recorded), but links its Python and Rust ports
/// directly to public GitHub repositories with NO gate. Two artifacts were
/// used, both fetched in full:
///
/// 1. **`github.com/artificialbeecolony/Python_ABC`** (the official
///    homepage's own "Python -- 27.05.2020" link; author "Omur Sahin",
///    under the `artificialbeecolony` GitHub organization the homepage
///    itself links to) -- `ABC.py`, `Config.py`, `ABCAlgorithm.py` fetched
///    verbatim via `git clone`.
/// 2. **`ABCorig.m`** (`github.com/tianxiangyi/MATLAB-ABC-Algorithm`,
///    `calculateFitness.m` from the same repository) -- a THIRD-PARTY
///    MIRROR, used because the official gated download could not be
///    fetched, but its own file header is Karaboga & Basturk's ORIGINAL
///    copyright notice verbatim ("Copyright (c) 2009 Erciyes University,
///    Intelligent Systems Research Group, The Dept. of Computer
///    Engineering", contact emails `karaboga@erciyes.edu.tr` /
///    `bahriye@erciyes.edu.tr`, citing TR-06 2005 + JGO 2007 + Applied Soft
///    Computing 2008 + Applied Math & Comp 2009 -- the SAME four
///    publications this module's own citations draw from) -- i.e. this is
///    Karaboga & Basturk's OWN authored code, mirrored, not a
///    reimplementation. **This is the primary artifact that GOVERNS**
///    whenever it and the Python port disagree (see the scout `>` vs `>=`
///    delta below) -- the Python port is corroborating, not primary,
///    despite also being linked from the official homepage.
///
/// Both artifacts agree, independently and byte-for-byte in the shared
/// formulas (`calculateFitness`, `calculate_probabilities`/`prob=...`,
/// the neighbor-move equation), on every mechanism below except the one
/// noted delta.
///
/// **`ABCorig.m`'s control-parameter section, quoted verbatim:**
/// ```text
/// NP=20; %/* The number of colony size (employed bees+onlooker bees)*/
/// FoodNumber=NP/2; %/*The number of food sources equals the half of the colony size*/
/// limit=100; %/*A food source which could not be improved through "limit" trials is abandoned by its employed bee*/
/// maxCycle=2500; %/*The number of cycles for foraging {a stopping criteria}*/
/// ```
///
/// **`ABCorig.m`'s three phases, quoted verbatim (`i`/`j`/`k` 1-indexed,
/// `D` = dimension, `sol`/`Foods`/`Fitness`/`trial` MATLAB row vectors):**
/// ```text
/// %%%%%%%%% EMPLOYED BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%
/// for i=1:(FoodNumber)
///     Param2Change=fix(rand*D)+1;
///     neighbour=fix(rand*(FoodNumber))+1;
///     while(neighbour==i)
///         neighbour=fix(rand*(FoodNumber))+1;
///     end;
///     sol=Foods(i,:);
///     sol(Param2Change)=Foods(i,Param2Change)+(Foods(i,Param2Change)-Foods(neighbour,Param2Change))*(rand-0.5)*2;
///     ind=find(sol<lb); sol(ind)=lb(ind);
///     ind=find(sol>ub); sol(ind)=ub(ind);
///     ObjValSol=feval(objfun,sol);
///     FitnessSol=calculateFitness(ObjValSol);
///     if (FitnessSol>Fitness(i))
///         Foods(i,:)=sol; Fitness(i)=FitnessSol; ObjVal(i)=ObjValSol; trial(i)=0;
///     else
///         trial(i)=trial(i)+1;
///     end;
/// end;
///
/// %%%%%%%%%%%%%%%%%%%%%%%% CalculateProbabilities %%%%%%%%%%%%%%%%%%%%%%%%
/// %/*or in a way used in the metot below prob(i)=a*fitness(i)/max(fitness)+b*/
/// prob=(0.9.*Fitness./max(Fitness))+0.1;
///
/// %%%%%%%%%%%%%%%%%%%%%%%% ONLOOKER BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
/// i=1; t=0;
/// while(t<FoodNumber)
///     if(rand<prob(i))
///         t=t+1;
///         [[SAME Param2Change/neighbour/sol move as the employed phase, on source i]]
///     end;
///     i=i+1;
///     if (i==(FoodNumber)+1) i=1; end;
/// end;
///
/// %%%%%%%%%%%% SCOUT BEE PHASE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
/// %In Basic ABC, only one scout is allowed to occur in each cycle*/
/// ind=find(trial==max(trial));
/// ind=ind(end);
/// if (trial(ind)>limit)
///     sol=(ub-lb).*rand(1,D)+lb;
///     ObjValSol=feval(objfun,sol);
///     FitnessSol=calculateFitness(ObjValSol);
///     Foods(ind,:)=sol; Fitness(ind)=FitnessSol; ObjVal(ind)=ObjValSol;
/// end;
/// ```
///
/// **`calculateFitness.m`, quoted verbatim (byte-identical to
/// `Python_ABC`'s `calculate_fitness`):**
/// ```text
/// function fFitness=calculateFitness(fObjV)
/// fFitness=zeros(size(fObjV));
/// ind=find(fObjV>=0);
/// fFitness(ind)=1./(fObjV(ind)+1);
/// ind=find(fObjV<0);
/// fFitness(ind)=1+abs(fObjV(ind));
/// ```
///
/// **`Python_ABC`'s main loop (`ABCAlgorithm.py`), confirming the phase
/// order used throughout this module:**
/// ```text
/// abc.initial(); abc.memorize_best_source()
/// while not abc.stopping_condition():
///     abc.send_employed_bees()
///     abc.calculate_probabilities()
///     abc.send_onlooker_bees()
///     abc.memorize_best_source()
///     abc.send_scout_bees()
///     abc.increase_cycle()
/// ```
///
/// ## Verified findings (deltas from the plan's sketch; SOURCE GOVERNS)
///
/// 1. **Pop <-> food-source convention -- settles the brief's flagged
///    ambiguity: sezgi's `pop_size` IS `SN`/`FoodNumber` directly, NOT
///    Karaboga's `NP` (colony size).** `FoodNumber=NP/2` in both artifacts,
///    and critically, there is only EVER one array of solutions
///    (`Foods`/`self.foods`, length `FoodNumber`) -- `NP` ("employed
///    bees+onlooker bees") is pure bookkeeping for "SN employed-bee-visits
///    + SN onlooker-bee-visits per cycle", not a second sub-population.
///
///    Onlookers do not have their own positions; they merely SELECT (via
///    roulette) one of the SAME `SN` food sources to perturb. This maps
///    directly onto sezgi's `Population` struct, which always holds exactly
///    `pop_size` individuals -- so `pop_size` = `SN`, and `NP` is discarded
///    entirely as a preset parameter (the brief's sketched "pop 40 (=20
///    sources?)" resolves to the SECOND reading: `presets::abc`'s
///    `pop_size` argument IS the food-source count directly, canonical 20).
/// 2. **`min_pop` -- ADJUSTED from the plan's sketched `3` down to `2`,
///    the SAME correction `fpa.rs`'s finding 4 and `tlbo.rs`'s finding 6
///    made for the identical reason:** the neighbour `k` (`neighbour` in
///    both artifacts) needs only be distinct from `i`, never from a THIRD
///    index -- `SN=2` already gives the employed/onlooker move its one and
///    only valid neighbour (the rejection loop for `k` terminates
///    immediately, no iterations needed), and the roulette in `abc_
///    probabilities` is well-defined (and non-degenerate: `prob` values
///    differ whenever `Fitness` values differ) for any `SN>=1` -- `SN=2` is
///    a HARD requirement only because of the `k!=i` rejection loop, not
///    because of anything roulette-specific. Enforced via `ComponentMeta::
///    with_min_pop(2)` on every ABC component (spec-validation) plus a
///    runtime `assert!` backstop on the generator, the crate's usual
///    two-layer idiom.
/// 3. **Employed-phase draw order, verified line-for-line:** `Param2Change`
///    (dimension `j`, ONE draw) THEN `neighbour` (`k`, `>=1` draw,
///    rejection-sampled until `!=i`) THEN the move's own `rand` (`phi =
///    (rand-0.5)*2 in [-1,1]`). Exactly ONE dimension changes per source per
///    visit -- `sol=Foods(i,:)` copies everything, only `sol(Param2Change)`
///    is overwritten -- see [`abc_candidate`]'s
///    `only_the_chosen_dimension_changes` test.
/// 4. **The neighbor-move equation, verified:** `v_ij = x_ij + phi*(x_ij -
///    x_kj)`, exactly the plan's sketch -- see [`abc_dim_step`].
/// 5. **Fitness transform (`calculateFitness`) -- the brief's flagged
///    "verify what the reference does with negative objectives" -- ANSWER:
///    the reference ALREADY handles `f<0` natively, no adaptation needed.**
///    `f>=0 -> 1/(f+1)`; `f<0 -> 1+|f|`. Both branches are STRICTLY
///    monotonically decreasing in `f` (`d/df[1/(f+1)] = -1/(f+1)^2 < 0`;
///    `d/df[1-f] = -1`), continuous AND C1 at the `f=0` boundary (both
///    branches evaluate to `1` at `f=0`, both have derivative `-1` there) --
///    see [`abc_fitness_transform`]'s tests, including the boundary-match
///    and negative-domain checks. **Consequence (a genuine simplification
///    opportunity, not a compromise): because the transform is strictly
///    monotonically decreasing and hence order-reversing, `Karaboga's own
///    greedy-accept test `FitnessSol > Fitness(i)` is PROVABLY EQUIVALENT to
///    the plain raw-objective comparison `f_candidate < f_parent`** -- so
///    [`AbcTrialGreedy`] never needs to call [`abc_fitness_transform`] at
///    all for the accept/reject decision; the transform is used ONLY by
///    [`abc_probabilities`] below, where the actual MAGNITUDE (not just the
///    order) of the transformed value matters. See the
///    `fitness_transform_order_matches_raw_comparison_property` test for
///    the hand-checked proof, exercised over same-sign AND mixed-sign
///    pairs.
/// 6. **Probability formula -- the brief's flagged "verify the exact
///    transform" -- a genuine delta from the plan's sketched `fit_i/
///    sum(fit)`:** `prob(i) = 0.9*(Fitness(i)/max(Fitness)) + 0.1`
///    (`calculate_probabilities`/`calculateProbabilities`), byte-identical
///    across both artifacts (the MATLAB comment even says so: `prob(i)=a*
///    fitness(i)/max(fitness)+b`). Range: `(0.1, 1.0]` (never below `0.1`,
///    since `Fitness` is always `>0`, guaranteeing the onlooker scan below
///    always terminates almost surely -- no `SN`-food-source-with-
///    zero-selection-probability degenerate case is possible) -- see
///    [`abc_probabilities`].
/// 7. **Onlooker scan -- NOT literal cumulative-weight roulette wheel
///    sampling; a repeated linear scan with per-visit accept/reject, VERIFIED
///    line-for-line, byte-identical across both artifacts:** `i=1; t=0;
///    while t<SN: if rand<prob(i): t+=1; [[move]]; i=(i+1) mod SN`. `prob`
///    is computed ONCE (before this loop) and never recomputed inside it,
///    even though `Fitness`/`Foods` DO mutate in place as visits succeed --
///    a genuine, faithfully-reproduced staleness in the reference itself,
///    not a sezgi simplification. The scan can (and, for skewed fitness
///    distributions, routinely does) visit the SAME source index multiple
///    times before `t` reaches `SN`, and some indices zero times -- see
///    "Phase-design adjudication" below for why this rules out the
///    `Generator`+`Replacer` shape TLBO used.
/// 8. **Scout phase -- at most ONE scout per cycle, VERIFIED in both
///    artifacts' own comments** ("In Basic ABC, only one scout is allowed
///    to occur in each cycle" / one `if np.amax(trial)>=LIMIT` check).
///    Target selection: `argmax(trial)`, **TIES BROKEN TO THE LAST
///    (highest) INDEX** -- `ind=find(trial==max(trial)); ind=ind(end)`
///    explicitly takes MATLAB `find`'s LAST match, not the first -- the
///    OPPOSITE tie convention from [`Population::best_index`]'s
///    first-seen-wins used throughout the rest of this crate -- see
///    [`abc_scout_target`]. **A SECOND, undisclosed cross-artifact
///    divergence, same resolution as finding 9 below (MATLAB governs):**
///    `Python_ABC`'s own scout target is `_self.trial.argmax(axis=0)` --
///    NumPy's `argmax` returns the FIRST occurrence of the maximum, the
///    OPPOSITE tie-break from `ABCorig.m`'s `ind(end)`. Neither port
///    documents this difference from the other; `ABCorig.m`'s own
///    last-index convention is what [`abc_scout_target`] pins.
/// 9. **A genuine cross-artifact delta -- MATLAB (primary) GOVERNS over the
///    Python port:** `ABCorig.m`'s scout condition is STRICT `trial(ind)>
///    limit`; `Python_ABC`'s `ABC.py` instead uses `np.amax(_self.trial) >=
///    _self.conf.LIMIT` (`>=`, not `>`). Since `ABCorig.m` is Karaboga &
///    Basturk's OWN authored/copyrighted code (see provenance above) and
///    `Python_ABC` is a later third-party-style reimplementation (even
///    though also linked from the official homepage), the strict `>` from
///    `ABCorig.m` is what this module pins -- see [`AbcOnlookerScout::
///    adapt`].
/// 10. **`limit` -- the brief's flagged "VERIFY the formula" -- ANSWER:
///     there is no formula in the reference code at all.** `ABCorig.m`'s own
///     demo hardcodes `limit=100` as a literal constant, completely
///     decoupled from both `SN` and `D` (in that same script, `D=100` --
///     the equality with `limit` is coincidental, not a rule). Since a
///     fixed, dimension-independent constant would not scale sensibly
///     across sezgi's varying `dim` presets, this module instead uses `limit
///     = SN*D`, the dimension-scaling rule of thumb from **Karaboga, D.,
///     Akay, B. (2009), "A comparative study of Artificial Bee Colony
///     algorithm", *Applied Mathematics and Computation* 214, 108-132**
///     (one of the four papers `ABCorig.m`'s own header cites) -- a
///     documented, paper-traceable, widely reused default, NOT `ABCorig.m`'s
///     own literal constant -- see [`abc_limit`]. Configurable via
///     [`AbcOnlookerScout::from_params`]'s optional `"limit"` override.
///
/// ## Phase-design adjudication (multi-stage vs. single generator vs. the
/// design actually used)
///
/// The brief posed a binary choice -- multi-stage (`StageSpec`s, TLBO's
/// precedent) vs. a single generator with internal phases -- but the
/// verified loop structure above rules out BOTH literal options, for two
/// independent reasons, and motivates a THIRD design:
///
/// - **The employed phase fits the crate's standard `Generator`+`Replacer`
///   idiom cleanly** (like every TLBO-precedent stage): it visits each
///   source index exactly once, in a fixed order, so it can be expressed as
///   "propose one candidate per index" (a pure [`Generator`], NO internal
///   evaluation) + the engine's own per-stage `eval.evaluate(&offspring)`
///   (giving EXACTLY `SN` evaluations, honestly, via the crate's existing
///   machinery) + a paired [`Replacer`] doing the greedy accept/reject AND
///   the coupled trial-counter update (finding 5's equivalence proof means
///   the accept test needs no fitness transform at all) -- the SAME
///   acceptance-coupled-replacer shape `ba.rs`'s `BaLoudnessGreedy`
///   established (T7-M2d-3-style precedent: a new replacer is justified
///   only because trial-counter bookkeeping needs `ctx.bb` access no
///   existing replacer has). **This is a genuine, tagged simplification,
///   NOT a necessity -- an Adapter-based, literally in-place employed
///   phase WAS available** (the onlooker phase below, `AbcOnlookerScout`,
///   is direct proof the shape works, with equally honest `SN`-eval
///   accounting: a self-contained `Adapter` could just as well have
///   evaluated each employed visit's candidate sequentially, in place,
///   the same way the onlooker adapter already does). The frozen-batch
///   design was chosen instead as a DELIBERATE structural trade: it lets
///   the employed phase reuse the crate's dominant `Generator`+`Replacer`
///   decomposition convention (every non-in-place preset in this crate --
///   SCA/JAYA/GWO/WOA/DE/TLBO/etc. -- already works this way; only
///   `ssa.rs`'s followers and `fa.rs`'s double loop are genuinely
///   in-place, and only because neither needs a fresh evaluation
///   mid-phase to decide anything), keeping the acceptance-coupled-
///   replacer idiom (`ba.rs`'s precedent) applicable here too, at the cost
///   of one documented positional-fidelity departure from `ABCorig.m`
///   (a later source `k<i` already updated earlier in this SAME employed
///   pass would, in the reference, be read in its POST-update state;
///   sezgi's `gen/abc-employed` instead proposes all `SN` candidates
///   against the FROZEN pop snapshot the stage started with) -- documented
///   here as `// sezgi simplification:` in [`AbcEmployedGenerator::
///   generate`]. The onlooker phase is NOT given the same treatment
///   (see below) specifically because its OWN structural constraint (the
///   scan's variable-length, possibly-repeated visits) makes the
///   `Generator`+`Replacer` shape genuinely inapplicable there, not merely
///   less convenient -- the two phases' designs answer two different
///   questions ("which shape is convenient and honest" for employed;
///   "which shape is even expressible" for onlooker).
/// - **The onlooker phase does NOT fit that idiom, for a structural reason,
///   not a stylistic one:** finding 7's scan can visit the SAME source index
///   multiple times (with the SECOND visit's move needing to see the FIRST
///   visit's accepted outcome to decide correctly) or zero times, before `t`
///   reaches `SN` -- this cannot be expressed as "exactly `SN` candidates,
///   index-aligned 1:1 with `pop`" the way every `Generator`+`Replacer`
///   stage in this crate assumes (`replace/one-to-one-greedy`'s and
///   `ba.rs`'s `BaLoudnessGreedy`'s both zip `off_i`/`off_f` against `pop`
///   BY POSITION). Attempting it anyway (e.g. an `HHO`-style `Generator`
///   that internally evaluates each visit's candidate itself) would ALSO
///   force the engine's own unavoidable per-stage `eval.evaluate(&offspring)`
///   to re-evaluate whatever is returned -- for `hho.rs` this "double eval"
///   is the FAITHFUL choice (`HHO.m` itself genuinely double-evaluates,
///   once per dive-trial and again next iteration's Loop 1); ABC has NO
///   analogous genuine double-eval in its reference, so paying it here would
///   be a real, unjustified 2x budget inflation, violating the brief's
///   explicit "onlooker = #onlookers evals" honesty requirement. Instead,
///   the onlooker phase (PLUS the scout phase, which the reference always
///   runs immediately after it, on the SAME cycle) is implemented as a
///   single self-contained [`Adapter`] ([`AbcOnlookerScout`]), following
///   `cs.rs`'s `AbandonWorstFraction` precedent directly: it reads/mutates
///   `pop` and the blackboard directly, and calls `ctx.eval.evaluate(..)`
///   itself for each visit's candidate (and, at most once, the scout's
///   reinitialized position) -- EXACTLY `SN + {0,1}` evaluations, no more,
///   no engine change required (per-stage adapters are never
///   auto-re-evaluated by `Engine::run`, confirmed by the existing `CS`
///   precedent -- the ONLY same-shape precedent: `hho.rs` does NOT use an
///   `Adapter` for its in-generator evaluation at all, its trial evals
///   live inside `gen/hho` itself and ARE genuinely double-evaluated by
///   design, per that module's own "In-generator evaluation" doc section --
///   citing it here would be a different, unrelated shape), and
///   `RunResult::best_f`/`best_x`/IOH visibility for free: `ctx.eval` is the
///   SAME `Evaluator` every other charged evaluation in a run passes
///   through, and `Evaluator::evaluate` updates its own best-tracking
///   (`best_so_far`/`best_x_so_far`) inside itself, on every call --
///   including this adapter's own `ctx.eval.evaluate(..)` calls -- so an
///   adapter-evaluated individual is visible the instant it is charged, with
///   no separate engine-side bookkeeping needed (M3-1 Task 1; see
///   `sezgi_core::problem::Evaluator`'s own doc). Because this adapter evaluates each
///   visit's candidate SEQUENTIALLY and mutates its own working copy in
///   place as it goes, it gets `ABCorig.m`'s literal in-place chaining
///   semantics FOR FREE, with NO simplification needed on this side --
///   the asymmetry (employed = frozen/simplified, onlooker = literal/exact)
///   reflects that the onlooker phase has NO OTHER expressible option
///   (the `Generator`+`Replacer` shape genuinely cannot represent its
///   variable-length, possibly-repeated scan), whereas the employed
///   phase's simplification was a CHOSEN trade among two available shapes
///   (see above), not a forced one.
/// - **Net result: ONE `StageSpec`, not two** (unlike TLBO's two symmetric
///   stages) -- `gen/abc-employed` + `replace/abc-trial-greedy` +
///   `adapter/abc-onlooker-scout`, all in the same stage, so one engine
///   `iteration` == one Karaboga "cycle" exactly, and the stage's adapter
///   runs immediately after its replacer, matching the reference's own
///   `employed -> probabilities -> onlooker -> memorize -> scout` order
///   term for term.
///
/// ## Blackboard state
///
/// `abc/trials`: `Vec<f64>`, length `pop_size` (`SN`), Karaboga's `trial`
/// array (kept as `f64` rather than an integer type purely to match this
/// crate's existing numeric-vector blackboard convention, e.g.
/// `mfo.rs`'s `mfo/flame_fitness`). [`AbcEmployedGenerator::meta`] declares
/// `requires` only, bootstrapping it to `SN` zeros on absence (generation 0
/// only) -- the SAME bootstrap-when-absent idiom `mfo.rs`'s generator uses,
/// so that spec-validation still catches a misconfigured spec that swaps
/// out `replace/abc-trial-greedy`/`adapter/abc-onlooker-scout` for
/// something that doesn't maintain this state (a silent-staleness bug,
/// not a crash, without this guard). [`AbcTrialGreedy`] and
/// [`AbcOnlookerScout`] both declare `requires` AND `provides` (they read
/// current counts to increment/reset, and write updated ones back) --
/// mirroring `mfo.rs`'s `MfoFlameAdapter` pairing exactly.
///
/// ## Eval accounting per phase (honest, per the brief's explicit ask)
///
/// - **Employed:** exactly `SN` evaluations -- the engine's own single
///   `eval.evaluate(&offspring)` call for `gen/abc-employed`'s `SN`
///   candidates.
/// - **Onlooker:** exactly `SN` evaluations -- one per accepted scan visit
///   (`t` runs `0..SN`), each via `AbcOnlookerScout::adapt`'s own
///   `ctx.eval.evaluate(..)` call on a single candidate. On budget
///   exhaustion mid-scan, the scan stops immediately (the failed attempt
///   charges nothing, per `Evaluator::evaluate`'s own all-or-nothing
///   contract) and the scout phase is skipped for that cycle.
/// - **Scout:** `0` or `1` evaluation -- at most one reinitialized food
///   source per cycle, per finding 8.
/// - **Total per cycle (no scout):** `2*SN`. **With a scout:** `2*SN+1`.
///
/// ## Tier note
///
/// Labeled metaphor preset (this crate's standard three-tier convention):
/// faithful to the verified primary artifacts'
/// equations and loop structure, pinned draw order, property-tested, but
/// NOT validated against any publication's reported benchmark numbers.
///
/// The per-source neighbor-move math is factored into [`abc_dim_step`]/
/// [`abc_candidate`], the fitness transform into [`abc_fitness_transform`],
/// the roulette probabilities into [`abc_probabilities`], the scout target
/// selection into [`abc_scout_target`], and the abandonment threshold into
/// [`abc_limit`], so each is unit-tested directly without needing to fake
/// `Ctx`/`RngStream`.
pub fn abc_dim_step(x_i_j: f64, x_k_j: f64, phi: f64) -> f64 {
    x_i_j + phi * (x_i_j - x_k_j)
}

/// Builds the full candidate vector for one neighbor-move visit: a copy of
/// `x_i` with dimension `j` replaced by `abc_dim_step(x_i[j], x_k[j], phi)`
/// -- every OTHER dimension is left bit-identical to `x_i` (finding 3:
/// `sol=Foods(i,:)` copies everything, only `sol(Param2Change)` is
/// overwritten).
pub fn abc_candidate(x_i: &[f64], x_k: &[f64], j: usize, phi: f64) -> Vec<f64> {
    let mut c = x_i.to_vec();
    c[j] = abc_dim_step(x_i[j], x_k[j], phi);
    c
}

/// `calculateFitness.m`'s minimization transform (finding 5): for `f>=0`,
/// `1/(f+1)`; for `f<0`, `1+|f|`. Always strictly positive, strictly
/// monotonically decreasing, and continuous (C1) at `f=0`.
pub fn abc_fitness_transform(f: f64) -> f64 {
    if f >= 0.0 { 1.0 / (f + 1.0) } else { 1.0 + f.abs() }
}

/// `calculate_probabilities`'s formula (finding 6): `prob(i) = 0.9*
/// (transformed(i)/max(transformed)) + 0.1`. `transformed` must be
/// non-empty with every entry `>0` (guaranteed by
/// [`abc_fitness_transform`]'s own range) -- returns values in `(0.1,
/// 1.0]`.
pub fn abc_probabilities(transformed: &[f64]) -> Vec<f64> {
    let max_fit = transformed.iter().cloned().fold(f64::MIN, f64::max);
    transformed.iter().map(|&f| 0.9 * (f / max_fit) + 0.1).collect()
}

/// `limit = SN*D` (finding 10) -- the dimension-scaling default this module
/// uses instead of `ABCorig.m`'s own dimension-independent literal
/// constant. Configurable via [`AbcOnlookerScout::from_params`]'s `"limit"`
/// override.
pub fn abc_limit(sn: usize, dim: usize) -> u64 {
    (sn * dim) as u64
}

/// Scout target selection (finding 8): `argmax(trials)`, ties broken to the
/// LAST (highest) index -- `ABCorig.m`'s `ind=find(trial==max(trial));
/// ind=ind(end)`.
pub fn abc_scout_target(trials: &[f64]) -> usize {
    let mut best = 0usize;
    for k in 1..trials.len() {
        if trials[k] >= trials[best] { best = k; }
    }
    best
}

/// Shared draw orchestration for one neighbor-move visit (finding 3): `j`
/// (dimension, one draw), then `k` (neighbour, rejection-sampled `!=i`),
/// then `phi = (rand-0.5)*2 in [-1,1]`. Used identically by the employed
/// generator and the onlooker scan (both artifacts' `Param2Change`/
/// `neighbour`/move-`rand` block is byte-identical between the two
/// phases).
fn abc_draw_move(sn: usize, dim: usize, i: usize, rng: &mut RngStream) -> (usize, usize, f64) {
    let j = rng.next_below(dim as u64) as usize;
    let k = loop {
        let cand = rng.next_below(sn as u64) as usize;
        if cand != i { break cand; }
    };
    let phi = (rng.next_f64() - 0.5) * 2.0;
    (j, k, phi)
}

fn floats(g: &Genotype) -> &Vec<f64> {
    match &g.blocks[0] { BlockValues::Float(x) => x, _ => unreachable!() }
}

fn bounds(space: &SearchSpace, dim: usize) -> (Vec<f64>, Vec<f64>) {
    match &space.blocks()[0] {
        Block::Float { lo, hi, n } => {
            debug_assert_eq!(*n, dim);
            (vec![*lo; dim], vec![*hi; dim])
        }
        _ => unreachable!("ABC components only support float blocks"),
    }
}

/// The employed-bee phase (see the module doc's "Phase-design
/// adjudication"): proposes exactly one candidate per food source, against
/// the FROZEN population snapshot `generate()` was called with.
///
/// // sezgi simplification: `ABCorig.m`'s own employed-phase loop updates
/// `Foods`/`Fitness` IN PLACE as it proceeds through `i=1..SN`, so a later
/// source's `neighbour` draw can land on an EARLIER source already improved
/// this same pass, reading its post-update position. This generator instead
/// proposes all `SN` candidates against the population as it stood at the
/// START of this call (every other non-in-place preset in this crate --
/// SCA/JAYA/GWO/WOA/DE/TLBO/etc. -- already works this way; only `ssa.rs`'s
/// followers and `fa.rs`'s double loop are genuinely in-place, and only
/// because neither needs a fresh evaluation mid-phase to decide anything).
/// This trade was CHOSEN, not forced: an `Adapter`-based, literally
/// in-place employed phase was available too -- `AbcOnlookerScout` below
/// is direct proof the shape works with equally honest `SN`-eval
/// accounting -- but reusing the crate's dominant `Generator`+`Replacer`
/// decomposition (and `ba.rs`'s acceptance-coupled-replacer idiom) was
/// preferred here. The onlooker phase (`AbcOnlookerScout`, below) is NOT
/// simplified this way -- it reproduces the reference's in-place chaining
/// exactly, since its own structural constraint (the scan's variable-
/// length, possibly-repeated visits) makes the `Generator`+`Replacer`
/// shape genuinely inexpressible there, not merely less convenient.
pub struct AbcEmployedGenerator;

impl AbcEmployedGenerator {
    pub fn from_params(_p: &serde_json::Value) -> Result<Self, ComponentError> {
        Ok(Self) // no tunable parameters -- j/k/phi are drawn, not configured
    }
}

impl Generator for AbcEmployedGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        let sn = pop.len();
        assert!(sn >= 2, "gen/abc-employed requires at least 2 food sources (pop_size={sn})");
        let dim = floats(&pop.individuals[0]).len();

        // Bootstrap trial counters on absence (generation 0 only) -- mirrors
        // mfo.rs's generator bootstrap-when-absent pattern.
        if !ctx.bb.contains("abc/trials") {
            ctx.bb.insert("abc/trials", vec![0.0_f64; sn]);
        }

        (0..sn).map(|i| {
            let x_i = floats(&pop.individuals[i]);
            let (j, k, phi) = abc_draw_move(sn, dim, i, ctx.rng);
            let x_k = floats(&pop.individuals[k]);
            let xs = abc_candidate(x_i, x_k, j, phi);
            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect()
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/abc-employed", SupportedBlocks::Only(vec!["float"]))
            .with_requires(vec![StateReq::of::<Vec<f64>>("abc/trials")])
            .with_min_pop(2)
    }
}

/// The new replacer this task adds (see the module doc's "Phase-design
/// adjudication" for the full rationale, `ba.rs`'s `BaLoudnessGreedy` T7-
/// M2d-3-style precedent): accepts offspring `i` over parent `i`
/// (same-index) iff `off_fit[i] < pop.fitness[i]` (finding 5's monotonicity
/// proof: equivalent to Karaboga's own `FitnessSol>Fitness(i)` test, no
/// fitness transform needed), resetting `abc/trials[i]` to `0` on accept or
/// incrementing it by `1` on reject.
pub struct AbcTrialGreedy;

impl Replacer for AbcTrialGreedy {
    fn replace(&self, pop: &mut Population, off_i: Vec<Genotype>, off_f: Vec<f64>, ctx: &mut Ctx) {
        let mut trials = ctx.bb.get::<Vec<f64>>("abc/trials")
            .expect("replace/abc-trial-greedy requires abc/trials -- pair with a component that provides it (e.g. gen/abc-employed's bootstrap)")
            .clone();
        for (i, (gi, fi)) in off_i.into_iter().zip(off_f).enumerate() {
            if i >= pop.len() { continue; }
            if fi < pop.fitness[i] {
                pop.individuals[i] = gi;
                pop.fitness[i] = fi;
                trials[i] = 0.0;
            } else {
                trials[i] += 1.0;
            }
        }
        ctx.bb.insert("abc/trials", trials);
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("replace/abc-trial-greedy", SupportedBlocks::All)
            .with_requires(vec![StateReq::of::<Vec<f64>>("abc/trials")])
            .with_provides(vec![StateReq::of::<Vec<f64>>("abc/trials")])
    }
}

/// The onlooker phase PLUS the scout phase (see the module doc's
/// "Phase-design adjudication" for why both live in one [`Adapter`]):
/// reproduces `ABCorig.m`'s repeated-scan-with-per-visit-accept/reject
/// (finding 7) and single-scout-per-cycle abandonment (findings 8-10)
/// exactly, including the reference's own literal in-place chaining (no
/// simplification needed on this side).
pub struct AbcOnlookerScout {
    limit_override: Option<u64>,
}

impl AbcOnlookerScout {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let limit_override = match p.get("limit") {
            None => None,
            Some(v) => Some(v.as_u64().ok_or_else(|| ComponentError::InvalidParams {
                kind: "adapter/abc-onlooker-scout".into(),
                reason: "limit must be a non-negative integer".into(),
            })?),
        };
        Ok(Self { limit_override })
    }
}

impl Adapter for AbcOnlookerScout {
    fn adapt(&self, pop: &mut Population, ctx: &mut Ctx) {
        let sn = pop.len();
        assert!(sn >= 2, "adapter/abc-onlooker-scout requires at least 2 food sources (pop_size={sn})");
        let dim = floats(&pop.individuals[0]).len();
        let (lo, hi) = bounds(ctx.space, dim);

        // calculate_probabilities (finding 6): computed ONCE, from the
        // POST-employed pop.fitness -- frozen for the whole scan below
        // (finding 7: ABCorig.m never recomputes `prob` mid-scan).
        let transformed: Vec<f64> = pop.fitness.iter().map(|&f| abc_fitness_transform(f)).collect();
        let prob = abc_probabilities(&transformed);

        // Working copy, mutated IN PLACE across the scan -- genuine
        // fidelity to ABCorig.m's own Foods/Fitness in-place updates (no
        // frozen-batch simplification needed here; see module doc).
        let mut positions: Vec<Vec<f64>> = pop.individuals.iter().map(|g| floats(g).clone()).collect();
        let mut fitness = pop.fitness.clone();
        let mut trials = ctx.bb.get::<Vec<f64>>("abc/trials")
            .expect("adapter/abc-onlooker-scout requires abc/trials -- pair with gen/abc-employed")
            .clone();

        let mut i = 0usize;
        let mut t = 0usize;
        let mut budget_exhausted = false;
        while t < sn && !budget_exhausted {
            let r = ctx.rng.next_f64();
            if r < prob[i] {
                t += 1;
                let (j, k, phi) = abc_draw_move(sn, dim, i, ctx.rng);
                let mut candidate = abc_candidate(&positions[i], &positions[k], j, phi);
                // Boundary clamp on the one changed dimension -- Adapters
                // bypass the engine's automatic per-stage boundary-repair
                // (which only wraps a Generator's returned offspring), so
                // this must be applied inline (finding: ABCorig.m clamps
                // BEFORE evaluating, `ind=find(sol<lb); sol(ind)=lb(ind)`).
                candidate[j] = candidate[j].clamp(lo[j], hi[j]);
                let cand_g = Genotype { blocks: vec![BlockValues::Float(candidate.clone())] };
                match ctx.eval.evaluate(std::slice::from_ref(&cand_g)) {
                    Ok(f) => {
                        if f[0] < fitness[i] {
                            positions[i] = candidate;
                            fitness[i] = f[0];
                            trials[i] = 0.0;
                        } else {
                            trials[i] += 1.0;
                        }
                    }
                    Err(_) => { budget_exhausted = true; }
                }
            }
            i = (i + 1) % sn;
        }

        for idx in 0..sn {
            pop.individuals[idx] = Genotype { blocks: vec![BlockValues::Float(positions[idx].clone())] };
            pop.fitness[idx] = fitness[idx];
        }
        ctx.bb.insert("abc/trials", trials.clone());

        if budget_exhausted { return; }

        // Scout phase (findings 8-10): at most ONE scout per cycle.
        let scout_idx = abc_scout_target(&trials);
        let limit = self.limit_override.unwrap_or_else(|| abc_limit(sn, dim));
        if trials[scout_idx] > limit as f64 {
            let new_pos: Vec<f64> = (0..dim).map(|d| lo[d] + (hi[d] - lo[d]) * ctx.rng.next_f64()).collect();
            let g = Genotype { blocks: vec![BlockValues::Float(new_pos)] };
            if let Ok(f) = ctx.eval.evaluate(std::slice::from_ref(&g)) {
                pop.individuals[scout_idx] = g;
                pop.fitness[scout_idx] = f[0];
                let mut trials2 = ctx.bb.get::<Vec<f64>>("abc/trials").unwrap().clone();
                trials2[scout_idx] = 0.0;
                ctx.bb.insert("abc/trials", trials2);
            }
        }
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("adapter/abc-onlooker-scout", SupportedBlocks::Only(vec!["float"]))
            .with_requires(vec![StateReq::of::<Vec<f64>>("abc/trials")])
            .with_provides(vec![StateReq::of::<Vec<f64>>("abc/trials")])
            .with_min_pop(2)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/abc-employed", |p| Ok(Box::new(AbcEmployedGenerator::from_params(p)?)));
    reg.register_replacer("replace/abc-trial-greedy", |_| Ok(Box::new(AbcTrialGreedy)));
    reg.register_adapter("adapter/abc-onlooker-scout", |p| Ok(Box::new(AbcOnlookerScout::from_params(p)?)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Problem, SphereShifted, Evaluator};
    use sezgi_core::state::Blackboard;

    fn g(xs: Vec<f64>) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs)] } }

    fn pop_nd(n: usize, dim: usize) -> Population {
        Population {
            individuals: (0..n).map(|i| g((0..dim).map(|d| (i * dim + d) as f64 * 0.37 - 1.0).collect())).collect(),
            fitness: (0..n).map(|i| (n - i) as f64).collect(),
        }
    }

    // ---- abc_dim_step / abc_candidate ----

    #[test]
    fn dim_step_matches_hand_computed_value() {
        // x=1.0, x_k=4.0, phi=0.5: 1.0 + 0.5*(1.0-4.0) = 1.0 - 1.5 = -0.5
        assert_eq!(abc_dim_step(1.0, 4.0, 0.5), -0.5);
    }

    #[test]
    fn dim_step_zero_phi_leaves_value_unchanged() {
        assert_eq!(abc_dim_step(3.0, 999.0, 0.0), 3.0);
    }

    #[test]
    fn dim_step_phi_one_moves_exactly_away_from_neighbor_by_the_full_gap() {
        // phi=1: x + 1*(x-x_k) = 2x - x_k.
        assert_eq!(abc_dim_step(2.0, 9.0, 1.0), 2.0 * 2.0 - 9.0);
    }

    #[test]
    fn only_the_chosen_dimension_changes() {
        // Finding 3: sol=Foods(i,:) copies everything, only Param2Change is
        // overwritten -- every other dimension must be BIT-IDENTICAL to x_i.
        let x_i = vec![1.0, 2.0, 3.0, 4.0];
        let x_k = vec![10.0, 20.0, 30.0, 40.0];
        for j in 0..4 {
            let c = abc_candidate(&x_i, &x_k, j, 0.7);
            for d in 0..4 {
                if d == j {
                    assert_eq!(c[d], abc_dim_step(x_i[d], x_k[d], 0.7));
                } else {
                    assert_eq!(c[d], x_i[d], "dimension {d} must be untouched when only dimension {j} is chosen");
                }
            }
        }
    }

    // ---- abc_fitness_transform: monotonicity, continuity, the negative-f handling ----

    #[test]
    fn fitness_transform_positive_domain_matches_formula() {
        assert_eq!(abc_fitness_transform(0.0), 1.0);
        assert_eq!(abc_fitness_transform(1.0), 0.5);
        assert_eq!(abc_fitness_transform(9.0), 0.1);
    }

    #[test]
    fn fitness_transform_negative_domain_matches_formula() {
        // The brief's flagged concern: does the reference handle f<0? Yes,
        // natively, via this branch (verified in both artifacts).
        assert_eq!(abc_fitness_transform(-1.0), 2.0);
        assert_eq!(abc_fitness_transform(-125.9497), 1.0 + 125.9497);
    }

    #[test]
    fn fitness_transform_continuous_at_the_f_zero_boundary() {
        // Both branches evaluate to exactly 1.0 at f=0 (the positive branch
        // is defined AT f=0; the negative branch's limit as f->0^- is also
        // 1.0) -- no discontinuity at the seam.
        let just_below = abc_fitness_transform(-1e-12);
        let at_zero = abc_fitness_transform(0.0);
        assert!((just_below - at_zero).abs() < 1e-10, "transform must be continuous across f=0: {just_below} vs {at_zero}");
    }

    #[test]
    fn fitness_transform_is_strictly_positive_everywhere() {
        for f in [-1000.0, -1.0, -1e-9, 0.0, 1e-9, 1.0, 1000.0] {
            assert!(abc_fitness_transform(f) > 0.0, "transform must be strictly positive at f={f}");
        }
    }

    #[test]
    fn fitness_transform_order_matches_raw_comparison_property() {
        // Finding 5's proof, hand-checked over same-sign AND mixed-sign
        // pairs: transform(f1) > transform(f2) iff f1 < f2 (strict
        // monotonic decrease is order-reversing). This is what licenses
        // AbcTrialGreedy to skip the transform entirely for accept/reject.
        let pairs = [
            (1.0, 2.0), (2.0, 1.0), (0.0, 5.0), (-1.0, 1.0), (-5.0, -2.0),
            (-2.0, -5.0), (-0.5, 0.5), (100.0, -100.0), (-125.9497, -1.0),
        ];
        for (f1, f2) in pairs {
            let raw_less = f1 < f2;
            let transform_greater = abc_fitness_transform(f1) > abc_fitness_transform(f2);
            assert_eq!(raw_less, transform_greater,
                "f1={f1}, f2={f2}: raw f1<f2 ({raw_less}) must match transform(f1)>transform(f2) ({transform_greater})");
        }
    }

    #[test]
    fn fitness_transform_ties_produce_ties() {
        assert_eq!(abc_fitness_transform(3.0), abc_fitness_transform(3.0));
        assert_eq!(abc_fitness_transform(-3.0), abc_fitness_transform(-3.0));
    }

    // ---- abc_probabilities ----

    #[test]
    fn probabilities_range_is_bounded_by_0_1_and_1_0() {
        let transformed = vec![0.1, 0.5, 1.0, 0.01];
        let prob = abc_probabilities(&transformed);
        for p in &prob {
            assert!(*p > 0.1 || (*p - 0.1).abs() < 1e-12, "prob must be >= 0.1, got {p}");
            assert!(*p <= 1.0 + 1e-12, "prob must be <= 1.0, got {p}");
        }
    }

    #[test]
    fn probabilities_max_entry_is_exactly_one() {
        let transformed = vec![0.2, 0.9, 0.3];
        let prob = abc_probabilities(&transformed);
        assert_eq!(prob[1], 1.0, "the entry equal to max(transformed) must get prob exactly 1.0");
    }

    #[test]
    fn probabilities_matches_hand_computed_value() {
        // transformed=[0.0909..(1/11), 0.5], max=0.5.
        // prob[0] = 0.9*(0.090909.../0.5)+0.1 = 0.9*0.181818...+0.1 = 0.263636...
        // prob[1] = 0.9*(0.5/0.5)+0.1 = 1.0
        let transformed = vec![1.0 / 11.0, 0.5];
        let prob = abc_probabilities(&transformed);
        assert!((prob[0] - 0.263636363636).abs() < 1e-9, "got {}", prob[0]);
        assert_eq!(prob[1], 1.0);
    }

    // ---- abc_limit ----

    #[test]
    fn limit_is_sn_times_dim() {
        assert_eq!(abc_limit(20, 5), 100);
        assert_eq!(abc_limit(1, 1), 1);
    }

    // ---- abc_scout_target: ties -> LAST index ----

    #[test]
    fn scout_target_picks_the_unique_max() {
        assert_eq!(abc_scout_target(&[1.0, 5.0, 2.0]), 1);
    }

    #[test]
    fn scout_target_ties_pick_the_last_index() {
        // Finding 8: ind(end) -- OPPOSITE of Population::best_index's
        // first-seen-wins convention used elsewhere in this crate.
        assert_eq!(abc_scout_target(&[3.0, 5.0, 5.0, 1.0]), 2);
        assert_eq!(abc_scout_target(&[5.0, 5.0]), 1);
    }

    #[test]
    fn scout_target_single_element() {
        assert_eq!(abc_scout_target(&[7.0]), 0);
    }

    // ---- AbcEmployedGenerator ----

    #[test]
    fn employed_determinism_same_seed_bit_identical() {
        let n = 6; let dim = 4;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let run = || {
            let mut evaluator = Evaluator::new(&p, 1000);
            let mut rng = RngStream::from_master(42, &[]);
            let mut bb = Blackboard::new();
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            AbcEmployedGenerator.generate(&pop, &mut ctx)
        };

        let a = run();
        let b = run();
        assert_eq!(a.len(), n);
        for (ga, gb) in a.iter().zip(b.iter()) {
            assert_eq!(floats(ga), floats(gb), "same seed must produce bit-identical offspring");
        }
    }

    #[test]
    fn employed_offspring_differ_from_parent_in_exactly_one_dimension() {
        // ABC-specific property test explicitly named in the task brief.
        let n = 5; let dim = 6;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let mut evaluator = Evaluator::new(&p, 1000);
        let mut rng = RngStream::from_master(3, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = AbcEmployedGenerator.generate(&pop, &mut ctx);

        #[allow(clippy::needless_range_loop)] // indexes two parallel arrays (pop.individuals and off) by the same i
        for i in 0..n {
            let parent = floats(&pop.individuals[i]);
            let child = floats(&off[i]);
            let changed: Vec<usize> = (0..dim).filter(|&d| child[d] != parent[d]).collect();
            assert_eq!(changed.len(), 1,
                "source {i}: expected exactly one changed dimension, got {changed:?} (parent={parent:?}, child={child:?})");
        }
    }

    #[test]
    fn employed_bootstraps_trials_to_zero_when_absent() {
        let n = 4; let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        assert!(!bb.contains("abc/trials"));
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        AbcEmployedGenerator.generate(&pop, &mut ctx);

        assert_eq!(ctx.bb.get::<Vec<f64>>("abc/trials").unwrap(), &vec![0.0; n]);
    }

    #[test]
    fn employed_does_not_re_bootstrap_existing_trials() {
        let n = 3; let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        bb.insert("abc/trials", vec![2.0, 5.0, 1.0]);
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        AbcEmployedGenerator.generate(&pop, &mut ctx);

        assert_eq!(ctx.bb.get::<Vec<f64>>("abc/trials").unwrap(), &vec![2.0, 5.0, 1.0],
            "generate() must not overwrite pre-existing trial counters");
    }

    #[test]
    fn employed_draw_order_matches_pinned_structure_via_raw_replay() {
        // Twin-stream raw-replay: per source, j (1 draw) then k (>=1
        // rejection-sampled draws) then phi (1 draw), in population order.
        let n = 6; let dim = 4;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);

        let mut rng = RngStream::from_master(13, &[]);
        let rng_before = rng.clone();
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut bb = Blackboard::new();
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let off = AbcEmployedGenerator.generate(&pop, &mut ctx);
            assert_eq!(off.len(), n);
        }

        let mut twin = rng_before;
        for i in 0..n {
            twin.next_below(dim as u64); // j
            loop {
                let cand = twin.next_below(n as u64) as usize;
                if cand != i { break; }
            } // k
            twin.next_f64(); // phi
        }
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "gen/abc-employed must consume exactly j, then k (rejection sampling), then phi, per source");
    }

    #[test]
    fn employed_min_pop_below_2_panics() {
        let n = 1; let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let pop = pop_nd(n, dim);
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            AbcEmployedGenerator.generate(&pop, &mut ctx)
        }));
        assert!(result.is_err(), "gen/abc-employed must reject pop_size < 2 at runtime as a backstop");
    }

    #[test]
    fn onlooker_scout_min_pop_below_2_panics() {
        // Two-layer idiom consistency (fix round 1, minor): the same
        // runtime assert() backstop gen/abc-employed has, on
        // adapter/abc-onlooker-scout too.
        let n = 1; let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let mut pop = pop_nd(n, dim);
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        bb.insert("abc/trials", vec![0.0]);
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            AbcOnlookerScout::from_params(&serde_json::json!({})).unwrap().adapt(&mut pop, &mut ctx)
        }));
        assert!(result.is_err(), "adapter/abc-onlooker-scout must reject pop_size < 2 at runtime as a backstop");
    }

    // ---- AbcTrialGreedy ----

    #[test]
    fn trial_greedy_accept_resets_reject_increments() {
        let dim = 1;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        bb.insert("abc/trials", vec![3.0, 4.0]);

        let mut pop = Population {
            individuals: vec![g(vec![1.0]), g(vec![2.0])],
            fitness: vec![10.0, 10.0],
        };
        // offspring 0 is better (accept, trial->0), offspring 1 is worse (reject, trial+1).
        let off_i = vec![g(vec![100.0]), g(vec![200.0])];
        let off_f = vec![5.0, 20.0];

        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        AbcTrialGreedy.replace(&mut pop, off_i, off_f, &mut ctx);

        assert_eq!(pop.fitness, vec![5.0, 10.0], "index 0 must accept (5.0<10.0), index 1 must reject (20.0 not < 10.0)");
        assert_eq!(pop.individuals[0], g(vec![100.0]));
        assert_eq!(pop.individuals[1], g(vec![2.0]), "rejected offspring must not overwrite the parent");
        assert_eq!(ctx.bb.get::<Vec<f64>>("abc/trials").unwrap(), &vec![0.0, 5.0]);
    }

    #[test]
    fn trial_greedy_tie_is_rejected_and_increments() {
        let dim = 1;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();
        bb.insert("abc/trials", vec![0.0]);
        let mut pop = Population { individuals: vec![g(vec![1.0])], fitness: vec![10.0] };
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        AbcTrialGreedy.replace(&mut pop, vec![g(vec![9.0])], vec![10.0], &mut ctx);
        assert_eq!(pop.individuals[0], g(vec![1.0]), "an exact tie (strict < fails) must be rejected");
        assert_eq!(ctx.bb.get::<Vec<f64>>("abc/trials").unwrap(), &vec![1.0]);
    }

    // ---- AbcOnlookerScout: determinism, in-place fidelity, scout trigger ----

    fn engine_evaluator(p: &SphereShifted, budget: u64) -> Evaluator<'_> { Evaluator::new(p, budget) }

    #[test]
    fn onlooker_determinism_same_seed_bit_identical() {
        let n = 5; let dim = 3;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();

        let run = || {
            let mut pop = pop_nd(n, dim);
            let mut evaluator = engine_evaluator(&p, 10_000);
            let mut rng = RngStream::from_master(21, &[]);
            let mut bb = Blackboard::new();
            bb.insert("abc/trials", vec![0.0; n]);
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            AbcOnlookerScout::from_params(&serde_json::json!({})).unwrap().adapt(&mut pop, &mut ctx);
            (pop.fitness, ctx.bb.get::<Vec<f64>>("abc/trials").unwrap().clone())
        };

        let (f1, t1) = run();
        let (f2, t2) = run();
        assert_eq!(f1, f2, "same seed must produce bit-identical fitness after onlooker+scout");
        assert_eq!(t1, t2, "same seed must produce bit-identical trial counters");
    }

    #[test]
    fn onlooker_visits_exactly_sn_and_charges_exactly_sn_evals_when_no_scout_fires() {
        let n = 4; let dim = 3;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let mut pop = pop_nd(n, dim);
        let mut evaluator = engine_evaluator(&p, 10_000);
        let mut rng = RngStream::from_master(5, &[]);
        let mut bb = Blackboard::new();
        bb.insert("abc/trials", vec![0.0; n]); // far from limit=n*dim=12, no scout will fire
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        AbcOnlookerScout::from_params(&serde_json::json!({})).unwrap().adapt(&mut pop, &mut ctx);
        assert_eq!(ctx.eval.used(), n as u64, "the onlooker phase alone must consume exactly SN evaluations (no scout)");
    }

    #[test]
    fn onlooker_gracefully_stops_on_budget_exhaustion_mid_scan() {
        let n = 6; let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let mut pop = pop_nd(n, dim);
        // Budget only allows 2 further evaluations.
        let mut evaluator = engine_evaluator(&p, 2);
        let mut rng = RngStream::from_master(9, &[]);
        let mut bb = Blackboard::new();
        bb.insert("abc/trials", vec![0.0; n]);
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            AbcOnlookerScout::from_params(&serde_json::json!({})).unwrap().adapt(&mut pop, &mut ctx);
        }));
        assert!(result.is_ok(), "budget exhaustion mid-scan must not panic");
        assert_eq!(ctx.eval.used(), 2, "no more than the available budget may be consumed");
    }

    #[test]
    fn onlooker_scout_fires_when_trial_strictly_exceeds_limit() {
        let n = 2; let dim = 1;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let limit = abc_limit(n, dim); // 2
        let mut pop = Population {
            individuals: vec![g(vec![1.0]), g(vec![2.0])],
            fitness: vec![10.0, 1.0],
        };
        let mut evaluator = engine_evaluator(&p, 10_000);
        let mut rng = RngStream::from_master(2, &[]);
        let mut bb = Blackboard::new();
        // Index 0's trial is already ABOVE limit; index 1's is at 0 --
        // after the onlooker scan (which cannot decrease trial[0] below
        // limit+1 even if it never improves), the scout must target index 0.
        bb.insert("abc/trials", vec![limit as f64 + 1.0, 0.0]);
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        AbcOnlookerScout::from_params(&serde_json::json!({})).unwrap().adapt(&mut pop, &mut ctx);

        let trials = ctx.bb.get::<Vec<f64>>("abc/trials").unwrap();
        assert_eq!(trials[0], 0.0, "the scouted source's trial counter must reset to 0");
        assert_ne!(pop.individuals[0], g(vec![1.0]), "the scouted source must be reinitialized");
    }

    #[test]
    fn onlooker_scout_does_not_fire_at_exactly_limit() {
        // Strict > (finding 9: MATLAB primary governs over the Python
        // port's >=): trial == limit exactly must NOT trigger the scout.
        //
        // Tightened (fix round 1, reviewer-requested): a loose
        // `trials[0] >= limit` bound cannot actually distinguish "the
        // scout correctly did not fire" from various other explanations
        // (e.g. an ordinary onlooker ACCEPT on index 0 also resets its
        // trial to 0.0, which such a bound wouldn't even have caught, or a
        // coincidental value that happens to still satisfy `>=`). Instead:
        // (a) full-state equality against the SAME independent reference
        // model (`onlooker_scout_reference_model`) the draw-order test
        // above uses, and (b) a DIRECT, unambiguous check that trials[0]
        // is exactly `limit` (not 0.0 -- the scout's only possible output
        // for a scouted slot -- and not left at some other elevated value)
        // AND that its position is bit-identical to its pre-call value
        // (proving index 0 was never even visited by the onlooker scan in
        // this fixture, let alone reinitialized by the scout).
        //
        // Reuses the draw-order test's exact fixture (seed 17, sn=2,
        // dim=1, positions [[10.0],[-1.0]], fitness [100.0,1.0]) but with
        // trials seeded at EXACTLY [limit, 0.0] (not limit+1.0) -- the
        // hand-traced scan (see the draw-order test's comment) never
        // visits-and-accepts index 0 at all in this fixture (both its
        // scan-check passes, at r=0.866751 and r=0.779688, reject against
        // prob[0]~=0.1178), so trials[0] ends the scan still at exactly
        // `limit`, and the scout check `limit > limit` is false.
        let sn = 2; let dim = 1;
        let lo = -1000.0; let hi = 1000.0;
        let p = SphereShifted::new(vec![0.0; dim], lo, hi);
        let space = p.space();
        let init_positions = vec![vec![10.0], vec![-1.0]];
        let init_fitness = vec![100.0, 1.0];
        let limit = abc_limit(sn, dim); // 2
        let init_trials = vec![limit as f64, 0.0];
        let mut pop = Population {
            individuals: init_positions.iter().cloned().map(g).collect(),
            fitness: init_fitness.clone(),
        };

        let mut evaluator = engine_evaluator(&p, 10_000);
        let mut rng = RngStream::from_master(17, &[]);
        let rng_before = rng.clone();
        let mut bb = Blackboard::new();
        bb.insert("abc/trials", init_trials.clone());

        let (actual_positions, actual_fitness, actual_trials) = {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            AbcOnlookerScout::from_params(&serde_json::json!({})).unwrap().adapt(&mut pop, &mut ctx);
            let xs: Vec<Vec<f64>> = pop.individuals.iter().map(|ind| floats(ind).clone()).collect();
            (xs, pop.fitness.clone(), ctx.bb.get::<Vec<f64>>("abc/trials").unwrap().clone())
        };

        let mut twin = rng_before;
        let (twin_positions, twin_fitness, twin_trials) =
            onlooker_scout_reference_model(&mut twin, init_positions, init_fitness, init_trials, lo, hi);

        assert_eq!(actual_positions, twin_positions, "final positions must match the independently-replayed twin");
        assert_eq!(actual_fitness, twin_fitness, "final fitness must match the independently-replayed twin");
        assert_eq!(actual_trials, twin_trials, "final trial counters must match the independently-replayed twin");

        assert_eq!(actual_trials[0], limit as f64,
            "trial==limit exactly must leave the source untouched by the scout (would be reset to exactly 0.0 if it had fired)");
        assert_eq!(actual_positions[0], vec![10.0],
            "index 0's position must be bit-identical to its pre-call value -- never visited by the onlooker scan, never scout-reinitialized");
    }

    #[test]
    fn onlooker_untouched_indices_leave_other_trials_alone() {
        // Scout-trigger property (brief): "others untouched" -- construct a
        // fixture where index 1 is far below the limit and verify it is
        // never reset to a fresh value by the scout logic (only index 0,
        // the argmax, can ever be scouted in one call).
        let n = 3; let dim = 1;
        let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
        let space = p.space();
        let limit = abc_limit(n, dim);
        let mut pop = Population {
            individuals: vec![g(vec![1.0]), g(vec![2.0]), g(vec![3.0])],
            fitness: vec![10.0, 1.0, 5.0],
        };
        let mut evaluator = engine_evaluator(&p, 10_000);
        let mut rng = RngStream::from_master(2, &[]);
        let mut bb = Blackboard::new();
        bb.insert("abc/trials", vec![limit as f64 + 1.0, 0.0, 0.0]);
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        AbcOnlookerScout::from_params(&serde_json::json!({})).unwrap().adapt(&mut pop, &mut ctx);

        // Only ONE scout per call (finding 8): index 1 and 2 (never the
        // argmax) must never be scout-reinitialized -- their trial values
        // may still move via ordinary onlooker visits, but exactly one
        // scout reset (to 0.0 on the argmax slot) is all that's permitted.
        let trials = ctx.bb.get::<Vec<f64>>("abc/trials").unwrap();
        assert_eq!(trials[0], 0.0, "index 0 (the argmax) must be the one scouted");
    }

    /// Independent "reference model" of `AbcOnlookerScout::adapt`, used by
    /// the twin-stream draw-order test and the tightened at-exactly-limit
    /// boundary test below: replays the SAME control-flow shape using ONLY
    /// the pure helpers the real adapter calls
    /// (`abc_probabilities`/`abc_draw_move`/`abc_candidate`/
    /// `abc_scout_target`/`abc_limit`) plus a hand-written objective
    /// matching `SphereShifted(shift=[0.0], ..)` exactly (`sum(x^2)`, no
    /// hidden randomness), so it runs entirely off a cloned `RngStream`
    /// with no `Ctx`/`Evaluator` at all -- if the real adapter's draw order
    /// or a branch condition ever drifts from the pinned structure, this
    /// model's independently-recomputed final state (and RNG-stream
    /// position) will disagree with the real adapter's.
    fn onlooker_scout_reference_model(
        rng: &mut RngStream,
        mut positions: Vec<Vec<f64>>,
        mut fitness: Vec<f64>,
        mut trials: Vec<f64>,
        lo: f64, hi: f64,
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let sn = positions.len();
        let dim = positions[0].len();
        let objective = |x: &[f64]| -> f64 { x.iter().map(|v| v * v).sum() };

        let transformed: Vec<f64> = fitness.iter().map(|&f| abc_fitness_transform(f)).collect();
        let prob = abc_probabilities(&transformed);

        let mut i = 0usize;
        let mut t = 0usize;
        while t < sn {
            let r = rng.next_f64(); // unconditional scan-check draw, every pass
            if r < prob[i] {
                t += 1;
                let (j, k, phi) = abc_draw_move(sn, dim, i, rng);
                let mut candidate = abc_candidate(&positions[i], &positions[k], j, phi);
                candidate[j] = candidate[j].clamp(lo, hi);
                let f = objective(&candidate);
                if f < fitness[i] {
                    positions[i] = candidate;
                    fitness[i] = f;
                    trials[i] = 0.0;
                } else {
                    trials[i] += 1.0;
                }
            }
            i = (i + 1) % sn;
        }

        let scout_idx = abc_scout_target(&trials);
        let limit = abc_limit(sn, dim);
        if trials[scout_idx] > limit as f64 {
            let new_pos: Vec<f64> = (0..dim).map(|_| lo + (hi - lo) * rng.next_f64()).collect();
            let f = objective(&new_pos);
            positions[scout_idx] = new_pos;
            fitness[scout_idx] = f;
            trials[scout_idx] = 0.0;
        }
        (positions, fitness, trials)
    }

    #[test]
    fn onlooker_scout_draw_order_matches_pinned_structure_via_raw_replay() {
        // Twin-stream raw-replay covering the FULL variable-length onlooker
        // scan (finding 7) plus the scout branch (findings 8-10) -- the
        // Task-6-M2d-3 technique, generalized for a DATA-DEPENDENT
        // structure via `onlooker_scout_reference_model` (see its doc).
        //
        // Fixture: sn=2, dim=1, bounds=[-1000,1000], positions
        // [[10.0],[-1.0]], fitness [100.0,1.0] -> transformed=[1/101,0.5],
        // max=0.5 -> prob=[0.9*(1/101)/0.5+0.1, 1.0]=[~0.11782, 1.0] --
        // source 1's prob is EXACTLY the max (1.0), so every scan-check
        // pass over index 1 unconditionally accepts; source 0's prob
        // (~0.1178) makes its scan-check draw accept only sometimes.
        // trials seeded at [limit+1.0, 0.0] (limit=SN*dim=2) so the scout
        // fires on index 0 after the scan.
        //
        // Hand-traced FULL per-pass sequence at seed 17 (captured by
        // running this exact fixture with instrumentation; the sequence
        // itself is DATA-derived from the seed+fixture, not hand-guessed --
        // reproduced here purely as documentation of what the twin below
        // independently re-derives from first principles, not as a
        // separate oracle):
        //   pass 1 (i=0): r=0.866751 >= prob[0] -> REJECT-SCAN (no draws, t stays 0)
        //   pass 2 (i=1): r=0.876100 <  prob[1] -> ACCEPT-SCAN (t=1);
        //     j=0, k=0, phi=-0.062620 -> candidate=[-0.311183], f=0.096835
        //     < fitness[1]=1.0 -> ACCEPT-MOVE (position/fitness updated, trial[1]=0)
        //   pass 3 (i=0): r=0.779688 >= prob[0] -> REJECT-SCAN (t stays 1)
        //   pass 4 (i=1): r=0.433437 <  prob[1] -> ACCEPT-SCAN (t=2, scan ends);
        //     j=0, k=0, phi=0.588841 -> candidate=[-6.382829], f=40.740506
        //     >= fitness[1]=0.096835 -> REJECT-MOVE (trial[1] += 1 -> 1.0)
        //   scout: argmax(trials)=[3.0,1.0] -> index 0; trials[0]=3.0 > limit=2
        //     -> SCOUT FIRES: 1 draw r=0.511349 -> new_pos=[22.697171],
        //     f=515.161557 (trial[0] reset to 0.0)
        //   final: positions=[[22.697171],[-0.311183]],
        //     fitness=[515.161557, 0.096835], trials=[0.0, 1.0]
        let sn = 2; let dim = 1;
        let lo = -1000.0; let hi = 1000.0;
        let p = SphereShifted::new(vec![0.0; dim], lo, hi);
        let space = p.space();
        let init_positions = vec![vec![10.0], vec![-1.0]];
        let init_fitness = vec![100.0, 1.0];
        let limit = abc_limit(sn, dim); // 2
        let init_trials = vec![limit as f64 + 1.0, 0.0];
        let mut pop = Population {
            individuals: init_positions.iter().cloned().map(g).collect(),
            fitness: init_fitness.clone(),
        };

        let mut evaluator = engine_evaluator(&p, 10_000);
        let mut rng = RngStream::from_master(17, &[]);
        let rng_before = rng.clone();
        let mut bb = Blackboard::new();
        bb.insert("abc/trials", init_trials.clone());

        let (actual_positions, actual_fitness, actual_trials) = {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            AbcOnlookerScout::from_params(&serde_json::json!({})).unwrap().adapt(&mut pop, &mut ctx);
            let xs: Vec<Vec<f64>> = pop.individuals.iter().map(|ind| floats(ind).clone()).collect();
            (xs, pop.fitness.clone(), ctx.bb.get::<Vec<f64>>("abc/trials").unwrap().clone())
        };

        let mut twin = rng_before;
        let (twin_positions, twin_fitness, twin_trials) =
            onlooker_scout_reference_model(&mut twin, init_positions, init_fitness, init_trials, lo, hi);

        assert_eq!(actual_positions, twin_positions, "final positions must match the independently-replayed twin");
        assert_eq!(actual_fitness, twin_fitness, "final fitness must match the independently-replayed twin");
        assert_eq!(actual_trials, twin_trials, "final trial counters must match the independently-replayed twin");
        assert_eq!(rng.next_f64(), twin.next_f64(),
            "after the full scan+scout, both streams must be at the identical position (draw order/count matches the pinned structure exactly)");
    }

    #[test]
    fn onlooker_limit_override_via_params() {
        let p_json = serde_json::json!({"limit": 0});
        let onlooker = AbcOnlookerScout::from_params(&p_json).unwrap();
        assert_eq!(onlooker.limit_override, Some(0));
    }

    // ---- End-to-end: two generations, hand-traced trial round-trip ----

    #[test]
    fn two_generations_trial_counters_round_trip() {
        let n = 3; let dim = 2;
        let p = SphereShifted::new(vec![0.0; dim], -100.0, 100.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100_000);
        let mut rng = RngStream::from_master(4, &[]);
        let mut bb = Blackboard::new();

        let mut pop = pop_nd(n, dim);

        // Generation 0: employed generate + trial-greedy replace + onlooker-scout adapt.
        let off0 = {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            AbcEmployedGenerator.generate(&pop, &mut ctx)
        };
        assert!(bb.contains("abc/trials"), "bootstrap must have run");
        let off0_fit: Vec<f64> = off0.iter().map(|ind| {
            let BlockValues::Float(xs) = &ind.blocks[0] else { unreachable!() };
            xs.iter().map(|x| x * x).sum()
        }).collect();
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            AbcTrialGreedy.replace(&mut pop, off0, off0_fit, &mut ctx);
        }
        let trials_after_employed_0 = bb.get::<Vec<f64>>("abc/trials").unwrap().clone();
        assert_eq!(trials_after_employed_0.len(), n);

        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            AbcOnlookerScout::from_params(&serde_json::json!({})).unwrap().adapt(&mut pop, &mut ctx);
        }
        let trials_after_gen0 = bb.get::<Vec<f64>>("abc/trials").unwrap().clone();

        // Generation 1: same sequence again -- trials must carry over (not reset to zero).
        let off1 = {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 1 };
            AbcEmployedGenerator.generate(&pop, &mut ctx)
        };
        let trials_before_replace_1 = bb.get::<Vec<f64>>("abc/trials").unwrap().clone();
        assert_eq!(trials_before_replace_1, trials_after_gen0,
            "generation 1's generate() must not reset or re-bootstrap existing trial counters");

        let off1_fit: Vec<f64> = off1.iter().map(|ind| {
            let BlockValues::Float(xs) = &ind.blocks[0] else { unreachable!() };
            xs.iter().map(|x| x * x).sum()
        }).collect();
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 1 };
            AbcTrialGreedy.replace(&mut pop, off1, off1_fit, &mut ctx);
        }
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 1 };
            AbcOnlookerScout::from_params(&serde_json::json!({})).unwrap().adapt(&mut pop, &mut ctx);
        }
        let trials_after_gen1 = bb.get::<Vec<f64>>("abc/trials").unwrap();
        assert_eq!(trials_after_gen1.len(), n, "trial vector length must remain SN across generations");
    }
}
