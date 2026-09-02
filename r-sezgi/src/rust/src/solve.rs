use savvy::{
    ffi::SEXP, savvy, savvy_err, FunctionArgs, FunctionSexp, ListSexp, NumericSexp,
    OwnedIntegerSexp, OwnedListSexp, OwnedLogicalSexp, OwnedRealSexp, Sexp, StringSexp,
};
use sezgi_components::{presets, register_builtins};
use sezgi_core::component::Registry;
use sezgi_core::dist::Distribution;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_core::problem::Problem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};
use sezgi_core::spec::AlgorithmSpec;
use sezgi_problems::{BbobProblem, CatMatch, Cec2014, Cec2017, Cec2022, IntQuadratic, OneMax, Tsp};
use std::panic;

/// Casts a non-negative-checked, WHOLE-NUMBER-checked `f64` (as passed
/// from R, which has no native unsigned integer type) to `u64`, rejecting
/// negative, non-finite, or fractional values. Duplicated from the
/// identically-named private helpers in `session.rs` / `experiment.rs` /
/// `stats.rs` (no shared private cross-module import -- same "no shared
/// private crate imports" rule as `parse_distribution`'s duplication
/// between `solve.rs` and py-sezgi's `lib.rs`).
///
/// T11 fix round 1: every `sz_preset_*` builder and `sz_solve_bbob` in this
/// file used to cast their `f64` params directly (`pop_size as usize`,
/// `budget as u64`, `master_seed as u64`, `run_id as u64`), bypassing this
/// helper entirely and silently truncating a fractional value -- the one
/// place in the R binding surface the "no silent truncation path remains"
/// contract (task 11, step 3) had missed. Every such cast in this file now
/// routes through `f64_to_u64`/`f64_to_usize` below.
fn f64_to_u64(name: &str, x: f64) -> savvy::Result<u64> {
    if !x.is_finite() || x < 0.0 {
        return Err(savvy_err!(
            "{} must be a non-negative finite number, got {}",
            name,
            x
        ));
    }
    if x.fract() != 0.0 {
        return Err(savvy_err!("{} expected a whole number, got {}", name, x));
    }
    Ok(x as u64)
}

/// Same contract as [`f64_to_u64`], returning `usize` -- for `pop_size`/
/// `dim`-typed params, which are consumed as `usize` on the Rust side.
fn f64_to_usize(name: &str, x: f64) -> savvy::Result<usize> {
    f64_to_u64(name, x).map(|v| v as usize)
}

/// Same contract as [`f64_to_u64`], returning `u32` -- for `k`-typed params
/// (`sz_solve_cat_match`'s `k` / `sz_solve_mixed_diagnostic`'s `k_cat`).
/// Duplicated from the identically-named private helper in `problems.rs`
/// (M3-8 Task 10) -- same "no shared private cross-module import" rule that
/// helper's own doc already documents.
fn f64_to_u32(name: &str, x: f64) -> savvy::Result<u32> {
    f64_to_u64(name, x).map(|v| v as u32)
}

/// Casts a WHOLE-NUMBER-checked `f64` to `i64`, rejecting non-finite or
/// fractional values (UNLIKE [`f64_to_u64`], negative values are accepted --
/// needed for `sz_solve_int_quadratic`'s `lo`/`hi`, which `IntQuadratic::new`
/// takes as signed `i64` and which a diagnostic bowl commonly straddles
/// zero, e.g. `lo = -10, hi = 10`). M3-8 Task 10.
fn f64_to_i64(name: &str, x: f64) -> savvy::Result<i64> {
    if !x.is_finite() {
        return Err(savvy_err!("{} must be a finite number, got {}", name, x));
    }
    if x.fract() != 0.0 {
        return Err(savvy_err!("{} expected a whole number, got {}", name, x));
    }
    Ok(x as i64)
}

/// Builds a DE/rand/1/bin algorithm spec (uniform init, clamp boundary,
/// one-to-one-greedy replacement) as JSON, ready to pass to
/// `sz_solve_bbob()`.
///
/// @param pop_size Population size.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_de_rand_1(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::de_rand_1(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a DE/best/1/bin algorithm spec (uniform init, clamp boundary,
/// one-to-one-greedy replacement) as JSON, ready to pass to
/// `sz_solve_bbob()`.
///
/// @param pop_size Population size.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_de_best_1(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::de_best_1(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a jDE algorithm spec (self-adaptive F/CR DE) as JSON, ready to pass
/// to `sz_solve_bbob()`.
///
/// @param pop_size Population size.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_jde(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::jde(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a real-coded GA (SBX crossover, polynomial mutation) algorithm spec
/// as JSON, ready to pass to `sz_solve_bbob()`.
///
/// @param pop_size Population size.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_ga_real(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::ga_real(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a PSO (Clerc-Kennedy constriction) algorithm spec as JSON, ready to
/// pass to `sz_solve_bbob()`.
///
/// @param pop_size Population size (swarm size).
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_pso(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::pso(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a Grey Wolf Optimizer algorithm spec (Mirjalili, Mirjalili & Lewis
/// 2014 -- a labeled metaphor preset, see `crates/components/src/gwo.rs`'s
/// module doc for the tier note and citations) as JSON, ready to pass to
/// `sz_solve_bbob()`.
///
/// @param pop_size Population size (pack size). Canonical is 30.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_gwo(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::gwo(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a Whale Optimization Algorithm spec (Mirjalili & Lewis 2016 -- a
/// labeled metaphor preset, see `crates/components/src/woa.rs`'s module doc
/// for the tier note and citations) as JSON, ready to pass to
/// `sz_solve_bbob()`.
///
/// @param pop_size Population size (school size). Canonical is 30.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_woa(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::woa(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a Harmony Search algorithm spec (Geem, Kim & Loganathan 2001 -- a
/// labeled metaphor preset, see `crates/components/src/hs.rs`'s module doc
/// for the tier note and citations) as JSON, ready to pass to
/// `sz_solve_bbob()`.
///
/// @param pop_size Population size (Harmony Memory Size, HMS). Canonical is 30.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_harmony_search(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::harmony_search(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a Cuckoo Search algorithm spec (Yang & Deb 2009 -- a labeled
/// metaphor preset, see `crates/components/src/cs.rs`'s module doc for the
/// tier note, citation and pinned draw order) as JSON, ready to pass to
/// `sz_solve_bbob()`.
///
/// @param pop_size Population size (nest count). Canonical is 25.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_cuckoo_search(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::cuckoo_search(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a Grasshopper Optimisation Algorithm spec (Saremi, Mirjalili &
/// Lewis 2017 -- a labeled metaphor preset, see
/// `crates/components/src/goa.rs`'s module doc for the tier note, citation,
/// the IMPLEMENTER-VERIFY distance-normalization resolution and the
/// zero-RNG-draw arithmetic-order pin) as JSON, ready to pass to
/// `sz_solve_bbob()`.
///
/// @param pop_size Population size (swarm size). Canonical is 30.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_goa(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::goa(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a Sine Cosine Algorithm spec (Mirjalili 2016 -- a labeled
/// metaphor preset, see `crates/components/src/sca.rs`'s module doc for the
/// tier note, citation, the `SCA.m`-verified pinned draw order and the
/// mealpy-`OriginalSCA` replacer delta) as JSON, ready to pass to
/// `sz_solve_bbob()`.
///
/// @param pop_size Population size (number of search agents). Canonical is 30.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_sca(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::sca(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a JAYA spec (Rao 2016 -- a labeled metaphor preset, see
/// `crates/components/src/jaya.rs`'s module doc for the tier note,
/// citation, the primary-paper-verified worked-example reproduction, the
/// shared-per-dimension-per-generation `r1`/`r2` draw finding and the
/// greedy-replacement delta vs mealpy's misleadingly-named `OriginalJA`) as
/// JSON, ready to pass to `sz_solve_bbob()`.
///
/// @param pop_size Population size (candidate count). Canonical is 30.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_jaya(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::jaya(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds an MFO (Moth-Flame Optimization; Mirjalili 2015 -- a labeled
/// metaphor preset, see `crates/components/src/mfo.rs`'s module doc for the
/// tier note, citation, the verified `MFO.m` loop structure, the two subtle
/// draw/index deltas found vs the plan's sketch, and the blackboard
/// flame-memory design) spec as JSON, ready to pass to `sz_solve_bbob()`.
///
/// @param pop_size Population size (number of search agents). Canonical is 30.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_mfo(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::mfo(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds an SSA (Salp Swarm Algorithm; Mirjalili et al. 2017 -- a labeled
/// metaphor preset, see `crates/components/src/ssa.rs`'s module doc for the
/// tier note, citation, the verified `SSA.m` half-population leader/follower
/// split, the leader sign-branch pin, the verified in-place follower-chain
/// semantics, and the persisted-food-vs-current-pop-best delta) spec as
/// JSON, ready to pass to `sz_solve_bbob()`.
///
/// @param pop_size Population size (number of salps). Canonical is 30.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_ssa(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::ssa(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a Firefly Algorithm (Yang, X.-S., *Nature-Inspired Metaheuristic
/// Algorithms*, 2nd ed., Luniver Press, 2010 -- a labeled metaphor preset,
/// see `crates/components/src/fa.rs`'s module doc for the tier note,
/// citation, the verified `fa_ndim.m`/`ffa_move.m` loop structure, the
/// floored attractiveness formula, the closed-form `alpha` decay, and the
/// hybrid in-place-self/live-distance/frozen-target double-loop semantics)
/// spec as JSON, ready to pass to `sz_solve_bbob()`.
///
/// @param pop_size Population size (number of fireflies). Canonical is 25.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_firefly(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::firefly(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a Bat Algorithm spec (Yang, X.-S. 2010, NICSO -- a labeled
/// metaphor preset, see `crates/components/src/ba.rs`'s module doc for the
/// tier note, citation, the verified `bat_algorithm.m` loop structure, the
/// verified fixed-loudness/pulse-rate finding, the two composing
/// sign-inversion deltas in the frequency draw and velocity term, and the
/// design adjudication for the new `replace/bat-loudness-greedy`
/// acceptance-coupled replacer) as JSON, ready to pass to `sz_solve_bbob()`.
///
/// @param pop_size Population size (number of bats). Canonical is 30.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_bat(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::bat(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a Flower Pollination Algorithm spec (Yang, X.-S. 2012, UCNC -- a
/// labeled metaphor preset, see `crates/components/src/fpa.rs`'s module doc
/// for the tier note, citation, the verified `fpa_demo.m` loop structure,
/// the switch-branch orientation delta, the global-step sign delta reusing
/// `cs.rs`'s `cs_dim_step` verbatim, the local-step self-selection-not-
/// excluded finding, and the min_pop adjustment from 3 down to 2) as JSON,
/// ready to pass to `sz_solve_bbob()`.
///
/// @param pop_size Population size (number of flowers). Canonical is 25.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_fpa(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::fpa(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a Teaching-Learning-Based Optimization spec (Rao, Savsani &
/// Vakharia 2011, Computer-Aided Design -- a labeled metaphor preset, and
/// sezgi's FIRST multi-stage preset: two `[[stages]]` (teacher, then
/// learner) run in sequence every generation. See
/// `crates/components/src/tlbo.rs`'s module doc for the full provenance
/// extraction against Yarpiz's `tlbo.m` -- explicitly labeled third-party,
/// not Rao's own code -- the per-learner teaching-factor finding, the
/// unconditionally-distinct partner-selection finding, the min_pop
/// adjustment from 3 down to 2, and the "parameter-free" framing's
/// Črepinšek/Liu/Mernik (2012) counterpoint) as JSON, ready to pass to
/// `sz_solve_bbob()`.
///
/// @param pop_size Population size (class size). Canonical is 30.
/// @param budget Evaluation budget. A full generation costs `2 * pop_size`
///   evaluations (both stages evaluate).
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_tlbo(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::tlbo(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a Harris Hawks Optimization spec (Heidari, Mirjalili, Faris,
/// Aljarah, Mafarja & Chen 2019, Future Generation Computer Systems -- a
/// labeled metaphor preset, and the wave's most structurally complex one: a
/// multi-branch escape-energy tree whose progressive rapid-dive
/// sub-branches evaluate mid-`generate()`. See
/// `crates/components/src/hho.rs`'s module doc for the full provenance
/// extraction against the paper author's own `HHO.m`, the hard/soft
/// besiege mapping delta, the mean(X)/random-hawk in-place semantics, and
/// the prominent in-generator-evaluation eval-accounting design decision)
/// as JSON, ready to pass to `sz_solve_bbob()`.
///
/// @param pop_size Population size (hawk count). Canonical is 30.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_hho(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::hho(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds an Ant Lion Optimizer spec (Mirjalili 2015, Advances in
/// Engineering Software -- a labeled metaphor preset; see
/// `crates/components/src/alo.rs`'s module doc for the full provenance
/// extraction against the author's own `ALO.m`/`Random_walk_around_
/// antlion.m`/`RouletteWheelSelection.m`, the faithful-full-walk cost
/// decision, and the elitism design adjudication -- the antlion population
/// itself is the persisted memory via `replace/mu-plus-lambda`, no
/// blackboard state needed) as JSON, ready to pass to `sz_solve_bbob()`.
///
/// @param pop_size Population size (ant/antlion count). Canonical is 25.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_alo(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::alo(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds an Artificial Bee Colony spec (Karaboga 2005, TR-06 / Karaboga &
/// Basturk 2007, Journal of Global Optimization -- a labeled metaphor
/// preset; see `crates/components/src/abc.rs`'s module doc for the full
/// provenance extraction against the author's own `ABCorig.m` plus the
/// official `Python_ABC` port, the pop<->food-source convention
/// resolution, the fitness-transform monotonicity proof, and the
/// phase-design adjudication -- a SINGLE stage, not two symmetric stages
/// like `tlbo`: `gen/abc-employed` + the new `replace/abc-trial-greedy`,
/// plus the new `adapter/abc-onlooker-scout` folding the onlooker AND
/// scout phases together) as JSON, ready to pass to `sz_solve_bbob()`.
///
/// @param pop_size Population size (food-source count `SN`, NOT Karaboga's
///   colony size `NP=2*SN`). Canonical is 20.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_abc(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::abc(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a Gravitational Search Algorithm spec (Rashedi, Nezamabadi-pour &
/// Saryazdi 2009, Information Sciences -- a labeled metaphor preset, and the
/// wave's LAST stateful/blackboard algorithm; see
/// `crates/components/src/gsa.rs`'s module doc for the full provenance
/// extraction against the author's own `GSA.m`/`Gconstant.m`/
/// `massCalculation.m`/`Gfield.m`/`move.m`, the verified `M_i`-free force
/// delta, and the confirmation that GSA's own `Fbest`/`Lbest` never feed
/// back into the mechanism) as JSON, ready to pass to `sz_solve_bbob()`.
///
/// @param pop_size Population size (agent count). Canonical is 30.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_gsa(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::gsa(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a simulated annealing (Metropolis, geometric cooling) algorithm
/// spec as JSON, ready to pass to `sz_solve_bbob()`.
///
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_sa(budget: f64) -> savvy::Result<Sexp> {
    let json = presets::sa(f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a SHADE algorithm spec as JSON, ready to pass to `sz_solve_bbob()`.
///
/// @param pop_size Population size.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_shade(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::shade(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds an L-SHADE algorithm spec (population linearly reduced from
/// `18 * dim`) as JSON, ready to pass to `sz_solve_bbob()`.
///
/// @param dim Problem dimension (determines the initial population size).
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_lshade(dim: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::lshade(f64_to_usize("dim", dim)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a (mu/mu_w,lambda)-CMA-ES algorithm spec as JSON, ready to pass to
/// `sz_solve_bbob()`.
///
/// @param pop_size Population size (lambda).
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_cmaes(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::cmaes(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a CMA-ES with IPOP-style stagnation restarts algorithm spec as
/// JSON, ready to pass to `sz_solve_bbob()`.
///
/// @param dim Problem dimension (determines the initial population size).
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_cmaes_ipop(dim: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::cmaes_ipop(f64_to_usize("dim", dim)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a Nelder-Mead simplex algorithm spec as JSON, ready to pass to
/// `sz_solve_bbob()`.
///
/// @param dim Problem dimension (population size is fixed to `dim + 1`).
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_nelder_mead(dim: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::nelder_mead(f64_to_usize("dim", dim)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a random search algorithm spec as JSON, ready to pass to
/// `sz_solve_bbob()`.
///
/// @param pop_size Population size (resampled uniformly each generation).
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_random_search(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::random_search(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a permutation-space GA spec (tournament selection, order
/// crossover, swap mutation -- `gen/ga-perm` + `replace/mu-plus-lambda`) as
/// JSON, ready to pass to `sz_solve_tsp()`. Binds
/// [`sezgi_components::presets::ga_perm`] exactly -- same preset py-sezgi's
/// `sezgi.presets.ga_perm` binds (M3-3 Task 9).
///
/// @param pop_size Population size.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_ga_perm(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::ga_perm(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a Binary-space GA spec (tournament selection, uniform crossover,
/// bit-flip mutation -- `gen/ga-bin` + `replace/mu-plus-lambda`) as JSON,
/// ready to pass to `sz_solve_onemax()`. Binds
/// [`sezgi_components::presets::ga_bin`] exactly -- M3-8 Task 10, mirroring
/// py-sezgi's `sezgi.presets.ga_bin` (M3-8 Task 9).
///
/// @param pop_size Population size.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_ga_bin(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::ga_bin(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds an Int-space GA spec (tournament selection, SBX-style integer
/// crossover, polynomial-style integer mutation -- `gen/ga-int` +
/// `replace/mu-plus-lambda`) as JSON, ready to pass to
/// `sz_solve_int_quadratic()`. Binds
/// [`sezgi_components::presets::ga_int`] exactly -- M3-8 Task 10, mirroring
/// py-sezgi's `sezgi.presets.ga_int` (M3-8 Task 9).
///
/// @param pop_size Population size.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_ga_int(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::ga_int(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Builds a Categorical-space GA spec (tournament selection, uniform
/// crossover, random-reset mutation -- `gen/ga-cat` + `replace/mu-plus-lambda`)
/// as JSON, ready to pass to `sz_solve_cat_match()`. Binds
/// [`sezgi_components::presets::ga_cat`] exactly -- M3-8 Task 10, mirroring
/// py-sezgi's `sezgi.presets.ga_cat` (M3-8 Task 9).
///
/// @param pop_size Population size.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_ga_cat(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::ga_cat(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?).to_json();
    json.try_into()
}

/// Parses the `dist` string + flattened distribution params accepted by
/// `sz_preset_es_mu_plus_lambda_raw()` into a `Distribution`. Duplicated
/// (rather than shared via a private cross-crate import) from py-sezgi's
/// identically-named helper in `py-sezgi/src/lib.rs` -- same rule as
/// `ln_gamma`'s independent implementations, see that precedent.
fn parse_distribution(
    dist: &str,
    mean: f64,
    sigma: f64,
    loc: f64,
    scale: f64,
    alpha: f64,
    nu: f64,
) -> Result<Distribution, String> {
    match dist {
        "uniform" => Ok(Distribution::Uniform),
        "gaussian" => Ok(Distribution::Gaussian { mean, sigma }),
        "cauchy" => Ok(Distribution::Cauchy { loc, scale }),
        "levy" => Ok(Distribution::Levy { alpha }),
        "student_t" => Ok(Distribution::StudentT { nu }),
        "laplace" => Ok(Distribution::Laplace { loc, scale }),
        other => Err(format!(
            "unknown distribution '{other}' (expected uniform|gaussian|cauchy|levy|student_t|laplace)"
        )),
    }
}

/// Builds a (mu/mu_w,lambda)-ES algorithm spec (mutation step drawn from
/// `dist`) as JSON, ready to pass to `sz_solve_bbob()`.
///
/// This is the raw savvy-generated binding (required args only; savvy has no
/// way to express a non-`NULL`/string default in the generated signature).
/// The public R entry point with R-native defaults is the hand-written
/// wrapper `sz_preset_es_mu_plus_lambda()` in `R/presets.R`, which calls this
/// function -- same raw/wrapper pattern as `sz_run_experiment()` /
/// `sz_run_experiment_raw()` and `sz_stats_bayesian_signed_rank()` /
/// `sz_stats_bayesian_signed_rank_raw()`.
///
/// @param pop_size Population size.
/// @param budget Evaluation budget.
/// @param dist Mutation distribution: one of `"uniform"`, `"gaussian"`,
///   `"cauchy"`, `"levy"`, `"student_t"`, `"laplace"`.
/// @param mean Gaussian mean (used only when `dist = "gaussian"`).
/// @param sigma Gaussian std-dev (used only when `dist = "gaussian"`).
/// @param loc Cauchy/Laplace location (used only when `dist` is `"cauchy"`
///   or `"laplace"`).
/// @param scale Cauchy/Laplace scale (used only when `dist` is `"cauchy"`
///   or `"laplace"`).
/// @param alpha Levy stability parameter (used only when `dist = "levy"`).
/// @param nu Student-t degrees of freedom (used only when `dist =
///   "student_t"`).
/// @returns A character scalar with the algorithm spec as JSON.
/// @noRd
#[savvy]
fn sz_preset_es_mu_plus_lambda_raw(
    pop_size: f64,
    budget: f64,
    dist: &str,
    mean: f64,
    sigma: f64,
    loc: f64,
    scale: f64,
    alpha: f64,
    nu: f64,
) -> savvy::Result<Sexp> {
    let d = parse_distribution(dist, mean, sigma, loc, scale, alpha, nu)
        .map_err(|e| savvy_err!("{e}"))?;
    let json = presets::es_mu_plus_lambda(f64_to_usize("pop_size", pop_size)?, f64_to_u64("budget", budget)?, d).to_json();
    json.try_into()
}

/// Mixed Float+Int+Categorical+Binary scaffold problem (M3-8 Task 10),
/// living only in this crate (not `crates/problems`) -- the R mirror of
/// py-sezgi's identically-named, identically-shaped `MixedDiagnostic`
/// (`py-sezgi/src/lib.rs`, M3-8 Task 9). It exists purely to give
/// `sz_solve_mixed_diagnostic()` a target whose search space has one block
/// of each non-Permutation kind, so a mixed-space TOML `AlgorithmSpec` using
/// `gen/compound` (Task 5) can be run end to end through the NORMAL R solve
/// path. Mirrors `crates/components/src/compound.rs`'s own test-local
/// `MixedProblem` byte-for-byte: `Block::Float{-5,5,n_float}` +
/// `Block::Int{-5,5,n_int}` + `Block::Categorical{k_cat,n_cat}` +
/// `Block::Binary{n_bin}`, and `evaluate_batch` = (sum of the float block) +
/// (sum of the int block) + (count of categorical genes != 0) + (count of
/// `false` bits). Not a benchmark and not one of Task 5's brief-pinned
/// diagnostics -- test scaffolding only; `optimum()` is not exposed by any
/// `sz_solve_*` binding (this file's whole convention never surfaces
/// `Problem::optimum()` -- see `sz_solve_bbob`'s own result shape), so no
/// "no verified target" caveat is even reachable from R the way py-sezgi's
/// `Inner::Mixed => None` is.
struct MixedDiagnostic {
    space: SearchSpace,
}

impl MixedDiagnostic {
    fn new(n_float: usize, n_int: usize, k_cat: u32, n_cat: usize, n_bin: usize) -> Self {
        let space = SearchSpace::new(vec![
            Block::Float { lo: -5.0, hi: 5.0, n: n_float },
            Block::Int { lo: -5, hi: 5, n: n_int },
            Block::Categorical { k: k_cat, n: n_cat },
            Block::Binary { n: n_bin },
        ])
        .expect("Float{-5,5,..}/Int{-5,5,..} bounds are fixed and valid (lo < hi); \
                 Categorical/Binary have no bounds for SearchSpace::new to reject");
        Self { space }
    }
}

impl Problem for MixedDiagnostic {
    fn space(&self) -> &SearchSpace { &self.space }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
        pop.iter()
            .map(|g| {
                g.blocks.iter().fold(0.0, |f, b| f + match b {
                    BlockValues::Float(xs) => xs.iter().sum::<f64>(),
                    BlockValues::Int(xs) => xs.iter().sum::<i64>() as f64,
                    BlockValues::Cat(xs) => xs.iter().filter(|&&c| c != 0).count() as f64,
                    BlockValues::Bin(xs) => xs.iter().filter(|&&b| !b).count() as f64,
                    BlockValues::Perm(_) => 0.0,
                })
            })
            .collect()
    }
}

/// Converts one [`BlockValues`] block into its natural R `Sexp` type -- the R
/// mirror of py-sezgi's `block_values_to_py` helper (`py-sezgi/src/lib.rs`,
/// M3-8 Task 9's "typed result genotype" design decision; see this task's
/// own report for the full mapping table). Used by `sz_solve_onemax` /
/// `sz_solve_int_quadratic` / `sz_solve_cat_match` / `sz_solve_mixed_diagnostic`
/// ONLY -- `sz_solve_bbob`/`sz_solve_cec2022`/`sz_solve_cec2014`/
/// `sz_solve_cec2017` read `BlockValues::Float` directly (unchanged, still
/// byte-identical), and `sz_solve_tsp` reads `BlockValues::Perm` directly
/// with its own 1-based conversion (`problems.rs`'s module doc, "Index-
/// convention decision") -- neither needs this generic dispatch.
///
/// - `Float` -> a numeric double vector (`OwnedRealSexp`) -- BYTE-IDENTICAL
///   to `sz_solve_bbob`'s own `best_x` (no behavior change for existing
///   callers of this crate's Float-space solves).
/// - `Int` -> an integer vector (`OwnedIntegerSexp`), each `i64` gene cast to
///   `i32` (R has no native 64-bit integer type; every value this crate's
///   own `Int`-typed diagnostics/presets produce fits comfortably in `i32`).
/// - `Cat` -> an integer vector (`OwnedIntegerSexp`) of category INDICES
///   `0..k` (`u32` cast to `i32`), not labels -- `CatMatch`/`gen/ga-cat` have
///   no label concept, same choice py-sezgi's `Cat -> list[int]` makes.
/// - `Bin` -> a logical vector (`OwnedLogicalSexp`), R's native boolean, one
///   per bit -- matching Rust's own `Vec<bool>` 1:1, and mirroring this same
///   file's own `sz_mo_read_moa`'s `MoArchiveGenotype::Binary ->
///   OwnedLogicalSexp` precedent (`mo.rs`) rather than a 0/1 numeric
///   encoding.
/// - `Perm` -> an integer vector (`OwnedIntegerSexp`), each `u32` gene cast
///   to `i32`, 0-BASED (NOT `sz_solve_tsp`'s own 1-based tour convention --
///   this generic helper mirrors py-sezgi's `block_value_to_py`'s
///   `Perm -> list[int]` exactly, `py-sezgi/src/lib.rs`, which is also
///   0-based). Unreachable from this helper's original four callers (none
///   of onemax/int_quadratic/cat_match/mixed_diagnostic ever build a
///   Permutation block), but reachable from its FIFTH caller added in M4-2
///   Task 2 (the R-callable `Problem` bridge's `evaluate` shim, via
///   [`genotype_blocks_to_r_list`]): an R6 `Problem` subclass's `space()`
///   may legally include an `sz_permutation()` block.
fn block_values_to_r(bv: &BlockValues) -> savvy::Result<Sexp> {
    Ok(match bv {
        BlockValues::Float(xs) => OwnedRealSexp::try_from_slice(xs.as_slice())?.into(),
        BlockValues::Int(xs) => {
            OwnedIntegerSexp::try_from_iter(xs.iter().map(|&x| x as i32))?.into()
        }
        BlockValues::Cat(xs) => {
            OwnedIntegerSexp::try_from_iter(xs.iter().map(|&x| x as i32))?.into()
        }
        BlockValues::Bin(xs) => OwnedLogicalSexp::try_from_slice(xs.as_slice())?.into(),
        BlockValues::Perm(xs) => {
            OwnedIntegerSexp::try_from_iter(xs.iter().map(|&x| x as i32))?.into()
        }
    })
}

/// Converts a full result genotype (`result.best_x`, possibly multi-block)
/// into its R `Sexp` shape -- single block => a flat typed vector
/// ([`block_values_to_r`] directly); multi-block (reachable only via
/// `sz_solve_mixed_diagnostic`) => an unnamed list of per-block vectors, one
/// per `SearchSpace::blocks()` entry in order. Mirrors py-sezgi's own
/// single-block-vs-multi-block `best_x` dispatch in `solve()`
/// (`py-sezgi/src/lib.rs`) exactly.
fn genotype_to_r(blocks: &[BlockValues]) -> savvy::Result<Sexp> {
    if blocks.len() == 1 {
        block_values_to_r(&blocks[0])
    } else {
        let mut out = OwnedListSexp::new(blocks.len(), false)?;
        for (i, b) in blocks.iter().enumerate() {
            out.set_value(i, block_values_to_r(b)?)?;
        }
        Ok(out.into())
    }
}

fn registry() -> Registry {
    let mut r = Registry::new();
    register_builtins(&mut r);
    r
}

/// Runs an algorithm spec on a BBOB problem and returns the result.
///
/// @param spec_json Algorithm spec as JSON (e.g. from `sz_preset_de_rand_1()`).
/// @param fid BBOB function id (>= 1).
/// @param dim Problem dimension (>= 1).
/// @param instance BBOB instance id (>= 1).
/// @param master_seed Master RNG seed.
/// @param run_id Run id (mixed into the seed for independent replicate streams).
/// @returns A named list with `best_f` (double), `evals` (double), and
///   `best_x` (double vector) -- the best EVALUATED point (paired with
///   `best_f`). For most algorithms this always lies within the declared
///   domain. It is not guaranteed to for every algorithm: some (e.g. HHO)
///   charge raw, pre-boundary-repair trial points against the budget before
///   boundary repair, and such a point can become the reported best if it
///   happens to be the run's own minimum.
/// @export
#[savvy]
fn sz_solve_bbob(
    spec_json: &str,
    fid: i32,
    dim: i32,
    instance: i32,
    master_seed: f64,
    run_id: f64,
) -> savvy::Result<Sexp> {
    if fid < 1 {
        return Err(savvy_err!("fid must be >= 1"));
    }
    if dim < 1 {
        return Err(savvy_err!("dim must be >= 1"));
    }
    if instance < 1 {
        return Err(savvy_err!("instance must be >= 1"));
    }

    let spec = AlgorithmSpec::from_json(spec_json).map_err(|e| savvy_err!("{e}"))?;
    let reg = registry();

    let problem =
        BbobProblem::new(fid as u32, dim as usize, instance as u32).map_err(|e| savvy_err!("{e}"))?;

    let engine =
        Engine::from_spec(&spec, &reg, problem.space()).map_err(|e| savvy_err!("{e}"))?;
    let result = engine
        .run(
            &problem,
            RunConfig {
                master_seed: f64_to_u64("master_seed", master_seed)?,
                run_id: f64_to_u64("run_id", run_id)?,
            },
            None,
        )
        .map_err(|e| savvy_err!("{e}"))?;

    let BlockValues::Float(xs) = &result.best_x.blocks[0] else {
        return Err(savvy_err!("unexpected genotype"));
    };

    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "best_f", OwnedRealSexp::try_from_scalar(result.best_f)?)?;
    out.set_name_and_value(
        1,
        "evals",
        OwnedRealSexp::try_from_scalar(result.evals_used as f64)?,
    )?;
    out.set_name_and_value(2, "best_x", OwnedRealSexp::try_from_slice(xs.as_slice())?)?;

    Ok(out.into())
}

/// Runs an algorithm spec on a CEC 2022 (Kumar, Price, Mohamed, Hadi &
/// Suganthan 2021) function via [`Cec2022::new`] and returns the result --
/// M3-5 Task 4, closing the M3-3 gap (`docs/DECISIONS.md`'s M3-3 record,
/// ruling (g)): r-sezgi previously bound only direct evaluation
/// (`sz_cec2022_evaluate`/`sz_cec2022_f_star`), with no `solve()`-integrated
/// path, unlike py-sezgi's `sezgi.problems.cec2022(...)` + `sezgi.solve()`.
/// Mirrors `sz_solve_bbob`/`sz_solve_tsp` exactly (`Engine::from_spec` +
/// `engine.run` + result conversion): same `best_f`/`evals`/`best_x` shape,
/// not py-sezgi's own `solve()` dict shape (`best_f`/`best_x`/`evals_used`/
/// `iterations`) -- the established r-sezgi `sz_solve_*` convention governs
/// here too. See `Cec2022::new`'s own doc for the exact `fid`/`dim` domain.
///
/// @param spec_json Algorithm spec as JSON (e.g. from `sz_preset_shade()`).
/// @param fid CEC 2022 function id, `1..=12` (double, cast to `u32`).
/// @param dim Problem dimension, one of `2`, `10`, `20` (double, cast to
///   `usize`); `dim = 2` is additionally rejected for a hybrid function
///   (`fid` 6-8).
/// @param master_seed Master RNG seed.
/// @param run_id Run id (mixed into the seed for independent replicate streams).
/// @returns A named list with `best_f` (double), `evals` (double), and
///   `best_x` (double vector) -- the best EVALUATED point (paired with
///   `best_f`). Same caveat as `sz_solve_bbob()`: not every algorithm's
///   reported best is guaranteed to lie within the declared domain.
///
/// # Errors
/// A savvy error for `fid` outside `1..=12`, `dim` outside `{2,10,20}`,
/// `dim = 2` for a hybrid function, any [`sezgi_core::spec`] parse error, or
/// any [`sezgi_core::engine`] run error.
/// @export
#[savvy]
fn sz_solve_cec2022(
    spec_json: &str,
    fid: f64,
    dim: f64,
    master_seed: f64,
    run_id: f64,
) -> savvy::Result<Sexp> {
    let spec = AlgorithmSpec::from_json(spec_json).map_err(|e| savvy_err!("{e}"))?;
    let reg = registry();

    let problem = Cec2022::new(f64_to_u64("fid", fid)? as u32, f64_to_usize("dim", dim)?)
        .map_err(|e| savvy_err!("{e}"))?;

    let engine = Engine::from_spec(&spec, &reg, problem.space()).map_err(|e| savvy_err!("{e}"))?;
    let result = engine
        .run(
            &problem,
            RunConfig {
                master_seed: f64_to_u64("master_seed", master_seed)?,
                run_id: f64_to_u64("run_id", run_id)?,
            },
            None,
        )
        .map_err(|e| savvy_err!("{e}"))?;

    let BlockValues::Float(xs) = &result.best_x.blocks[0] else {
        return Err(savvy_err!("unexpected genotype"));
    };

    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "best_f", OwnedRealSexp::try_from_scalar(result.best_f)?)?;
    out.set_name_and_value(
        1,
        "evals",
        OwnedRealSexp::try_from_scalar(result.evals_used as f64)?,
    )?;
    out.set_name_and_value(2, "best_x", OwnedRealSexp::try_from_slice(xs.as_slice())?)?;

    Ok(out.into())
}

/// Runs an algorithm spec on a CEC 2014 (Liang, Qu & Suganthan 2013)
/// function via [`Cec2014::new`] and returns the result -- M3-6 Task 10,
/// mirroring `sz_solve_cec2022` exactly (`Engine::from_spec` + `engine.run`
/// + result conversion): same `best_f`/`evals`/`best_x` shape. See
/// [`Cec2014::new`]'s own doc for the exact `fid`/`dim` domain.
///
/// @param spec_json Algorithm spec as JSON (e.g. from `sz_preset_shade()`).
/// @param fid CEC 2014 function id, `1..=30` (double, cast to `u32`).
/// @param dim Problem dimension, one of `10`, `30` (double, cast to
///   `usize`).
/// @param master_seed Master RNG seed.
/// @param run_id Run id (mixed into the seed for independent replicate streams).
/// @returns A named list with `best_f` (double), `evals` (double), and
///   `best_x` (double vector) -- the best EVALUATED point (paired with
///   `best_f`). Same caveat as `sz_solve_bbob()`: not every algorithm's
///   reported best is guaranteed to lie within the declared domain.
///
/// # Errors
/// A savvy error for `fid` outside `1..=30`, `dim` outside `{10,30}`, any
/// [`sezgi_core::spec`] parse error, or any [`sezgi_core::engine`] run error.
/// @export
#[savvy]
fn sz_solve_cec2014(
    spec_json: &str,
    fid: f64,
    dim: f64,
    master_seed: f64,
    run_id: f64,
) -> savvy::Result<Sexp> {
    let spec = AlgorithmSpec::from_json(spec_json).map_err(|e| savvy_err!("{e}"))?;
    let reg = registry();

    let problem = Cec2014::new(f64_to_u64("fid", fid)? as u32, f64_to_usize("dim", dim)?)
        .map_err(|e| savvy_err!("{e}"))?;

    let engine = Engine::from_spec(&spec, &reg, problem.space()).map_err(|e| savvy_err!("{e}"))?;
    let result = engine
        .run(
            &problem,
            RunConfig {
                master_seed: f64_to_u64("master_seed", master_seed)?,
                run_id: f64_to_u64("run_id", run_id)?,
            },
            None,
        )
        .map_err(|e| savvy_err!("{e}"))?;

    let BlockValues::Float(xs) = &result.best_x.blocks[0] else {
        return Err(savvy_err!("unexpected genotype"));
    };

    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "best_f", OwnedRealSexp::try_from_scalar(result.best_f)?)?;
    out.set_name_and_value(
        1,
        "evals",
        OwnedRealSexp::try_from_scalar(result.evals_used as f64)?,
    )?;
    out.set_name_and_value(2, "best_x", OwnedRealSexp::try_from_slice(xs.as_slice())?)?;

    Ok(out.into())
}

/// Runs an algorithm spec on a CEC 2017 (Awad, Ali, Liang, Qu & Suganthan
/// 2016) function via [`Cec2017::new`] and returns the result -- M3-6 Task
/// 10, mirroring `sz_solve_cec2014` exactly. See [`Cec2017::new`]'s own doc
/// for the exact `fid`/`dim` domain.
///
/// @param spec_json Algorithm spec as JSON (e.g. from `sz_preset_shade()`).
/// @param fid CEC 2017 function id, `1` or `3..=30` (double, cast to `u32`);
///   `fid = 2` was officially withdrawn from the suite.
/// @param dim Problem dimension, one of `10`, `30` (double, cast to
///   `usize`).
/// @param master_seed Master RNG seed.
/// @param run_id Run id (mixed into the seed for independent replicate streams).
/// @returns A named list with `best_f` (double), `evals` (double), and
///   `best_x` (double vector) -- the best EVALUATED point (paired with
///   `best_f`). Same caveat as `sz_solve_bbob()`: not every algorithm's
///   reported best is guaranteed to lie within the declared domain.
///
/// # Errors
/// A savvy error for `fid` outside `{1} union {3..=30}`, `dim` outside
/// `{10,30}`, any [`sezgi_core::spec`] parse error, or any
/// [`sezgi_core::engine`] run error. `fid = 2` raises a dedicated error --
/// the Rust [`sezgi_problems::Cec2017Error::Withdrawn`] message is surfaced
/// VERBATIM.
/// @export
#[savvy]
fn sz_solve_cec2017(
    spec_json: &str,
    fid: f64,
    dim: f64,
    master_seed: f64,
    run_id: f64,
) -> savvy::Result<Sexp> {
    let spec = AlgorithmSpec::from_json(spec_json).map_err(|e| savvy_err!("{e}"))?;
    let reg = registry();

    let problem = Cec2017::new(f64_to_u64("fid", fid)? as u32, f64_to_usize("dim", dim)?)
        .map_err(|e| savvy_err!("{e}"))?;

    let engine = Engine::from_spec(&spec, &reg, problem.space()).map_err(|e| savvy_err!("{e}"))?;
    let result = engine
        .run(
            &problem,
            RunConfig {
                master_seed: f64_to_u64("master_seed", master_seed)?,
                run_id: f64_to_u64("run_id", run_id)?,
            },
            None,
        )
        .map_err(|e| savvy_err!("{e}"))?;

    let BlockValues::Float(xs) = &result.best_x.blocks[0] else {
        return Err(savvy_err!("unexpected genotype"));
    };

    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "best_f", OwnedRealSexp::try_from_scalar(result.best_f)?)?;
    out.set_name_and_value(
        1,
        "evals",
        OwnedRealSexp::try_from_scalar(result.evals_used as f64)?,
    )?;
    out.set_name_and_value(2, "best_x", OwnedRealSexp::try_from_slice(xs.as_slice())?)?;

    Ok(out.into())
}

/// Runs an algorithm spec on a TSPLIB VENDORED instance (`"berlin52"`,
/// `"eil51"`, `"st70"` -- via [`Tsp::vendored`]; UNLIKE `sz_tsp_load()`/
/// `sz_tsp_tour_length()` in `problems.rs`, raw TSPLIB text is not accepted
/// here -- mirrors py-sezgi's `sezgi.problems.tsp(name)`, which is likewise
/// vendored-only) and returns the result. Same output shape as
/// `sz_solve_bbob()` (`best_f`/`evals`/`best_x`), not py-sezgi's own
/// `solve()` dict shape (`best_f`/`best_x`/`evals_used`/`iterations`) -- the
/// established r-sezgi `sz_solve_*` convention governs here, not py-sezgi's
/// key names (see `problems.rs`'s module doc, "Index-convention decision",
/// for the general 1-based-vs-0-based rule this function's `best_x` also
/// follows).
///
/// @param spec_json Algorithm spec as JSON (e.g. from `sz_preset_ga_perm()`).
/// @param name A vendored TSPLIB instance name.
/// @param master_seed Master RNG seed.
/// @param run_id Run id (mixed into the seed for independent replicate streams).
/// @returns A named list with `best_f` (double), `evals` (double), and
///   `best_x` (double vector -- a 1-based permutation of `1:n_cities`, the
///   best EVALUATED tour, paired with `best_f`).
///
/// # Errors
/// A savvy error if `name` is not one of the three vendored instances, for
/// any [`sezgi_core::spec`] parse error, or any [`sezgi_core::engine`] run
/// error.
/// @export
#[savvy]
fn sz_solve_tsp(spec_json: &str, name: &str, master_seed: f64, run_id: f64) -> savvy::Result<Sexp> {
    let spec = AlgorithmSpec::from_json(spec_json).map_err(|e| savvy_err!("{e}"))?;
    let reg = registry();

    let problem = Tsp::vendored(name).map_err(|e| savvy_err!("{e}"))?;

    let engine = Engine::from_spec(&spec, &reg, problem.space()).map_err(|e| savvy_err!("{e}"))?;
    let result = engine
        .run(
            &problem,
            RunConfig {
                master_seed: f64_to_u64("master_seed", master_seed)?,
                run_id: f64_to_u64("run_id", run_id)?,
            },
            None,
        )
        .map_err(|e| savvy_err!("{e}"))?;

    let BlockValues::Perm(order) = &result.best_x.blocks[0] else {
        return Err(savvy_err!("unexpected genotype"));
    };
    // 0-based (Rust) -> 1-based (R) -- see `problems.rs`'s module doc,
    // "Index-convention decision".
    let best_x: Vec<f64> = order.iter().map(|&c| (c + 1) as f64).collect();

    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "best_f", OwnedRealSexp::try_from_scalar(result.best_f)?)?;
    out.set_name_and_value(
        1,
        "evals",
        OwnedRealSexp::try_from_scalar(result.evals_used as f64)?,
    )?;
    out.set_name_and_value(2, "best_x", OwnedRealSexp::try_from_slice(best_x.as_slice())?)?;

    Ok(out.into())
}

/// Runs an algorithm spec on [`OneMax`] (Goldberg 1989's classic
/// Binary-block GA diagnostic; `sezgi_problems::diagnostics::OneMax`) and
/// returns the result -- M3-8 Task 10, mirroring `sz_solve_tsp`/
/// `sz_solve_cec2022` exactly (`Engine::from_spec` + `engine.run`), except
/// `best_x` is now typed via [`genotype_to_r`] rather than assumed `Float`
/// (see that helper's own doc for the full type-mapping table). Pairs with
/// `sz_preset_ga_bin(...)`. Diagnostic only -- not a benchmark, see
/// `OneMax`'s own module doc.
///
/// @param spec_json Algorithm spec as JSON (e.g. from `sz_preset_ga_bin()`).
/// @param n_bits Length of the single `Block::Binary` (double, cast to
///   `usize`).
/// @param master_seed Master RNG seed.
/// @param run_id Run id (mixed into the seed for independent replicate streams).
/// @returns A named list with `best_f` (double), `evals` (double), and
///   `best_x` (a LOGICAL vector, one per bit -- see [`genotype_to_r`]'s doc).
///
/// # Errors
/// A savvy error for any [`sezgi_core::spec`] parse error, or any
/// [`sezgi_core::engine`] run error.
/// @export
#[savvy]
fn sz_solve_onemax(spec_json: &str, n_bits: f64, master_seed: f64, run_id: f64) -> savvy::Result<Sexp> {
    let spec = AlgorithmSpec::from_json(spec_json).map_err(|e| savvy_err!("{e}"))?;
    let reg = registry();

    let problem = OneMax::new(f64_to_usize("n_bits", n_bits)?);

    let engine = Engine::from_spec(&spec, &reg, problem.space()).map_err(|e| savvy_err!("{e}"))?;
    let result = engine
        .run(
            &problem,
            RunConfig {
                master_seed: f64_to_u64("master_seed", master_seed)?,
                run_id: f64_to_u64("run_id", run_id)?,
            },
            None,
        )
        .map_err(|e| savvy_err!("{e}"))?;

    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "best_f", OwnedRealSexp::try_from_scalar(result.best_f)?)?;
    out.set_name_and_value(
        1,
        "evals",
        OwnedRealSexp::try_from_scalar(result.evals_used as f64)?,
    )?;
    out.set_name_and_value(2, "best_x", genotype_to_r(&result.best_x.blocks)?)?;

    Ok(out.into())
}

/// Runs an algorithm spec on [`IntQuadratic`] (an Int-block quadratic bowl
/// around a fixed, deterministically-derived target;
/// `sezgi_problems::diagnostics::IntQuadratic`) and returns the result --
/// M3-8 Task 10, mirroring `sz_solve_onemax` exactly. Pairs with
/// `sz_preset_ga_int(...)`. Diagnostic only, see `IntQuadratic`'s own module
/// doc.
///
/// @param spec_json Algorithm spec as JSON (e.g. from `sz_preset_ga_int()`).
/// @param lo Inclusive lower bound of the single `Block::Int` (double, cast
///   to `i64`; may be negative).
/// @param hi Inclusive upper bound of the single `Block::Int` (double, cast
///   to `i64`; may be negative). Must be `> lo`.
/// @param n Length of the single `Block::Int` (double, cast to `usize`).
/// @param master_seed Master RNG seed.
/// @param run_id Run id (mixed into the seed for independent replicate streams).
/// @returns A named list with `best_f` (double), `evals` (double), and
///   `best_x` (an INTEGER vector -- see `genotype_to_r`'s doc).
///
/// # Errors
/// A savvy error if `lo >= hi`, for any [`sezgi_core::spec`] parse error, or
/// any [`sezgi_core::engine`] run error.
/// @export
#[savvy]
fn sz_solve_int_quadratic(
    spec_json: &str,
    lo: f64,
    hi: f64,
    n: f64,
    master_seed: f64,
    run_id: f64,
) -> savvy::Result<Sexp> {
    let lo_i = f64_to_i64("lo", lo)?;
    let hi_i = f64_to_i64("hi", hi)?;
    if lo_i >= hi_i {
        return Err(savvy_err!("lo ({}) must be < hi ({})", lo_i, hi_i));
    }
    let n_u = f64_to_usize("n", n)?;

    let spec = AlgorithmSpec::from_json(spec_json).map_err(|e| savvy_err!("{e}"))?;
    let reg = registry();

    let problem = IntQuadratic::new(lo_i, hi_i, n_u);

    let engine = Engine::from_spec(&spec, &reg, problem.space()).map_err(|e| savvy_err!("{e}"))?;
    let result = engine
        .run(
            &problem,
            RunConfig {
                master_seed: f64_to_u64("master_seed", master_seed)?,
                run_id: f64_to_u64("run_id", run_id)?,
            },
            None,
        )
        .map_err(|e| savvy_err!("{e}"))?;

    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "best_f", OwnedRealSexp::try_from_scalar(result.best_f)?)?;
    out.set_name_and_value(
        1,
        "evals",
        OwnedRealSexp::try_from_scalar(result.evals_used as f64)?,
    )?;
    out.set_name_and_value(2, "best_x", genotype_to_r(&result.best_x.blocks)?)?;

    Ok(out.into())
}

/// Runs an algorithm spec on [`CatMatch`] (a Categorical-block Hamming-
/// distance-to-target matching problem; `sezgi_problems::diagnostics::CatMatch`)
/// and returns the result -- M3-8 Task 10, mirroring `sz_solve_onemax`
/// exactly. Pairs with `sz_preset_ga_cat(...)`. Diagnostic only, see
/// `CatMatch`'s own module doc.
///
/// @param spec_json Algorithm spec as JSON (e.g. from `sz_preset_ga_cat()`).
/// @param k Category count per gene (double, cast to `u32`).
/// @param n Length of the single `Block::Categorical` (double, cast to `usize`).
/// @param seed Master seed the target category vector is drawn from (double,
///   cast to `u64`) -- UNLIKE `sz_solve_int_quadratic`'s `lo`/`hi`, this is
///   an explicit caller-supplied construction parameter, not derived.
/// @param master_seed Master RNG seed for the solve itself.
/// @param run_id Run id (mixed into the seed for independent replicate streams).
/// @returns A named list with `best_f` (double), `evals` (double), and
///   `best_x` (an INTEGER vector of category INDICES `0..k` -- see
///   `genotype_to_r`'s doc).
///
/// # Errors
/// A savvy error for any [`sezgi_core::spec`] parse error, or any
/// [`sezgi_core::engine`] run error.
/// @export
#[savvy]
fn sz_solve_cat_match(
    spec_json: &str,
    k: f64,
    n: f64,
    seed: f64,
    master_seed: f64,
    run_id: f64,
) -> savvy::Result<Sexp> {
    let spec = AlgorithmSpec::from_json(spec_json).map_err(|e| savvy_err!("{e}"))?;
    let reg = registry();

    let problem = CatMatch::new(
        f64_to_u32("k", k)?,
        f64_to_usize("n", n)?,
        f64_to_u64("seed", seed)?,
    );

    let engine = Engine::from_spec(&spec, &reg, problem.space()).map_err(|e| savvy_err!("{e}"))?;
    let result = engine
        .run(
            &problem,
            RunConfig {
                master_seed: f64_to_u64("master_seed", master_seed)?,
                run_id: f64_to_u64("run_id", run_id)?,
            },
            None,
        )
        .map_err(|e| savvy_err!("{e}"))?;

    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "best_f", OwnedRealSexp::try_from_scalar(result.best_f)?)?;
    out.set_name_and_value(
        1,
        "evals",
        OwnedRealSexp::try_from_scalar(result.evals_used as f64)?,
    )?;
    out.set_name_and_value(2, "best_x", genotype_to_r(&result.best_x.blocks)?)?;

    Ok(out.into())
}

/// Runs an algorithm spec on [`MixedDiagnostic`] (this file's own
/// Float+Int+Categorical+Binary mixed-space scaffold problem, see its own
/// doc) and returns the result -- M3-8 Task 10. Added SOLELY so
/// `gen/compound` (Task 5) is reachable end to end through the NORMAL R
/// solve path, proven with a mixed-space `AlgorithmSpec` authored as TOML
/// (this task's own test) -- UNLIKE its three siblings above (which take
/// `spec_json`, pairing with `sz_preset_ga_bin/ga_int/ga_cat`'s own
/// `.to_json()` presets), this function takes `spec_toml` directly and
/// parses it via [`AlgorithmSpec::from_toml`], the SAME entry point
/// `sz_run_experiment_raw`'s `ExperimentSpec::from_toml` already establishes
/// the "hand a raw TOML document straight to the Rust core" convention for
/// (`experiment.rs`) -- no R-side TOML library exists or is needed (r-sezgi
/// has none in `DESCRIPTION`'s `Suggests`; unlike py-sezgi's test, which
/// parses TOML with the stdlib's own `tomllib` into a dict before handing it
/// to `solve()`, R has no such stdlib module, so parsing happens in Rust
/// instead -- `AlgorithmSpec::from_toml`/`::from_json` are just two
/// serializations of the identical schema, so this is not a private
/// shortcut, only a different serialization entry point already used
/// elsewhere in this same file's crate). Mirrors `sz_solve_onemax`
/// otherwise, EXCEPT `run_id` is dropped (fixed to `0` internally) rather
/// than taken as an explicit parameter -- with `spec_toml` this function
/// already sits at 7 R-facing parameters; adding `run_id` would push it to 8
/// and trip this workspace's `clippy::too_many_arguments` gate (threshold
/// 7, this file's ONE pre-existing exception is
/// `sz_preset_es_mu_plus_lambda_raw`, not to be joined by a second). `run_id
/// = 0` matches how this scaffold is actually exercised (this task's own
/// TOML test, mirroring py-sezgi's `test_gen_compound_mixed_space_toml_spec_
/// solves_end_to_end`, calls `solve(spec, problem, master_seed=42)` with no
/// `run_id` override either -- `solve()`'s own Python signature defaults
/// `run_id=0`). Test scaffolding only -- NOT one of Task 5's brief-pinned
/// diagnostics, and (unlike onemax/int_quadratic/cat_match) has no verified
/// target: this file's own convention never surfaces `Problem::optimum()`
/// in a result anyway (see `sz_solve_bbob`'s own `best_f`/`evals`/`best_x`
/// shape), so that caveat needs no separate plumbing here.
///
/// @param spec_toml Algorithm spec as TOML text (e.g. a mixed-space
///   `gen/compound` document).
/// @param n_float Length of the `Block::Float{-5,5,..}` block (double, cast
///   to `usize`).
/// @param n_int Length of the `Block::Int{-5,5,..}` block (double, cast to
///   `usize`).
/// @param k_cat Category count per gene of the `Block::Categorical` block
///   (double, cast to `u32`).
/// @param n_cat Length of the `Block::Categorical` block (double, cast to
///   `usize`).
/// @param n_bin Length of the `Block::Binary` block (double, cast to `usize`).
/// @param master_seed Master RNG seed. `run_id` is fixed to `0` (see this
///   function's own doc for why it is not a parameter here).
/// @returns A named list with `best_f` (double), `evals` (double), and
///   `best_x` -- a MULTI-block genotype, surfaced as an unnamed list of 4
///   per-block vectors in `SearchSpace::blocks()` order (Float numeric, Int
///   integer, Categorical integer, Binary logical -- see `genotype_to_r`'s
///   doc).
///
/// # Errors
/// A savvy error for any [`sezgi_core::spec`] parse error, or any
/// [`sezgi_core::engine`] run error.
/// @export
#[savvy]
fn sz_solve_mixed_diagnostic(
    spec_toml: &str,
    n_float: f64,
    n_int: f64,
    k_cat: f64,
    n_cat: f64,
    n_bin: f64,
    master_seed: f64,
) -> savvy::Result<Sexp> {
    let spec = AlgorithmSpec::from_toml(spec_toml).map_err(|e| savvy_err!("{e}"))?;
    let reg = registry();

    let problem = MixedDiagnostic::new(
        f64_to_usize("n_float", n_float)?,
        f64_to_usize("n_int", n_int)?,
        f64_to_u32("k_cat", k_cat)?,
        f64_to_usize("n_cat", n_cat)?,
        f64_to_usize("n_bin", n_bin)?,
    );

    let engine = Engine::from_spec(&spec, &reg, problem.space()).map_err(|e| savvy_err!("{e}"))?;
    let result = engine
        .run(
            &problem,
            RunConfig {
                master_seed: f64_to_u64("master_seed", master_seed)?,
                run_id: 0,
            },
            None,
        )
        .map_err(|e| savvy_err!("{e}"))?;

    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "best_f", OwnedRealSexp::try_from_scalar(result.best_f)?)?;
    out.set_name_and_value(
        1,
        "evals",
        OwnedRealSexp::try_from_scalar(result.evals_used as f64)?,
    )?;
    out.set_name_and_value(2, "best_x", genotype_to_r(&result.best_x.blocks)?)?;

    Ok(out.into())
}

// ===========================================================================
// M4-2 Task 2: R-callable problem bridge -- lets an R6 `Problem` subclass
// instance (`R/problem.R`) be solved by the SAME `Engine` every other
// `sz_solve_*` binding uses, closing the long-standing "R callable-objective
// sessions" deferral (`docs/DECISIONS.md:839-845`). Mirrors py-sezgi's
// `solve_with_py_generator`/`from_callable_spaced` shape
// (`py-sezgi/src/lib.rs`), but with R's own GC-protection and
// error-propagation design -- see this milestone's research doc
// (`docs/superpowers/research/2026-09-02-r-class-front-door.md`) §C1-§C4.
// ===========================================================================

/// RAII guard over `savvy::protect::{insert,release}_from_preserved_list` --
/// the ONLY way this milestone's code holds a SEXP across a nested
/// `Rf_eval` (research doc §C1). A bare `FunctionSexp`/raw `SEXP` field is
/// NOT itself protected from R's GC: `FunctionSexp` is a newtype over a raw
/// pointer with no `Drop` impl, and while R protects a `.Call` argument for
/// the duration of that call, that protection does not extend across a
/// LATER nested `Rf_eval` inside `Engine::run`'s own generation loop.
/// `obj` is the raw pointer this guard protects (exposed so a caller can
/// hand it to a struct -- [`RProblem`] -- that must outlive this guard's
/// own stack frame while remaining bounded by it); `token` is the
/// continuation handle `release_from_preserved_list` needs to release it.
struct Preserved {
    obj: SEXP,
    token: SEXP,
}

impl Preserved {
    fn new(obj: SEXP) -> Self {
        let token = savvy::protect::insert_to_preserved_list(obj);
        Self { obj, token }
    }
}

impl Drop for Preserved {
    fn drop(&mut self) {
        savvy::protect::release_from_preserved_list(self.token);
    }
}

/// Panic payload carrying R's `R_UnwindProtect` continuation token
/// (research doc §C3). `savvy::Error` itself is NOT `Send` (its `Aborted`
/// variant holds a `SEXP`, and Rust does not derive `Send` per-variant), so
/// this thin wrapper is what actually crosses `panic::panic_any`/
/// `catch_unwind` -- the same role `AbortToken`/`throw_r_error` play in the
/// research doc's compile-verified probe.
struct AbortToken(SEXP);

/// # Safety
///
/// This token never crosses a real OS-thread boundary. `Engine::run` is
/// single-threaded (see [`RProblem`]'s own `# Safety` note below, points
/// 1-3), and [`run_with_r_bridge`]'s `catch_unwind` only unwinds Rust stack
/// frames WITHIN THAT SAME THREAD, between [`call_r_evaluate`]'s
/// `panic::panic_any` call site and `run_with_r_bridge`'s own call to
/// `catch_unwind` a few frames up the same call stack -- no R frame is ever
/// crossed by the unwind itself, because the R-side longjmp was already
/// intercepted at the `R_UnwindProtect` boundary INSIDE
/// `FunctionSexp::call` (research doc §C3), well before this panic is
/// thrown.
unsafe impl Send for AbortToken {}

/// Shared `catch_unwind` wrapper (research doc §C3) -- consumed by T2's
/// [`sz_solve_r_problem`] and reserved for T3's engine-hosted generator
/// bridge (`Generator::generate` has no `Result` return channel either, so
/// it needs this exact same wrapper around its own `engine.run(...)`
/// call). Catches an [`AbortToken`] panic payload -- thrown by
/// [`call_r_evaluate`] when `FunctionSexp::call` returns
/// `Err(savvy::Error::Aborted(tok))`, i.e. an R-side error/condition
/// propagating out of the callback -- and re-raises it as
/// `Err(savvy::Error::Aborted(tok))`; the generated C shim's own
/// `handle_result` then calls `R_ContinueUnwind(tok)` (savvy-bindgen
/// `src/codegen/c.rs:162-187`, `savvy::handle_error`,
/// `src/lib.rs:131-136`), re-raising the ORIGINAL R condition -- class,
/// message, call, custom condition classes and all -- UNCHANGED in the R
/// session. Any OTHER panic payload (a malformed `evaluate()` return
/// value, a core `assert!`/`unreachable!()`, or any other error converted
/// to a `String` at its own throw site, since `savvy::Error` is not
/// `Send`) is converted to a plain savvy error message instead.
fn run_with_r_bridge<T>(f: impl FnOnce() -> T) -> savvy::Result<T> {
    match panic::catch_unwind(panic::AssertUnwindSafe(f)) {
        Ok(v) => Ok(v),
        Err(payload) => match payload.downcast::<AbortToken>() {
            Ok(tok) => Err(savvy::Error::Aborted(tok.0)),
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "unknown panic in an R callback".to_string());
                Err(savvy_err!("{}", msg))
            }
        },
    }
}

/// Parses `blocks` -- the flat R `list` `.sz_space_to_blocks()` produces
/// (`R/spaces.R`: one entry per block, each a named `list` with a `type`
/// field plus that block's own numeric fields) -- into a `Vec<Block>`, the
/// same way [`MixedDiagnostic::new`] constructs `Block` values by hand,
/// just driven by R-supplied data instead of fixed literals.
/// `SearchSpace::new` (called by [`sz_solve_r_problem`] right after this)
/// is where an invalid `lo >= hi` bound is actually rejected -- mirroring
/// every other `sz_solve_*` binding's "bound checking happens at solve
/// time, not at builder time" convention (`R/spaces.R`'s own module doc).
///
/// Int block `lo`/`hi` are read as `f64` (not narrowed to `i32`) -- see
/// `.sz_block_to_list`'s own doc (`R/spaces.R`, this task's fix) for why:
/// `Block::Int`'s bounds are `i64` (`crates/core/src/space.rs`), and
/// [`f64_to_i64`] below performs the exact same f64->i64 cast/whole-number
/// check `sz_solve_int_quadratic`'s own `lo`/`hi` params already use.
fn blocks_from_r(blocks: &ListSexp) -> savvy::Result<Vec<Block>> {
    let mut out = Vec::with_capacity(blocks.len());
    for i in 0..blocks.len() {
        let elt = blocks
            .get_by_index(i)
            .ok_or_else(|| savvy_err!("blocks[[{}]] is missing", i + 1))?;
        let entry: ListSexp = elt.try_into()?;

        let get_num = |name: &str| -> savvy::Result<f64> {
            let s = entry
                .get(name)
                .ok_or_else(|| savvy_err!("blocks[[{}]] missing '{}'", i + 1, name))?;
            let num: NumericSexp = s.try_into()?;
            num.as_slice_f64()
                .first()
                .copied()
                .ok_or_else(|| savvy_err!("blocks[[{}]].{} is empty", i + 1, name))
        };

        let type_sexp = entry
            .get("type")
            .ok_or_else(|| savvy_err!("blocks[[{}]] missing 'type'", i + 1))?;
        let type_str: StringSexp = type_sexp.try_into()?;
        let kind = type_str
            .iter()
            .next()
            .ok_or_else(|| savvy_err!("blocks[[{}]].type is empty", i + 1))?;

        let block = match kind {
            "float" => Block::Float {
                lo: get_num("lo")?,
                hi: get_num("hi")?,
                n: f64_to_usize("n", get_num("n")?)?,
            },
            "int" => Block::Int {
                lo: f64_to_i64("lo", get_num("lo")?)?,
                hi: f64_to_i64("hi", get_num("hi")?)?,
                n: f64_to_usize("n", get_num("n")?)?,
            },
            "categorical" => Block::Categorical {
                k: f64_to_u32("k", get_num("k")?)?,
                n: f64_to_usize("n", get_num("n")?)?,
            },
            "binary" => Block::Binary {
                n: f64_to_usize("n", get_num("n")?)?,
            },
            "permutation" => Block::Permutation {
                n: f64_to_usize("n", get_num("n")?)?,
            },
            other => {
                return Err(savvy_err!(
                    "blocks[[{}]]: unknown block type '{}' \
                     (expected float|int|categorical|binary|permutation)",
                    i + 1,
                    other
                ));
            }
        };
        out.push(block);
    }
    Ok(out)
}

/// Converts a full genotype into the ALWAYS-a-list shape
/// `.sz_make_evaluate_shim()`'s R closure expects as its single argument --
/// one entry per block, in space order, each already typed by
/// [`block_values_to_r`]. UNLIKE [`genotype_to_r`] (which bare-unwraps a
/// single-block genotype for `sz_solve_*`'s own `best_x` return value),
/// this ALWAYS returns a list, even for a single block -- it is the R-side
/// shim (`R/problem.R`'s `.sz_make_evaluate_shim`), not this function, that
/// decides bare-vs-list for `evaluate(x)`'s own `x` argument, so the wire
/// shape built here needs to stay uniform for the shim to dispatch on
/// `length(blocks)`.
fn genotype_blocks_to_r_list(blocks: &[BlockValues]) -> savvy::Result<Sexp> {
    let mut out = OwnedListSexp::new(blocks.len(), false)?;
    for (i, b) in blocks.iter().enumerate() {
        out.set_value(i, block_values_to_r(b)?)?;
    }
    Ok(out.into())
}

/// Calls the R-side `evaluate` shim (`.sz_make_evaluate_shim(prob)`'s
/// return value, `R/problem.R`) with ONE individual's genotype, reshaped by
/// [`genotype_blocks_to_r_list`], and extracts the returned numeric scalar.
///
/// Every error path is thrown via `panic::panic_any`, NEVER by returning a
/// sentinel/empty value (research doc §C3's documented hazard: signalling
/// an error by returning no offspring/result is an INFINITE HANG, not an
/// error, because `Evaluator::evaluate` on an empty slice trivially
/// succeeds with zero cost). Specifically:
/// - `Err(savvy::Error::Aborted(tok))` (an R-side error/condition
///   propagating out of `prob$evaluate(x)` itself, via `FunctionSexp::
///   call`'s own `unwind_protect`) becomes `panic::panic_any(AbortToken
///   (tok))`, so [`run_with_r_bridge`]'s `catch_unwind` can re-raise the
///   ORIGINAL R condition unchanged.
/// - Any OTHER error (a malformed return value -- not a numeric scalar --
///   or any other savvy error at the FFI boundary itself) is thrown as a
///   plain `String` payload instead (`savvy::Error` is not `Send`, so its
///   message is captured into an owned `String` first); `run_with_r_bridge`
///   converts that into a plain savvy error message.
fn call_r_evaluate(f: &FunctionSexp, g: &Genotype) -> f64 {
    let blocks_sexp = match genotype_blocks_to_r_list(&g.blocks) {
        Ok(s) => s,
        Err(e) => panic::panic_any(e.to_string()),
    };
    let mut args = FunctionArgs::new();
    if let Err(e) = args.add("", blocks_sexp) {
        panic::panic_any(e.to_string());
    }
    match f.call(args) {
        Ok(eval_result) => {
            let sexp: Sexp = eval_result.into();
            let num: NumericSexp = match sexp.try_into() {
                Ok(n) => n,
                Err(e) => panic::panic_any(format!(
                    "Problem$evaluate(x) must return a numeric scalar: {e}"
                )),
            };
            let slice = num.as_slice_f64();
            if slice.len() != 1 {
                panic::panic_any(format!(
                    "Problem$evaluate(x) must return a numeric scalar, got length {}",
                    slice.len()
                ));
            }
            slice[0]
        }
        Err(savvy::Error::Aborted(tok)) => panic::panic_any(AbortToken(tok)),
        Err(e) => panic::panic_any(e.to_string()),
    }
}

/// A [`Problem`] implementation over an R6 `Problem` subclass instance,
/// evaluated via a callback into R (`evaluate`, `.sz_make_evaluate_shim`'s
/// return value -- `R/problem.R`). `evaluate` is a bare, UNPROTECTED
/// `SEXP` -- protection across `Engine::run`'s nested `Rf_eval` calls is
/// the CALLER's job: [`sz_solve_r_problem`] holds a [`Preserved`] guard
/// over the exact same pointer for the whole call, so this struct never
/// needs to (and, being `unsafe impl Send + Sync`, never runs a `Drop` the
/// engine could observe from a wrong thread anyway).
struct RProblem {
    evaluate: SEXP,
    space: SearchSpace,
}

/// # Safety
///
/// `SEXP` is `*mut c_void` (`savvy-ffi`) and therefore neither `Send` nor
/// `Sync` by default; `Problem: Send + Sync` (`crates/core/src/
/// problem.rs:3`) requires it anyway. This is sound because:
///
/// 1. `Engine::run` is single-threaded (`crates/core/src/engine.rs:
///    149-240`, zero threading deps in `crates/core/Cargo.toml`).
/// 2. The only parallel path (`sz_run_experiment` -> `crates/bench`) is
///    driven by TOML specs that structurally cannot name a synthetic R
///    kind or carry a SEXP.
/// 3. The registry, engine, and guard never escape the `.Call` frame.
unsafe impl Send for RProblem {}

/// # Safety
/// See the `Send` impl's `# Safety` note immediately above -- the same
/// three-point justification applies verbatim.
unsafe impl Sync for RProblem {}

impl Problem for RProblem {
    fn space(&self) -> &SearchSpace {
        &self.space
    }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
        // No closure is registered with the `Registry` in this bridge
        // (unlike Task 3's engine-hosted generator bridge, where
        // `reg.register_generator(name, move |_params| ...)` will need
        // `let held = &wrapper;` inside the closure to force whole-struct
        // capture -- research doc §C1's documented Rust-2021
        // disjoint-capture gotcha), so that gotcha does not arise HERE:
        // `self.evaluate` is read once, directly, with no `move` closure
        // capturing it disjointly.
        let f = FunctionSexp(self.evaluate);
        pop.iter().map(|g| call_r_evaluate(&f, g)).collect()
    }
}

/// R-callable Problem bridge (M4-2 Task 2) -- lets an R6 `Problem`
/// subclass instance (via `.sz_make_evaluate_shim(prob)`, `R/problem.R`)
/// be solved by the SAME engine every other `sz_solve_*` binding uses.
/// INTERNAL entry point -- not exported (`R/problem.R` is this task's sole
/// R-facing surface; there is no `sz_solve_problem()` convenience wrapper
/// yet -- a later task's wrapper-class `$run()` methods are expected to
/// call this directly, the same way `sz_preset_es_mu_plus_lambda_raw` is
/// called only from its own hand-written R wrapper).
///
/// Body, per research doc §C3: build a [`Preserved`] guard on `evaluate`
/// -> build [`RProblem`] -> [`registry()`] -> `Engine::from_spec` ->
/// [`run_with_r_bridge`]`(|| engine.run(...))` -> return the SAME 3-field
/// list every other `sz_solve_*` returns.
///
/// DEVIATION from this task's own brief sketch: there is no separate
/// `budget` parameter. Every OTHER `sz_solve_*` binding in this file takes
/// `spec_json` alone and reads the run's budget from `AlgorithmSpec::
/// termination.budget` (already baked in by whichever `sz_preset_*()`
/// built `spec_json`, e.g. `sz_preset_gwo(pop_size, budget)`) -- a second,
/// redundant `budget` input here would depart from that established
/// convention for no benefit (nothing in `Engine::run`'s own signature
/// even has a place to put it) -- see this task's own report for the full
/// rationale.
///
/// @param spec_json Algorithm spec as JSON (e.g. from `sz_preset_gwo()`).
/// @param blocks The flat block-list `.sz_space_to_blocks()` produces
///   (`R/spaces.R`) -- one named list per block (`type` + that block's own
///   numeric fields). Rebuilt into a `SearchSpace` here; `lo >= hi` is
///   rejected at THIS point (`SearchSpace::new`), not at `sz_float()`/
///   `sz_int()` construction time (Task 1 ruling).
/// @param evaluate `.sz_make_evaluate_shim(prob)`'s return value -- an R
///   closure of one argument, called once PER INDIVIDUAL with a `list` of
///   per-block typed vectors (Float->double, Int->integer,
///   Categorical->integer category indices, Binary->logical,
///   Permutation->integer 0-based -- see [`block_values_to_r`]'s doc for
///   the full table), one entry per block in space order; must return a
///   numeric scalar.
/// @param master_seed Master RNG seed.
/// @param run_id Run id (mixed into the seed for independent replicate
///   streams).
/// @returns A named list with `best_f` (double), `evals` (double), and
///   `best_x` (see [`genotype_to_r`]'s doc for the type mapping).
///
/// # Errors
/// A savvy error for an invalid `spec_json`/`blocks` shape, `lo >= hi` in
/// any Float/Int block, any [`sezgi_core::engine`] run error, OR -- via
/// [`run_with_r_bridge`] -- the R condition `evaluate` itself raised,
/// re-raised UNCHANGED in the R session.
/// @noRd
#[savvy]
fn sz_solve_r_problem(
    spec_json: &str,
    blocks: ListSexp,
    evaluate: FunctionSexp,
    master_seed: f64,
    run_id: f64,
) -> savvy::Result<Sexp> {
    let spec = AlgorithmSpec::from_json(spec_json).map_err(|e| savvy_err!("{e}"))?;
    let reg = registry();

    let block_vec = blocks_from_r(&blocks)?;
    let space = SearchSpace::new(block_vec).map_err(|e| savvy_err!("{e}"))?;

    let master_seed_u = f64_to_u64("master_seed", master_seed)?;
    let run_id_u = f64_to_u64("run_id", run_id)?;

    // Preserved guard (research doc §C1): the ONLY thing keeping
    // `evaluate` alive across `engine.run`'s nested `Rf_eval` calls --
    // `RProblem` itself holds only the bare, unprotected SEXP.
    let guard = Preserved::new(evaluate.0);
    let problem = RProblem {
        evaluate: guard.obj,
        space,
    };

    let engine =
        Engine::from_spec(&spec, &reg, problem.space()).map_err(|e| savvy_err!("{e}"))?;

    let run_result = run_with_r_bridge(|| {
        engine.run(
            &problem,
            RunConfig {
                master_seed: master_seed_u,
                run_id: run_id_u,
            },
            None,
        )
    })?;
    let result = run_result.map_err(|e| savvy_err!("{e}"))?;

    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "best_f", OwnedRealSexp::try_from_scalar(result.best_f)?)?;
    out.set_name_and_value(
        1,
        "evals",
        OwnedRealSexp::try_from_scalar(result.evals_used as f64)?,
    )?;
    out.set_name_and_value(2, "best_x", genotype_to_r(&result.best_x.blocks)?)?;

    Ok(out.into())
}
