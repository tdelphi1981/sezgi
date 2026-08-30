use savvy::{savvy, savvy_err, OwnedListSexp, OwnedRealSexp, Sexp};
use sezgi_components::{presets, register_builtins};
use sezgi_core::component::Registry;
use sezgi_core::dist::Distribution;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_core::problem::Problem;
use sezgi_core::space::BlockValues;
use sezgi_core::spec::AlgorithmSpec;
use sezgi_problems::BbobProblem;

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
