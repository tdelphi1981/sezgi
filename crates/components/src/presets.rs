use sezgi_core::dist::Distribution;
use sezgi_core::spec::*;

fn comp(kind: &str, params: serde_json::Value) -> ComponentSpec {
    ComponentSpec { kind: kind.into(), params }
}

pub fn de_rand_1(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "de/rand/1/bin".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/de", serde_json::json!(
                {"strategy": "rand1", "f": 0.5, "cr": 0.9})),
            replacer: comp("replace/one-to-one-greedy", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

pub fn de_best_1(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "de/best/1/bin".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/de", serde_json::json!(
                {"strategy": "best1", "f": 0.5, "cr": 0.9})),
            replacer: comp("replace/one-to-one-greedy", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

pub fn jde(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "de/jde".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/de-jde", serde_json::json!(
                {"f_lower": 0.1, "f_upper": 0.9, "tau1": 0.1, "tau2": 0.1})),
            replacer: comp("replace/one-to-one-greedy", serde_json::json!({})),
            adapter: Some(comp("adapter/jde-commit", serde_json::json!({}))),
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

pub fn es_mu_plus_lambda(pop_size: usize, budget: u64, dist: Distribution) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "es/mu+lambda".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/step", serde_json::json!(
                {"dist": serde_json::to_value(&dist).unwrap(), "rate": 1.0})),
            replacer: comp("replace/mu-plus-lambda", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

pub fn ga_real(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "ga/real-sbx".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/ga-real", serde_json::json!(
                {"tournament_k": 2, "pc": 0.9, "eta_c": 15.0, "eta_m": 20.0})),
            replacer: comp("replace/mu-plus-lambda", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Permutation GA (M3-3 Task 4) -- a labeled preset validated on TSP
/// instances (see `crates/problems/src/tsp.rs`'s `Tsp`). Mirrors `ga_real`'s
/// own preset structure exactly (`init` + `boundary` + one `[[stages]]`
/// entry pairing a fused crossover+mutation `Generator` with
/// `replace/mu-plus-lambda`), swapped to the Permutation representation:
/// `init/perm-random` (Fisher-Yates, per `perm.rs`'s own doc) instead of
/// `init/uniform`, `gen/ga-perm` ([`crate::perm::GaPermGenerator`] -- the
/// FUSED OX-crossover + swap-mutation generator, per the controller ruling
/// recorded in `perm.rs`'s module doc "Composition decision": the
/// standalone `gen/ox`/`gen/perm-swap` pair, composed as two engine stages,
/// would yield a TLBO-style two-phase algorithm instead of the classic GA
/// loop) instead of `gen/ga-real`, and `replace/mu-plus-lambda` REUSED
/// as-is (same elitist merge-sort-truncate replacer `ga_real` uses --
/// `SupportedBlocks::All`, so it needs no permutation-specific variant).
/// `boundary/clamp` is likewise reused as-is: its `Block::Permutation`
/// branch is a documented no-op (`boundary.rs`: "cat/perm/bin: structurally
/// cannot go out of bounds"), so it is safe here purely for spec-shape
/// uniformity with every other preset in this crate, not because
/// permutations need repairing. `pop_size` is the population size (no
/// canonical value from a single source -- the caller picks it, same as
/// `ga_real`). `min_pop = 2` (tournament selection needs a population of at
/// least 2), enforced via `AlgorithmSpec::validate`.
pub fn ga_perm(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "ga-perm".into(), pop_size,
        init: comp("init/perm-random", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/ga-perm", serde_json::json!(
                {"tournament_k": 2, "pc": 0.8})),
            replacer: comp("replace/mu-plus-lambda", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Binary GA (M3-8 Task 2) -- mirrors `ga_perm`'s own preset structure
/// exactly (`init` + `boundary` + one `[[stages]]` entry pairing a fused
/// crossover+mutation `Generator` with `replace/mu-plus-lambda`), swapped to
/// the Binary representation: `init/uniform` (its own `Block::Binary`
/// branch already samples one fair coin flip per bit, per `init.rs`'s own
/// doc, so there is no dedicated `init/bin-random` the way there is
/// `init/perm-random` -- nothing would differ) instead of `init/perm-random`,
/// `gen/ga-bin` ([`crate::bin_ops::GaBinGenerator`] -- the FUSED two-point-
/// crossover + bit-flip-mutation generator, per `bin_ops.rs`'s module doc)
/// instead of `gen/ga-perm`, and `replace/mu-plus-lambda` reused as-is (same
/// elitist merge-sort-truncate replacer `ga_real`/`ga_perm` both use --
/// `SupportedBlocks::All`). `boundary/clamp` is likewise reused as-is: its
/// `Block::Binary` branch is a documented no-op (structurally cannot go out
/// of bounds), included purely for spec-shape uniformity with every other
/// preset in this crate. `pop_size` is the caller's choice (no canonical
/// value from a single source, same as `ga_real`/`ga_perm`). `min_pop = 2`
/// (tournament selection needs a population of at least 2), enforced via
/// `AlgorithmSpec::validate`.
pub fn ga_bin(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "ga-bin".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/ga-bin", serde_json::json!(
                {"tournament_k": 2, "p_c": 0.9})),
            replacer: comp("replace/mu-plus-lambda", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Integer GA (M3-8 Task 3) -- mirrors `ga_bin`'s/`ga_perm`'s own preset
/// structure exactly (`init` + `boundary` + one `[[stages]]` entry pairing a
/// fused crossover+mutation `Generator` with `replace/mu-plus-lambda`),
/// swapped to the Int representation: `init/uniform` (its own `Block::Int`
/// branch already samples `lo + next_below(hi-lo+1)`, per `init.rs`'s own
/// doc, so there is no dedicated `init/int-random` the way there is
/// `init/perm-random` -- nothing would differ) and `gen/ga-int`
/// ([`crate::int_ops::GaIntGenerator`] -- the FUSED SBX-crossover +
/// polynomial-mutation generator, both computed in float then rounded via
/// pymoo 0.6.2's `RoundingRepair` convention, per `int_ops.rs`'s module doc)
/// instead of `gen/ga-perm`/`gen/ga-bin`, and `replace/mu-plus-lambda`
/// reused as-is (same elitist merge-sort-truncate replacer `ga_real`/
/// `ga_perm`/`ga_bin` all use -- `SupportedBlocks::All`). `boundary/clamp`
/// is likewise reused as-is: its `Block::Int` branch already clamps
/// (`boundary.rs`), included both for spec-shape uniformity with every other
/// preset in this crate AND as a real backstop here (unlike the structurally
/// bound-proof binary/permutation cases) -- though `gen/ga-int`'s own
/// SBX/PM cores already clamp+round into `[lo,hi]` internally (see
/// `int_ops.rs`'s "PROVENANCE" doc section), so this is redundant-but-safe
/// belt-and-suspenders, not load-bearing. `eta_c`/`eta_m`/`p_c` are set
/// explicitly to `gen/ga-int`'s own verified pymoo defaults (`15.0`/`20.0`/
/// `0.9`) for documentation clarity, even though omitting them would resolve
/// to the same values. `pop_size` is the caller's choice (no canonical value
/// from a single source, same as `ga_real`/`ga_perm`/`ga_bin`). `min_pop = 2`
/// (tournament selection needs a population of at least 2), enforced via
/// `AlgorithmSpec::validate`.
pub fn ga_int(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "ga-int".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/ga-int", serde_json::json!(
                {"tournament_k": 2, "p_c": 0.9, "eta_c": 15.0, "eta_m": 20.0})),
            replacer: comp("replace/mu-plus-lambda", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Categorical GA (M3-8 Task 4) -- mirrors `ga_int`'s/`ga_bin`'s/`ga_perm`'s
/// own preset structure exactly (`init` + `boundary` + one `[[stages]]`
/// entry pairing a fused crossover+mutation `Generator` with
/// `replace/mu-plus-lambda`), swapped to the Categorical representation:
/// `init/uniform` (its own `Block::Categorical` branch already samples
/// `rng.next_below(k)`, per `init.rs`'s own `sample_uniform`, so there is
/// no dedicated `init/cat-random` the way there is `init/perm-random` --
/// nothing would differ) and `gen/ga-cat` ([`crate::cat_ops::GaCatGenerator`]
/// -- the FUSED uniform-crossover + random-reset-mutation generator, per
/// pymoo 0.6.2's `Choice` wiring, see `cat_ops.rs`'s module doc) instead of
/// `gen/ga-int`/`gen/ga-bin`/`gen/ga-perm`, and `replace/mu-plus-lambda`
/// reused as-is (same elitist merge-sort-truncate replacer every other `ga_*`
/// preset uses -- `SupportedBlocks::All`). `boundary/clamp` is likewise
/// reused as-is: its `Block::Categorical` branch is a documented no-op
/// (structurally cannot go out of bounds -- `gen/ga-cat`'s own cores only
/// ever copy existing valid category indices or resample fresh ones via
/// `rng.next_below(k)`, per `cat_ops.rs`'s "PROVENANCE" doc), included purely
/// for spec-shape uniformity with every other preset in this crate. `p_c`/
/// `p_m` are left at `gen/ga-cat`'s own verified pymoo defaults (`0.9`/
/// `1/n`) -- both omitted here (unlike `ga_bin`'s own preset, which sets
/// `p_c` explicitly and omits only `p_m`) because they resolve identically
/// whether stated or not. `pop_size` is the
/// caller's choice (no canonical value from a single source, same as
/// `ga_real`/`ga_perm`/`ga_bin`/`ga_int`). `min_pop = 2` (tournament
/// selection needs a population of at least 2), enforced via
/// `AlgorithmSpec::validate`.
pub fn ga_cat(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "ga-cat".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/ga-cat", serde_json::json!({"tournament_k": 2})),
            replacer: comp("replace/mu-plus-lambda", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

pub fn pso(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "pso/clerc-kennedy".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/pso", serde_json::json!(
                {"w": 0.7298, "c1": 1.49618, "c2": 1.49618})),
            replacer: comp("replace/pso-commit", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Grey Wolf Optimizer (Mirjalili, Mirjalili & Lewis 2014) -- a labeled
/// metaphor preset (see `gwo.rs`'s module doc for the tier note, citations
/// and pinned draw order). Uniform init, `boundary/clamp` (same as `pso`),
/// unconditional generational replacement (`replace/generational` -- GWO is
/// non-elitist by construction, same rationale as `pso`/`cma-es`).
/// `pop_size` is the pack size; canonical is 30 per the source paper.
/// `min_pop = 3` (alpha/beta/delta), enforced via `AlgorithmSpec::validate`.
pub fn gwo(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "gwo".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/gwo", serde_json::json!({})),
            replacer: comp("replace/generational", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Whale Optimization Algorithm (Mirjalili & Lewis 2016) -- a labeled
/// metaphor preset (see `woa.rs`'s module doc for the tier note, citations
/// and pinned draw order). Uniform init, `boundary/clamp` (same as `gwo`),
/// unconditional generational replacement (`replace/generational` -- WOA is
/// non-elitist by construction, same rationale as `gwo`/`pso`/`cma-es`).
/// `pop_size` is the school size; canonical is 30 per the source paper.
/// `min_pop = 2` (best-so-far plus at least one other whale), enforced via
/// `AlgorithmSpec::validate`.
pub fn woa(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "woa".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/woa", serde_json::json!({})),
            replacer: comp("replace/generational", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Harmony Search (Geem, Kim & Loganathan 2001) -- a labeled metaphor
/// preset (see `hs.rs`'s module doc for the tier note, citations and pinned
/// draw order). Uniform init, `boundary/clamp` (same as `gwo`/`woa` --
/// pitch adjustment can push a coordinate outside `[lo, hi]`),
/// `replace/worst-if-better` (added for this task -- see `replace.rs`'s doc
/// comment for why neither existing replacer kind fits an in-place
/// worst-replacement contract). `pop_size` is HMS (Harmony Memory Size);
/// canonical is 30 per the source paper. `min_pop = 1` (memory
/// consideration degenerates gracefully with a single harmony), enforced
/// via `AlgorithmSpec::validate`.
pub fn harmony_search(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "harmony".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/hs", serde_json::json!({})),
            replacer: comp("replace/worst-if-better", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Cuckoo Search (Yang & Deb 2009) -- a labeled metaphor preset (see
/// `cs.rs`'s module doc for the tier note, citation and pinned draw order).
/// Uniform init, `boundary/clamp` (same as `gwo`/`woa`/`harmony_search` --
/// a Lévy step can push a coordinate outside `[lo, hi]`),
/// `replace/one-to-one-greedy` (DE's kind, reused as-is for the greedy
/// same-index replacement -- sezgi simplification: reuses DE's same-index
/// greedy replacer (offspring i vs parent i) instead of the paper's
/// random-nest comparison; see `cs.rs`'s module doc for the full
/// rationale), plus the new `adapter/abandon-worst-fraction` (worst
/// `pa = 0.25` fraction re-randomized after replacement each iteration --
/// see `cs.rs`'s `AbandonWorstFraction` doc for the component-chain-placement
/// and RNG-stream rationale). `pop_size` is the nest count; canonical is 25
/// per the source paper. `min_pop = 2` (needs a best distinct from `i` to
/// move), enforced via `AlgorithmSpec::validate`.
pub fn cuckoo_search(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "cuckoo-search".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/cuckoo_levy", serde_json::json!({})),
            // sezgi simplification: reuses DE's same-index greedy replacer
            // (offspring i vs parent i) instead of the paper's random-nest
            // comparison -- see cs.rs's module doc for the full rationale.
            replacer: comp("replace/one-to-one-greedy", serde_json::json!({})),
            adapter: Some(comp("adapter/abandon-worst-fraction", serde_json::json!({}))),
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Grasshopper Optimisation Algorithm (Saremi, Mirjalili & Lewis 2017) -- a
/// labeled metaphor preset (see `goa.rs`'s module doc for the tier note,
/// citation, the IMPLEMENTER-VERIFY distance-normalization resolution and
/// the zero-RNG-draw arithmetic-order pin). Uniform init, `boundary/clamp`
/// (same as `gwo`/`woa`/`harmony_search`/`cuckoo_search` -- the swarm term
/// can push a coordinate outside `[lo, hi]`), unconditional generational
/// replacement (`replace/generational`, GWO's kind, reused as-is -- GOA is
/// non-elitist by construction, same rationale as `gwo`/`woa`/`pso`/
/// `cma-es`). `pop_size` is the swarm size; canonical is 30 per the source
/// paper. `min_pop = 2` (needs a best-so-far distinct from `i` for the
/// swarm interaction term to be meaningful), enforced via
/// `AlgorithmSpec::validate`.
pub fn goa(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "goa".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/goa", serde_json::json!({})),
            replacer: comp("replace/generational", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Sine Cosine Algorithm (Mirjalili 2016) -- a labeled metaphor preset (see
/// `sca.rs`'s module doc for the tier note, citation, verified-against-
/// `SCA.m` pinned draw order and the mealpy-`OriginalSCA` replacer delta).
/// Uniform init, `boundary/clamp` (same as `gwo`/`woa`/`goa`), unconditional
/// generational replacement (`replace/generational`, GWO's kind, reused as
/// -- `SCA.m`'s own reference loop overwrites every agent's position every
/// iteration with no per-agent fitness-improvement test). `pop_size` is the
/// number of search agents; canonical is 30 per the source paper. `min_pop
/// = 2` (needs a best-so-far distinct from `i` for the update to be
/// meaningful), enforced via `AlgorithmSpec::validate`.
pub fn sca(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "sca".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/sca", serde_json::json!({})),
            replacer: comp("replace/generational", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// JAYA (Rao 2016) -- a labeled metaphor preset (see `jaya.rs`'s module doc
/// for the tier note, citation, the primary-paper-verified worked-example
/// reproduction, the shared-per-dimension-per-generation `r1`/`r2` draw
/// finding and the greedy-replacement delta vs mealpy's misleadingly-named
/// `OriginalJA`). Uniform init, `boundary/clamp` (same as `gwo`/`woa`/`sca`),
/// greedy same-index replacement (`replace/one-to-one-greedy`, DE's kind,
/// reused as-is -- the paper's own Fig. 1 flowchart and worked Table 3
/// confirm per-candidate greedy acceptance). `pop_size` is the candidate
/// count; canonical is 30 per this wave's convention (the paper itself uses
/// a demonstration population of 5). `min_pop = 2` (needs a best AND a
/// worst distinct selection for the update to be meaningful), enforced via
/// `AlgorithmSpec::validate`.
pub fn jaya(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "jaya".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/jaya", serde_json::json!({})),
            replacer: comp("replace/one-to-one-greedy", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Moth-Flame Optimization (Mirjalili 2015) -- a labeled metaphor preset
/// (see `mfo.rs`'s module doc for the full provenance extraction, the
/// verified `MFO.m` loop structure, the two subtle draw/index deltas found
/// vs the plan's sketch, and the blackboard flame-memory design). Uniform
/// init, `boundary/clamp` (same as `gwo`/`woa`/`sca`/`jaya`), `gen/mfo`
/// paired with `adapter/mfo-flame-update` (the flame memory's canonical
/// owner -- merge-sort-truncate each generation, per `MFO.m`), generational
/// replacement (`replace/generational`, reused as-is -- the flames, not the
/// moth population, carry the elitism). `pop_size` is the number of search
/// agents; canonical is 30 per the source paper. `min_pop = 2`, enforced
/// via `AlgorithmSpec::validate`.
pub fn mfo(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "mfo".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/mfo", serde_json::json!({})),
            replacer: comp("replace/generational", serde_json::json!({})),
            adapter: Some(comp("adapter/mfo-flame-update", serde_json::json!({}))),
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Salp Swarm Algorithm (Mirjalili et al. 2017) -- a labeled metaphor
/// preset (see `ssa.rs`'s module doc for the full provenance extraction,
/// the verified `SSA.m` half-population leader/follower split, the leader
/// sign-branch pin, the verified in-place follower-chain semantics, and the
/// persisted-food-vs-current-pop-best delta). Uniform init, `boundary/clamp`
/// (same as `gwo`/`woa`/`sca`/`jaya`/`mfo`), unconditional generational
/// replacement (`replace/generational`, reused as-is -- `SSA.m`'s own
/// reference loop overwrites every salp's position every iteration with no
/// per-agent fitness-improvement test). `pop_size` is the number of salps;
/// canonical is 30 per the source paper. `min_pop = 2` (one leader, one
/// follower, the minimal meaningful split), enforced via
/// `AlgorithmSpec::validate`.
pub fn ssa(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "ssa".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/ssa", serde_json::json!({})),
            replacer: comp("replace/generational", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Firefly Algorithm (Yang, X.-S., *Nature-Inspired Metaheuristic
/// Algorithms*, 2nd ed., Luniver Press, 2010) -- a labeled metaphor preset
/// (see `fa.rs`'s module doc for the full provenance extraction, the
/// verified `fa_ndim.m`/`ffa_move.m` loop structure, the floored
/// attractiveness formula and its non-vanishing `gamma -> infinity` limit,
/// the closed-form `alpha` decay, and the hybrid in-place-self/live-
/// distance/frozen-target double-loop semantics). Uniform init,
/// `boundary/clamp` (same as `gwo`/`woa`/`sca`/`jaya`/`mfo`/`ssa`),
/// generational replacement (`replace/generational`, reused as-is --
/// `fa_ndim.m`'s own reference loop overwrites the whole population every
/// generation with no per-firefly fitness-improvement test). `pop_size` is
/// the number of fireflies; canonical is 25 per this wave's convention (the
/// source's own demo uses 20). `min_pop = 2`, enforced via
/// `AlgorithmSpec::validate`. `O(pop_size^2 * dim)` per generation -- see
/// `fa.rs`'s "Cost note".
pub fn firefly(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "firefly".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/fa", serde_json::json!({})),
            replacer: comp("replace/generational", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Bat Algorithm (Yang, X.-S. 2010, NICSO) -- a labeled metaphor preset (see
/// `ba.rs`'s module doc for the full provenance extraction, the verified
/// `bat_algorithm.m` loop structure, the verified fixed-loudness/pulse-rate
/// finding that rules out the plan's sketched `A_i`/`r_i` decay dynamics,
/// the two composing sign-inversion deltas in the frequency draw and
/// velocity term, and the design adjudication for the new
/// `replace/bat-loudness-greedy` acceptance-coupled replacer). Uniform init,
/// `boundary/clamp` (same as `gwo`/`woa`/`sca`/`jaya`/`mfo`/`ssa`/`firefly`),
/// `gen/ba` (owns the persisted `ba/velocity` blackboard state, no separate
/// adapter -- same self-owning shape as `pso`) paired with the new
/// `replace/bat-loudness-greedy` (fixed `loudness = BA_A0 = 0.5`, the
/// verified source's own default). `pop_size` is the number of bats;
/// canonical is 30 per this wave's convention (the source's own demo uses
/// 20). `min_pop = 2`, enforced via `AlgorithmSpec::validate`.
pub fn bat(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "bat".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/ba", serde_json::json!({})),
            replacer: comp("replace/bat-loudness-greedy", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Flower Pollination Algorithm (Yang, X.-S. 2012, UCNC) -- a labeled
/// metaphor preset (see `fpa.rs`'s module doc for the full provenance
/// extraction, the verified `fpa_demo.m` loop structure, the switch-branch
/// orientation delta -- `rand>p` selects global, not `u<p` -- the
/// global-step sign delta reusing `cs.rs`'s `cs_dim_step` verbatim, the
/// local-step `j,k` self-selection-not-excluded finding, and the
/// min_pop adjustment from the plan's sketched 3 down to 2). Uniform init,
/// `boundary/clamp` (same as `cs`/`gwo`/`woa`/`sca`/`jaya`/`mfo`/`ssa`/
/// `firefly`/`bat`), greedy same-index replacement
/// (`replace/one-to-one-greedy`, DE's kind, reused as-is -- the verified
/// source's own `if (Fnew<=Fitness(i))` acceptance). `pop_size` is the
/// flower/pollen-gamete count; canonical is 25 per the source's demo.
/// `min_pop = 2` (a HARD requirement -- `pick_two_distinct`'s rejection
/// loop for `k` never terminates with `n < 2`), enforced via
/// `AlgorithmSpec::validate`.
pub fn fpa(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "fpa".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/fpa", serde_json::json!({})),
            replacer: comp("replace/one-to-one-greedy", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Teaching-Learning-Based Optimization (Rao, Savsani & Vakharia 2011,
/// Computer-Aided Design) -- a labeled metaphor preset, and sezgi's FIRST
/// **multi-stage** preset: two `[[stages]]` run in sequence every
/// generation, `gen/tlbo-teacher` then `gen/tlbo-learner` (see `tlbo.rs`'s
/// module doc for the full provenance extraction against Yarpiz's `tlbo.m`
/// -- explicitly labeled third-party, not Rao's own code -- the per-learner
/// teaching-factor finding, the unconditionally-distinct partner-selection
/// finding, the min_pop adjustment from the plan's sketched 3 down to 2, and
/// the "parameter-free" framing's Črepinšek/Liu/Mernik (2012) counterpoint).
/// Both stages: uniform init, `boundary/clamp` (same as every other preset
/// in this crate), `replace/one-to-one-greedy` (DE's kind, reused as-is --
/// each phase's own `if newsol.Cost<pop(i).Cost` acceptance). A full
/// generation costs `2 * pop_size` evaluations (both stages evaluate; see
/// `sezgi_core::engine`'s two-stage budget-accounting test). `pop_size` is
/// the class size; canonical is 30 per the source paper. `min_pop = 2` --
/// the learner phase's partner selection is a HARD requirement (needs
/// exactly one OTHER member at minimum), enforced via `AlgorithmSpec::
/// validate`.
pub fn tlbo(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "tlbo".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![
            StageSpec {
                generator: comp("gen/tlbo-teacher", serde_json::json!({})),
                replacer: comp("replace/one-to-one-greedy", serde_json::json!({})),
                adapter: None,
            },
            StageSpec {
                generator: comp("gen/tlbo-learner", serde_json::json!({})),
                replacer: comp("replace/one-to-one-greedy", serde_json::json!({})),
                adapter: None,
            },
        ],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Harris Hawks Optimization (Heidari, Mirjalili, Faris, Aljarah, Mafarja &
/// Chen 2019, *Future Generation Computer Systems*) -- a labeled metaphor
/// preset (see `hho.rs`'s module doc for the full provenance extraction
/// against the paper AUTHOR's own `HHO.m`, the wave's most complex
/// multi-branch escape-energy tree, the hard/soft besiege mapping delta,
/// the rapid-dive in-generator-evaluation design decision, and the
/// mean(X)/random-hawk in-place semantics). Uniform init, `boundary/clamp`
/// (same as every other preset in this crate), `replace/generational`
/// (finding 10 -- the source's own exploration/besiege-without-dive
/// branches overwrite unconditionally; the dive branches' accept/reject is
/// resolved entirely inside `gen/hho` itself). `pop_size` is the hawk
/// count; canonical is 30 per the source's own demo. `min_pop = 2`,
/// enforced via `AlgorithmSpec::validate`.
pub fn hho(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "hho".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/hho", serde_json::json!({})),
            replacer: comp("replace/generational", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Ant Lion Optimizer (Mirjalili 2015) -- a labeled metaphor preset (see
/// `alo.rs`'s module doc for the full provenance extraction against the
/// author's own `ALO.m`/`Random_walk_around_antlion.m`/
/// `RouletteWheelSelection.m`, the faithful-full-walk cost decision, the
/// elitism design adjudication -- reusing `replace/mu-plus-lambda` as-is,
/// no blackboard state needed -- and the negative-fitness roulette-weight
/// simplification). Uniform init, `boundary/clamp` (same as every other
/// preset in this crate), `gen/alo` paired with `replace/mu-plus-lambda`
/// (ES's kind, reused as-is -- `ALO.m`'s own antlion-update merge-sort-
/// truncate mechanism, verified to be exactly this shape). `pop_size` is
/// the ant/antlion count; canonical is 25 per this wave's convention.
/// `min_pop = 2`, enforced via `AlgorithmSpec::validate`.
pub fn alo(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "alo".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/alo", serde_json::json!({})),
            replacer: comp("replace/mu-plus-lambda", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Artificial Bee Colony (Karaboga 2005, TR-06 / Karaboga & Basturk 2007,
/// *Journal of Global Optimization*) -- a labeled metaphor preset (see
/// `abc.rs`'s module doc for the full provenance extraction against the
/// author's own `ABCorig.m` plus the official `Python_ABC` port, the
/// pop<->food-source convention resolution, the fitness-transform
/// monotonicity proof, the probability-formula delta from the plan's
/// sketch, and the phase-design adjudication -- a single stage, not two
/// symmetric stages like `tlbo`). `gen/abc-employed` paired with the new
/// `replace/abc-trial-greedy` (the trial counters' bootstrap-then-owning
/// pair, mirroring `mfo.rs`'s flame-memory pattern) plus the new
/// `adapter/abc-onlooker-scout` (the onlooker AND scout phases, folded into
/// one self-contained adapter -- see the module doc for why the onlooker
/// scan cannot fit the `Generator`+`Replacer` shape). `pop_size` IS `SN`
/// (the food-source count) directly, NOT Karaboga's colony size `NP=2*SN`;
/// canonical is 20 per this module's resolved convention. `min_pop = 2`
/// (the neighbour `k!=i` rejection loop's hard floor), enforced via
/// `AlgorithmSpec::validate`. A full cycle costs `2*pop_size` evaluations
/// (`+1` when a scout fires) -- see the module doc's "Eval accounting".
pub fn abc(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "abc".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/abc-employed", serde_json::json!({})),
            replacer: comp("replace/abc-trial-greedy", serde_json::json!({})),
            adapter: Some(comp("adapter/abc-onlooker-scout", serde_json::json!({}))),
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// Gravitational Search Algorithm (Rashedi, Nezamabadi-pour & Saryazdi 2009,
/// *Information Sciences*) -- a labeled metaphor preset, and the wave's LAST
/// stateful/blackboard algorithm (see `gsa.rs`'s module doc for the full
/// provenance extraction against the author's own `GSA.m`/`Gconstant.m`/
/// `massCalculation.m`/`Gfield.m`/`move.m`, the verified `M_i`-free force/
/// acceleration delta from the plan's sketch, the per-`(i,j,d)` rand-
/// placement finding, and the confirmation that GSA's own persisted
/// `Fbest`/`Lbest` are pure reporting bookkeeping never fed back into the
/// mechanism -- settling the current-pop-convention controller ruling).
/// Uniform init, `boundary/clamp` (`GSA.m`'s own `space_bound`, same as
/// every other preset in this crate), `gen/gsa` (owns the persisted
/// `gsa/velocity` blackboard state, no separate adapter -- same self-owning
/// shape as `pso`/`bat`) paired with unconditional generational replacement
/// (`replace/generational` -- `move.m` overwrites every agent's position
/// unconditionally, no per-agent fitness-improvement test). `pop_size` is
/// the agent count; canonical is 30 per this wave's convention. `min_pop =
/// 2` (the force loop needs a distinct `j != i` to contribute at all),
/// enforced via `AlgorithmSpec::validate`.
pub fn gsa(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "gsa".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/gsa", serde_json::json!({})),
            replacer: comp("replace/generational", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

pub fn sa(budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "sa/metropolis-geometric".into(),
        pop_size: 1,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/step", serde_json::json!(
                {"dist": {"kind": "gaussian", "mean": 0.0, "sigma": 0.5}, "rate": 1.0})),
            replacer: comp("replace/metropolis", serde_json::json!(
                {"t0": 1.0, "alpha": 0.999})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

pub fn shade(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "de/shade".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/de-shade", serde_json::json!({"h": 6, "p": 0.11})),
            replacer: comp("replace/shade", serde_json::json!({})),
            adapter: Some(comp("adapter/shade-history", serde_json::json!({}))),
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

pub fn lshade(dim: usize, budget: u64) -> AlgorithmSpec {
    let pop_size = 18 * dim;
    AlgorithmSpec {
        name: "de/l-shade".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/de-shade", serde_json::json!({"h": 6, "p": 0.11})),
            replacer: comp("replace/shade", serde_json::json!({})),
            adapter: Some(comp("adapter/shade-lshade", serde_json::json!(
                {"n_init": pop_size, "n_min": 4}))),
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// (μ/μ_w,λ)-CMA-ES (Hansen's tutorial form, positive-weights variant — see
/// `cma.rs`'s module doc for the two documented sezgi simplifications).
/// `pop_size` is λ; the caller picks it (sezgi does not auto-derive it from
/// `dim`). A common guideline (Hansen's default) is
/// `λ = 4 + ⌊3·ln(dim)⌋`.
pub fn cmaes(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "cma-es".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/cma", serde_json::json!({})),
            replacer: comp("replace/cma-update", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

/// (μ/μ_w,λ)-CMA-ES with IPOP-style stagnation restarts (M2b Task 12):
/// `λ = 4 + ⌊3·ln(dim)⌋` (Hansen's default, computed here rather than left to
/// the caller, since `restart/stagnation`'s `Sizing::Ipop` scales from this
/// starting population every time it fires). Same `gen/cma` +
/// `replace/cma-update` stage as `cmaes`, plus `restart/stagnation` with
/// `patience: 2000, sizing: "ipop", factor: 2.0, max_pop: 512`.
pub fn cmaes_ipop(dim: usize, budget: u64) -> AlgorithmSpec {
    let pop_size = 4 + (3.0 * (dim as f64).ln()).floor() as usize;
    AlgorithmSpec {
        name: "cma-es/ipop".into(), pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/cma", serde_json::json!({})),
            replacer: comp("replace/cma-update", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: Some(comp("restart/stagnation", serde_json::json!(
            {"patience": 2000, "sizing": "ipop", "factor": 2.0, "max_pop": 512}))),
    }
}

/// Nelder–Mead simplex (M2b Task 13; see `nm.rs`'s module doc for the
/// standard coefficients and the batch-engine state-machine adaptation).
/// `pop_size` is fixed to `dim + 1` (the population IS the simplex).
pub fn nelder_mead(dim: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "nelder-mead".into(),
        pop_size: dim + 1,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/nelder-mead", serde_json::json!({})),
            replacer: comp("replace/nelder-mead", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}

pub fn random_search(pop_size: usize, budget: u64) -> AlgorithmSpec {
    AlgorithmSpec {
        name: "random-search".into(),
        pop_size,
        init: comp("init/uniform", serde_json::json!({})),
        boundary: comp("boundary/clamp", serde_json::json!({})),
        stages: vec![StageSpec {
            generator: comp("gen/uniform-resample", serde_json::json!({})),
            replacer: comp("replace/mu-plus-lambda", serde_json::json!({})),
            adapter: None,
        }],
        termination: TerminationSpec { budget, target: None },
        restart: None,
    }
}
