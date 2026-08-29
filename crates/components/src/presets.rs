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
