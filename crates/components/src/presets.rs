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
