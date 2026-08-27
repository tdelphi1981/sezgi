use sezgi_core::component::Registry;
use sezgi_core::engine::{Engine, RunConfig, RunResult};
use sezgi_core::problem::Problem;
use sezgi_core::spec::AlgorithmSpec;
use sezgi_problems::BbobProblem;
use serde::{Deserialize, Serialize};

pub const RNG_ENGINE_ID: &str = "splitmix64+xoshiro256++/v1";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "suite", rename_all = "snake_case")]
pub enum ProblemRef {
    Bbob { fid: u32, dim: usize, instance: u32 },
}

impl ProblemRef {
    fn build(&self) -> Result<Box<dyn Problem>, ManifestError> {
        match *self {
            ProblemRef::Bbob { fid, dim, instance } =>
                Ok(Box::new(BbobProblem::new(fid, dim, instance)
                    .map_err(|e| ManifestError::Problem(e.to_string()))?)),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub sezgi_version: String,
    pub rng_engine: String,
    pub master_seed: u64,
    pub run_id: u64,
    pub best_f: f64,
    pub evals_used: u64,
    pub wall_time_s: f64,
    pub problem: ProblemRef,
    pub algorithm: AlgorithmSpec,
}

#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("ayrıştırma: {0}")]
    Parse(String),
    #[error("problem kurulamadı: {0}")]
    Problem(String),
    #[error("motor: {0}")]
    Engine(String),
}

impl RunManifest {
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("manifest serileştirilemedi")
    }
    pub fn from_toml(s: &str) -> Result<Self, ManifestError> {
        toml::from_str(s).map_err(|e| ManifestError::Parse(e.to_string()))
    }
}

pub fn run_with_manifest(spec: &AlgorithmSpec, pref: ProblemRef, reg: &Registry,
                         cfg: RunConfig) -> Result<(RunResult, RunManifest), ManifestError> {
    let problem = pref.build()?;
    let engine = Engine::from_spec(spec, reg, problem.space())
        .map_err(|e| ManifestError::Engine(e.to_string()))?;
    let t0 = std::time::Instant::now();
    let res = engine.run(problem.as_ref(), cfg, None)
        .map_err(|e| ManifestError::Engine(e.to_string()))?;
    let m = RunManifest {
        sezgi_version: env!("CARGO_PKG_VERSION").to_string(),
        rng_engine: RNG_ENGINE_ID.to_string(),
        master_seed: cfg.master_seed,
        run_id: cfg.run_id,
        best_f: res.best_f,
        evals_used: res.evals_used,
        wall_time_s: t0.elapsed().as_secs_f64(),
        problem: pref,
        algorithm: spec.clone(),
    };
    Ok((res, m))
}

pub fn verify(m: &RunManifest, reg: &Registry) -> Result<bool, ManifestError> {
    let problem = m.problem.build()?;
    let engine = Engine::from_spec(&m.algorithm, reg, problem.space())
        .map_err(|e| ManifestError::Engine(e.to_string()))?;
    let res = engine.run(problem.as_ref(),
                         RunConfig { master_seed: m.master_seed, run_id: m.run_id }, None)
        .map_err(|e| ManifestError::Engine(e.to_string()))?;
    Ok(res.best_f.to_bits() == m.best_f.to_bits() && res.evals_used == m.evals_used)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_components::{presets, register_builtins};
    use sezgi_core::component::Registry;
    use sezgi_core::engine::RunConfig;

    fn reg() -> Registry {
        let mut r = Registry::new();
        register_builtins(&mut r);
        r
    }

    #[test]
    fn manifest_roundtrip_and_verify() {
        let spec = presets::de_rand_1(20, 2_000);
        let pref = ProblemRef::Bbob { fid: 1, dim: 5, instance: 1 };
        let (_res, m) = run_with_manifest(&spec, pref, &reg(),
            RunConfig { master_seed: 42, run_id: 0 }).unwrap();
        let m2 = RunManifest::from_toml(&m.to_toml()).unwrap();
        assert!(verify(&m2, &reg()).unwrap(), "yeniden koşu bit-uyumlu olmalı");
    }

    #[test]
    fn tampered_manifest_fails_verify() {
        let spec = presets::de_rand_1(20, 2_000);
        let pref = ProblemRef::Bbob { fid: 1, dim: 5, instance: 1 };
        let (_r, mut m) = run_with_manifest(&spec, pref, &reg(),
            RunConfig { master_seed: 42, run_id: 0 }).unwrap();
        m.best_f += 1.0;
        assert!(!verify(&m, &reg()).unwrap());
    }
}
