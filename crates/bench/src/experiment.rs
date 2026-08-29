//! Experiment spec (TOML) + deterministic sequential executor (M2c Task 7).
//!
//! An [`ExperimentSpec`] declares a cartesian sweep over algorithms,
//! problems (bbob only, for M2c), instances, seeds and budgets. [`enumerate`]
//! expands that sweep into an ordered [`Vec<PlannedRun>`], validating every
//! algorithm against every problem's search space up front (fail fast,
//! before any run is executed). [`run_experiment_sequential`] enumerates and
//! then runs each [`PlannedRun`] through [`Engine::run`] in order, calling a
//! caller-supplied hook after each completed run.
//!
//! ## Preset kind -> [`sezgi_components::presets`] mapping
//!
//! An `[[algorithms]]` entry can reference a built-in preset by `kind`
//! instead of embedding a full [`AlgorithmSpec`]. Presets fall into three
//! groups:
//!
//! | `kind`               | `pop_size`                        | dim-linked? |
//! |-----------------------|------------------------------------|-------------|
//! | `de_rand_1`           | required                            | no |
//! | `de_best_1`           | required                            | no |
//! | `jde`                 | required                            | no |
//! | `es_mu_plus_lambda`   | required                            | no |
//! | `ga_real`             | required                            | no |
//! | `pso`                 | required                            | no |
//! | `shade`               | required                            | no |
//! | `cmaes`               | required                            | no |
//! | `random_search`       | required                            | no |
//! | `lshade`              | ignored (pop_size = 18 * dim)        | yes |
//! | `cmaes_ipop`          | ignored (pop_size = 4 + floor(3 ln dim)) | yes |
//! | `nelder_mead`         | ignored (pop_size = dim + 1)         | yes |
//! | `sa`                  | ignored (pop_size fixed at 1)        | no |
//!
//! Dim-linked presets receive the enumerated problem's `dim` at enumeration
//! time and ignore any `pop_size` given in the TOML. Any other kind requires
//! `pop_size` in the TOML and errors (`MissingPopSize`) if it is absent.
//! An unknown `kind` errors listing the valid kinds.
//!
//! SIMPLIFICATION: `es_mu_plus_lambda` additionally takes a step
//! [`Distribution`] in `sezgi_components::presets::es_mu_plus_lambda`, which
//! a bare preset ref cannot select; the executor fixes it to
//! `Gaussian { mean: 0.0, sigma: 0.5 }`. Use `spec_toml` instead of `preset`
//! to choose a different step distribution.
//!
//! Regardless of source (`preset` or `spec_toml`), `spec.termination.budget`
//! is ALWAYS overridden to the enumerated run's budget.

use serde::de::Error as DeError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sezgi_components::presets;
use sezgi_core::component::Registry;
use sezgi_core::dist::Distribution;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_core::problem::Problem;
use sezgi_core::spec::{AlgorithmSpec, SpecError};
use sezgi_problems::BbobProblem;
use std::fmt;

// ---------------------------------------------------------------------
// Experiment spec (TOML)
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExperimentSpec {
    pub name: String,
    /// Master seeds. `run_id` for a seed is that seed's INDEX in this list
    /// (pinned, not the seed value itself) — reordering `seeds` changes the
    /// runs that are produced.
    pub seeds: Vec<u64>,
    pub budgets: Vec<u64>,
    pub algorithms: Vec<AlgoEntry>,
    pub problems: Vec<ProblemEntry>,
}

impl ExperimentSpec {
    pub fn from_toml(src: &str) -> Result<Self, ExperimentError> {
        toml::from_str(src).map_err(|e| ExperimentError::Parse(e.to_string()))
    }
    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("experiment spec could not be serialized")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AlgoEntry {
    pub name: String,
    pub source: AlgoSource,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlgoSource {
    Preset { kind: String, pop_size: Option<usize> },
    SpecToml(String),
}

/// Wire-format twin of [`AlgoEntry`]: `preset = { kind, pop_size }` XOR
/// `spec_toml = "..."` sit as sibling keys alongside `name`. Kept as a
/// private struct + hand-written (de)serialization on [`AlgoEntry`] so the
/// public type can carry a proper `source: AlgoSource` enum.
#[derive(Serialize, Deserialize)]
struct RawAlgoEntry {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    preset: Option<RawPreset>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    spec_toml: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct RawPreset {
    kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pop_size: Option<usize>,
}

impl Serialize for AlgoEntry {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let raw = match &self.source {
            AlgoSource::Preset { kind, pop_size } => RawAlgoEntry {
                name: self.name.clone(),
                preset: Some(RawPreset { kind: kind.clone(), pop_size: *pop_size }),
                spec_toml: None,
            },
            AlgoSource::SpecToml(s) => RawAlgoEntry {
                name: self.name.clone(),
                preset: None,
                spec_toml: Some(s.clone()),
            },
        };
        raw.serialize(s)
    }
}

impl<'de> Deserialize<'de> for AlgoEntry {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = RawAlgoEntry::deserialize(d)?;
        let source = match (raw.preset, raw.spec_toml) {
            (Some(p), None) => AlgoSource::Preset { kind: p.kind, pop_size: p.pop_size },
            (None, Some(s)) => AlgoSource::SpecToml(s),
            (Some(_), Some(_)) => return Err(DeError::custom(format!(
                "algorithm `{}` must have either `preset` or `spec_toml`, not both", raw.name))),
            (None, None) => return Err(DeError::custom(format!(
                "algorithm `{}` must have either `preset` or `spec_toml`", raw.name))),
        };
        Ok(AlgoEntry { name: raw.name, source })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProblemEntry {
    /// M2c: only `"bbob"` is supported.
    pub suite: String,
    pub fid: u32,
    pub dim: usize,
    pub instances: Vec<u32>,
}

// ---------------------------------------------------------------------
// Run identity + result
// ---------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RunKey {
    pub algo: String,
    pub fid: u32,
    pub dim: usize,
    pub instance: u32,
    pub seed: u64,
    pub budget: u64,
}

impl fmt::Display for RunKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/f{}d{}i{}/s{}/b{}",
               self.algo, self.fid, self.dim, self.instance, self.seed, self.budget)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunRecord {
    pub key: RunKey,
    pub best_f: f64,
    pub f_opt: f64,
    pub evals_used: u64,
    pub wall_secs: f64,
}

// ---------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ExperimentError {
    #[error("could not parse experiment spec: {0}")]
    Parse(String),
    #[error("unknown preset kind `{kind}` (valid kinds: {valid})")]
    UnknownPresetKind { kind: String, valid: String },
    #[error("preset `{kind}` requires `pop_size` but none was given")]
    MissingPopSize { kind: String },
    #[error("algorithm `{algo}` is invalid for problem `{problem}`: {source}")]
    Validation { algo: String, problem: String, #[source] source: SpecError },
    #[error("could not build problem: {0}")]
    Problem(String),
    #[error("engine error: {0}")]
    Engine(String),
}

const VALID_PRESET_KINDS: &[&str] = &[
    "de_rand_1", "de_best_1", "jde", "es_mu_plus_lambda", "ga_real", "pso",
    "shade", "cmaes", "random_search", "lshade", "cmaes_ipop", "nelder_mead", "sa",
];

fn require_pop_size(kind: &str, pop_size: Option<usize>) -> Result<usize, ExperimentError> {
    pop_size.ok_or_else(|| ExperimentError::MissingPopSize { kind: kind.to_string() })
}

/// See the module doc's mapping table.
fn build_preset(kind: &str, pop_size: Option<usize>, dim: usize, budget: u64)
    -> Result<AlgorithmSpec, ExperimentError>
{
    let mut spec = match kind {
        "de_rand_1" => presets::de_rand_1(require_pop_size(kind, pop_size)?, budget),
        "de_best_1" => presets::de_best_1(require_pop_size(kind, pop_size)?, budget),
        "jde" => presets::jde(require_pop_size(kind, pop_size)?, budget),
        "es_mu_plus_lambda" => presets::es_mu_plus_lambda(
            require_pop_size(kind, pop_size)?, budget,
            Distribution::Gaussian { mean: 0.0, sigma: 0.5 }),
        "ga_real" => presets::ga_real(require_pop_size(kind, pop_size)?, budget),
        "pso" => presets::pso(require_pop_size(kind, pop_size)?, budget),
        "shade" => presets::shade(require_pop_size(kind, pop_size)?, budget),
        "cmaes" => presets::cmaes(require_pop_size(kind, pop_size)?, budget),
        "random_search" => presets::random_search(require_pop_size(kind, pop_size)?, budget),
        "lshade" => presets::lshade(dim, budget),
        "cmaes_ipop" => presets::cmaes_ipop(dim, budget),
        "nelder_mead" => presets::nelder_mead(dim, budget),
        "sa" => presets::sa(budget),
        other => return Err(ExperimentError::UnknownPresetKind {
            kind: other.to_string(),
            valid: VALID_PRESET_KINDS.join(", "),
        }),
    };
    spec.termination.budget = budget; // always overridden, per contract
    Ok(spec)
}

fn build_algo_spec(source: &AlgoSource, dim: usize, budget: u64)
    -> Result<AlgorithmSpec, ExperimentError>
{
    match source {
        AlgoSource::Preset { kind, pop_size } => build_preset(kind, *pop_size, dim, budget),
        AlgoSource::SpecToml(s) => {
            let mut spec = AlgorithmSpec::from_toml(s)
                .map_err(|e| ExperimentError::Parse(e.to_string()))?;
            spec.termination.budget = budget; // always overridden, per contract
            Ok(spec)
        }
    }
}

// ---------------------------------------------------------------------
// Enumeration
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PlannedRun {
    pub key: RunKey,
    pub algo_spec: AlgorithmSpec,
    pub fid: u32,
    pub dim: usize,
    pub instance: u32,
    pub seed: u64,
    pub run_id: u64,
}

fn problem_label(fid: u32, dim: usize) -> String {
    format!("bbob/f{fid}d{dim}")
}

/// Cartesian product `algorithms x problems x instances x seeds x budgets`,
/// in declaration order. Validates every algorithm spec against every
/// problem's search space up front — before building any [`PlannedRun`] —
/// so a bad combination fails fast, before any run is executed.
pub fn enumerate(spec: &ExperimentSpec) -> Result<Vec<PlannedRun>, ExperimentError> {
    if spec.budgets.is_empty() {
        return Err(ExperimentError::Parse("`budgets` must not be empty".into()));
    }

    let mut reg = Registry::new();
    sezgi_components::register_builtins(&mut reg);

    // Fail-fast validation pass: every algorithm x problem pair, before any
    // PlannedRun is constructed.
    for algo in &spec.algorithms {
        for prob in &spec.problems {
            if prob.suite != "bbob" {
                return Err(ExperimentError::Parse(format!(
                    "unsupported suite `{}` (M2c supports `bbob` only)", prob.suite)));
            }
            let instance = *prob.instances.first().ok_or_else(|| ExperimentError::Parse(
                format!("problem {} has no instances", problem_label(prob.fid, prob.dim))))?;
            let problem = BbobProblem::new(prob.fid, prob.dim, instance)
                .map_err(|e| ExperimentError::Problem(e.to_string()))?;
            let algo_spec = build_algo_spec(&algo.source, prob.dim, spec.budgets[0])?;
            algo_spec.validate(&reg, problem.space()).map_err(|source| ExperimentError::Validation {
                algo: algo.name.clone(),
                problem: problem_label(prob.fid, prob.dim),
                source,
            })?;
        }
    }

    // Cartesian expansion.
    let mut planned = Vec::new();
    for algo in &spec.algorithms {
        for prob in &spec.problems {
            for &instance in &prob.instances {
                for (seed_idx, &seed) in spec.seeds.iter().enumerate() {
                    for &budget in &spec.budgets {
                        let algo_spec = build_algo_spec(&algo.source, prob.dim, budget)?;
                        let key = RunKey {
                            algo: algo.name.clone(),
                            fid: prob.fid,
                            dim: prob.dim,
                            instance,
                            seed,
                            budget,
                        };
                        planned.push(PlannedRun {
                            key,
                            algo_spec,
                            fid: prob.fid,
                            dim: prob.dim,
                            instance,
                            seed,
                            run_id: seed_idx as u64,
                        });
                    }
                }
            }
        }
    }
    Ok(planned)
}

// ---------------------------------------------------------------------
// Sequential executor
// ---------------------------------------------------------------------

/// Enumerates `spec`, then runs each [`PlannedRun`] through [`Engine::run`]
/// in enumeration order, calling `on_record` after each completed run.
pub fn run_experiment_sequential(
    spec: &ExperimentSpec,
    mut on_record: impl FnMut(&RunRecord),
) -> Result<Vec<RunRecord>, ExperimentError> {
    let planned = enumerate(spec)?;

    let mut reg = Registry::new();
    sezgi_components::register_builtins(&mut reg);

    let mut records = Vec::with_capacity(planned.len());
    for run in planned {
        let problem = BbobProblem::new(run.fid, run.dim, run.instance)
            .map_err(|e| ExperimentError::Problem(e.to_string()))?;
        let engine = Engine::from_spec(&run.algo_spec, &reg, problem.space())
            .map_err(|e| ExperimentError::Engine(e.to_string()))?;
        let t0 = std::time::Instant::now();
        let res = engine
            .run(&problem, RunConfig { master_seed: run.seed, run_id: run.run_id }, None)
            .map_err(|e| ExperimentError::Engine(e.to_string()))?;
        let record = RunRecord {
            key: run.key,
            best_f: res.best_f,
            f_opt: problem.f_opt(),
            evals_used: res.evals_used,
            wall_secs: t0.elapsed().as_secs_f64(),
        };
        on_record(&record);
        records.push(record);
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn demo_toml() -> &'static str {
        r#"
            name = "demo"
            seeds = [1, 2, 3]
            budgets = [1000, 10000]

            [[algorithms]]
            name = "de"
            preset = { kind = "de_rand_1", pop_size = 30 }

            [[problems]]
            suite = "bbob"
            fid = 1
            dim = 5
            instances = [1, 2]
        "#
    }

    #[test]
    fn toml_roundtrip() {
        let spec = ExperimentSpec::from_toml(demo_toml()).unwrap();
        assert_eq!(spec.name, "demo");
        assert_eq!(spec.seeds, vec![1, 2, 3]);
        assert_eq!(spec.budgets, vec![1000, 10000]);
        assert_eq!(spec.algorithms.len(), 1);
        assert_eq!(spec.algorithms[0].name, "de");
        assert_eq!(spec.algorithms[0].source,
            AlgoSource::Preset { kind: "de_rand_1".into(), pop_size: Some(30) });
        assert_eq!(spec.problems.len(), 1);
        assert_eq!(spec.problems[0], ProblemEntry {
            suite: "bbob".into(), fid: 1, dim: 5, instances: vec![1, 2],
        });

        let spec2 = ExperimentSpec::from_toml(&spec.to_toml()).unwrap();
        assert_eq!(spec, spec2);
    }

    #[test]
    fn spec_toml_algo_source_roundtrips() {
        let toml_src = r#"
            name = "custom"
            seeds = [1]
            budgets = [100]
            [[algorithms]]
            name = "custom-de"
            spec_toml = "inline"
            [[problems]]
            suite = "bbob"
            fid = 1
            dim = 5
            instances = [1]
        "#;
        let spec = ExperimentSpec::from_toml(toml_src).unwrap();
        assert_eq!(spec.algorithms[0].source, AlgoSource::SpecToml("inline".into()));
        let spec2 = ExperimentSpec::from_toml(&spec.to_toml()).unwrap();
        assert_eq!(spec, spec2);
    }

    #[test]
    fn algo_entry_rejects_both_preset_and_spec_toml() {
        let toml_src = r#"
            name = "de"
            preset = { kind = "de_rand_1", pop_size = 10 }
            spec_toml = "x"
        "#;
        assert!(toml::from_str::<AlgoEntry>(toml_src).is_err());
    }

    #[test]
    fn algo_entry_rejects_neither_preset_nor_spec_toml() {
        let toml_src = r#"name = "de""#;
        assert!(toml::from_str::<AlgoEntry>(toml_src).is_err());
    }

    fn two_algo_spec() -> ExperimentSpec {
        ExperimentSpec {
            name: "sweep".into(),
            seeds: vec![10, 20],
            budgets: vec![500, 1000],
            algorithms: vec![
                AlgoEntry {
                    name: "de".into(),
                    source: AlgoSource::Preset { kind: "de_rand_1".into(), pop_size: Some(10) },
                },
                AlgoEntry {
                    name: "rs".into(),
                    source: AlgoSource::Preset { kind: "random_search".into(), pop_size: Some(10) },
                },
            ],
            problems: vec![ProblemEntry {
                suite: "bbob".into(), fid: 1, dim: 5, instances: vec![1, 2],
            }],
        }
    }

    #[test]
    fn enumerate_count_and_order() {
        let spec = two_algo_spec();
        let planned = enumerate(&spec).unwrap();
        // 2 algos x 1 problem x 2 instances x 2 seeds x 2 budgets = 16
        assert_eq!(planned.len(), 16);

        let first = &planned[0].key;
        assert_eq!(first.algo, "de");
        assert_eq!(first.fid, 1);
        assert_eq!(first.dim, 5);
        assert_eq!(first.instance, 1);
        assert_eq!(first.seed, 10);
        assert_eq!(first.budget, 500);
        assert_eq!(planned[0].run_id, 0);

        let last = &planned[15].key;
        assert_eq!(last.algo, "rs");
        assert_eq!(last.fid, 1);
        assert_eq!(last.dim, 5);
        assert_eq!(last.instance, 2);
        assert_eq!(last.seed, 20);
        assert_eq!(last.budget, 1000);
        assert_eq!(planned[15].run_id, 1);
    }

    #[test]
    fn run_key_display_is_stable() {
        let key = RunKey { algo: "de".into(), fid: 1, dim: 5, instance: 2, seed: 42, budget: 1000 };
        assert_eq!(key.to_string(), "de/f1d5i2/s42/b1000");
    }

    #[test]
    fn unknown_preset_kind_is_rejected() {
        let mut spec = two_algo_spec();
        spec.algorithms[0].source = AlgoSource::Preset { kind: "bogus".into(), pop_size: Some(10) };
        let err = enumerate(&spec).unwrap_err();
        assert!(matches!(err, ExperimentError::UnknownPresetKind { .. }), "got {err:?}");
    }

    #[test]
    fn fail_fast_validation_runs_before_any_run() {
        // de_rand_1's gen/de generator has min_pop 4; pop_size 3 is too small.
        let mut spec = two_algo_spec();
        spec.algorithms[0].source = AlgoSource::Preset { kind: "de_rand_1".into(), pop_size: Some(3) };

        let err = enumerate(&spec).unwrap_err();
        assert!(matches!(err, ExperimentError::Validation { .. }), "got {err:?}");

        let mut runs_started = 0usize;
        let result = run_experiment_sequential(&spec, |_| runs_started += 1);
        assert!(result.is_err(), "sequential run must fail fast on validation error");
        assert_eq!(runs_started, 0, "no run must execute before fail-fast validation errors out");
    }

    #[test]
    fn sequential_smoke_de_and_random_search() {
        let spec = ExperimentSpec {
            name: "smoke".into(),
            seeds: vec![1, 2],
            budgets: vec![500],
            algorithms: vec![
                AlgoEntry {
                    name: "de".into(),
                    source: AlgoSource::Preset { kind: "de_rand_1".into(), pop_size: Some(10) },
                },
                AlgoEntry {
                    name: "rs".into(),
                    source: AlgoSource::Preset { kind: "random_search".into(), pop_size: Some(10) },
                },
            ],
            problems: vec![ProblemEntry {
                suite: "bbob".into(), fid: 1, dim: 5, instances: vec![1],
            }],
        };

        let mut recorded = Vec::new();
        let records = run_experiment_sequential(&spec, |r| recorded.push(r.clone())).unwrap();

        assert_eq!(records.len(), 4, "2 algos x 1 problem x 1 instance x 2 seeds x 1 budget");
        assert_eq!(recorded.len(), 4, "on_record must be called once per completed run");
        for r in &records {
            assert!(r.best_f >= r.f_opt - 1e-9,
                "best_f ({}) must not undercut f_opt ({}) by more than numerical noise", r.best_f, r.f_opt);
            assert!(r.evals_used <= 500, "evals_used ({}) must respect the budget", r.evals_used);
        }
    }
}
