use crate::component::{
    Adapter, BoundaryHandler, Ctx, Generator, Initializer, Registry, Replacer, Restart,
};
use crate::problem::{EvalObserver, Evaluator, Population, Problem};
use crate::rng::RngStream;
use crate::space::{Genotype, SearchSpace};
use crate::spec::{AlgorithmSpec, SpecError};
use crate::state::Blackboard;

pub struct Engine {
    init: Box<dyn Initializer>,
    boundary: Box<dyn BoundaryHandler>,
    stages: Vec<(Box<dyn Generator>, Box<dyn Replacer>)>,
    adapters: Vec<Option<Box<dyn Adapter>>>,
    restart: Option<Box<dyn Restart>>,
    pop_size: usize,
    budget: u64,
    target: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct RunConfig { pub master_seed: u64, pub run_id: u64 }

/// A completed run's outcome. `best_f`/`best_x` are sourced directly from
/// `Evaluator`'s own best-tracking (see [`crate::problem::Evaluator`]'s
/// doc): the best fitness/genotype pair over EVERY charged evaluation of the
/// run, from anywhere -- the engine's own per-stage/setup/restart calls, and
/// any `Generator`/`Adapter`'s own internal `ctx.eval.evaluate(..)` calls.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub best_f: f64,
    /// The best evaluated point (paired with [`Self::best_f`]). For most
    /// presets this always lies within the problem's declared domain (every
    /// charged evaluation was itself boundary-repaired first). It is NOT
    /// guaranteed to for every preset: a generator whose internal trial
    /// evaluations run on a raw, PRE-boundary-repair candidate (e.g.
    /// `gen/hho`'s dive-trial evaluations, see `sezgi_components::hho`'s
    /// module doc) can have that raw, possibly out-of-domain point become
    /// the run's own best -- because it was itself a charged evaluation, and
    /// `Evaluator` tracks the best over every charged evaluation, not only
    /// the ones the engine's own stage loop re-evaluates after repair.
    pub best_x: Genotype,
    pub evals_used: u64,
    pub iterations: u64,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error(transparent)]
    Spec(#[from] SpecError),
    #[error("initializer produced an empty population")]
    EmptyPopulation,
    #[error("budget ({budget}) is smaller than population size ({pop_size})")]
    BudgetSmallerThanPopulation { budget: u64, pop_size: usize },
    /// A component emitted a genotype whose shape (block count, per-block
    /// kind or length) does not match the search space. Every flat generator
    /// reads `blocks[0]` by documented contract (see
    /// `sezgi_components::compound`'s module doc), so a flat component run on
    /// a multi-block space used to corrupt genotypes silently; this guard
    /// turns that into a fast, named error. Checked BEFORE boundary repair.
    ///
    /// `stage` is the 0-based index of the stage whose generator produced the
    /// offending offspring, or [`INIT_STAGE`] (`usize::MAX`) when the
    /// initializer's population was the offender (this covers both the
    /// initial population and restart re-initialization). `detail` names
    /// expected vs got.
    #[error("genotype shape mismatch at {}: {detail}; a multi-block space needs a gen/compound generator (flat generators are single-block; eligible non-GA generators can ride inside gen/compound)", stage_label(*stage))]
    GenotypeShapeMismatch { stage: usize, detail: String },
}

/// `EngineError::GenotypeShapeMismatch::stage` value meaning "the initial
/// population or a restart re-initialization" (as opposed to a generator
/// stage index).
pub const INIT_STAGE: usize = usize::MAX;

fn stage_label(stage: usize) -> String {
    if stage == INIT_STAGE { "initialization".into() } else { format!("stage {stage}") }
}

impl Engine {
    pub fn from_spec(spec: &AlgorithmSpec, reg: &Registry, space: &SearchSpace)
        -> Result<Self, SpecError> {
        spec.validate(reg, space)?;
        Ok(Self {
            init: reg.build_initializer(&spec.init.kind, &spec.init.params)?,
            boundary: reg.build_boundary(&spec.boundary.kind, &spec.boundary.params)?,
            stages: spec.stages.iter().map(|st| Ok((
                reg.build_generator(&st.generator.kind, &st.generator.params)?,
                reg.build_replacer(&st.replacer.kind, &st.replacer.params)?,
            ))).collect::<Result<_, SpecError>>()?,
            adapters: spec.stages.iter().map(|st| {
                st.adapter.as_ref()
                    .map(|a| reg.build_adapter(&a.kind, &a.params))
                    .transpose()
                    .map_err(SpecError::from)
            }).collect::<Result<_, SpecError>>()?,
            restart: spec.restart.as_ref()
                .map(|r| reg.build_restart(&r.kind, &r.params))
                .transpose()?,
            pop_size: spec.pop_size,
            budget: spec.termination.budget,
            target: spec.termination.target,
        })
    }

    pub fn run(&self, problem: &dyn Problem, cfg: RunConfig,
               observer: Option<Box<dyn EvalObserver>>) -> Result<RunResult, EngineError> {
        let space = problem.space();
        let mut eval = Evaluator::new(problem, self.budget);
        if let Some(obs) = observer { eval.set_observer(obs); }
        let mut bb = Blackboard::new();
        let mut iterations = 0u64;

        // RNG stream path contract — the manifest records these paths (Task 16)
        let mut init_rng = RngStream::from_master(cfg.master_seed, &[cfg.run_id, 0]);
        // NOTE: collides with the replacer path at 500+ stages (2+2*499=1000); widen the offset if the engine path space is split further.
        let mut boundary_rng = RngStream::from_master(cfg.master_seed, &[cfg.run_id, 1000]);
        let mut stage_rngs: Vec<(RngStream, RngStream)> = (0..self.stages.len()).map(|i| (
            RngStream::from_master(cfg.master_seed, &[cfg.run_id, 1 + 2 * i as u64]),
            RngStream::from_master(cfg.master_seed, &[cfg.run_id, 2 + 2 * i as u64]),
        )).collect();
        // Adapter streams: one per stage, path [run_id, 1_000_000 + stage_index].
        let mut adapter_rngs: Vec<RngStream> = (0..self.stages.len()).map(|i|
            RngStream::from_master(cfg.master_seed, &[cfg.run_id, 1_000_000 + i as u64])
        ).collect();
        // Restart stream: path [run_id, 2_000_000]; successive restarts derive
        // a distinct child via restart_rng.split(restart_count).
        let mut restart_rng = RngStream::from_master(cfg.master_seed, &[cfg.run_id, 2_000_000]);
        let mut restart_count: u64 = 0;

        // Initialization
        let individuals = {
            let mut ctx = Ctx { space, rng: &mut init_rng, bb: &mut bb,
                                eval: &mut eval, iteration: 0 };
            self.init.initialize(self.pop_size, &mut ctx)
        };
        if individuals.is_empty() { return Err(EngineError::EmptyPopulation); }
        for g in &individuals {
            space.validate_genotype(g).map_err(|e| EngineError::GenotypeShapeMismatch {
                stage: INIT_STAGE, detail: e.to_string() })?;
        }
        let fitness = match eval.evaluate(&individuals) {
            Ok(f) => f,
            Err(_) => return Err(EngineError::BudgetSmallerThanPopulation {
                budget: self.budget,
                pop_size: self.pop_size,
            }),
        };
        let mut pop = Population { individuals, fitness };

        // Best-so-far: sourced from `eval` itself (`Evaluator::best_so_far`/
        // `best_x_so_far`), NOT tracked separately here. `Evaluator::evaluate`
        // is the sole gateway every charged evaluation in this crate passes
        // through -- this engine's own per-stage/setup/restart calls AND any
        // `Generator`/`Adapter`'s own internal `ctx.eval.evaluate(..)` calls
        // (e.g. `gen/hho`'s dive-trial evaluations, which evaluate a RAW
        // point the engine itself never sees again after boundary repair) --
        // so it is the only place that can see the true minimum over every
        // charged evaluation, from anywhere (M3-1 Task 1; see `problem.rs`'s
        // `Evaluator` doc comment). An engine-local `global_best` mirrored
        // ONLY the engine's own call sites (the setup eval below, each
        // stage's own offspring-evaluate, the post-adapter `pop` scan, and
        // restart re-init) and so necessarily missed any such internal
        // generator/adapter-issued evaluate call whose winning point was
        // never itself returned to the engine -- this was a real, measured
        // gap for `gen/hho`. Reading directly from `eval`
        // closes it for HHO and remains a no-op for every other preset (a
        // sweep already found best_f == observed-min 10/10 for all 16
        // non-HHO presets).

        let reached = |eval: &Evaluator| -> bool {
            matches!((self.target, eval.best_so_far()),
                     (Some(t), Some(b)) if b <= t)
        };

        'outer: while !reached(&eval) {
            for (si, (gen, rep)) in self.stages.iter().enumerate() {
                let mut offspring = {
                    let (gr, _) = &mut stage_rngs[si];
                    let mut ctx = Ctx { space, rng: gr, bb: &mut bb,
                                        eval: &mut eval, iteration: iterations };
                    gen.generate(&pop, &mut ctx)
                };
                for g in &offspring {
                    space.validate_genotype(g).map_err(|e| EngineError::GenotypeShapeMismatch {
                        stage: si, detail: e.to_string() })?;
                }
                for g in &mut offspring {
                    let mut ctx = Ctx { space, rng: &mut boundary_rng, bb: &mut bb,
                                        eval: &mut eval, iteration: iterations };
                    self.boundary.repair(g, space, &mut ctx);
                }
                let off_fit = match eval.evaluate(&offspring) {
                    Ok(f) => f,
                    Err(_) => break 'outer, // budget exhausted: clean exit
                };
                {
                    let (_, rr) = &mut stage_rngs[si];
                    let mut ctx = Ctx { space, rng: rr, bb: &mut bb,
                                        eval: &mut eval, iteration: iterations };
                    rep.replace(&mut pop, offspring, off_fit, &mut ctx);
                }
                if let Some(adapter) = &self.adapters[si] {
                    let mut ctx = Ctx { space, rng: &mut adapter_rngs[si], bb: &mut bb,
                                        eval: &mut eval, iteration: iterations };
                    adapter.adapt(&mut pop, &mut ctx);
                }
                // Post-adapter shape sweep: an adapter may write new
                // individuals straight into `pop` (bypassing the generator
                // guard above), so re-check the whole population here.
                for g in &pop.individuals {
                    space.validate_genotype(g).map_err(|e| EngineError::GenotypeShapeMismatch {
                        stage: si, detail: e.to_string() })?;
                }
                //
                // An adapter above may have evaluated brand-new individuals
                // via `ctx.eval.evaluate(..)` and written them straight into
                // `pop` without going through this loop's own generator-stage
                // `eval.evaluate(&offspring)` call (M2d-3 Task 8's
                // `adapter/abandon-worst-fraction` is the first such adapter;
                // see `sezgi_components::cs`'s module doc). Pre-M3-1 this
                // engine had to separately re-scan `pop` right here to catch
                // that case for its own `global_best`; now that best-tracking
                // lives in `eval` itself (M3-1 Task 1), that adapter-issued
                // evaluate call already updated `eval.best_so_far()`/
                // `best_x_so_far()` the instant it happened, so no scan is
                // needed -- `reached(&eval)` below sees it directly.
                if reached(&eval) { break 'outer; }
            }

            if let Some(restart) = &self.restart {
                let directive = {
                    let mut ctx = Ctx { space, rng: &mut restart_rng, bb: &mut bb,
                                        eval: &mut eval, iteration: iterations };
                    restart.check(&pop, &mut ctx)
                };
                if let Some(directive) = directive {
                    // Preserve the restart component's own declared state
                    // across the blackboard clear.
                    let provide_keys: Vec<String> =
                        restart.meta().provides.iter().map(|r| r.key.clone()).collect();
                    let mut preserved = Vec::with_capacity(provide_keys.len());
                    for k in &provide_keys {
                        if let Some(v) = bb.take_raw(k) { preserved.push((k.clone(), v)); }
                    }
                    bb = Blackboard::new();
                    for (k, v) in preserved { bb.put_raw(&k, v); }

                    let new_size = if directive.new_pop_size == 0 {
                        pop.len()
                    } else {
                        directive.new_pop_size
                    };
                    let mut reinit_rng = restart_rng.split(restart_count);
                    restart_count += 1;

                    let individuals = {
                        let mut ctx = Ctx { space, rng: &mut reinit_rng, bb: &mut bb,
                                            eval: &mut eval, iteration: iterations };
                        self.init.initialize(new_size, &mut ctx)
                    };
                    if individuals.is_empty() { return Err(EngineError::EmptyPopulation); }
                    // Restart re-initialization is an init-site: same guard,
                    // same `INIT_STAGE` encoding, BEFORE evaluation.
                    for g in &individuals {
                        space.validate_genotype(g).map_err(|e| EngineError::GenotypeShapeMismatch {
                            stage: INIT_STAGE, detail: e.to_string() })?;
                    }
                    match eval.evaluate(&individuals) {
                        Ok(fitness) => {
                            pop = Population { individuals, fitness };
                        }
                        Err(_) => break 'outer, // budget exhausted: clean exit, best-so-far kept
                    }
                    if reached(&eval) { break 'outer; }
                }
            }

            iterations += 1;
        }

        let best_f = eval.best_so_far()
            .expect("at least the initialization should have been evaluated");
        let best_x = eval.best_x_so_far()
            .expect("best_so_far and best_x_so_far are set together, on the same evaluate() call")
            .clone();
        Ok(RunResult {
            best_f,
            best_x,
            evals_used: eval.used(),
            iterations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::*;
    use crate::problem::{EvalObserver, Population, SphereShifted};
    use crate::space::{Block, BlockValues, Genotype, SearchSpace};
    use crate::spec::*;
    use crate::state::StateReq;

    fn uniform_sample(space: &SearchSpace, rng: &mut crate::rng::RngStream) -> Genotype {
        let blocks = space.blocks().iter().map(|b| match *b {
            Block::Float { lo, hi, n } =>
                BlockValues::Float((0..n).map(|_| lo + (hi - lo) * rng.next_f64()).collect()),
            _ => unreachable!("this test only uses float"),
        }).collect();
        Genotype { blocks }
    }

    struct UInit;
    impl Initializer for UInit {
        fn initialize(&self, n: usize, ctx: &mut Ctx) -> Vec<Genotype> {
            (0..n).map(|_| uniform_sample(ctx.space, ctx.rng)).collect()
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta::new("u-init", SupportedBlocks::All)
        }
    }
    struct Resample;
    impl Generator for Resample {
        fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
            (0..pop.len()).map(|_| uniform_sample(ctx.space, ctx.rng)).collect()
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta::new("resample", SupportedBlocks::All)
        }
    }
    struct Greedy;
    impl Replacer for Greedy {
        fn replace(&self, pop: &mut Population, off_i: Vec<Genotype>,
                   off_f: Vec<f64>, _c: &mut Ctx) {
            for (i, (gi, fi)) in off_i.into_iter().zip(off_f).enumerate() {
                if fi < pop.fitness[i] { pop.individuals[i] = gi; pop.fitness[i] = fi; }
            }
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta::new("greedy", SupportedBlocks::All)
        }
    }
    struct NoB;
    impl BoundaryHandler for NoB {
        fn repair(&self, _g: &mut Genotype, _s: &SearchSpace, _c: &mut Ctx) {}
        fn meta(&self) -> ComponentMeta {
            ComponentMeta::new("no-b", SupportedBlocks::All)
        }
    }

    fn setup() -> (Registry, AlgorithmSpec, SphereShifted) {
        let mut reg = Registry::new();
        reg.register_initializer("u-init", |_| Ok(Box::new(UInit)));
        reg.register_generator("resample", |_| Ok(Box::new(Resample)));
        reg.register_replacer("greedy", |_| Ok(Box::new(Greedy)));
        reg.register_boundary("no-b", |_| Ok(Box::new(NoB)));
        let spec = AlgorithmSpec {
            name: "random-search".into(), pop_size: 10,
            init: ComponentSpec { kind: "u-init".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "no-b".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "resample".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "greedy".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 500, target: None },
            restart: None,
        };
        (reg, spec, SphereShifted::new(vec![1.0, -2.0], -5.0, 5.0))
    }

    #[test]
    fn respects_budget_exactly() {
        let (reg, spec, p) = setup();
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
        assert!(r.evals_used <= 500);
        assert!(r.evals_used >= 500 - spec.pop_size as u64, "should use close to the full budget");
    }

    #[test]
    fn deterministic_given_seed() {
        let (reg, spec, p) = setup();
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r1 = e.run(&p, RunConfig { master_seed: 7, run_id: 3 }, None).unwrap();
        let r2 = e.run(&p, RunConfig { master_seed: 7, run_id: 3 }, None).unwrap();
        assert_eq!(r1.best_f, r2.best_f);
        assert_eq!(r1.best_x, r2.best_x);
    }

    #[test]
    fn different_run_ids_differ() {
        let (reg, spec, p) = setup();
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r1 = e.run(&p, RunConfig { master_seed: 7, run_id: 0 }, None).unwrap();
        let r2 = e.run(&p, RunConfig { master_seed: 7, run_id: 1 }, None).unwrap();
        assert_ne!(r1.best_f, r2.best_f);
    }

    #[test]
    fn target_stops_early() {
        let (reg, mut spec, p) = setup();
        spec.termination.target = Some(1e9); // every value satisfies the target
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 7, run_id: 0 }, None).unwrap();
        assert_eq!(r.evals_used, spec.pop_size as u64, "should stop right after init");
    }

    #[test]
    fn random_search_improves_over_init() {
        let (reg, spec, p) = setup();
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
        assert!(r.best_f < 5.0, "reasonable progress on sphere in 500 evaluations: {}", r.best_f);
    }

    /// A NON-elitist replacer: accepts incoming offspring unconditionally
    /// (the population can regress — like PSO). The engine must still
    /// correctly return the best seen so far (best_so_far).
    struct Unconditional;
    impl Replacer for Unconditional {
        fn replace(&self, pop: &mut Population, off_i: Vec<Genotype>,
                   off_fit: Vec<f64>, _c: &mut Ctx) {
            pop.individuals = off_i;
            pop.fitness = off_fit;
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta::new("unconditional", SupportedBlocks::All)
        }
    }

    #[test]
    fn non_elitist_replacer_keeps_global_best() {
        let mut reg = Registry::new();
        reg.register_initializer("u-init", |_| Ok(Box::new(UInit)));
        reg.register_generator("resample", |_| Ok(Box::new(Resample)));
        reg.register_replacer("unconditional", |_| Ok(Box::new(Unconditional)));
        reg.register_boundary("no-b", |_| Ok(Box::new(NoB)));
        let spec = AlgorithmSpec {
            name: "unconditional-search".into(), pop_size: 10,
            init: ComponentSpec { kind: "u-init".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "no-b".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "resample".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "unconditional".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 500, target: None },
            restart: None,
        };
        let p = SphereShifted::new(vec![1.0, -2.0], -5.0, 5.0);
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();

        struct MinObserver(std::sync::Arc<std::sync::Mutex<f64>>);
        impl EvalObserver for MinObserver {
            fn on_eval(&mut self, _eval_index: u64, _f: f64, best_so_far: f64) {
                let mut m = self.0.lock().unwrap();
                if best_so_far < *m { *m = best_so_far; }
            }
        }
        let observed_min = std::sync::Arc::new(std::sync::Mutex::new(f64::INFINITY));
        let obs = MinObserver(observed_min.clone());

        let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, Some(Box::new(obs))).unwrap();
        let observed_min = *observed_min.lock().unwrap();
        assert_eq!(r.best_f, observed_min,
            "the engine must not lose the global best under an unconditional (non-elitist) replacer");
    }

    #[test]
    fn budget_smaller_than_pop_is_clear_error() {
        let (reg, mut spec, p) = setup();
        spec.termination.budget = 5; // smaller than pop_size=10
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let err = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap_err();
        assert!(matches!(err, EngineError::BudgetSmallerThanPopulation { budget: 5, pop_size: 10 }),
            "should return a distinct error when the budget is smaller than the population, got: {:?}", err);
    }

    // ---- Adapter / Restart engine extension tests (M2b Task 4) ----

    /// Absence of `adapter`/`restart` in the spec must be byte-identical to
    /// today's behavior: a serde round-trip through JSON (where the fields
    /// are omitted via skip_serializing_if) must produce the same run.
    #[test]
    fn absent_adapter_and_restart_change_nothing() {
        let (reg, spec, p) = setup();
        let e1 = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r1 = e1.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();

        let json = spec.to_json();
        assert!(!json.contains("\"adapter\""),
            "adapter field must be omitted from JSON when absent (skip_serializing_if)");
        assert!(!json.contains("\"restart\""),
            "restart field must be omitted from JSON when absent (skip_serializing_if)");
        let spec2 = AlgorithmSpec::from_json(&json).unwrap();
        assert_eq!(spec2.stages[0].adapter, None);
        assert_eq!(spec2.restart, None);

        let e2 = Engine::from_spec(&spec2, &reg, p.space()).unwrap();
        let r2 = e2.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();

        assert_eq!(r1.best_f, r2.best_f);
        assert_eq!(r1.best_x, r2.best_x);
        assert_eq!(r1.evals_used, r2.evals_used);
        assert_eq!(r1.iterations, r2.iterations);
    }

    struct CountAdapter(std::sync::Arc<std::sync::atomic::AtomicU64>);
    impl Adapter for CountAdapter {
        fn adapt(&self, _pop: &mut Population, ctx: &mut Ctx) {
            let new_val = match ctx.bb.get_mut::<u64>("adapt_count") {
                Some(c) => { *c += 1; *c }
                None => { ctx.bb.insert("adapt_count", 1u64); 1 }
            };
            self.0.store(new_val, std::sync::atomic::Ordering::SeqCst);
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta::new("count-adapter", SupportedBlocks::All)
                .with_provides(vec![StateReq::of::<u64>("adapt_count")])
        }
    }

    #[test]
    fn adapter_runs_after_replacer() {
        let (mut reg, mut spec, p) = setup();
        let counter = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        {
            let counter = counter.clone();
            reg.register_adapter("count-adapter", move |_| {
                Ok(Box::new(CountAdapter(counter.clone())) as Box<dyn Adapter>)
            });
        }
        spec.stages[0].adapter = Some(ComponentSpec { kind: "count-adapter".into(), params: serde_json::json!({}) });
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
        assert_eq!(counter.load(std::sync::atomic::Ordering::SeqCst), r.iterations,
            "the adapter must run exactly once per iteration, after the replacer");
    }

    /// Evaluates a fixed point and writes it directly into `pop`, bypassing
    /// the stage's own generator-evaluate path entirely -- the same shape as
    /// `sezgi_components::cs::AbandonWorstFraction` (M2d-3 Task 8), which
    /// evaluates re-randomized nests via `ctx.eval.evaluate(..)` and installs
    /// them into `pop` directly, never through `offspring`/`off_fit`.
    struct PlantOptimum { shift: Vec<f64> }
    impl Adapter for PlantOptimum {
        fn adapt(&self, pop: &mut Population, ctx: &mut Ctx) {
            let g = Genotype { blocks: vec![BlockValues::Float(self.shift.clone())] };
            if let Ok(fitness) = ctx.eval.evaluate(std::slice::from_ref(&g)) {
                pop.individuals[0] = g;
                pop.fitness[0] = fitness[0];
            }
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta::new("plant-optimum", SupportedBlocks::All)
        }
    }

    /// Regression test (M2d-3 Task 8, fix round 1): an adapter that
    /// evaluates and installs a BRAND NEW individual into `pop` (never
    /// through the stage's own `eval.evaluate(&offspring)` call) must still
    /// have that individual's fitness/genotype captured by `global_best`,
    /// and therefore reflected in `RunResult::best_f`/`best_x`. `setup()`'s
    /// problem is `SphereShifted` with optimum at `shift = [1.0, -2.0]`
    /// (f = 0.0 there); `resample`/uniform-init sample continuously over
    /// `[-5,5]`, so an exact-bit hit on the optimum by chance is
    /// vanishingly improbable -- the adapter-planted point is, with
    /// overwhelming probability, the unique run-wide best. Before the
    /// engine.rs fix in this round, `Engine::run`'s `global_best` only
    /// updated from the generator stage's own `eval.evaluate(&offspring)`
    /// call and from `Restart` re-init, so this adapter-issued evaluate was
    /// invisible to it and `best_f` would NOT be exactly `0.0`.
    #[test]
    fn adapter_evaluated_individual_is_captured_in_global_best() {
        let (mut reg, mut spec, p) = setup();
        let shift = vec![1.0, -2.0];
        {
            let shift = shift.clone();
            reg.register_adapter("plant-optimum", move |_| {
                Ok(Box::new(PlantOptimum { shift: shift.clone() }) as Box<dyn Adapter>)
            });
        }
        spec.stages[0].adapter = Some(ComponentSpec { kind: "plant-optimum".into(), params: serde_json::json!({}) });
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();

        assert_eq!(r.best_f, 0.0,
            "the adapter-planted exact optimum must be captured as RunResult::best_f");
        let BlockValues::Float(xs) = &r.best_x.blocks[0] else { panic!("expected a float block") };
        assert_eq!(xs, &shift, "RunResult::best_x must be the adapter-planted point");
    }

    /// Regression test (M2d-3 Task 8, fix round 2): an adapter's OWN
    /// evaluation can itself be what triggers `target` termination
    /// (`reached(&eval)`, checked once per stage, right after that stage's
    /// adapter call) -- the exact case fix round 1's end-of-generation-only
    /// scan (placed AFTER the whole per-stage `for` loop, i.e. after BOTH
    /// `break 'outer` sites) missed: the target-hit `break 'outer` fires
    /// before an end-of-loop scan ever runs, dropping the adapter-planted
    /// point from `global_best` even though it is exactly the point that
    /// satisfied the target. `target` is set to the exact optimum value
    /// (`0.0`) so `PlantOptimum`'s own `ctx.eval.evaluate(..)` call -- not
    /// any earlier generator-stage evaluate -- is what flips
    /// `reached(&eval)` to true, on the very first stage of the very first
    /// generation (`resample`ing continuously over `[-5,5]^2` has
    /// vanishingly improbable odds of hitting `f=0.0` by chance first).
    #[test]
    fn adapter_triggered_target_hit_still_captures_the_planted_optimum() {
        let (mut reg, mut spec, p) = setup();
        let shift = vec![1.0, -2.0];
        {
            let shift = shift.clone();
            reg.register_adapter("plant-optimum", move |_| {
                Ok(Box::new(PlantOptimum { shift: shift.clone() }) as Box<dyn Adapter>)
            });
        }
        spec.stages[0].adapter = Some(ComponentSpec { kind: "plant-optimum".into(), params: serde_json::json!({}) });
        spec.termination.target = Some(0.0); // the adapter's own evaluate() must be what triggers this
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();

        assert_eq!(r.iterations, 0,
            "target should be hit within the FIRST stage of the first generation, via the adapter's own evaluate -- never reaching the bottom of the outer loop");
        assert_eq!(r.best_f, 0.0,
            "the adapter-planted optimum must be captured as RunResult::best_f even when its own evaluation is what triggers target termination");
        let BlockValues::Float(xs) = &r.best_x.blocks[0] else { panic!("expected a float block") };
        assert_eq!(xs, &shift, "RunResult::best_x must be the adapter-planted point");
    }

    struct FireAt3 { new_pop_size: usize }
    impl Restart for FireAt3 {
        fn check(&self, _pop: &Population, ctx: &mut Ctx) -> Option<RestartDirective> {
            if ctx.iteration == 3 { Some(RestartDirective { new_pop_size: self.new_pop_size }) } else { None }
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta::new("fire-at-3", SupportedBlocks::All)
        }
    }

    struct SizeProbe(std::sync::Arc<std::sync::Mutex<usize>>);
    impl Adapter for SizeProbe {
        fn adapt(&self, pop: &mut Population, _ctx: &mut Ctx) {
            *self.0.lock().unwrap() = pop.len();
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta::new("size-probe", SupportedBlocks::All)
        }
    }

    #[test]
    fn restart_reinitializes_at_new_size() {
        let (mut reg, mut spec, p) = setup();
        reg.register_restart("fire-at-3", |_| Ok(Box::new(FireAt3 { new_pop_size: 7 }) as Box<dyn Restart>));
        spec.restart = Some(ComponentSpec { kind: "fire-at-3".into(), params: serde_json::json!({}) });

        // Baseline: the same spec/seed, but with the budget capped exactly at
        // the eval count reached right before the restart's re-init fires
        // (10 init + 4 stage passes * 10 = 50). Since the adapter/restart
        // streams are independent of the generator/replacer/boundary streams,
        // this baseline's global best is bit-identical to the full run's
        // global best at the moment restart fires — i.e. "the best seen
        // before the restart".
        let mut baseline_spec = spec.clone();
        baseline_spec.termination.budget = 50;
        let e_baseline = Engine::from_spec(&baseline_spec, &reg, p.space()).unwrap();
        let r_baseline = e_baseline.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();

        let probe = std::sync::Arc::new(std::sync::Mutex::new(0usize));
        {
            let probe = probe.clone();
            reg.register_adapter("size-probe", move |_| Ok(Box::new(SizeProbe(probe.clone())) as Box<dyn Adapter>));
        }
        spec.stages[0].adapter = Some(ComponentSpec { kind: "size-probe".into(), params: serde_json::json!({}) });
        spec.termination.budget = 500;
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();

        assert_eq!(*probe.lock().unwrap(), 7,
            "population size must be 7 after the restart fires");
        assert!(r.best_f <= r_baseline.best_f,
            "global best after restart ({}) must be at least as good as before it ({})",
            r.best_f, r_baseline.best_f);
    }

    #[test]
    fn restart_budget_exhaustion_is_clean() {
        let (mut reg, mut spec, p) = setup();
        // A directive requesting a huge population, fired when only a
        // handful of evaluations remain: the engine must return Ok with the
        // pre-restart global best rather than erroring.
        reg.register_restart("fire-at-3-huge", |_| Ok(Box::new(FireAt3 { new_pop_size: 10_000 }) as Box<dyn Restart>));
        spec.restart = Some(ComponentSpec { kind: "fire-at-3-huge".into(), params: serde_json::json!({}) });
        spec.termination.budget = 55; // 10 init + 4*10 stage evals = 50, leaves only 5 spare
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
        assert_eq!(r.evals_used, 50,
            "the failed restart re-init batch must not be partially evaluated");
    }

    struct EveryTwoIters {
        fires: std::sync::Arc<std::sync::atomic::AtomicU64>,
        observed: std::sync::Arc<std::sync::Mutex<Vec<u64>>>,
    }
    impl Restart for EveryTwoIters {
        fn check(&self, pop: &Population, ctx: &mut Ctx) -> Option<RestartDirective> {
            let count = match ctx.bb.get_mut::<u64>("restart_counter") {
                Some(c) => { *c += 1; *c }
                None => { ctx.bb.insert("restart_counter", 1u64); 1 }
            };
            self.observed.lock().unwrap().push(count);
            if ctx.iteration > 0 && ctx.iteration.is_multiple_of(2) {
                self.fires.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Some(RestartDirective { new_pop_size: pop.len() })
            } else {
                None
            }
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta::new("every-two-iters", SupportedBlocks::All)
                .with_provides(vec![StateReq::of::<u64>("restart_counter")])
        }
    }

    /// Controller ruling (plan T12): when a restart fires, the engine clears
    /// the blackboard but preserves the restart component's own declared
    /// state (its `meta().provides` keys). This test's restart component
    /// increments a bb-resident counter on every `check` call, every single
    /// iteration of the whole run — a sequence that can only stay strictly
    /// increasing (never resetting to 1) if that counter survives every
    /// clear-and-restore cycle across repeated restarts.
    #[test]
    fn restart_state_survives_clear() {
        let (mut reg, mut spec, p) = setup();
        let fires = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let observed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        {
            let fires = fires.clone();
            let observed = observed.clone();
            reg.register_restart("every-two-iters", move |_| {
                Ok(Box::new(EveryTwoIters { fires: fires.clone(), observed: observed.clone() }) as Box<dyn Restart>)
            });
        }
        spec.restart = Some(ComponentSpec { kind: "every-two-iters".into(), params: serde_json::json!({}) });
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
        assert!(r.iterations > 4, "test needs enough iterations for multiple restarts to be meaningful");

        let fire_count = fires.load(std::sync::atomic::Ordering::SeqCst);
        assert!(fire_count > 1, "restart should have fired more than once, got {}", fire_count);

        let seen = observed.lock().unwrap();
        for w in seen.windows(2) {
            assert_eq!(w[1], w[0] + 1,
                "restart_counter must survive the blackboard clear across restarts, got sequence {:?}", *seen);
        }
    }

    // ---- Multi-stage engine-interaction tests (M2d-4 Task 8: TLBO is the
    // first preset with more than one `[[stages]]` entry -- these exercise
    // the generic engine machinery, not any specific algorithm, using the
    // same minimal test components as the rest of this module. ----

    fn two_stage_spec(base: &AlgorithmSpec) -> AlgorithmSpec {
        let mut s = base.clone();
        s.stages.push(base.stages[0].clone());
        s
    }

    /// (a) Budget accounting across two stages per generation: EVERY stage
    /// evaluates, so one generation of a two-stage spec consumes
    /// `2 * pop_size` evaluations, on top of the one-time `pop_size` init
    /// batch. Chosen so the budget divides EXACTLY (no partial generation),
    /// making the expected `evals_used`/`iterations` arithmetic exact.
    #[test]
    fn two_stage_budget_accounting_consumes_2x_pop_per_generation() {
        let (reg, spec, p) = setup(); // pop_size = 10 (see setup())
        let mut spec = two_stage_spec(&spec);
        let generations = 4u64;
        let pop_size = spec.pop_size as u64;
        spec.termination.budget = pop_size + generations * 2 * pop_size; // 10 + 4*20 = 90
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
        assert_eq!(r.evals_used, spec.termination.budget,
            "an exactly-divisible budget must be used up exactly: init + generations*2*pop_size");
        assert_eq!(r.iterations, generations,
            "each iteration must run BOTH stages (2*pop_size evals) before incrementing");
    }

    /// A generator that plants a caller-given genotype for every offspring
    /// (deterministic, consumes no RNG) -- used by (b) to force stage 0 to
    /// evaluate the exact optimum.
    struct PlantExactGen { point: Vec<f64> }
    impl Generator for PlantExactGen {
        fn generate(&self, pop: &Population, _c: &mut Ctx) -> Vec<Genotype> {
            (0..pop.len())
                .map(|_| Genotype { blocks: vec![BlockValues::Float(self.point.clone())] })
                .collect()
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta::new("plant-exact-gen", SupportedBlocks::All)
        }
    }

    /// A generator that increments a shared counter every time it is
    /// called, so a test can prove it was (or was not) invoked -- used by
    /// (b) as stage 1, which must NEVER run once stage 0's own evaluation
    /// already satisfies `target`.
    struct CountingGen(std::sync::Arc<std::sync::atomic::AtomicU64>);
    impl Generator for CountingGen {
        fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            (0..pop.len()).map(|_| uniform_sample(ctx.space, ctx.rng)).collect()
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta::new("counting-gen", SupportedBlocks::All)
        }
    }

    /// (b) A target-hit triggered by STAGE 0's own generator-evaluate call
    /// must stop the run cleanly BEFORE stage 1 ever runs (the per-stage
    /// `reached()` check, right after stage 0's own adapter-scan slot --
    /// same placement the M2d-3 Task 8 engine fix established for
    /// adapter-triggered hits, exercised here for the plain generator-stage
    /// path with a SECOND stage present). `setup()`'s problem is
    /// `SphereShifted` with optimum at `shift = [1.0, -2.0]` (f=0.0 there).
    #[test]
    fn target_hit_in_stage_0_stops_before_stage_1_runs() {
        let (mut reg, spec, p) = setup();
        let shift = vec![1.0, -2.0];
        let stage1_calls = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        {
            let shift = shift.clone();
            reg.register_generator("plant-exact-gen", move |_| {
                Ok(Box::new(PlantExactGen { point: shift.clone() }) as Box<dyn Generator>)
            });
        }
        {
            let stage1_calls = stage1_calls.clone();
            reg.register_generator("counting-gen", move |_| {
                Ok(Box::new(CountingGen(stage1_calls.clone())) as Box<dyn Generator>)
            });
        }
        let mut spec = spec;
        spec.stages[0].generator = ComponentSpec { kind: "plant-exact-gen".into(), params: serde_json::json!({}) };
        spec.stages.push(StageSpec {
            generator: ComponentSpec { kind: "counting-gen".into(), params: serde_json::json!({}) },
            replacer: ComponentSpec { kind: "greedy".into(), params: serde_json::json!({}) },
            adapter: None,
        });
        spec.termination.target = Some(0.0); // stage 0's own evaluate() must be what triggers this

        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();

        assert_eq!(r.iterations, 0,
            "target must be hit within stage 0 of the FIRST generation, never reaching the bottom of the outer loop");
        assert_eq!(stage1_calls.load(std::sync::atomic::Ordering::SeqCst), 0,
            "stage 1's generator must NEVER run once stage 0's own evaluate already satisfied target");
        assert_eq!(r.best_f, 0.0, "the exact optimum planted by stage 0 must be captured as best_f");
        let BlockValues::Float(xs) = &r.best_x.blocks[0] else { panic!("expected a float block") };
        assert_eq!(xs, &shift);
        assert_eq!(r.evals_used, spec.pop_size as u64 * 2,
            "only the init batch plus stage 0's own evaluate -- stage 1 must not have evaluated anything");
    }

    /// A generator that records every raw `next_f64()` draw it consumes
    /// (in order) into a shared log, and draws EXACTLY ONE value per
    /// offspring (no further RNG consumption -- unlike `uniform_sample`,
    /// which would draw once per dimension and break the 1-draw-per-
    /// individual stride this test relies on) -- used by (c) to compare a
    /// stage's ACTUAL consumed stream against an independently-
    /// reconstructed `RngStream` at that stage's documented path.
    struct ProbeGen(std::sync::Arc<std::sync::Mutex<Vec<f64>>>);
    impl Generator for ProbeGen {
        fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
            let BlockValues::Float(x0) = &pop.individuals[0].blocks[0] else {
                unreachable!("this test only uses float blocks")
            };
            let dim = x0.len();
            (0..pop.len()).map(|_| {
                let v = ctx.rng.next_f64();
                self.0.lock().unwrap().push(v);
                Genotype { blocks: vec![BlockValues::Float(vec![v; dim])] }
            }).collect()
        }
        fn meta(&self) -> ComponentMeta {
            ComponentMeta::new("probe-gen", SupportedBlocks::All)
        }
    }

    /// (c) Stage RNG streams: per the engine's `stage_rngs` derivation
    /// (`[run_id, 1 + 2*i]` for stage `i`'s generator stream), stage 0's
    /// generator must draw from `RngStream::from_master(seed, &[run_id,
    /// 1])` and stage 1's generator from `RngStream::from_master(seed,
    /// &[run_id, 3])` -- entirely independent streams. This test documents
    /// AND verifies both indices directly, by reconstructing each stream
    /// standalone and comparing its raw draws to what each stage's probe
    /// actually consumed.
    #[test]
    fn stage_rng_streams_use_the_documented_per_stage_indices() {
        let (mut reg, spec, p) = setup();
        let log0 = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let log1 = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        {
            let log0 = log0.clone();
            reg.register_generator("probe-gen-0", move |_| {
                Ok(Box::new(ProbeGen(log0.clone())) as Box<dyn Generator>)
            });
        }
        {
            let log1 = log1.clone();
            reg.register_generator("probe-gen-1", move |_| {
                Ok(Box::new(ProbeGen(log1.clone())) as Box<dyn Generator>)
            });
        }
        let mut spec = spec;
        spec.stages[0].generator = ComponentSpec { kind: "probe-gen-0".into(), params: serde_json::json!({}) };
        spec.stages.push(StageSpec {
            generator: ComponentSpec { kind: "probe-gen-1".into(), params: serde_json::json!({}) },
            replacer: ComponentSpec { kind: "greedy".into(), params: serde_json::json!({}) },
            adapter: None,
        });
        spec.termination.budget = spec.pop_size as u64 * 3; // init + exactly ONE generation (2 stages)

        let seed = 99u64;
        let run_id = 5u64;
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: seed, run_id }, None).unwrap();
        assert_eq!(r.iterations, 1, "exactly one generation must have run");

        // No `target` is set, so the outer `while !reached(&eval)` loop
        // always attempts a SECOND generation, and `gen.generate()` runs
        // speculatively BEFORE the budget check that then fails inside
        // `eval.evaluate(&offspring)` (see `Engine::run`'s `'outer` loop) --
        // so stage 0's probe logs `pop_size` MORE draws from that doomed
        // second attempt (stage 1's probe never runs a second time, since
        // the `break 'outer` on stage 0's failed evaluate happens first).
        // Only the FIRST `pop_size` entries of stage 0's log belong to the
        // one completed generation this test is about.
        let got0_all = log0.lock().unwrap().clone();
        let got1 = log1.lock().unwrap().clone();
        assert_eq!(got1.len(), spec.pop_size);
        assert!(got0_all.len() >= spec.pop_size);
        let got0 = got0_all[..spec.pop_size].to_vec();
        assert_ne!(got0, got1, "the two stages' generator streams must be independent, not the same sequence");

        // Documented path: stage i's generator stream is [run_id, 1 + 2*i].
        let mut expected0 = RngStream::from_master(seed, &[run_id, 1]); // stage 0
        let expected0: Vec<f64> = (0..spec.pop_size).map(|_| expected0.next_f64()).collect();
        assert_eq!(got0, expected0,
            "stage 0's generator must draw from RngStream::from_master(seed, &[run_id, 1])");

        let mut expected1 = RngStream::from_master(seed, &[run_id, 3]); // stage 1 (1 + 2*1 = 3)
        let expected1: Vec<f64> = (0..spec.pop_size).map(|_| expected1.next_f64()).collect();
        assert_eq!(got1, expected1,
            "stage 1's generator must draw from RngStream::from_master(seed, &[run_id, 3])");
    }

    // ---- genotype shape guard ----

    /// Two-block problem (Float n=2, Binary n=3); fitness = sum of floats.
    struct TwoBlock { space: SearchSpace }
    impl TwoBlock {
        fn new() -> Self {
            Self { space: SearchSpace::new(vec![
                Block::Float { lo: -5.0, hi: 5.0, n: 2 },
                Block::Binary { n: 3 },
            ]).unwrap() }
        }
    }
    impl Problem for TwoBlock {
        fn space(&self) -> &SearchSpace { &self.space }
        fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
            pop.iter().map(|g| match &g.blocks[0] {
                BlockValues::Float(xs) => xs.iter().sum(),
                _ => f64::INFINITY,
            }).collect()
        }
    }

    fn good_two_block() -> Genotype {
        Genotype { blocks: vec![
            BlockValues::Float(vec![1.0, 2.0]),
            BlockValues::Bin(vec![true, false, true]),
        ]}
    }

    /// Initializer/generator emitting a fixed-shape genotype.
    #[derive(Clone)]
    struct FixedShape(Genotype);
    impl Initializer for FixedShape {
        fn initialize(&self, n: usize, _c: &mut Ctx) -> Vec<Genotype> { vec![self.0.clone(); n] }
        fn meta(&self) -> ComponentMeta { ComponentMeta::new("fixed-init", SupportedBlocks::All) }
    }
    impl Generator for FixedShape {
        fn generate(&self, pop: &Population, _c: &mut Ctx) -> Vec<Genotype> {
            vec![self.0.clone(); pop.len()]
        }
        fn meta(&self) -> ComponentMeta { ComponentMeta::new("fixed-gen", SupportedBlocks::All) }
    }

    fn shape_run(init: Genotype, gen: Genotype) -> Result<RunResult, EngineError> {
        let mut reg = Registry::new();
        reg.register_initializer("fixed-init", move |_| Ok(Box::new(FixedShape(init.clone()))));
        reg.register_generator("fixed-gen", move |_| Ok(Box::new(FixedShape(gen.clone()))));
        reg.register_replacer("greedy", |_| Ok(Box::new(Greedy)));
        reg.register_boundary("no-b", |_| Ok(Box::new(NoB)));
        let spec = AlgorithmSpec {
            name: "shape".into(), pop_size: 4,
            init: ComponentSpec { kind: "fixed-init".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "no-b".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "fixed-gen".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "greedy".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 40, target: None },
            restart: None,
        };
        let p = TwoBlock::new();
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        e.run(&p, RunConfig { master_seed: 1, run_id: 0 }, None)
    }

    struct CorruptingAdapter;
    impl Adapter for CorruptingAdapter {
        fn adapt(&self, pop: &mut Population, _c: &mut Ctx) {
            // Shape-corrupt one individual without evaluating anything.
            pop.individuals[0] = Genotype { blocks: vec![BlockValues::Float(vec![1.0])] };
        }
        fn meta(&self) -> ComponentMeta { ComponentMeta::new("corrupt", SupportedBlocks::All) }
    }

    #[test]
    fn adapter_corrupting_shape_fails_fast() {
        let mut reg = Registry::new();
        reg.register_initializer("fixed-init", |_| Ok(Box::new(FixedShape(good_two_block()))));
        reg.register_generator("fixed-gen", |_| Ok(Box::new(FixedShape(good_two_block()))));
        reg.register_replacer("greedy", |_| Ok(Box::new(Greedy)));
        reg.register_boundary("no-b", |_| Ok(Box::new(NoB)));
        reg.register_adapter("corrupt", |_| Ok(Box::new(CorruptingAdapter) as Box<dyn Adapter>));
        let spec = AlgorithmSpec {
            name: "shape".into(), pop_size: 4,
            init: ComponentSpec { kind: "fixed-init".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "no-b".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "fixed-gen".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "greedy".into(), params: serde_json::json!({}) },
                adapter: Some(ComponentSpec { kind: "corrupt".into(), params: serde_json::json!({}) }),
            }],
            termination: TerminationSpec { budget: 400, target: None },
            restart: None,
        };
        let p = TwoBlock::new();
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        match e.run(&p, RunConfig { master_seed: 1, run_id: 0 }, None) {
            Err(EngineError::GenotypeShapeMismatch { stage, detail }) => {
                assert_eq!(stage, 0);
                assert!(detail.contains("expected 2") && detail.contains("got 1"), "{detail}");
            }
            other => panic!("expected GenotypeShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn flat_generator_on_multiblock_space_is_rejected() {
        let flat = Genotype { blocks: vec![BlockValues::Float(vec![1.0, 2.0])] };
        match shape_run(good_two_block(), flat) {
            Err(EngineError::GenotypeShapeMismatch { stage, detail }) => {
                assert_eq!(stage, 0);
                assert!(detail.contains("expected 2") && detail.contains("got 1"), "{detail}");
            }
            other => panic!("expected GenotypeShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn compound_shaped_generator_on_multiblock_space_runs() {
        let r = shape_run(good_two_block(), good_two_block()).unwrap();
        assert_eq!(r.best_x.blocks.len(), 2);
    }

    #[test]
    fn bad_init_shape_reports_init_stage() {
        let flat = Genotype { blocks: vec![BlockValues::Float(vec![1.0, 2.0])] };
        match shape_run(flat, good_two_block()) {
            Err(EngineError::GenotypeShapeMismatch { stage, .. }) => assert_eq!(stage, INIT_STAGE),
            other => panic!("expected init-site GenotypeShapeMismatch, got {other:?}"),
        }
    }

    #[test]
    fn right_block_count_wrong_length_is_rejected() {
        let bad = Genotype { blocks: vec![
            BlockValues::Float(vec![1.0, 2.0]),
            BlockValues::Bin(vec![true, false]), // n should be 3
        ]};
        match shape_run(good_two_block(), bad) {
            Err(EngineError::GenotypeShapeMismatch { stage, detail }) => {
                assert_eq!(stage, 0);
                assert!(detail.contains("block 1") && detail.contains("length 3")
                        && detail.contains("length 2"), "{detail}");
            }
            other => panic!("expected GenotypeShapeMismatch, got {other:?}"),
        }
    }

    /// Initializer that is shape-correct on its first call and emits a flat
    /// (wrong) shape on every later call, i.e. on restart re-initialization.
    struct FlipInit { calls: std::sync::atomic::AtomicUsize }
    impl Initializer for FlipInit {
        fn initialize(&self, n: usize, _c: &mut Ctx) -> Vec<Genotype> {
            let k = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let g = if k == 0 { good_two_block() }
                    else { Genotype { blocks: vec![BlockValues::Float(vec![1.0, 2.0])] } };
            vec![g; n]
        }
        fn meta(&self) -> ComponentMeta { ComponentMeta::new("flip-init", SupportedBlocks::All) }
    }

    #[test]
    fn restart_reinit_with_wrong_shape_is_rejected() {
        let mut reg = Registry::new();
        reg.register_initializer("flip-init", |_| Ok(Box::new(FlipInit {
            calls: std::sync::atomic::AtomicUsize::new(0) })));
        reg.register_generator("fixed-gen", |_| Ok(Box::new(FixedShape(good_two_block()))));
        reg.register_replacer("greedy", |_| Ok(Box::new(Greedy)));
        reg.register_boundary("no-b", |_| Ok(Box::new(NoB)));
        reg.register_restart("fire-at-3", |_| Ok(Box::new(FireAt3 { new_pop_size: 0 }) as Box<dyn Restart>));
        let spec = AlgorithmSpec {
            name: "flip".into(), pop_size: 4,
            init: ComponentSpec { kind: "flip-init".into(), params: serde_json::json!({}) },
            boundary: ComponentSpec { kind: "no-b".into(), params: serde_json::json!({}) },
            stages: vec![StageSpec {
                generator: ComponentSpec { kind: "fixed-gen".into(), params: serde_json::json!({}) },
                replacer: ComponentSpec { kind: "greedy".into(), params: serde_json::json!({}) },
                adapter: None,
            }],
            termination: TerminationSpec { budget: 500, target: None },
            restart: Some(ComponentSpec { kind: "fire-at-3".into(), params: serde_json::json!({}) }),
        };
        let p = TwoBlock::new();
        let e = Engine::from_spec(&spec, &reg, p.space()).unwrap();
        match e.run(&p, RunConfig { master_seed: 1, run_id: 0 }, None) {
            Err(EngineError::GenotypeShapeMismatch { stage, .. }) => assert_eq!(stage, INIT_STAGE),
            other => panic!("expected restart-site GenotypeShapeMismatch, got {other:?}"),
        }
    }
}
