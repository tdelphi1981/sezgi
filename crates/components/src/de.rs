use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::{BlockValues, Genotype};
use sezgi_core::state::StateReq;

#[derive(Debug, Clone, Copy)]
pub enum DeStrategy { Rand1, Best1 }

pub struct DeGenerator { pub strategy: DeStrategy, pub f: f64, pub cr: f64 }

impl DeGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams {
            kind: "gen/de".into(), reason };
        let strategy = match p.get("strategy").and_then(|v| v.as_str()).unwrap_or("rand1") {
            "rand1" => DeStrategy::Rand1,
            "best1" => DeStrategy::Best1,
            s => return Err(err(format!("unknown strategy: {s}"))),
        };
        let f = p.get("f").and_then(|v| v.as_f64()).unwrap_or(0.5);
        let cr = p.get("cr").and_then(|v| v.as_f64()).unwrap_or(0.9);
        if !(0.0..=2.0).contains(&f) { return Err(err(format!("f outside [0,2]: {f}"))); }
        if !(0.0..=1.0).contains(&cr) { return Err(err(format!("cr outside [0,1]: {cr}"))); }
        Ok(Self { strategy, f, cr })
    }

    fn float_view(g: &Genotype) -> &Vec<f64> {
        match &g.blocks[0] { BlockValues::Float(xs) => xs, _ => unreachable!() }
    }
}

impl Generator for DeGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 4, "gen/de requires a population of at least 4 (pop_size={})", pop.len());
        let n = pop.len();
        let best = pop.best_index().unwrap_or(0);
        (0..n).map(|i| {
            let mut pick_distinct = |excluded: &[usize]| loop {
                let r = ctx.rng.next_below(n as u64) as usize;
                if r != i && !excluded.contains(&r) { break r; }
            };

            let (base, r2, r3) = match self.strategy {
                DeStrategy::Rand1 => {
                    // Draw three indices: r1, r2, r3 all distinct from each other and from i
                    let r1 = pick_distinct(&[]);
                    let r2 = pick_distinct(&[r1]);
                    let r3 = pick_distinct(&[r1, r2]);
                    (Self::float_view(&pop.individuals[r1]), r2, r3)
                },
                DeStrategy::Best1 => {
                    // Draw two indices: r2, r3 distinct from each other, from i, and from best
                    let r2 = pick_distinct(&[best]);
                    let r3 = pick_distinct(&[best, r2]);
                    (Self::float_view(&pop.individuals[best]), r2, r3)
                },
            };

            let a = Self::float_view(&pop.individuals[r2]);
            let b = Self::float_view(&pop.individuals[r3]);
            let target = Self::float_view(&pop.individuals[i]);
            let d = target.len();
            let j_rand = ctx.rng.next_below(d as u64) as usize;
            let xs = (0..d).map(|j| {
                if j == j_rand || ctx.rng.next_f64() < self.cr {
                    base[j] + self.f * (a[j] - b[j])
                } else { target[j] }
            }).collect();
            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect()
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/de", SupportedBlocks::Only(vec!["float"])).with_min_pop(4)
    }
}

/// jDE: self-adaptive DE/rand/1/bin (Brest et al. 2006).
///
/// Each individual carries its own `F_i`/`CR_i`, stored on the blackboard as
/// `jde_f`/`jde_cr` (parallel to the population, indexed by `i`). Before
/// generating individual `i`'s trial, the parameters have a small chance
/// (`tau1`, `tau2`) of being redrawn:
///
/// - `u1 = rng.next_f64()`; if `u1 < tau1`, `F_i = f_lower + f_upper * rng.next_f64()`.
/// - `u2 = rng.next_f64()`; if `u2 < tau2`, `CR_i = rng.next_f64()`.
///
/// This draw order is pinned (part of the RNG-stream contract): F before CR,
/// always both draws (the "coin flip" `u1`/`u2` is consumed even when it
/// doesn't trigger an update), then the usual rand/1 donor draws and
/// binomial crossover using the (possibly just-updated) `F_i`/`CR_i`.
///
/// **Survival coupling.** Classic jDE keeps the new `F_i`/`CR_i` only when
/// the trial replaces the parent (i.e. the mutation is "good"). sezgi has no
/// generator-replacer coupling, so this is implemented as a companion
/// adapter (`adapter/jde-commit`, below) that runs after the replacer: the
/// generator snapshots the pre-update `F`/`CR`/fitness into
/// `jde_f_prev`/`jde_cr_prev`/`jde_fit_prev`, and the adapter rolls back
/// `jde_f[i]`/`jde_cr[i]` to the `_prev` values wherever the parent survived
/// (`pop.fitness[i]` unchanged from `jde_fit_prev[i]`).
pub struct JdeGenerator { pub f_lower: f64, pub f_upper: f64, pub tau1: f64, pub tau2: f64 }

impl JdeGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams {
            kind: "gen/de-jde".into(), reason };
        let f_lower = p.get("f_lower").and_then(|v| v.as_f64()).unwrap_or(0.1);
        let f_upper = p.get("f_upper").and_then(|v| v.as_f64()).unwrap_or(0.9);
        let tau1 = p.get("tau1").and_then(|v| v.as_f64()).unwrap_or(0.1);
        let tau2 = p.get("tau2").and_then(|v| v.as_f64()).unwrap_or(0.1);
        if !f_lower.is_finite() || f_lower <= 0.0 {
            return Err(err(format!("f_lower must be finite and > 0: {f_lower}")));
        }
        if !f_upper.is_finite() || f_upper <= 0.0 {
            return Err(err(format!("f_upper must be finite and > 0: {f_upper}")));
        }
        if !tau1.is_finite() || !(0.0..=1.0).contains(&tau1) {
            return Err(err(format!("tau1 must be finite and in [0,1]: {tau1}")));
        }
        if !tau2.is_finite() || !(0.0..=1.0).contains(&tau2) {
            return Err(err(format!("tau2 must be finite and in [0,1]: {tau2}")));
        }
        Ok(Self { f_lower, f_upper, tau1, tau2 })
    }

    fn float_view(g: &Genotype) -> &Vec<f64> {
        match &g.blocks[0] { BlockValues::Float(xs) => xs, _ => unreachable!() }
    }
}

impl Generator for JdeGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 4, "gen/de-jde requires a population of at least 4 (pop_size={})", pop.len());
        let n = pop.len();

        // Lazily initialize on first call; re-initialize if the population
        // size changed underneath us (e.g. a restart resized the pop).
        let needs_init = ctx.bb.get::<Vec<f64>>("jde_f").map(|v| v.len() != n).unwrap_or(true);
        if !needs_init {
            // Since 0.1.6 the survival rollback lives here (it used to run in
            // `adapter/jde-commit` at the end of the previous iteration): an
            // individual whose fitness is unchanged from the pre-generation
            // snapshot kept its parent, so its F/CR revert to the pre-update
            // values. RNG-free.
            let prev = (
                ctx.bb.get::<Vec<f64>>("jde_fit_prev").cloned(),
                ctx.bb.get::<Vec<f64>>("jde_f_prev").cloned(),
                ctx.bb.get::<Vec<f64>>("jde_cr_prev").cloned(),
            );
            if let (Some(fit_prev), Some(f_prev), Some(cr_prev)) = prev {
                let m = n.min(fit_prev.len()).min(f_prev.len()).min(cr_prev.len());
                let survived = |i: usize| pop.fitness[i].total_cmp(&fit_prev[i])
                    == std::cmp::Ordering::Equal;
                if let Some(f) = ctx.bb.get_mut::<Vec<f64>>("jde_f") {
                    for i in 0..m { if survived(i) { f[i] = f_prev[i]; } }
                }
                if let Some(cr) = ctx.bb.get_mut::<Vec<f64>>("jde_cr") {
                    for i in 0..m { if survived(i) { cr[i] = cr_prev[i]; } }
                }
            }
        }
        if needs_init {
            ctx.bb.insert("jde_f", vec![0.5f64; n]);
            ctx.bb.insert("jde_cr", vec![0.9f64; n]);
        }

        // Snapshot pre-update state BEFORE the per-individual loop: the
        // commit adapter needs both the values-as-they-were and the
        // fitness-as-it-was to detect "did the parent survive".
        let f_prev = ctx.bb.get::<Vec<f64>>("jde_f").unwrap().clone();
        let cr_prev = ctx.bb.get::<Vec<f64>>("jde_cr").unwrap().clone();
        let fit_prev = pop.fitness.clone();

        let mut f_new = f_prev.clone();
        let mut cr_new = cr_prev.clone();

        let offspring: Vec<Genotype> = (0..n).map(|i| {
            // Pinned draw order: u1/F, then u2/CR, then the rand/1 donors.
            let u1 = ctx.rng.next_f64();
            if u1 < self.tau1 {
                f_new[i] = self.f_lower + self.f_upper * ctx.rng.next_f64();
            }
            let u2 = ctx.rng.next_f64();
            if u2 < self.tau2 {
                cr_new[i] = ctx.rng.next_f64();
            }

            let mut pick_distinct = |excluded: &[usize]| loop {
                let r = ctx.rng.next_below(n as u64) as usize;
                if r != i && !excluded.contains(&r) { break r; }
            };

            let r1 = pick_distinct(&[]);
            let r2 = pick_distinct(&[r1]);
            let r3 = pick_distinct(&[r1, r2]);

            let base = Self::float_view(&pop.individuals[r1]);
            let a = Self::float_view(&pop.individuals[r2]);
            let b = Self::float_view(&pop.individuals[r3]);
            let target = Self::float_view(&pop.individuals[i]);
            let d = target.len();
            let j_rand = ctx.rng.next_below(d as u64) as usize;
            let (f_i, cr_i) = (f_new[i], cr_new[i]);
            let xs = (0..d).map(|j| {
                if j == j_rand || ctx.rng.next_f64() < cr_i {
                    base[j] + f_i * (a[j] - b[j])
                } else { target[j] }
            }).collect();
            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect();

        ctx.bb.insert("jde_f", f_new);
        ctx.bb.insert("jde_cr", cr_new);
        ctx.bb.insert("jde_f_prev", f_prev);
        ctx.bb.insert("jde_cr_prev", cr_prev);
        ctx.bb.insert("jde_fit_prev", fit_prev);

        offspring
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/de-jde", SupportedBlocks::Only(vec!["float"]))
            .with_provides(vec![
                StateReq::of::<Vec<f64>>("jde_f"),
                StateReq::of::<Vec<f64>>("jde_cr"),
                StateReq::of::<Vec<f64>>("jde_f_prev"),
                StateReq::of::<Vec<f64>>("jde_cr_prev"),
                StateReq::of::<Vec<f64>>("jde_fit_prev"),
            ])
            .with_min_pop(4)
    }
}

/// Companion adapter for `gen/de-jde`.
///
/// Since 0.1.6 the survival-coupled F/CR rollback lives in `gen/de-jde` (it
/// runs at the top of the next `generate()` call, comparing the committed
/// population's fitness against `jde_fit_prev`); this component is kept for
/// spec compatibility and is a total no-op.
pub struct JdeCommitAdapter;

impl Adapter for JdeCommitAdapter {
    fn adapt(&self, _pop: &mut Population, _ctx: &mut Ctx) {}
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("adapter/jde-commit", SupportedBlocks::All)
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/de", |p| Ok(Box::new(DeGenerator::from_params(p)?)));
    reg.register_generator("gen/de-jde", |p| Ok(Box::new(JdeGenerator::from_params(p)?)));
    reg.register_adapter("adapter/jde-commit", |_| Ok(Box::new(JdeCommitAdapter)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Problem, SphereShifted, Evaluator};
    use sezgi_core::rng::RngStream;
    use sezgi_core::state::Blackboard;

    #[test]
    fn params_parse_and_validate() {
        let g = DeGenerator::from_params(&serde_json::json!(
            {"strategy": "best1", "f": 0.7, "cr": 0.8})).unwrap();
        assert!(matches!(g.strategy, DeStrategy::Best1));
        assert!(DeGenerator::from_params(&serde_json::json!({"f": -1.0})).is_err());
        assert!(DeGenerator::from_params(&serde_json::json!({"cr": 2.0})).is_err());
    }

    #[test]
    #[should_panic(expected = "at least 4")]
    fn small_population_panics() {
        use sezgi_core::problem::Population;

        let p = SphereShifted::new(vec![0.0], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(42, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };

        // Create 3-individual population to trigger panic
        let pop = Population {
            individuals: vec![
                Genotype { blocks: vec![BlockValues::Float(vec![0.0])] },
                Genotype { blocks: vec![BlockValues::Float(vec![0.0])] },
                Genotype { blocks: vec![BlockValues::Float(vec![0.0])] },
            ],
            fitness: vec![0.0; 3],
        };

        let gen = DeGenerator::from_params(&serde_json::json!({})).unwrap();
        let _ = gen.generate(&pop, &mut ctx);
    }

    #[test]
    fn best1_draws_two_donors_distinct_from_best() {
        use sezgi_core::problem::Population;

        let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(42, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };

        // Create 6-individual population with known fitness so best is deterministic
        let pop = Population {
            individuals: vec![
                Genotype { blocks: vec![BlockValues::Float(vec![0.0; 5])] },
                Genotype { blocks: vec![BlockValues::Float(vec![1.0; 5])] },
                Genotype { blocks: vec![BlockValues::Float(vec![2.0; 5])] },
                Genotype { blocks: vec![BlockValues::Float(vec![3.0; 5])] },
                Genotype { blocks: vec![BlockValues::Float(vec![4.0; 5])] },
                Genotype { blocks: vec![BlockValues::Float(vec![5.0; 5])] },
            ],
            fitness: vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        };

        let gen = DeGenerator::from_params(
            &serde_json::json!({"strategy": "best1", "f": 0.5, "cr": 0.9})).unwrap();
        let offspring = gen.generate(&pop, &mut ctx);
        // Should not panic and should return pop-size offspring
        assert_eq!(offspring.len(), pop.len());
    }

    #[test]
    fn jde_params_validate() {
        let g = JdeGenerator::from_params(&serde_json::json!(
            {"f_lower": 0.2, "f_upper": 0.8, "tau1": 0.05, "tau2": 0.05})).unwrap();
        assert_eq!(g.f_lower, 0.2);
        assert_eq!(g.f_upper, 0.8);
        assert_eq!(g.tau1, 0.05);
        assert_eq!(g.tau2, 0.05);

        // Defaults per Brest et al. 2006.
        let d = JdeGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!((d.f_lower, d.f_upper, d.tau1, d.tau2), (0.1, 0.9, 0.1, 0.1));

        assert!(JdeGenerator::from_params(&serde_json::json!({"f_lower": 0.0})).is_err());
        assert!(JdeGenerator::from_params(&serde_json::json!({"f_lower": -0.1})).is_err());
        assert!(JdeGenerator::from_params(&serde_json::json!({"f_upper": 0.0})).is_err());
        assert!(JdeGenerator::from_params(&serde_json::json!({"f_upper": -1.0})).is_err());
        assert!(JdeGenerator::from_params(&serde_json::json!({"tau1": -0.1})).is_err());
        assert!(JdeGenerator::from_params(&serde_json::json!({"tau1": 1.1})).is_err());
        assert!(JdeGenerator::from_params(&serde_json::json!({"tau2": -0.1})).is_err());
        assert!(JdeGenerator::from_params(&serde_json::json!({"tau2": 1.1})).is_err());
    }

    #[test]
    fn jde_generate_rolls_back_on_parent_survival() {
        use sezgi_core::problem::Population;

        let p = SphereShifted::new(vec![0.0; 2], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(42, &[]);
        let mut bb = Blackboard::new();

        fn g(x: f64) -> Genotype { Genotype { blocks: vec![BlockValues::Float(vec![x, x])] } }

        // 4 individuals. Post-replacement fitness: 0 and 2 are unchanged from
        // jde_fit_prev (the parent survived); 1 and 3 improved (the trial won).
        let jde_fit_prev = vec![10.0, 20.0, 30.0, 40.0];
        let mut pop = Population {
            individuals: vec![g(0.0), g(1.0), g(2.0), g(3.0)],
            fitness: vec![10.0, 5.0, 30.0, 1.0],
        };

        bb.insert("jde_f", vec![0.9, 0.9, 0.9, 0.9]);
        bb.insert("jde_cr", vec![0.3, 0.3, 0.3, 0.3]);
        bb.insert("jde_f_prev", vec![0.5, 0.5, 0.5, 0.5]);
        bb.insert("jde_cr_prev", vec![0.9, 0.9, 0.9, 0.9]);
        bb.insert("jde_fit_prev", jde_fit_prev);

        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        // The companion adapter is a total no-op since 0.1.6.
        JdeCommitAdapter.adapt(&mut pop, &mut ctx);
        assert_eq!(ctx.bb.get::<Vec<f64>>("jde_f").unwrap(), &vec![0.9, 0.9, 0.9, 0.9]);

        // The rollback now happens at the top of generate(). tau1 = tau2 = 0
        // so no F/CR resampling occurs; the snapshots written at the end of
        // generate() are the post-rollback values.
        let gen = JdeGenerator { f_lower: 0.1, f_upper: 0.9, tau1: 0.0, tau2: 0.0 };
        let _ = gen.generate(&pop, &mut ctx);
        let f = ctx.bb.get::<Vec<f64>>("jde_f").unwrap();
        let cr = ctx.bb.get::<Vec<f64>>("jde_cr").unwrap();
        // Rolled back exactly at 0 and 2 (survived); kept the new values at 1 and 3 (won).
        assert_eq!(f, &vec![0.5, 0.9, 0.5, 0.9]);
        assert_eq!(cr, &vec![0.9, 0.3, 0.9, 0.3]);
    }
}
