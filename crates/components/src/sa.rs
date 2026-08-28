use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::Genotype;

pub struct MetropolisReplacer {
    pub t0: f64,
    pub alpha: f64,
}

impl MetropolisReplacer {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams {
            kind: "replace/metropolis".into(),
            reason,
        };

        let t0 = p.get("t0").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let alpha = p.get("alpha").and_then(|v| v.as_f64()).unwrap_or(0.995);

        if !t0.is_finite() || t0 <= 0.0 {
            return Err(err(format!(
                "t0 must be positive and finite, got {}",
                t0
            )));
        }
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(err(format!(
                "alpha must be in open interval (0, 1), got {}",
                alpha
            )));
        }

        Ok(Self { t0, alpha })
    }
}

impl Replacer for MetropolisReplacer {
    fn replace(&self, pop: &mut Population, oi: Vec<Genotype>, of: Vec<f64>, ctx: &mut Ctx) {
        let t = (self.t0 * self.alpha.powi(ctx.iteration as i32)).max(1e-12);

        for (i, (gi, fi)) in oi.into_iter().zip(of).enumerate() {
            if i < pop.len() {
                let delta = fi - pop.fitness[i];
                let accept = if delta < 0.0 {
                    true
                } else {
                    ctx.rng.next_f64() < (-delta / t).exp()
                };

                if accept {
                    pop.individuals[i] = gi;
                    pop.fitness[i] = fi;
                }
            }
        }
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta {
            kind: "replace/metropolis",
            supported_blocks: SupportedBlocks::All,
            requires: vec![],
            provides: vec![],
        }
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_replacer("replace/metropolis", |p| {
        Ok(Box::new(MetropolisReplacer::from_params(p)?))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::Population;
    use sezgi_core::space::{BlockValues, Genotype};

    fn g(x: f64) -> Genotype {
        Genotype {
            blocks: vec![BlockValues::Float(vec![x])],
        }
    }

    #[test]
    fn metropolis_always_accepts_improvement() {
        let mut pop = Population {
            individuals: vec![g(1.0), g(2.0)],
            fitness: vec![10.0, 20.0],
        };

        // Offspring with strictly better fitness (lower is better)
        let offspring = vec![g(1.5), g(2.5)];
        let offspring_f = vec![5.0, 8.0];

        let replacer = MetropolisReplacer::from_params(&serde_json::json!({})).unwrap();

        // Test with any seed
        use sezgi_core::problem::{SphereShifted, Problem};
        use sezgi_core::rng::RngStream;
        use sezgi_core::state::Blackboard;

        let p = SphereShifted::new(vec![0.0], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = sezgi_core::problem::Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(42, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx {
            space,
            rng: &mut rng,
            eval: &mut evaluator,
            bb: &mut bb,
            iteration: 0,
        };

        replacer.replace(&mut pop, offspring, offspring_f, &mut ctx);

        // Both offspring should be accepted since they're better (delta < 0)
        assert_eq!(pop.fitness, vec![5.0, 8.0]);
    }

    #[test]
    fn metropolis_param_validation() {
        // t0 = 0.0 should be rejected
        assert!(MetropolisReplacer::from_params(&serde_json::json!({"t0": 0.0})).is_err());

        // t0 < 0 should be rejected
        assert!(MetropolisReplacer::from_params(&serde_json::json!({"t0": -1.0})).is_err());

        // alpha = 0.0 should be rejected
        assert!(MetropolisReplacer::from_params(&serde_json::json!({"alpha": 0.0})).is_err());

        // alpha = 1.0 should be rejected
        assert!(MetropolisReplacer::from_params(&serde_json::json!({"alpha": 1.0})).is_err());

        // alpha > 1.0 should be rejected
        assert!(MetropolisReplacer::from_params(&serde_json::json!({"alpha": 1.5})).is_err());

        // Valid params should work
        let r = MetropolisReplacer::from_params(&serde_json::json!({"t0": 1.5, "alpha": 0.99}));
        assert!(r.is_ok());
    }
}
