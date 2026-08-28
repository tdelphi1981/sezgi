use sezgi_core::component::*;
use sezgi_core::dist::Distribution;
use sezgi_core::problem::Population;
use sezgi_core::space::{BlockValues, Genotype};

/// Dağılım-parametreli adım mutasyonu: her float geni `rate` olasılıkla
/// x + dist.sample() olarak günceller. Gaussian → klasik ES mutasyonu,
/// Cauchy → hızlı ES, Levy → Lévy uçuşu (spec §6).
pub struct StepMutation { pub dist: Distribution, pub rate: f64 }

impl StepMutation {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams {
            kind: "gen/step".into(), reason };
        let dist: Distribution = serde_json::from_value(
            p.get("dist").cloned()
             .unwrap_or(serde_json::json!({"kind": "gaussian", "mean": 0.0, "sigma": 0.1})))
            .map_err(|e| err(e.to_string()))?;
        dist.validate().map_err(|reason| err(reason))?;
        let rate = p.get("rate").and_then(|v| v.as_f64()).unwrap_or(1.0);
        if !(0.0..=1.0).contains(&rate) {
            return Err(err(format!("rate [0,1] dışında: {rate}")));
        }
        Ok(Self { dist, rate })
    }
}

impl Generator for StepMutation {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        pop.individuals.iter().map(|g| {
            let mut child = g.clone();
            for v in &mut child.blocks {
                if let BlockValues::Float(xs) = v {
                    for x in xs.iter_mut() {
                        if ctx.rng.next_f64() < self.rate {
                            *x += self.dist.sample(ctx.rng);
                        }
                    }
                }
            }
            child
        }).collect()
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta { kind: "gen/step",
            supported_blocks: SupportedBlocks::Only(vec!["float"]),
            requires: vec![], provides: vec![] }
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/step", |p| Ok(Box::new(StepMutation::from_params(p)?)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::dist::Distribution;

    #[test]
    fn step_params_parse_from_json() {
        let p = serde_json::json!({
            "dist": {"kind": "cauchy", "loc": 0.0, "scale": 0.1}, "rate": 0.5
        });
        let s = StepMutation::from_params(&p).unwrap();
        assert_eq!(s.rate, 0.5);
        assert!(matches!(s.dist, Distribution::Cauchy { .. }));
    }

    #[test]
    fn bad_params_are_component_error() {
        let p = serde_json::json!({"dist": {"kind": "yok"}});
        assert!(StepMutation::from_params(&p).is_err());
    }

    #[test]
    fn invalid_dist_params_rejected_at_parse() {
        let p = serde_json::json!({"dist": {"kind": "levy", "alpha": 3.0}});
        assert!(matches!(StepMutation::from_params(&p),
            Err(sezgi_core::component::ComponentError::InvalidParams { .. })));
    }
}
