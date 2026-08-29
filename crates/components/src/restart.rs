//! `restart/stagnation`: IPOP/BIPOP-style stagnation restarts.
//!
//! Watches the evaluator's global-best-so-far (`ctx.eval.best_so_far()`) for
//! `patience` evaluations without a strict improvement (> `1e-12`), and when
//! that window elapses, asks the engine (via `RestartDirective`) to
//! re-initialize the population at a new size — chosen by `Sizing`:
//!
//! - `Ipop { factor, max_pop }`: every restart doubles (or `factor`-scales)
//!   the current population size, capped at `max_pop` — the classic IPOP-CMA-ES
//!   scheme (Auger & Hansen 2005), monotonically growing the population to
//!   trade exploration for exploitation as restarts accumulate.
//! - `Bipop { base, factor, max_pop }`: a **documented simplification** of
//!   BIPOP-CMA-ES's budget-based regime alternation (Hansen 2009), which in
//!   the original picks whichever of a "small" or "large" population regime
//!   has consumed less of the remaining budget so far. sezgi has no
//!   per-regime budget accounting in this component, so it alternates
//!   strictly by restart parity instead: odd `restart_count` uses the small
//!   `base` population, even `restart_count` uses the IPOP-style
//!   `factor`-scaled (capped) population.
//!
//! **Blackboard state** (all owned by this component; declared as `provides`
//! so the engine preserves them across the blackboard clear a restart
//! triggers — see `sezgi_core::engine`'s restart-state-preservation logic,
//! M2b Task 4):
//! - `restart_last_best`: `f64` — the best `ctx.eval.best_so_far()` value
//!   observed as of the last improvement (or the last restart fire).
//! - `restart_last_improve_eval`: `u64` — the evaluator's `used()` count as
//!   of the last improvement (or the last restart fire).
//! - `restart_count`: `u64` — number of restarts fired so far.
//!
//! **Lazy init** (first `check` call, i.e. these keys absent from the
//! blackboard): `restart_last_best` = the current best, `restart_last_improve_eval`
//! = the current `used()`, `restart_count` = 0. No restart fires on this call.
//!
//! **`check` per call thereafter:**
//! 1. `ctx.eval.best_so_far()` is `None` (no evals yet at all) → `None`, no state touched.
//! 2. Strict improvement (`last_best - best > 1e-12`) → update `restart_last_best`/
//!    `restart_last_improve_eval`, return `None`.
//! 3. Else, if `used - restart_last_improve_eval >= patience` → increment
//!    `restart_count`, reset `restart_last_improve_eval = used` and
//!    `restart_last_best = best` (so the next stagnation window starts
//!    fresh), and return `Some(RestartDirective { new_pop_size })` per the
//!    `Sizing` rule above (population size is read from the *current*
//!    population, `pop.len()`, not tracked separately).
//! 4. Else → `None`, no state touched.

use sezgi_core::component::{
    ComponentError, ComponentMeta, Ctx, Registry, Restart, RestartDirective, SupportedBlocks,
};
use sezgi_core::problem::Population;
use sezgi_core::state::StateReq;

/// How a fired restart picks the new population size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Sizing {
    Ipop { factor: f64, max_pop: usize },
    Bipop { base: usize, factor: f64, max_pop: usize },
}

pub struct StagnationRestart {
    /// Evaluations without a strict global-best improvement before a restart fires.
    pub patience: u64,
    pub sizing: Sizing,
}

fn ipop_size(current_pop: usize, factor: f64, max_pop: usize) -> usize {
    let scaled = (current_pop as f64 * factor).round();
    (scaled as usize).min(max_pop)
}

impl StagnationRestart {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams {
            kind: "restart/stagnation".into(),
            reason,
        };

        let patience = p
            .get("patience")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| err("patience must be a non-negative integer".into()))?;
        if patience < 1 {
            return Err(err(format!("patience must be >= 1: {patience}")));
        }

        let sizing_tag = p
            .get("sizing")
            .and_then(|v| v.as_str())
            .ok_or_else(|| err("sizing must be a string (\"ipop\" or \"bipop\")".into()))?;

        let factor = match p.get("factor") {
            None | Some(serde_json::Value::Null) => 2.0,
            Some(v) => v
                .as_f64()
                .ok_or_else(|| err("factor must be a number".into()))?,
        };
        if !(factor.is_finite() && factor > 1.0) {
            return Err(err(format!("factor must be finite and > 1: {factor}")));
        }

        let max_pop = p
            .get("max_pop")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| err("max_pop must be a non-negative integer".into()))?
            as usize;
        if max_pop < 1 {
            return Err(err(format!("max_pop must be >= 1: {max_pop}")));
        }

        let sizing = match sizing_tag {
            "ipop" => Sizing::Ipop { factor, max_pop },
            "bipop" => {
                let base = p
                    .get("base")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| err("bipop sizing requires \"base\" (non-negative integer)".into()))?
                    as usize;
                if base < 1 {
                    return Err(err(format!("base must be >= 1: {base}")));
                }
                Sizing::Bipop { base, factor, max_pop }
            }
            other => return Err(err(format!("unknown sizing kind: {other}"))),
        };

        Ok(Self { patience, sizing })
    }
}

impl Restart for StagnationRestart {
    fn check(&self, pop: &Population, ctx: &mut Ctx) -> Option<RestartDirective> {
        let best = ctx.eval.best_so_far()?;
        let used = ctx.eval.used();

        if !ctx.bb.contains("restart_last_best") {
            // Lazy init: first call ever, no restart fires yet.
            ctx.bb.insert("restart_last_best", best);
            ctx.bb.insert("restart_last_improve_eval", used);
            ctx.bb.insert("restart_count", 0u64);
            return None;
        }

        let last_best = *ctx.bb.get::<f64>("restart_last_best").unwrap();
        if last_best - best > 1e-12 {
            ctx.bb.insert("restart_last_best", best);
            ctx.bb.insert("restart_last_improve_eval", used);
            return None;
        }

        let last_improve_eval = *ctx.bb.get::<u64>("restart_last_improve_eval").unwrap();
        if used.saturating_sub(last_improve_eval) < self.patience {
            return None;
        }

        // Stagnation window elapsed: fire a restart.
        let count = ctx.bb.get::<u64>("restart_count").copied().unwrap_or(0) + 1;
        ctx.bb.insert("restart_count", count);
        // Reset the window so it starts fresh after the restart re-init.
        ctx.bb.insert("restart_last_improve_eval", used);
        ctx.bb.insert("restart_last_best", best);

        let current_pop = pop.len();
        let new_pop_size = match self.sizing {
            Sizing::Ipop { factor, max_pop } => ipop_size(current_pop, factor, max_pop),
            Sizing::Bipop { base, factor, max_pop } => {
                if count % 2 == 1 { base } else { ipop_size(current_pop, factor, max_pop) }
            }
        };
        Some(RestartDirective { new_pop_size })
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta {
            kind: "restart/stagnation",
            supported_blocks: SupportedBlocks::All,
            requires: vec![],
            provides: vec![
                StateReq::of::<f64>("restart_last_best"),
                StateReq::of::<u64>("restart_last_improve_eval"),
                StateReq::of::<u64>("restart_count"),
            ],
        }
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_restart("restart/stagnation", |p| {
        Ok(Box::new(StagnationRestart::from_params(p)?) as Box<dyn Restart>)
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Evaluator, Problem, SphereShifted};
    use sezgi_core::rng::RngStream;
    use sezgi_core::space::{BlockValues, Genotype};
    use sezgi_core::state::Blackboard;

    #[test]
    fn restart_params_validate() {
        assert!(StagnationRestart::from_params(&serde_json::json!(
            {"patience": 2000, "sizing": "ipop", "factor": 2.0, "max_pop": 512}
        )).is_ok());
        assert!(StagnationRestart::from_params(&serde_json::json!(
            {"patience": 2000, "sizing": "bipop", "base": 8, "factor": 2.0, "max_pop": 512}
        )).is_ok());

        // factor defaults to 2.0
        let r = StagnationRestart::from_params(&serde_json::json!(
            {"patience": 10, "sizing": "ipop", "max_pop": 100}
        )).unwrap();
        match r.sizing {
            Sizing::Ipop { factor, .. } => assert_eq!(factor, 2.0),
            _ => panic!("expected ipop sizing"),
        }

        // bad patience
        assert!(StagnationRestart::from_params(&serde_json::json!(
            {"patience": 0, "sizing": "ipop", "max_pop": 100}
        )).is_err());

        // bad factor
        assert!(StagnationRestart::from_params(&serde_json::json!(
            {"patience": 10, "sizing": "ipop", "factor": 1.0, "max_pop": 100}
        )).is_err());
        assert!(StagnationRestart::from_params(&serde_json::json!(
            {"patience": 10, "sizing": "ipop", "factor": 0.5, "max_pop": 100}
        )).is_err());

        // bad max_pop
        assert!(StagnationRestart::from_params(&serde_json::json!(
            {"patience": 10, "sizing": "ipop", "max_pop": 0}
        )).is_err());

        // bad bipop base
        assert!(StagnationRestart::from_params(&serde_json::json!(
            {"patience": 10, "sizing": "bipop", "base": 0, "factor": 2.0, "max_pop": 100}
        )).is_err());
    }

    fn plateau_point() -> Genotype {
        Genotype { blocks: vec![BlockValues::Float(vec![0.0, 0.0])] }
    }

    #[test]
    fn stagnation_fires_after_patience() {
        let restart = StagnationRestart { patience: 50, sizing: Sizing::Ipop { factor: 2.0, max_pop: 512 } };
        let p = SphereShifted::new(vec![0.0; 2], -5.0, 5.0);
        let space = p.space();
        let mut ev = Evaluator::new(&p, 10_000);
        let mut bb = Blackboard::new();
        let mut rng = RngStream::from_master(1, &[]);

        let point = plateau_point(); // f == 0 at every evaluation: a permanent fitness plateau
        let pop = Population { individuals: vec![point.clone(); 10], fitness: vec![0.0; 10] };

        // First batch establishes best_so_far (used=5).
        ev.evaluate(&vec![point.clone(); 5]).unwrap();
        {
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut ev, iteration: 0 };
            assert!(restart.check(&pop, &mut ctx).is_none(), "first call is lazy init, no fire");
        }
        assert_eq!(*bb.get::<u64>("restart_last_improve_eval").unwrap(), 5);
        assert_eq!(*bb.get::<f64>("restart_last_best").unwrap(), 0.0);
        assert_eq!(*bb.get::<u64>("restart_count").unwrap(), 0);

        // Advance used without improvement, short of patience (used=49, delta=44 < 50).
        ev.evaluate(&vec![point.clone(); 44]).unwrap();
        {
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut ev, iteration: 1 };
            assert!(restart.check(&pop, &mut ctx).is_none(), "should not fire before patience elapses");
        }

        // Cross the patience threshold (used=55, delta=50 >= 50).
        ev.evaluate(&vec![point.clone(); 6]).unwrap();
        let directive = {
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut ev, iteration: 2 };
            restart.check(&pop, &mut ctx)
        };
        let directive = directive.expect("should fire once the stagnation window elapses");
        assert_eq!(directive.new_pop_size, 20, "ipop should double the current pop size (10 -> 20)");
        assert_eq!(*bb.get::<u64>("restart_count").unwrap(), 1);
    }

    #[test]
    fn bipop_alternates_small_large() {
        let restart = StagnationRestart { patience: 1, sizing: Sizing::Bipop { base: 8, factor: 2.0, max_pop: 100 } };
        let p = SphereShifted::new(vec![0.0; 2], -5.0, 5.0);
        let space = p.space();
        let mut ev = Evaluator::new(&p, 10_000);
        let mut bb = Blackboard::new();
        let mut rng = RngStream::from_master(1, &[]);

        let point = plateau_point();
        let pop = Population { individuals: vec![point.clone(); 20], fitness: vec![0.0; 20] };

        ev.evaluate(std::slice::from_ref(&point)).unwrap(); // used=1
        {
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut ev, iteration: 0 };
            assert!(restart.check(&pop, &mut ctx).is_none(), "lazy init");
        }

        ev.evaluate(std::slice::from_ref(&point)).unwrap(); // used=2, delta=1 >= patience(1)
        let d1 = {
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut ev, iteration: 1 };
            restart.check(&pop, &mut ctx).expect("first fire")
        };
        assert_eq!(d1.new_pop_size, 8, "odd restart_count (1) uses the small base population");
        assert_eq!(*bb.get::<u64>("restart_count").unwrap(), 1);

        ev.evaluate(std::slice::from_ref(&point)).unwrap(); // used=3, delta=1 >= patience(1)
        let d2 = {
            let mut ctx = Ctx { space, rng: &mut rng, bb: &mut bb, eval: &mut ev, iteration: 2 };
            restart.check(&pop, &mut ctx).expect("second fire")
        };
        assert_eq!(d2.new_pop_size, 40, "even restart_count (2) uses the factor-scaled population (20*2)");
        assert_eq!(*bb.get::<u64>("restart_count").unwrap(), 2);
    }
}
