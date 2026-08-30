//! f0: the BIAS-toolbox "random function" null problem.
//!
//! `F0Random` has no landscape at all — every charged evaluation returns an
//! independent U(0,1) draw from a seeded stream, uncorrelated with the point
//! being queried. It exists purely as a probe: an algorithm with no
//! structural bias of its own should leave its final positions uniformly
//! distributed over the domain when run repeatedly against f0, so any
//! measured departure from uniformity is evidence about the ALGORITHM's
//! operators (boundary handling, update-rule geometry, etc.), never about
//! the fitness landscape (there isn't one).
//!
//! # Pinned behavioral contract
//!
//! The k-th charged evaluation of a given `F0Random` instance returns the
//! k-th draw of that instance's RNG stream, REGARDLESS of:
//! - the point `x` being evaluated (the value never depends on `x`);
//! - how evaluations are grouped into `evaluate_batch` calls (one call of 10
//!   and ten calls of 1 produce the identical 10-value sequence).
//!
//! This is exercised directly by this module's tests and MUST NOT change
//! silently — any future revision to the draw order is a pin change.
//!
//! # Domain
//!
//! The domain is `[0,1]^d` per the BIAS-toolbox convention, per dimension
//! `d = dim`. This is a single, named constant pair ([`F0_LO`], [`F0_HI`])
//! so a future source-verification pass (M3-1 Task 4) can retarget it in
//! one place if the verified toolbox method uses a different bound.
//!
//! # RNG stream path
//!
//! `F0Random::new` treats its own `seed` argument as the RNG master (never
//! the engine's `cfg.master_seed`/`run_id`), the same pattern
//! `sezgi_problems::BbobProblem` uses for its own per-function master
//! (`BBOB_SEED_BASE + fid`) — a problem owns and derives its own RNG
//! namespace, independent of whatever engine run later evaluates it. Under
//! that master, f0 draws from a single stream at path `[3_000_000]`.
//!
//! This cannot collide with the engine's per-run stream family
//! (`crates/core/src/engine.rs`: `[run_id, 0]` init, `[run_id, 1000]`
//! boundary, `[run_id, 1 + 2*i]`/`[run_id, 2 + 2*i]` per-stage
//! generator/replacer, `[run_id, 1_000_000 + i]` per-stage adapter,
//! `[run_id, 2_000_000]` restart) for two independent reasons: (1) a
//! `Problem`'s `seed` and the engine's `cfg.master_seed` are different,
//! caller-chosen values in normal use, and (2) EVEN IF a caller reuses the
//! same numeric value for both, `RngStream::from_master`'s path folding
//! makes collision impossible by construction — f0's single-element path
//! performs exactly one SplitMix64 fold from the shared master, while every
//! engine stream's two-element path performs a SECOND fold on top of that
//! (a second fold is never the identity of the first digest), so the final
//! internal state always differs. `3_000_000` also sits well above every
//! existing engine tag range (max `2_000_000 + restart-split`), leaving
//! headroom for future f1..fN bias-toolbox problems in this crate to claim
//! their own tags in the same style without approaching that boundary.

use std::sync::Mutex;

use sezgi_core::problem::Problem;
use sezgi_core::rng::RngStream;
use sezgi_core::space::{Block, Genotype, SearchSpace};

/// f0's domain lower bound, `[0,1]^d` per the BIAS-toolbox convention.
/// Change this (and [`F0_HI`]) in this one place if Task 4's source
/// verification finds the toolbox uses a different bound.
pub const F0_LO: f64 = 0.0;
/// f0's domain upper bound — see [`F0_LO`].
pub const F0_HI: f64 = 1.0;

pub struct F0Random {
    space: SearchSpace,
    rng: Mutex<RngStream>,
}

impl F0Random {
    pub fn new(dim: usize, seed: u64) -> Self {
        let space = SearchSpace::new(vec![Block::Float { lo: F0_LO, hi: F0_HI, n: dim }])
            .expect("F0_LO < F0_HI by construction, so SearchSpace::new cannot fail here");
        // stream path: [3_000_000] under this instance's own `seed` as master —
        // see the module doc's "RNG stream path" section for the collision analysis.
        let rng = RngStream::from_master(seed, &[3_000_000]);
        Self { space, rng: Mutex::new(rng) }
    }
}

impl Problem for F0Random {
    fn space(&self) -> &SearchSpace { &self.space }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
        let mut rng = self.rng.lock().expect("F0Random RNG mutex poisoned");
        pop.iter().map(|_| rng.next_f64()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{EvalObserver, Evaluator};
    use sezgi_core::space::BlockValues;
    use std::sync::Arc;

    fn g(xs: &[f64]) -> Genotype {
        Genotype { blocks: vec![BlockValues::Float(xs.to_vec())] }
    }

    #[test]
    fn deterministic_same_seed_same_sequence() {
        let a = F0Random::new(2, 7);
        let b = F0Random::new(2, 7);
        let pop = vec![g(&[0.1, 0.2]), g(&[0.3, 0.4]), g(&[0.9, 0.9])];
        assert_eq!(a.evaluate_batch(&pop), b.evaluate_batch(&pop));
    }

    #[test]
    fn values_independent_of_x() {
        let a = F0Random::new(2, 7);
        let b = F0Random::new(2, 7);
        let pop_a = vec![g(&[0.0, 0.0]), g(&[0.5, 0.5]), g(&[1.0, 1.0])];
        // Different points, same count/order: values must still match --
        // proof that only call order (not x) drives the draw.
        let pop_b = vec![g(&[0.99, 0.01]), g(&[0.2, 0.8]), g(&[0.5, 0.5])];
        assert_eq!(a.evaluate_batch(&pop_a), b.evaluate_batch(&pop_b));
    }

    #[test]
    fn kth_eval_matches_kth_raw_stream_draw() {
        let seed = 99;
        let p = F0Random::new(3, seed);
        let mut raw = RngStream::from_master(seed, &[3_000_000]);
        let expected: Vec<f64> = (0..5).map(|_| raw.next_f64()).collect();
        let pop: Vec<Genotype> = (0..5).map(|i| g(&[i as f64, i as f64, i as f64])).collect();
        assert_eq!(p.evaluate_batch(&pop), expected);
    }

    #[test]
    fn stream_order_independent_of_batch_grouping() {
        let seed = 5;
        let one_shot = {
            let p = F0Random::new(1, seed);
            let pop: Vec<Genotype> = (0..6).map(|i| g(&[i as f64 * 0.1])).collect();
            p.evaluate_batch(&pop)
        };
        let split = {
            let p = F0Random::new(1, seed);
            let mut out = vec![];
            for chunk in [2usize, 1, 3] {
                let pop: Vec<Genotype> = (0..chunk).map(|i| g(&[i as f64])).collect();
                out.extend(p.evaluate_batch(&pop));
            }
            out
        };
        assert_eq!(one_shot, split);
    }

    #[test]
    fn draws_within_declared_domain() {
        let p = F0Random::new(1, 4242);
        let pop: Vec<Genotype> = (0..1000).map(|i| g(&[(i as f64) / 1000.0])).collect();
        let fs = p.evaluate_batch(&pop);
        assert!(fs.iter().all(|&x| (F0_LO..F0_HI).contains(&x)),
            "next_f64 is documented as [0,1), so every f0 draw must land in [F0_LO, F0_HI)");
    }

    #[test]
    fn space_matches_declared_domain_constants() {
        let p = F0Random::new(4, 1);
        match &p.space().blocks()[0] {
            Block::Float { lo, hi, n } => {
                assert_eq!(*lo, F0_LO);
                assert_eq!(*hi, F0_HI);
                assert_eq!(*n, 4);
            }
            other => panic!("expected a single Float block, got {other:?}"),
        }
    }

    #[test]
    fn evaluator_integration_counts_tracks_and_observes() {
        struct Probe(Arc<Mutex<Vec<u64>>>);
        impl EvalObserver for Probe {
            fn on_eval(&mut self, i: u64, _f: f64, _best: f64) { self.0.lock().unwrap().push(i); }
        }
        let p = F0Random::new(2, 123);
        let seen = Arc::new(Mutex::new(vec![]));
        let mut ev = Evaluator::new(&p, 10);
        ev.set_observer(Box::new(Probe(seen.clone())));
        let pop = vec![g(&[0.1, 0.1]), g(&[0.2, 0.2]), g(&[0.3, 0.3])];
        let fs = ev.evaluate(&pop).unwrap();

        assert_eq!(ev.used(), 3);
        assert_eq!(*seen.lock().unwrap(), vec![1, 2, 3]);

        let expected_best = fs.iter().cloned().fold(f64::INFINITY, f64::min);
        assert_eq!(ev.best_so_far(), Some(expected_best));

        let bx = ev.best_x_so_far().expect("best_x_so_far must be set after a successful evaluate");
        let BlockValues::Float(best_xs) = &bx.blocks[0] else { panic!("expected float block") };
        assert!(pop.iter().any(|cand| {
            let BlockValues::Float(cand_xs) = &cand.blocks[0] else { unreachable!() };
            cand_xs == best_xs
        }), "best_x_so_far must be one of the population's own genotypes");
    }

    #[test]
    fn distribution_sanity_10k_draws() {
        let p = F0Random::new(1, 2026);
        let pop: Vec<Genotype> = (0..10_000).map(|i| g(&[(i as f64) / 10_000.0])).collect();
        let fs = p.evaluate_batch(&pop);
        let mean: f64 = fs.iter().sum::<f64>() / fs.len() as f64;
        // Measured mean at seed=2026, n=10_000: 0.503289303825652.
        // U(0,1)'s own natural per-draw spread is std = 1/sqrt(12) ~= 0.2887
        // (the sample mean's spread, SE = std/sqrt(n) ~= 0.00289, is far
        // tighter, but a coarse sanity smoke test -- not the rigorous KS/AD
        // battery that lands in Task 3 -- should not assert down near SE
        // resolution). Bounds are anchored around the MEASURED value with
        // +/-0.1 headroom on each side: headroom (~0.097 to the measured
        // mean's low side, ~0.103 to its high side) clears 0.3 * std
        // (~0.0866), i.e. at least "0.3 natural-spread units" of headroom,
        // matching this project's other anchored-threshold tests'
        // ">=0.3 headroom" convention but expressed relative to f0's own
        // natural (per-draw) spread rather than an absolute optimization-gap
        // scale.
        assert!((0.40..0.60).contains(&mean),
            "f0 mean over 10k draws should be near 0.5 (U(0,1)); got {mean}");
    }
}
