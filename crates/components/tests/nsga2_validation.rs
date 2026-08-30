//! Task 8 (M3-2) + Task 1 (M3-3): NSGA-II reference validation run.
//!
//! Runs the merged NSGA-II runner (`sezgi_components::nsga2::nsga2_run`,
//! T6) against the ZDT1/ZDT2/ZDT3/ZDT4 (T2) and DTLZ2 (T2) benchmark
//! suites, with the paper-pinned standard settings (T6's `Nsga2Config` doc:
//! SBX `eta_c=20`, polynomial mutation `eta_m=20`, `p_c=0.9`, `p_m=None` ->
//! `1/n_variables` -- Deb et al. 2002, "A Fast and Elitist Multiobjective
//! Genetic Algorithm: NSGA-II", *IEEE TEC* 6(2), Sec. IV.A's own quoted
//! experimental settings), `pop_size=100` (a multiple of 4, per T6's
//! `popsize % 4 == 0` requirement), `budget=25_000` evaluations. ZDT4
//! (M3-3 Task 1, carried deferral) is the MIXED-BOUNDS case (`x1 in
//! [0,1]`, `x2,...,x10 in [-5,5]`, two `Block::Float`s) and additionally
//! asserts every final individual's variables land within those bounds --
//! guarding `nsga2_run`'s per-variable bounds extraction end-to-end, not
//! just its convergence.
//!
//! # Anchored-threshold discipline (this task's brief)
//!
//! Every assert below is anchored to a VALUE MEASURED FROM THIS
//! IMPLEMENTATION (recorded in the per-test comments), rounded to a
//! round-number threshold with comfortable headroom -- not chased to the
//! literature's own numbers. Context only: Deb et al. 2002 Table V reports
//! a "convergence metric" (their own gamma/Eq. 4 distance-to-a-500-point
//! discretized front measure, NOT this crate's IGD, and computed against
//! their own C implementation's operators/RNG) of order 1e-3 to 6e-3 (mean)
//! for ZDT1/ZDT2/ZDT3 at generation 250 (pop 100, so also a 25,000-eval
//! budget) in their Table V. That is CONTEXT for "the metric magnitude is
//! in the right ballpark," not a target: our IGD values below (Ishibuchi et
//! al. 2015's p=1 mean-nearest-distance metric, `sezgi_stats::igd`) are a
//! DIFFERENT metric on a DIFFERENT (though behaviorally similar)
//! implementation, so an apples-to-apples numeric comparison is not
//! meaningful and is deliberately not attempted here.
//!
//! # Multi-seed
//!
//! 3 DISTINCT seeds per problem (1, 2, 3), each asserted INDIVIDUALLY (no
//! seed is averaged into another, and no cross-algorithm comparison is
//! made -- this is one algorithm against fixed anchors, which the M2b
//! controller ruling explicitly allows, unlike single-seed cross-algorithm
//! comparative claims).
//!
//! # Hypervolume reference points
//!
//! ZDT1/ZDT2/ZDT3/ZDT4 (`sezgi_problems::Zdt`) map `x1 in [0,1]` directly
//! to `f1`, and every measured converged front0 here has `f1 in [0,
//! ~1.0]`; ZDT1/ZDT2/ZDT4's `f2` also lands in `[0, ~1.01]` (`f2 = g*h`,
//! converged `g~1`, `h in [0,1]`) but ZDT3's `h` includes a `sin` term that
//! pushes converged `f2` as low as about -0.77 (still comfortably above
//! the DTLZ-style "unbounded" case). `ref_point = [1.1, 1.1]` was verified
//! (via a scratch probe run at all 3 seeds) to sit strictly outside every
//! converged front0's `f1` AND `f2` range for all four problems (measured
//! maxima: ZDT1 f1<=0.9999, f2<=1.0013; ZDT2 f1<=1.0000, f2<=1.0030; ZDT3
//! f1<=0.8525, f2<=1.0025; ZDT4 f1<=0.9998, f2<=1.0098), so it is used
//! uniformly for all four. DTLZ2 here uses `m=3` objectives, so
//! `hypervolume_2d` (2D-only by design, see `sezgi_stats::moo_indicators`'s
//! module doc: "General-M (M > 2 objectives) hypervolume requires the WFG
//! algorithm... explicitly OUT of scope here") does not apply -- DTLZ2 is
//! validated by IGD only.
//!
//! # Budget/runtime
//!
//! All 15 runs (5 problems x 3 seeds) at `pop=100`/`budget=25_000` complete
//! in well under 1 second combined in `--release` (measured: ~0.6s total on
//! the development machine, ZDT4's own 3 runs well under 100ms of that) --
//! nowhere near the brief's ~60s ceiling, so the paper's own
//! `pop=100`/`budget=25_000` (Deb et al. 2002 Sec. IV.A: "a population size
//! of 100 [and] the algorithms were run for 250 generations") setting is
//! used AS-IS, not reduced.

use sezgi_components::nsga2::{nsga2_run, Nsga2Config};
use sezgi_core::mo::MoProblem;
use sezgi_core::space::BlockValues;
use sezgi_problems::{Dtlz, Zdt};
use sezgi_stats::{hypervolume_2d, igd};

const POP_SIZE: usize = 100;
const BUDGET: u64 = 25_000;
const SEEDS: [u64; 3] = [1, 2, 3];
const ZDT_REF_POINT: [f64; 2] = [1.1, 1.1];

fn standard_cfg(seed: u64) -> Nsga2Config {
    // T6-pinned standard settings (Deb et al. 2002 Sec. IV.A): SBX eta_c=20,
    // polynomial-mutation eta_m=20, p_c=0.9, p_m=None -> 1/n_variables.
    Nsga2Config { pop_size: POP_SIZE, budget: BUDGET, seed, eta_c: 20.0, eta_m: 20.0, p_c: 0.9, p_m: None }
}

fn front0_objectives(result: &sezgi_components::nsga2::MoRunResult) -> Vec<Vec<f64>> {
    result.front0.iter().map(|&i| result.objectives[i].clone()).collect()
}

// ---- ZDT1 (dim=30): convex front, f2 = 1 - sqrt(f1) at g=1 -------------

#[test]
fn nsga2_reference_zdt1() {
    let problem = Zdt::new(1, 30).unwrap();
    let reference = problem.pareto_front(500).unwrap();

    // Measured at seeds 1/2/3 (scratch probe, pop=100, budget=25000):
    //   seed=1: igd=0.0048697802 hv=0.8694170884
    //   seed=2: igd=0.0048561454 hv=0.8694992442
    //   seed=3: igd=0.0049627449 hv=0.8693303039
    // IGD anchor: round UP from the max measured (~0.00496) to 0.01 (>2x
    // headroom). HV anchor: round DOWN from the min measured (~0.86933) to
    // 0.8 (~0.069 absolute headroom, ~8%).
    for &seed in &SEEDS {
        let result = nsga2_run(&problem, &standard_cfg(seed)).unwrap();
        let front0 = front0_objectives(&result);
        let igd_val = igd(&front0, &reference).unwrap();
        let hv_val = hypervolume_2d(&front0, &ZDT_REF_POINT).unwrap();
        assert!(igd_val < 0.01, "ZDT1 seed={seed}: measured IGD {igd_val} exceeds anchored threshold 0.01");
        assert!(hv_val > 0.8, "ZDT1 seed={seed}: measured HV {hv_val} below anchored threshold 0.8");
    }
}

// ---- ZDT2 (dim=30): nonconvex front, f2 = 1 - f1^2 at g=1 --------------

#[test]
fn nsga2_reference_zdt2() {
    let problem = Zdt::new(2, 30).unwrap();
    let reference = problem.pareto_front(500).unwrap();

    // Measured at seeds 1/2/3 (scratch probe, pop=100, budget=25000):
    //   seed=1: igd=0.0057804324 hv=0.5349867848
    //   seed=2: igd=0.0050811601 hv=0.5358517959
    //   seed=3: igd=0.0052496322 hv=0.5362012265
    // IGD anchor: round UP from the max measured (~0.00578) to 0.01 (>1.7x
    // headroom). HV anchor: round DOWN from the min measured (~0.53499) to
    // 0.5 (~0.035 absolute headroom, ~6.5%).
    for &seed in &SEEDS {
        let result = nsga2_run(&problem, &standard_cfg(seed)).unwrap();
        let front0 = front0_objectives(&result);
        let igd_val = igd(&front0, &reference).unwrap();
        let hv_val = hypervolume_2d(&front0, &ZDT_REF_POINT).unwrap();
        assert!(igd_val < 0.01, "ZDT2 seed={seed}: measured IGD {igd_val} exceeds anchored threshold 0.01");
        assert!(hv_val > 0.5, "ZDT2 seed={seed}: measured HV {hv_val} below anchored threshold 0.5");
    }
}

// ---- ZDT3 (dim=30): disconnected front, includes a sin term -------------

#[test]
fn nsga2_reference_zdt3() {
    let problem = Zdt::new(3, 30).unwrap();
    let reference = problem.pareto_front(500).unwrap();

    // Measured at seeds 1/2/3 (scratch probe, pop=100, budget=25000):
    //   seed=1: igd=0.0052712900 hv=1.3272551314
    //   seed=2: igd=0.0053503052 hv=1.3275529650
    //   seed=3: igd=0.0058141850 hv=1.3269902510
    // (hv > 1 here is expected and correct: ZDT3's converged front dips to
    // f2 as low as ~-0.77, so with ref_point=[1.1,1.1] the enclosed
    // rectangles' total height routinely exceeds 1 -- see the module doc's
    // "Hypervolume reference points" section.)
    // IGD anchor: round UP from the max measured (~0.00581) to 0.01 (>1.7x
    // headroom). HV anchor: round DOWN from the min measured (~1.32699) to
    // 1.2 (~0.127 absolute headroom, ~9.6%).
    for &seed in &SEEDS {
        let result = nsga2_run(&problem, &standard_cfg(seed)).unwrap();
        let front0 = front0_objectives(&result);
        let igd_val = igd(&front0, &reference).unwrap();
        let hv_val = hypervolume_2d(&front0, &ZDT_REF_POINT).unwrap();
        assert!(igd_val < 0.01, "ZDT3 seed={seed}: measured IGD {igd_val} exceeds anchored threshold 0.01");
        assert!(hv_val > 1.2, "ZDT3 seed={seed}: measured HV {hv_val} below anchored threshold 1.2");
    }
}

// ---- DTLZ2 (m=3, dim=12): unit-sphere front, IGD only (no 2D hypervolume) --

#[test]
fn nsga2_reference_dtlz2_m3() {
    // k=10 is DTLZ2's own chapter-quoted standard distance-group size
    // (crates/problems/src/dtlz.rs module doc); dim = m + k - 1 = 3+10-1 = 12.
    let problem = Dtlz::new(2, 3, 12).unwrap();
    let reference = problem.pareto_front(1000).unwrap();

    // Measured at seeds 1/2/3 (scratch probe, pop=100, budget=25000):
    //   seed=1: igd=0.0701752952
    //   seed=2: igd=0.0635028641
    //   seed=3: igd=0.0678758310
    // 3-objective front: hypervolume_2d does not apply here (2D-only by
    // design -- see the module doc's "Hypervolume reference points"
    // section), so this problem is validated by IGD only.
    // IGD anchor: round UP from the max measured (~0.07018) to 0.1 (~0.03
    // absolute headroom, ~42%).
    for &seed in &SEEDS {
        let result = nsga2_run(&problem, &standard_cfg(seed)).unwrap();
        let front0 = front0_objectives(&result);
        let igd_val = igd(&front0, &reference).unwrap();
        assert!(igd_val < 0.1, "DTLZ2 m=3 seed={seed}: measured IGD {igd_val} exceeds anchored threshold 0.1");
    }
}

// ---- ZDT4 (dim=10): mixed-bounds case, x1 in [0,1], x2..x10 in [-5,5] --
// (Task 1, M3-3: this is the test's whole point -- nsga2_run's bounds
// extraction is exercised end-to-end against a REAL two-block mixed-bound
// space, not just ZDT1-3/DTLZ2's uniform-bound ones. Front shape matches
// ZDT1's, g=1: f2 = 1 - sqrt(f1) -- see crates/problems/src/zdt.rs's module
// doc and `zdt4_pareto_front_shape_matches_zdt1`.)

#[test]
fn nsga2_reference_zdt4() {
    let problem = Zdt::new(4, 10).unwrap();
    let reference = problem.pareto_front(500).unwrap();

    // Measured at seeds 1/2/3 (scratch probe, pop=100, budget=25000):
    //   seed=1: igd=0.0079982877 hv=0.8624179760 f1<=0.999731 f2<=1.009773
    //   seed=2: igd=0.0055300303 hv=0.8675769151 f1<=0.999831 f2<=1.005066
    //   seed=3: igd=0.0071724697 hv=0.8635810105 f1<=0.998880 f2<=1.009193
    // (TDD: this test was first written with placeholder-tight anchors --
    // igd<0.0001, hv>999.0 -- run, observed failing on the real measured
    // values above via the assertion message, then the anchors below were
    // hand-set from that measurement, per this task's brief.)
    // IGD anchor: round UP from the max measured (~0.00800) to 0.02 (2.5x
    // headroom -- more than ZDT1-3's ~1.7-2x, since ZDT4's multimodal g
    // (Rastrigin-like, T4 in crates/problems/src/zdt.rs's module doc) makes
    // per-run variance a real risk this suite must tolerate). HV anchor:
    // round DOWN from the min measured (~0.86242) to 0.8 (~0.062 absolute
    // headroom, ~7.2%), matching ZDT1's own 0.8 floor for the same
    // front-shape family (f2 = 1 - sqrt(f1) at g=1, see the module doc's
    // "ZDT4: same front shape as ZDT1" section). ref_point=[1.1,1.1] (same
    // `ZDT_REF_POINT` as ZDT1-3, per the module doc's "Hypervolume
    // reference points" section) is verified to sit strictly outside every
    // measured front0's f1/f2 range above (max f1<=0.999831, max
    // f2<=1.009773, both < 1.1).
    for &seed in &SEEDS {
        let result = nsga2_run(&problem, &standard_cfg(seed)).unwrap();

        // (a) bounds: every final individual's x1 in [0,1], x2..x10 in
        // [-5,5] -- the mixed-bounds path this test exists to guard.
        for individual in &result.individuals {
            match (individual.blocks.first(), individual.blocks.get(1)) {
                (Some(BlockValues::Float(x1)), Some(BlockValues::Float(rest))) => {
                    assert_eq!(x1.len(), 1, "ZDT4 seed={seed}: expected 1 x1 variable, got {x1:?}");
                    assert!(
                        (0.0..=1.0).contains(&x1[0]),
                        "ZDT4 seed={seed}: x1={} out of [0,1]",
                        x1[0]
                    );
                    assert_eq!(rest.len(), 9, "ZDT4 seed={seed}: expected 9 rest variables, got {rest:?}");
                    for &xi in rest {
                        assert!((-5.0..=5.0).contains(&xi), "ZDT4 seed={seed}: xi={xi} out of [-5,5]");
                    }
                }
                other => panic!("ZDT4 seed={seed}: expected two Float blocks, got {other:?}"),
            }
        }

        let front0 = front0_objectives(&result);
        let igd_val = igd(&front0, &reference).unwrap();
        let hv_val = hypervolume_2d(&front0, &ZDT_REF_POINT).unwrap();
        assert!(igd_val < 0.02, "ZDT4 seed={seed}: measured IGD {igd_val} exceeds anchored threshold 0.02");
        assert!(hv_val > 0.8, "ZDT4 seed={seed}: measured HV {hv_val} below anchored threshold 0.8");
    }
}
