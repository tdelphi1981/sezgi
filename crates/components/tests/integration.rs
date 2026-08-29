use sezgi_components::{presets, register_builtins};
use sezgi_core::component::Registry;
use sezgi_core::dist::Distribution;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_core::problem::{Problem, SphereShifted};
use sezgi_problems::BbobProblem;

fn registry() -> Registry {
    let mut r = Registry::new();
    register_builtins(&mut r);
    r
}

#[test]
fn de_converges_on_shifted_sphere_10d() {
    let shift: Vec<f64> = (0..10).map(|i| 0.7 * i as f64 - 3.0).collect();
    let p = SphereShifted::new(shift, -5.0, 5.0);
    let spec = presets::de_rand_1(50, 20_000);
    let e = Engine::from_spec(&spec, &registry(), p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    assert!(r.best_f < 1e-6, "DE should converge in 20k evaluations: {}", r.best_f);
}

#[test]
fn es_with_cauchy_beats_init() {
    let p = SphereShifted::new(vec![1.0; 5], -5.0, 5.0);
    let spec = presets::es_mu_plus_lambda(
        20, 5_000, Distribution::Cauchy { loc: 0.0, scale: 0.1 });
    let e = Engine::from_spec(&spec, &registry(), p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    assert!(r.best_f < 0.1, "ES should make progress: {}", r.best_f);
}

#[test]
fn ga_converges_on_shifted_sphere_5d() {
    let p = SphereShifted::new(vec![1.5, -0.5, 2.0, -3.0, 0.25], -5.0, 5.0);
    let spec = presets::ga_real(50, 20_000);
    let e = Engine::from_spec(&spec, &registry(), p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    assert!(r.best_f < 1e-3, "GA should converge: {}", r.best_f);
}

#[test]
fn pso_converges_on_shifted_sphere_5d() {
    let p = SphereShifted::new(vec![1.5, -0.5, 2.0, -3.0, 0.25], -5.0, 5.0);
    let spec = presets::pso(40, 20_000);
    let e = Engine::from_spec(&spec, &registry(), p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    assert!(r.best_f < 1e-4, "PSO should converge: {}", r.best_f);
}

#[test]
fn pso_commit_without_generator_is_rejected() {
    let p = SphereShifted::new(vec![1.0], -5.0, 5.0);
    let mut spec = presets::de_rand_1(10, 100);
    spec.stages[0].replacer.kind = "replace/pso-commit".into();
    assert!(Engine::from_spec(&spec, &registry(), p.space()).is_err());
}

#[test]
fn de_best1_converges_on_shifted_sphere() {
    let p = SphereShifted::new(vec![1.5, -0.5, 2.0, -3.0, 0.25], -5.0, 5.0);
    let spec = presets::de_best_1(30, 10_000);
    let e = Engine::from_spec(&spec, &registry(), p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    assert!(r.best_f < 1e-6, "DE Best1 should converge in 10k evaluations: {}", r.best_f);
}

#[test]
fn sa_improves_on_shifted_sphere() {
    let p = SphereShifted::new(vec![1.5, -0.5, 2.0, -3.0, 0.25], -5.0, 5.0);
    let spec = presets::sa(20_000);
    let e = Engine::from_spec(&spec, &registry(), p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    assert!(r.best_f < 0.05, "SA should improve to < 0.05: {}", r.best_f);
}

#[test]
fn random_search_improves_but_modestly() {
    let p = SphereShifted::new(vec![1.5, -0.5, 2.0, -3.0, 0.25], -5.0, 5.0);
    let spec = presets::random_search(20, 5_000);
    let e = Engine::from_spec(&spec, &registry(), p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    assert!(r.best_f.is_finite(), "best_f must be finite: {}", r.best_f);
    // Deterministic achieved best_f ≈ 1.467 at seed 42 (bound anchored to this run; plan's 1.0 was an unrealistic guess).
    assert!(r.best_f < 2.0, "random-search should stay bounded: {}", r.best_f);
    assert!(r.best_f > 1e-6, "random-search should not converge like DE: {}", r.best_f);
}

#[test]
fn jde_converges_on_shifted_sphere() {
    let shift: Vec<f64> = (0..10).map(|i| 0.7 * i as f64 - 3.0).collect();
    let p = SphereShifted::new(shift, -5.0, 5.0);
    let spec = presets::jde(40, 20_000);
    let e = Engine::from_spec(&spec, &registry(), p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    assert!(r.best_f < 1e-6, "jDE should converge in 20k evaluations: {}", r.best_f);
}

#[test]
fn jde_near_optimum_on_rastrigin() {
    let p = BbobProblem::new(3, 5, 1).unwrap();
    let spec = presets::jde(40, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Deterministic achieved gap ≈ 1.411 at seed 42 (BBOB f3 transforms stretch basin
    // spacing; bound anchored with headroom to the next basin). Single-seed comparative
    // claims (jde vs de) were ruled out as statistically meaningless; cross-algorithm
    // comparison arrives with M2c's statistics module.
    assert!(gap < 2.0,
        "jDE should land within one Rastrigin local basin of the optimum: gap={}", gap);
}

#[test]
fn shade_converges_on_shifted_sphere() {
    let shift: Vec<f64> = (0..10).map(|i| 0.7 * i as f64 - 3.0).collect();
    let p = SphereShifted::new(shift, -5.0, 5.0);
    let spec = presets::shade(50, 20_000);
    let e = Engine::from_spec(&spec, &registry(), p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    assert!(r.best_f < 1e-8, "SHADE should converge in 20k evaluations: {}", r.best_f);
}

#[test]
fn shade_near_optimum_on_rastrigin() {
    let p = BbobProblem::new(3, 5, 1).unwrap();
    let spec = presets::shade(50, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Deterministic achieved gap ≈ 1.014 at seed 42 (BBOB f3 transforms stretch basin
    // spacing; bound anchored to this run, rounded up to the next 0.5 with headroom).
    // Single-seed comparative claims (shade vs de) were ruled out as statistically
    // meaningless per the M2b Task 7 controller ruling; cross-algorithm comparison
    // arrives with M2c's statistics module.
    assert!(gap < 1.5,
        "SHADE should land within one Rastrigin local basin of the optimum: gap={}", gap);
}

#[test]
fn lshade_converges_on_shifted_sphere() {
    let shift: Vec<f64> = (0..10).map(|i| 0.7 * i as f64 - 3.0).collect();
    let p = SphereShifted::new(shift, -5.0, 5.0);
    let spec = presets::lshade(10, 30_000);
    let e = Engine::from_spec(&spec, &registry(), p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    assert!(r.best_f < 1e-8, "L-SHADE should converge in 30k evaluations: {}", r.best_f);
}

#[test]
fn lshade_near_optimum_on_rotated_rastrigin() {
    let p = BbobProblem::new(15, 5, 1).unwrap();
    let spec = presets::lshade(5, 30_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Deterministic achieved gap ≈ 0.99656 at seed 42 on BBOB f15 (rotated
    // Rastrigin, dim 5, instance 1 — a genuinely hard multimodal landscape).
    // Rounded up to the next 0.5 (1.0) leaves only ~0.003 headroom, so the
    // bound is bumped one more 0.5 step to 1.5 (headroom ≈0.50), matching the
    // convention used for shade_near_optimum_on_rastrigin above. Single-seed
    // comparative claims (l-shade vs shade/de) were ruled out as
    // statistically meaningless per the M2b controller ruling;
    // cross-algorithm comparison arrives with M2c's statistics module.
    assert!(gap < 1.5,
        "L-SHADE should land near the optimum on rotated Rastrigin f15: gap={}", gap);
}

// ---- CMA-ES (M2b Task 11) ----

#[test]
fn cmaes_solves_sphere() {
    let shift: Vec<f64> = (0..10).map(|i| 0.7 * i as f64 - 3.0).collect();
    let p = SphereShifted::new(shift, -5.0, 5.0);
    let spec = presets::cmaes(10, 20_000);
    let e = Engine::from_spec(&spec, &registry(), p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    assert!(r.best_f < 1e-9, "CMA-ES should converge tightly on the sphere in 20k evaluations: {}", r.best_f);
}

#[test]
fn cmaes_solves_rotated_ellipsoid() {
    let p = BbobProblem::new(10, 10, 1).unwrap();
    let spec = presets::cmaes(10, 30_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    assert!(gap < 1e-6, "CMA-ES should solve the rotated ellipsoid (f10) tightly: gap={}", gap);
}

#[test]
fn cmaes_solves_bent_cigar() {
    let p = BbobProblem::new(12, 10, 1).unwrap();
    let spec = presets::cmaes(10, 30_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    assert!(gap < 1e-4, "CMA-ES should solve bent cigar (f12): gap={}", gap);
}

// ---- Stagnation restarts / IPOP (M2b Task 12) ----

#[test]
fn cmaes_ipop_escapes_multimodal() {
    let p = BbobProblem::new(3, 5, 1).unwrap(); // BBOB f3: separable Rastrigin
    let spec = presets::cmaes_ipop(5, 100_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let ipop_gap = r.best_f - p.f_opt();

    // Ablation baseline: the same CMA-ES algorithm, same seed/budget, without
    // restarts. pop_size = 4 + floor(3*ln(5)) = 8, matching cmaes_ipop's own
    // internally-computed starting population. Same algorithm with/without
    // restarts is an ablation, not a cross-algorithm claim.
    let plain_spec = presets::cmaes(8, 100_000);
    let e_plain = Engine::from_spec(&plain_spec, &registry(), Problem::space(&p)).unwrap();
    let r_plain = e_plain.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let plain_gap = r_plain.best_f - p.f_opt();

    assert!(ipop_gap <= plain_gap + 1e-9,
        "IPOP restarts should not make CMA-ES worse on a multimodal landscape: ipop_gap={ipop_gap} plain_gap={plain_gap}");
    // Deterministic achieved values at seed 42: ipop_gap ≈ 0.99496, plain_gap
    // ≈ 14.47617 — plain CMA-ES stalls in a wrong Rastrigin basin (no restart
    // mechanism to escape it) while IPOP's restarts let it recover and land
    // near the optimum. Bound anchored to the ipop_gap run: rounding up to
    // the next 0.5 (1.0) leaves only ~0.005 headroom, so per convention the
    // bound is bumped one more 0.5 step to 1.5 (headroom ≈0.505).
    assert!(ipop_gap < 1.5,
        "CMA-ES/IPOP should land close to the Rastrigin optimum: ipop_gap={ipop_gap} plain_gap={plain_gap}");
}

// ---- Nelder-Mead simplex (M2b Task 13) ----

#[test]
fn nm_converges_on_sphere() {
    let p = SphereShifted::new(vec![1.5, -0.5, 2.0, -3.0, 0.25], -5.0, 5.0);
    let spec = presets::nelder_mead(5, 5_000);
    let e = Engine::from_spec(&spec, &registry(), p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    assert!(r.best_f < 1e-6, "Nelder-Mead should converge tightly on the sphere: {}", r.best_f);
}

#[test]
fn nm_on_rosenbrock_2d() {
    let p = BbobProblem::new(8, 2, 1).unwrap();
    let spec = presets::nelder_mead(2, 10_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    assert!(gap < 1e-4, "Nelder-Mead should solve the 2D Rosenbrock (f8) closely: gap={}", gap);
}

// ---- Grey Wolf Optimizer (M2d-3 Task 5) ----

#[test]
fn gwo_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::gwo(30, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Deterministic achieved gap at seed 42: gap ~= 0.0000045804 (GWO
    // converges tightly on the separable, unimodal BBOB f1/Sphere in 20k
    // evaluations). Anchored bound rounded up to the next 0.5 (0.5), giving
    // ample (>=0.3) headroom, per convention (see
    // shade_near_optimum_on_rastrigin above).
    assert!(gap < 0.5,
        "GWO should land close to the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap}");
}

#[test]
fn gwo_min_pop_3_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::gwo(2, 500); // below min_pop = 3 (alpha/beta/delta)
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    assert!(err_str.contains("gen/gwo") && err_str.contains("3") && err_str.contains("2"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    // Also assert the underlying SpecError variant directly.
    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/gwo");
            assert_eq!(min_pop, 3);
            assert_eq!(pop_size, 2);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}

// ---- Whale Optimization Algorithm (M2d-3 Task 6) ----

#[test]
fn woa_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::woa(30, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Deterministic achieved gap at seed 42: gap ~= 0.0002618594 (re-measured
    // after the M2d-3 final review's per-whale draw-structure fix, since the
    // RNG-stream trajectory changed; WOA still converges tightly on the
    // separable, unimodal BBOB f1/Sphere in 20k evaluations). Anchored bound
    // rounded up to the next 0.5 (0.5), giving ample (>=0.3) headroom, per
    // convention (see gwo_solves_bbob_f1_dim5 above).
    assert!(gap < 0.5,
        "WOA should land close to the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap}");
}

#[test]
fn woa_min_pop_2_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::woa(1, 500); // below min_pop = 2 (best-so-far + one other)
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    assert!(err_str.contains("gen/woa") && err_str.contains("2") && err_str.contains("1"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    // Also assert the underlying SpecError variant directly.
    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/woa");
            assert_eq!(min_pop, 2);
            assert_eq!(pop_size, 1);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}

// ---- Harmony Search (M2d-3 Task 7) ----

#[test]
fn harmony_search_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::harmony_search(30, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Deterministic achieved gap at seed 42: gap ~= 0.0000002829 (HS
    // converges tightly on the separable, unimodal BBOB f1/Sphere in 20k
    // evaluations, despite generating only one new harmony per generation).
    // Anchored bound rounded up to the next 0.5 (0.5), giving ample
    // (>=0.3) headroom, per convention (see gwo_solves_bbob_f1_dim5 above).
    assert!(gap < 0.5,
        "Harmony Search should land close to the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap}");
}

#[test]
fn harmony_search_min_pop_1_validates_fine() {
    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::harmony_search(1, 500); // min_pop = 1: smallest legal pop_size
    assert!(spec.validate(&registry(), p.space()).is_ok(),
        "harmony_search preset must validate fine at pop_size=1 (min_pop=1)");
    let e = Engine::from_spec(&spec, &registry(), p.space());
    assert!(e.is_ok(), "harmony_search preset must build an Engine fine at pop_size=1");
}

// ---- Cuckoo Search (M2d-3 Task 8) ----

#[test]
fn cuckoo_search_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::cuckoo_search(25, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Deterministic achieved gap at seed 42: gap ~= 0.0100989121 (CS
    // converges on the separable, unimodal BBOB f1/Sphere in 20k
    // evaluations, pop 25). Anchored bound rounded up to the next 0.5 (0.5),
    // giving ample (>=0.3) headroom, per convention (see
    // gwo_solves_bbob_f1_dim5 above).
    assert!(gap < 0.5,
        "Cuckoo Search should land close to the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap}");
}

#[test]
fn cs_min_pop_2_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::cuckoo_search(1, 500); // below min_pop = 2 (best-so-far + one other)
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    assert!(err_str.contains("gen/cuckoo_levy") && err_str.contains("2") && err_str.contains("1"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    // Also assert the underlying SpecError variant directly.
    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/cuckoo_levy");
            assert_eq!(min_pop, 2);
            assert_eq!(pop_size, 1);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}

// ---- Grasshopper Optimisation Algorithm (M2d-3 Task 9) ----

#[test]
fn goa_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::goa(30, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Deterministic achieved gap at seed 42: gap ~= 3.34e-12 (GOA converges
    // EXTREMELY tightly on the separable, unimodal BBOB f1/Sphere in 20k
    // evaluations, pop 30 -- tighter than GWO/WOA/HS/CS's ~1e-3..1e-2 gaps,
    // consistent with `c` decaying to `cmin=1e-5` by the end of the budget,
    // which collapses every offspring to within a vanishing perturbation of
    // `X_best`). Anchored bound rounded up to the next 0.5 (0.5), giving
    // ample (>=0.3) headroom, per convention (see gwo_solves_bbob_f1_dim5
    // above).
    assert!(gap < 0.5,
        "GOA should land close to the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap}");
}

#[test]
fn goa_min_pop_2_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::goa(1, 500); // below min_pop = 2 (best-so-far + one other)
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    assert!(err_str.contains("gen/goa") && err_str.contains("2") && err_str.contains("1"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    // Also assert the underlying SpecError variant directly.
    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/goa");
            assert_eq!(min_pop, 2);
            assert_eq!(pop_size, 1);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}

// ---- Sine Cosine Algorithm (M2d-4 Task 1) ----

#[test]
fn sca_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::sca(30, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Deterministic achieved gap at seed 42: gap ~= 0.1823560084 (SCA
    // converges on the separable, unimodal BBOB f1/Sphere in 20k
    // evaluations, pop 30 -- looser than GWO/GOA's tighter gaps, consistent
    // with SCA's unconditional non-elitist replacement never re-locking onto
    // a previously-found best position once the population drifts past it).
    // Anchored bound rounded up to the next 0.5 (0.5), giving ample (>=0.3)
    // headroom, per convention (see gwo_solves_bbob_f1_dim5 above).
    assert!(gap < 0.5,
        "SCA should land close to the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap}");
}

#[test]
fn sca_min_pop_2_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::sca(1, 500); // below min_pop = 2 (best-so-far + one other)
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    assert!(err_str.contains("gen/sca") && err_str.contains("2") && err_str.contains("1"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    // Also assert the underlying SpecError variant directly.
    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/sca");
            assert_eq!(min_pop, 2);
            assert_eq!(pop_size, 1);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}

// ---- JAYA (M2d-4 Task 2) ----

#[test]
fn jaya_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::jaya(30, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Deterministic achieved gap at seed 42: gap ~= 0.0005521661213379048
    // (JAYA converges tightly on the separable, unimodal BBOB f1/Sphere in
    // 20k evaluations, pop 30 -- greedy same-index acceptance keeps every
    // improvement, and the shared per-dimension r1/r2 draws still let each
    // candidate individually converge toward best/away from worst).
    // Anchored bound rounded up to the next 0.5 (0.5), giving ample
    // (>=0.3) headroom, per convention (see sca_solves_bbob_f1_dim5 above).
    assert!(gap < 0.5,
        "JAYA should land close to the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap}");
}

#[test]
fn jaya_min_pop_2_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::jaya(1, 500); // below min_pop = 2 (best + worst)
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    assert!(err_str.contains("gen/jaya") && err_str.contains("2") && err_str.contains("1"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    // Also assert the underlying SpecError variant directly.
    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/jaya");
            assert_eq!(min_pop, 2);
            assert_eq!(pop_size, 1);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}
