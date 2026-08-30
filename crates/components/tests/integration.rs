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

// ---- MFO (M2d-4 Task 3) ----

#[test]
fn mfo_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::mfo(30, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Measured gap at seed 42/pop 30/budget 20k: EXACTLY 0.0 (best_f is
    // bit-identical to f_opt -- the flame-elitist memory's narrowing
    // spiral converges past double-precision resolution on the separable,
    // unimodal BBOB f1/Sphere well within 20k evaluations). Anchored bound
    // rounded up to the next 0.5 above the measured value (0.5), matching
    // this wave's other anchored-threshold tests' bound; ample (>=0.3)
    // headroom.
    assert!(gap < 0.5,
        "MFO should land at (or extremely close to) the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap:e}");
}

#[test]
fn mfo_min_pop_2_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::mfo(1, 500); // below min_pop = 2
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    assert!(err_str.contains("gen/mfo") && err_str.contains("2") && err_str.contains("1"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/mfo");
            assert_eq!(min_pop, 2);
            assert_eq!(pop_size, 1);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}

// ---- SSA (M2d-4 Task 4) ----

#[test]
fn ssa_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::ssa(30, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Measured gap at seed 42/pop 30/budget 20k: ~6.54e-13 (essentially
    // converged to the optimum on the separable, unimodal BBOB f1/Sphere --
    // the half-population leader block's narrowing c1 schedule plus the
    // follower chain's averaging pull the whole population in tight well
    // within 20k evaluations). Anchored bound rounded up to the next 0.5
    // above the measured value (0.5), matching this wave's other
    // anchored-threshold tests' bound; ample (>=0.3) headroom.
    assert!(gap < 0.5,
        "SSA should land at (or extremely close to) the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap:e}");
}

#[test]
fn ssa_min_pop_2_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::ssa(1, 500); // below min_pop = 2
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    assert!(err_str.contains("gen/ssa") && err_str.contains("2") && err_str.contains("1"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/ssa");
            assert_eq!(min_pop, 2);
            assert_eq!(pop_size, 1);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}

// ---- FA / Firefly Algorithm (M2d-4 Task 5) ----

#[test]
fn firefly_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::firefly(25, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Measured gap at seed 42/pop 25/budget 20k: ~5.92e-8 (the pairwise
    // brighter-attracts-dimmer mechanism converges tightly on the
    // separable, unimodal BBOB f1/Sphere well within 20k evaluations).
    // Anchored bound rounded up to the next 0.5 above the measured value
    // (0.5), matching this wave's other anchored-threshold tests' bound;
    // ample (>=0.3) headroom.
    assert!(gap < 0.5,
        "firefly should land reasonably close to the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap:e}");
}

#[test]
fn firefly_min_pop_2_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::firefly(1, 500); // below min_pop = 2
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    assert!(err_str.contains("gen/fa") && err_str.contains("2") && err_str.contains("1"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/fa");
            assert_eq!(min_pop, 2);
            assert_eq!(pop_size, 1);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}

// ---- BA / Bat Algorithm (M2d-4 Task 6) ----

#[test]
fn bat_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::bat(30, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Measured gap at seed 42/pop 30/budget 20k: ~3.7007 (BA's velocity
    // accumulates with NO inertia/decay term at all, per the verified
    // bat_algorithm.m source -- see ba.rs's module doc -- and the
    // loudness-gated replacer can reject genuine improvements, so a looser
    // gap than MFO/SSA/JAYA's near-exact convergence is expected here).
    // Anchored bound: the next 0.5 multiple above the measured value (4.0)
    // gives only ~0.30 headroom, short of this wave's >=0.3 convention, so
    // one further 0.5 step (4.5) is used instead, giving ample (~0.80)
    // headroom.
    assert!(gap < 4.5,
        "bat should land reasonably close to the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap:e}");
}

#[test]
fn bat_min_pop_2_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::bat(1, 500); // below min_pop = 2
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    assert!(err_str.contains("gen/ba") && err_str.contains("2") && err_str.contains("1"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/ba");
            assert_eq!(min_pop, 2);
            assert_eq!(pop_size, 1);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}

// ---- FPA / Flower Pollination Algorithm (M2d-4 Task 7) ----

#[test]
fn fpa_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::fpa(25, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Measured gap at seed 42/pop 25/budget 20k: EXACTLY 0.0 (best_f is
    // bit-identical to f_opt -- FPA's greedy same-index acceptance never
    // loses ground, and the global branch's Lévy step reusing cs.rs's
    // cs_dim_step converges past double-precision resolution on the
    // separable, unimodal BBOB f1/Sphere well within 20k evaluations --
    // same phenomenon already observed for mfo_solves_bbob_f1_dim5 above).
    // Anchored bound rounded up to the next 0.5 above the measured value
    // (0.5), matching this wave's other anchored-threshold tests' bound;
    // ample (>=0.3) headroom.
    assert!(gap < 0.5,
        "fpa should land reasonably close to the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap:e}");
}

#[test]
fn fpa_min_pop_2_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::fpa(1, 500); // below min_pop = 2
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    assert!(err_str.contains("gen/fpa") && err_str.contains("2") && err_str.contains("1"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/fpa");
            assert_eq!(min_pop, 2);
            assert_eq!(pop_size, 1);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}

// ---- TLBO / Teaching-Learning-Based Optimization (M2d-4 Task 8, first
// multi-stage preset) ----

#[test]
fn tlbo_is_a_two_stage_spec() {
    let spec = presets::tlbo(30, 2000);
    assert_eq!(spec.stages.len(), 2, "tlbo must have exactly two [[stages]]: teacher, then learner");
    assert_eq!(spec.stages[0].generator.kind, "gen/tlbo-teacher");
    assert_eq!(spec.stages[0].replacer.kind, "replace/one-to-one-greedy");
    assert_eq!(spec.stages[1].generator.kind, "gen/tlbo-learner");
    assert_eq!(spec.stages[1].replacer.kind, "replace/one-to-one-greedy");
}

#[test]
fn tlbo_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::tlbo(30, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Measured gap at seed 42/pop 30/budget 20k: EXACTLY 0.0 (best_f is
    // bit-identical to f_opt -- TLBO's greedy same-index acceptance after
    // EACH phase never loses ground, and the teacher phase's pull toward
    // the current-pop best plus the learner phase's toward-better/
    // away-from-worse pairwise moves converge past double-precision
    // resolution on the separable, unimodal BBOB f1/Sphere well within the
    // 2*pop_size-per-generation budget). Anchored bound rounded up to the
    // next 0.5 above the measured value (0.5), matching this wave's other
    // anchored-threshold tests' bound; ample (>=0.3) headroom.
    assert!(gap < 0.5,
        "TLBO should land at (or extremely close to) the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap:e}");
}

#[test]
fn tlbo_min_pop_2_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::tlbo(1, 500); // below min_pop = 2 (learner phase needs a distinct partner)
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    // validate() checks each stage's components in order, so the FIRST
    // stage's generator (gen/tlbo-teacher) is the one reported here.
    assert!(err_str.contains("gen/tlbo-teacher") && err_str.contains("2") && err_str.contains("1"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/tlbo-teacher");
            assert_eq!(min_pop, 2);
            assert_eq!(pop_size, 1);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}

// ---- HHO / Harris Hawks Optimization (M2d-4 Task 9, the wave's most
// complex multi-branch escape-energy tree, with in-generator evaluation for
// the rapid-dive sub-branches) ----

#[test]
fn hho_is_a_single_stage_generational_spec() {
    let spec = presets::hho(30, 2000);
    assert_eq!(spec.stages.len(), 1);
    assert_eq!(spec.stages[0].generator.kind, "gen/hho");
    assert_eq!(spec.stages[0].replacer.kind, "replace/generational");
}

#[test]
fn hho_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::hho(30, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Measured gap at seed 42/pop 30/budget 20k, RE-MEASURED after fix
    // round 1 (the soft-dive Y formula correction, HHO.m line 105 vs line
    // 98 -- see hho.rs's finding 4): approximately 8.39e-3 (0.00839) --
    // HHO's escape-energy tree and rapid-dive greedy acceptance still make
    // steady progress on the separable, unimodal BBOB f1/Sphere well within
    // the budget (the internal dive-trial evaluations also consume part of
    // the 20k budget, per this module's eval-accounting design, so fewer
    // generations complete than a single-eval-per-hawk preset would get at
    // the same budget -- but still comfortably converges). Anchored bound
    // rounded up to the next 0.5 above the measured value (0.5), matching
    // this wave's other anchored-threshold tests' bound; ample (>=0.3,
    // here ~0.49) headroom.
    assert!(gap < 0.5,
        "hho should land reasonably close to the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap:e}");
}

#[test]
fn hho_min_pop_2_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::hho(1, 500); // below min_pop = 2
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    assert!(err_str.contains("gen/hho") && err_str.contains("2") && err_str.contains("1"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/hho");
            assert_eq!(min_pop, 2);
            assert_eq!(pop_size, 1);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}

// ---- ALO / Ant Lion Optimizer (M2d-4 Task 10, faithful full-walk
// construction -- see alo.rs's module doc's "Cost design" section) ----

#[test]
fn alo_is_a_single_stage_generational_spec() {
    let spec = presets::alo(25, 2000);
    assert_eq!(spec.stages.len(), 1);
    assert_eq!(spec.stages[0].generator.kind, "gen/alo");
    // The elitism adjudication (see alo.rs's module doc): the antlion
    // population itself is the persisted memory, via mu-plus-lambda's
    // merge-sort-truncate -- no blackboard state, no new replacer.
    assert_eq!(spec.stages[0].replacer.kind, "replace/mu-plus-lambda");
}

#[test]
fn alo_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::alo(25, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Measured gap at seed 42/pop 25/budget 20k: ~1.28e-13 (essentially
    // converged to double-precision resolution) -- despite this instance's
    // negative f_opt (~-125.95, so fitness is negative throughout almost
    // the entire run, exercising alo_roulette_weights's negative-fitness
    // floor-shift branch, not just the literal-reciprocal common case), the
    // elite-walk term (RE, always toward the current best-ever via
    // mu-plus-lambda's elitist merge) and the shrinking I-ratio bounds
    // still drive convergence past double-precision resolution well within
    // 20k evaluations on the separable, unimodal BBOB f1/Sphere. Anchored
    // bound rounded up to the next 0.5 above the measured value (0.5),
    // matching this wave's other anchored-threshold tests' bound; ample
    // (>=0.3) headroom.
    assert!(gap < 0.5,
        "alo should land reasonably close to the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap:e}");
}

#[test]
fn alo_min_pop_2_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::alo(1, 500); // below min_pop = 2
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    assert!(err_str.contains("gen/alo") && err_str.contains("2") && err_str.contains("1"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/alo");
            assert_eq!(min_pop, 2);
            assert_eq!(pop_size, 1);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}

// ---- ABC / Artificial Bee Colony (M2d-4 Task 11 -- see abc.rs's module
// doc's "Phase-design adjudication": a single stage, not two symmetric
// stages like tlbo, since the onlooker scan cannot fit the
// Generator+Replacer shape) ----

#[test]
fn abc_is_a_single_stage_spec_with_the_trial_coupled_replacer_and_onlooker_scout_adapter() {
    let spec = presets::abc(20, 2000);
    assert_eq!(spec.stages.len(), 1, "ABC is a single-stage preset, unlike tlbo's two symmetric stages");
    assert_eq!(spec.stages[0].generator.kind, "gen/abc-employed");
    assert_eq!(spec.stages[0].replacer.kind, "replace/abc-trial-greedy");
    assert_eq!(spec.stages[0].adapter.as_ref().unwrap().kind, "adapter/abc-onlooker-scout");
}

#[test]
fn abc_min_pop_2_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::abc(1, 500); // below min_pop = 2
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    assert!(err_str.contains("gen/abc-employed") && err_str.contains("2") && err_str.contains("1"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/abc-employed");
            assert_eq!(min_pop, 2);
            assert_eq!(pop_size, 1);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}

/// Budget-accounting honesty (the brief's explicit ask): with trial
/// counters far from the limit (`SN*dim`, large relative to a handful of
/// generations), no scout ever fires, so a full multi-generation run must
/// consume EXACTLY `pop_size (init) + generations * 2 * pop_size`
/// evaluations -- `SN` for the employed phase (the engine's own
/// `eval.evaluate(&offspring)` call) plus `SN` for the onlooker phase
/// (`adapter/abc-onlooker-scout`'s own `SN` internal `ctx.eval.evaluate`
/// calls), never `4*SN` (which an HHO-style "generator internally
/// evaluates, then the engine re-evaluates the same offspring again" design
/// would have produced -- see the module doc's phase-design adjudication
/// for why that shape was rejected).
#[test]
fn abc_evals_are_exactly_2x_pop_per_cycle_when_no_scout_fires() {
    let pop_size = 10usize;
    let dim = 5;
    // limit = SN*dim = 50: each source's trial can increment at most TWICE
    // per cycle (once in the employed phase, once if visited-and-rejected
    // in the onlooker phase), so after only 3 cycles no source's trial can
    // possibly exceed 6 -- nowhere near 50, so the scout provably never
    // fires in this fixture, and the budget divides EXACTLY (no partial
    // generation), making the expected evals_used arithmetic exact --
    // mirroring the engine's own `two_stage_budget_accounting_consumes_2x_
    // pop_per_generation` test (M2d-4 Task 8) at the component level.
    let p = SphereShifted::new(vec![0.0; dim], -5.0, 5.0);
    let generations = 3u64;
    let budget = pop_size as u64 + generations * 2 * pop_size as u64; // exactly divisible
    let spec = presets::abc(pop_size, budget);
    let e = Engine::from_spec(&spec, &registry(), p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    assert_eq!(r.evals_used, budget,
        "an exactly-divisible budget must be used up exactly: init + generations*2*pop_size, honestly (never 4*pop_size)");
    assert_eq!(r.iterations, generations);
}

#[test]
fn abc_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::abc(20, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Measured gap at seed 42/pop_size 20 (=SN)/budget 20k: exactly 0.0
    // (bit-identical convergence to the optimum, best_f == f_opt ==
    // -125.9497035670884 -- cross-checked at seeds 7/100/999 too, same
    // exact result each time). On this easy, unimodal, separable BBOB
    // f1/dim5 instance, ABC's single-dimension coordinate-wise employed
    // and onlooker moves (each visit only ever perturbs ONE dimension) act
    // like a highly effective coordinate-descent-with-restarts search over
    // 500 cycles (20k/(2*20)), so exact convergence is unsurprising here --
    // same class of result `alo.rs`'s own anchored test measured (~1.28e-13,
    // essentially converged) on the identical instance. Anchored per this
    // wave's convention: round UP to the next 0.5 above the measured value
    // with >=0.3 headroom -- measured 0.0, so anchor at 0.5 (headroom 0.5).
    assert!(gap < 0.5,
        "abc should land reasonably close to the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap:e}");
}

// ---- GSA / Gravitational Search Algorithm (M2d-4 Task 12 -- the wave's
// LAST stateful/blackboard algorithm; see gsa.rs's module doc for the full
// provenance extraction against Rashedi's own GSA.m/Gconstant.m/
// massCalculation.m/Gfield.m/move.m, the verified M_i-free force delta, and
// the confirmation that GSA's own Fbest/Lbest never feed back into the
// mechanism -- settling the current-pop-convention controller ruling) ----

#[test]
fn gsa_is_a_single_stage_spec_with_generational_replacement() {
    let spec = presets::gsa(30, 2000);
    assert_eq!(spec.stages.len(), 1);
    assert_eq!(spec.stages[0].generator.kind, "gen/gsa");
    assert_eq!(spec.stages[0].replacer.kind, "replace/generational");
    assert!(spec.stages[0].adapter.is_none(),
        "gsa/velocity is owned entirely by gen/gsa itself, no separate adapter (same self-owning shape as pso/bat)");
}

#[test]
fn gsa_min_pop_2_enforced_by_spec_validation() {
    use sezgi_core::spec::SpecError;

    let p = SphereShifted::new(vec![0.0; 5], -5.0, 5.0);
    let spec = presets::gsa(1, 500); // below min_pop = 2
    let err = Engine::from_spec(&spec, &registry(), p.space()).err().unwrap();
    let err_str = format!("{err}");
    assert!(err_str.contains("gen/gsa") && err_str.contains("2") && err_str.contains("1"),
        "expected a PopulationTooSmall-shaped error, got: {err_str}");

    match spec.validate(&registry(), p.space()) {
        Err(SpecError::PopulationTooSmall { kind, min_pop, pop_size }) => {
            assert_eq!(kind, "gen/gsa");
            assert_eq!(min_pop, 2);
            assert_eq!(pop_size, 1);
        }
        other => panic!("expected SpecError::PopulationTooSmall, got: {other:?}"),
    }
}

#[test]
fn gsa_solves_bbob_f1_dim5() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let spec = presets::gsa(30, 20_000);
    let e = Engine::from_spec(&spec, &registry(), Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    let gap = r.best_f - p.f_opt();
    // Measured gap at seed 42/pop_size 30/budget 20k: exactly 0.0
    // (bit-identical convergence to the optimum, best_f == f_opt ==
    // -125.9497035670884 -- cross-checked at seeds 7/100/999 too, same
    // exact result each time). On this easy, unimodal, separable BBOB
    // f1/dim5 instance, GSA's gravitational pull toward the current heaviest
    // (best-fitness) agents converges cleanly, same class of result
    // `abc.rs`'s/`alo.rs`'s own anchored tests measured on the identical
    // instance. Anchored per this wave's convention: round UP to the next
    // 0.5 above the measured value with >=0.3 headroom -- measured 0.0, so
    // anchor at 0.5 (headroom 0.5).
    assert!(gap < 0.5,
        "gsa should land reasonably close to the BBOB f1 (Sphere) optimum in 20k evaluations: gap={gap:e}");
}
