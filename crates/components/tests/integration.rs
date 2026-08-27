use sezgi_components::{presets, register_builtins};
use sezgi_core::component::Registry;
use sezgi_core::dist::Distribution;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_core::problem::{Problem, SphereShifted};

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
    assert!(r.best_f < 1e-6, "DE 20k değerlendirmede yakınsamalı: {}", r.best_f);
}

#[test]
fn es_with_cauchy_beats_init() {
    let p = SphereShifted::new(vec![1.0; 5], -5.0, 5.0);
    let spec = presets::es_mu_plus_lambda(
        20, 5_000, Distribution::Cauchy { loc: 0.0, scale: 0.1 });
    let e = Engine::from_spec(&spec, &registry(), p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    assert!(r.best_f < 0.1, "ES ilerlemeli: {}", r.best_f);
}

#[test]
fn ga_converges_on_shifted_sphere_5d() {
    let p = SphereShifted::new(vec![1.5, -0.5, 2.0, -3.0, 0.25], -5.0, 5.0);
    let spec = presets::ga_real(50, 20_000);
    let e = Engine::from_spec(&spec, &registry(), p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();
    assert!(r.best_f < 1e-3, "GA yakınsamalı: {}", r.best_f);
}
