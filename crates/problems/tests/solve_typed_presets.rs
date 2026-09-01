//! Task 5 required test 2: each M3-8 typed GA preset must solve its own
//! diagnostic problem (`sezgi_problems::diagnostics`) to (or effectively at)
//! the known optimum, from a fixed seed -- anchored final-quality
//! thresholds, mirroring `suite_properties.rs`'s own `de_smoke_on_new_fids`
//! shape (`Engine::from_spec` + `Engine::run`, `RunConfig { master_seed,
//! run_id: 0 }`, assert on `RunResult::best_f` against the problem's own
//! `optimum()`).

use sezgi_components::{presets, register_builtins};
use sezgi_core::component::Registry;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_core::problem::Problem;
use sezgi_problems::{CatMatch, IntQuadratic, OneMax};

#[test]
fn ga_bin_solves_one_max() {
    let p = OneMax::new(20);
    let mut reg = Registry::new();
    register_builtins(&mut reg);
    let e = Engine::from_spec(&presets::ga_bin(40, 6_000), &reg, p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 1, run_id: 0 }, None).unwrap();
    assert_eq!(r.best_f, Problem::optimum(&p).unwrap(), "ga_bin must reach OneMax's exact optimum (best_f={})", r.best_f);
}

#[test]
fn ga_int_solves_int_quadratic() {
    let p = IntQuadratic::new(-10, 10, 5);
    let mut reg = Registry::new();
    register_builtins(&mut reg);
    let e = Engine::from_spec(&presets::ga_int(40, 8_000), &reg, p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 3, run_id: 0 }, None).unwrap();
    assert_eq!(r.best_f, Problem::optimum(&p).unwrap(), "ga_int must reach IntQuadratic's exact optimum (best_f={})", r.best_f);
}

#[test]
fn ga_cat_solves_cat_match() {
    let p = CatMatch::new(4, 8, 123);
    let mut reg = Registry::new();
    register_builtins(&mut reg);
    let e = Engine::from_spec(&presets::ga_cat(40, 8_000), &reg, p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 5, run_id: 0 }, None).unwrap();
    assert_eq!(r.best_f, Problem::optimum(&p).unwrap(), "ga_cat must reach CatMatch's exact optimum (best_f={})", r.best_f);
}
