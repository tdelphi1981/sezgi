use sezgi_components::{presets, register_builtins};
use sezgi_core::component::Registry;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_problems::BbobProblem;

#[test]
fn de_solves_bbob_f1() {
    let p = BbobProblem::new(1, 10, 1).unwrap();
    let mut reg = Registry::new();
    register_builtins(&mut reg);
    let e = Engine::from_spec(&presets::de_rand_1(50, 20_000), &reg,
                              sezgi_core::problem::Problem::space(&p)).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 1, run_id: 0 }, None).unwrap();
    assert!(r.best_f - p.f_opt() < 1e-6, "fark: {}", r.best_f - p.f_opt());
}
