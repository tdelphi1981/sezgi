use sezgi_components::{presets, register_builtins};
use sezgi_core::component::Registry;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_core::problem::Problem;
use sezgi_problems::BbobProblem;

#[test]
fn golden_de_bbob_f1_seed42() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let mut reg = Registry::new();
    register_builtins(&mut reg);
    let e = Engine::from_spec(&presets::de_rand_1(20, 2_000), &reg, p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();

    let golden: serde_json::Value = serde_json::from_str(
        include_str!("../../../tests/golden/de_bbob_f1_seed42.json")).unwrap();
    let expected = golden["best_f_bits"].as_str().unwrap();
    let got = format!("{:016x}", r.best_f.to_bits());
    // If the value in the file is still "PIN-ME" on the first run: run this test with
    // --nocapture, write the printed value into the JSON, then re-run. The value is fixed after that.
    if expected == "PIN-ME" {
        panic!("pin the golden value: best_f_bits = {got}");
    }
    assert_eq!(got, expected, "Rust trajectory drifted from the golden value");
}

#[test]
fn golden_de_best1_bbob_f1_seed42() {
    let p = BbobProblem::new(1, 5, 1).unwrap();
    let mut reg = Registry::new();
    register_builtins(&mut reg);
    let e = Engine::from_spec(&presets::de_best_1(20, 2_000), &reg, p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: 42, run_id: 0 }, None).unwrap();

    let golden: serde_json::Value = serde_json::from_str(
        include_str!("../../../tests/golden/de_best1_bbob_f1_seed42.json")).unwrap();
    let expected = golden["best_f_bits"].as_str().unwrap();
    let got = format!("{:016x}", r.best_f.to_bits());
    // If the value in the file is still "PIN-ME" on the first run: run this test with
    // --nocapture, write the printed value into the JSON, then re-run. The value is fixed after that.
    if expected == "PIN-ME" {
        panic!("pin the golden value: best_f_bits = {got}");
    }
    assert_eq!(got, expected, "Rust trajectory drifted from the golden value");
}
