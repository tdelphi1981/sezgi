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
    // İlk koşuda dosyadaki değer "PIN-ME" ise: bu testi --nocapture ile koştur,
    // yazdırılan değeri JSON'a işle, testi tekrar koştur. Sonrasında değer sabittir.
    if expected == "PIN-ME" {
        panic!("altın değeri sabitle: best_f_bits = {got}");
    }
    assert_eq!(got, expected, "Rust yörüngesi altın değerden saptı");
}
