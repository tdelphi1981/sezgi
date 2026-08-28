use sezgi_core::problem::Problem;
use sezgi_core::space::{BlockValues, Genotype};
use sezgi_problems::BbobProblem;

fn g(xs: Vec<f64>) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs)] } }

#[test]
fn all_24_functions_complete_suite_properties() {
    for fid in 1u32..=24 {
        for dim in [2usize, 5, 10] {
            let p = BbobProblem::new(fid, dim, 1)
                .unwrap_or_else(|e| panic!("fid {fid} dim {dim}: {e}"));
            // 1) optimum erişimi
            let at_opt = p.evaluate_batch(&[g(p.x_opt().to_vec())])[0];
            assert!((at_opt - p.f_opt()).abs() < 1e-6, "fid {fid} dim {dim}: {}", at_opt - p.f_opt());
            // 2) optimum asla tam merkezde değil
            assert!(p.x_opt().iter().any(|&x| x.abs() > 1e-3), "fid {fid}: merkez optimum!");
            // 3) instance determinizmi + farklılığı
            let p_same = BbobProblem::new(fid, dim, 1).unwrap();
            let p_diff = BbobProblem::new(fid, dim, 2).unwrap();
            assert_eq!(p.x_opt(), p_same.x_opt());
            assert_ne!(p.x_opt(), p_diff.x_opt(), "fid {fid}: instance'lar aynı");
            // 4) sonlu değerler (NaN/Inf sızıntısı yok)
            let f = p.evaluate_batch(&[g(vec![1.234; dim]), g(vec![-4.9; dim])]);
            assert!(f.iter().all(|v| v.is_finite()), "fid {fid} dim {dim}: {f:?}");
        }
    }
}

#[test]
fn de_smoke_on_new_fids() {
    use sezgi_components::{presets, register_builtins};
    use sezgi_core::component::Registry;
    use sezgi_core::engine::{Engine, RunConfig};
    let mut reg = Registry::new();
    register_builtins(&mut reg);
    for fid in [6u32, 12, 15, 21] {
        let p = BbobProblem::new(fid, 5, 1).unwrap();
        let e = Engine::from_spec(&presets::de_rand_1(30, 6_000), &reg, p.space()).unwrap();
        let r = e.run(&p, RunConfig { master_seed: 7, run_id: 0 }, None).unwrap();
        assert!(r.best_f.is_finite() && r.best_f - p.f_opt() < 1e3,
                "fid {fid}: DE anlamlı ilerleme kaydedemedi ({})", r.best_f - p.f_opt());
    }
}
