//! Fixed-seed bit-identity goldens for the pso / de-jde / mfo presets on a
//! single-Float-block problem. Captured BEFORE the 0.1.6 generator-owned-state
//! refactor and required to pass UNCHANGED after it.
use sezgi_components::{presets, register_builtins};
use sezgi_core::component::Registry;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_core::problem::{Problem, SphereShifted};
use sezgi_core::space::BlockValues;
use sezgi_core::spec::AlgorithmSpec;

const DIM: usize = 6;
const BUDGET: u64 = 3000;

fn run(spec: &AlgorithmSpec, seed: u64) -> (u64, Vec<u64>) {
    let shift: Vec<f64> = (0..DIM).map(|i| 0.5 + 0.25 * i as f64).collect();
    let p = SphereShifted::new(shift, -5.0, 5.0);
    let mut reg = Registry::new();
    register_builtins(&mut reg);
    let e = Engine::from_spec(spec, &reg, p.space()).unwrap();
    let r = e.run(&p, RunConfig { master_seed: seed, run_id: 0 }, None).unwrap();
    let BlockValues::Float(xs) = &r.best_x.blocks[0] else { panic!("float block") };
    (r.best_f.to_bits(), xs.iter().map(|x| x.to_bits()).collect())
}

fn check(name: &str, spec: AlgorithmSpec, seed: u64, f_bits: u64, x_bits: &[u64]) {
    let (f, x) = run(&spec, seed);
    if f_bits == 0 {
        println!("CAPTURE {name} seed{seed}: {f:#018x} {x:#018x?}");
        panic!("uncaptured golden");
    }
    assert_eq!(f, f_bits, "{name} seed {seed}: best_f bits drifted");
    assert_eq!(x, x_bits, "{name} seed {seed}: best_x bits drifted");
}

#[test]
fn golden_pso_seed1() { check("pso", presets::pso(20, BUDGET), 1, 0x3e0cc16233356d45,
        &[0x3fe00017a6dbc3a3, 0x3fe7ffe6cabf6049, 0x3ff000030dc379a5, 0x3ff3fff11470571f, 0x3ff7ffec54f20fd4, 0x3ffbfffe5c6f5955]); }
#[test]
fn golden_pso_seed2() { check("pso", presets::pso(20, BUDGET), 2, 0x3e107b7ae1288b09,
        &[0x3fe0001b4d428512, 0x3fe80007f8dd49a1, 0x3ff0001935bc18ff, 0x3ff3fffcc69828b9, 0x3ff7fffc25760107, 0x3ffbfff226287fc6]); }
#[test]
fn golden_jde_seed1() { check("jde", presets::jde(20, BUDGET), 1, 0x3e0469709ce70fa7,
        &[0x3fe00015f7250613, 0x3fe7fff485338955, 0x3fefffdae8aa9750, 0x3ff4000b749f6c53, 0x3ff800046d14b565, 0x3ffbfffdcbdef287]); }
#[test]
fn golden_jde_seed2() { check("jde", presets::jde(20, BUDGET), 2, 0x3d980280dd2d4f66,
        &[0x3fe00002804033e3, 0x3fe7fffcdedfaa6c, 0x3feffffd737fec04, 0x3ff4000070e557c6, 0x3ff7ffff9f4547e0, 0x3ffc00002ba237ac]); }
#[test]
fn golden_mfo_seed1() { check("mfo", presets::mfo(30, BUDGET), 1, 0x3e602a1c9811e43b,
        &[0x3fe0007ec8099ea8, 0x3fe7ffebeba4b4c4, 0x3ff0005a370087d0, 0x3ff3ffb5f3cde497, 0x3ff7ff87a1e3293e, 0x3ffc001d97152de5]); }
#[test]
fn golden_mfo_seed2() { check("mfo", presets::mfo(30, BUDGET), 2, 0x3ea262f71273e47b,
        &[0x3fe0011bee6855c0, 0x3fe8016b60dbc858, 0x3ff002191e2babfa, 0x3ff40085f7e19852, 0x3ff7fe7d5f32741c, 0x3ffc01319e06e87d]); }
