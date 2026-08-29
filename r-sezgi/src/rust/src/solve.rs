use savvy::{savvy, savvy_err, OwnedListSexp, OwnedRealSexp, Sexp};
use sezgi_components::{presets, register_builtins};
use sezgi_core::component::Registry;
use sezgi_core::engine::{Engine, RunConfig};
use sezgi_core::problem::Problem;
use sezgi_core::space::BlockValues;
use sezgi_core::spec::AlgorithmSpec;
use sezgi_problems::BbobProblem;

/// Builds a DE/rand/1/bin algorithm spec (uniform init, clamp boundary,
/// one-to-one-greedy replacement) as JSON, ready to pass to
/// `sz_solve_bbob()`.
///
/// @param pop_size Population size.
/// @param budget Evaluation budget.
/// @returns A character scalar with the algorithm spec as JSON.
/// @export
#[savvy]
fn sz_preset_de_rand_1(pop_size: f64, budget: f64) -> savvy::Result<Sexp> {
    let json = presets::de_rand_1(pop_size as usize, budget as u64).to_json();
    json.try_into()
}

fn registry() -> Registry {
    let mut r = Registry::new();
    register_builtins(&mut r);
    r
}

/// Runs an algorithm spec on a BBOB problem and returns the result.
///
/// @param spec_json Algorithm spec as JSON (e.g. from `sz_preset_de_rand_1()`).
/// @param fid BBOB function id (>= 1).
/// @param dim Problem dimension (>= 1).
/// @param instance BBOB instance id (>= 1).
/// @param master_seed Master RNG seed.
/// @param run_id Run id (mixed into the seed for independent replicate streams).
/// @returns A named list with `best_f` (double), `evals` (double), and
///   `best_x` (double vector).
/// @export
#[savvy]
fn sz_solve_bbob(
    spec_json: &str,
    fid: i32,
    dim: i32,
    instance: i32,
    master_seed: f64,
    run_id: f64,
) -> savvy::Result<Sexp> {
    if fid < 1 {
        return Err(savvy_err!("fid must be >= 1"));
    }
    if dim < 1 {
        return Err(savvy_err!("dim must be >= 1"));
    }
    if instance < 1 {
        return Err(savvy_err!("instance must be >= 1"));
    }

    let spec = AlgorithmSpec::from_json(spec_json).map_err(|e| savvy_err!("{e}"))?;
    let reg = registry();

    let problem =
        BbobProblem::new(fid as u32, dim as usize, instance as u32).map_err(|e| savvy_err!("{e}"))?;

    let engine =
        Engine::from_spec(&spec, &reg, problem.space()).map_err(|e| savvy_err!("{e}"))?;
    let result = engine
        .run(
            &problem,
            RunConfig {
                master_seed: master_seed as u64,
                run_id: run_id as u64,
            },
            None,
        )
        .map_err(|e| savvy_err!("{e}"))?;

    let BlockValues::Float(xs) = &result.best_x.blocks[0] else {
        return Err(savvy_err!("unexpected genotype"));
    };

    let mut out = OwnedListSexp::new(3, true)?;
    out.set_name_and_value(0, "best_f", OwnedRealSexp::try_from_scalar(result.best_f)?)?;
    out.set_name_and_value(
        1,
        "evals",
        OwnedRealSexp::try_from_scalar(result.evals_used as f64)?,
    )?;
    out.set_name_and_value(2, "best_x", OwnedRealSexp::try_from_slice(xs.as_slice())?)?;

    Ok(out.into())
}
