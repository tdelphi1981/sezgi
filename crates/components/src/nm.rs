use sezgi_core::component::*;
use sezgi_core::problem::Population;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};
use sezgi_core::state::StateReq;

/// Nelder–Mead simplex (Nelder & Mead 1965, "A Simplex Method for Function
/// Minimization"; standard coefficients α=1, γ=2, ρ=0.5, σ=0.5 — the
/// reflection/expansion/contraction/shrink coefficients as given in the
/// original paper and the standard Wikipedia formulation).
///
/// **The batch-engine adaptation (documented sezgi simplification).**
/// Textbook Nelder–Mead evaluates one candidate point at a time and reacts
/// immediately (a strictly sequential, single-evaluation-per-decision
/// algorithm). sezgi's engine evaluates a `Vec<Genotype>` batch per
/// `generate`/`replace` round. To fit that shape without changing NM's
/// actual decision logic, the *population IS the simplex*
/// (`pop_size == dim + 1`) and the algorithm is expressed as an explicit
/// state machine on the blackboard: each `generate` call emits exactly the
/// one (or, in Shrink, `dim`) candidate(s) the *next* textbook decision
/// needs, and `replace` applies the textbook accept/reject rule for that
/// step and advances the machine to the next phase. Across a full "round"
/// (Reflect, and possibly TryExpand or a Try-Contract phase) the simplex is
/// otherwise untouched, so `nm_centroid` / `nm_worst_idx` computed once at
/// Reflect stay valid for every phase that follows in the same round. No
/// RNG draws happen anywhere in this pair (fully deterministic after
/// initialization), and every index tie is pinned by the direction Rust's
/// own `Iterator::min_by`/`max_by` already resolve ties (documented at each
/// use site below) rather than by any extra tie-breaking code.
///
/// **Blackboard state** (owned by `gen/nelder-mead`; required by
/// `replace/nelder-mead`):
/// - `nm_phase`: `u8` — 0=Reflect, 1=TryExpand, 2=TryContractOutside,
///   3=TryContractInside, 4=Shrink. Absent (fresh blackboard) is treated as
///   0 (Reflect) — the machine's start state.
/// - `nm_centroid`: `Vec<f64>` (len `dim`) — centroid of all vertices except
///   the worst, computed fresh at every Reflect and reused by TryExpand /
///   TryContractOutside / TryContractInside until the next Reflect.
/// - `nm_reflected`: `Vec<f64>` (len `dim`) — the Reflect phase's `x_r`,
///   needed by TryExpand (`x_e = c + γ(x_r − c)`) and TryContractOutside
///   (`x_c = c + ρ(x_r − c)`). The generator's own `generate` call stores a
///   *pre-repair* value here (it has no way to see the engine's boundary
///   repair, which runs after `generate` and before `evaluate`/`replace`);
///   `replace/nelder-mead`'s Reflect branch overwrites this entry with the
///   *repaired* offspring it actually receives before consulting the
///   decision table, so every later insertion of `x_r` (TryExpand's
///   reject-expansion branch included) uses the genotype the evaluated
///   fitness actually belongs to. See the "reflected point is stored
///   post-repair" note on `NmReplacer::replace` below.
/// - `nm_reflected_f`: `f64` — `x_r`'s fitness, only known once `replace`
///   sees the evaluated Reflect offspring; needed by TryExpand's
///   best-of-two comparison and TryContractOutside's `f_c ≤ f_r` test.
/// - `nm_worst_idx`: `usize` — the vertex index Reflect/TryExpand/
///   TryContractOutside/TryContractInside all replace on acceptance.
///
/// **Index tie-breaks** (pinned, both simply the natural behavior of the
/// standard-library iterator adaptor used — no separate tie-break code):
/// - Best (lowest fitness): `Population::best_index` uses `Iterator::min_by`,
///   which returns the *first* of equally-minimal elements — i.e. ties go to
///   the **lowest index**.
/// - Worst (highest fitness): `worst_index` (below) uses `Iterator::max_by`,
///   which returns the *last* of equally-maximal elements — i.e. ties go to
///   the **highest index**.
pub struct NmGenerator;

const ALPHA: f64 = 1.0;
const GAMMA: f64 = 2.0;
const RHO: f64 = 0.5;
const SIGMA: f64 = 0.5;

fn float_view(g: &Genotype) -> &Vec<f64> {
    match &g.blocks[0] { BlockValues::Float(xs) => xs, _ => unreachable!() }
}

fn dim_of(space: &SearchSpace) -> usize {
    match space.blocks()[0] {
        Block::Float { n, .. } => n,
        _ => unreachable!("gen/nelder-mead requires a float-only space (meta().supported_blocks)"),
    }
}

/// Index of the worst (highest-fitness) individual. Ties go to the
/// **highest** index (`Iterator::max_by` returns the last of equal maxima).
fn worst_index(fitness: &[f64]) -> usize {
    fitness.iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("worst_index requires a non-empty population")
}

/// Centroid of every vertex except `exclude`, over the first `dim` float
/// coordinates.
fn centroid_excluding(pop: &Population, exclude: usize, dim: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; dim];
    let mut count = 0usize;
    for (i, ind) in pop.individuals.iter().enumerate() {
        if i == exclude { continue; }
        let xs = float_view(ind);
        for j in 0..dim { c[j] += xs[j]; }
        count += 1;
    }
    for v in &mut c { *v /= count as f64; }
    c
}

impl Generator for NmGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        let dim = dim_of(ctx.space);
        assert_eq!(pop.len(), dim + 1,
            "gen/nelder-mead: the population IS the simplex, so pop_size must equal dim+1 \
             (dim={dim} requires pop_size={}, got pop_size={})", dim + 1, pop.len());

        let phase = ctx.bb.get::<u8>("nm_phase").copied().unwrap_or(0);

        match phase {
            // Reflect: c = centroid of all but the worst; x_r = c + α(c − x_worst).
            0 => {
                let worst_idx = worst_index(&pop.fitness);
                let centroid = centroid_excluding(pop, worst_idx, dim);
                let x_worst = float_view(&pop.individuals[worst_idx]);
                let x_r: Vec<f64> = (0..dim)
                    .map(|j| centroid[j] + ALPHA * (centroid[j] - x_worst[j]))
                    .collect();
                ctx.bb.insert("nm_phase", 0u8);
                ctx.bb.insert("nm_worst_idx", worst_idx);
                ctx.bb.insert("nm_centroid", centroid);
                ctx.bb.insert("nm_reflected", x_r.clone());
                vec![Genotype { blocks: vec![BlockValues::Float(x_r)] }]
            }
            // TryExpand: x_e = c + γ(x_r − c).
            1 => {
                let centroid = ctx.bb.get::<Vec<f64>>("nm_centroid")
                    .expect("gen/nelder-mead: TryExpand requires nm_centroid from a prior Reflect")
                    .clone();
                let x_r = ctx.bb.get::<Vec<f64>>("nm_reflected")
                    .expect("gen/nelder-mead: TryExpand requires nm_reflected from a prior Reflect")
                    .clone();
                let x_e: Vec<f64> = (0..dim)
                    .map(|j| centroid[j] + GAMMA * (x_r[j] - centroid[j]))
                    .collect();
                ctx.bb.insert("nm_phase", 1u8);
                vec![Genotype { blocks: vec![BlockValues::Float(x_e)] }]
            }
            // TryContractOutside: x_c = c + ρ(x_r − c).
            2 => {
                let centroid = ctx.bb.get::<Vec<f64>>("nm_centroid")
                    .expect("gen/nelder-mead: TryContractOutside requires nm_centroid from a prior Reflect")
                    .clone();
                let x_r = ctx.bb.get::<Vec<f64>>("nm_reflected")
                    .expect("gen/nelder-mead: TryContractOutside requires nm_reflected from a prior Reflect")
                    .clone();
                let x_c: Vec<f64> = (0..dim)
                    .map(|j| centroid[j] + RHO * (x_r[j] - centroid[j]))
                    .collect();
                ctx.bb.insert("nm_phase", 2u8);
                vec![Genotype { blocks: vec![BlockValues::Float(x_c)] }]
            }
            // TryContractInside: x_cc = c + ρ(x_worst − c) (== c − ρ(x_r − c)).
            3 => {
                let centroid = ctx.bb.get::<Vec<f64>>("nm_centroid")
                    .expect("gen/nelder-mead: TryContractInside requires nm_centroid from a prior Reflect")
                    .clone();
                let worst_idx = *ctx.bb.get::<usize>("nm_worst_idx")
                    .expect("gen/nelder-mead: TryContractInside requires nm_worst_idx from a prior Reflect");
                let x_worst = float_view(&pop.individuals[worst_idx]).clone();
                let x_cc: Vec<f64> = (0..dim)
                    .map(|j| centroid[j] + RHO * (x_worst[j] - centroid[j]))
                    .collect();
                ctx.bb.insert("nm_phase", 3u8);
                vec![Genotype { blocks: vec![BlockValues::Float(x_cc)] }]
            }
            // Shrink: for every non-best vertex i, x_i' = x_best + σ(x_i − x_best),
            // emitted in ascending index order.
            4 => {
                let best_idx = pop.best_index()
                    .expect("gen/nelder-mead requires a non-empty population");
                let x_best = float_view(&pop.individuals[best_idx]).clone();
                let mut out = Vec::with_capacity(dim);
                for (i, ind) in pop.individuals.iter().enumerate() {
                    if i == best_idx { continue; }
                    let x_i = float_view(ind);
                    let x_shrunk: Vec<f64> = (0..dim)
                        .map(|j| x_best[j] + SIGMA * (x_i[j] - x_best[j]))
                        .collect();
                    out.push(Genotype { blocks: vec![BlockValues::Float(x_shrunk)] });
                }
                ctx.bb.insert("nm_phase", 4u8);
                out
            }
            p => unreachable!("gen/nelder-mead: invalid nm_phase byte {p}"),
        }
    }

    /// `pop_size` here must equal `dim + 1` (the population IS the simplex);
    /// that's dimension-dependent and can't be expressed as a fixed
    /// `min_pop` constant, so `min_pop` stays at the `ComponentMeta::new`
    /// default of 1 and the exact requirement is enforced only by the
    /// runtime assert in `generate` (see `nm_asserts_simplex_size` below).
    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("gen/nelder-mead", SupportedBlocks::Only(vec!["float"]))
            .with_offspring_one()
            .with_provides(vec![
                StateReq::of::<u8>("nm_phase"),
                StateReq::of::<Vec<f64>>("nm_centroid"),
                StateReq::of::<Vec<f64>>("nm_reflected"),
                StateReq::of::<f64>("nm_reflected_f"),
                StateReq::of::<usize>("nm_worst_idx"),
            ])
    }
}

/// `replace/nelder-mead`: the standard Nelder–Mead accept/reject decision
/// table, applied to whichever phase `gen/nelder-mead` just produced
/// offspring for — see the module doc for the full blackboard contract and
/// the batch-engine state-machine framing.
///
/// **Reflected point is stored post-repair.** The Reflect branch (`nm_phase
/// == 0`) overwrites `nm_reflected` with the offspring it is actually
/// handed (`oi[0]`) before consulting the decision table, rather than
/// trusting the pre-repair value `gen/nelder-mead` wrote. The engine repairs
/// offspring in place between `generate` and `evaluate`/`replace`, so `oi[0]`
/// is the post-repair, in-bounds point whose fitness is `of[0]`; keeping
/// `nm_reflected` in sync with it means every later insertion of `x_r`
/// (e.g. TryExpand's reject-expansion branch) stays genotype↔fitness
/// consistent under boundary clamping.
pub struct NmReplacer;

impl Replacer for NmReplacer {
    fn replace(&self, pop: &mut Population, oi: Vec<Genotype>, of: Vec<f64>, ctx: &mut Ctx) {
        let phase = *ctx.bb.get::<u8>("nm_phase")
            .expect("replace/nelder-mead requires nm_phase (run gen/nelder-mead first)");

        match phase {
            // After Reflect, with f_r = of[0]: compare against f_best / f_second_worst
            // (both computed from the CURRENT population) / f_worst.
            0 => {
                let worst_idx = *ctx.bb.get::<usize>("nm_worst_idx")
                    .expect("replace/nelder-mead: Reflect requires nm_worst_idx");
                // `oi[0]` is the offspring the replacer actually receives, i.e.
                // POST-repair (the engine repairs offspring in place between
                // `generate` and `evaluate`/`replace`). The generator's own
                // `nm_reflected` write happened before that repair, so it can
                // disagree with the point `f_r` was actually evaluated at.
                // Overwrite it here, before the decision table below, so every
                // later use of x_r (including TryExpand's reject-expansion
                // branch, which reconstructs a genotype straight from
                // `nm_reflected`) stays genotype/fitness-consistent and inside
                // the search space.
                ctx.bb.insert("nm_reflected", float_view(&oi[0]).clone());
                let f_r = of[0];
                let f_worst = pop.fitness[worst_idx];
                let f_best = pop.fitness.iter().cloned().fold(f64::INFINITY, f64::min);
                let f_second_worst = pop.fitness.iter().enumerate()
                    .filter(|&(i, _)| i != worst_idx)
                    .map(|(_, &f)| f)
                    .fold(f64::NEG_INFINITY, f64::max);

                if f_r < f_best {
                    // Better than the best: don't accept yet, try expanding.
                    ctx.bb.insert("nm_reflected_f", f_r);
                    ctx.bb.insert("nm_phase", 1u8);
                } else if f_r < f_second_worst {
                    // f_best <= f_r < f_second_worst: accept the reflection outright.
                    pop.individuals[worst_idx] = oi.into_iter().next().unwrap();
                    pop.fitness[worst_idx] = f_r;
                    ctx.bb.insert("nm_phase", 0u8);
                } else if f_r < f_worst {
                    // f_second_worst <= f_r < f_worst: try an outside contraction.
                    ctx.bb.insert("nm_reflected_f", f_r);
                    ctx.bb.insert("nm_phase", 2u8);
                } else {
                    // f_r >= f_worst: try an inside contraction.
                    ctx.bb.insert("nm_reflected_f", f_r);
                    ctx.bb.insert("nm_phase", 3u8);
                }
            }
            // After TryExpand, with f_e = of[0]: accept the better of (x_e, f_e)
            // and the stored (x_r, f_r).
            1 => {
                let worst_idx = *ctx.bb.get::<usize>("nm_worst_idx")
                    .expect("replace/nelder-mead: TryExpand requires nm_worst_idx");
                let f_e = of[0];
                let f_r = *ctx.bb.get::<f64>("nm_reflected_f")
                    .expect("replace/nelder-mead: TryExpand requires nm_reflected_f from Reflect");
                if f_e < f_r {
                    pop.individuals[worst_idx] = oi.into_iter().next().unwrap();
                    pop.fitness[worst_idx] = f_e;
                } else {
                    let x_r = ctx.bb.get::<Vec<f64>>("nm_reflected")
                        .expect("replace/nelder-mead: TryExpand requires nm_reflected from Reflect")
                        .clone();
                    pop.individuals[worst_idx] = Genotype { blocks: vec![BlockValues::Float(x_r)] };
                    pop.fitness[worst_idx] = f_r;
                }
                ctx.bb.insert("nm_phase", 0u8);
            }
            // After TryContractOutside, with f_c = of[0]: accept iff f_c <= f_r,
            // else shrink.
            2 => {
                let worst_idx = *ctx.bb.get::<usize>("nm_worst_idx")
                    .expect("replace/nelder-mead: TryContractOutside requires nm_worst_idx");
                let f_c = of[0];
                let f_r = *ctx.bb.get::<f64>("nm_reflected_f")
                    .expect("replace/nelder-mead: TryContractOutside requires nm_reflected_f from Reflect");
                if f_c <= f_r {
                    pop.individuals[worst_idx] = oi.into_iter().next().unwrap();
                    pop.fitness[worst_idx] = f_c;
                    ctx.bb.insert("nm_phase", 0u8);
                } else {
                    ctx.bb.insert("nm_phase", 4u8);
                }
            }
            // After TryContractInside, with f_cc = of[0]: accept iff f_cc < f_worst,
            // else shrink.
            3 => {
                let worst_idx = *ctx.bb.get::<usize>("nm_worst_idx")
                    .expect("replace/nelder-mead: TryContractInside requires nm_worst_idx");
                let f_cc = of[0];
                let f_worst = pop.fitness[worst_idx];
                if f_cc < f_worst {
                    pop.individuals[worst_idx] = oi.into_iter().next().unwrap();
                    pop.fitness[worst_idx] = f_cc;
                    ctx.bb.insert("nm_phase", 0u8);
                } else {
                    ctx.bb.insert("nm_phase", 4u8);
                }
            }
            // After Shrink: write the `dim` offspring (ascending non-best index
            // order — the same order gen/nelder-mead produced them in) into their
            // slots wholesale.
            4 => {
                let best_idx = pop.best_index()
                    .expect("replace/nelder-mead requires a non-empty population");
                let mut oi_iter = oi.into_iter();
                let mut of_iter = of.into_iter();
                for i in 0..pop.len() {
                    if i == best_idx { continue; }
                    pop.individuals[i] = oi_iter.next()
                        .expect("replace/nelder-mead: Shrink expects dim offspring individuals");
                    pop.fitness[i] = of_iter.next()
                        .expect("replace/nelder-mead: Shrink expects dim offspring fitness values");
                }
                ctx.bb.insert("nm_phase", 0u8);
            }
            p => unreachable!("replace/nelder-mead: invalid nm_phase byte {p}"),
        }
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta::new("replace/nelder-mead", SupportedBlocks::Only(vec!["float"]))
            .with_requires(vec![
                StateReq::of::<u8>("nm_phase"),
                StateReq::of::<Vec<f64>>("nm_centroid"),
                StateReq::of::<Vec<f64>>("nm_reflected"),
                StateReq::of::<f64>("nm_reflected_f"),
                StateReq::of::<usize>("nm_worst_idx"),
            ])
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/nelder-mead", |_| Ok(Box::new(NmGenerator)));
    reg.register_replacer("replace/nelder-mead", |_| Ok(Box::new(NmReplacer)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nm_meta_reports_single_offspring() {
        let m = NmGenerator.meta();
        assert_eq!(m.kind, "gen/nelder-mead");
        assert_eq!(m.offspring, OffspringCount::One); assert!(!m.internal_eval);
    }
    use sezgi_core::problem::{Evaluator, Problem, SphereShifted};
    use sezgi_core::rng::RngStream;
    use sezgi_core::state::Blackboard;

    fn g(xs: &[f64]) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs.to_vec())] } }
    fn sq(xs: &[f64]) -> f64 { xs.iter().map(|x| x * x).sum() }

    fn ctx_pieces() -> (SphereShifted, Blackboard, RngStream) {
        // f(x)=x.x's own problem is only needed to build a Ctx (space +
        // evaluator plumbing) — these tests drive generate/replace by hand,
        // supplying fitness values directly, never calling `evaluate`.
        let p = SphereShifted::new(vec![0.0; 2], -5.0, 5.0);
        (p, Blackboard::new(), RngStream::from_master(42, &[]))
    }

    /// Hand-derived (the arithmetic for each step is spelled out in the
    /// per-round comments below): starting simplex v0=(1,0) f=1, v1=(0,1)
    /// f=1, v2=(1,1) f=2 on f(x)=x·x, driven through its first four phase
    /// transitions.
    #[test]
    fn nm_hand_derived_first_four_transitions() {
        let (p, mut bb, mut rng) = ctx_pieces();
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);

        let mut pop = Population {
            individuals: vec![g(&[1.0, 0.0]), g(&[0.0, 1.0]), g(&[1.0, 1.0])],
            fitness: vec![1.0, 1.0, 2.0],
        };

        let gen = NmGenerator;
        let rep = NmReplacer;

        // Round 1: Reflect. worst_idx=2 (unique max f=2). centroid=(0.5,0.5).
        // x_r = 2c - x_worst = (1,1)-(1,1) = (0,0). f_r = 0.
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), 1);
            let x_r = float_view(&off[0]).clone();
            assert_eq!(x_r, vec![0.0, 0.0]);
            assert_eq!(*ctx.bb.get::<usize>("nm_worst_idx").unwrap(), 2);
            assert_eq!(ctx.bb.get::<Vec<f64>>("nm_centroid").unwrap(), &vec![0.5, 0.5]);

            let f_r = sq(&x_r);
            assert_eq!(f_r, 0.0);
            rep.replace(&mut pop, off, vec![f_r], &mut ctx);
            // f_r=0 < f_best=1: not accepted yet, move to TryExpand.
            assert_eq!(*ctx.bb.get::<u8>("nm_phase").unwrap(), 1);
            assert_eq!(*ctx.bb.get::<f64>("nm_reflected_f").unwrap(), 0.0);
        }
        // Population unchanged so far.
        assert_eq!(pop.fitness, vec![1.0, 1.0, 2.0]);

        // Round 2: TryExpand. x_e = c + 2*(x_r-c) = (0.5,0.5)+2*(-0.5,-0.5) = (-0.5,-0.5).
        // f_e = 0.5, which is NOT < f_r=0, so the reflected point (0,0)/f=0 is
        // accepted into the worst slot (index 2) instead of the expansion point.
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 1 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), 1);
            let x_e = float_view(&off[0]).clone();
            assert_eq!(x_e, vec![-0.5, -0.5]);
            let f_e = sq(&x_e);
            assert_eq!(f_e, 0.5);
            rep.replace(&mut pop, off, vec![f_e], &mut ctx);
            assert_eq!(*ctx.bb.get::<u8>("nm_phase").unwrap(), 0);
        }
        assert_eq!(pop.individuals[2], g(&[0.0, 0.0]));
        assert_eq!(pop.fitness, vec![1.0, 1.0, 0.0]);

        // Round 3: Reflect on the updated simplex v0=(1,0)f1, v1=(0,1)f1, v2=(0,0)f0.
        // worst_idx: max fitness is 1, tied between index0 and index1 -> highest
        // index (1) wins. centroid = avg(v0,v2) = (0.5,0.0).
        // x_r = 2c - x_worst = (1,0)-(0,1) = (1,-1). f_r = 2.
        // f_best=0, f_second_worst=max(f0=1,f2=0)=1, f_worst=1(index1).
        // f_r=2 >= f_worst=1 -> TryContractInside.
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 2 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(*ctx.bb.get::<usize>("nm_worst_idx").unwrap(), 1);
            let x_r = float_view(&off[0]).clone();
            assert_eq!(x_r, vec![1.0, -1.0]);
            let f_r = sq(&x_r);
            assert_eq!(f_r, 2.0);
            rep.replace(&mut pop, off, vec![f_r], &mut ctx);
            assert_eq!(*ctx.bb.get::<u8>("nm_phase").unwrap(), 3);
            assert_eq!(*ctx.bb.get::<f64>("nm_reflected_f").unwrap(), 2.0);
        }
        assert_eq!(pop.fitness, vec![1.0, 1.0, 0.0]); // still untouched

        // Round 4: TryContractInside. x_cc = c + 0.5*(x_worst-c)
        // = (0.5,0)+0.5*((0,1)-(0.5,0)) = (0.5,0)+(-0.25,0.5) = (0.25,0.5).
        // f_cc = 0.3125 < f_worst=1 -> accept into slot 1.
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 3 };
            let off = gen.generate(&pop, &mut ctx);
            let x_cc = float_view(&off[0]).clone();
            assert_eq!(x_cc, vec![0.25, 0.5]);
            let f_cc = sq(&x_cc);
            assert_eq!(f_cc, 0.3125);
            rep.replace(&mut pop, off, vec![f_cc], &mut ctx);
            assert_eq!(*ctx.bb.get::<u8>("nm_phase").unwrap(), 0);
        }
        assert_eq!(pop.individuals[1], g(&[0.25, 0.5]));
        assert_eq!(pop.fitness, vec![1.0, 0.3125, 0.0]);
    }

    /// Reflect's direct-accept branch: f_best <= f_r < f_second_worst.
    #[test]
    fn nm_reflect_direct_accept() {
        let (p, mut bb, mut rng) = ctx_pieces();
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);

        // v0=best f=0, v1 f=3 (second worst), v2=worst f=10.
        let mut pop = Population {
            individuals: vec![g(&[0.0, 0.0]), g(&[1.0, 1.0]), g(&[2.0, 2.0])],
            fitness: vec![0.0, 3.0, 10.0],
        };
        let gen = NmGenerator;
        let rep = NmReplacer;

        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let off = gen.generate(&pop, &mut ctx);
        assert_eq!(*ctx.bb.get::<usize>("nm_worst_idx").unwrap(), 2);
        // Directly supply an f_r landing in [f_best=0, f_second_worst=3).
        rep.replace(&mut pop, off, vec![1.5], &mut ctx);

        assert_eq!(*ctx.bb.get::<u8>("nm_phase").unwrap(), 0, "direct accept stays in Reflect");
        assert_eq!(pop.fitness[2], 1.5, "worst slot replaced with the reflected fitness");
    }

    /// TryExpand's accept-expansion branch: f_e < f_r.
    #[test]
    fn nm_try_expand_accepts_expansion() {
        let (p, mut bb, mut rng) = ctx_pieces();
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);

        let mut pop = Population {
            individuals: vec![g(&[0.0, 0.0]), g(&[1.0, 1.0]), g(&[2.0, 2.0])],
            fitness: vec![0.0, 1.0, 5.0],
        };
        let gen = NmGenerator;
        let rep = NmReplacer;

        // Round 1: Reflect, force f_r < f_best so we move to TryExpand.
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            rep.replace(&mut pop, off, vec![-10.0], &mut ctx); // f_r=-10 < f_best=0
            assert_eq!(*ctx.bb.get::<u8>("nm_phase").unwrap(), 1);
            assert_eq!(*ctx.bb.get::<f64>("nm_reflected_f").unwrap(), -10.0);
        }
        // Round 2: TryExpand, supply f_e < f_r=-10 -> expansion accepted.
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 1 };
            let off = gen.generate(&pop, &mut ctx);
            let x_e = float_view(&off[0]).clone();
            rep.replace(&mut pop, off, vec![-20.0], &mut ctx);
            assert_eq!(*ctx.bb.get::<u8>("nm_phase").unwrap(), 0);
            assert_eq!(pop.individuals[2], g(&x_e), "expansion point accepted, not the reflected point");
            assert_eq!(pop.fitness[2], -20.0);
        }
    }

    /// TryContractOutside's accept branch: f_c <= f_r.
    #[test]
    fn nm_try_contract_outside_accepts() {
        let (p, mut bb, mut rng) = ctx_pieces();
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);

        let mut pop = Population {
            individuals: vec![g(&[0.0, 0.0]), g(&[1.0, 1.0]), g(&[2.0, 2.0])],
            fitness: vec![0.0, 3.0, 10.0],
        };
        let gen = NmGenerator;
        let rep = NmReplacer;

        // Round 1: Reflect, force f_second_worst <= f_r < f_worst (f_r=5, in [3,10)).
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            rep.replace(&mut pop, off, vec![5.0], &mut ctx);
            assert_eq!(*ctx.bb.get::<u8>("nm_phase").unwrap(), 2);
            assert_eq!(*ctx.bb.get::<f64>("nm_reflected_f").unwrap(), 5.0);
        }
        // Round 2: TryContractOutside, f_c=4 <= f_r=5 -> accept.
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 1 };
            let off = gen.generate(&pop, &mut ctx);
            rep.replace(&mut pop, off, vec![4.0], &mut ctx);
            assert_eq!(*ctx.bb.get::<u8>("nm_phase").unwrap(), 0);
            assert_eq!(pop.fitness[2], 4.0);
        }
    }

    /// TryContractOutside's reject branch falls through to Shrink, and Shrink
    /// writes its `dim` offspring into the non-best slots in ascending index
    /// order.
    #[test]
    fn nm_try_contract_outside_rejects_to_shrink() {
        let (p, mut bb, mut rng) = ctx_pieces();
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);

        let mut pop = Population {
            individuals: vec![g(&[0.0, 0.0]), g(&[1.0, 1.0]), g(&[2.0, 2.0])],
            fitness: vec![0.0, 3.0, 10.0],
        };
        let gen = NmGenerator;
        let rep = NmReplacer;

        // Round 1: Reflect -> TryContractOutside (f_r=5, in [3,10)).
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            rep.replace(&mut pop, off, vec![5.0], &mut ctx);
            assert_eq!(*ctx.bb.get::<u8>("nm_phase").unwrap(), 2);
        }
        // Round 2: TryContractOutside, f_c=6 > f_r=5 -> Shrink.
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 1 };
            let off = gen.generate(&pop, &mut ctx);
            rep.replace(&mut pop, off, vec![6.0], &mut ctx);
            assert_eq!(*ctx.bb.get::<u8>("nm_phase").unwrap(), 4);
        }
        assert_eq!(pop.fitness, vec![0.0, 3.0, 10.0], "unchanged until shrink is applied");

        // Round 3: Shrink. best_idx=0 (f=0). Non-best indices in ascending
        // order: 1, 2. x_1' = x0 + 0.5*(x1-x0) = (0.5,0.5); x_2' = (1,1).
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 2 };
            let off = gen.generate(&pop, &mut ctx);
            assert_eq!(off.len(), 2);
            assert_eq!(float_view(&off[0]), &vec![0.5, 0.5]);
            assert_eq!(float_view(&off[1]), &vec![1.0, 1.0]);
            rep.replace(&mut pop, off, vec![0.5, 2.0], &mut ctx);
            assert_eq!(*ctx.bb.get::<u8>("nm_phase").unwrap(), 0);
        }
        assert_eq!(pop.individuals, vec![g(&[0.0, 0.0]), g(&[0.5, 0.5]), g(&[1.0, 1.0])]);
        assert_eq!(pop.fitness, vec![0.0, 0.5, 2.0]);
    }

    /// TryContractInside's reject branch also falls through to Shrink.
    #[test]
    fn nm_try_contract_inside_rejects_to_shrink() {
        let (p, mut bb, mut rng) = ctx_pieces();
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);

        let mut pop = Population {
            individuals: vec![g(&[0.0, 0.0]), g(&[1.0, 1.0]), g(&[2.0, 2.0])],
            fitness: vec![0.0, 3.0, 10.0],
        };
        let gen = NmGenerator;
        let rep = NmReplacer;

        // Round 1: Reflect -> TryContractInside (f_r=20 >= f_worst=10).
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
            let off = gen.generate(&pop, &mut ctx);
            rep.replace(&mut pop, off, vec![20.0], &mut ctx);
            assert_eq!(*ctx.bb.get::<u8>("nm_phase").unwrap(), 3);
        }
        // Round 2: TryContractInside, f_cc=12 >= f_worst=10 -> Shrink.
        {
            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 1 };
            let off = gen.generate(&pop, &mut ctx);
            rep.replace(&mut pop, off, vec![12.0], &mut ctx);
            assert_eq!(*ctx.bb.get::<u8>("nm_phase").unwrap(), 4);
        }
        assert_eq!(pop.fitness, vec![0.0, 3.0, 10.0]);
    }

    /// The replacer must trust the offspring it is actually handed (which,
    /// in the real engine, is POST-repair), not whatever `nm_reflected`
    /// value the generator wrote pre-repair. Simulate that divergence by
    /// hand: seed `nm_reflected` with a bogus "pre-repair" value the
    /// generator supposedly computed, then call `replace` with a `oi[0]`
    /// that differs (standing in for the engine's boundary-repaired point).
    /// After `replace`, `nm_reflected` must equal the offspring `replace`
    /// received, not the stale pre-repair value.
    #[test]
    fn reflect_stores_repaired_point() {
        let (p, mut bb, mut rng) = ctx_pieces();
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);

        let pop_before = vec![0.0, 1.0, 10.0]; // f_best=0.0
        let mut pop = Population {
            individuals: vec![g(&[0.0, 0.0]), g(&[1.0, 1.0]), g(&[2.0, 2.0])],
            fitness: pop_before,
        };

        // Hand-build the blackboard state gen/nelder-mead would have left
        // after a Reflect call, but with nm_reflected set to a stale
        // "pre-repair" value that does NOT match the offspring replace is
        // about to receive (simulating the generator's pre-repair x_r being
        // clamped by the engine's boundary repair before replace sees it).
        bb.insert("nm_phase", 0u8);
        bb.insert("nm_worst_idx", 2usize);
        bb.insert("nm_centroid", vec![0.5, 0.5]);
        bb.insert("nm_reflected", vec![99.0, 99.0]); // stale pre-repair value

        let repaired_offspring = g(&[-1.0, -1.0]); // what replace actually receives
        let f_r = -50.0; // f_r < f_best=0.0 -> TryExpand path, stores nm_reflected_f

        let rep = NmReplacer;
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        rep.replace(&mut pop, vec![repaired_offspring.clone()], vec![f_r], &mut ctx);

        assert_eq!(*ctx.bb.get::<u8>("nm_phase").unwrap(), 1, "f_r < f_best routes to TryExpand");
        assert_eq!(
            ctx.bb.get::<Vec<f64>>("nm_reflected").unwrap(),
            float_view(&repaired_offspring),
            "nm_reflected must be overwritten with the REPLACER-received (post-repair) offspring, \
             not the generator's stale pre-repair value"
        );
    }

    #[test]
    #[should_panic(expected = "pop_size must equal dim+1")]
    fn nm_asserts_simplex_size() {
        let (p, mut bb, mut rng) = ctx_pieces();
        let space = p.space(); // dim=2, so a valid simplex needs pop_size=3
        let mut evaluator = Evaluator::new(&p, 1000);
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };

        let pop = Population {
            individuals: vec![g(&[0.0, 0.0]), g(&[1.0, 1.0])], // only 2, not 3
            fitness: vec![0.0, 1.0],
        };
        let gen = NmGenerator;
        let _ = gen.generate(&pop, &mut ctx);
    }
}
