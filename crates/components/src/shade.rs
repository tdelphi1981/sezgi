use sezgi_core::component::*;
use sezgi_core::dist::Distribution;
use sezgi_core::problem::Population;
use sezgi_core::space::{BlockValues, Genotype};
use sezgi_core::state::StateReq;

/// SHADE: success-history adaptive DE with external archive (Tanabe &
/// Fukunaga 2013). `current-to-pbest/1` mutation with a per-individual `F_i`
/// (Cauchy) / `CR_i` (Gaussian) drawn from a historical memory of `h` slots,
/// plus binomial crossover with a guaranteed `j_rand`.
///
/// **Blackboard state** (all owned by this generator; see `meta().provides`):
/// - `shade_mf` / `shade_mcr`: `Vec<f64>` of length `h`, the success-history
///   memory (init 0.5 each slot).
/// - `shade_k`: `usize` memory write cursor (init 0; advanced by the
///   companion adapter).
/// - `shade_archive`: `Vec<Genotype>`, the external archive of replaced
///   parents (init empty).
/// - `shade_f` / `shade_cr`: `Vec<f64>` of length `n`, this generation's
///   per-individual `F_i`/`CR_i` (scratch, consumed by `replace/shade`).
/// - `shade_fit_prev`: `Vec<f64>`, a snapshot of `pop.fitness` taken *before*
///   generation (scratch, consumed by `replace/shade` as the authoritative
///   "parent fitness" — pinned so the replacer never has to assume the
///   engine hasn't touched `pop` in between).
///
/// **Pinned per-individual draw order** (part of the RNG-stream contract):
/// memory slot `r = next_below(h)` → `CR_i = clamp(Gaussian(M_CR[r], 0.1), 0, 1)`
/// → `F_i` = rejection-resample `Cauchy(M_F[r], 0.1)` until `> 0`, then
/// `min(F_i, 1.0)` (each rejected attempt consumes exactly one Cauchy draw)
/// → pbest index uniform among the `ceil(p·n)` best (tie-break by index) →
/// `r1` uniform `≠ i` (classic SHADE: **not** required `≠ pbest`, pinned) →
/// `r2` uniform over `population ∪ archive` (indices `[0,n)` = population,
/// `[n, n+archive.len())` = archive), `≠ i`, `≠ r1` — an archive-space
/// candidate can never coincide with `i` or `r1` (both population indices),
/// so the rejection only re-rolls when the candidate falls in `[0,n)` and
/// equals `i` or `r1` → `j_rand = next_below(d)` → per-gene binomial coin
/// (skipped, short-circuit, when `j == j_rand`).
///
/// Mutant: `v = x_i + F_i(x_pbest - x_i) + F_i(x_r1 - x_r2)`.
pub struct ShadeGenerator { pub h: usize, pub p: f64 }

impl ShadeGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams {
            kind: "gen/de-shade".into(), reason };
        let h = p.get("h").and_then(|v| v.as_u64()).unwrap_or(6) as usize;
        let pfrac = p.get("p").and_then(|v| v.as_f64()).unwrap_or(0.11);
        if h == 0 { return Err(err("h must be > 0".into())); }
        if !pfrac.is_finite() || !(0.0..=1.0).contains(&pfrac) || pfrac <= 0.0 {
            return Err(err(format!("p must be finite and in (0,1]: {pfrac}")));
        }
        Ok(Self { h, p: pfrac })
    }

    fn float_view(g: &Genotype) -> &Vec<f64> {
        match &g.blocks[0] { BlockValues::Float(xs) => xs, _ => unreachable!() }
    }
}

impl Generator for ShadeGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 4, "gen/de-shade requires a population of at least 4 (pop_size={})", pop.len());
        let n = pop.len();
        let h = self.h;

        // Lazily initialize the success-history memory (re-init if `h`
        // changed underneath us); the archive independently defaults empty.
        let needs_hist_init = ctx.bb.get::<Vec<f64>>("shade_mf").map(|v| v.len() != h).unwrap_or(true);
        if needs_hist_init {
            ctx.bb.insert("shade_mf", vec![0.5f64; h]);
            ctx.bb.insert("shade_mcr", vec![0.5f64; h]);
            ctx.bb.insert("shade_k", 0usize);
        }
        if !ctx.bb.contains("shade_archive") {
            ctx.bb.insert("shade_archive", Vec::<Genotype>::new());
        }

        // Snapshot the memory/archive out of the blackboard so the
        // per-individual loop only needs `ctx.rng` (mirrors the pattern in
        // pso.rs: pull blackboard state out before the closure).
        let mf = ctx.bb.get::<Vec<f64>>("shade_mf").unwrap().clone();
        let mcr = ctx.bb.get::<Vec<f64>>("shade_mcr").unwrap().clone();
        let archive = ctx.bb.get::<Vec<Genotype>>("shade_archive").unwrap().clone();
        let archive_len = archive.len();

        // pbest pool: ceil(p*n) best individuals, tie-break (fitness, index) — pinned.
        let pbest_count = ((self.p * n as f64).ceil() as usize).clamp(1, n);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| pop.fitness[a].total_cmp(&pop.fitness[b]).then(a.cmp(&b)));
        let pbest_pool = order[..pbest_count].to_vec();

        let fit_prev = pop.fitness.clone();
        let mut f_new = vec![0.0f64; n];
        let mut cr_new = vec![0.0f64; n];

        let offspring: Vec<Genotype> = (0..n).map(|i| {
            // Pinned draw order: slot r -> CR -> F (rejection) -> pbest -> r1 -> r2 -> j_rand -> crossover coins.
            let r = ctx.rng.next_below(h as u64) as usize;

            let cr_i = Distribution::Gaussian { mean: mcr[r], sigma: 0.1 }
                .sample(ctx.rng).clamp(0.0, 1.0);

            let f_i = loop {
                let f = Distribution::Cauchy { loc: mf[r], scale: 0.1 }.sample(ctx.rng);
                if f > 0.0 { break f.min(1.0); }
            };

            let pbest_idx = pbest_pool[ctx.rng.next_below(pbest_count as u64) as usize];

            let r1 = loop {
                let cand = ctx.rng.next_below(n as u64) as usize;
                if cand != i { break cand; }
            };

            // r2 over population (indices [0,n)) union archive (indices
            // [n, n+archive_len)); reject only when the candidate lands in
            // the population range and equals i or r1 (an archive-range
            // candidate is, by construction, never equal to a population
            // index, so it is always accepted immediately).
            let r2_idx = loop {
                let cand = ctx.rng.next_below((n + archive_len) as u64) as usize;
                if cand >= n || (cand != i && cand != r1) { break cand; }
            };

            let target = Self::float_view(&pop.individuals[i]);
            let pbest = Self::float_view(&pop.individuals[pbest_idx]);
            let x_r1 = Self::float_view(&pop.individuals[r1]);
            let x_r2 = if r2_idx < n {
                Self::float_view(&pop.individuals[r2_idx])
            } else {
                Self::float_view(&archive[r2_idx - n])
            };

            let d = target.len();
            let j_rand = ctx.rng.next_below(d as u64) as usize;
            let xs = (0..d).map(|j| {
                if j == j_rand || ctx.rng.next_f64() < cr_i {
                    target[j] + f_i * (pbest[j] - target[j]) + f_i * (x_r1[j] - x_r2[j])
                } else { target[j] }
            }).collect();

            f_new[i] = f_i;
            cr_new[i] = cr_i;

            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect();

        ctx.bb.insert("shade_f", f_new);
        ctx.bb.insert("shade_cr", cr_new);
        ctx.bb.insert("shade_fit_prev", fit_prev);

        offspring
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta { kind: "gen/de-shade",
            supported_blocks: SupportedBlocks::Only(vec!["float"]),
            requires: vec![],
            provides: vec![
                StateReq::of::<Vec<f64>>("shade_mf"),
                StateReq::of::<Vec<f64>>("shade_mcr"),
                StateReq::of::<usize>("shade_k"),
                StateReq::of::<Vec<Genotype>>("shade_archive"),
                StateReq::of::<Vec<f64>>("shade_f"),
                StateReq::of::<Vec<f64>>("shade_cr"),
                StateReq::of::<Vec<f64>>("shade_fit_prev"),
            ] }
    }
}

/// One-to-one greedy replacement, extended for SHADE's bookkeeping: for each
/// individual where the trial strictly improves on the parent, the parent's
/// genotype is pushed onto `shade_archive` (evicting a uniformly random
/// entry via `ctx.rng` while the archive exceeds `pop.len()` — pinned
/// eviction policy), and `(F_i, CR_i, Δf = parent_fit − off_fit)` is
/// recorded into fresh per-generation success lists `shade_sf`/`shade_scr`/
/// `shade_dw` (overwritten each call — these are pure per-generation
/// scratch, cleared again once consumed by `adapter/shade-history`).
pub struct ShadeReplacer;

impl Replacer for ShadeReplacer {
    fn replace(&self, pop: &mut Population, oi: Vec<Genotype>, of: Vec<f64>, ctx: &mut Ctx) {
        let n = pop.len();
        let f = ctx.bb.get::<Vec<f64>>("shade_f").unwrap().clone();
        let cr = ctx.bb.get::<Vec<f64>>("shade_cr").unwrap().clone();
        let fit_prev = ctx.bb.get::<Vec<f64>>("shade_fit_prev").unwrap().clone();

        let mut sf = Vec::new();
        let mut scr = Vec::new();
        let mut dw = Vec::new();

        for (i, (gi, fi)) in oi.into_iter().zip(of).enumerate() {
            if i >= n { break; }
            let parent_fit = fit_prev[i];
            if fi < parent_fit {
                let parent_genotype = std::mem::replace(&mut pop.individuals[i], gi);
                pop.fitness[i] = fi;

                sf.push(f[i]);
                scr.push(cr[i]);
                dw.push(parent_fit - fi);

                let archive = ctx.bb.get_mut::<Vec<Genotype>>("shade_archive").unwrap();
                archive.push(parent_genotype);
                while archive.len() > n {
                    let idx = ctx.rng.next_below(archive.len() as u64) as usize;
                    archive.remove(idx);
                }
            }
        }

        ctx.bb.insert("shade_sf", sf);
        ctx.bb.insert("shade_scr", scr);
        ctx.bb.insert("shade_dw", dw);
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta { kind: "replace/shade",
            supported_blocks: SupportedBlocks::All,
            requires: vec![
                StateReq::of::<Vec<f64>>("shade_f"),
                StateReq::of::<Vec<f64>>("shade_cr"),
                StateReq::of::<Vec<f64>>("shade_fit_prev"),
                StateReq::of::<Vec<Genotype>>("shade_archive"),
            ],
            provides: vec![
                StateReq::of::<Vec<f64>>("shade_sf"),
                StateReq::of::<Vec<f64>>("shade_scr"),
                StateReq::of::<Vec<f64>>("shade_dw"),
            ] }
    }
}

/// Companion adapter for `gen/de-shade` / `replace/shade`: the success-history
/// memory update (Tanabe & Fukunaga 2013 eq. 5-7). If this generation's
/// success lists are empty, does nothing (RNG-free either way). Otherwise,
/// with weights `w_k = Δf_k / Σ Δf`:
/// - `M_F[k_cursor] = (Σ w·F²) / (Σ w·F)` — the weighted Lehmer mean.
/// - `M_CR[k_cursor] = Σ w·CR` — the weighted arithmetic mean.
///
/// `shade_k` then advances cyclically (`(k_cursor + 1) % h`) and the success
/// lists are cleared.
pub struct ShadeHistoryAdapter;

impl Adapter for ShadeHistoryAdapter {
    fn adapt(&self, _pop: &mut Population, ctx: &mut Ctx) {
        let sf = ctx.bb.get::<Vec<f64>>("shade_sf").cloned().unwrap_or_default();
        let scr = ctx.bb.get::<Vec<f64>>("shade_scr").cloned().unwrap_or_default();
        let dw = ctx.bb.get::<Vec<f64>>("shade_dw").cloned().unwrap_or_default();

        if sf.is_empty() { return; }

        let sum_dw: f64 = dw.iter().sum();
        let mut num_f = 0.0;
        let mut den_f = 0.0;
        let mut mcr_new = 0.0;
        for k in 0..sf.len() {
            let w = dw[k] / sum_dw;
            num_f += w * sf[k] * sf[k];
            den_f += w * sf[k];
            mcr_new += w * scr[k];
        }
        let mf_new = num_f / den_f;

        let k_cursor = *ctx.bb.get::<usize>("shade_k").unwrap();
        let h = ctx.bb.get::<Vec<f64>>("shade_mf").unwrap().len();
        let slot = k_cursor % h;

        ctx.bb.get_mut::<Vec<f64>>("shade_mf").unwrap()[slot] = mf_new;
        ctx.bb.get_mut::<Vec<f64>>("shade_mcr").unwrap()[slot] = mcr_new;
        ctx.bb.insert("shade_k", (k_cursor + 1) % h);

        ctx.bb.insert("shade_sf", Vec::<f64>::new());
        ctx.bb.insert("shade_scr", Vec::<f64>::new());
        ctx.bb.insert("shade_dw", Vec::<f64>::new());
    }
    fn meta(&self) -> ComponentMeta {
        ComponentMeta { kind: "adapter/shade-history",
            supported_blocks: SupportedBlocks::All,
            requires: vec![
                StateReq::of::<Vec<f64>>("shade_mf"),
                StateReq::of::<Vec<f64>>("shade_mcr"),
                StateReq::of::<usize>("shade_k"),
                StateReq::of::<Vec<f64>>("shade_sf"),
                StateReq::of::<Vec<f64>>("shade_scr"),
                StateReq::of::<Vec<f64>>("shade_dw"),
            ],
            provides: vec![] }
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/de-shade", |p| Ok(Box::new(ShadeGenerator::from_params(p)?)));
    reg.register_replacer("replace/shade", |_| Ok(Box::new(ShadeReplacer)));
    reg.register_adapter("adapter/shade-history", |_| Ok(Box::new(ShadeHistoryAdapter)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Problem, SphereShifted, Evaluator};
    use sezgi_core::rng::RngStream;
    use sezgi_core::state::Blackboard;

    #[test]
    fn params_parse_and_validate() {
        let g = ShadeGenerator::from_params(&serde_json::json!({"h": 10, "p": 0.2})).unwrap();
        assert_eq!(g.h, 10);
        assert_eq!(g.p, 0.2);

        let d = ShadeGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!((d.h, d.p), (6, 0.11));

        assert!(ShadeGenerator::from_params(&serde_json::json!({"h": 0})).is_err());
        assert!(ShadeGenerator::from_params(&serde_json::json!({"p": 0.0})).is_err());
        assert!(ShadeGenerator::from_params(&serde_json::json!({"p": 1.5})).is_err());
    }

    /// Hand-computed (see task brief): successes F=[0.5,1.0], CR=[0.2,0.6],
    /// Δf=[1.0,3.0] ⇒ w=[0.25,0.75] ⇒
    /// M_F = (0.25·0.25 + 0.75·1.0) / (0.25·0.5 + 0.75·1.0) = 0.8125/0.875 ≈ 0.928571428571...
    /// M_CR = 0.25·0.2 + 0.75·0.6 = 0.5
    #[test]
    fn shade_memory_updates_use_lehmer_mean() {
        let p = SphereShifted::new(vec![0.0], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(42, &[]);
        let mut bb = Blackboard::new();

        bb.insert("shade_mf", vec![0.5f64, 0.5]);
        bb.insert("shade_mcr", vec![0.5f64, 0.5]);
        bb.insert("shade_k", 0usize);
        bb.insert("shade_sf", vec![0.5f64, 1.0]);
        bb.insert("shade_scr", vec![0.2f64, 0.6]);
        bb.insert("shade_dw", vec![1.0f64, 3.0]);

        let mut pop = Population { individuals: vec![], fitness: vec![] };
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let adapter = ShadeHistoryAdapter;
        adapter.adapt(&mut pop, &mut ctx);

        let mf = ctx.bb.get::<Vec<f64>>("shade_mf").unwrap();
        let mcr = ctx.bb.get::<Vec<f64>>("shade_mcr").unwrap();
        assert!((mf[0] - 0.9285714285714286).abs() < 1e-12, "M_F={}", mf[0]);
        assert!((mcr[0] - 0.5).abs() < 1e-12, "M_CR={}", mcr[0]);
        // untouched slot stays at init
        assert_eq!(mf[1], 0.5);
        assert_eq!(mcr[1], 0.5);

        assert_eq!(*ctx.bb.get::<usize>("shade_k").unwrap(), 1, "cursor should advance cyclically");
        assert!(ctx.bb.get::<Vec<f64>>("shade_sf").unwrap().is_empty());
        assert!(ctx.bb.get::<Vec<f64>>("shade_scr").unwrap().is_empty());
        assert!(ctx.bb.get::<Vec<f64>>("shade_dw").unwrap().is_empty());
    }

    #[test]
    fn shade_history_adapter_skips_empty_success_lists() {
        let p = SphereShifted::new(vec![0.0], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(42, &[]);
        let mut bb = Blackboard::new();

        bb.insert("shade_mf", vec![0.5f64, 0.5]);
        bb.insert("shade_mcr", vec![0.5f64, 0.5]);
        bb.insert("shade_k", 0usize);
        bb.insert("shade_sf", Vec::<f64>::new());
        bb.insert("shade_scr", Vec::<f64>::new());
        bb.insert("shade_dw", Vec::<f64>::new());

        let mut pop = Population { individuals: vec![], fitness: vec![] };
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let adapter = ShadeHistoryAdapter;
        adapter.adapt(&mut pop, &mut ctx);

        assert_eq!(ctx.bb.get::<Vec<f64>>("shade_mf").unwrap(), &vec![0.5, 0.5]);
        assert_eq!(*ctx.bb.get::<usize>("shade_k").unwrap(), 0, "cursor must not advance on a no-op generation");
    }

    #[test]
    fn shade_archive_bounded() {
        fn g(x: f64) -> Genotype { Genotype { blocks: vec![BlockValues::Float(vec![x])] } }

        let p = SphereShifted::new(vec![0.0], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 10_000);
        let mut rng = RngStream::from_master(1, &[]);
        let mut bb = Blackboard::new();

        let n = 4usize;
        bb.insert("shade_archive", Vec::<Genotype>::new());

        let mut pop = Population {
            individuals: (0..n).map(|i| g(i as f64)).collect(),
            fitness: vec![100.0; n],
        };

        let replacer = ShadeReplacer;
        for gen in 0..10u64 {
            let fit_prev = pop.fitness.clone();
            bb.insert("shade_f", vec![0.5f64; n]);
            bb.insert("shade_cr", vec![0.9f64; n]);
            bb.insert("shade_fit_prev", fit_prev.clone());

            // Every offspring strictly improves on its parent, so every
            // individual is replaced and pushed to the archive every
            // generation — this is the stress case for the eviction bound.
            let offspring: Vec<Genotype> = (0..n).map(|i| g(i as f64 + gen as f64)).collect();
            let off_fit: Vec<f64> = fit_prev.iter().map(|f| f - 1.0).collect();

            let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: gen };
            replacer.replace(&mut pop, offspring, off_fit, &mut ctx);

            let archive_len = ctx.bb.get::<Vec<Genotype>>("shade_archive").unwrap().len();
            assert!(archive_len <= pop.len(),
                "archive length {archive_len} must never exceed pop.len() {}", pop.len());
        }

        let final_len = bb.get::<Vec<Genotype>>("shade_archive").unwrap().len();
        assert_eq!(final_len, n, "archive should saturate at pop.len() after repeated replacements");
    }
}
