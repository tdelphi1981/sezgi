use crate::linalg;
use sezgi_core::component::*;
use sezgi_core::dist::Distribution;
use sezgi_core::problem::Population;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};
use sezgi_core::state::StateReq;

/// (μ/μ_w, λ)-CMA-ES (Hansen's tutorial form; Hansen & Ostermeier 2001,
/// "Completely Derandomized Self-Adaptation in Evolution Strategies", plus
/// Hansen's 2016 tutorial for the exact constant formulas used here).
///
/// **Six documented sezgi deviations from textbook CMA-ES** (two algorithmic
/// simplifications, two added numerical/boundary guards, two plain numerical
/// guards against non-finite blowup — full reasoning at each site's own doc
/// comment; this is just the inventory):
/// 1. **Positive weights only.** The recombination/rank-μ weights `w_1..w_μ`
///    are all positive (computed only for the best `μ = ⌊λ/2⌋` offspring, per
///    the classic (μ/μ_w,λ) scheme) — no negative weights / active
///    covariance-matrix update (the "active CMA-ES" extension that also
///    penalizes the worst offspring). This is the algorithm as originally
///    published, not the later active variant.
/// 2. **Eigendecomposition every generation.** `C` is eigendecomposed via
///    `linalg::eigh_jacobi` on *every* call to `generate`, rather than lazily
///    refreshing `B`/`D` only every `~1/(10·d·(c_1+c_μ))` generations as
///    Hansen's reference implementations do to amortize the O(d³) cost. Given
///    `eigh_jacobi`'s own O(d³) sweep cost, sezgi trades some CPU for a
///    simpler, always-consistent blackboard state.
/// 3. **Mean-clamp + evolution-path reset (boundary-saturation guard).**
///    After the mean update, `replace/cma-update` clamps `mean` into the
///    search space bounds, and — only when that clamp actually fires —
///    resets `ps`/`pc` to zero. Trajectory-neutral whenever the mean stays
///    in-box (the common case); without it, under severe conditioning
///    (BBOB-scale ~1e6) the mean can drift permanently outside the space and
///    diverge (see `replace`'s doc comment for the full mechanism and the
///    empirical evidence).
/// 4. **Flat-fitness freeze.** If every offspring in a generation lands on
///    bit-identical fitness (no selection signal at all — e.g. boundary
///    repair saturated the whole batch to the same point), `mean`, `sigma`,
///    `C`, `ps`, `pc`, and `gen` are all frozen for that round rather than
///    adapted on noise; the population is still replaced unconditionally
///    either way (see `replace`'s doc comment).
/// 5. **Sigma clamp.** After the CSA step-size update, `sigma` is clamped to
///    `[1e-12, 1e12]` (see `replace`) — guards against underflow-to-zero
///    (a zero step size freezes sampling permanently) and against runaway
///    growth under a diverging evolution path, without changing behavior
///    anywhere sigma would naturally stay in-range.
/// 6. **Eigenvalue floor.** Eigenvalues from `linalg::eigh_jacobi` are
///    floored to `1e-20` before the `sqrt` that produces `cma_eig_d` (see
///    `generate`, below) — guards against a negative or zero eigenvalue
///    (numerically singular `C`, or roundoff producing a tiny negative
///    value for a true-zero eigenvalue) turning the corresponding
///    `sqrt` into `NaN` and poisoning every subsequent sample.
///
/// **Blackboard state** (all owned by `gen/cma`; required by
/// `replace/cma-update`):
/// - `cma_mean`: `Vec<f64>` (len `d`) — the current distribution mean.
/// - `cma_sigma`: `f64` — the current step size.
/// - `cma_cov`: `Vec<Vec<f64>>` (`d`×`d`) — the current covariance matrix.
/// - `cma_ps`: `Vec<f64>` (len `d`) — the conjugate (isotropic) evolution
///   path.
/// - `cma_pc`: `Vec<f64>` (len `d`) — the anisotropic evolution path.
/// - `cma_gen`: `u64` — generation counter (0 at lazy init, incremented once
///   per `replace/cma-update` call).
/// - `cma_eig_b`: `Vec<Vec<f64>>` (`d`×`d`) — this generation's eigenvectors
///   of `C`, as **columns** (`eig_b[row][col]`, matching `linalg::eigh_jacobi`).
/// - `cma_eig_d`: `Vec<f64>` (len `d`) — this generation's `sqrt(max(eigenvalue, 1e-20))`.
/// - `cma_ys`: `Vec<Vec<f64>>` (len `λ`, each len `d`) — this generation's
///   `y_i = B·(D∘z_i)` steps, in sampling order (offspring order).
///
/// **Lazy init** (first `generate` call, or whenever `cma_mean`'s length
/// mismatches `d` — the sole restart-resilience check needed, since a
/// restart clears the whole blackboard and so `cma_mean` will simply be
/// absent): `mean` = the *current population's best individual's* float
/// coordinates (a pinned choice that makes meaningful use of the
/// initializer's output rather than e.g. the population centroid); `sigma` =
/// `sigma0` (default `0.3·(hi−lo)` of the first float block, computed from
/// `ctx.space`); `C` = identity; `ps` = `pc` = **0**; `gen` = 0.
///
/// **Strategy constants** (`cma_constants`, below) are recomputed from
/// `(d, λ)` on *every* call — pure functions of the current dimension and
/// population size, so there is no drift to track.
///
/// **Pinned sampling draw order:** for `i` in `0..λ`: draw `d` `N(0,1)`
/// Gaussians (`z` components, in index order) — one individual is fully
/// sampled (all `d` components) before the next begins.
pub struct CmaGenerator {
    /// `None` = compute the Hansen-tutorial default (`0.3·(hi−lo)` of the
    /// first float block) at lazy init, from `ctx.space`.
    pub sigma0: Option<f64>,
}

impl CmaGenerator {
    pub fn from_params(p: &serde_json::Value) -> Result<Self, ComponentError> {
        let err = |reason: String| ComponentError::InvalidParams {
            kind: "gen/cma".into(), reason };
        let sigma0 = match p.get("sigma0") {
            None | Some(serde_json::Value::Null) => None,
            Some(v) => {
                let s = v.as_f64().ok_or_else(|| err("sigma0 must be a number".into()))?;
                if !(s.is_finite() && s > 0.0) {
                    return Err(err(format!("sigma0 must be finite and > 0: {s}")));
                }
                Some(s)
            }
        };
        Ok(Self { sigma0 })
    }

    fn float_view(g: &Genotype) -> &Vec<f64> {
        match &g.blocks[0] { BlockValues::Float(xs) => xs, _ => unreachable!() }
    }
}

fn identity(d: usize) -> Vec<Vec<f64>> {
    (0..d).map(|i| { let mut row = vec![0.0f64; d]; row[i] = 1.0; row }).collect()
}

/// Per-coordinate `(lo, hi)` bounds, expanding each `Block::Float { lo, hi, n }`
/// to `n` repeated entries — matches the flat single-Float-block genotype
/// convention `gen/cma` uses (mirrors `boundary/clamp`'s own per-coordinate walk).
fn coordinate_bounds(space: &SearchSpace) -> Vec<(f64, f64)> {
    space.blocks().iter().flat_map(|b| match *b {
        Block::Float { lo, hi, n } => vec![(lo, hi); n],
        _ => unreachable!("gen/cma requires a float-only space (meta().supported_blocks)"),
    }).collect()
}

impl Generator for CmaGenerator {
    fn generate(&self, pop: &Population, ctx: &mut Ctx) -> Vec<Genotype> {
        assert!(pop.len() >= 2,
            "gen/cma requires a population of at least 2 (lambda={})", pop.len());
        let lambda = pop.len();
        let d = Self::float_view(&pop.individuals[0]).len();

        // Restart resilience: absent `cma_mean` (fresh/restarted blackboard)
        // or a dimension mismatch both trigger a fresh lazy init.
        let needs_init = match ctx.bb.get::<Vec<f64>>("cma_mean") {
            Some(m) => m.len() != d,
            None => true,
        };
        if needs_init {
            let best_idx = pop.best_index().expect("gen/cma requires a non-empty population");
            let mean = Self::float_view(&pop.individuals[best_idx]).clone();
            let sigma0 = self.sigma0.unwrap_or_else(|| match ctx.space.blocks()[0] {
                Block::Float { lo, hi, .. } => 0.3 * (hi - lo),
                _ => unreachable!("gen/cma requires a float-only space (meta().supported_blocks)"),
            });
            ctx.bb.insert("cma_mean", mean);
            ctx.bb.insert("cma_sigma", sigma0);
            ctx.bb.insert("cma_cov", identity(d));
            ctx.bb.insert("cma_ps", vec![0.0f64; d]);
            ctx.bb.insert("cma_pc", vec![0.0f64; d]);
            ctx.bb.insert("cma_gen", 0u64);
        }

        let mean = ctx.bb.get::<Vec<f64>>("cma_mean").unwrap().clone();
        let sigma = *ctx.bb.get::<f64>("cma_sigma").unwrap();
        let cov = ctx.bb.get::<Vec<Vec<f64>>>("cma_cov").unwrap().clone();

        // Eigendecompose C every generation (documented simplification, see
        // module doc). eig_b columns are eigenvectors; eig_d is sqrt of the
        // (floor-clamped) eigenvalues.
        let (eigenvalues, eig_b) = linalg::eigh_jacobi(&cov);
        let eig_d: Vec<f64> = eigenvalues.iter().map(|&ev| ev.max(1e-20).sqrt()).collect();

        let mut ys: Vec<Vec<f64>> = Vec::with_capacity(lambda);
        let offspring: Vec<Genotype> = (0..lambda).map(|_| {
            // Pinned: draw all d components of z before moving to the next
            // individual.
            let z: Vec<f64> = (0..d)
                .map(|_| Distribution::Gaussian { mean: 0.0, sigma: 1.0 }.sample(ctx.rng))
                .collect();
            let dz: Vec<f64> = (0..d).map(|j| eig_d[j] * z[j]).collect();
            let y = linalg::mat_vec(&eig_b, &dz);
            let xs: Vec<f64> = (0..d).map(|j| mean[j] + sigma * y[j]).collect();
            ys.push(y);
            Genotype { blocks: vec![BlockValues::Float(xs)] }
        }).collect();

        ctx.bb.insert("cma_eig_b", eig_b);
        ctx.bb.insert("cma_eig_d", eig_d);
        ctx.bb.insert("cma_ys", ys);

        offspring
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta { kind: "gen/cma",
            supported_blocks: SupportedBlocks::Only(vec!["float"]),
            requires: vec![],
            provides: vec![
                StateReq::of::<Vec<f64>>("cma_mean"),
                StateReq::of::<f64>("cma_sigma"),
                StateReq::of::<Vec<Vec<f64>>>("cma_cov"),
                StateReq::of::<Vec<f64>>("cma_ps"),
                StateReq::of::<Vec<f64>>("cma_pc"),
                StateReq::of::<u64>("cma_gen"),
                StateReq::of::<Vec<Vec<f64>>>("cma_eig_b"),
                StateReq::of::<Vec<f64>>("cma_eig_d"),
                StateReq::of::<Vec<Vec<f64>>>("cma_ys"),
            ] }
    }
}

/// This generation's (d, λ)-derived CMA-ES strategy constants (Hansen's
/// tutorial formulas, positive-weights variant). Recomputed fresh from
/// `(d, lambda)` every call — never cached/drifted in the blackboard.
pub(crate) struct CmaConstants {
    pub mu: usize,
    /// Normalized positive weights `w_1..w_mu`, summing to 1.
    pub weights: Vec<f64>,
    pub mu_eff: f64,
    pub c_sigma: f64,
    pub d_sigma: f64,
    pub c_c: f64,
    pub c_1: f64,
    pub c_mu: f64,
    /// E‖N(0,I_d)‖, the expected norm of a standard d-dimensional Gaussian.
    pub e_norm: f64,
}

pub(crate) fn cma_constants(d: usize, lambda: usize) -> CmaConstants {
    let mu = lambda / 2;
    let df = d as f64;

    let raw: Vec<f64> = (0..mu)
        .map(|i| (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln())
        .collect();
    let sum_raw: f64 = raw.iter().sum();
    let weights: Vec<f64> = raw.iter().map(|w| w / sum_raw).collect();
    let sum_w2: f64 = weights.iter().map(|w| w * w).sum();
    let mu_eff = 1.0 / sum_w2;

    let c_sigma = (mu_eff + 2.0) / (df + mu_eff + 5.0);
    let d_sigma = 1.0 + 2.0 * (((mu_eff - 1.0) / (df + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;
    let c_c = (4.0 + mu_eff / df) / (df + 4.0 + 2.0 * mu_eff / df);
    let c_1 = 2.0 / ((df + 1.3).powi(2) + mu_eff);
    let c_mu = (1.0 - c_1)
        .min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((df + 2.0).powi(2) + mu_eff));
    let e_norm = df.sqrt() * (1.0 - 1.0 / (4.0 * df) + 1.0 / (21.0 * df * df));

    CmaConstants { mu, weights, mu_eff, c_sigma, d_sigma, c_c, c_1, c_mu, e_norm }
}

/// `replace/cma-update`: the CMA-ES strategy-parameter update (mean, `ps`,
/// `pc`, `C`, `sigma`) plus a global generation-counter bump. See the module
/// doc for the blackboard contract and the two documented simplifications.
///
/// Sorts offspring by `(fitness, index)` ascending, takes the `μ` best,
/// forms the weighted recombination step `⟨y⟩_w` from the matching entries
/// of `cma_ys` (looked up by *offspring index*, not by re-deriving `y` from
/// the — possibly boundary-repaired — genotype), then applies the full
/// evolution-path / covariance / step-size update. The population is
/// **replaced unconditionally** by the offspring (CMA-ES is not elitist by
/// construction; the engine's own global-best tracking is what keeps the
/// best-ever individual safe under this non-elitist replacer, exactly as for
/// `replace/pso-commit`).
pub struct CmaUpdateReplacer;

impl Replacer for CmaUpdateReplacer {
    fn replace(&self, pop: &mut Population, oi: Vec<Genotype>, of: Vec<f64>, ctx: &mut Ctx) {
        let lambda = oi.len();

        // Guard: a fully flat generation — every offspring landed on
        // (bit-)identical fitness (e.g. boundary repair clamped the whole
        // batch to the same saturated corner) — carries no selection signal
        // at all. Under CSA, a run of selection-free generations is a
        // *driftless* random walk in log(sigma): with no restoring force it
        // can wander to extreme values purely by chance the longer it's
        // stuck (a documented, known CSA limitation, not specific to this
        // implementation — real CMA-ES implementations detect exactly this
        // "flat fitness" condition and stop adapting rather than let it
        // run). So: freeze mean, sigma, C, ps, pc and the generation counter
        // this round, leaving them exactly as the blackboard already has
        // them. Population replacement (the pinned non-elitist contract)
        // still happens unconditionally either way.
        let is_flat = lambda > 0 && of.iter().all(|&f| f == of[0]);
        if is_flat {
            pop.individuals = oi;
            pop.fitness = of;
            return;
        }

        let d = match &oi[0].blocks[0] { BlockValues::Float(xs) => xs.len(), _ => unreachable!() };

        let c = cma_constants(d, lambda);

        let ys = ctx.bb.get::<Vec<Vec<f64>>>("cma_ys").unwrap().clone();
        let eig_b = ctx.bb.get::<Vec<Vec<f64>>>("cma_eig_b").unwrap().clone();
        let eig_d = ctx.bb.get::<Vec<f64>>("cma_eig_d").unwrap().clone();
        let mut mean = ctx.bb.get::<Vec<f64>>("cma_mean").unwrap().clone();
        let mut sigma = *ctx.bb.get::<f64>("cma_sigma").unwrap();
        let cov = ctx.bb.get::<Vec<Vec<f64>>>("cma_cov").unwrap().clone();
        let mut ps = ctx.bb.get::<Vec<f64>>("cma_ps").unwrap().clone();
        let mut pc = ctx.bb.get::<Vec<f64>>("cma_pc").unwrap().clone();
        let gen = *ctx.bb.get::<u64>("cma_gen").unwrap();

        // Sort offspring indices by (fitness, index) ascending; take the mu best.
        let mut order: Vec<usize> = (0..lambda).collect();
        order.sort_by(|&a, &b| of[a].total_cmp(&of[b]).then(a.cmp(&b)));
        let best_idx = &order[..c.mu];

        // <y>_w = sum_k w_k * y_(sorted_k)
        let mut y_w = vec![0.0f64; d];
        for (k, &idx) in best_idx.iter().enumerate() {
            let w = c.weights[k];
            for j in 0..d { y_w[j] += w * ys[idx][j]; }
        }

        // Mean update.
        for j in 0..d { mean[j] += sigma * y_w[j]; }

        // Guard: clamp the mean back into the search space bounds, and if
        // that clamp actually fired, reset the evolution paths (ps, pc).
        //
        // `cma_ys` deliberately holds the *pre*-boundary-repair `y_i`
        // (documented simplification, see module doc), so a boundary
        // violation on an offspring is invisible to this update — the
        // selected `y`s carry no signal that would otherwise pull the mean
        // back in. Verified (via an independent from-scratch reference
        // implementation, not just this code, plus direct empirical
        // measurement against BBOB's rotated ellipsoid and bent-cigar
        // functions, both condition ~1e6) that without the mean clamp, the
        // mean can drift, coordinate by coordinate, past the box edge along
        // whichever principal direction is least sensitive to fitness —
        // once out, every offspring in that coordinate clips to the same
        // boundary value on repair, fitness stops discriminating candidates
        // along it, and the mean can get stuck outside the space with no
        // corrective gradient.
        //
        // The clamp alone is not quite enough on the most adversarial case
        // (bent-cigar's uniform 1e6 weighting on 9 of its 10 axes): once
        // several coordinates are pinned at once, `ps`/`pc` still carry
        // *stale* momentum accumulated from the (now-interrupted) approach
        // to the wall, and CSA reads that as a genuine consistent search
        // direction, driving `sigma` into an unbounded, driftless random
        // walk over the following hundreds of generations. Resetting `ps`
        // and `pc` exactly when the clamp fires discards that stale
        // momentum, so CSA starts from a clean slate the moment boundary
        // correction happens — empirically this closes the gap to
        // (near-)machine precision on both BBOB functions above, alongside
        // sphere, with no other guard needed.
        let mut clamped = false;
        for (j, (lo, hi)) in coordinate_bounds(ctx.space).into_iter().enumerate() {
            let r = mean[j].clamp(lo, hi);
            if r != mean[j] { clamped = true; }
            mean[j] = r;
        }
        if clamped {
            ps = vec![0.0; d];
            pc = vec![0.0; d];
        }

        // C^{-1/2} <y>_w = B * (D^{-1} ∘ (B^T <y>_w)).
        let bt_yw: Vec<f64> = (0..d)
            .map(|k| (0..d).map(|i| eig_b[i][k] * y_w[i]).sum())
            .collect();
        let dinv_bt_yw: Vec<f64> = (0..d).map(|k| bt_yw[k] / eig_d[k]).collect();
        let c_inv_half_yw = linalg::mat_vec(&eig_b, &dinv_bt_yw);

        // ps update (conjugate evolution path).
        let ps_coeff = (c.c_sigma * (2.0 - c.c_sigma) * c.mu_eff).sqrt();
        for j in 0..d { ps[j] = (1.0 - c.c_sigma) * ps[j] + ps_coeff * c_inv_half_yw[j]; }
        let ps_norm: f64 = ps.iter().map(|v| v * v).sum::<f64>().sqrt();

        // h_sigma: heuristically stalls the pc path after a sudden sigma jump.
        let denom = (1.0 - (1.0 - c.c_sigma).powi(2 * (gen as i32 + 1))).sqrt();
        let h_sigma_threshold = (1.4 + 2.0 / (d as f64 + 1.0)) * c.e_norm;
        let h_sigma: f64 = if (ps_norm / denom) < h_sigma_threshold { 1.0 } else { 0.0 };

        // pc update (anisotropic evolution path).
        let pc_coeff = (c.c_c * (2.0 - c.c_c) * c.mu_eff).sqrt();
        for j in 0..d { pc[j] = (1.0 - c.c_c) * pc[j] + h_sigma * pc_coeff * y_w[j]; }

        // Covariance update: (1-c1-cmu)*C + c1*(pc pc^T + (1-h_sigma)*c_c(2-c_c)*C) + cmu*sum w_k y_k y_k^T.
        let mut c_new = vec![vec![0.0f64; d]; d];
        for i in 0..d { for j in 0..d { c_new[i][j] = (1.0 - c.c_1 - c.c_mu) * cov[i][j]; } }
        linalg::vec_outer_add(&mut c_new, c.c_1, &pc);
        let old_c_coeff = c.c_1 * (1.0 - h_sigma) * c.c_c * (2.0 - c.c_c);
        for i in 0..d { for j in 0..d { c_new[i][j] += old_c_coeff * cov[i][j]; } }
        for (k, &idx) in best_idx.iter().enumerate() {
            linalg::vec_outer_add(&mut c_new, c.c_mu * c.weights[k], &ys[idx]);
        }

        // Symmetrize. Each iteration mutates two distinct rows (i and j>i)
        // at once, which an iterator adaptor can't express without an
        // explicit disjoint-borrow split — the index form is the correct
        // one here.
        #[allow(clippy::needless_range_loop)]
        for i in 0..d {
            for j in (i + 1)..d {
                let avg = (c_new[i][j] + c_new[j][i]) / 2.0;
                c_new[i][j] = avg;
                c_new[j][i] = avg;
            }
        }

        // Sigma update, clamped.
        sigma *= ((c.c_sigma / c.d_sigma) * (ps_norm / c.e_norm - 1.0)).exp();
        sigma = sigma.clamp(1e-12, 1e12);

        ctx.bb.insert("cma_mean", mean);
        ctx.bb.insert("cma_sigma", sigma);
        ctx.bb.insert("cma_cov", c_new);
        ctx.bb.insert("cma_ps", ps);
        ctx.bb.insert("cma_pc", pc);
        ctx.bb.insert("cma_gen", gen + 1);

        // Unconditional (non-elitist) population replacement.
        pop.individuals = oi;
        pop.fitness = of;
    }

    fn meta(&self) -> ComponentMeta {
        ComponentMeta { kind: "replace/cma-update",
            supported_blocks: SupportedBlocks::Only(vec!["float"]),
            requires: vec![
                StateReq::of::<Vec<f64>>("cma_mean"),
                StateReq::of::<f64>("cma_sigma"),
                StateReq::of::<Vec<Vec<f64>>>("cma_cov"),
                StateReq::of::<Vec<f64>>("cma_ps"),
                StateReq::of::<Vec<f64>>("cma_pc"),
                StateReq::of::<u64>("cma_gen"),
                StateReq::of::<Vec<Vec<f64>>>("cma_eig_b"),
                StateReq::of::<Vec<f64>>("cma_eig_d"),
                StateReq::of::<Vec<Vec<f64>>>("cma_ys"),
            ],
            provides: vec![] }
    }
}

pub fn register(reg: &mut Registry) {
    reg.register_generator("gen/cma", |p| Ok(Box::new(CmaGenerator::from_params(p)?)));
    reg.register_replacer("replace/cma-update", |_| Ok(Box::new(CmaUpdateReplacer)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::{Evaluator, Problem, SphereShifted};
    use sezgi_core::rng::RngStream;
    use sezgi_core::state::Blackboard;

    fn f(x: &[f64]) -> Genotype { Genotype { blocks: vec![BlockValues::Float(x.to_vec())] } }

    #[test]
    fn params_parse_and_validate() {
        let g = CmaGenerator::from_params(&serde_json::json!({"sigma0": 0.5})).unwrap();
        assert_eq!(g.sigma0, Some(0.5));

        let d = CmaGenerator::from_params(&serde_json::json!({})).unwrap();
        assert_eq!(d.sigma0, None);

        assert!(CmaGenerator::from_params(&serde_json::json!({"sigma0": 0.0})).is_err());
        assert!(CmaGenerator::from_params(&serde_json::json!({"sigma0": -1.0})).is_err());
        assert!(CmaGenerator::from_params(&serde_json::json!({"sigma0": "nope"})).is_err());
    }

    #[test]
    #[should_panic(expected = "at least 2")]
    fn small_population_panics() {
        use sezgi_core::problem::Population;

        let p = SphereShifted::new(vec![0.0], -5.0, 5.0);
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 100);
        let mut rng = RngStream::from_master(42, &[]);
        let mut bb = Blackboard::new();
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };

        // Single-individual population (lambda=1) to trigger the panic.
        let pop = Population {
            individuals: vec![Genotype { blocks: vec![BlockValues::Float(vec![0.0])] }],
            fitness: vec![0.0],
        };

        let gen = CmaGenerator::from_params(&serde_json::json!({})).unwrap();
        let _ = gen.generate(&pop, &mut ctx);
    }

    /// Hand-derived from Hansen's 2016 tutorial formulas for d=10, lambda=10
    /// (mu=5): weights via the log-rank recombination formula, then mu_eff,
    /// c_sigma, d_sigma, c_c, c_1, c_mu, and e_norm (`E||N(0,I_d)||`) each
    /// from their closed-form expression in terms of `(d, mu, mu_eff)`.
    #[test]
    fn cma_constants_match_hansen_for_d10_lambda10() {
        let c = cma_constants(10, 10);
        assert_eq!(c.mu, 5);

        let expected_weights = [
            0.45627264690340597,
            0.2707530970017852,
            0.16223111715866978,
            0.08523354710016448,
            0.025509591835974777,
        ];
        assert_eq!(c.weights.len(), 5);
        for (got, want) in c.weights.iter().zip(expected_weights.iter()) {
            assert!((got - want).abs() < 1e-12, "weight got={got} want={want}");
        }
        let sum: f64 = c.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "weights must sum to 1: {sum}");

        assert!((c.mu_eff - 3.1672992814107017).abs() < 1e-12, "mu_eff={}", c.mu_eff);
        assert!((c.c_sigma - 0.28442858794636744).abs() < 1e-12, "c_sigma={}", c.c_sigma);
        assert!((c.d_sigma - 1.2844285879463675).abs() < 1e-12, "d_sigma={}", c.d_sigma);
        assert!((c.c_c - 0.29499038303562225).abs() < 1e-12, "c_c={}", c.c_c);
        assert!((c.c_1 - 0.015283824524751714).abs() < 1e-12, "c_1={}", c.c_1);
        assert!((c.c_mu - 0.02015428276120837).abs() < 1e-12, "c_mu={}", c.c_mu);
        assert!((c.e_norm - 3.0847265651690123).abs() < 1e-12, "e_norm={}", c.e_norm);
    }

    fn ctx_pieces() -> (SphereShifted, Blackboard, RngStream) {
        let p = SphereShifted::new(vec![0.0; 4], -5.0, 5.0);
        (p, Blackboard::new(), RngStream::from_master(42, &[]))
    }

    #[test]
    fn lazy_init_uses_best_individual_and_default_sigma0() {
        let (p, mut bb, mut rng) = ctx_pieces();
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);

        let pop = Population {
            individuals: vec![f(&[1.0, 1.0, 1.0, 1.0]), f(&[0.1, 0.1, 0.1, 0.1]), f(&[2.0, 2.0, 2.0, 2.0])],
            fitness: vec![4.0, 0.04, 16.0], // index 1 is best
        };

        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let gen = CmaGenerator { sigma0: None };
        let offspring = gen.generate(&pop, &mut ctx);

        assert_eq!(offspring.len(), 3);
        assert_eq!(ctx.bb.get::<Vec<f64>>("cma_mean").unwrap(), &vec![0.1, 0.1, 0.1, 0.1]);
        assert_eq!(*ctx.bb.get::<f64>("cma_sigma").unwrap(), 0.3 * 10.0); // 0.3*(hi-lo), hi-lo=10
        assert_eq!(ctx.bb.get::<Vec<Vec<f64>>>("cma_cov").unwrap(), &identity(4));
        assert_eq!(ctx.bb.get::<Vec<f64>>("cma_ps").unwrap(), &vec![0.0; 4]);
        assert_eq!(ctx.bb.get::<Vec<f64>>("cma_pc").unwrap(), &vec![0.0; 4]);
        assert_eq!(*ctx.bb.get::<u64>("cma_gen").unwrap(), 0);
        assert_eq!(ctx.bb.get::<Vec<Vec<f64>>>("cma_eig_b").unwrap().len(), 4);
        assert_eq!(ctx.bb.get::<Vec<f64>>("cma_eig_d").unwrap().len(), 4);
        assert_eq!(ctx.bb.get::<Vec<Vec<f64>>>("cma_ys").unwrap().len(), 3);
        for y in ctx.bb.get::<Vec<Vec<f64>>>("cma_ys").unwrap() { assert_eq!(y.len(), 4); }
    }

    #[test]
    fn replace_moves_mean_toward_best_and_bumps_gen() {
        let (p, mut bb, mut rng) = ctx_pieces();
        let space = p.space();
        let mut evaluator = Evaluator::new(&p, 1000);

        let d = 4usize;
        bb.insert("cma_mean", vec![0.0f64; d]);
        bb.insert("cma_sigma", 1.0f64);
        bb.insert("cma_cov", identity(d));
        bb.insert("cma_ps", vec![0.0f64; d]);
        bb.insert("cma_pc", vec![0.0f64; d]);
        bb.insert("cma_gen", 0u64);
        // With C = I, eig_b = I and eig_d = [1,1,1,1] is a valid (if not the
        // canonical Jacobi-produced) eigendecomposition of the identity.
        bb.insert("cma_eig_b", identity(d));
        bb.insert("cma_eig_d", vec![1.0f64; d]);

        let lambda = 10usize;
        // ys chosen so the best (lowest-fitness) mu=5 offspring all have a
        // clearly positive y in every coordinate: the weighted mean should
        // move mean in the +direction.
        let ys: Vec<Vec<f64>> = (0..lambda).map(|i| vec![(i as f64 + 1.0) * 0.1; d]).collect();
        bb.insert("cma_ys", ys.clone());

        let offspring: Vec<Genotype> = ys.iter().map(|y| f(y)).collect();
        // fitness ascending with index (so sorted order == index order; the
        // mu=5 best are indices 0..5, all with positive small y).
        let fitness: Vec<f64> = (0..lambda).map(|i| i as f64).collect();

        let mut pop = Population { individuals: offspring.clone(), fitness: vec![999.0; lambda] };
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let rep = CmaUpdateReplacer;
        rep.replace(&mut pop, offspring.clone(), fitness.clone(), &mut ctx);

        let mean = ctx.bb.get::<Vec<f64>>("cma_mean").unwrap();
        for &m in mean { assert!(m > 0.0, "mean should move toward the best (positive-y) offspring: {mean:?}"); }
        assert_eq!(*ctx.bb.get::<u64>("cma_gen").unwrap(), 1);

        // Population replaced unconditionally.
        assert_eq!(pop.individuals, offspring);
        assert_eq!(pop.fitness, fitness);

        // C stays symmetric.
        let c = ctx.bb.get::<Vec<Vec<f64>>>("cma_cov").unwrap();
        for (i, row) in c.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!((val - c[j][i]).abs() < 1e-15, "C must be symmetric at [{i}][{j}]");
            }
        }

        let sigma = *ctx.bb.get::<f64>("cma_sigma").unwrap();
        assert!(sigma > 0.0 && sigma.is_finite());
    }

    /// A mean update that would overshoot the space bounds must be clamped
    /// back in, and `ps`/`pc` must be reset to zero when that clamp fires
    /// (the boundary-interaction guard — see `replace`'s doc comment).
    #[test]
    fn replace_clamps_mean_and_resets_paths_on_boundary_violation() {
        let (p, mut bb, mut rng) = ctx_pieces();
        let space = p.space(); // Float { lo: -5.0, hi: 5.0, n: 4 }
        let mut evaluator = Evaluator::new(&p, 1000);

        let d = 4usize;
        bb.insert("cma_mean", vec![4.5f64; d]);
        bb.insert("cma_sigma", 1.0f64);
        bb.insert("cma_cov", identity(d));
        bb.insert("cma_ps", vec![3.0f64; d]); // nonzero "stale" paths
        bb.insert("cma_pc", vec![3.0f64; d]);
        bb.insert("cma_gen", 5u64);
        bb.insert("cma_eig_b", identity(d));
        bb.insert("cma_eig_d", vec![1.0f64; d]);

        let lambda = 10usize;
        // Every y pushes strongly in the +direction: mean (4.5) + 1.0*y will
        // overshoot hi=5.0 for the selected (best) individuals.
        let ys: Vec<Vec<f64>> = (0..lambda).map(|i| vec![(i as f64 + 1.0) * 0.5; d]).collect();
        bb.insert("cma_ys", ys.clone());

        let offspring: Vec<Genotype> = ys.iter().map(|y| f(y)).collect();
        let fitness: Vec<f64> = (0..lambda).map(|i| i as f64).collect();

        let mut pop = Population { individuals: offspring.clone(), fitness: vec![999.0; lambda] };
        let mut ctx = Ctx { space, rng: &mut rng, eval: &mut evaluator, bb: &mut bb, iteration: 0 };
        let rep = CmaUpdateReplacer;
        rep.replace(&mut pop, offspring, fitness, &mut ctx);

        let mean = ctx.bb.get::<Vec<f64>>("cma_mean").unwrap();
        for &m in mean { assert!(m <= 5.0, "mean must be clamped to the space bounds: {mean:?}"); }

        // ps was recomputed fresh this generation (not simply zero — the
        // reset applies to the *carried-over* history before this
        // generation's own contribution is added), so it should differ from
        // the stale pre-clamp value of 3.0 in every coordinate, and in
        // particular should not equal what it would have been had the stale
        // history of 3.0 been kept (i.e. no trace of the old value's scale).
        let ps = ctx.bb.get::<Vec<f64>>("cma_ps").unwrap();
        for &v in ps { assert_ne!(v, 3.0, "ps must not retain the stale pre-clamp value: {ps:?}"); }
    }
}
