//! sezgi-bbob: a reimplementation that follows the BBOB function DEFINITIONS —
//! numerical bit-for-bit agreement with COCO instances is NOT a goal (we use our
//! own RNG/instance model). Known simplifications are marked in the relevant
//! functions' comments with the "sezgi-bbob simplification" tag.

pub mod functions;
pub mod transform;

use std::f64::consts::PI;
use sezgi_core::problem::Problem;
use sezgi_core::rng::RngStream;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

pub const BBOB_SEED_BASE: u64 = 0x5EC1;

#[derive(Debug, Clone)]
pub struct GallagherData {
    pub peaks: Vec<Vec<f64>>,
    pub weights: Vec<f64>,
    pub alphas: Vec<f64>,
}

#[derive(Debug, thiserror::Error)]
pub enum BbobError {
    #[error("fid {0} is not implemented yet")]
    NotImplemented(u32),
    #[error("dim must be >= 2")]
    BadDim,
    /// Returned by [`BbobProblem::recentered`] for a fid whose objective
    /// formula consumes `x_opt` directly beyond a coordinate shift (see
    /// [`BbobProblem::is_translation_invariant`]) -- forcing `x_opt` to the
    /// domain center for such a fid would change the landscape's SHAPE,
    /// not just relocate its optimum, so `recentered` refuses rather than
    /// silently producing a wrong landscape.
    #[error(
        "fid {0} is not translation-invariant: its formula consumes x_opt directly \
         (beyond a coordinate shift), so BbobProblem::recentered cannot move its optimum \
         to the domain center without changing the landscape's shape"
    )]
    NotTranslationInvariant(u32),
}

#[derive(Debug, Clone, Copy)]
enum XOptPolicy {
    Uniform44,
    BoundaryPm5,
    ScaledPattern { scale: f64 },
}

fn needs_r(fid: u32) -> bool {
    !matches!(fid, 1 | 4 | 5 | 20)
}

fn needs_q(fid: u32) -> bool {
    matches!(fid, 6 | 7 | 13 | 15 | 16 | 17 | 18 | 23 | 24)
}

fn x_opt_policy(fid: u32) -> XOptPolicy {
    match fid {
        5 => XOptPolicy::BoundaryPm5,
        20 => XOptPolicy::ScaledPattern { scale: 4.2096874633 / 2.0 },
        24 => XOptPolicy::ScaledPattern { scale: 2.5 / 2.0 },
        _ => XOptPolicy::Uniform44,
    }
}

fn shift(xs: &[f64], x_opt: &[f64]) -> Vec<f64> {
    xs.iter().zip(x_opt).map(|(x, o)| x - o).collect()
}

/// z = Q·Λ^10·R·(x - x_opt) — the f6/f13-style pipeline
fn apply2(p: &BbobProblem, xs: &[f64]) -> Vec<f64> {
    let s = shift(xs, &p.x_opt);
    let r = transform::apply(p.rot.as_ref().unwrap(), &s);
    let l = transform::lambda_alpha(&r, 10.0);
    transform::apply(p.rot2.as_ref().unwrap(), &l)
}

pub struct BbobProblem {
    fid: u32,
    x_opt: Vec<f64>,
    f_opt: f64,
    rot: Option<Vec<Vec<f64>>>,
    rot2: Option<Vec<Vec<f64>>>,
    gallagher: Option<GallagherData>,
    space: SearchSpace,
    pub instance: u32,
}

impl BbobProblem {
    pub fn new(fid: u32, dim: usize, instance: u32) -> Result<Self, BbobError> {
        if dim < 2 { return Err(BbobError::BadDim); }
        if !(1..=24).contains(&fid) { return Err(BbobError::NotImplemented(fid)); }

        let mut rng = RngStream::from_master(BBOB_SEED_BASE + fid as u64,
                                             &[instance as u64]);

        // Drawing order (CRITICAL): [x_opt d draws][f_opt 1 draw][R seed if needed][Q seed if needed]
        let x_opt: Vec<f64> = match x_opt_policy(fid) {
            XOptPolicy::Uniform44 =>
                (0..dim).map(|_| -4.0 + 8.0 * rng.next_f64()).collect(),
            XOptPolicy::BoundaryPm5 =>
                (0..dim).map(|_| if rng.next_f64() < 0.5 { -5.0 } else { 5.0 }).collect(),
            XOptPolicy::ScaledPattern { scale } =>
                (0..dim).map(|_| if rng.next_f64() < 0.5 { -scale } else { scale }).collect(),
        };
        let f_opt = -200.0 + 400.0 * rng.next_f64();
        let rot = needs_r(fid).then(|| transform::rotation_matrix(dim, rng.next_u64()));
        let rot2 = needs_q(fid).then(|| transform::rotation_matrix(dim, rng.next_u64()));

        // Gallagher peak data (for fid 21/22)
        // sezgi-bbob simplification: alphas are drawn as 1000^(2u) at random instead of a permuted fixed set; C matrices are diagonal (Λ^alpha), no per-peak rotation.
        let gallagher = matches!(fid, 21 | 22).then(|| {
            let p = if fid == 21 { 101usize } else { 21 };
            let mut peaks = Vec::with_capacity(p);
            peaks.push(x_opt.clone());
            for _ in 1..p {
                peaks.push((0..dim).map(|_| -4.9 + 9.8 * rng.next_f64()).collect());
            }
            let weights: Vec<f64> = (0..p).map(|i| if i == 0 { 10.0 }
                else { 1.1 + 8.0 * (i as f64 - 1.0) / (p as f64 - 2.0) }).collect();
            let alphas: Vec<f64> = (0..p).map(|i| if i == 0 { 1000.0 }
                else { 1000f64.powf(2.0 * rng.next_f64()) }).collect();
            GallagherData { peaks, weights, alphas }
        });

        let space = SearchSpace::new(vec![Block::Float { lo: -5.0, hi: 5.0, n: dim }])
            .expect("fixed bounds are valid");

        Ok(Self { fid, x_opt, f_opt, rot, rot2, gallagher, space, instance })
    }

    pub fn x_opt(&self) -> &[f64] { &self.x_opt }
    pub fn f_opt(&self) -> f64 { self.f_opt }
    pub fn fid(&self) -> u32 { self.fid }

    /// `true` iff `fid`'s objective formula uses `x_opt` ONLY as the origin
    /// of a coordinate shift (`shift(xs, x_opt) = xs - x_opt`, feeding an
    /// otherwise `x_opt`-independent core function, with any boundary
    /// penalty `f_pen` computed on the RAW, untranslated `xs`) -- i.e.
    /// whether [`BbobProblem::recentered`] can move `x_opt` to the domain
    /// center while leaving every other aspect of the landscape (shape,
    /// conditioning, rotation) untouched.
    ///
    /// `false` for fids 5 (`LinearSlope`), 6 (`AttractiveSector`), 20
    /// (`Schwefel`), and 24 (`LunacekBiRastrigin`): each consumes `x_opt`
    /// DIRECTLY inside its core formula beyond a shift --
    /// `functions::linear_slope`'s per-axis slope sign `oi.signum()`,
    /// `functions::attractive_sector`'s per-axis sector selection
    /// `zi * oi > 0.0`, and fid 20/24's own `o.signum()`-driven
    /// `xhat`/asymmetry construction (this module's `evaluate_batch`).
    /// Forcing `x_opt` to the domain center (all zeros) would zero every
    /// one of those per-dimension signs uniformly (`0.0_f64.signum() ==
    /// 1.0`), collapsing the intended random per-dimension asymmetry into a
    /// degenerate, structurally SIMPLER landscape -- an actual change of
    /// shape, not a relocation of the same one.
    pub fn is_translation_invariant(fid: u32) -> bool {
        !matches!(fid, 5 | 6 | 20 | 24)
    }

    /// Returns a variant of this instance with `x_opt` -- and, for fid
    /// 21/22, the Gallagher data's own coincident first peak -- FORCED to
    /// the exact center of the declared domain, instead of wherever this
    /// instance's own construction drew it. `f_opt`, every rotation
    /// matrix, and (for 21/22) every OTHER Gallagher peak/weight/alpha are
    /// UNCHANGED: only the optimum's location moves.
    ///
    /// This is the NATIVE "centered" condition for a central-bias scan
    /// (`sezgi_bias::central`): unlike a wrapper that translates evaluated
    /// POINTS to relocate the apparent optimum, this translates the
    /// INSTANCE's own stored optimum once, up front -- so every
    /// subsequent `evaluate_batch` call runs on the caller's own,
    /// untranslated coordinates, and any boundary penalty (`f_pen`, used
    /// by roughly a third of the 24 fids) is computed on the SAME
    /// coordinates the caller actually queried, never on a translated
    /// (and potentially out-of-declared-domain) point.
    ///
    /// # Errors
    /// [`BbobError::NotTranslationInvariant`] if
    /// [`BbobProblem::is_translation_invariant`] is `false` for this
    /// instance's `fid` -- see that method's doc for exactly which fids
    /// and why recentering them is refused rather than silently producing
    /// a different-shaped landscape.
    pub fn recentered(mut self) -> Result<Self, BbobError> {
        if !Self::is_translation_invariant(self.fid) {
            return Err(BbobError::NotTranslationInvariant(self.fid));
        }
        let center = match &self.space.blocks()[0] {
            Block::Float { lo, hi, .. } => (lo + hi) / 2.0,
            other => unreachable!(
                "BbobProblem's space is always a single Float block by construction, got {other:?}"
            ),
        };
        self.x_opt = vec![center; self.x_opt.len()];
        if let Some(gd) = self.gallagher.as_mut() {
            // fid 21/22's first peak IS x_opt by construction (`peaks.push(x_opt.clone())`
            // in `BbobProblem::new`); keep that invariant after recentering.
            gd.peaks[0] = self.x_opt.clone();
        }
        Ok(self)
    }
    pub fn name(&self) -> &'static str {
        match self.fid { 1 => "Sphere", 2 => "Ellipsoidal", 3 => "Rastrigin", 4 => "BucheRastrigin",
                         5 => "LinearSlope", 6 => "AttractiveSector", 7 => "StepEllipsoidal", 8 => "Rosenbrock", 9 => "RosenbrockRotated",
                         10 => "EllipsoidalRotated", 11 => "Discus", 12 => "BentCigar", 13 => "SharpRidge", 14 => "DifferentPowers",
                         15 => "RastriginRotated", 16 => "Weierstrass", 17 => "SchaffersF7", 18 => "SchaffersF7Ill", 19 => "GriewankRosenbrock",
                         20 => "Schwefel", 21 => "Gallagher101", 22 => "Gallagher21", 23 => "Katsuura", 24 => "LunacekBiRastrigin", _ => unreachable!() }
    }
}

impl Problem for BbobProblem {
    fn space(&self) -> &SearchSpace { &self.space }
    fn optimum(&self) -> Option<f64> { Some(self.f_opt) }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
        pop.iter().map(|g| {
            let BlockValues::Float(xs) = &g.blocks[0] else { return f64::INFINITY };
            let raw = match self.fid {
                1 | 2 | 3 | 8 => {
                    let shifted: Vec<f64> = xs.iter().zip(&self.x_opt).map(|(x, o)| x - o).collect();
                    let z = match &self.rot {
                        Some(r) => transform::apply(r, &shifted),
                        None => shifted,
                    };
                    match self.fid {
                        1 => functions::sphere(&z),
                        2 => functions::ellipsoidal(&z),
                        3 => functions::rastrigin(&z),
                        8 => {
                            // z=0 at the optimum → shift to w=1
                            let w: Vec<f64> = z.iter().map(|v| v + 1.0).collect();
                            functions::rosenbrock(&w)
                        }
                        _ => unreachable!(),
                    }
                }
                4 => {
                    let shifted: Vec<f64> = xs.iter().zip(&self.x_opt).map(|(x, o)| x - o).collect();
                    functions::buche_rastrigin(&transform::t_osz(&shifted)) + 100.0 * transform::f_pen(xs)
                }
                5 => functions::linear_slope(xs, &self.x_opt),
                6 => {
                    let z = apply2(self, xs);
                    functions::attractive_sector(&z, &self.x_opt)
                }
                7 => {
                    let shifted = shift(xs, &self.x_opt);
                    let zhat = transform::lambda_alpha(&transform::apply(self.rot.as_ref().unwrap(), &shifted), 10.0);
                    let ztilde: Vec<f64> = zhat.iter().map(|&v| {
                        if v.abs() > 0.5 { (0.5 + v).floor() } else { (0.5 + 10.0 * v).floor() / 10.0 }
                    }).collect();
                    let z = transform::apply(self.rot2.as_ref().unwrap(), &ztilde);
                    functions::step_ellipsoidal(zhat[0], &z) + transform::f_pen(xs)
                }
                9 => {
                    let shifted = shift(xs, &self.x_opt);
                    let scale = 1f64.max((xs.len() as f64).sqrt() / 8.0);
                    let z: Vec<f64> = transform::apply(self.rot.as_ref().unwrap(), &shifted)
                        .iter().map(|v| scale * v + 1.0).collect();
                    functions::rosenbrock(&z)
                }
                10 => {
                    let z = transform::t_osz(&transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt)));
                    functions::ellipsoidal(&z)
                }
                11 => {
                    let z = transform::t_osz(&transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt)));
                    functions::discus(&z)
                }
                12 => {
                    // sezgi-bbob simplification: same R applied twice instead of the canonical second independent rotation.
                    let r1 = transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt));
                    let z = transform::apply(self.rot.as_ref().unwrap(), &transform::t_asy(&r1, 0.5));
                    functions::bent_cigar(&z)
                }
                13 => functions::sharp_ridge(&apply2(self, xs)),
                14 => functions::different_powers(&transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt))),
                15 => {
                    let r1 = transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt));
                    let a = transform::t_asy(&transform::t_osz(&r1), 0.2);
                    let q = transform::apply(self.rot2.as_ref().unwrap(), &a);
                    let z = transform::apply(self.rot.as_ref().unwrap(), &transform::lambda_alpha(&q, 10.0));
                    functions::rastrigin(&z)
                }
                16 => {
                    let r1 = transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt));
                    let q = transform::apply(self.rot2.as_ref().unwrap(), &transform::t_osz(&r1));
                    let z = transform::apply(self.rot.as_ref().unwrap(), &transform::lambda_alpha(&q, 0.01));
                    functions::weierstrass(&z) + 10.0 / xs.len() as f64 * transform::f_pen(xs)
                }
                17 | 18 => {
                    let alpha = if self.fid == 17 { 10.0 } else { 1000.0 };
                    let r1 = transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt));
                    let q = transform::apply(self.rot2.as_ref().unwrap(), &transform::t_asy(&r1, 0.5));
                    let z = transform::lambda_alpha(&q, alpha);
                    functions::schaffers_f7(&z) + 10.0 * transform::f_pen(xs)
                }
                19 => {
                    let scale = 1f64.max((xs.len() as f64).sqrt() / 8.0);
                    let z: Vec<f64> = transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt))
                        .iter().map(|v| scale * v + 1.0).collect();
                    functions::griewank_rosenbrock(&z)
                }
                20 => {
                    // sezgi-bbob simplification: a variant formula of the canonical BBOB Schwefel; Λ^10 diagonal scaling, no rotation.
                    let d = xs.len();
                    let xhat: Vec<f64> = xs.iter().zip(&self.x_opt).map(|(x, o)| 2.0 * o.signum() * x).collect();
                    let two_abs: Vec<f64> = self.x_opt.iter().map(|o| 2.0 * o.abs()).collect();
                    let mut zhat = xhat.clone();
                    for i in 1..d { zhat[i] = xhat[i] + 0.25 * (xhat[i - 1] - two_abs[i - 1]); }
                    let inner: Vec<f64> = zhat.iter().zip(&two_abs).map(|(z, t)| z - t).collect();
                    let z: Vec<f64> = transform::lambda_alpha(&inner, 10.0).iter()
                        .zip(&two_abs).map(|(l, t)| 100.0 * (l + t)).collect();
                    let s: f64 = z.iter().map(|&zi| zi * (zi.abs().sqrt()).sin()).sum();
                    let z100: Vec<f64> = z.iter().map(|v| v / 100.0).collect();
                    -s / (100.0 * d as f64) + 4.189828872724339 + 100.0 * transform::f_pen(&z100)
                }
                21 | 22 => {
                    let gd = self.gallagher.as_ref().unwrap();
                    let d = xs.len() as f64;
                    let best: f64 = gd.peaks.iter().zip(&gd.weights).zip(&gd.alphas).map(|((y, &w), &a)| {
                        let diff = shift(xs, y);
                        let rd = transform::apply(self.rot.as_ref().unwrap(), &diff);
                        let c = transform::lambda_alpha(&rd, a);
                        let q: f64 = c.iter().zip(&rd).map(|(ci, ri)| ci * ri).sum::<f64>() / a.powf(0.25);
                        w * (-q / (2.0 * d)).exp()
                    }).fold(f64::NEG_INFINITY, f64::max);
                    transform::t_osz(&[10.0 - best])[0].powi(2) + transform::f_pen(xs)
                }
                23 => {
                    let r1 = transform::apply(self.rot.as_ref().unwrap(), &shift(xs, &self.x_opt));
                    let z = transform::apply(self.rot2.as_ref().unwrap(), &transform::lambda_alpha(&r1, 100.0));
                    let d = xs.len() as f64;
                    let prod: f64 = z.iter().enumerate().map(|(i, &zi)| {
                        let s: f64 = (1..=32).map(|j| {
                            let t = 2f64.powi(j) * zi;
                            (t - t.round()).abs() / 2f64.powi(j)
                        }).sum();
                        (1.0 + (i as f64 + 1.0) * s).powf(10.0 / d.powf(1.2))
                    }).product();
                    10.0 / (d * d) * prod - 10.0 / (d * d) + transform::f_pen(xs)
                }
                24 => {
                    let d = xs.len() as f64;
                    let mu0 = 2.5f64;
                    let s = 1.0 - 1.0 / (2.0 * (d + 20.0).sqrt() - 8.2);
                    let mu1 = -((mu0 * mu0 - 1.0) / s).sqrt();
                    let xhat: Vec<f64> = xs.iter().zip(&self.x_opt).map(|(x, o)| 2.0 * o.signum() * x).collect();
                    let inner: Vec<f64> = xhat.iter().map(|v| v - mu0).collect();
                    let r1 = transform::apply(self.rot.as_ref().unwrap(), &inner);
                    let z = transform::apply(self.rot2.as_ref().unwrap(), &transform::lambda_alpha(&r1, 100.0));
                    let sum0: f64 = xhat.iter().map(|v| (v - mu0).powi(2)).sum();
                    let sum1: f64 = xhat.iter().map(|v| (v - mu1).powi(2)).sum();
                    let cos_sum: f64 = z.iter().map(|v| (2.0 * PI * v).cos()).sum();
                    sum0.min(d + s * sum1) + 10.0 * (d - cos_sum) + 1e4 * transform::f_pen(xs)
                }
                _ => unreachable!(),
            };
            raw + self.f_opt
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::Problem;
    use sezgi_core::space::{BlockValues, Genotype};

    fn g(xs: Vec<f64>) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs)] } }

    #[test]
    fn optimum_is_attained_at_x_opt() {
        for fid in [1u32, 2, 3, 8] {
            let p = BbobProblem::new(fid, 5, 1).unwrap();
            let f = p.evaluate_batch(&[g(p.x_opt().to_vec())])[0];
            assert!((f - p.f_opt()).abs() < 1e-9, "fid={fid}: {f} != {}", p.f_opt());
        }
    }

    #[test]
    fn x_opt_not_at_center() {
        let p = BbobProblem::new(1, 10, 1).unwrap();
        assert!(p.x_opt().iter().any(|&x| x.abs() > 1e-3), "the optimum should not be at the center");
    }

    #[test]
    fn instances_differ_and_are_deterministic() {
        let a1 = BbobProblem::new(3, 5, 1).unwrap();
        let a2 = BbobProblem::new(3, 5, 1).unwrap();
        let b = BbobProblem::new(3, 5, 2).unwrap();
        assert_eq!(a1.x_opt(), a2.x_opt());
        assert_ne!(a1.x_opt(), b.x_opt());
    }

    #[test]
    fn away_from_optimum_is_worse() {
        let p = BbobProblem::new(2, 5, 3).unwrap();
        let mut x = p.x_opt().to_vec();
        x[0] += 1.0;
        assert!(p.evaluate_batch(&[g(x)])[0] > p.f_opt());
    }

    #[test]
    fn unknown_fid_is_error() {
        assert!(matches!(BbobProblem::new(99, 5, 1), Err(BbobError::NotImplemented(99))));
    }

    #[test]
    fn rotation_is_orthogonal() {
        let r = crate::bbob::transform::rotation_matrix(6, 12345);
        for i in 0..6 {
            for j in 0..6 {
                let dot: f64 = (0..6).map(|k| r[k][i] * r[k][j]).sum();
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn m1_fids_instance_values_pinned() {
        // These values are taken from the existing code and pinned on the FIRST run of
        // this test (same PIN-ME procedure as T17/M1): the f value's bits at
        // instance=1, dim=5, x=[0.5,-1.0,2.0,0.0,-3.0] for each fid.
        let probe = g(vec![0.5, -1.0, 2.0, 0.0, -3.0]);
        for (fid, expected_hex) in [(1u32, "c05a1069cca05d30"), (2, "414d7d483f2ad056"), (3, "c03b12604f3ed308"), (8, "40d41e5080f1a8d2")] {
            let p = BbobProblem::new(fid, 5, 1).unwrap();
            let f = p.evaluate_batch(std::slice::from_ref(&probe))[0];
            let got = format!("{:016x}", f.to_bits());
            assert_eq!(got, expected_hex, "fid {fid}: M1 behavior drifted!");
        }
    }

    #[test]
    fn all_fids_instance_values_pinned() {
        // All 24 BBOB functions: instance=1, dim=5, probe x=[0.5,-1.0,2.0,0.0,-3.0].
        // These hex values freeze the instance stream against future refactors.
        let probe = g(vec![0.5, -1.0, 2.0, 0.0, -3.0]);
        let expected = [
            // fid 1..=24
            "c05a1069cca05d30", "414d7d483f2ad056", "c03b12604f3ed308", "40a545a7309c9755",
            "40607335c8aef256", "40f743ba7e8bb236", "40746d5f43627b37", "40d41e5080f1a8d2",
            "4103e765d446c8e1", "4173a9297a3b0679", "40ac2b30cfdd105a", "416377d9b7b67eea",
            "4090c3e12c344d12", "406310739d351b0c", "407d4ee8344261cd", "c03a392ffc338080",
            "406c208e87ed6518", "406f6f65180551a1", "40662f4941524515", "40f1145c37c4c1f8",
            "405a384b7b1d61ee", "4056a1a3130da00e", "c057f3b723d68878", "40670626da36ea4c",
        ];
        for fid in 1u32..=24 {
            let p = BbobProblem::new(fid, 5, 1).unwrap();
            let f = p.evaluate_batch(std::slice::from_ref(&probe))[0];
            let got = format!("{:016x}", f.to_bits());
            assert_eq!(got, expected[(fid - 1) as usize], "fid {fid}: value drifted!");
        }
    }

    #[test]
    fn f4_f5_optimum_attained_and_offcenter_worse() {
        for fid in [4u32, 5] {
            let p = BbobProblem::new(fid, 5, 1).unwrap();
            let at_opt = p.evaluate_batch(&[g(p.x_opt().to_vec())])[0];
            assert!((at_opt - p.f_opt()).abs() < 1e-6, "fid {fid}: {at_opt} != {}", p.f_opt());
            let mut x = p.x_opt().to_vec();
            x[0] = (x[0] - 0.7).clamp(-5.0, 5.0); // for f5, push inward from the boundary
            if x == p.x_opt() { x[0] += 0.7; }
            assert!(p.evaluate_batch(&[g(x)])[0] > p.f_opt(), "fid {fid}: off-optimum came out better");
        }
    }

    #[test]
    fn f5_optimum_on_boundary() {
        let p = BbobProblem::new(5, 6, 2).unwrap();
        assert!(p.x_opt().iter().all(|&x| x == 5.0 || x == -5.0));
    }

    #[test]
    fn f6_f7_f9_optimum_and_determinism() {
        for fid in [6u32, 7, 9] {
            let p = BbobProblem::new(fid, 5, 1).unwrap();
            let at_opt = p.evaluate_batch(&[g(p.x_opt().to_vec())])[0];
            assert!((at_opt - p.f_opt()).abs() < 1e-6, "fid {fid}: {}", at_opt - p.f_opt());
            let mut x = p.x_opt().to_vec();
            x[1] += 1.0;
            assert!(p.evaluate_batch(&[g(x)])[0] > p.f_opt(), "fid {fid}");
            let p2 = BbobProblem::new(fid, 5, 1).unwrap();
            assert_eq!(p.x_opt(), p2.x_opt(), "fid {fid} instance determinism");
        }
    }

    #[test]
    fn f7_has_plateaus() {
        // Step-ellipsoidal: two nearby points far enough from the optimum should give the same f
        let p = BbobProblem::new(7, 5, 1).unwrap();
        let mut a = p.x_opt().to_vec(); a[0] += 2.0;
        let mut b = a.clone(); b[0] += 1e-4;
        let fa = p.evaluate_batch(&[g(a)])[0];
        let fb = p.evaluate_batch(&[g(b)])[0];
        assert_eq!(fa, fb, "expected a step plateau");
    }

    #[test]
    fn f10_to_f14_optimum_and_conditioning() {
        for fid in 10u32..=14 {
            let p = BbobProblem::new(fid, 5, 3).unwrap();
            let at_opt = p.evaluate_batch(&[g(p.x_opt().to_vec())])[0];
            assert!((at_opt - p.f_opt()).abs() < 1e-6, "fid {fid}: {}", at_opt - p.f_opt());
            let mut x = p.x_opt().to_vec();
            x[2] -= 0.5;
            assert!(p.evaluate_batch(&[g(x)])[0] > p.f_opt(), "fid {fid}");
        }
    }

    #[test]
    fn discus_and_cigar_axis_asymmetry() {
        // pure core test: discus has the first axis 1e6x heavier, cigar is the reverse
        assert!(functions::discus(&[1.0, 0.0]) > functions::discus(&[0.0, 1.0]) * 1e5);
        assert!(functions::bent_cigar(&[0.0, 1.0]) > functions::bent_cigar(&[1.0, 0.0]) * 1e5);
    }

    #[test]
    fn f15_to_f19_optimum_attained() {
        for fid in 15u32..=19 {
            let p = BbobProblem::new(fid, 5, 1).unwrap();
            let at_opt = p.evaluate_batch(&[g(p.x_opt().to_vec())])[0];
            assert!((at_opt - p.f_opt()).abs() < 1e-6, "fid {fid}: diff {}", at_opt - p.f_opt());
            let mut x = p.x_opt().to_vec();
            x[0] += 0.9; x[3] -= 0.4;
            assert!(p.evaluate_batch(&[g(x)])[0] > p.f_opt(), "fid {fid}");
        }
    }

    #[test]
    fn weierstrass_core_zero_at_origin() {
        assert!(functions::weierstrass(&[0.0; 5]).abs() < 1e-9);
    }

    #[test]
    fn schaffers_core_zero_at_origin_positive_elsewhere() {
        assert!(functions::schaffers_f7(&[0.0; 4]).abs() < 1e-12);
        assert!(functions::schaffers_f7(&[1.0, -2.0, 0.5, 3.0]) > 0.0);
    }

    #[test]
    fn f20_to_f24_optimum_attained() {
        for fid in 20u32..=24 {
            let p = BbobProblem::new(fid, 5, 1).unwrap();
            let at_opt = p.evaluate_batch(&[g(p.x_opt().to_vec())])[0];
            assert!((at_opt - p.f_opt()).abs() < 1e-6, "fid {fid}: diff {}", at_opt - p.f_opt());

            // Off-optimum probe: perturb x[0] by ±0.5 and verify worse value
            let mut x_probe = p.x_opt().to_vec();
            x_probe[0] = (x_probe[0] + 0.5).clamp(-5.0, 5.0);
            if x_probe[0] == p.x_opt()[0] {
                x_probe[0] = (x_probe[0] - 0.5).clamp(-5.0, 5.0);
            }
            let f_probe = p.evaluate_batch(&[g(x_probe)])[0];
            assert!(f_probe > p.f_opt(), "fid {fid}: off-optimum ({f_probe}) should be worse than optimum ({}, f_opt={})", f_probe, p.f_opt());
        }
    }

    #[test]
    fn gallagher_first_peak_dominates() {
        let p = BbobProblem::new(21, 5, 1).unwrap();
        // far from the optimum the value should approach f_opt+10² (a rough bound without t_osz) but stay below it
        let far = g(vec![4.9; 5]);
        let f_far = p.evaluate_batch(&[far])[0];
        assert!(f_far > p.f_opt());
    }

    // ---- BbobProblem::recentered / is_translation_invariant (review fix) ----

    #[test]
    fn is_translation_invariant_excludes_expected_fids() {
        for fid in 1u32..=24 {
            let expected = !matches!(fid, 5 | 6 | 20 | 24);
            assert_eq!(BbobProblem::is_translation_invariant(fid), expected, "fid {fid}");
        }
    }

    #[test]
    fn recentered_rejects_non_translation_invariant_fids() {
        for fid in [5u32, 6, 20, 24] {
            let p = BbobProblem::new(fid, 5, 1).unwrap();
            match p.recentered() {
                Ok(_) => panic!("fid {fid}: expected NotTranslationInvariant error, got Ok"),
                Err(BbobError::NotTranslationInvariant(f)) => assert_eq!(f, fid),
                Err(_) => panic!("fid {fid}: expected NotTranslationInvariant, got a different BbobError"),
            }
        }
    }

    #[test]
    fn recentered_moves_optimum_to_domain_center_exactly() {
        for fid in [1u32, 4, 8, 21] {
            let p = BbobProblem::new(fid, 5, 3).unwrap().recentered().expect("translation-invariant fid");
            let center: Vec<f64> = match &p.space().blocks()[0] {
                Block::Float { lo, hi, n } => vec![(lo + hi) / 2.0; *n],
                other => panic!("expected a single Float block, got {other:?}"),
            };
            let f = p.evaluate_batch(&[g(center)])[0];
            assert!((f - p.f_opt()).abs() < 1e-6, "fid {fid}: {f} != {}", p.f_opt());
        }
    }

    // Proves `recentered` is a PURE relocation of the optimum for
    // translation-invariant fids, AWAY FROM THE PENALTY REGION: evaluating
    // `x_opt_orig + delta` on the original (shifted) instance must equal
    // evaluating `center + delta` on its `.recentered()` counterpart, for
    // the SAME instance (same fid, dim, instance number -- so identical
    // rotation/Gallagher data), for every fid category this fix touches
    // (plain shift: 1; f_pen fids: 4, 16, 23; Gallagher: 21).
    //
    // Scope note (does NOT overclaim landscape-wide equality): for the
    // f_pen-bearing fids (4, 16, 23), `recentered` deliberately does NOT
    // translate the boundary-penalty envelope `f_pen` -- that asymmetry is
    // the whole point of the `recentered` fix (see this module's own doc,
    // `is_translation_invariant`/`recentered`, and `central.rs`'s "why
    // `central_bias_scan` no longer wraps evaluated points" section). This
    // fixture's `delta` values are small enough that every probed point
    // stays in the penalty-free interior (`f_pen = 0`) on BOTH sides, so the
    // test only proves the CORE landscape (the part `f_pen` doesn't touch)
    // is a pure translation -- it does not, and cannot, prove `f_pen` itself
    // translates (it does not, by design).
    #[test]
    fn recentered_is_a_pure_translation_of_the_core_landscape_away_from_the_penalty_region() {
        let dim = 5;
        let delta = [0.3, -0.2, 0.1, 0.0, -0.15];
        for fid in [1u32, 4, 16, 21, 23] {
            let original = BbobProblem::new(fid, dim, 2).unwrap();
            let center = match &original.space().blocks()[0] {
                Block::Float { lo, hi, .. } => (lo + hi) / 2.0,
                other => panic!("expected a single Float block, got {other:?}"),
            };
            let x_opt_orig = original.x_opt().to_vec();
            let recentered = BbobProblem::new(fid, dim, 2).unwrap().recentered().expect("translation-invariant fid");

            let x_shifted: Vec<f64> = x_opt_orig.iter().zip(&delta).map(|(o, d)| o + d).collect();
            let x_centered: Vec<f64> = delta.iter().map(|&d| center + d).collect();

            let f_shifted = original.evaluate_batch(&[g(x_shifted)])[0];
            let f_centered = recentered.evaluate_batch(&[g(x_centered)])[0];
            assert!(
                (f_shifted - f_centered).abs() < 1e-9,
                "fid {fid}: recentered must be a pure translation of the core landscape (away \
                 from the penalty region), got f_shifted={f_shifted} vs f_centered={f_centered}"
            );
        }
    }

    #[test]
    fn recentered_preserves_f_opt() {
        for fid in [1u32, 4, 10, 21] {
            let original = BbobProblem::new(fid, 5, 4).unwrap();
            let f_opt = original.f_opt();
            let recentered = original.recentered().expect("translation-invariant fid");
            assert_eq!(recentered.f_opt(), f_opt, "fid {fid}");
        }
    }
}
