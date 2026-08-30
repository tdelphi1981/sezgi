//! The CEC 2022 Special Session and Competition benchmark suite (basic
//! functions f1-f5; fid 6-12, the hybrid/composition functions, are
//! DEFERRED to T6/T7 -- see `data.rs`'s "Extension points" doc).
//!
//! Source (PROVENANCE, fetched and read directly, not from memory):
//! Abhishek Kumar, Kenneth V. Price, Ali Wagdy Mohamed, Anas A. Hadi, P. N.
//! Suganthan, "Problem Definitions and Evaluation Criteria for the CEC 2022
//! Special Session and Competition on Single Objective Bound Constrained
//! Numerical Optimization", Technical Report, December 2021. PDF fetched
//! directly from the authors' own official repository:
//! `https://raw.githubusercontent.com/P-N-Suganthan/2022-SO-BO/main/CEC2022%20TR.pdf`
//! (20 pages; every quote below is transcribed from that fetched PDF via
//! `pdftotext`, cross-checked with and without `-layout`). The reference C
//! implementation and vendored data were fetched from the SAME repository's
//! `CEC2022.zip` -> `CEC2022/C-Code/{cec22_test_func.cpp, input_data/}`
//! (`https://raw.githubusercontent.com/P-N-Suganthan/2022-SO-BO/main/CEC2022.zip`).
//!
//! ## Licensing finding (milestone scope ruling 3)
//!
//! The repository root (`https://api.github.com/repos/P-N-Suganthan/2022-SO-BO/contents/`,
//! fetched directly) lists exactly five files -- `2022-Full-Download.zip`,
//! `CEC2022 TR.pdf`, `CEC2022.zip`, `Python-CEC2022.zip`, `R-package.txt` --
//! and NONE of `LICENSE`, `LICENSE.md`, `LICENSE.txt` exist at the repo root
//! (each probed directly, all HTTP 404); no license file exists anywhere
//! inside the extracted `CEC2022.zip` either (`find ... -iname '*licen*'`
//! found nothing). This confirms the scope ruling's premise: the repo
//! carries no explicit license. Per that ruling, the files below are
//! vendored with prominent attribution (this doc) rather than withheld.
//!
//! ## 1.2. Summary table (quoted; note the report's OWN header typo, "CEC'21
//! Test Suite", carried over verbatim from a prior year's report into this
//! CEC'22 document -- reproduced here exactly as printed, not corrected)
//!
//! | No. | Functions | `F_i*` |
//! |---|---|---|
//! | 1 | Shifted and full Rotated Zakharov Function | 300 |
//! | 2 | Shifted and full Rotated Rosenbrock's Function | 400 |
//! | 3 | Shifted and full Rotated Expanded Schaffer's f6 Function | 600 |
//! | 4 | Shifted and full Rotated Non-Continuous Rastrigin's Function | 800 |
//! | 5 | Shifted and full Rotated Levy Function | 900 |
//! | 6 | Hybrid Function 1 (N = 3) | 1800 |
//! | 7 | Hybrid Function 2 (N = 6) | 2000 |
//! | 8 | Hybrid Function 3 (N = 5) | 2200 |
//! | 9 | Composition Function 1 (N = 5) | 2300 |
//! | 10 | Composition Function 2 (N = 4) | 2400 |
//! | 11 | Composition Function 3 (N = 5) | 2600 |
//! | 12 | Composition Function 4 (N = 6) | 2700 |
//!
//! "Search range: `[-100,100]^D`." confirms the brief's sketch domain. This
//! module implements fid 1-5 only, VERIFIED against the C reference's
//! dispatch (`cec22_test_func.cpp`'s `switch(func_num)`, quoted below) --
//! `f2 += 400.0` for `rosenbrock_func` and `f4 += 800.0` for
//! `step_rastrigin_func` confirms both the sketch's `F*` values AND that fid
//! 4 dispatches to `step_rastrigin_func` ("non-continuous", per its own
//! name), NOT plain `rastrigin_func` -- see the F4 discrepancy note below
//! for what "non-continuous" actually computes once the reference bug is
//! accounted for.
//!
//! ## Supported dims -- {2,10,20}, and a verified dim=2 GAP for fid 6/7/8
//!
//! `cec22_test_func.cpp`'s initializer, quoted verbatim:
//! ```text
//! if (!(nx==2||nx==10||nx==20))
//!     printf("\nError: Test functions are only defined for D=2,10,20.\n");
//! if (nx==2&&(func_num==6||func_num==7||func_num==8))
//!     printf("\nError:  NOT defined for D=2.\n");
//! ```
//! Confirmed independently by the vendored `input_data/` file listing
//! itself: `M_7_D2.txt` (and its `shift`/`shuffle` counterparts for fid 6-8)
//! is simply ABSENT from the archive, while `M_1_D2.txt` through
//! `M_5_D2.txt` and `M_9_D2.txt` through `M_12_D2.txt` are all present. Not
//! relevant to fid 1-5 (this task: all five ship complete dim=2 data, see
//! `data.rs`'s data-integrity tests), but recorded here for T6/T7, which
//! will need `Cec2022::new` to reject `(fid in 6..=8, dim == 2)` explicitly
//! rather than fail obscurely on a missing embed.
//!
//! ## 1.3. Definitions of the Basic Functions (quoted, section numbers as
//! printed) -- the UNSHIFTED, UNROTATED core `f_i`, before section 1.4's
//! shift/rotate/scale composition
//!
//! **1) Zakharov Function**
//! `f1(x) = sum_{i=1}^{D} x_i^2 + (sum_{i=1}^{D} 0.5 x_i)^2 + (sum_{i=1}^{D} 0.5 x_i)^4`
//!
//! **2) Rosenbrock's Function**
//! `f2(x) = sum_{i=1}^{D-1} (100 (x_i^2 - x_{i+1})^2 + (x_i - 1)^2)`
//!
//! **3) Expanded Schaffer's Function** -- "Schaffer's Function: `g(x,y) = 0.5 + (sin^2(sqrt(x^2+y^2)) - 0.5) / (1 + 0.001(x^2+y^2))^2`", `f3(x) = g(x1,x2) + g(x2,x3) + ... + g(x_{D-1},x_D) + g(x_D,x1)` (CYCLIC, D terms).
//!
//! **NOTE**: this is NOT the formula the reference code actually uses for standalone problem 3 -- see the F3 discrepancy note below.
//!
//! **4) Rastrigin's Function**
//! `f4(x) = sum_{i=1}^{D} (x_i^2 - 10 cos(2 pi x_i) + 10)`
//!
//! **5) Levy Function**
//! `f5(x) = sin^2(pi w1) + sum_{i=1}^{D-1} (w_i-1)^2 [1 + 10 sin^2(pi w_i + 1)] + (w_D-1)^2 [1 + sin^2(2 pi w_D)]`,
//! `w_i = 1 + (x_i - 1)/4` for `i = 1,...,D`. (The report's own printed
//! exponent term reads ambiguously as `sin^2(pi*w_i - 1)` under PDF text
//! extraction; the vendored C reference's `levy_func`, unambiguous source
//! code, computes `sin(PI*wi+1)` -- `+1` inside the sine, not `-1` outside
//! it -- which this module follows.)
//!
//! **16) Schaffer's F7 Function** (listed among the OTHER basic functions,
//! used by hybrid/composition fid -- and, per the F3 discrepancy below,
//! ALSO the actual formula standalone problem 3 dispatches to):
//! `f16(x) = [1/(D-1) sum_{i=1}^{D-1} (sqrt(s_i) (sin(50 s_i^0.2) + 1))]^2`,
//! `s_i = sqrt(x_i^2 + x_{i+1}^2)` (NON-cyclic, `D-1` terms).
//!
//! ## 1.4.A. Basic Functions -- shift/rotate/scale composition (quoted
//! equations 16-20 as printed; see the discrepancy notes for where the
//! printed equation and the vendored C reference code disagree, and which
//! one this module follows)
//!
//! - Eq (16): `F1(x) = f1(M(x - o1)) + F1*` -- no scale factor.
//! - Eq (17): `F2(x) = f2(M(2.048(x-o2)/100) + 1) + F2*` -- scale
//!   `2.048/100`, `+1` applied to the ROTATED vector (shifts Rosenbrock's
//!   `f2`, whose unshifted optimum is at `1`, back near the domain origin).
//! - Eq (18): `F3(x) = f3(M(0.5(x-o3)/100)) + F3*` -- scale `0.5/100`,
//!   printed `f3` (section 1.3's Expanded Schaffer's). **DISCREPANCY --
//!   does not match the reference code; see below.**
//! - Eq (19): `F4(x) = f4(M(5.12(x-o4)/100)) + F4*` -- scale `5.12/100`.
//! - Eq (20): `F5(x) = f5(M(5.12(x-o5)/100)) + F5*` -- scale `5.12/100` AS
//!   PRINTED. **DISCREPANCY -- does not match the reference code; see
//!   below.**
//!
//! ## VERIFIED DISCREPANCIES between the printed report and the vendored
//! reference C code (`cec22_test_func.cpp`) -- found by directly reading
//! the fetched, byte-identical C source (not memory, not guesswork); this
//! module implements the C REFERENCE CODE's actual, executable behavior
//! (the ground truth every submitted competition algorithm is literally
//! scored against), documenting each divergence from the printed report
//! text for T8's independent (opfunu) cross-check to adjudicate further.
//!
//! ### F3: report says "Expanded Schaffer's" with scale `0.5/100` and full
//! rotation; the code computes plain "Schaffer's F7" (eq 16) with NO scale
//! and, due to what reads as a buffer-reuse bug, NO effective rotation.
//!
//! `cec22_test_func`'s dispatch (quoted): `case 3:
//! schaffer_F7_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=600.0;` -- calls
//! `schaffer_F7_func`, NOT `escaffer6_func` (the function that actually
//! implements section 1.3's eq (3), and which the report's own eq (18)
//! symbol `f3` should mean). `escaffer6_func` is used ONLY inside the
//! composition functions (`cf06`, `cf07`), never for standalone problem 3 --
//! confirmed by grepping every call site of both functions in the vendored
//! source. The report's OWN section headers additionally disagree with each
//! other on the name: "3) Shifted and full Rotated Expanded Schaffer's F7"
//! (section 1.4 heading) vs. "Figure 3 Shifted and full Rotated Expanded
//! Schaffer's f6 Function" (the very same section's figure caption, two
//! lines later) -- so the report text is self-contradictory on this point,
//! independent of the code.
//!
//! `schaffer_F7_func`, quoted: `sr_func (x, z, nx, Os, Mr, 1.0, s_flag,
//! r_flag); for (i=0;i<nx-1;i++) { z[i]=pow(y[i]*y[i]+y[i+1]*y[i+1],0.5);
//! ... }`. `sr_func`'s `sh_rate` argument is `1.0` -- NO scale (not
//! `0.5/100`). Worse: `sr_func` computes the FULLY shifted+rotated result
//! into `z` (its `sr_x` output parameter), but the very next loop
//! OVERWRITES `z` using `y` -- the GLOBAL intermediate buffer `sr_func`
//! itself populated with the shifted-but-NOT-YET-rotated vector
//! (`shiftfunc(x,y,nx,Os)` inside `sr_func`, quoted in `data.rs`'s sibling
//! reading) -- so the rotation `sr_func` computed into `z` is silently
//! discarded; only the SHIFT (`x - o3`) reaches the formula. This module's
//! fid-3 branch therefore applies shift only, no scale, no rotation --
//! `M_3_D*.txt` is still parsed and stored (`Cec2022`'s `m` field, for
//! struct/API uniformity and because T6/T7's hybrid function 2 reuses this
//! same `f16` core WITH its own rotation in a different context), but
//! deliberately left UNUSED by fid 3's `evaluate_batch` arm, matching the
//! verified reference behavior.
//!
//! ### F4: report's own function name ("Non-Continuous Rastrigin's") and
//! dispatch (`step_rastrigin_func`) both signal a genuinely non-continuous
//! function; the code's non-continuous transform is DEAD CODE due to an
//! argument-passing bug, so the code's actual output is PLAIN continuous
//! Rastrigin.
//!
//! `step_rastrigin_func`, quoted in full:
//! ```text
//! for (i=0; i<nx; i++)
//!     if (fabs(y[i]-Os[i])>0.5)
//!         y[i]=Os[i]+floor(2*(y[i]-Os[i])+0.5)/2;
//! sr_func (x, z, nx, Os, Mr, 5.12/100.0, s_flag, r_flag);
//! for (i=0; i<nx; i++)
//!     f[0] += (z[i]*z[i] - 10.0*cos(2.0*PI*z[i]) + 10.0);
//! ```
//! The first loop computes a step-quantized value into the global `y`
//! buffer -- but `sr_func` is then called with `x` (the function's raw
//! input parameter), NOT `y`, so the step-quantized `y` is never read
//! again: the shift/scale/rotate pipeline runs on the UNMODIFIED `x`, and
//! the step transform has no effect on `f[0]` whatsoever. (Independently,
//! `y[i]` is also read BEFORE this call ever writes it -- on a process's
//! first evaluation `y` is uninitialized `malloc`'d memory, and on later
//! evaluations it holds a stale value left over from a PRIOR individual's
//! unrelated computation -- a second, compounding bug in the same function.)
//! Taken together, `step_rastrigin_func` as literally executed is
//! `rastrigin_func` with an inert, unreachable step-quantization loop
//! bolted on. Per the brief's explicit instruction to verify (not assume)
//! whether fid 4 is plain or non-continuous, and given this is a verified
//! reading of the actual reference source (not a guess): this module
//! implements PLAIN Rastrigin for fid 4 -- `F4(x) = rastrigin(M(5.12(x-o4)/100)) + F4*`
//! -- matching what the reference code actually computes, not what its
//! function name promises. Flagged here explicitly for T8's opfunu
//! cross-check: if an independent implementation disagrees by applying the
//! (arguably "intended", but not what ships) non-continuous transform, that
//! is the discrepancy to adjudicate there, per this report's own
//! DISCREPANCY PROTOCOL.
//!
//! ### F5: the report's printed Eq (20) shows scale `5.12/100` -- IDENTICAL
//! to Eq (19)'s Rastrigin scale directly above it, strongly suggesting a
//! copy-paste error in the PDF's typesetting (confirmed by extracting the
//! PDF text two independent ways, `pdftotext` with and without `-layout`:
//! both agree the printed glyphs read "5.12(x-o5)/100", not some other
//! value the extractor might have mangled) -- but `levy_func`'s call is
//! `sr_func (x, z, nx, Os, Mr, 1.0, s_flag, r_flag)`: `sh_rate = 1.0`, NO
//! scale. This module follows the code: fid 5 shifts and rotates with no
//! extra scale factor.
//!
//! ## `x = o` pin (Step 2): why `F_i(o_i) == F_i*` EXACTLY for every fid
//! 1-5, independent of every discrepancy above
//!
//! At `x = o`, the shift `x - o = 0` REGARDLESS of which scale factor
//! (`sr`) is applied afterward (`0 * sr == 0`) and regardless of rotation
//! (`M * 0 == 0`, `M` orthogonal or not) -- so every fid's `z` (or, for fid
//! 3, `y`) is the ALL-ZERO vector, except fid 2, whose pipeline adds `1`
//! AFTER rotation, giving the ALL-ONES vector. Each base function is then:
//! `f1(0) = 0 + 0^2 + 0^4 = 0`; `f2(1,1,...,1) = sum(100*(1-1)^2+(1-1)^2) =
//! 0`; `f16(0) = [sum of sqrt(0)*(...)]^2 = 0` (every `s_i = 0`); `f4(0) =
//! sum(0 - 10*cos(0) + 10) = sum(0 - 10 + 10) = 0`; `f5` at `z=0` gives
//! `w_i = 1` for all `i`, so `sin^2(pi*1) = 0`, every `(w_i-1)^2` term is
//! `0`, and `(w_D-1)^2[...] = 0` -- `f5 = 0`. Every base function is `0` at
//! its pipeline's origin, so `F_i(o_i) = 0 + F_i* = F_i*` EXACTLY (no
//! floating-point residue: the shift subtraction `o_i - o_i` is exact for
//! identical `f64` operands, and every subsequent operation on the
//! resulting exact `0.0` stays exact through multiplication, `M * 0`, and
//! `+1`).
//!
//! ## Genotype mapping
//!
//! [`Cec2022::space`] is one [`Block::Float`] of `dim` variables, bounds
//! `[-100.0, 100.0]` (module doc, "Search range" quote above).

mod data;

use sezgi_core::problem::Problem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

#[derive(Debug, thiserror::Error)]
pub enum Cec2022Error {
    #[error("fid must be in 1..=12 (the CEC 2022 12-function table), got {0}")]
    UnknownFid(u32),
    #[error(
        "fid {0} is not implemented yet (only the basic functions f1-f5 are implemented so \
         far -- hybrid/composition functions fid 6-12 are deferred, see this module's doc)"
    )]
    NotImplemented(u32),
    #[error(
        "dim must be one of {{2,10,20}} (the report's supported dimensions -- the vendored \
         input_data files ship only these), got {0}"
    )]
    BadDim(usize),
}

/// One instance of a CEC 2022 basic function (fid 1-5): embedded official
/// shift vector `o` and rotation matrix `M` for the requested `(fid, dim)`,
/// plus the pinned `F_i*` bias (module doc, section 1.2's table). See the
/// module doc for each fid's exact shift/scale/rotate pipeline, including
/// the verified discrepancies (fid 3, 4, 5) between the report's printed
/// equations and the vendored reference C code this module actually
/// follows.
pub struct Cec2022 {
    fid: u32,
    dim: usize,
    o: Vec<f64>,
    m: Vec<Vec<f64>>,
    space: SearchSpace,
}

impl Cec2022 {
    /// `fid` in `1..=5` (module doc: fid 6-12 are [`Cec2022Error::NotImplemented`],
    /// deferred to T6/T7; fid outside `1..=12` entirely is
    /// [`Cec2022Error::UnknownFid`]), `dim` in `{2,10,20}` (module doc: the
    /// only dims the report supports and the vendored data ships).
    pub fn new(fid: u32, dim: usize) -> Result<Cec2022, Cec2022Error> {
        if !(1..=12).contains(&fid) {
            return Err(Cec2022Error::UnknownFid(fid));
        }
        if !(1..=5).contains(&fid) {
            return Err(Cec2022Error::NotImplemented(fid));
        }
        if !matches!(dim, 2 | 10 | 20) {
            return Err(Cec2022Error::BadDim(dim));
        }
        let o = data::shift_vector(fid, dim);
        let m = data::rotation_matrix(fid, dim);
        let space = SearchSpace::new(vec![Block::Float { lo: -100.0, hi: 100.0, n: dim }])
            .expect("CEC 2022 bounds (-100 < 100) are always valid");
        Ok(Self { fid, dim, o, m, space })
    }

    pub fn fid(&self) -> u32 { self.fid }
    pub fn dim(&self) -> usize { self.dim }

    /// The report's `F_i*` bias (module doc, section 1.2's table, quoted).
    pub fn f_star(&self) -> f64 {
        match self.fid {
            1 => 300.0,
            2 => 400.0,
            3 => 600.0,
            4 => 800.0,
            5 => 900.0,
            other => unreachable!("Cec2022::new rejects fid outside 1..=5, got {other}"),
        }
    }

    /// `(x - o) * sr`, optionally rotated by `self.m` (module doc's `sr_func`
    /// quote: shift, then shrink-scale, then rotate -- in that order).
    fn shift_scale_rotate(&self, xs: &[f64], sr: f64, rotate: bool) -> Vec<f64> {
        let scaled: Vec<f64> = xs.iter().zip(&self.o).map(|(x, o)| (x - o) * sr).collect();
        if rotate {
            crate::bbob::transform::apply(&self.m, &scaled)
        } else {
            scaled
        }
    }

    /// Basic function 1 (module doc, section 1.3): Zakharov.
    fn f1_base(z: &[f64]) -> f64 {
        let sum1: f64 = z.iter().map(|&zi| zi * zi).sum();
        let sum2: f64 = z.iter().enumerate().map(|(i, &zi)| 0.5 * (i + 1) as f64 * zi).sum();
        sum1 + sum2.powi(2) + sum2.powi(4)
    }

    /// Basic function 2 (module doc, section 1.3): Rosenbrock's.
    fn f2_base(z: &[f64]) -> f64 {
        (0..z.len() - 1)
            .map(|i| 100.0 * (z[i] * z[i] - z[i + 1]).powi(2) + (z[i] - 1.0).powi(2))
            .sum()
    }

    /// Basic function 16 (module doc, section 1.3): Schaffer's F7 -- the
    /// formula the reference code actually dispatches problem 3 to (module
    /// doc's F3 discrepancy note).
    fn f16_schaffer_f7_base(y: &[f64]) -> f64 {
        let sum: f64 = (0..y.len() - 1)
            .map(|i| {
                let si = (y[i] * y[i] + y[i + 1] * y[i + 1]).sqrt();
                si.sqrt() + si.sqrt() * (50.0 * si.powf(0.2)).sin().powi(2)
            })
            .sum();
        sum * sum / (y.len() - 1).pow(2) as f64
    }

    /// Basic function 4 (module doc, section 1.3): Rastrigin's -- PLAIN,
    /// per the module doc's F4 discrepancy note (the reference code's
    /// non-continuous transform is dead code).
    fn f4_base(z: &[f64]) -> f64 {
        z.iter().map(|&zi| zi * zi - 10.0 * (2.0 * std::f64::consts::PI * zi).cos() + 10.0).sum()
    }

    /// Basic function 5 (module doc, section 1.3): Levy.
    fn f5_base(z: &[f64]) -> f64 {
        let pi = std::f64::consts::PI;
        let w: Vec<f64> = z.iter().map(|&zi| 1.0 + zi / 4.0).collect();
        let term1 = (pi * w[0]).sin().powi(2);
        let mid: f64 = (0..w.len() - 1)
            .map(|i| (w[i] - 1.0).powi(2) * (1.0 + 10.0 * (pi * w[i] + 1.0).sin().powi(2)))
            .sum();
        let last = w.len() - 1;
        let term3 = (w[last] - 1.0).powi(2) * (1.0 + (2.0 * pi * w[last]).sin().powi(2));
        term1 + mid + term3
    }

    /// The full pipeline (shift/scale/rotate + base function + `F_i*` bias)
    /// for one already-flattened decision vector, per this module's doc
    /// (each fid's exact `sr`/rotate choice, including the verified fid
    /// 3/4/5 discrepancies from the printed report).
    fn eval_one(&self, xs: &[f64]) -> f64 {
        match self.fid {
            1 => Self::f1_base(&self.shift_scale_rotate(xs, 1.0, true)) + self.f_star(),
            2 => {
                let mut z = self.shift_scale_rotate(xs, 2.048 / 100.0, true);
                for zi in &mut z {
                    *zi += 1.0;
                }
                Self::f2_base(&z) + self.f_star()
            }
            3 => {
                // No scale, no rotation -- module doc's F3 discrepancy note.
                let y = self.shift_scale_rotate(xs, 1.0, false);
                Self::f16_schaffer_f7_base(&y) + self.f_star()
            }
            4 => Self::f4_base(&self.shift_scale_rotate(xs, 5.12 / 100.0, true)) + self.f_star(),
            5 => {
                // No extra scale -- module doc's F5 discrepancy note.
                Self::f5_base(&self.shift_scale_rotate(xs, 1.0, true)) + self.f_star()
            }
            other => unreachable!("Cec2022::new rejects fid outside 1..=5, got {other}"),
        }
    }
}

impl Problem for Cec2022 {
    fn space(&self) -> &SearchSpace { &self.space }
    fn optimum(&self) -> Option<f64> { Some(self.f_star()) }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
        pop.iter()
            .map(|g| match g.blocks.first() {
                Some(BlockValues::Float(xs)) if xs.len() == self.dim => self.eval_one(xs),
                _ => f64::INFINITY,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::Evaluator;

    fn g(xs: Vec<f64>) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs)] } }

    // ---- construction / errors ----

    #[test]
    fn fid_0_and_13_are_unknown() {
        for fid in [0u32, 13, 100] {
            assert!(matches!(Cec2022::new(fid, 10), Err(Cec2022Error::UnknownFid(f)) if f == fid), "fid={fid}");
        }
    }

    #[test]
    fn fid_6_to_12_are_not_implemented_yet() {
        for fid in 6u32..=12 {
            assert!(
                matches!(Cec2022::new(fid, 10), Err(Cec2022Error::NotImplemented(f)) if f == fid),
                "fid={fid}"
            );
        }
    }

    #[test]
    fn dim_5_is_bad_dim_for_every_implemented_fid() {
        for fid in 1u32..=5 {
            assert!(matches!(Cec2022::new(fid, 5), Err(Cec2022Error::BadDim(5))), "fid={fid}");
        }
    }

    #[test]
    fn space_is_single_float_block_pm100() {
        for fid in 1u32..=5 {
            let p = Cec2022::new(fid, 10).unwrap();
            assert_eq!(p.space().blocks(), &[Block::Float { lo: -100.0, hi: 100.0, n: 10 }], "fid={fid}");
            assert_eq!(p.space().dim(), 10);
        }
    }

    #[test]
    fn f_star_matches_report_table() {
        let expect = [(1u32, 300.0), (2, 400.0), (3, 600.0), (4, 800.0), (5, 900.0)];
        for (fid, fstar) in expect {
            assert_eq!(Cec2022::new(fid, 10).unwrap().f_star(), fstar, "fid={fid}");
        }
    }

    // ---- x = o pin (module doc's derivation): F(o) == F* EXACTLY, dim=10 ----

    #[test]
    fn x_equals_o_pins_f_star_exactly_dim10() {
        for fid in 1u32..=5 {
            let p = Cec2022::new(fid, 10).unwrap();
            let out = p.evaluate_batch(&[g(p.o.clone())]);
            assert_eq!(out, vec![p.f_star()], "fid={fid}: F(o) must equal F* EXACTLY");
        }
    }

    // ---- hand fixtures at dim=2, x = o + [1, -1] (module doc's discrepancy
    // notes govern each fid's pipeline; arithmetic scratch-verified in
    // Python outside this crate, values transcribed here). ----

    #[test]
    fn f1_zakharov_hand_fixture_dim2() {
        // o1=[-55.938326705218444, 4.5430653935964642], x=o1+[1,-1].
        // shift=[1,-1] (sr=1, no further scale), M_1_D2=
        //   [ 0.999658775378376  -0.026121500894966159]
        //   [-0.026121500894966111 -0.999658775378376  ]  (approx, see data.rs)
        // z = M*[1,-1] = [1.0257802762733421, 0.9735372744834098]
        // f1(z) = sum(z^2) + (sum 0.5*(i+1)*z_i)^2 + (...)^4 = 9.09120845986969
        // F1 = f1 + 300 = 309.0912084598697
        let p = Cec2022::new(1, 2).unwrap();
        let x = vec![p.o[0] + 1.0, p.o[1] - 1.0];
        let out = p.evaluate_batch(&[g(x)])[0];
        assert!((out - 309.091_208_459_869_7).abs() < 1e-9, "{out}");
    }

    #[test]
    fn f2_rosenbrock_hand_fixture_dim2() {
        // scale=2.048/100, rotate by M_2_D2, then +1.
        // z(post-rotate) = [1.0052713575694814, 1.0284793537387116] pre-+1... already includes +1 below
        // f2(z) = 0.032100483018396256, F2 = f2+400 = 400.0321004830184
        let p = Cec2022::new(2, 2).unwrap();
        let x = vec![p.o[0] + 1.0, p.o[1] - 1.0];
        let out = p.evaluate_batch(&[g(x)])[0];
        assert!((out - 400.032_100_483_018_4).abs() < 1e-9, "{out}");
    }

    #[test]
    fn f3_schaffer_f7_hand_fixture_dim2() {
        // NO scale, NO rotation (module doc discrepancy note): y = x-o3 = [1,-1].
        // s_0 = sqrt(1^2+(-1)^2) = sqrt(2); f16(y) = 1.5079726648501366
        // F3 = f16+600 = 601.5079726648502
        let p = Cec2022::new(3, 2).unwrap();
        let x = vec![p.o[0] + 1.0, p.o[1] - 1.0];
        let out = p.evaluate_batch(&[g(x)])[0];
        assert!((out - 601.507_972_664_850_2).abs() < 1e-9, "{out}");
    }

    #[test]
    fn f4_rastrigin_hand_fixture_dim2() {
        // scale=5.12/100, rotate by M_4_D2.
        // z = [0.07198630529613456, -0.007800759566331264]
        // f4(z) = 1.0228235446323275, F4 = f4+800 = 801.0228235446323
        let p = Cec2022::new(4, 2).unwrap();
        let x = vec![p.o[0] + 1.0, p.o[1] - 1.0];
        let out = p.evaluate_batch(&[g(x)])[0];
        assert!((out - 801.022_823_544_632_3).abs() < 1e-9, "{out}");
    }

    #[test]
    fn f5_levy_hand_fixture_dim2() {
        // NO extra scale (module doc discrepancy note), rotate by M_5_D2.
        // z = [-1.2258801295325246, 0.7051368009239913]
        // w = [0.6935299676168689, 1.176284200230998]
        // f5(z) = 0.8248785312697374, F5 = f5+900 = 900.8248785312697
        let p = Cec2022::new(5, 2).unwrap();
        let x = vec![p.o[0] + 1.0, p.o[1] - 1.0];
        let out = p.evaluate_batch(&[g(x)])[0];
        assert!((out - 900.824_878_531_269_7).abs() < 1e-9, "{out}");
    }

    // ---- Evaluator integration + budget counting ----

    #[test]
    fn evaluator_counts_budget_and_tracks_best() {
        let p = Cec2022::new(1, 10).unwrap();
        let mut ev = Evaluator::new(&p, 10);
        let at_opt = g(p.o.clone());
        let mut worse = p.o.clone();
        worse[0] += 5.0;
        ev.evaluate(&[at_opt, g(worse)]).unwrap();
        assert_eq!(ev.used(), 2);
        assert_eq!(ev.best_so_far(), Some(p.f_star()));
    }

    #[test]
    fn optimum_maps_to_f_star() {
        for fid in 1u32..=5 {
            let p = Cec2022::new(fid, 10).unwrap();
            assert_eq!(Problem::optimum(&p), Some(p.f_star()), "fid={fid}");
        }
    }

    // ---- malformed genotype handling ----

    #[test]
    fn wrong_length_float_block_evaluates_to_infinity_not_panic() {
        let p = Cec2022::new(1, 10).unwrap();
        let out = p.evaluate_batch(&[g(vec![0.0; 3])]);
        assert_eq!(out, vec![f64::INFINITY]);
    }

    #[test]
    fn non_float_block_evaluates_to_infinity() {
        let p = Cec2022::new(1, 10).unwrap();
        let bad = Genotype { blocks: vec![BlockValues::Perm(vec![0, 1, 2])] };
        assert_eq!(p.evaluate_batch(&[bad]), vec![f64::INFINITY]);
    }

    // ---- full sweep: fid 1..=5 x dims {2,10,20}, one evaluation, no panic ----

    #[test]
    fn full_sweep_fid_1_to_5_dims_2_10_20_evaluates_without_panic() {
        for fid in 1u32..=5 {
            for &dim in &[2usize, 10, 20] {
                let p = Cec2022::new(fid, dim).unwrap();
                let x = vec![1.5; dim];
                let out = p.evaluate_batch(&[g(x)])[0];
                assert!(out.is_finite(), "fid={fid} dim={dim}: {out}");
            }
        }
    }

    #[test]
    fn away_from_optimum_is_worse_for_every_fid() {
        for fid in 1u32..=5 {
            let p = Cec2022::new(fid, 10).unwrap();
            let mut x = p.o.clone();
            x[0] += 10.0;
            let out = p.evaluate_batch(&[g(x)])[0];
            assert!(out > p.f_star(), "fid={fid}: {out} not > {}", p.f_star());
        }
    }
}
