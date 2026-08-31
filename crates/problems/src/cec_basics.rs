//! Shared CEC basic-function library, extracted verbatim from
//! `cec2022::Cec2022` in M3-6 T1 for cross-suite reuse (CEC 2014, CEC 2017,
//! and any later CEC year that shares the same basic-function/`sr_func`/
//! `cf_cal` design lineage). This is a PURE RELOCATION -- every function
//! body below is byte-identical to the code it was extracted from (only
//! visibility changed, private `impl Cec2022` associated function to
//! `pub(crate)` free function; [`shift_scale_rotate`] additionally lost its
//! `&self` receiver in favor of explicit `o`/`m` parameters so both of
//! cec2022's two near-duplicate shift/rotate call sites can share it -- see
//! that function's own doc for the detail). Each function's own provenance
//! doc comment (report quote, C reference quote, verified-discrepancy notes
//! where applicable) travels with it unmodified; do not re-derive or
//! "improve" any formula here without re-reading `cec2022/mod.rs`'s module
//! doc first, since several of these are pinned to a VERIFIED reference-C
//! behavior that intentionally diverges from the printed report text (see
//! `cec2022/mod.rs`'s module doc, "VERIFIED DISCREPANCIES" section, for
//! which ones and why).
//!
//! // sezgi decision: what did NOT move here, and why (M3-6 T1) --
//! `HybridComponent`/`CompFn` (cec2022's own dispatch enums over these
//! basics), `hybrid_spec`/`composition_spec` (cec2022's own fid-specific
//! proportion/delta/bias/component tables), `hybrid_fitness` (entangled
//! with cec2022's VERIFIED fid-7 `SchafferF7Buggy` dead-segment replication
//! -- the bug lives in how cec2022 slices its shuffled `y`, not in
//! [`f16_schaffer_f7_base`] itself, which is plain suite-agnostic Schaffer's
//! F7 and DID move), `composition_fitness`/`eval_one` (cec2022's own
//! per-fid orchestration, F* biases, and fid-specific scale/rotate choices),
//! and the shuffle application (`z[shuffle[i]]`, a one-line inlined index
//! op in `eval_one` -- there was no separable "shuffle machinery" function
//! to extract). All of these stay in `cec2022/mod.rs`, unmodified. Whether
//! CEC 2014/2017 need their own `SchafferF7`-shaped buggy-component
//! replication, or any shuffle helper of their own, is for those suites'
//! own tasks to judge from their own reference C -- nothing here assumes it.

/// Shift, scale, and (optionally) rotate one decision vector: `(x - o) *
/// sr`, then `M * result` if `rotate` -- mirrors the CEC reference
/// `sr_func`'s shift-then-scale-then-rotate order (`cec2022/mod.rs`'s module
/// doc, `sr_func` quote). Extracted from `Cec2022::shift_scale_rotate`
/// (M3-6 T1); `o`/`m` are explicit parameters here instead of `&self`
/// fields so this single function also serves what was
/// `Cec2022::comp_shift_scale_rotate` (byte-identical logic, differing only
/// in which of `Cec2022`'s two shift/rotation-matrix stores it read --
/// `self.o`/`self.m` vs. `self.comp_shift[idx]`/`self.comp_rotation[idx]`)
/// -- both cec2022 call sites now delegate here.
pub(crate) fn shift_scale_rotate(xs: &[f64], o: &[f64], m: &[Vec<f64>], sr: f64, rotate: bool) -> Vec<f64> {
    let scaled: Vec<f64> = xs.iter().zip(o).map(|(x, o)| (x - o) * sr).collect();
    if rotate {
        crate::bbob::transform::apply(m, &scaled)
    } else {
        scaled
    }
}

/// Basic function 1 (CEC 2022 report, section 1.3): Zakharov.
pub(crate) fn f1_base(z: &[f64]) -> f64 {
    let sum1: f64 = z.iter().map(|&zi| zi * zi).sum();
    let sum2: f64 = z.iter().enumerate().map(|(i, &zi)| 0.5 * (i + 1) as f64 * zi).sum();
    sum1 + sum2.powi(2) + sum2.powi(4)
}

/// Basic function 2 (CEC 2022 report, section 1.3): Rosenbrock's.
pub(crate) fn f2_base(z: &[f64]) -> f64 {
    (0..z.len() - 1)
        .map(|i| 100.0 * (z[i] * z[i] - z[i + 1]).powi(2) + (z[i] - 1.0).powi(2))
        .sum()
}

/// Basic function 16 (CEC 2022 report, section 1.3): Schaffer's F7 -- the
/// formula the CEC 2022 reference code actually dispatches problem 3 to
/// (`cec2022/mod.rs`'s module doc, F3 discrepancy note: the printed report's
/// sine term is NOT squared, the code's is -- this implementation follows
/// the code, `.sin().powi(2)`).
pub(crate) fn f16_schaffer_f7_base(y: &[f64]) -> f64 {
    let sum: f64 = (0..y.len() - 1)
        .map(|i| {
            let si = (y[i] * y[i] + y[i + 1] * y[i + 1]).sqrt();
            si.sqrt() + si.sqrt() * (50.0 * si.powf(0.2)).sin().powi(2)
        })
        .sum();
    sum * sum / (y.len() - 1).pow(2) as f64
}

/// Basic function 4 (CEC 2022 report, section 1.3): Rastrigin's -- PLAIN,
/// per `cec2022/mod.rs`'s module doc F4 discrepancy note (the reference
/// code's non-continuous transform is dead code due to a verified
/// argument-passing bug).
pub(crate) fn f4_base(z: &[f64]) -> f64 {
    z.iter().map(|&zi| zi * zi - 10.0 * (2.0 * std::f64::consts::PI * zi).cos() + 10.0).sum()
}

/// Basic function 5 (CEC 2022 report, section 1.3): Levy.
pub(crate) fn f5_base(z: &[f64]) -> f64 {
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

/// Basic function 6 (CEC 2022 report, "New component basic functions"):
/// Bent Cigar.
pub(crate) fn bent_cigar_base(z: &[f64]) -> f64 {
    z[0] * z[0] + z[1..].iter().map(|&zi| 1.0e6 * zi * zi).sum::<f64>()
}

/// Basic function 7 (CEC 2022 report): HGBat.
pub(crate) fn hgbat_base(z: &[f64]) -> f64 {
    const ALPHA: f64 = 1.0 / 4.0;
    let n = z.len() as f64;
    let zs: Vec<f64> = z.iter().map(|&zi| zi - 1.0).collect();
    let r2: f64 = zs.iter().map(|&zi| zi * zi).sum();
    let sum_z: f64 = zs.iter().sum();
    (r2.powf(2.0) - sum_z.powf(2.0)).abs().powf(2.0 * ALPHA) + (0.5 * r2 + sum_z) / n + 0.5
}

/// Basic function 9 (CEC 2022 report): Katsuura.
pub(crate) fn katsuura_base(z: &[f64]) -> f64 {
    let n = z.len() as f64;
    let tmp3 = n.powf(1.2);
    let mut f = 1.0;
    for (i, &zi) in z.iter().enumerate() {
        let mut temp = 0.0;
        for j in 1..=32 {
            let tmp1 = 2f64.powi(j);
            let tmp2 = tmp1 * zi;
            temp += (tmp2 - (tmp2 + 0.5).floor()).abs() / tmp1;
        }
        f *= (1.0 + (i + 1) as f64 * temp).powf(10.0 / tmp3);
    }
    let tmp1 = 10.0 / (n * n);
    f * tmp1 - tmp1
}

/// Basic function 13 (CEC 2022 report): Ackley's.
pub(crate) fn ackley_base(z: &[f64]) -> f64 {
    let n = z.len() as f64;
    let sum1: f64 = z.iter().map(|&zi| zi * zi).sum();
    let sum2: f64 = z.iter().map(|&zi| (2.0 * std::f64::consts::PI * zi).cos()).sum();
    let sum1 = -0.2 * (sum1 / n).sqrt();
    let sum2 = sum2 / n;
    std::f64::consts::E - 20.0 * sum1.exp() - sum2.exp() + 20.0
}

/// Basic function 12 (CEC 2022 report): Modified Schwefel's -- written to
/// mirror `schwefel_func`'s three branches literally (`cec2022/mod.rs`'s
/// module doc, quoted there in full), not algebraically simplified, so it
/// stays diffable against the quoted C.
pub(crate) fn schwefel_base(z: &[f64]) -> f64 {
    let n = z.len() as f64;
    let mut f = 0.0;
    for &z0 in z {
        let zi = z0 + 4.209687462275036e+02;
        if zi > 500.0 {
            let r = 500.0 - (zi % 500.0);
            f -= r * r.sqrt().sin();
            let tmp = (zi - 500.0) / 100.0;
            f += tmp * tmp / n;
        } else if zi < -500.0 {
            let faz = zi.abs() % 500.0;
            let mult = -500.0 + faz;
            let arg = (500.0 - faz).sqrt();
            f -= mult * arg.sin();
            let tmp = (zi + 500.0) / 100.0;
            f += tmp * tmp / n;
        } else {
            f -= zi * zi.abs().sqrt().sin();
        }
    }
    f + 4.189828872724338e+02 * n
}

/// Basic function 10 (CEC 2022 report): HappyCat.
pub(crate) fn happycat_base(z: &[f64]) -> f64 {
    const ALPHA: f64 = 1.0 / 8.0;
    let n = z.len() as f64;
    let zs: Vec<f64> = z.iter().map(|&zi| zi - 1.0).collect();
    let r2: f64 = zs.iter().map(|&zi| zi * zi).sum();
    let sum_z: f64 = zs.iter().sum();
    (r2 - n).abs().powf(2.0 * ALPHA) + (0.5 * r2 + sum_z) / n + 0.5
}

/// Basic function 11 (CEC 2022 report): Expanded Griewank's plus
/// Rosenbrock's (CYCLIC).
pub(crate) fn grie_rosen_base(z: &[f64]) -> f64 {
    let n = z.len();
    let z1: Vec<f64> = z.iter().map(|&zi| zi + 1.0).collect();
    let mut f = 0.0;
    for i in 0..n - 1 {
        let tmp1 = z1[i] * z1[i] - z1[i + 1];
        let tmp2 = z1[i] - 1.0;
        let temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
        f += temp * temp / 4000.0 - temp.cos() + 1.0;
    }
    let tmp1 = z1[n - 1] * z1[n - 1] - z1[0];
    let tmp2 = z1[n - 1] - 1.0;
    let temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2;
    f + temp * temp / 4000.0 - temp.cos() + 1.0
}

// ---- CEC 2022 T7: composition-only new basic functions (module doc's T7
// section quotes each C body) ----

/// `ellips_func` (Ellipsoidal, CEC 2022 report T7 section): `f(z) =
/// sum_{i=0}^{D-1} 10^(6i/(D-1)) z_i^2` (0-based `i`).
pub(crate) fn ellips_base(z: &[f64]) -> f64 {
    let n = z.len();
    z.iter().enumerate().map(|(i, &zi)| 10f64.powf(6.0 * i as f64 / (n - 1) as f64) * zi * zi).sum()
}

/// `discus_func` (Discus, CEC 2022 report T7 section): `f(z) = 10^6 z_0^2 +
/// sum_{i=1}^{D-1} z_i^2`.
pub(crate) fn discus_base(z: &[f64]) -> f64 {
    1.0e6 * z[0] * z[0] + z[1..].iter().map(|&zi| zi * zi).sum::<f64>()
}

/// `griewank_func` (Griewank's, CEC 2022 report T7 section): `f(z) = 1 +
/// sum(z_i^2)/4000 - prod(cos(z_i/sqrt(1+i)))` (0-based `i`, standalone --
/// NOT the [`grie_rosen_base`] cyclic Rosenbrock composite above).
pub(crate) fn griewank_base(z: &[f64]) -> f64 {
    let s: f64 = z.iter().map(|&zi| zi * zi).sum();
    let p: f64 = z.iter().enumerate().map(|(i, &zi)| (zi / (1.0 + i as f64).sqrt()).cos()).product();
    1.0 + s / 4000.0 - p
}

/// `escaffer6_func` (Expanded Scaffer's F6, CEC 2022 report T7 section): the
/// CYCLIC `g(x,y) = 0.5 + (sin^2(sqrt(x^2+y^2)) - 0.5)/(1+0.001(x^2+y^2))^2`
/// summed over every consecutive pair PLUS the wrap-around `(D-1,0)` pair --
/// section 1.3's ORIGINAL "Expanded Schaffer's" formula, used (per
/// `cec2022/mod.rs`'s module doc F3 discrepancy note) ONLY inside
/// compositions there, never for standalone fid 3 (which dispatches to the
/// DIFFERENT [`f16_schaffer_f7_base`] instead).
pub(crate) fn escaffer6_base(z: &[f64]) -> f64 {
    let n = z.len();
    let g = |a: f64, b: f64| {
        let s = a * a + b * b;
        let t1 = s.sqrt().sin().powi(2);
        let t2 = 1.0 + 0.001 * s;
        0.5 + (t1 - 0.5) / (t2 * t2)
    };
    let mut f = 0.0;
    for i in 0..n - 1 {
        f += g(z[i], z[i + 1]);
    }
    f + g(z[n - 1], z[0])
}

/// `G_nx` (CEC 2022 report Eq (21) / `hf02` quote, `cec2022/mod.rs`'s module
/// doc): `ceil(p_i*dim)` for every proportion but the last, which absorbs
/// the remainder (`dim - sum(the rest)`) so the segments always sum to
/// exactly `dim`.
pub(crate) fn segment_sizes(gp: &[f64], dim: usize) -> Vec<usize> {
    let n = gp.len();
    let mut sizes = vec![0usize; n];
    let mut total = 0usize;
    for (i, size) in sizes.iter_mut().enumerate().take(n - 1) {
        *size = (gp[i] * dim as f64).ceil() as usize;
        total += *size;
    }
    sizes[n - 1] = dim - total;
    sizes
}

/// `G` (CEC 2022 report Eq (21) / `hf02` quote): cumulative segment start
/// offsets from `sizes`.
pub(crate) fn segment_starts(sizes: &[usize]) -> Vec<usize> {
    let mut starts = vec![0usize; sizes.len()];
    for i in 1..sizes.len() {
        starts[i] = starts[i - 1] + sizes[i - 1];
    }
    starts
}

/// Weierstrass Function (CEC 2014 report, section 1.3, eq (6)): `f(z) =
/// sum_{i=1}^{D} sum_{k=0}^{kmax} [a^k cos(2*pi*b^k*(z_i+0.5))] - D *
/// sum_{k=0}^{kmax} [a^k cos(2*pi*b^k*0.5)]`, `a=0.5, b=3, kmax=20`
/// (`crates/problems/src/cec2014/mod.rs`'s module doc has the full
/// PROVENANCE quote and the compiled-C hand fixture this was verified
/// against; new for M3-6 T2, CEC 2014 does not appear in CEC 2022's basic-
/// function set so this has no cec2022 sibling to reuse). `weierstrass_func`,
/// quoted in full: `a=0.5;b=3.0;k_max=20;f[0]=0.0; ... for(i=0;i<nx;i++) {
/// sum=0.0;sum2=0.0; for(j=0;j<=k_max;j++) {
/// sum+=pow(a,j)*cos(2.0*PI*pow(b,j)*(z[i]+0.5));
/// sum2+=pow(a,j)*cos(2.0*PI*pow(b,j)*0.5); } f[0]+=sum; } f[0]-=nx*sum2;` --
/// `sum2` does NOT depend on `i` (recomputed identically on every outer-loop
/// pass, a redundant-but-harmless C quirk); this Rust port keeps the same
/// redundant recomputation rather than hoisting it out, so the two stay
/// line-for-line diffable against the quoted C.
pub(crate) fn weierstrass_base(z: &[f64]) -> f64 {
    const A: f64 = 0.5;
    const B: f64 = 3.0;
    const K_MAX: i32 = 20;
    let pi = std::f64::consts::PI;
    let n = z.len() as f64;
    let mut f = 0.0;
    let mut sum2 = 0.0;
    for &zi in z {
        let mut sum = 0.0;
        sum2 = 0.0;
        for k in 0..=K_MAX {
            sum += A.powi(k) * (2.0 * pi * B.powi(k) * (zi + 0.5)).cos();
            sum2 += A.powi(k) * (2.0 * pi * B.powi(k) * 0.5).cos();
        }
        f += sum;
    }
    f - n * sum2
}

/// `cf_cal`'s weight formula (`cec2022/mod.rs`'s module doc T7 section,
/// quoted there in full): `w_i = (1/D_i)^0.5 * exp(-D_i / (2 * dim *
/// delta_i^2))` where `D_i = ||x - o_i||^2` (the RAW, UNSHIFTED distance --
/// `cf_cal` reads `x` directly, not any component's shift-rotated `z`),
/// `w_i = INF (1e99, the C's own finite sentinel, NOT IEEE infinity) if
/// D_i==0` exactly, then normalized by `sum(w)` -- EXCEPT if every `w_i` is
/// `0` (`w_max==0`, only possible if `dim==0`, unreachable in practice), in
/// which case the C falls back to a uniform `1/cf_num` each. A pure
/// function of `(xs, shifts, delta)` -- generalizes cleanly across any CEC
/// composition-function suite that reuses `cf_cal`'s exact formula.
pub(crate) fn composition_weights(xs: &[f64], shifts: &[Vec<f64>], delta: &[f64]) -> Vec<f64> {
    let cf_num = shifts.len();
    let dim = xs.len() as f64;
    let mut w = vec![0.0f64; cf_num];
    let mut w_max = 0.0f64;
    for i in 0..cf_num {
        let d2: f64 = xs.iter().zip(&shifts[i]).map(|(&x, &o)| (x - o) * (x - o)).sum();
        w[i] = if d2 != 0.0 {
            (1.0 / d2).sqrt() * (-d2 / 2.0 / dim / (delta[i] * delta[i])).exp()
        } else {
            1.0e99 // `cf_cal`'s `INF` -- a finite sentinel, not IEEE inf.
        };
        if w[i] > w_max {
            w_max = w[i];
        }
    }
    if w_max == 0.0 {
        return vec![1.0 / cf_num as f64; cf_num];
    }
    let w_sum: f64 = w.iter().sum();
    w.iter().map(|&wi| wi / w_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- M3-6 T2 hand fixture: weierstrass_base cross-checked against the
    // compiled CEC 2014 reference C (`weierstrass_func`, called directly
    // with `s_flag=0,r_flag=0` so no shift/rotation intervenes -- only
    // `weierstrass_func`'s OWN internal `sh_rate=0.5/100.0` scale, per
    // `sr_func`'s `s_flag==0,r_flag==0` branch: `sr_x[i]=x[i]*sh_rate`; to
    // land on z=[0.1,-0.2,0.3] the probe fed `x = z / (0.5/100.0) =
    // z*200.0`). This task's report has the isolated probe driver's full
    // source and the exact `g++`/run transcript.
    #[test]
    fn weierstrass_base_matches_compiled_c_hand_fixture() {
        // Compiled reference output (`%.20f`):
        // 5.12731920344174341153 for z=[0.1,-0.2,0.3].
        let z = [0.1_f64, -0.2, 0.3];
        let got = weierstrass_base(&z);
        let expect = 5.127_319_203_441_743_f64;
        assert!((got - expect).abs() < 1e-12, "got {got}, expect {expect}");
    }
}
