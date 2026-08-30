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
//! Checked PRECISELY (not just "fid 6-8 dim=2 is missing") against the
//! vendored `input_data/` file listing, since the exact shape of the gap
//! matters for T6's own data decisions: `M_6_D2.txt` and `M_8_D2.txt` DO
//! exist (fid 6 and fid 8 each have a complete `M_*_D2.txt` rotation-matrix
//! file); only `M_7_D2.txt` is genuinely ABSENT. What IS uniformly absent
//! for all three of fid 6/7/8, however, is dim=2 SHUFFLE data --
//! `shuffle_data_6_D2.txt`, `shuffle_data_7_D2.txt`, and
//! `shuffle_data_8_D2.txt` all do not exist (only their `_D10`/`_D20`
//! counterparts are present) -- which is what actually makes dim=2 unusable
//! for these hybrid functions regardless of fid 6/8's otherwise-present M
//! matrix, and is consistent with the C code's own blanket guard (quoted
//! above) rejecting `nx==2` for `func_num` 6, 7, AND 8 uniformly. `M_1_D2.txt`
//! through `M_5_D2.txt` and `M_9_D2.txt` through `M_12_D2.txt` are all
//! present. Not relevant to fid 1-5 (this task: all five ship complete
//! dim=2 data, see `data.rs`'s data-integrity tests), but recorded here
//! PRECISELY for T6/T7: `Cec2022::new` will need to reject `(fid in 6..=8,
//! dim == 2)` explicitly because of the missing SHUFFLE data (and, for fid
//! 7 specifically, the missing M matrix too) -- not because every fid 6-8
//! M matrix is missing at dim=2, which it is not.
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
//! `s_i = sqrt(x_i^2 + x_{i+1}^2)` (NON-cyclic, `D-1` terms), AS PRINTED --
//! this is the report's literal text; it does NOT match the code's squared
//! sine (a fourth discrepancy, documented in its own entry below, right
//! after F3's).
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
//! ### f16 formula (the Schaffer's F7 core F3 dispatches to): report prints
//! `sin(50 s_i^0.2)`, code computes `sin^2(50 s_i^0.2)`.
//!
//! Report eq (16), quoted above: `f16(x) = [1/(D-1) sum (sqrt(s_i)
//! (sin(50 s_i^0.2) + 1))]^2` -- the sine term is NOT squared as printed.
//! `schaffer_F7_func`, quoted: `tmp=sin(50.0*pow(z[i],0.2));
//! f[0] += pow(z[i],0.5)+pow(z[i],0.5)*tmp*tmp;` -- `tmp*tmp` is `sin^2`,
//! multiplying `sqrt(z_i)`, so the actual per-term contribution is
//! `sqrt(s_i) * (1 + sin^2(50 s_i^0.2))`, not `sqrt(s_i) * (1 + sin(50
//! s_i^0.2))`. This module (`f16_schaffer_f7_base`) follows the code:
//! `.sin().powi(2)`, matching `tmp*tmp` exactly, not the report's printed
//! (unsquared) form.
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
//! ### CONTROLLER RULING (recorded decision, Task 5 fix round 1): sezgi
//! follows the OFFICIAL C CODE wherever it and the printed report diverge
//! (F3, F4, F5, and the f16 sine-squaring above), because the C code is
//! what scored the CEC 2022 competition and produced its published results
//! -- the code IS the benchmark; the report is its (imperfect) documentation.
//! A consequence worth stating plainly: any re-implementation that instead
//! follows the printed report on these points (e.g. `opfunu`, if it does)
//! will disagree with sezgi on F3/F4/F5/f16 BY CONSTRUCTION, not by either
//! side's error. T8's cross-validation therefore treats the COMPILED
//! official C code as the primary reference for these functions, with
//! `opfunu` as a secondary check whose expected disagreements on exactly
//! these points are documented rather than "fixed" to force agreement.
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
//! ## T6: Hybrid Functions (fid 6-8)
//!
//! Source (PROVENANCE, re-fetched directly for this task, not reused from
//! memory): the SAME two artifacts T5 fetched -- `CEC2022%20TR.pdf` (this
//! time section "B. Hybrid Functions" and the seven `Basic Functions`
//! entries `f6`-`f13` that hadn't been needed yet) and `CEC2022.zip`'s
//! `cec22_test_func.cpp` (this time `hf02`/`hf06`/`hf10` and `sr_func`'s
//! `s_flag==0` branch). Both re-verified byte-identical to T5's fetch
//! (`CEC2022.zip` re-hashed to the SAME `sha1=176e677743774c8aa5397542caa7ae77776253f3`
//! T5's report recorded). A THIRD artifact was added for this task
//! specifically because the hybrid pipeline's pointer/buffer aliasing is too
//! easy to misread by eye (see the fid-7 discrepancy below, found this way):
//! the vendored C was actually COMPILED (`g++`, stripping only the
//! `WINDOWS.H` include and supplying the four `extern` globals `main.cpp`
//! normally defines) and RUN, both through the unmodified
//! `cec22_test_func()` entry point (for `F(o)` and hand-fixture ground
//! truth) and through `hf02`/`hf06`/`hf10` called directly with a
//! synthetic identity shift (`Os=0`), identity rotation (`Mr=I`), and
//! identity shuffle (`SS[i]=i+1`) to isolate the shuffle/segmentation logic
//! from the shift-rotate pipeline. This task's report reproduces the
//! driver/probe source and every command run.
//!
//! ### Dispatch: which C function each fid actually calls (verified, NOT
//! assumed from the internal function names -- see the naming curiosity
//! below)
//!
//! `cec22_test_func`'s `switch`, quoted: `case 6: hf02(&x[i*nx],&f[i],nx,
//! OShift,M,SS,1,1); f[i]+=1800.0; break; case 7: hf10(&x[i*nx],&f[i],nx,
//! OShift,M,SS,1,1); f[i]+=2000.0; break; case 8: hf06(&x[i*nx],&f[i],nx,
//! OShift,M,SS,1,1); f[i]+=2200.0; break;`. **Naming curiosity (cosmetic
//! only, verified to NOT affect output)**: the C function *names* don't
//! match the report's "Hybrid Function 1/2/3" numbering at all -- fid 6
//! ("Hybrid Function 1") calls a function literally named `hf02` (source
//! comment: `/* Hybrid Function 2 */`); fid 7 ("Hybrid Function 2") calls
//! `hf10` (comment: `/* Hybrid Function 6 */`); fid 8 ("Hybrid Function 3")
//! calls `hf06` (comment: `/* Hybrid Function 6 */`, duplicating fid 7's
//! stale comment). This reads as leftover internal numbering from a prior
//! year's function library that was never renamed to match CEC'22's own
//! table -- since the `switch` dispatches by `func_num` (not by name), it
//! does not change any computed value; recorded here only so a reader
//! diffing this module against the C source isn't confused by `hf02`
//! appearing under fid 6's arm despite the "Hybrid Function 1" label.
//!
//! ### Eq (21): the general hybrid formula (quoted) and Gp/segmentation
//! arithmetic (quoted) -- verified against the C's EXACT `ceil`-based
//! computation, not assumed
//!
//! `F(x) = g1(M1 z1) + g2(M2 z2) + ... + gN(MN zN) + F*(x)` (eq 21), `z =
//! [z1,...,zN]`, `z1 = [y_S1,...,y_Sn1]`, `z2 = [y_S(n1+1),...,y_S(n1+n2)]`,
//! ..., `y = x - o`, `S = randperm(1:D)`, `p_i` controls `g_i`'s share of
//! `D`, `n1 = ceil(p1*D), n2 = ceil(p2*D), ..., n(N-1) = ceil(p(N-1)*D), nN
//! = D - sum_{i=1}^{N-1} n_i`.
//!
//! `hf02`/`hf06`/`hf10`, quoted (identical shape in all three, `hf10`'s
//! `cf_num=6` shown): `G_nx[i] = ceil(Gp[i]*nx);` for `i` in `0..cf_num-1`,
//! `tmp += G_nx[i]`; then `G_nx[cf_num-1]=nx-tmp;` (the LAST segment absorbs
//! the remainder -- matches eq 21's `nN` exactly, confirmed identical C
//! logic in all three hybrid functions, not just described once and
//! assumed to generalize). `G[0]=0; G[i]=G[i-1]+G_nx[i-1];` (segment start
//! offsets). This module's [`Cec2022::segment_sizes`] /
//! [`Cec2022::segment_starts`] mirror this arithmetic exactly (same
//! `f64.ceil() as usize`, same last-segment-absorbs-remainder rule).
//!
//! **Report vs. code on rotation, eq (21)'s own notation**: eq (21) writes
//! `g_i(M_i z_i)`, suggesting each sub-function gets its OWN rotation
//! matrix `M_i`. The C does NOT do this: `hf02`/`hf06`/`hf10` call
//! `sr_func(x, z, nx, Os, Mr, 1.0, s_flag, r_flag)` exactly ONCE, at the
//! very top (`s_flag=1, r_flag=1` from the dispatch), producing ONE fully
//! shifted-AND-rotated `z` for the WHOLE `nx`-length vector; shuffling and
//! segmentation happen AFTER that single rotation, and every subsequent
//! per-component call passes `r_flag=0` (quoted below) -- so no component
//! is EVER separately rotated. This is also why only ONE `nx`x`nx` matrix
//! is vendored per `(fid, dim)` (`M_6_D10.txt` is 2520 bytes, the SAME size
//! as `M_1_D10.txt`, not `cf_num` times bigger as an `N`-separate-rotations
//! reading of eq (21) would require) -- confirmed by inspecting the
//! vendored file sizes directly (this task's report), not inferred. This
//! module's [`Cec2022::eval_one`] therefore applies [`Cec2022::shift_scale_rotate`]
//! (full shift+rotate, `sr=1.0`) exactly once per hybrid evaluation, before
//! shuffling.
//!
//! ### Shuffle: `y[i] = z[S[i]-1]` applied to the ROTATED-SHIFTED `z`
//! (verified, not assumed) -- 1-based file indices
//!
//! `hf02`, quoted: `sr_func (x, z, nx, Os, Mr, 1.0, s_flag, r_flag); for
//! (i=0; i<nx; i++) { y[i]=z[S[i]-1]; }` -- the shuffle reads `z`, `sr_func`'s
//! OUTPUT (fully shifted+rotated), confirming the report's informal
//! `y = x - o` (pre-eq-21 text) undersells its own pipeline: the vector
//! that actually gets shuffled and split into `z1..zN` (eq 21's notation)
//! is shift-AND-rotate applied, not shift-only. `S` is 1-based in the
//! vendored `shuffle_data_<fid>_D<dim>.txt` (`data.rs`'s
//! [`data::shuffle_indices`] converts to 0-based on load, module doc
//! there). This module's [`Cec2022::eval_one`] builds `y` the same way:
//! `y[i] = z[self.shuffle[i]]` (`self.shuffle` already 0-based).
//!
//! ### Per-component evaluation: NO shift, NO rotation, ONLY the
//! component's own inner scale (`sh_rate`) -- verified from `sr_func`'s
//! `s_flag==0` branch, quoted
//!
//! Every component call passes `s_flag=0, r_flag=0`, e.g. `hf02`: `i=0;
//! bent_cigar_func(&y[G[i]],&fit[i],G_nx[i],Os,Mr,0,0);`. Inside each base
//! function, e.g. `bent_cigar_func`: `sr_func (x, z, nx, Os, Mr, 1.0,
//! s_flag, r_flag);` -- `sr_func`'s body, quoted (the relevant branch):
//! `else { if (r_flag==1) {...} else for (i=0;i<nx;i++)
//! sr_x[i]=x[i]*sh_rate; }` -- with `s_flag=0` the FIRST `if(s_flag==1)`
//! branch is skipped entirely (`Os` never read), and with `r_flag=0` the
//! inner `if(r_flag==1)` is also skipped (`Mr` never read) -- so a
//! component's `x` parameter (`&y[G[i]]`, its assigned segment of the
//! ALREADY shift-rotated-shuffled `y`) is simply multiplied by that
//! FUNCTION's OWN `sh_rate` constant (the same constant it would use
//! standalone, e.g. `bent_cigar_func`'s is `1.0`, `hgbat_func`'s is
//! `5.0/100.0`) and nothing else. This module's component `eval`s
//! (`Cec2022::HybridComponent::eval`, below) take an already-scaled slice
//! for exactly this reason -- [`Cec2022::eval_one`]'s hybrid arm applies
//! `yi * comp.sh_rate()` per element before calling the component, mirroring
//! this exact branch.
//!
//! ### Hybrid composition tables (report section "B. Hybrid Functions",
//! quoted) -- cross-checked against the C's `Gp`/component-call arrays,
//! MATCHES for fid 6 and 8, ONE discrepancy for fid 7 (below)
//!
//! **fid 6, "Hybrid Function 1"**: `N=3`, `p=[0.4, 0.4, 0.2]`, `g1`: Bent
//! Cigar `f6`, `g2`: HGBat `f7`, `g3`: Rastrigin's `f4`. Matches `hf02`'s
//! `Gp={0.4,0.4,0.2}` and call order `bent_cigar_func, hgbat_func,
//! rastrigin_func` exactly.
//!
//! **fid 7, "Hybrid Function 2"**: `N=6`, printed `p=[0.1, 0.2, 0.2, 0.2,
//! 0.2, 0.1, 0.2]` (SEVEN values -- **DISCREPANCY**: one too many for
//! `N=6`, almost certainly a duplicated "0.2" typo in the PDF's list
//! typesetting), `g1`: HGBat `f7`, `g2`: Katsuura `f9`, `g3`: Ackley's
//! `f13`, `g4`: Rastrigin's `f4`, `g5`: Modified Schwefel's `f12`, `g6`:
//! Schaffer's F7 `f16`. `hf10`'s `Gp={0.1,0.2,0.2,0.2,0.1,0.2}` is
//! unambiguously SIX values (matching `cf_num=6`) and its call order is
//! `hgbat_func, katsuura_func, ackley_func, rastrigin_func, schwefel_func,
//! schaffer_F7_func` -- this module follows the CODE's six-value `Gp`
//! (dropping the report's apparent duplicate), per the milestone's
//! code-over-report ruling.
//!
//! **fid 8, "Hybrid Function 3"**: `N=5`, `p=[0.3, 0.2, 0.2, 0.1, 0.2]`,
//! `g1`: Katsuura `f9`, `g2`: HappyCat `f10`, `g3`: Expanded Griewank's plus
//! Rosenbrock's `f11`, `g4`: Modified Schwefel's `f12`, `g5`: Ackley's
//! `f13`. Matches `hf06`'s `Gp={0.3,0.2,0.2,0.1,0.2}` and call order
//! `katsuura_func, happycat_func, grie_rosen_func, schwefel_func,
//! ackley_func` exactly.
//!
//! ### New component basic functions (report section "A. Basic Functions",
//! `f6`-`f13`, quoted; equation numbers as printed) -- verified against
//! each one's C body, quoted where it clarifies an ambiguous printed glyph
//!
//! **f6) Bent Cigar Function** (eq 6): `f6(x) = x1^2 + 10^6 sum_{i=2}^{D}
//! xi^2`. `bent_cigar_func`, quoted: `f[0] = z[0]*z[0]; for (i=1;i<nx;i++)
//! f[0] += pow(10.0,6.0)*z[i]*z[i];` -- matches (0-based `i=1..nx-1` is
//! 1-based `i=2..D`).
//!
//! **f7) HGBat Function** (eq 7): `f7(x) = |(sum xi^2)^2 - (sum xi)^2|^0.5
//! + (0.5 sum xi^2 + sum xi)/D + 0.5`. `hgbat_func`, quoted: `alpha=1.0/4.0;
//!   ... f[0]=pow(fabs(pow(r2,2.0)-pow(sum_z,2.0)),2*alpha) + (0.5*r2 +
//!   sum_z)/nx + 0.5;` where `r2 = sum((z_i-1)^2)`, `sum_z = sum(z_i-1)` (the
//! `z[i]=z[i]-1.0` "shift to origin" loop runs BEFORE `r2`/`sum_z` are
//! accumulated) -- `2*alpha = 0.5` matches the printed exponent exactly.
//!
//! **f9) Katsuura Function** (eq 9): `f9(x) = (10/D^2) prod_{i=1}^{D} (1 +
//! i sum_{j=1}^{32} |2^j xi - round(2^j xi)| / 2^j)^{10/D^1.2} - 10/D^2`.
//! `katsuura_func`, quoted: `f[0]=1.0; tmp3=pow(1.0*nx,1.2); for(i..) {
//! temp=0.0; for(j=1;j<=32;j++) { tmp1=pow(2.0,j); tmp2=tmp1*z[i]; temp +=
//! fabs(tmp2-floor(tmp2+0.5))/tmp1; } f[0] *=
//! pow(1.0+(i+1)*temp,10.0/tmp3); } tmp1=10.0/nx/nx;
//! f[0]=f[0]*tmp1-tmp1;` -- `floor(tmp2+0.5)` is round-to-nearest, matching
//! the printed `round(...)`; `(i+1)` because the C loop is 0-based but the
//! formula's `i` multiplier is 1-based.
//!
//! **f10) HappyCat Function** (eq 10): `f10(x) = |sum xi^2 - D|^{0.25} +
//! (0.5 sum xi^2 + sum xi)/D + 0.5`. `happycat_func`, quoted:
//! `alpha=1.0/8.0; ... f[0]=pow(fabs(r2-nx),2*alpha) + (0.5*r2 +
//! sum_z)/nx + 0.5;`, same `z[i]=z[i]-1.0` shift-to-origin as HGBat --
//! `2*alpha=0.25` matches. (Note the exponent's ARGUMENT differs from
//! HGBat's: `r2-nx` here, a scalar difference, vs. HGBat's `r2^2-sum_z^2`.)
//!
//! **f11) Expanded Rosenbrock's plus Griewank's Function** (eq 11):
//! `f11(x) = f15(f2(x1,x2)) + f15(f2(x2,x3)) + ... + f15(f2(xD,x1))`
//! (CYCLIC composition of Griewank's `f15` applied to each consecutive
//! Rosenbrock `f2` pair). `grie_rosen_func`, quoted: (after the same
//! `z[i]+=1` "shift to origin" convention Rosenbrock's `f2` uses)
//! `tmp1=z[i]*z[i]-z[i+1]; tmp2=z[i]-1.0; temp=100.0*tmp1*tmp1+tmp2*tmp2;
//! f[0] += (temp*temp)/4000.0 - cos(temp) + 1.0;` per cyclic pair -- each
//! term is exactly Griewank's single-dimension form (`f15` at `i=0`:
//! `temp^2/4000 - cos(temp/sqrt(1)) + 1`) applied to the SCALAR Rosenbrock
//! value `temp` for that pair, matching the report's `f15(f2(...))`
//! composition precisely.
//!
//! **f12) Modified Schwefel's Function** (eq 12): `f12(x) = 418.9829*D -
//! sum g(zi)`, `zi = xi + 4.209687462275036e+02`, piecewise `g` (three
//! cases on `|zi|` vs. `500`, quoted in full in `schwefel_func` below --
//! IDENTICAL formula and constants to fid 4's already-implemented
//! `Self::f4_base`... no: this is a DIFFERENT function from fid 4's plain
//! Rastrigin; the "418.9829" constant match is with fid 4's UNRELATED
//! Schwefel name only by coincidence of both being classic benchmark
//! functions -- no code is shared with `f4_base`). `schwefel_func`, quoted
//! in full: `z[i] += 4.209687462275036e+002; if (z[i]>500) { f[0]
//! -=(500.0-fmod(z[i],500))*sin(pow(500.0-fmod(z[i],500),0.5));
//! tmp=(z[i]-500.0)/100; f[0]+= tmp*tmp/nx; } else if (z[i]<-500) { f[0]
//! -=(-500.0+fmod(fabs(z[i]),500))*sin(pow(500.0-fmod(fabs(z[i]),500),0.5));
//! tmp=(z[i]+500.0)/100; f[0]+= tmp*tmp/nx; } else f[0]
//! -=z[i]*sin(pow(fabs(z[i]),0.5)); ... f[0] +=4.189828872724338e+002*nx;`
//! -- matches the report's piecewise `g` exactly (the printed "418.9829" is
//! the report's own 4-decimal ROUNDING of the code's full-precision
//! `4.189828872724338e+002`, not a value discrepancy -- same convention as
//! T5's unflagged rounded-display constants).
//!
//! **f13) Ackley's Function** (eq 13): `f13(x) = -20 exp(-0.2 sqrt((1/D)
//! sum xi^2)) - exp((1/D) sum cos(2*pi*xi)) + 20 + e`. `ackley_func`,
//! quoted: `sum1 += z[i]*z[i]; sum2 += cos(2.0*PI*z[i]); ... sum1 =
//! -0.2*sqrt(sum1/nx); sum2 /= nx; f[0] = E - 20.0*exp(sum1) - exp(sum2) +
//! 20.0;` -- same formula, terms reordered (report's `+ e` at the end vs.
//! the code's `E -` at the start); `E` is the code's own `#define E
//! 2.7182818284590452353602874713526625`, i.e. the SAME constant.
//!
//! (`f4` Rastrigin's -- reused verbatim from `Self::f4_base`, fid 4's
//! plain-Rastrigin core, since `rastrigin_func`'s formula is byte-identical
//! to `step_rastrigin_func`'s dead-code-adjusted actual behavior already
//! implemented there. `f16` Schaffer's F7 -- reused from
//! `Self::f16_schaffer_f7_base`, fid 3's core, EXCEPT for fid 7's buggy
//! component call, next section.)
//!
//! ### VERIFIED DISCREPANCY (severe): fid 7's `schaffer_F7` hybrid
//! component NEVER reads its assigned segment -- the whole segment is dead
//! weight in the search space
//!
//! `schaffer_F7_func`, quoted (same function T5 already flagged as buggy
//! for standalone fid 3): `sr_func (x, z, nx, Os, Mr, 1.0, s_flag,
//! r_flag); for (i=0;i<nx-1;i++) { z[i]=pow(y[i]*y[i]+y[i+1]*y[i+1],0.5);
//! ... }` -- the loop reads the GLOBAL `y[i]`/`y[i+1]`, NOT the function's
//! OWN `x` parameter (which `hf10` passes as `&y[G[5]]`, this component's
//! correctly shuffled/assigned tail segment) and NOT `z` (what `sr_func`
//! just computed, itself a no-op copy here since `s_flag=0,r_flag=0` and
//! `sh_rate=1.0`). Because `y` is the SAME GLOBAL buffer `hf10` itself
//! populated via `y[i]=z[S[i]-1]` for `i` in `0..nx` (the FULL shuffled
//! vector, not a per-component-offset one), and the loop indexes it from
//! `i=0`, this component ALWAYS reads the first `G_nx[5]-1` adjacent pairs
//! of the WHOLE shuffled `y` -- i.e. (with `hf10`'s segment order `HGBat,
//! Katsuura, Ackley, Rastrigin, Schwefel, SchafferF7`) a PREFIX overlapping
//! the `HGBat`/`Katsuura` segments -- REGARDLESS of the pointer offset
//! `&y[G[5]]` it was actually called with.
//!
//! **Verified two independent ways** (this task's report has the full
//! transcripts): (1) the compiled reference, called through the real
//! `cec22_test_func` entry point with the REAL vendored data (dim=10,
//! fid=7): perturbing `x[8]` and `x[9]` alone (which land in the assigned
//! LAST shuffled segment, `G[5]=8, G_nx[5]=2`) by up to `1e6` changes the
//! output by exactly `0.0`; perturbing `x[0]` (the FIRST shuffled
//! position, HGBat's segment, also what the buggy loop reads) changes the
//! output substantially. (2) an isolated probe calling `hf10` directly
//! with a synthetic IDENTITY shift/rotation/shuffle (so `y == x` exactly,
//! removing rotation as a confound): the SAME result -- `x[8]`/`x[9]`
//! (dim=10) or `x[16..20]` (dim=20) are COMPLETELY INERT, any value at all,
//! `f` is bit-identical.
//!
//! **Consequence for sezgi's implementation**: this module replicates the
//! bug exactly (PROVENANCE-FIRST, the milestone's code-over-report ruling
//! -- the C code is the actual competition scorer) --
//! [`Cec2022::eval_one`]'s `HybridComponent::SchafferF7Buggy` arm evaluates
//! [`Cec2022::f16_schaffer_f7_base`] on `y[0..G_nx[5]]` (a PREFIX of the
//! WHOLE shuffled `y`, not `y[G[5]..G[5]+G_nx[5]]`), and the true last
//! segment `y[G[5]..]` is simply never read by fid 7's evaluator, matching
//! the reference exactly. Flagged here explicitly, in the same severity
//! class as F3/F4/F5, for T8's opfunu cross-check: an independent
//! implementation that instead honors the pointer offset `&y[G[5]]` "as
//! intended" will disagree with sezgi (and with the actual competition
//! scorer) on fid 7 for any `x` where the true last segment's coordinates
//! matter -- this is the discrepancy to adjudicate there, not a sezgi bug.
//! Only fid 7 is affected (`hf02`/`hf06`, fid 6/8, never call
//! `schaffer_F7_func` -- verified: neither's component list includes it).
//!
//! **A real-data corroboration, found while writing this module's tests**:
//! the vendored `M_7_D10.txt` is itself sparse enough for the bug to be
//! observable through the PUBLIC `x`-space API, no synthetic identity
//! needed -- `x = o7 + [10,0,0,...,0]` (perturbing ONLY the first
//! coordinate) rotates to a `z` whose only two nonzero entries (verified,
//! this task's report) land, after shuffling, in `y[8]` and `y[9]` -- fid
//! 7's dead LAST segment at dim=10 -- so `F7(o7 + [10,0,...,0])` is
//! EXACTLY `F7* = 2000.0` (verified against the compiled reference AND
//! this module's own test), even though the input is nowhere near the
//! optimum. This module's test suite documents this exact case
//! (`fid7_x0_alone_lands_entirely_in_the_dead_segment_dim10`) rather than
//! papering over it with a different perturbation choice.
//!
//! ### `dim == 2` rejected for fid 6-8 (module doc's earlier "Supported
//! dims" section already recorded the data finding; [`Cec2022::new`]
//! enforces it as [`Cec2022Error::HybridDim2Unsupported`])
//!
//! Mirrors the C's own blanket guard, quoted above: `if
//! (nx==2&&(func_num==6||func_num==7||func_num==8)) printf("\nError:  NOT
//! defined for D=2.\n");`. No `shuffle_data_{6,7,8}_D2.txt` is vendored
//! (confirmed absent upstream), so this module cannot even attempt dim=2
//! for these fids -- the explicit check gives a clear error instead of a
//! panic reaching into `data::shuffle_indices`.
//!
//! ### `x = o` pin (Step 2) for fid 6-8: why `F_i(o_i) == F_i*` EXACTLY,
//! dim=10 AND dim=20 -- derived per-component, then VERIFIED against the
//! compiled reference (not just derived by hand)
//!
//! At `x=o`: `z = M*(x-o)*1.0 = M*0 = 0` (the SAME all-zero argument as
//! fid 1-5's pin, module doc above) -- so `y[i] = z[S[i]] = 0` for EVERY
//! `i`, regardless of the shuffle permutation (shuffling an all-zero vector
//! yields an all-zero vector). Every segment `y[G[j]..G[j]+G_nx[j]]` is
//! therefore also all-zero, for every hybrid, every dim, every shuffle.
//! Per component (all inputs `0`, scaled by that component's `sh_rate`,
//! still exactly `0.0`):
//! - **Bent Cigar**: `0^2 + 10^6*sum(0^2) = 0` -- trivial.
//! - **HGBat**: shift-to-origin gives `z_i=-1` for all `i` (size `n`); `r2 =
//!   n`, `sum_z = -n`; `r2^2 - sum_z^2 = n^2 - n^2 = 0` EXACTLY (both are
//!   `pow(x,2.0)` of the SAME magnitude `n`, so bit-identical); `0^0.5=0`;
//!   remaining term `(0.5*n + (-n))/n = -0.5`; total `0 + (-0.5) + 0.5 = 0`
//!   -- exact, via cancellation, NOT because the inputs were trivially
//!   zero.
//! - **Rastrigin's**: `Self::f4_base` at all-zero: `0 - 10*cos(0) + 10 = 0`
//!   per term (`cos(0)=1.0` exact) -- trivial, same as fid 4's pin.
//! - **Katsuura**: inner `tmp2 = 2^j * 0 = 0`; `floor(0+0.5) = floor(0.5) =
//!   0` EXACTLY (`0.5` is exactly representable, `floor` of it is `0`
//!   exactly); `|0-0|/2^j = 0` for every `j` -- `f` stays `1.0` through the
//!   product loop, then `f*tmp1 - tmp1 = tmp1 - tmp1 = 0` EXACTLY (same
//!   `tmp1` value subtracted from itself).
//! - **Ackley's**: `sum1=0`, `sum2 = sum(cos(0)) = n`; `sum1' =
//!   -0.2*sqrt(0/n) = 0`; `sum2' = n/n = 1`; `f = E - 20*exp(0) - exp(1) +
//!   20 = E - 20 - exp(1.0) + 20 = E - exp(1.0)`. This is `0.0` EXACTLY
//!   only if the `E` literal and a live `exp(1.0)` call round to the
//!   IDENTICAL `f64` -- verified empirically (this task's report) rather
//!   than assumed: both the compiled C reference's `F(o)` AND this
//!   module's Rust test observe `E - 1f64.exp() == 0.0` bit-exact on the
//!   platform these tests run on (`std::f64::consts::E` vs. `1.0_f64.exp()`).
//! - **HappyCat**: same shift-to-origin as HGBat, `r2=n`, `sum_z=-n`; but
//!   the exponent's ARGUMENT here is the scalar `r2-nx = n-n = 0` (not
//!   HGBat's squared-difference) -- `|0|^0.25=0`; remaining term identical
//!   to HGBat's, `-0.5`; total `0 + (-0.5) + 0.5 = 0` -- exact.
//! - **Griewank's+Rosenbrock's**: shift-to-origin gives `z_i=1` for all
//!   `i`; every cyclic pair: `tmp1 = 1-1=0`, `tmp2=1-1=0`, `temp=0`;
//!   `temp^2/4000 - cos(0) + 1 = 0 - 1 + 1 = 0` per term -- exact.
//! - **Modified Schwefel's**: `zi = 0 + 4.209687462275036e+02` exactly
//!   (same `f64` literal both directions), which is `<= 500`, so the
//!   `else` branch runs: `f -= zi*sin(sqrt(zi))` per term, then `f +=
//!   4.189828872724338e+02 * n`. Unlike the other seven components, this
//!   one is NOT obviously `0` from algebra alone (it depends on `sin`/
//!   `sqrt` of a specific non-round `f64`, cancelled against a SEPARATE
//!   truncated-decimal constant) -- so this module does NOT claim an
//!   algebraic proof; instead, VERIFIED EMPIRICALLY (this task's report):
//!   the compiled C reference's `F(o)` for fid 7 AND fid 8 (both include
//!   this component) returns EXACTLY `2000.0`/`2200.0` at dim=10 AND
//!   dim=20 (`%.20f`-formatted, all-zero fractional digits), meaning the
//!   two `f64` constants were evidently chosen (or happen) to cancel
//!   bit-exactly for this specific per-element value -- confirmed
//!   independently by this module's own Rust `x=o` pin test producing the
//!   SAME exact result with the SAME embedded data.
//! - **Schaffer's F7 (fid 7's buggy component)**: regardless of WHICH
//!   prefix of `y` it reads (previous section's discrepancy), that prefix
//!   is STILL all-zero at `x=o` (every entry of `y` is `0`), so
//!   `Self::f16_schaffer_f7_base` returns `0` exactly the same way fid 3's
//!   pin does -- the bug does not disturb the `x=o` pin, only general
//!   points.
//!
//! Summing eight (fid 7) or five (fid 6: three; fid 8: five) exact `0.0`
//! components gives `F_i(o_i) = 0 + F_i* = F_i*` EXACTLY -- confirmed by
//! BOTH the algebraic derivation above AND the compiled C reference AND
//! this module's own tests, for every `(fid, dim)` in `{6,7,8} x {10,20}`.
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
    #[error(
        "fid {0} (a hybrid function, fid 6-8) is not defined for dim=2 -- mirrors the official \
         C reference's own guard (\"nx==2&&(func_num==6||func_num==7||func_num==8)\") and the \
         vendored data (no shuffle_data_{{6,7,8}}_D2.txt exists upstream); use dim 10 or 20"
    )]
    HybridDim2Unsupported(u32),
}

/// One instance of a CEC 2022 basic (fid 1-5) or hybrid (fid 6-8) function:
/// embedded official shift vector `o` and rotation matrix `M` for the
/// requested `(fid, dim)`, plus (fid 6-8 only) the shuffle permutation
/// `shuffle` (0-based, empty for fid 1-5), plus the pinned `F_i*` bias
/// (module doc, section 1.2's table). See the module doc for each fid's
/// exact shift/scale/rotate(/shuffle/segment, fid 6-8) pipeline, including
/// the verified discrepancies (fid 3, 4, 5, and fid 7's dead hybrid
/// segment) between the report's printed equations and the vendored
/// reference C code this module actually follows.
pub struct Cec2022 {
    fid: u32,
    dim: usize,
    o: Vec<f64>,
    m: Vec<Vec<f64>>,
    shuffle: Vec<usize>,
    space: SearchSpace,
}

/// One hybrid function's sub-component (module doc's "Per-component
/// evaluation" section): pairs the base function with the `sh_rate`
/// constant the C reference's OWN per-component `sr_func` call uses
/// (`s_flag=0,r_flag=0` branch -- pure scale, no shift, no rotate).
/// `SchafferF7Buggy` is fid 7's verified-buggy last component (module doc's
/// dedicated discrepancy section) -- it does not use `sh_rate` at all (its
/// C counterpart's `sr_func` output is computed but never read; the loop
/// reads the unscaled global `y` directly).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HybridComponent {
    BentCigar,
    HGBat,
    Rastrigin,
    Katsuura,
    Ackley,
    Schwefel,
    HappyCat,
    GrieRosen,
    SchafferF7Buggy,
}

impl HybridComponent {
    /// This component's own inner scale constant (module doc: the SAME
    /// constant it would use standalone).
    fn sh_rate(self) -> f64 {
        match self {
            Self::BentCigar => 1.0,
            Self::HGBat => 5.0 / 100.0,
            Self::Rastrigin => 5.12 / 100.0,
            Self::Katsuura => 5.0 / 100.0,
            Self::Ackley => 1.0,
            Self::Schwefel => 1000.0 / 100.0,
            Self::HappyCat => 5.0 / 100.0,
            Self::GrieRosen => 5.0 / 100.0,
            Self::SchafferF7Buggy => 1.0,
        }
    }

    /// Evaluate this component's base function on an ALREADY-scaled
    /// segment (caller applies `sh_rate`, module doc). Never called for
    /// `SchafferF7Buggy` -- that arm is handled directly in
    /// [`Cec2022::hybrid_fitness`] since it reads a DIFFERENT slice of `y`
    /// entirely (the verified discrepancy), not this component's own
    /// scaled segment.
    fn eval(self, seg: &[f64]) -> f64 {
        match self {
            Self::BentCigar => Cec2022::bent_cigar_base(seg),
            Self::HGBat => Cec2022::hgbat_base(seg),
            Self::Rastrigin => Cec2022::f4_base(seg),
            Self::Katsuura => Cec2022::katsuura_base(seg),
            Self::Ackley => Cec2022::ackley_base(seg),
            Self::Schwefel => Cec2022::schwefel_base(seg),
            Self::HappyCat => Cec2022::happycat_base(seg),
            Self::GrieRosen => Cec2022::grie_rosen_base(seg),
            Self::SchafferF7Buggy => {
                unreachable!("SchafferF7Buggy is handled directly in hybrid_fitness")
            }
        }
    }
}

impl Cec2022 {
    /// `fid` in `1..=8` (module doc: fid 9-12, the composition functions,
    /// are [`Cec2022Error::NotImplemented`], deferred to T7; fid outside
    /// `1..=12` entirely is [`Cec2022Error::UnknownFid`]), `dim` in
    /// `{2,10,20}` for fid 1-5, `{10,20}` for fid 6-8 (module doc's T6
    /// addendum: `dim=2` is [`Cec2022Error::HybridDim2Unsupported`] for fid
    /// 6-8, mirroring the C reference's own guard).
    pub fn new(fid: u32, dim: usize) -> Result<Cec2022, Cec2022Error> {
        if !(1..=12).contains(&fid) {
            return Err(Cec2022Error::UnknownFid(fid));
        }
        if !(1..=8).contains(&fid) {
            return Err(Cec2022Error::NotImplemented(fid));
        }
        if !matches!(dim, 2 | 10 | 20) {
            return Err(Cec2022Error::BadDim(dim));
        }
        if (6..=8).contains(&fid) && dim == 2 {
            return Err(Cec2022Error::HybridDim2Unsupported(fid));
        }
        let o = data::shift_vector(fid, dim);
        let m = data::rotation_matrix(fid, dim);
        let shuffle =
            if (6..=8).contains(&fid) { data::shuffle_indices(fid, dim) } else { Vec::new() };
        let space = SearchSpace::new(vec![Block::Float { lo: -100.0, hi: 100.0, n: dim }])
            .expect("CEC 2022 bounds (-100 < 100) are always valid");
        Ok(Self { fid, dim, o, m, shuffle, space })
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
            6 => 1800.0,
            7 => 2000.0,
            8 => 2200.0,
            other => unreachable!("Cec2022::new rejects fid outside 1..=8, got {other}"),
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

    /// Basic function 6 (module doc, "New component basic functions"):
    /// Bent Cigar.
    fn bent_cigar_base(z: &[f64]) -> f64 {
        z[0] * z[0] + z[1..].iter().map(|&zi| 1.0e6 * zi * zi).sum::<f64>()
    }

    /// Basic function 7 (module doc): HGBat.
    fn hgbat_base(z: &[f64]) -> f64 {
        const ALPHA: f64 = 1.0 / 4.0;
        let n = z.len() as f64;
        let zs: Vec<f64> = z.iter().map(|&zi| zi - 1.0).collect();
        let r2: f64 = zs.iter().map(|&zi| zi * zi).sum();
        let sum_z: f64 = zs.iter().sum();
        (r2.powf(2.0) - sum_z.powf(2.0)).abs().powf(2.0 * ALPHA) + (0.5 * r2 + sum_z) / n + 0.5
    }

    /// Basic function 9 (module doc): Katsuura.
    fn katsuura_base(z: &[f64]) -> f64 {
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

    /// Basic function 13 (module doc): Ackley's.
    fn ackley_base(z: &[f64]) -> f64 {
        let n = z.len() as f64;
        let sum1: f64 = z.iter().map(|&zi| zi * zi).sum();
        let sum2: f64 = z.iter().map(|&zi| (2.0 * std::f64::consts::PI * zi).cos()).sum();
        let sum1 = -0.2 * (sum1 / n).sqrt();
        let sum2 = sum2 / n;
        std::f64::consts::E - 20.0 * sum1.exp() - sum2.exp() + 20.0
    }

    /// Basic function 12 (module doc): Modified Schwefel's -- written to
    /// mirror `schwefel_func`'s three branches literally (module doc's
    /// quote), not algebraically simplified, so it stays diffable against
    /// the quoted C.
    fn schwefel_base(z: &[f64]) -> f64 {
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

    /// Basic function 10 (module doc): HappyCat.
    fn happycat_base(z: &[f64]) -> f64 {
        const ALPHA: f64 = 1.0 / 8.0;
        let n = z.len() as f64;
        let zs: Vec<f64> = z.iter().map(|&zi| zi - 1.0).collect();
        let r2: f64 = zs.iter().map(|&zi| zi * zi).sum();
        let sum_z: f64 = zs.iter().sum();
        (r2 - n).abs().powf(2.0 * ALPHA) + (0.5 * r2 + sum_z) / n + 0.5
    }

    /// Basic function 11 (module doc): Expanded Griewank's plus
    /// Rosenbrock's (CYCLIC).
    fn grie_rosen_base(z: &[f64]) -> f64 {
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

    /// `fid`'s (`6..=8`) proportions `p` and component list, in call order
    /// (module doc's hybrid composition tables, cross-checked against the
    /// C's `Gp`/component-call arrays -- fid 7 follows the CODE's six-value
    /// `Gp`, module doc's discrepancy note on the report's seven-value
    /// printed array).
    fn hybrid_spec(fid: u32) -> (&'static [f64], &'static [HybridComponent]) {
        use HybridComponent::*;
        match fid {
            6 => (&[0.4, 0.4, 0.2], &[BentCigar, HGBat, Rastrigin]),
            7 => (
                &[0.1, 0.2, 0.2, 0.2, 0.1, 0.2],
                &[HGBat, Katsuura, Ackley, Rastrigin, Schwefel, SchafferF7Buggy],
            ),
            8 => (&[0.3, 0.2, 0.2, 0.1, 0.2], &[Katsuura, HappyCat, GrieRosen, Schwefel, Ackley]),
            other => unreachable!("hybrid_spec called with unsupported fid {other}"),
        }
    }

    /// `G_nx` (module doc's Eq (21)/`hf02` quote): `ceil(p_i*dim)` for every
    /// proportion but the last, which absorbs the remainder
    /// (`dim - sum(the rest)`) so the segments always sum to exactly `dim`.
    fn segment_sizes(gp: &[f64], dim: usize) -> Vec<usize> {
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

    /// `G` (module doc's Eq (21)/`hf02` quote): cumulative segment start
    /// offsets from `sizes`.
    fn segment_starts(sizes: &[usize]) -> Vec<usize> {
        let mut starts = vec![0usize; sizes.len()];
        for i in 1..sizes.len() {
            starts[i] = starts[i - 1] + sizes[i - 1];
        }
        starts
    }

    /// One hybrid function's component-sum (BEFORE `F_i*`), given the
    /// already shift-rotate-shuffled `y` (module doc: `y[i] = z[S[i]]`, `z`
    /// from [`Self::shift_scale_rotate`]). A separate method (not inlined
    /// into [`Self::eval_one`]) so Step 2's tests can probe the
    /// segmentation/shuffle/dead-segment logic directly on a hand-built
    /// `y`, independent of the shift/rotate/embedded-data plumbing.
    fn hybrid_fitness(fid: u32, dim: usize, y: &[f64]) -> f64 {
        let (gp, comps) = Self::hybrid_spec(fid);
        let sizes = Self::segment_sizes(gp, dim);
        let starts = Self::segment_starts(&sizes);
        comps
            .iter()
            .enumerate()
            .map(|(idx, &comp)| {
                let start = starts[idx];
                let len = sizes[idx];
                if comp == HybridComponent::SchafferF7Buggy {
                    // Module doc's verified discrepancy: reads a PREFIX of
                    // the WHOLE shuffled `y` (`y[0..len]`), NOT this
                    // component's own assigned segment `y[start..start+len]`
                    // -- replicates the reference C bug exactly.
                    Self::f16_schaffer_f7_base(&y[0..len])
                } else {
                    let seg: Vec<f64> =
                        y[start..start + len].iter().map(|&yi| yi * comp.sh_rate()).collect();
                    comp.eval(&seg)
                }
            })
            .sum()
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
            6..=8 => {
                // Full shift+rotate ONCE (module doc: eq (21)'s per-component
                // `M_i` is NOT what the C does), then shuffle, then segment +
                // per-component scale-only evaluation (module doc's fid 6-8
                // section, including fid 7's verified dead-segment bug).
                let z = self.shift_scale_rotate(xs, 1.0, true);
                let y: Vec<f64> = (0..self.dim).map(|i| z[self.shuffle[i]]).collect();
                Self::hybrid_fitness(self.fid, self.dim, &y) + self.f_star()
            }
            other => unreachable!("Cec2022::new rejects fid outside 1..=8, got {other}"),
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
    fn fid_9_to_12_are_not_implemented_yet() {
        // T6 implements fid 6-8 (the hybrid functions); fid 9-12 (the
        // composition functions) remain deferred to T7 -- this boundary
        // moved from 6..=12 (T5) to 9..=12 (this task).
        for fid in 9u32..=12 {
            assert!(
                matches!(Cec2022::new(fid, 10), Err(Cec2022Error::NotImplemented(f)) if f == fid),
                "fid={fid}"
            );
        }
    }

    #[test]
    fn dim_5_is_bad_dim_for_every_implemented_fid() {
        for fid in 1u32..=8 {
            assert!(matches!(Cec2022::new(fid, 5), Err(Cec2022Error::BadDim(5))), "fid={fid}");
        }
    }

    #[test]
    fn dim_2_is_rejected_for_fid_6_to_8_with_the_hybrid_specific_error() {
        // Module doc's T6 addendum: mirrors the C's own blanket guard, NOT
        // the generic BadDim (dim=2 IS otherwise a supported dim).
        for fid in 6u32..=8 {
            assert!(
                matches!(Cec2022::new(fid, 2), Err(Cec2022Error::HybridDim2Unsupported(f)) if f == fid),
                "fid={fid}"
            );
        }
    }

    #[test]
    fn space_is_single_float_block_pm100() {
        for fid in 1u32..=5 {
            let p = Cec2022::new(fid, 10).unwrap();
            assert_eq!(p.space().blocks(), &[Block::Float { lo: -100.0, hi: 100.0, n: 10 }], "fid={fid}");
            assert_eq!(p.space().dim(), 10);
        }
        for fid in 6u32..=8 {
            for &dim in &[10usize, 20] {
                let p = Cec2022::new(fid, dim).unwrap();
                assert_eq!(
                    p.space().blocks(),
                    &[Block::Float { lo: -100.0, hi: 100.0, n: dim }],
                    "fid={fid} dim={dim}"
                );
                assert_eq!(p.space().dim(), dim);
            }
        }
    }

    #[test]
    fn f_star_matches_report_table() {
        let expect =
            [(1u32, 300.0), (2, 400.0), (3, 600.0), (4, 800.0), (5, 900.0), (6, 1800.0), (7, 2000.0), (8, 2200.0)];
        for (fid, fstar) in expect {
            assert_eq!(Cec2022::new(fid, 10).unwrap().f_star(), fstar, "fid={fid}");
        }
    }

    // ---- construction sweep: fid 6-8 x {10,20} ----

    #[test]
    fn construction_sweep_fid_6_to_8_dims_10_20() {
        for fid in 6u32..=8 {
            for &dim in &[10usize, 20] {
                let p = Cec2022::new(fid, dim).unwrap();
                assert_eq!(p.fid(), fid);
                assert_eq!(p.dim(), dim);
                assert_eq!(p.shuffle.len(), dim, "fid={fid} dim={dim}");
            }
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

    // ---- x = o pin for fid 6-8 (module doc's per-component derivation),
    // dim=10 AND dim=20 -- exact equality, cross-checked against the
    // compiled C reference in this task's report. ----

    #[test]
    fn x_equals_o_pins_f_star_exactly_hybrid_dim10_and_dim20() {
        for fid in 6u32..=8 {
            for &dim in &[10usize, 20] {
                let p = Cec2022::new(fid, dim).unwrap();
                let out = p.evaluate_batch(&[g(p.o.clone())]);
                assert_eq!(out, vec![p.f_star()], "fid={fid} dim={dim}: F(o) must equal F* EXACTLY");
            }
        }
    }

    // ---- ackley's E vs. exp(1.0) exactness assumption the x=o pin's
    // derivation relies on (module doc): confirm it holds on this platform. ----

    #[test]
    fn ackley_e_constant_matches_live_exp_of_one_bit_exact() {
        assert_eq!(std::f64::consts::E, 1.0_f64.exp());
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

    // ---- full sweep: fid 6..=8 x dims {10,20} (module doc: dim=2 is
    // unsupported for these fids), one evaluation, no panic ----

    #[test]
    fn full_sweep_fid_6_to_8_dims_10_20_evaluates_without_panic() {
        for fid in 6u32..=8 {
            for &dim in &[10usize, 20] {
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

    // ---- fid 6-8 need a DIFFERENT perturbation than fid 1-5's single-
    // coordinate `x[0] += 10.0`: fid 7's rotation matrix `M_7_D10` is
    // (verified, this task's report) column-sparse enough that `x[0]`'s
    // rotated contribution lands ENTIRELY in the last shuffled segment --
    // fid 7's verified dead segment (module doc) -- so `x[0] += 10.0` alone
    // leaves fid 7 EXACTLY at `F*` (cross-checked against the compiled C
    // reference, not just this Rust code). Perturbing every coordinate
    // guarantees at least one LIVE segment is hit, for every fid. ----

    #[test]
    fn away_from_optimum_is_worse_for_every_hybrid_fid() {
        for fid in 6u32..=8 {
            let p = Cec2022::new(fid, 10).unwrap();
            let x: Vec<f64> = p.o.iter().map(|&oi| oi + 10.0).collect();
            let out = p.evaluate_batch(&[g(x)])[0];
            assert!(out > p.f_star(), "fid={fid}: {out} not > {}", p.f_star());
        }
    }

    #[test]
    fn fid7_x0_alone_lands_entirely_in_the_dead_segment_dim10() {
        // Direct regression, through the PUBLIC evaluate_batch API (not
        // just the private hybrid_fitness probe above), for the exact
        // discrepancy this task's report documents: verified against the
        // compiled C reference to be EXACTLY F* (2000.0), not approximately.
        let p = Cec2022::new(7, 10).unwrap();
        let mut x = p.o.clone();
        x[0] += 10.0;
        let out = p.evaluate_batch(&[g(x)])[0];
        assert_eq!(out, p.f_star(), "{out}");
    }

    // ---- T6: hybrid functions f6-f8 ----

    // ---- segmentation arithmetic (module doc's Eq (21)/`hf02` quote):
    // G_nx[i] = ceil(p_i*dim) for i < N-1, last absorbs the remainder.
    // Hand-computed here (transcribed into the assertions) so the test
    // itself is the derivation, not just a check against the same formula
    // the implementation uses. ----

    #[test]
    fn segment_sizes_match_hand_computed_ceil_arithmetic() {
        // fid 6, Gp=[0.4,0.4,0.2], N=3:
        //   dim=10: ceil(0.4*10)=4, ceil(0.4*10)=4, tmp=8, last=10-8=2 -> [4,4,2]
        //   dim=20: ceil(0.4*20)=8, ceil(0.4*20)=8, tmp=16, last=20-16=4 -> [8,8,4]
        assert_eq!(Cec2022::segment_sizes(&[0.4, 0.4, 0.2], 10), vec![4, 4, 2]);
        assert_eq!(Cec2022::segment_sizes(&[0.4, 0.4, 0.2], 20), vec![8, 8, 4]);

        // fid 7, Gp=[0.1,0.2,0.2,0.2,0.1,0.2], N=6:
        //   dim=10: ceil(1)=1, ceil(2)=2, ceil(2)=2, ceil(2)=2, ceil(1)=1,
        //           tmp=1+2+2+2+1=8, last=10-8=2 -> [1,2,2,2,1,2]
        //   dim=20: ceil(2)=2, ceil(4)=4, ceil(4)=4, ceil(4)=4, ceil(2)=2,
        //           tmp=2+4+4+4+2=16, last=20-16=4 -> [2,4,4,4,2,4]
        let gp7 = [0.1, 0.2, 0.2, 0.2, 0.1, 0.2];
        assert_eq!(Cec2022::segment_sizes(&gp7, 10), vec![1, 2, 2, 2, 1, 2]);
        assert_eq!(Cec2022::segment_sizes(&gp7, 20), vec![2, 4, 4, 4, 2, 4]);

        // fid 8, Gp=[0.3,0.2,0.2,0.1,0.2], N=5:
        //   dim=10: ceil(3)=3, ceil(2)=2, ceil(2)=2, ceil(1)=1,
        //           tmp=3+2+2+1=8, last=10-8=2 -> [3,2,2,1,2]
        //   dim=20: ceil(6)=6, ceil(4)=4, ceil(4)=4, ceil(2)=2,
        //           tmp=6+4+4+2=16, last=20-16=4 -> [6,4,4,2,4]
        let gp8 = [0.3, 0.2, 0.2, 0.1, 0.2];
        assert_eq!(Cec2022::segment_sizes(&gp8, 10), vec![3, 2, 2, 1, 2]);
        assert_eq!(Cec2022::segment_sizes(&gp8, 20), vec![6, 4, 4, 2, 4]);

        // Every fid's sizes sum to exactly `dim`, both dims.
        for (gp, n) in [(&[0.4, 0.4, 0.2][..], 3), (&gp7[..], 6), (&gp8[..], 5)] {
            for &dim in &[10usize, 20] {
                let sizes = Cec2022::segment_sizes(gp, dim);
                assert_eq!(sizes.len(), n);
                assert_eq!(sizes.iter().sum::<usize>(), dim, "gp={gp:?} dim={dim}");
            }
        }
    }

    #[test]
    fn segment_starts_are_cumulative_sums_of_sizes() {
        assert_eq!(Cec2022::segment_starts(&[4, 4, 2]), vec![0, 4, 8]);
        assert_eq!(Cec2022::segment_starts(&[1, 2, 2, 2, 1, 2]), vec![0, 1, 3, 5, 7, 8]);
        assert_eq!(Cec2022::segment_starts(&[3, 2, 2, 1, 2]), vec![0, 3, 5, 7, 8]);
    }

    // ---- shuffle-application unit check (brief, Step 2): a hand-made probe
    // vector plus the EMBEDDED shuffle data (read directly, not through
    // `Cec2022::new`), asserting the shuffled/segmented coordinates land in
    // the pinned per-component segments. ----

    #[test]
    fn shuffle_application_lands_coordinates_in_the_pinned_segments_fid6_dim10() {
        // Probe vector z[i] = i as f64 (index-as-value) makes the shuffle's
        // effect trivially traceable: y[i] = z[S[i]] = S[i] as f64 exactly.
        let dim = 10;
        let shuffle = super::data::shuffle_indices(6, dim); // embedded data, read directly
        assert_eq!(shuffle, vec![3, 6, 8, 2, 4, 1, 9, 7, 5, 0]); // 0-based, spot-checked in data.rs too
        let z: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..dim).map(|i| z[shuffle[i]]).collect();
        assert_eq!(y, vec![3.0, 6.0, 8.0, 2.0, 4.0, 1.0, 9.0, 7.0, 5.0, 0.0]);

        // fid 6 segmentation at dim=10: sizes=[4,4,2], starts=[0,4,8]
        // (segment_sizes test above) -- bent_cigar gets y[0..4], hgbat
        // y[4..8], rastrigin y[8..10].
        let bent_cigar_seg = &y[0..4];
        let hgbat_seg = &y[4..8];
        let rastrigin_seg = &y[8..10];
        assert_eq!(bent_cigar_seg, &[3.0, 6.0, 8.0, 2.0]);
        assert_eq!(hgbat_seg, &[4.0, 1.0, 9.0, 7.0]);
        assert_eq!(rastrigin_seg, &[5.0, 0.0]);
        // Every original index 0..dim appears in EXACTLY one segment
        // (segments partition the shuffled vector).
        let mut all: Vec<f64> =
            bent_cigar_seg.iter().chain(hgbat_seg).chain(rastrigin_seg).copied().collect();
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(all, (0..dim).map(|i| i as f64).collect::<Vec<_>>());
    }

    #[test]
    fn shuffle_indices_used_by_eval_one_match_embedded_data_for_every_fid_6_to_8_dim() {
        // Cross-check the field `Cec2022::new` populates against the same
        // embedded-data parser `data.rs`'s own tests spot-check directly --
        // confirms the two aren't drifting apart.
        for fid in 6u32..=8 {
            for &dim in &[10usize, 20] {
                let p = Cec2022::new(fid, dim).unwrap();
                assert_eq!(p.shuffle, super::data::shuffle_indices(fid, dim), "fid={fid} dim={dim}");
            }
        }
    }

    // ---- fid 7's verified dead-segment discrepancy (module doc): the last
    // shuffled segment (schaffer_F7's assigned block) never affects the
    // output, at ANY dim. Probed directly on `hybrid_fitness` (bypassing
    // shift/rotate) so the regression is about the segmentation/shuffle
    // logic itself, not confounded by the top-level rotation mixing every
    // coordinate together (module doc explains why testing this through
    // the public `x`-space API would be confounded). ----

    #[test]
    fn fid7_last_shuffled_segment_is_dead_due_to_verified_schaffer_f7_hybrid_bug() {
        for &dim in &[10usize, 20] {
            let y: Vec<f64> = (1..=dim).map(|i| i as f64).collect();
            let base = Cec2022::hybrid_fitness(7, dim, &y);

            // fid 7's last segment (schaffer_F7, index 5): starts at
            // sum(sizes[0..5]) = dim - sizes[5] (segment_sizes test above:
            // sizes[5] = 2 at dim=10, 4 at dim=20).
            let sizes = Cec2022::segment_sizes(&[0.1, 0.2, 0.2, 0.2, 0.1, 0.2], dim);
            let last_len = sizes[5];
            let start = dim - last_len;

            let mut perturbed = y.clone();
            for v in &mut perturbed[start..] {
                *v = 987_654.321; // arbitrary, large -- must not matter
            }
            let out = Cec2022::hybrid_fitness(7, dim, &perturbed);
            assert_eq!(base, out, "dim={dim}: fid 7's last segment must be dead (verified C bug)");
        }
    }

    #[test]
    fn fid6_and_fid8_have_no_dead_segment_every_coordinate_matters() {
        // Contrast case: neither fid 6 nor fid 8 uses the buggy
        // SchafferF7Buggy component (module doc), so perturbing ANY
        // segment, including the last, must change the output.
        for (fid, gp) in [(6u32, &[0.4, 0.4, 0.2][..]), (8, &[0.3, 0.2, 0.2, 0.1, 0.2][..])] {
            for &dim in &[10usize, 20] {
                let y: Vec<f64> = (1..=dim).map(|i| i as f64).collect();
                let base = Cec2022::hybrid_fitness(fid, dim, &y);
                let sizes = Cec2022::segment_sizes(gp, dim);
                let last_len = *sizes.last().unwrap();
                let start = dim - last_len;
                let mut perturbed = y.clone();
                for v in &mut perturbed[start..] {
                    *v = 987_654.321;
                }
                let out = Cec2022::hybrid_fitness(fid, dim, &perturbed);
                assert_ne!(out, base, "fid={fid} dim={dim}: last segment must NOT be dead");
            }
        }
    }

    // ---- hand fixture: f6, dim=10, x = o + delta (a simple non-o point).
    // Full pipeline (shift -> rotate -> shuffle -> segment -> per-component
    // scale+base -> sum -> +F*) computed independently in Python outside
    // this crate, reading the SAME vendored data files directly (not the
    // Rust parser) -- this task's report has the script and its output.
    // Cross-checked against the COMPILED official C reference
    // (`cec22_test_func`, unmodified) called with the SAME `x`, matching
    // to the digits asserted below. ----

    #[test]
    fn f6_hybrid_hand_fixture_dim10() {
        // delta = [0.5,0.3,0.2,0.1,0.4,0.2,0.1,0.4,0.3,0.5], x = o6 + delta.
        // Pipeline stages (dim=10, o6/M_6_D10/shuffle_data_6_D10 = the
        // vendored files):
        //   shifted = delta (shift subtracts o exactly)
        //   z = M_6_D10 * shifted (full rotation)
        //   y[i] = z[S[i]] (S = shuffle_indices(6,10) = [3,6,8,2,4,1,9,7,5,0])
        //   segments (sizes [4,4,2], starts [0,4,8]):
        //     bent_cigar_seg = y[0..4], sh_rate=1.0 -> f0 = 132609.1748847605
        //     hgbat_seg      = y[4..8], sh_rate=0.05 -> f1 = 0.2661460749418409
        //     rastrigin_seg  = y[8..10], sh_rate=0.0512 -> f2 = 0.15891343303704986
        //   total = f0+f1+f2 = 132609.59994426847
        //   F6 = total + 1800.0 = 134409.59994426847
        // (compiled C reference, SAME x: 134409.59994426846969872713 --
        // matches to double precision.)
        let p = Cec2022::new(6, 10).unwrap();
        let delta = [0.5, 0.3, 0.2, 0.1, 0.4, 0.2, 0.1, 0.4, 0.3, 0.5];
        let x: Vec<f64> = p.o.iter().zip(delta).map(|(&oi, d)| oi + d).collect();
        let out = p.evaluate_batch(&[g(x)])[0];
        assert!((out - 134_409.599_944_268_47).abs() < 1e-6, "{out}");
    }
}
