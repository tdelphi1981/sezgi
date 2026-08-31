//! The CEC 2017 Special Session and Competition benchmark suite: fid 1
//! (Bent Cigar), fid 3-10 (Zakharov, Rosenbrock, Rastrigin, Schaffer's F7,
//! Lunacek bi-Rastrigin, Non-Continuous Rastrigin, Levy, Modified Schwefel
//! -- M3-6 T6), fid 11-20 (hybrid functions -- M3-6 T7), and fid 21-30
//! (composition functions -- M3-6 T8, THIS TASK: the suite's own `cf01`
//! .."cf10"). Fid 2 is PERMANENTLY WITHDRAWN (module doc's "F2 ruling"
//! section below); `Cec2017::new(2, _)` returns a dedicated
//! [`Cec2017Error::Withdrawn`] error, distinct from
//! [`Cec2017Error::UnknownFid`]. `Cec2017::new` now covers the suite's FULL
//! usable range, `fid` in `{1} ∪ {3..=30}` (mirrors `cec2014`'s own
//! T2 -> T3 -> T4 staging, now complete for this suite too).
//!
//! Source (PROVENANCE, fetched and read directly, not from memory): N. H.
//! Awad, M. Z. Ali, P. N. Suganthan, J. J. Liang, B. Y. Qu, "Problem
//! Definitions and Evaluation Criteria for the CEC 2017 Special Session and
//! Competition on Single Objective Real-Parameter Numerical Optimization",
//! Technical Report, Nanyang Technological University, Singapore / Jordan
//! University of Science and Technology / Zhengzhou University, "Modified
//! on October 15th 2016". PDF fetched directly from the authors' own
//! official repository:
//! `https://raw.githubusercontent.com/P-N-Suganthan/CEC2017-BoundContrained/master/Definitions%20of%20%20CEC2017%20benchmark%20suite%20final%20version%20updated.pdf`
//! (SHA-256 `b69f52f047f6bca888787ac19f3f1224293000c0a2446bdd5c355795efa9684a`,
//! git blob SHA `53b532c3bf4179de98f01f086d54660babc0694f`, size 2,399,144
//! bytes; every quote below is transcribed from that fetched PDF via
//! `pdftotext -layout`). **Repo-name caution (verified, real, not a typo of
//! this doc)**: the correct repository is
//! `github.com/P-N-Suganthan/CEC2017-BoundContrained` -- the plain
//! `P-N-Suganthan/CEC2017` repo is a DIFFERENT, unrelated CEC'17
//! *constrained*-optimization competition, not this bound-constrained
//! suite; not fetched, not used anywhere in this module.
//!
//! The reference C implementation and vendored data were fetched from the
//! SAME repository's `codes.rar` -> `codes/C version/{cec17_test_func.cpp,
//! main.cpp, readme.txt, input_data/}`
//! (`https://raw.githubusercontent.com/P-N-Suganthan/CEC2017-BoundContrained/master/codes.rar`,
//! SHA-256 `4fd2ac6913db34c61821476deb402e17e1a5c26a1ab016f71da5c9011317be89`,
//! git blob SHA `a4842e1bacceba91765ad91550b28f5579778e61`, size 7,465,979
//! bytes -- re-fetched fresh for this task and independently re-hashed,
//! matching a prior research pass's own fetch byte-for-byte; this task's
//! report has the full `curl`/`shasum` transcript). The rar also carries a
//! `Matlab/` copy with its own duplicate `input_data/` (same files, same
//! bytes) -- not used, only the C version was vendored from.
//!
//! ## Licensing finding (mirrors `cec2014/mod.rs`'s own finding)
//!
//! The repository root (`https://api.github.com/repos/P-N-Suganthan/CEC2017-BoundContrained/contents/`,
//! fetched directly) lists `Bound-Constrained-Comparisons.pdf`,
//! `CEC17_fast_pow-C++.zip`, `Codes-of-Top-Methods-and-results.zip`, the
//! definitions PDF, `Shifting and Rotation for CEC 2017.rar`, `codes.rar`,
//! and `readme--R-codes.md` -- NONE of `LICENSE`, `LICENSE.md`,
//! `LICENSE.txt` -- no license file exists at the repo root. Per the
//! milestone's scope ruling, the files below are vendored with prominent
//! attribution (this doc) rather than withheld.
//!
//! ## F2 ruling (user-approved scope ruling, binding) -- verified against
//! the compiled reference C
//!
//! CEC 2017's function 2 ("Sum of Different Powers", per the commented-out
//! call) was officially withdrawn. `cec17_test_func.cpp`'s dispatch switch,
//! quoted VERBATIM (the commented-out call lines are the C's OWN comments,
//! not this doc's elision):
//! ```text
//! case 2:
//!     //sum_diff_pow_func(&x[i*nx],&f[i],nx,OShift,M,1,1);
//!     //f[i]+=200.0;
//!     printf("\nError: This function (F2) has been deleted\n");
//!     break;
//! ```
//! Compiled and run directly (this task's probe driver,
//! `./probe 2 10 0 0 0 0 0 0 0 0 0 0`): prints exactly `Error: This
//! function (F2) has been deleted` to stdout and leaves `f[0]` COMPLETELY
//! UNSET (undefined behavior -- the C never writes `f[i]` on this branch;
//! the probe's own uninitialized stack double happened to read back as
//! `0.0` in this run, not a guarantee). `sezgi` follows the C's OWN
//! behavior, not merely its printed message: [`Cec2017::new`] with
//! `fid==2` returns [`Cec2017Error::Withdrawn`] BEFORE touching any
//! `input_data/` file (no `M_2_D*.txt`/`shift_data_2.txt` were even
//! vendored, module doc's "Vendoring scope" section below) -- a Rust
//! `Result::Err`, never an uninitialized/garbage `f64`. Numbering stays
//! GAPPED to match the C's own `func_num`: `Cec2017::new` accepts `fid==1`
//! and `fid` in `3..=10` (this task), never `2`; `F_i* = 100*fid` is
//! preserved as-is (fid 3's own `F*` is `300`, NOT renumbered down to
//! `200`), mirroring the C's own `f[i]+=<100*func_num>.0` lines exactly
//! (quoted in the dispatch section below) -- NOT the printed report's own
//! contiguous 1-29 Table I renumbering (see the next section for why the
//! two disagree).
//!
//! ## Report-table vs. code `func_num` numbering (verified, both read
//! directly)
//!
//! The report's OWN "Table I. Summary of the CEC'17 Test Functions" lists
//! a clean, contiguous 1-29 (no gap), with `Fi*=100*i`: row 1 Bent Cigar
//! (100), row 2 Zakharov (200), row 3 Rosenbrock's (300), row 4 Rastrigin's
//! (400), row 5 "Expanded Scaffer's F6 Function" (500), row 6 "Lunacek
//! Bi_Rastrigin" (600), row 7 "Non-Continuous Rastrigin's" (700), row 8
//! Levy (800), row 9 "Schwefel's Function" (900), row 10-19 Hybrid 1-10,
//! row 20-29 Composition 1-10. But the report's OWN section 1.3 formula
//! bodies use SUBSCRIPTS that already match the CODE's gapped numbering,
//! not the table's: Bent Cigar is written `f1(x)=...` (matches), Zakharov
//! `f3(x)=...` (table says row 2, formula says f3 -- ALREADY off by one),
//! Rosenbrock `f4(x)=...`, Rastrigin `f5(x)=...`, item 5 ("Expanded
//! Schaffer's F6", table row 5) is written `f6(x)=...`, item 6 (Lunacek,
//! table row 6) `f7(x)=...`, item 7 (Non-Continuous Rastrigin, table row 7)
//! `f8(x)=...`, item 8 (Levy) `f9(x)=w...` and item 9 (Modified Schwefel)
//! `f10(x)=...`. So the report's OWN prose is internally split: Table I
//! uses one numbering (contiguous, no gap, `Fi*=100*i`), section 1.3's
//! formula subscripts use ANOTHER (gapped at 2, matching the code's
//! `func_num`, `Fi*=100*func_num`) -- and the compiled C's dispatch switch
//! (quoted in full below) matches section 1.3's subscripts and biases
//! EXACTLY, not the table. `sezgi`'s own `fid` follows the code's
//! `func_num` (= section 1.3's own subscripts), per the F2 ruling above.
//!
//! ## VERIFIED DISCREPANCY: fid 6's name (report Table I / eq. 5) does NOT
//! match what the compiled C actually computes for `func_num==6`
//!
//! The report's Table I row 5 (which -- per the numbering note above -- is
//! the row whose `func_num` is 6) is titled "Expanded Scaffer's F6
//! Function", and section 1.3 item 5 gives its OWN formula body for that
//! name, quoted VERBATIM:
//! ```text
//! Schaffer's F6 Function: g(x,y) = 0.5 + (sin^2(sqrt(x^2+y^2)) - 0.5)
//!                                         / (1 + 0.001(x^2+y^2))^2
//! f6(x) = g(x1,x2) + g(x2,x3) + ... + g(x_{D-1},x_D) + g(x_D,x1)     (5)
//! ```
//! a CYCLIC sum (including the wrap-around `g(x_D,x1)` term) -- this is
//! `crate::cec_basics::escaffer6_base`'s exact shape. But the compiled C's
//! dispatch switch (quoted in full below) sends `func_num==6` to
//! `schaffer_F7_func`, whose body (quoted in full below, in the "Reused
//! bases" table's fid-6 row) is a COMPLETELY DIFFERENT, non-cyclic formula
//! (pairwise `y[i],y[i+1]` for `i=0..nx-2`, a `sin(50*s^0.2)` term, divided
//! by `(D-1)^2`) -- the SAME formula the report's OWN section 1.3 item 19
//! separately defines as "Schaffer's F7 Function" (a component-only
//! definition normally reserved for hybrid/composition sub-functions, per
//! the report's own later section). Compiled and probed directly (this
//! task's probe driver): running `func_num=6` reproduces the
//! `schaffer_F7_func`/item-19 formula, NOT the `escaffer6`/item-5 formula
//! (confirmed numerically: `f16_schaffer_f7_base` on the shift-only `y`
//! matches the compiled C to ~1e-15 relative at every probed point, module
//! doc's "Reused bases" table). `sezgi` follows the COMPILED C (this
//! project's standing PROVENANCE-FIRST rule: the reference implementation
//! governs over report prose when the two disagree), so `Cec2017`'s fid 6
//! computes the Schaffer's-F7-shaped formula, NOT Expanded Schaffer's F6 --
//! this is a genuine inconsistency in the officially shipped reference kit
//! itself (report text vs. shipped code), not a `sezgi`-side choice of
//! convenience.
//!
//! ## Vendoring scope (M3-6 T6, user-approved, dims {10, 30} only)
//!
//! `crates/problems/data/cec2017/` carries EVERY `input_data/` file the
//! reference C's own loader reads for `fid` in `{1,3..=30}` (fid 2
//! excluded entirely -- its dead `input_data` files are never vendored,
//! F2-ruling section above) at `dim` 10 and 30, so T7/T8 (which extend
//! `Cec2017` to fid 11-30) do not need to re-fetch or re-copy anything:
//! every `shift_data_<fid>.txt` for `fid` in `{1,3..=30}` (dimension-
//! independent, one copy each), every `M_<fid>_D{10,30}.txt` for `fid` in
//! `{1,3..=30}` (the C's loader unconditionally reads an `M` file for
//! every `func_num`, quoted below), and `shuffle_data_<fid>_D{10,30}.txt`
//! ONLY for the fids the loader's own conditional actually reads shuffle
//! data for -- `fid` in `{11..=20, 29, 30}` (quoted in the loader excerpt
//! below).
//!
//! **Measured total** (`find crates/problems/data/cec2017 -type f -exec
//! stat -f%z {} \; | awk '{s+=$1} END {print s}'`, cross-checked with an
//! independent Python `os.path.getsize` sum during the copy step): **111
//! files, 3,285,318 bytes.** T2 precedent: estimate superseded by
//! measurement -- an earlier scratchpad research pass's own estimate
//! ("D10+D30 for CEC2017: 3,021,840 bytes", carried into this milestone's
//! brief) undercounted because it summed the upstream repo's raw
//! per-dimension file listing (`D10`+`D30` `input_data/` byte totals from
//! the GitHub API) without separating out `shift_data_<fid>.txt` (which
//! has no dimension suffix and is shared across ALL dims, so summing "the
//! D10 total" and "the D30 total" independently double-counts every shift
//! file) AND without excluding fid 2's dead files -- both corrected here.
//! This module's own `include_str!`/parser functions (`data.rs`) cover
//! ONLY fid `{1,3..=10}` (this task's functional scope, 27 of the 111
//! vendored files: 9 `shift_data_<fid>.txt` + 18 `M_<fid>_D{10,30}.txt`);
//! the other 84 files (fid 11-30) sit unreferenced on disk until T7/T8.
//!
//! Supported dims -- `{2,10,20,30,50,100}` per the C
//! (`cec17_test_func.cpp` line 103: `if (!(nx==2||nx==10||nx==20||nx==30||nx==50||nx==100))
//! printf("\nError: Test functions are only defined for D=2,10,20,30,50,100.\n");`),
//! `{10,30}` vendored here (this task's user-approved scope) --
//! [`Cec2017Error::BadDim`] for anything else.
//!
//! ## Data-loading excerpt (`cec17_test_func.cpp`'s `cec17_test_func`
//! initializer, `func_num<20` branch -- the branch that applies to every
//! fid this module serves, quoted VERBATIM)
//!
//! ```text
//! sprintf(FileName, "input_data/M_%d_D%d.txt", func_num,nx);
//! fpt = fopen(FileName,"r");
//! if (func_num<20) {
//!     M=(double*)malloc(nx*nx*sizeof(double));
//!     for (i=0; i<nx*nx; i++) fscanf(fpt,"%lf",&M[i]);   // fixed up, see below
//! }
//! sprintf(FileName, "input_data/shift_data_%d.txt", func_num);
//! fpt = fopen(FileName,"r");
//! if (func_num<20) {
//!     OShift=(double *)malloc(nx*sizeof(double));
//!     for(i=0;i<nx;i++) fscanf(fpt,"%lf",&OShift[i]);    // fixed up, see below
//! }
//! if (func_num>=11&&func_num<=20) {
//!     sprintf(FileName, "input_data/shuffle_data_%d_D%d.txt", func_num, nx);
//!     ...
//! } else if (func_num==29||func_num==30) {
//!     sprintf(FileName, "input_data/shuffle_data_%d_D%d.txt", func_num, nx);
//!     ...
//! }
//! ```
//! Confirms: an `M`/`shift_data` file is loaded for EVERY `func_num`
//! (including fid 6, whose own dispatch ends up never using the rotated
//! result in its formula -- module doc's fid-6 discrepancy note above and
//! the "Reused bases" table's fid-6 row); `shuffle_data` is loaded ONLY for
//! `func_num` 11-20 or 29-30.
//!
//! ## Compile edits (portability only, no behavior change)
//!
//! Compiled with `g++ -O2` in the scratchpad
//! (`/private/tmp/.../scratchpad/t6_build/src/`), never in the repo. Same
//! TWO-edit class T2's report already documented for CEC2014's
//! `cec14_test_func.cpp` (`readme.txt` itself, vendored alongside, names
//! this exact fix: *"For Linux Users: Please change %xx in fscanf and
//! fprintf and do use "WINDOWS.H"."*):
//!
//! 1. `#include <WINDOWS.H>` and `#include <malloc.h>` removed; `#include
//!    <cstdlib>` added.
//! 2. All 5 `fscanf(fpt,"%Lf", &<double*>)` call sites (lines 126, 136,
//!    156, 168, 174 of the vendored `cec17_test_func.cpp` -- independently
//!    grepped, exactly 5, same count as CEC2014's own 5) rewritten to
//!    `%lf`: `%Lf` scans a `long double` into an 8-byte `double*`
//!    destination -- undefined behavior off Windows/MSVC that can corrupt
//!    adjacent heap memory.
//!
//! A custom probe driver (`probe.cpp`, `argv: fid dim x0 x1 ...` ->
//! `%.20f`-printed `f(x)`) drove every o-pin and random-point check below.
//! A second isolated driver (`probe2.cpp`) called `bi_rastrigin_func`
//! directly (bypassing the file-loading dispatcher, hand-supplied
//! `Os`/`M`/`x`) for fid 7's hand fixtures, mirroring T2's own
//! `probe_weierstrass.cpp` pattern. This task's report has the full
//! `g++`/run transcripts for both.
//!
//! ## Dispatch (`cec17_test_func.cpp`'s switch, `case 1` and `case 3..10`,
//! quoted VERBATIM -- every case passes `s_flag=1,r_flag=1` to its base
//! function)
//!
//! ```text
//! case 1:  bent_cigar_func(&x[i*nx],&f[i],nx,OShift,M,1,1);   f[i]+=100.0;  break;
//! case 2:  printf("\nError: This function (F2) has been deleted\n");        break;
//! case 3:  zakharov_func(&x[i*nx],&f[i],nx,OShift,M,1,1);     f[i]+=300.0;  break;
//! case 4:  rosenbrock_func(&x[i*nx],&f[i],nx,OShift,M,1,1);   f[i]+=400.0;  break;
//! case 5:  rastrigin_func(&x[i*nx],&f[i],nx,OShift,M,1,1);    f[i]+=500.0;  break;
//! case 6:  schaffer_F7_func(&x[i*nx],&f[i],nx,OShift,M,1,1);  f[i]+=600.0;  break;
//! case 7:  bi_rastrigin_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=700.0;  break;
//! case 8:  step_rastrigin_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=800.0; break;
//! case 9:  levy_func(&x[i*nx],&f[i],nx,OShift,M,1,1);         f[i]+=900.0;  break;
//! case 10: schwefel_func(&x[i*nx],&f[i],nx,OShift,M,1,1);     f[i]+=1000.0; break;
//! ```
//!
//! ## Reused bases (`crate::cec_basics`) vs. NEW (this module) -- each
//! probe-verified against the compiled C, not trusted by name alone
//!
//! | fid | C function | `sr_func` scale, rotate | sezgi | new? | note |
//! |---|---|---|---|---|---|
//! | 1 | `bent_cigar_func` | `1.0`, yes | `cb::bent_cigar_base` | no | `f[0]=z0^2+1e6*sum(z[1..]^2)` -- byte-identical formula to CEC2014/2022's own, probe-verified (o-pin + 6 random points, this task's report). |
//! | 3 | `zakharov_func` | `1.0`, yes | `cb::f1_base` | no | `sum1+sum2^2+sum2^4` -- byte-identical to CEC2022's own Zakharov (extracted into `cec_basics` as `f1_base`), probe-verified. |
//! | 4 | `rosenbrock_func` | `2.048/100.0`, yes, then `z[i]+=1.0` for every `i` (shift-to-origin) | `cb::f2_base` on the `+1`'d `z` | no | SAME `+1` pattern CEC2014's fid 4 and CEC2022's fid 2 already use `f2_base` for -- probe-verified. |
//! | 5 | `rastrigin_func` | `5.12/100.0`, yes | `cb::f4_base` | no | Plain Rastrigin, byte-identical to CEC2014's fid 9 usage -- probe-verified. |
//! | 6 | `schaffer_F7_func` | `1.0`, yes (but see below) | `cb::f16_schaffer_f7_base` on the SHIFT-ONLY `y` | no | **VERIFIED DISCREPANCY** (module doc section above has the name mismatch): `schaffer_F7_func`'s body, quoted VERBATIM: `for(i=0;i<nx-1;i++){ z[i]=pow(y[i]*y[i]+y[i+1]*y[i+1],0.5); tmp=sin(50.0*pow(z[i],0.2)); f[0]+=pow(z[i],0.5)+pow(z[i],0.5)*tmp*tmp; } f[0]=f[0]*f[0]/(nx-1)/(nx-1);` -- reads `y[i]`/`y[i+1]`, the SHIFT-ONLY intermediate `sr_func` computes internally BEFORE rotating into `z` (`sr_func`'s `s_flag==1,r_flag==1` branch: `shiftfunc(x,y,...); y[i]*=sh_rate; rotatefunc(y,sr_x,...)`), and OVERWRITES the module's `z` array with its own local, unrelated `z[i]=sqrt(y_i^2+y_{i+1}^2)` computation -- the ACTUAL rotated `sr_func` output is silently discarded. So even though `r_flag=1` is passed, the rotation `M` is loaded but never affects fid 6's fitness -- the SAME "M vendored but sometimes unused" shape CEC2022's own fid-3 discrepancy note already documents, and `sezgi` mirrors that exact call shape: `f16_schaffer_f7_base(&shift_scale_rotate(xs, 1.0, false))` (shift-only, `rotate=false`). |
//! | 7 | `bi_rastrigin_func` | custom (no plain `sr_func` call) | `f7_bi_rastrigin` (module-local, THIS TASK, new) | **yes** | Full C body quoted at [`f7_bi_rastrigin`]'s own doc. Module-local (not `cec_basics`) per this task's own judgment call: unlike every other base here, its formula is NOT a pure function of an already shift-scale-rotated `z` -- it reads the RAW shift vector `Os` a second time (for a per-dimension sign flip keyed on `Os[i]<0.0`) AFTER the shift/scale step, entangling shift-data semantics with the base formula itself; extracting it to `cec_basics` would require redesigning that shared module's `(z) -> f64` contract for one caller, and no second CEC generation's C has been read to confirm this exact entanglement recurs identically elsewhere -- left suite-local per the "unpinnable => module-local" judgment the brief's item 5 directs. |
//! | 8 | `step_rastrigin_func` | `5.12/100.0`, yes | `cb::f4_base` | no | **VERIFIED DISCREPANCY**, SAME BUG CLASS as CEC2022's own fid-4 discrepancy (`cec_basics.rs`'s `f4_base` doc references it): the C's non-continuous pre-transform is DEAD CODE. Body quoted VERBATIM: `for(i=0;i<nx;i++){ if(fabs(y[i]-Os[i])>0.5) y[i]=Os[i]+floor(2*(y[i]-Os[i])+0.5)/2; } sr_func(x,z,nx,Os,Mr,5.12/100.0,s_flag,r_flag); for(i=0;i<nx;i++) f[0]+=(z[i]*z[i]-10.0*cos(2.0*PI*z[i])+10.0);` -- the pre-loop reads/writes the GLOBAL `y` buffer, but the very next `sr_func` call's `s_flag==1` branch immediately overwrites EVERY `y[i]` via `shiftfunc(x,y,nx,Os)` (`y[i]=x[i]-Os[i]`) before `y` is used for anything else -- the pre-loop's edit is unconditionally clobbered before it can affect `z` (or anything downstream), regardless of what `y` held on entry (even garbage from a prior, unrelated call). Probe-verified: fid 5 (plain Rastrigin) and fid 8 (this one) produce IDENTICAL compiled-C output at every probed point in this task's report (same formula, same shift/rotate pipeline, only the vendored `M_8_D*.txt`/`shift_data_8.txt` differ from fid 5's own -- which is why fid 8 does NOT reduce to fid 5's numeric output, just its FORMULA). |
//! | 9 | `levy_func` | `1.0`, yes | `cb::f5_base` on the `-1`'d `z` | no (adjusted) | **VERIFIED CROSS-GENERATION CONSTANT DIVERGENCE** (brief item 5's own warning: "same-named functions across CEC generations sometimes differ in constants -- probe, don't trust names"): `cb::f5_base` was extracted from CEC 2022's OWN Levy and computes `w[i]=1.0+z[i]/4.0`; CEC 2017's `levy_func` (quoted VERBATIM) computes `w[i] = 1.0 + (z[i] - 1.0)/4.0` -- an EXTRA `-1.0` CEC 2022 does not have. First attempt (calling `f5_base` directly on `z`, no adjustment) was caught by this task's OWN random-point probes: it passed compile but FAILED every probe (`x=o` gave exactly `900.0` instead of the compiled C's `901.4426...`, and every random point diverged by 1-25% relative, not ~1e-15 -- module doc's report has the failing-then-fixed transcript). Fix: pre-subtract `1.0` from every `z[i]` before calling `f5_base`, so its own `1.0+zi/4.0` computes the exact `w` CEC 2017's formula wants -- the mirror-image of fid 4's `+1.0` `f2_base` compensation. After the fix: **`x=o` STILL does NOT pin to `F_i*` exactly** for this fid (every other fid 1,3-8,10 does -- module doc's "x=o pin" test section) -- this part is a genuine property of the (now-correctly-ported) Levy formula itself, not a bug: plain Levy's own unconstrained minimum (`f=0`) is at `w=1` i.e. `z=1` (all-ones), NOT `z=0` -- unlike fid 4 (Rosenbrock), the C's `levy_func` applies NO compensating `+1` offset to `z` before evaluating (contrast `rosenbrock_func`'s explicit `z[0]+=1.0;...z[i+1]+=1.0;` loop, fid-4 row above), so `x=o` (which gives `z=0`) lands away from Levy's own minimum. Measured (compiled C, `x=o`): `f(o)=901.44260098705274...` at dim=10, `903.25949206939231...` at dim=30 (both `f_star+`~1-4, NOT exactly `900`) -- the TRUE global optimum (`f=900` exactly) IS still attainable somewhere in the `[-100,100]^dim` box (at `z=1`, i.e. some `x != o`), so [`Cec2017::optimum`] still correctly returns `f_star` for fid 9; only the `x=o` PIN itself is the verified exception, tested against the measured constant instead of `f_star` (this module's own test module). |
//! | 10 | `schwefel_func` | `1000.0/100.0`, yes | `cb::schwefel_base` | no | Byte-identical to CEC2014's fid 11 usage -- probe-verified. |
//!
//! `shift_scale_rotate` (this module's own thin wrapper over
//! `crate::cec_basics::shift_scale_rotate`, `(x-o)*sr` then optionally
//! `M*result`) is the SAME shared helper CEC2014/CEC2022 use -- no new
//! shift/rotate arithmetic this task.
//!
//! ## `x = o` pin (Step 2): why `F_i(o_i) == F_i*` EXACTLY for fid 1,3-8,10
//!
//! At `x = o`: the shift `x - o = 0` exactly (identical `f64` operands,
//! IEEE 754 subtraction of equal values is exact), so `z = M*0*sr = 0`
//! exactly for every fid whose base takes a plain shift-scale-rotated `z`
//! with no extra additive offset -- `bent_cigar_base(0)=0`, `f1_base(0)=0`
//! (Zakharov), `f4_base(0)=0` (Rastrigin, `cos(0)=1.0` exact),
//! `schwefel_base(0)`: `zi=0+4.209687462275036e+02`, hits the `else`
//! branch (`|zi|<500`), `-zi*sin(sqrt(|zi|))` at the EXACT constant the
//! trailing `+4.189828872724338e+02*n` is defined to cancel (same
//! derivation CEC2014's own module doc gives for its own Schwefel fids).
//! `f2_base` (fid 4, Rosenbrock) at the `+1`'d all-ones vector:
//! `100*(1-1)^2+(1-1)^2=0` per term. Fid 6's `f16_schaffer_f7_base` on an
//! all-zero `y`: every pairwise term is `0`, `f=0`. Fid 7 (Lunacek) is
//! its OWN base with its own derivation, not reused from this list --
//! `x=o` gives `y=0`, `z=0` (per [`f7_bi_rastrigin`]'s own body: `2*0=0`
//! regardless of sign), `tmp1=0`, `tmp2=d*n` (`>0` for `n>0`, so `min` picks
//! `tmp1=0`), `cos_sum=n*cos(0)=n`, `f=0+10*(n-n)=0` -- confirmed
//! algebraically AND probe-verified (compiled C at `x=o`, this task's
//! report). Fid 9 (Levy) is the documented exception (table above).
//!
//! ## M3-6 T7: Hybrid Functions (fid 11-20)
//!
//! Dispatch (`cec17_test_func.cpp`'s switch, `case 11..20`, quoted
//! VERBATIM -- a CLEAN 1:1 mapping, fid 11 calls `hf01` ("Hybrid Function
//! 1"), fid 12 calls `hf02`, ..., fid 20 calls `hf10`, same shape
//! `cec2014/mod.rs`'s own T3 fid-17-22 dispatch table already documents for
//! that suite, UNLIKE `cec2022`'s own hybrid dispatch whose internal
//! `hf02`/`hf06`/`hf10` names don't track its report numbering at all):
//! ```text
//! case 11:  hf01(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=1100.0; break;
//! case 12:  hf02(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=1200.0; break;
//! case 13:  hf03(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=1300.0; break;
//! case 14:  hf04(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=1400.0; break;
//! case 15:  hf05(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=1500.0; break;
//! case 16:  hf06(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=1600.0; break;
//! case 17:  hf07(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=1700.0; break;
//! case 18:  hf08(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=1800.0; break;
//! case 19:  hf09(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=1900.0; break;
//! case 20:  hf10(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=2000.0; break;
//! ```
//!
//! ### Top-level dataflow -- VERIFIED to hold for CEC 2017's own `hf01`
//! (quoted VERBATIM, cross-checked against every one of `hf02`..`hf10`,
//! which share the identical shape modulo `cf_num`/`Gp`/component list):
//! ```text
//! void hf01 (double *x, double *f, int nx, double *Os,double *Mr,int *S,int s_flag,int r_flag)
//! {
//!     int i,tmp,cf_num=3;
//!     double fit[3];
//!     int G[3],G_nx[3];
//!     double Gp[3]={0.2,0.4,0.4};
//!     tmp=0;
//!     for (i=0; i<cf_num-1; i++) { G_nx[i]=ceil(Gp[i]*nx); tmp+=G_nx[i]; }
//!     G_nx[cf_num-1]=nx-tmp;
//!     G[0]=0;
//!     for (i=1; i<cf_num; i++) { G[i]=G[i-1]+G_nx[i-1]; }
//!     sr_func (x, z, nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */
//!     for (i=0; i<nx; i++) { y[i]=z[S[i]-1]; }
//!     i=0; zakharov_func(&y[G[i]],&fit[i],G_nx[i],Os,Mr,0,0);
//!     i=1; rosenbrock_func(&y[G[i]],&fit[i],G_nx[i],Os,Mr,0,0);
//!     i=2; rastrigin_func(&y[G[i]],&fit[i],G_nx[i],Os,Mr,0,0);
//!     f[0]=0.0;
//!     for(i=0;i<cf_num;i++) { f[0] += fit[i]; }
//! }
//! ```
//! CONFIRMS the brief's own hint: full shift+rotate FIRST (`sr_func`, always
//! `sh_rate=1.0`, `s_flag=1,r_flag=1` from the dispatch above), THEN shuffle
//! (`y[i]=z[S[i]-1]`, 1-based lookup), THEN segment (`G`/`G_nx`, EXACTLY
//! [`crate::cec_basics::segment_sizes`]/[`segment_starts`], already reused
//! unmodified by `cec2014`'s own T3) -- byte-identical dataflow shape to
//! `cec2014/mod.rs`'s own T3 hybrids, no new segmentation/shuffle machinery
//! needed this task. EVERY component call passes `s_flag=0,r_flag=0` (never
//! `1,1`) -- per `sr_func`'s own `s_flag==0,r_flag==0` branch (quoted in
//! T6's module doc, `mod.rs`'s "Compile edits" section is a different quote;
//! the exact branch is `sr_func`'s trailing `else` arm: `for(i=0;i<nx;i++)
//! sr_x[i]=x[i]*sh_rate;`, PLAIN scale, no shift, no rotate, `Os`/`Mr`
//! UNTOUCHED) -- so each component base function receives its own segment
//! `y[G[i]..G[i]+G_nx[i]-1]`, scaled by ONLY that base function's OWN
//! internal `sh_rate` constant (e.g. `rosenbrock_func`'s own
//! `2.048/100.0`), no further shift or rotation.
//!
//! ### Hybrid composition tables -- quoted from the C (`Gp` arrays +
//! component call order, ALL TEN `hf0N` read directly, this task's own
//! compiled probe cross-checks below)
//!
//! | fid | `hf0N` | `Gp` (C) | Component call order (C) |
//! |---|---|---|---|
//! | 11 | `hf01` | `{0.2,0.4,0.4}` | `zakharov_func, rosenbrock_func, rastrigin_func` |
//! | 12 | `hf02` | `{0.3,0.3,0.4}` | `ellips_func, schwefel_func, bent_cigar_func` |
//! | 13 | `hf03` | `{0.3,0.3,0.4}` | `bent_cigar_func, rosenbrock_func, bi_rastrigin_func` |
//! | 14 | `hf04` | `{0.2,0.2,0.2,0.4}` | `ellips_func, ackley_func, schaffer_F7_func, rastrigin_func` |
//! | 15 | `hf05` | `{0.2,0.2,0.3,0.3}` | `bent_cigar_func, hgbat_func, rastrigin_func, rosenbrock_func` |
//! | 16 | `hf06` | `{0.2,0.2,0.3,0.3}` | `escaffer6_func, hgbat_func, rosenbrock_func, schwefel_func` |
//! | 17 | `hf07` | `{0.1,0.2,0.2,0.2,0.3}` | `katsuura_func, ackley_func, grie_rosen_func, schwefel_func, rastrigin_func` |
//! | 18 | `hf08` | `{0.2,0.2,0.2,0.2,0.2}` | `ellips_func, ackley_func, rastrigin_func, hgbat_func, discus_func` |
//! | 19 | `hf09` | `{0.2,0.2,0.2,0.2,0.2}` | `bent_cigar_func, rastrigin_func, grie_rosen_func, weierstrass_func, escaffer6_func` |
//! | 20 | `hf10` | `{0.1,0.1,0.2,0.2,0.2,0.2}` | `hgbat_func, katsuura_func, ackley_func, rastrigin_func, schwefel_func, schaffer_F7_func` |
//!
//! Report-vs-code divergence in this table (found in review, code-over-report
//! ruling applies): the definitions PDF lists fid 20's FIRST component as
//! "Happycat Function (f17)", but the shipped C's `hf10` calls `hgbat_func`
//! (f18, quoted above) -- sezgi follows the C, as everywhere else.
//!
//! ### New component enum ([`HybridComponent`]) -- every base function
//! ALREADY exists in `crate::cec_basics` (T6/T2/T1's own extractions;
//! `zakharov`=[`crate::cec_basics::f1_base`], `rosenbrock`=[`crate::cec_basics::f2_base`]
//! `+1`'d per fid 4's own pattern, `rastrigin`=[`crate::cec_basics::f4_base`],
//! plus `ellips_base`/`schwefel_base`/`bent_cigar_base`/`ackley_base`/
//! `hgbat_base`/`escaffer6_base`/`katsuura_base`/`grie_rosen_base`/
//! `weierstrass_base`/`discus_base`) -- NO new base function needed for any
//! of these ten, EXCEPT the two VERIFIED-BUGGY components below, which are
//! NOT pure functions of their own segment and so are special-cased
//! directly in [`Cec2017::hybrid_fitness`] rather than routed through
//! `HybridComponent::eval`'s uniform `(seg) -> f64` contract (same
//! structural choice `cec2022::HybridComponent::SchafferF7Buggy` already
//! established for that suite's own fid-7 hybrid; this task ADDS a second,
//! CEC-2017-only buggy variant with no `cec2022`/`cec2014` precedent).
//!
//! Every base above was probe-verified against the compiled C for its
//! HYBRID usage specifically (not just assumed identical to its T6
//! standalone usage): this task wrote a from-scratch Python replica of ALL
//! TEN `hf0N` (mirroring the C's OWN per-component `sh_rate` scale, since
//! `s_flag=0,r_flag=0` still applies that constant even though it skips
//! shift/rotate) and cross-checked it against the compiled C at 3 random
//! points x 2 dims x 10 fids = 60 probes; every one matched to
//! **~1e-15 relative or exact** (max measured: `1.9644724722821017e-15`,
//! fid=18, dim=30 -- this task's report has the full probe transcript). The
//! SAME replica, ported line-for-line to this module's
//! [`Cec2017::hybrid_fitness`]/[`HybridComponent`], is what the Rust test
//! suite below re-verifies (`hybrid_random_probe_*` tests, using DIFFERENT
//! random points than the Python cross-check, generated independently by
//! `t7_gen_fixtures.py` calling the compiled C directly for the literal
//! values).
//!
//! ### VERIFIED DISCREPANCY 1: `schaffer_F7_func` as a hybrid component
//! (fid 14's `hf04`, fid 20's `hf10`) reads a PREFIX of the WHOLE shuffled
//! `y`, not its own assigned segment -- the SAME bug class T6 already found
//! for fid 6's STANDALONE usage (module doc above), but here it also
//! survives the segment-offset call convention
//!
//! `schaffer_F7_func`'s body (T6's module doc quotes it in full) computes
//! its formula from the GLOBAL `y[i]`/`y[i+1]` (module-level C globals, NOT
//! its own `x` parameter, NOT the `z` its own internal `sr_func` call
//! writes) for `i=0..nx_local-2`. When called as a hybrid component
//! (`schaffer_F7_func(&y[G[i]],&fit[i],G_nx[i],Os,Mr,0,0)`), the GLOBAL `y`
//! identifier is the SAME dim-length buffer the outer `hf0N` just wrote via
//! its own shuffle step (`y[i]=z[S[i]-1]` for `i=0..nx-1`) -- but
//! `schaffer_F7_func`'s own loop indexes THAT buffer starting from absolute
//! position `0`, NOT from `G[i]` (its caller's intended offset, which only
//! reaches the function through the IGNORED `x` parameter). Net effect:
//! this component ALWAYS evaluates Schaffer's F7 on `y[0..G_nx[component]-1]`
//! -- the FIRST `G_nx[component]` elements of the FULL shuffled vector --
//! regardless of where its own segment actually sits. Verified with a
//! targeted compiled-C probe (fid 14/`hf04`, `x` chosen so segments are
//! numerically distinguishable): the "correct segment" hypothesis
//! (`f16_schaffer_f7_base` on `y[G[2]..G[2]+2]`) gave a component value of
//! `519.7396346671704`; the "reads `y[0..2]`" hypothesis gave
//! `262.971892911507`; the compiled C's actual total
//! (`515861491.82161021232604980469`) matches ONLY the second hypothesis
//! bit-for-bit once summed with the other three (correctly-behaved)
//! components -- confirmed again independently for fid 20/`hf10` (component
//! at the LAST position, `G[5]=8`, same "reads from 0" behavior). `sezgi`
//! mirrors this: [`Cec2017::hybrid_fitness`]'s `SchafferF7Buggy` arm calls
//! `crate::cec_basics::f16_schaffer_f7_base(&y[0..len])`, `len` = this
//! component's OWN segment length, but sliced from index `0` of the FULL
//! `y`, never `start`.
//!
//! ### VERIFIED DISCREPANCY 2: `bi_rastrigin_func` as a hybrid component
//! (fid 13's `hf03`, position 3) reads the WRONG slice of the ORIGINAL
//! shift vector `Os` for its sign-flip test -- a NEW bug, no `cec2022`/
//! `cec2014` precedent
//!
//! `bi_rastrigin_func`'s body (T6's module doc quotes it in full) reads the
//! RAW shift vector `Os[i]` a second time (independent of its own `s_flag`-
//! gated shift step) to decide a per-dimension sign flip:
//! `if (Os[i] < 0.0) tmpx[i] *= -1.;`. When called as a hybrid component
//! (`bi_rastrigin_func(&y[G[2]],&fit[2],G_nx[2],Os,Mr,0,0)`), `Os` is passed
//! as the SAME unoffset, full dim-length pointer every component call uses
//! (never `&Os[G[2]]`) -- so `Os[i]` for LOCAL `i=0..G_nx[2]-1` reads
//! `Os[0..G_nx[2]-1]`, the FIRST `G_nx[2]` values of the ORIGINAL
//! (unshuffled, unrotated) shift vector, NOT the values aligned with this
//! component's actual position `G[2]` in the shuffled/segmented vector.
//! (With `s_flag=0`, the function's OWN `y[i]=x[i]` shift-step branch DOES
//! correctly use its `x` parameter -- so the segment CONTENT is right, only
//! the SIGN-FLIP REFERENCE is misaligned.) Verified with a targeted
//! compiled-C probe (fid 13/`hf03`): the "correct offset" hypothesis
//! (`Os[G[2]..G[2]+3]` as the sign reference) gave a component value of
//! `311.712370364593`; the "reads `Os[0..3]`" hypothesis gave
//! `669.6574913888649`; the compiled C's actual total
//! (`3950498384.45452404022216796875`) matches ONLY the second hypothesis
//! bit-for-bit. `sezgi` mirrors this: [`Cec2017::hybrid_fitness`]'s
//! `BiRastriginBuggy` arm calls [`bi_rastrigin_hybrid_component`] with
//! `sign_ref = &self.o[0..len]`, never `&self.o[start..start+len]`.
//!
//! ### `x = o` pin (fid 11-20): exactly `F_i*`, same as fid 1,3-8,10
//!
//! At `x=o`: the outer `sr_func`'s shift is exact-zero (`x-o=0`, IEEE 754
//! subtraction of equal `f64` operands), so `z=0` exactly for every index,
//! and the shuffle (`y[i]=z[S[i]-1]`) leaves `y` all-zero too (a
//! permutation of an all-zero vector is still all-zero) -- so EVERY
//! component (including the two verified-buggy ones above, since their
//! bugs only misalign WHICH zero they read, and `0.0` reads identically
//! from any index) evaluates its base formula on an all-zero segment, which
//! is `0` for every base here (same per-fid derivation T6's module doc
//! already gives for fid 1,3-8,10's own standalone bases, all reused
//! unmodified). Compiled-C probe confirms all 10 fids, both dims, hit
//! EXACTLY `100*fid` at `x=o` (this task's report has the full transcript,
//! `t7_xo_pin.py`) -- no exception this task, unlike fid 9's own Levy
//! exception above.
//!
//! ### Measured deviations (this task)
//!
//! 60-probe Python-vs-compiled-C cross-check (above): max
//! `1.9644724722821017e-15` relative (fid=18, dim=30). The Rust test
//! suite's own independent 60-probe cross-check (different random points,
//! `hybrid_random_probe_*` tests below) uses the SAME `1e-9`
//! [`tests::assert_close`] tolerance the fid-1-10 tests already use,
//! comfortably above both measured figures.
//!
//! ## M3-6 T8: Composition Functions (fid 21-30)
//!
//! Source (PROVENANCE, re-verified this task): the SAME two artifacts T6/T7
//! fetched (module doc's opening PROVENANCE section), re-hashed this task
//! against the scratchpad build (`shasum -a 256`, `cec17_test_func.cpp` at
//! `scratchpad/t6_build/src/cec17_test_func.cpp`,
//! SHA-256 `63dd4b36a33ca5cb2c5debd94dbbae4cc702609468552b47821e57328d461516`
//! -- IDENTICAL to T6/T7's own pinned copy, no re-fetch needed). Re-used the
//! SAME compiled `probe` binary T6/T7 already built (`g++ -O2`, unmodified
//! `probe.cpp`, dispatches through the unmodified `cec17_test_func()` entry
//! point for ANY `fid`, so fid 21-30 needed no probe-side change). All
//! `input_data/` for fid 21-30 was ALREADY vendored in
//! `crates/problems/data/cec2017/` from T6's full-suite vendoring pass
//! (module doc's "Vendoring scope" section, `{1,3..=30}`) -- confirmed
//! present, byte-identical to the scratchpad probe's own copy (`M_<fid>_D
//! <dim>.txt`/`shift_data_<fid>.txt`/`shuffle_data_{29,30}_D<dim>.txt`), no
//! re-vendoring this task.
//!
//! ### Dispatch (`cec17_test_func.cpp`'s switch, `case 21..30`, quoted
//! VERBATIM -- a CLEAN 1:1 mapping, fid 21 calls `cf01` ("Composition
//! Function 1"), fid 22 calls `cf02`, ..., fid 30 calls `cf10`, TEN
//! composition functions -- unlike `cec2014`'s own suite, which only has
//! EIGHT (`cf01..cf08`, fid 23-30 there))
//! ```text
//! case 21:  cf01(&x[i*nx],&f[i],nx,OShift,M,1); f[i]+=2100.0; break;
//! case 22:  cf02(&x[i*nx],&f[i],nx,OShift,M,1); f[i]+=2200.0; break;
//! case 23:  cf03(&x[i*nx],&f[i],nx,OShift,M,1); f[i]+=2300.0; break;
//! case 24:  cf04(&x[i*nx],&f[i],nx,OShift,M,1); f[i]+=2400.0; break;
//! case 25:  cf05(&x[i*nx],&f[i],nx,OShift,M,1); f[i]+=2500.0; break;
//! case 26:  cf06(&x[i*nx],&f[i],nx,OShift,M,1); f[i]+=2600.0; break;
//! case 27:  cf07(&x[i*nx],&f[i],nx,OShift,M,1); f[i]+=2700.0; break;
//! case 28:  cf08(&x[i*nx],&f[i],nx,OShift,M,1); f[i]+=2800.0; break;
//! case 29:  cf09(&x[i*nx],&f[i],nx,OShift,M,SS,1); f[i]+=2900.0; break;
//! case 30:  cf10(&x[i*nx],&f[i],nx,OShift,M,SS,1); f[i]+=3000.0; break;
//! ```
//! `Fi* = 2100..3000` continues the SAME `100*fid` closed form fid 1-20
//! already established -- `Cec2017::f_star`'s existing formula needs no
//! per-fid override for fid 21-30 either. The outer `r_flag` literal passed
//! to every `cf0N` call is always `1` (never `0`) -- and UNLIKE
//! `cec2014`'s own `cf01`/`cf02` (which each hardcode ONE component's
//! `r_flag` to a literal `0`, module doc's `cec2014` sibling note), every
//! single component call in ALL TEN of `cf01`..`cf10` here passes the
//! `r_flag` PARAMETER through unmodified (verified by reading all ten
//! bodies directly, quoted in full below) -- so EVERY composition component
//! in this suite is rotated, no hardcoded-unrotated exception exists here.
//!
//! ### The `cf_num=10`-hardcoded data-loading quirk (SAME shape
//! `cec2014/mod.rs`'s own T4 section documents, DIFFERENT threshold: this
//! suite's loader branches on `func_num<20`, not `cec2014`'s `func_num<23`
//! -- verified directly, not assumed from the brief)
//!
//! `cec17_test_func`'s initializer declares `int cf_num=10` as a LOCAL at
//! the very top of the function (quoted in full in this module's "Data-
//! loading excerpt" section above, T6), and its `M`/`OShift`/`SS` loading
//! branches on `func_num<20` (not `<23`) -- so EVERY `func_num>=20`,
//! INCLUDING fid 20 itself (`hf10`, a hybrid, T7's own scope) as well as fid
//! 21-30 (this task), loads `M`/`OShift` as `cf_num`=10 blocks (T7's own
//! module doc already establishes this is harmless for fid 20, since `hf10`
//! only ever reads block 0). For fid 21-30, every one of the ten
//! composition functions' OWN `cf_num` local (3 for `cf01`/`cf02`/`cf09`/
//! `cf10`, 4 for `cf03`/`cf04`, 5 for `cf05`/`cf06`, 6 for `cf07`/`cf08`,
//! quoted per-fid below) is `<=6`, always LESS than the loader's hardcoded
//! `10` -- confirmed: every vendored `shift_data_<fid>.txt` (fid 21-30)
//! carries exactly 10 rows (`wc -l`), every `M_<fid>_D<dim>.txt` carries
//! exactly `10*dim*dim` values (`wc -w`: `1000`/`9000` at dim 10/30), and
//! `shuffle_data_{29,30}_D<dim>.txt` carries exactly `10*dim` ints (`wc -w`:
//! `100`/`300`). [`data::composition_shift_blocks`] /
//! [`data::composition_rotation_blocks`] / [`data::composition_shuffle_blocks`]
//! (data.rs's own doc comments) take the CALLER's actual `cf_num` and parse
//! only that many rows/blocks, byte-identical shape to
//! `cec2014::data`'s own composition parsers (just a different loader
//! threshold, `>=20` vs `>=23`).
//!
//! ### cf01-cf08 tables (base-function compositions) -- quoted from the C
//! (`cf01`..`cf08` bodies, `cf_num`/`delta`/`bias`/component-call-order/
//! post-eval-rescale) AND CROSS-CHECKED against the report (section "D.
//! Composition Functions", `pdftotext -layout`)
//!
//! ```text
//! void cf01 (...) { cf_num=3; delta={10,20,30}; bias={0,100,200};
//!     i=0; rosenbrock_func(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],1,r_flag);
//!     i=1; ellips_func(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],1,r_flag); fit[i]=10000*fit[i]/1e+10;
//!     i=2; rastrigin_func(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],1,r_flag);
//!     cf_cal(x, f, nx, Os, delta,bias,fit,cf_num); }
//! void cf02 (...) { cf_num=3; delta={10,20,30}; bias={0,100,200};
//!     i=0; rastrigin_func(...,r_flag);
//!     i=1; griewank_func(...,r_flag);  fit[i]=1000*fit[i]/100;
//!     i=2; schwefel_func(...,r_flag);
//!     cf_cal(...); }
//! void cf03 (...) { cf_num=4; delta={10,20,30,40}; bias={0,100,200,300};
//!     i=0; rosenbrock_func(...,r_flag);
//!     i=1; ackley_func(...,r_flag);    fit[i]=1000*fit[i]/100;
//!     i=2; schwefel_func(...,r_flag);
//!     i=3; rastrigin_func(...,r_flag);
//!     cf_cal(...); }
//! void cf04 (...) { cf_num=4; delta={10,20,30,40}; bias={0,100,200,300};
//!     i=0; ackley_func(...,r_flag);    fit[i]=1000*fit[i]/100;
//!     i=1; ellips_func(...,r_flag);    fit[i]=10000*fit[i]/1e+10;
//!     i=2; griewank_func(...,r_flag);  fit[i]=1000*fit[i]/100;
//!     i=3; rastrigin_func(...,r_flag);
//!     cf_cal(...); }
//! void cf05 (...) { cf_num=5; delta={10,20,30,40,50}; bias={0,100,200,300,400};
//!     i=0; rastrigin_func(...,r_flag);  fit[i]=10000*fit[i]/1e+3;
//!     i=1; happycat_func(...,r_flag);   fit[i]=1000*fit[i]/1e+3;
//!     i=2; ackley_func(...,r_flag);     fit[i]=1000*fit[i]/100;
//!     i=3; discus_func(...,r_flag);     fit[i]=10000*fit[i]/1e+10;
//!     i=4; rosenbrock_func(...,r_flag);
//!     cf_cal(...); }
//! void cf06 (...) { cf_num=5; delta={10,20,20,30,40}; bias={0,100,200,300,400};
//!     i=0; escaffer6_func(...,r_flag);  fit[i]=10000*fit[i]/2e+7;
//!     i=1; schwefel_func(...,r_flag);
//!     i=2; griewank_func(...,r_flag);   fit[i]=1000*fit[i]/100;
//!     i=3; rosenbrock_func(...,r_flag);
//!     i=4; rastrigin_func(...,r_flag);  fit[i]=10000*fit[i]/1e+3;
//!     cf_cal(...); }
//! void cf07 (...) { cf_num=6; delta={10,20,30,40,50,60}; bias={0,100,200,300,400,500};
//!     i=0; hgbat_func(...,r_flag);      fit[i]=10000*fit[i]/1000;
//!     i=1; rastrigin_func(...,r_flag);  fit[i]=10000*fit[i]/1e+3;
//!     i=2; schwefel_func(...,r_flag);   fit[i]=10000*fit[i]/4e+3;
//!     i=3; bent_cigar_func(...,r_flag); fit[i]=10000*fit[i]/1e+30;
//!     i=4; ellips_func(...,r_flag);     fit[i]=10000*fit[i]/1e+10;
//!     i=5; escaffer6_func(...,r_flag);  fit[i]=10000*fit[i]/2e+7;
//!     cf_cal(...); }
//! void cf08 (...) { cf_num=6; delta={10,20,30,40,50,60}; bias={0,100,200,300,400,500};
//!     i=0; ackley_func(...,r_flag);     fit[i]=1000*fit[i]/100;
//!     i=1; griewank_func(...,r_flag);   fit[i]=1000*fit[i]/100;
//!     i=2; discus_func(...,r_flag);     fit[i]=10000*fit[i]/1e+10;
//!     i=3; rosenbrock_func(...,r_flag);
//!     i=4; happycat_func(...,r_flag);   fit[i]=1000*fit[i]/1e+3;
//!     i=5; escaffer6_func(...,r_flag);  fit[i]=10000*fit[i]/2e+7;
//!     cf_cal(...); }
//! ```
//! (`...` elides the repeated `x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx]`
//! parameter prefix each call shares, for readability -- every ellided call
//! passes the SAME arguments the fully-spelled-out `cf01`/`i=0` line above
//! shows, confirmed by reading each body directly, this task's report has
//! the byte-for-byte full transcript of all ten functions.)
//!
//! | fid | `cf0N` | `cf_num` | `delta` | `bias` | components `(fn, rescale)` (rotate ALWAYS true) |
//! |---|---|---|---|---|---|
//! | 21 | `cf01` | 3 | `[10,20,30]` | `[0,100,200]` | `(Rosenbrock,1.0)`, `(Ellips,1e-6)`, `(Rastrigin,1.0)` |
//! | 22 | `cf02` | 3 | `[10,20,30]` | `[0,100,200]` | `(Rastrigin,1.0)`, `(Griewank,10.0)`, `(Schwefel,1.0)` |
//! | 23 | `cf03` | 4 | `[10,20,30,40]` | `[0,100,200,300]` | `(Rosenbrock,1.0)`, `(Ackley,10.0)`, `(Schwefel,1.0)`, `(Rastrigin,1.0)` |
//! | 24 | `cf04` | 4 | `[10,20,30,40]` | `[0,100,200,300]` | `(Ackley,10.0)`, `(Ellips,1e-6)`, `(Griewank,10.0)`, `(Rastrigin,1.0)` |
//! | 25 | `cf05` | 5 | `[10,20,30,40,50]` | `[0,100,200,300,400]` | `(Rastrigin,10.0)`, `(HappyCat,1.0)`, `(Ackley,10.0)`, `(Discus,1e-6)`, `(Rosenbrock,1.0)` |
//! | 26 | `cf06` | 5 | `[10,20,20,30,40]` | `[0,100,200,300,400]` | `(EScaffer6,5e-4)`, `(Schwefel,1.0)`, `(Griewank,10.0)`, `(Rosenbrock,1.0)`, `(Rastrigin,10.0)` |
//! | 27 | `cf07` | 6 | `[10,20,30,40,50,60]` | `[0,100,200,300,400,500]` | `(HGBat,10.0)`, `(Rastrigin,10.0)`, `(Schwefel,2.5)`, `(BentCigar,1e-26)`, `(Ellips,1e-6)`, `(EScaffer6,5e-4)` |
//! | 28 | `cf08` | 6 | `[10,20,30,40,50,60]` | `[0,100,200,300,400,500]` | `(Ackley,10.0)`, `(Griewank,10.0)`, `(Discus,1e-6)`, `(Rosenbrock,1.0)`, `(HappyCat,1.0)`, `(EScaffer6,5e-4)` |
//!
//! Every `delta`/`bias`/component-name/component-order/rescale value above
//! was independently re-derived from the C's own `fit[i]=K*fit[i]/D` lines
//! (e.g. `10000/1e10=1e-6`, `10000/1e30=1e-26`, `1000/100=10.0`) -- NOT
//! copied from the report's own printed lambda arrays, because two of the
//! report's own composition-function entries are independently VERIFIED
//! WRONG (below); the C is authoritative throughout (standing
//! PROVENANCE-FIRST ruling).
//!
//! Report (PDF `pdftotext -layout`, section "D. Composition Functions",
//! this section's OWN item numbers `20)`.."29)" use the SAME CONTIGUOUS,
//! no-F2-gap numbering Table I uses -- NOT section 1.3's gapped
//! code-`func_num` numbering (module doc's "Report-table vs. code
//! `func_num` numbering" section above already establishes this split for
//! fid 1-10; it recurs here identically: report item `20)` = code
//! `func_num` 21, ..., item `29)` = code `func_num` 30):
//! ```text
//! 20) Composition Function 1: N=3, sigma=[10,20,30], lambda=[1,1e-6,1], bias=[0,100,200]
//!     g1: Rosenbrock's Function F4'  g2: High Conditioned Elliptic Function F1'  g3 Rastrigin's Function F4'
//! 21) Composition Function 2: N=3, sigma=[10,20,30], lambda=[1,10,1], bias=[0,100,200]
//!     g1: Rastrigin's Function F5'  g2: Griewank's Function F15'  g3 Modifed Schwefel's Function F10'
//! 22) Composition Function 3: N=4, sigma=[10,20,30,40], lambda=[1,10,1,1], bias=[0,100,200,300]
//!     g1: Rosenbrock's Function F4'  g2: Ackley's Function F13'  g3: Modified Schwefel's Function F10'  g4: Rastrigin's Function F5'
//! 23) Composition Function 4: N=4, sigma=[10,20,30,40], lambda=[10,1e-6,10,1], bias=[0,100,200,300]
//!     g1: Ackley's Function F13'  g2: High Conditioned Elliptic Function F11'  g3: Girewank Function F15'  g4: Rastrigin's Function F5'
//! 24) Composition Function 5: N=5, sigma=[10,20,30,40,50], lambda=[10,1,10,1e-6,1], bias=[0,100,200,300,400]
//!     g1: Rastrigin's Function F5'  g2: Happycat Function F17'  g3: Ackley Function F13'  g4: Discus Function F12'  g5: Rosenbrock's Function F4'
//! 25) Composition Function 6: N=5, sigma=[10,20,20,30,40], lambda=[1e-26,10,1e-6,10,5e-4], bias=[0,100,200,300,400]
//!     g1: Expanded Scaffer's F6 Function F6'  g2: Modified Schwefel's Function F10'  g3: Griewank's Function F15'  g4: Rosenbrock's Function F4'  g5: Rastrigin's Function F5'
//! 26) Composition Function 7: N=6, sigma=[10,20,30,40,50,60], lambda=[10,10,2.5,1e-26,1e-6,5e-4], bias=[0,100,200,300,400,500]
//!     g1: HGBat Function F18'  g2: Rastrigin's Function F5'  g3: Modified Schwefel's Function F10'  g4: Bent-Cigar Function F11'  g4(sic): High Conditioned Elliptic Function F11'  g5(sic): Expanded Scaffer's F6 Function F6'
//! 27) Composition Function 8: N=6, sigma=[10,20,30,40,50,60], lambda=[10,10,1e-6,1,1,5e-4], bias=[0,100,200,300,400,500]
//!     g1: Ackley's Function F13'  g2: Griewank's Function F15'  g3: Discus Function F12'  g4: Rosenbrock's Function F4'  g4(sic): HappyCat Function F17'  g5(sic): Expanded Scaffer's F6 Function F6'
//! 28) Composition Function 10 (SIC -- swapped title, see VERIFIED DISCREPANCY below): N=3, sigma=[10,30,50], lambda=[1,1,1], bias=[0,100,200]
//!     g1: Hybrid Function 5 F5'  g2: Hybrid Function 6 F6'  g3: Hybrid Function 7 F7'
//! 29) Composition Function 9 (SIC -- swapped title, see VERIFIED DISCREPANCY below): N=3, sigma=[10,30,50], lambda=[1,1,1], bias=[0,100,200]
//!     g1: Hybrid Function 5 F5'  g2: Hybrid Function 8 F8'  g3: Hybrid Function 9 F9'
//! ```
//!
//! ### VERIFIED DISCREPANCY 1: item `25)`'s (fid 26, `cf06`) printed `lambda`
//! array does not match its own printed component order, or the C, at all
//! (found via cross-check, not merely transcribed) -- `sigma`/`bias`/
//! component ORDER are all correct, only `lambda` is wrong
//!
//! Item `25)`'s own component list (`g1..g5`: EScaffer6, Schwefel,
//! Griewank, Rosenbrock, Rastrigin) matches `cf06`'s C call order EXACTLY
//! (quoted above). But the report prints `lambda=[1e-26, 10, 1e-6, 10,
//! 5e-4]` for that SAME order -- the C's own `fit[i]=K*fit[i]/D` lines for
//! THOSE FIVE components (in that order) compute `[5e-4, 1.0, 10.0, 1.0,
//! 10.0]` (escaffer6: `10000/2e7=5e-4`; schwefel: no rescale line, `1.0`;
//! griewank: `1000/100=10.0`; rosenbrock: no rescale line, `1.0`;
//! rastrigin: `10000/1e3=10.0`) -- NONE of the report's five printed values
//! match the C-derived ones in ANY permutation of this component list
//! (`1e-26` and `1e-6` appear nowhere in `cf06`'s own rescale lines at all;
//! those two exponents only appear in `cf07`'s own lambda, `bent_cigar`
//! `1e-26` and `ellips` `1e-6` -- plausibly a PDF table-row bleed from the
//! adjacent item, though this task does not need to diagnose the printing
//! mechanism to apply the standing ruling). `sezgi` follows the C-derived
//! `[5e-4, 1.0, 10.0, 1.0, 10.0]` (this module doc's fid-26/`cf06` table row
//! above), confirmed correct by this task's OWN `x=comp_shift[0]` pin and
//! random-point probes (below) both matching the compiled C.
//!
//! ### VERIFIED DISCREPANCY 2: items `28)`/`29)`'s titles are SWAPPED
//! relative to Table I's own naming (and relative to the C's own `cf09`/
//! `cf10` function names) -- their CONTENT (component lists) is correct and
//! unambiguous, only the two-word title text is wrong
//!
//! Table I (module doc's numbering section above) names row 28 (code
//! `func_num` 29) "Composition Function 9" and row 29 (code `func_num` 30)
//! "Composition Function 10" -- consistent with the C's own function names
//! (`cf09` is dispatched at `case 29`, `cf10` at `case 30`, quoted in this
//! section's dispatch block above). But section D's OWN item `28)` (SAME
//! position, `func_num` 29) is titled "Composition Function **10**", and
//! item `29)` (`func_num` 30) is titled "Composition Function **9**" -- the
//! two titles are swapped relative to Table I and the C's own naming.
//! Despite the swapped TITLES, each item's CONTENT is unambiguous and
//! matches the C's actual component list AT THAT SEQUENTIAL POSITION
//! exactly: item `28)` (func_num 29) lists `g1..g3` = Hybrid Function
//! 5/6/7, which matches `cf09`'s C body (`hf05, hf06, hf07`, quoted below)
//! bit-for-bit; item `29)` (func_num 30) lists Hybrid Function 5/8/9, which
//! matches `cf10`'s C body (`hf05, hf08, hf09`) bit-for-bit. `sezgi` follows
//! the C's function-name-to-`func_num` mapping (`cf09`->fid 29, `cf10`->fid
//! 30) throughout this module (`Cec2017::composition_hybrid_spec`'s own doc
//! and code), never the report's swapped title text -- this is purely a
//! documentation-quality finding (the CONTENT was never ambiguous), noted
//! here per this task's PROVENANCE-FIRST reporting requirement.
//!
//! ### cf09/cf10 -- hybrid-in-composition (the report's own note, quoted
//! above the per-fid list in T6's/`cec2014`'s own precedent: "In CEC'14,
//! the hybrid functions are also used as the basic functions for
//! composition functions") -- quoted from the C, CROSS-CHECKED against the
//! report (content, not the swapped titles above)
//!
//! ```text
//! void cf09 (double *x,double *f,int nx,double *Os,double *Mr,int *SS,int r_flag) {
//!     cf_num=3; delta={10,30,50}; bias={0,100,200};
//!     i=0; hf05(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],&SS[i*nx],1,r_flag);
//!     i=1; hf06(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],&SS[i*nx],1,r_flag);
//!     i=2; hf07(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],&SS[i*nx],1,r_flag);
//!     cf_cal(x, f, nx, Os, delta,bias,fit,cf_num); }
//! void cf10 (double *x,double *f,int nx,double *Os,double *Mr,int *SS,int r_flag) {
//!     cf_num=3; delta={10,30,50}; bias={0,100,200};
//!     i=0; hf05(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],&SS[i*nx],1,r_flag);
//!     i=1; hf08(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],&SS[i*nx],1,r_flag);
//!     i=2; hf09(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],&SS[i*nx],1,r_flag);
//!     cf_cal(x, f, nx, Os, delta,bias,fit,cf_num); }
//! ```
//! `lambda=[1,1,1]` for both (report, unaffected by the title-swap bug
//! above -- content matches C exactly) confirms NEITHER `cf09` NOR `cf10`
//! post-scales any `fit[i]` (unlike `cf01`-`cf08`'s `fit[i]=K*fit[i]/D`
//! lines) -- `cf_cal` adds `bias[i]` only. `Os[i*nx]`/`Mr[i*nx*nx]`/
//! `SS[i*nx]` give EACH hybrid component its OWN FULL shift+rotate+shuffle
//! (a DIFFERENT `(o,M,S)` triple per component `i`), exactly the same shape
//! `cec2014`'s own `cf07`/`cf08` already established --
//! [`Cec2017::composition_hybrid_fitness`] threads this through explicitly,
//! reusing [`Cec2017::hybrid_fitness`] (T7's helper, unmodified) at the
//! matching `hf0N`<->fid mapping T7's own dispatch section already
//! establishes: `hf05`->fid 15, `hf06`->fid 16, `hf07`->fid 17 (`cf09`);
//! `hf05`->fid 15, `hf08`->fid 18, `hf09`->fid 19 (`cf10`). Neither `hf05`,
//! `hf06`, `hf07`, `hf08`, nor `hf09` includes either of T7's two
//! VERIFIED-BUGGY components (`SchafferF7Buggy` is only in `hf04`/fid 14 and
//! `hf10`/fid 20; `BiRastriginBuggy` is only in `hf03`/fid 13) -- confirmed
//! by re-reading T7's own `hybrid_spec` table (module doc above): NONE of
//! `cf09`/`cf10`'s five component hybrids (`hf05..hf09`) are among fid
//! `{13,14,20}`, so this task's composition-of-hybrids wiring inherits
//! NEITHER T7 bug, unlike a hypothetical composition that reused `hf03`,
//! `hf04`, or `hf10`.
//!
//! ### `cf_cal` weight formula -- SAME shared [`crate::cec_basics::composition_weights`]
//! function T6's/`cec2014`'s own T4 section already verifies BIT-FOR-BIT
//! identical to this suite's own `cf_cal` (re-diffed this task against the
//! quoted body in `cec17_test_func.cpp` line 1604 -- zero difference from
//! `cec2014`'s copy, same `#define INF 1.0e99` sentinel) -- no new
//! weight-formula code this task, [`Cec2017::composition_fitness`] and
//! [`Cec2017::composition_hybrid_fitness`] both call
//! [`crate::cec_basics::composition_weights`] directly.
//!
//! ### New component variants ([`HybridComponent::Griewank`],
//! [`HybridComponent::HappyCat`]) -- the only two base functions this
//! suite's fid 1-20 (T6/T7) never needed, both ALREADY exist in
//! `crate::cec_basics` (`griewank_base`/`happycat_base`, extracted for
//! `cec2014`'s own T2/T4) -- probe-verified against `griewank_func`
//! (`sh_rate=600.0/100.0`) and `happycat_func` (`sh_rate=5.0/100.0`, its own
//! internal `z[i]-=1.0` "shift to origin" already baked into
//! `cec_basics::happycat_base`, same convention `HybridComponent::HappyCat`
//! already uses for `cec2014`) reading the C bodies directly (this module's
//! own T8 dispatch quote above). Every OTHER component `cf01`-`cf10` needs
//! (`Rosenbrock`, `Ellips`, `Rastrigin`, `Schwefel`, `Ackley`, `Discus`,
//! `BentCigar`, `HGBat`, `EScaffer6`) already exists as a
//! [`HybridComponent`] variant from T7 -- no other new variant needed.
//!
//! ### `x = comp_shift[0]` pin (Step 2) for fid 21-30: SAME "INF-sentinel
//! absorption" argument `cec2014/mod.rs`'s own T4 section already makes
//! (re-derived here for this suite's own ten tables, VERIFIED against the
//! compiled C for every fid 21-30 x every dim in {10,30}, this task's own
//! probe: `t8_xo1_pin.py`)
//!
//! At `x = comp_shift[0]`: component 1's raw squared distance `D_0 = 0`
//! exactly, so `cf_cal`'s `w_0 = INF = 1e99` absorbs `w_sum`, making
//! `w_0/w_sum` round to EXACTLY `1.0`; component 1's own shift-rotated
//! argument is the all-zero vector, so its base evaluation is EXACTLY `0`
//! (every `cf0N`'s FIRST component here -- Rosenbrock/Rastrigin x2/Ackley/
//! Rastrigin/EScaffer6/HGBat/Ackley/`hf05`(x2) -- is one of the bases this
//! module doc's earlier pin sections, or T7's hybrid pin section, already
//! prove `=0` at an all-zero argument), and `bias[0]==0` for EVERY ONE of
//! the ten tables above -- so component 1's raw contribution is `0*1.0=0`
//! exactly, and the other components' contributions are picoscopically
//! small (SAME caveat as every earlier composition pin in this crate) but
//! empirically absorbed by `F_i*`'s ULP. Measured (compiled C, `x=
//! comp_shift[0]`, `%.20f`): all TWENTY `(fid,dim)` pairs in `{21..=30} x
//! {10,30}` print the exact integer `100*fid`, zero fractional residue
//! (this task's report has the transcript) -- confirmed by
//! `fid_21_to_30_x_eq_comp_shift0_is_exactly_f_star_dim10_and_dim30` below.
//!
//! ### Measured deviations (this task)
//!
//! 20-point `x=comp_shift[0]` pin probe (above): EXACT (`0` residue) at
//! every point, both dims. 60-probe random-point cross-check (`composition_
//! random_probe_*` tests below, seed `20170721`, this task's report has the
//! full transcript): same `1e-9` [`tests::assert_close`] relative tolerance
//! reused from every earlier section of this module; every probe passed.
//!
//! ## Struct/impl notes
//!
//! [`Cec2017::new`] rejects `dim` outside `{10,30}` (vendoring scope) and
//! `fid` outside `{1} ∪ {3..=30}` (T6+T7+T8's combined, now COMPLETE
//! functional scope: unimodal/simple-multimodal F1/F3-F10, hybrid F11-F20,
//! composition F21-F30 -- `fid==2` gets its own dedicated
//! [`Cec2017Error::Withdrawn`], everything else outside `{1,3..=30}` (`0`,
//! `31` and up) gets [`Cec2017Error::UnknownFid`], mirrors
//! `cec2014/mod.rs`'s own now-complete T2->T3->T4 `UnknownFid` scoping).

use sezgi_core::problem::Problem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

mod data;

#[derive(Debug, thiserror::Error)]
pub enum Cec2017Error {
    #[error(
        "fid must be 1 or in 3..=30 (T6+T7+T8's combined, now complete functional scope of the \
         CEC 2017 suite: unimodal F1/F3, simple-multimodal F4-F10, hybrid F11-F20, composition \
         F21-F30), got {0}"
    )]
    UnknownFid(u32),
    #[error(
        "fid 2 (\"Sum of Different Powers\") was officially withdrawn from the CEC 2017 suite; \
         the reference C's own case 2 prints \"Error: This function (F2) has been deleted\" and \
         leaves its result unset -- sezgi returns this dedicated error instead, never a garbage \
         f64 (module doc's F2 ruling section has the full quoted C and probe transcript)"
    )]
    Withdrawn,
    #[error(
        "dim must be one of {{10,30}} (this crate vendors only these two dimensions of the \
         CEC 2017 input_data set -- the official suite also supports 2,20,50,100), got {0}"
    )]
    BadDim(usize),
}

/// One instance of a CEC 2017 unimodal (fid 1,3) or simple-multimodal (fid
/// 4-10) function: embedded official shift vector `o` and rotation matrix
/// `M` for the requested `(fid, dim)`, plus the pinned `F_i* = 100*fid`
/// bias (module doc's dispatch section). See the module doc for each fid's
/// exact shift/scale/rotate pipeline, including the verified fid 6/8/9
/// discrepancies between the printed report and the vendored reference C
/// code this module actually follows.
pub struct Cec2017 {
    fid: u32,
    dim: usize,
    o: Vec<f64>,
    m: Vec<Vec<f64>>,
    /// fid 11-20 only (M3-6 T7): 0-based shuffle permutation applied to the
    /// shift-rotated vector before segmenting into hybrid components
    /// (module doc's T7 section: `y[i] = z[shuffle[i]]`). Empty for fid 1,
    /// 3-10 and fid 21-30 (fid 29-30 use [`Self::comp_shuffle`] instead, one
    /// permutation PER component).
    shuffle: Vec<usize>,
    /// fid 21-30 only (M3-6 T8): each component's OWN shift row
    /// (`comp_shift[idx]`, `cf_num` rows total -- 3, 4, 5, or 6). Empty for
    /// fid 1, 3-20. `o` mirrors `comp_shift[0]` for API uniformity with the
    /// `x = o` pin tests (same `cec2014::Cec2014`'s own fid-23-30 struct doc
    /// decision, quoted there).
    comp_shift: Vec<Vec<f64>>,
    /// fid 21-30 only: each component's OWN `dim`x`dim` rotation matrix,
    /// parallel to [`Self::comp_shift`]. Empty for fid 1, 3-20.
    comp_rotation: Vec<Vec<Vec<f64>>>,
    /// fid 29-30 only (M3-6 T8, `cf09`/`cf10`'s hybrid-in-composition
    /// wiring): each of the 3 hybrid components' OWN 0-based shuffle
    /// permutation, parallel to [`Self::comp_shift`]. Empty for every other
    /// fid (including fid 21-28, whose components are plain base functions
    /// with no shuffle at all).
    comp_shuffle: Vec<Vec<usize>>,
    space: SearchSpace,
}

impl Cec2017 {
    /// `fid` must be `1` or in `3..=30` ([`Cec2017Error::UnknownFid`]
    /// otherwise); `fid==2` specifically returns
    /// [`Cec2017Error::Withdrawn`] (module doc's F2 ruling). `dim` must be
    /// one of `{10,30}` ([`Cec2017Error::BadDim`] otherwise, module doc's
    /// "Vendoring scope" section).
    pub fn new(fid: u32, dim: usize) -> Result<Cec2017, Cec2017Error> {
        if fid == 2 {
            return Err(Cec2017Error::Withdrawn);
        }
        if !(fid == 1 || (3..=30).contains(&fid)) {
            return Err(Cec2017Error::UnknownFid(fid));
        }
        if !matches!(dim, 10 | 30) {
            return Err(Cec2017Error::BadDim(dim));
        }
        let (o, m, shuffle, comp_shift, comp_rotation, comp_shuffle) = if (21..=30).contains(&fid) {
            let cf_num = Self::composition_cf_num(fid);
            let comp_shift = data::composition_shift_blocks(fid, dim, cf_num);
            let comp_rotation = data::composition_rotation_blocks(fid, dim, cf_num);
            let comp_shuffle = if (29..=30).contains(&fid) {
                data::composition_shuffle_blocks(fid, dim, cf_num)
            } else {
                Vec::new()
            };
            // `o` mirrors component 1's own shift (struct doc, same
            // `cec2014::Cec2014` decision for its own fid 23-30) so the
            // `x = o` pin tests can stay uniform across the whole `{1,
            // 3..=30}` range.
            let o = comp_shift[0].clone();
            (o, Vec::new(), Vec::new(), comp_shift, comp_rotation, comp_shuffle)
        } else {
            let o = data::shift_vector(fid, dim);
            let m = data::rotation_matrix(fid, dim);
            let shuffle =
                if (11..=20).contains(&fid) { data::shuffle_indices(fid, dim) } else { Vec::new() };
            (o, m, shuffle, Vec::new(), Vec::new(), Vec::new())
        };
        let space = SearchSpace::new(vec![Block::Float { lo: -100.0, hi: 100.0, n: dim }])
            .expect("CEC 2017 bounds (-100 < 100) are always valid");
        Ok(Self { fid, dim, o, m, shuffle, comp_shift, comp_rotation, comp_shuffle, space })
    }

    /// This composition fid's own `cf_num` (module doc's T8 cf01-cf10
    /// table, transcribed from each `cf0N`'s own local `cf_num` -- NOT the
    /// loader's unrelated hardcoded `cf_num=10`, module doc's dedicated note
    /// on that distinction). `fid` must be `21..=30` (caller's
    /// responsibility).
    fn composition_cf_num(fid: u32) -> usize {
        match fid {
            21 | 22 | 29 | 30 => 3,
            23 | 24 => 4,
            25 | 26 => 5,
            27 | 28 => 6,
            other => unreachable!("composition_cf_num called with unsupported fid {other}"),
        }
    }

    pub fn fid(&self) -> u32 { self.fid }
    pub fn dim(&self) -> usize { self.dim }

    /// The dispatch's own `f[i]+=<N>00.0` lines (module doc's dispatch
    /// quote), `F_i* = 100*fid` -- the C's OWN gapped `func_num`-based
    /// bias, NOT the report Table I's contiguous-renumbering bias (module
    /// doc's numbering-divergence section).
    pub fn f_star(&self) -> f64 { 100.0 * f64::from(self.fid) }

    /// `(x - o) * sr`, optionally rotated by `self.m` (module doc's
    /// `sr_func` quote via `cec_basics`, byte-identical to
    /// `cec2014`/`cec2022`'s own -- delegates to the SAME shared helper, no
    /// new shift/rotate arithmetic this task).
    fn shift_scale_rotate(&self, xs: &[f64], sr: f64, rotate: bool) -> Vec<f64> {
        crate::cec_basics::shift_scale_rotate(xs, &self.o, &self.m, sr, rotate)
    }

    /// The full pipeline (shift/scale/rotate + base function + `F_i*`
    /// bias) for one already-flattened decision vector, per this module's
    /// doc ("Reused bases" table has each fid's exact `sr`/rotate choice,
    /// including the verified fid 6/8 discrepancies).
    fn eval_one(&self, xs: &[f64]) -> f64 {
        use crate::cec_basics as cb;
        let value = match self.fid {
            1 => cb::bent_cigar_base(&self.shift_scale_rotate(xs, 1.0, true)),
            3 => cb::f1_base(&self.shift_scale_rotate(xs, 1.0, true)),
            4 => {
                let mut z = self.shift_scale_rotate(xs, 2.048 / 100.0, true);
                for zi in &mut z {
                    *zi += 1.0;
                }
                cb::f2_base(&z)
            }
            5 => cb::f4_base(&self.shift_scale_rotate(xs, 5.12 / 100.0, true)),
            // No rotation applied to the FORMULA (module doc's fid-6
            // discrepancy note: `schaffer_F7_func` reads the shift-only
            // `y`, silently discarding `sr_func`'s rotated output, even
            // though `r_flag=1` is passed and `M` IS loaded/vendored).
            6 => cb::f16_schaffer_f7_base(&self.shift_scale_rotate(xs, 1.0, false)),
            7 => f7_bi_rastrigin(xs, &self.o, &self.m, true),
            // Dead non-continuous pre-transform (module doc's fid-8
            // discrepancy note) -- reduces to plain `f4_base`, same as fid
            // 5, just with fid 8's OWN vendored shift/rotation data.
            8 => cb::f4_base(&self.shift_scale_rotate(xs, 5.12 / 100.0, true)),
            9 => {
                // sezgi decision: reuse `cb::f5_base` with a `-1.0`
                // pre-shift instead of adding a second Levy base --
                // exactly equivalent algebraically (see below), keeps
                // one Levy formula in the codebase.
                // `cb::f5_base` (extracted from CEC 2022's OWN Levy) uses
                // `w=1+zi/4.0` -- CEC 2017's `levy_func` uses `w=1+(zi-
                // 1.0)/4.0` (module doc's fid-9 row: DIFFERENT constant
                // across generations, same name, per this task's own
                // probe-verification finding, not assumed from the name).
                // Pre-subtracting 1.0 from every `z[i]` before calling
                // `f5_base` makes its `1+zi/4.0` compute the SAME `w` CEC
                // 2017's own formula does -- the mirror-image of fid 4's
                // `+1.0` compensation for `f2_base` above.
                let mut z = self.shift_scale_rotate(xs, 1.0, true);
                for zi in &mut z {
                    *zi -= 1.0;
                }
                cb::f5_base(&z)
            }
            10 => cb::schwefel_base(&self.shift_scale_rotate(xs, 1000.0 / 100.0, true)),
            11..=20 => {
                // Module doc's T7 section: outer `sr_func` is ALWAYS
                // `sh_rate=1.0, s_flag=1,r_flag=1` for every `hf0N`, THEN
                // shuffle, THEN segment/dispatch to components.
                let z = self.shift_scale_rotate(xs, 1.0, true);
                let y: Vec<f64> = (0..self.dim).map(|i| z[self.shuffle[i]]).collect();
                Self::hybrid_fitness(self.fid, self.dim, &y, &self.o)
            }
            21..=28 => self.composition_fitness(xs),
            29 | 30 => self.composition_hybrid_fitness(xs),
            other => unreachable!("Cec2017::new rejects fid outside {{1,3..=30}}, got {other}"),
        };
        value + self.f_star()
    }

    /// One hybrid function's (fid 11-20) component-sum (BEFORE `F_i*`),
    /// given the already shift-rotate-shuffled `y` (module doc's T7
    /// section: `y[i] = z[shuffle[i]]`) and the RAW (unshuffled, unrotated)
    /// shift vector `o` (needed ONLY by the `BiRastriginBuggy` arm's own
    /// verified-buggy sign-flip reference, module doc's "VERIFIED
    /// DISCREPANCY 2" section). A separate associated function (not inlined
    /// into [`Self::eval_one`]) so the segmentation/shuffle logic can be
    /// probed directly on a hand-built `y`, independent of the shift/
    /// rotate/embedded-data plumbing -- mirrors `cec2014::Cec2014::
    /// hybrid_fitness`'s own separation.
    fn hybrid_fitness(fid: u32, dim: usize, y: &[f64], o: &[f64]) -> f64 {
        use crate::cec_basics as cb;
        let (gp, comps) = Self::hybrid_spec(fid);
        let sizes = cb::segment_sizes(gp, dim);
        let starts = cb::segment_starts(&sizes);
        comps
            .iter()
            .enumerate()
            .map(|(idx, &comp)| {
                let start = starts[idx];
                let len = sizes[idx];
                match comp {
                    // Module doc's "VERIFIED DISCREPANCY 1": reads a PREFIX
                    // of the WHOLE shuffled `y` (`y[0..len]`), NOT this
                    // component's own assigned segment
                    // `y[start..start+len]` -- replicates the reference C
                    // bug exactly (same structural shape
                    // `cec2022::HybridComponent::SchafferF7Buggy` already
                    // established for that suite's own fid-7 hybrid).
                    HybridComponent::SchafferF7Buggy => cb::f16_schaffer_f7_base(&y[0..len]),
                    // Module doc's "VERIFIED DISCREPANCY 2": the segment
                    // CONTENT is this component's own correctly-offset
                    // `y[start..start+len]`, but the sign-flip reference
                    // reads `o[0..len]` (a PREFIX of the ORIGINAL,
                    // unshuffled shift vector), NOT `o[start..start+len]`.
                    HybridComponent::BiRastriginBuggy => {
                        bi_rastrigin_hybrid_component(&y[start..start + len], &o[0..len])
                    }
                    _ => {
                        let seg: Vec<f64> =
                            y[start..start + len].iter().map(|&yi| yi * comp.sh_rate()).collect();
                        comp.eval(&seg)
                    }
                }
            })
            .sum()
    }

    /// `fid`'s (`11..=20`) proportions `p` and component list, in call order
    /// (module doc's T7 "Hybrid composition tables" section, transcribed
    /// verbatim from the cross-checked C, all ten `hf0N` read directly).
    fn hybrid_spec(fid: u32) -> (&'static [f64], &'static [HybridComponent]) {
        use HybridComponent::*;
        match fid {
            11 => (&[0.2, 0.4, 0.4], &[Zakharov, Rosenbrock, Rastrigin]),
            12 => (&[0.3, 0.3, 0.4], &[Ellips, Schwefel, BentCigar]),
            13 => (&[0.3, 0.3, 0.4], &[BentCigar, Rosenbrock, BiRastriginBuggy]),
            14 => (&[0.2, 0.2, 0.2, 0.4], &[Ellips, Ackley, SchafferF7Buggy, Rastrigin]),
            15 => (&[0.2, 0.2, 0.3, 0.3], &[BentCigar, HGBat, Rastrigin, Rosenbrock]),
            16 => (&[0.2, 0.2, 0.3, 0.3], &[EScaffer6, HGBat, Rosenbrock, Schwefel]),
            17 => (&[0.1, 0.2, 0.2, 0.2, 0.3], &[Katsuura, Ackley, GrieRosen, Schwefel, Rastrigin]),
            18 => (&[0.2, 0.2, 0.2, 0.2, 0.2], &[Ellips, Ackley, Rastrigin, HGBat, Discus]),
            19 => {
                (&[0.2, 0.2, 0.2, 0.2, 0.2], &[BentCigar, Rastrigin, GrieRosen, Weierstrass, EScaffer6])
            }
            20 => (
                &[0.1, 0.1, 0.2, 0.2, 0.2, 0.2],
                &[HGBat, Katsuura, Ackley, Rastrigin, Schwefel, SchafferF7Buggy],
            ),
            other => unreachable!("hybrid_spec called with unsupported fid {other}"),
        }
    }

    /// One composition function's (fid 21-28, [`Self::composition_spec`])
    /// `(delta, bias, components)` triple -- `components[i]` is `(base
    /// function, this component's own post-eval rescale)`. Unlike
    /// `cec2014::Cec2014::composition_spec`, no per-component `rotate: bool`
    /// field is needed here: module doc's T8 dispatch note verifies EVERY
    /// component in ALL TEN `cf01`..`cf10` passes the `r_flag` PARAMETER
    /// through unmodified (no hardcoded-`0` site exists in this suite's own
    /// composition functions, unlike `cec2014`'s `cf01`/`cf02`) -- so every
    /// component here is always rotated, and [`Self::composition_fitness`]
    /// hardcodes `rotate=true` for all of them.
    // sezgi decision (M3-6 T8): named to keep the signature readable
    // (clippy's `type_complexity`), same shape `cec2014::Cec2014`'s own
    // `composition_spec` alias uses (minus the unneeded `rotate` bool,
    // above).
    #[allow(clippy::type_complexity)]
    fn composition_spec(fid: u32) -> (&'static [f64], &'static [f64], &'static [(HybridComponent, f64)]) {
        use HybridComponent::*;
        match fid {
            21 => (
                &[10.0, 20.0, 30.0],
                &[0.0, 100.0, 200.0],
                &[(Rosenbrock, 1.0), (Ellips, 1.0e-6), (Rastrigin, 1.0)],
            ),
            22 => (
                &[10.0, 20.0, 30.0],
                &[0.0, 100.0, 200.0],
                &[(Rastrigin, 1.0), (Griewank, 10.0), (Schwefel, 1.0)],
            ),
            23 => (
                &[10.0, 20.0, 30.0, 40.0],
                &[0.0, 100.0, 200.0, 300.0],
                &[(Rosenbrock, 1.0), (Ackley, 10.0), (Schwefel, 1.0), (Rastrigin, 1.0)],
            ),
            24 => (
                &[10.0, 20.0, 30.0, 40.0],
                &[0.0, 100.0, 200.0, 300.0],
                &[(Ackley, 10.0), (Ellips, 1.0e-6), (Griewank, 10.0), (Rastrigin, 1.0)],
            ),
            25 => (
                &[10.0, 20.0, 30.0, 40.0, 50.0],
                &[0.0, 100.0, 200.0, 300.0, 400.0],
                &[(Rastrigin, 10.0), (HappyCat, 1.0), (Ackley, 10.0), (Discus, 1.0e-6), (Rosenbrock, 1.0)],
            ),
            26 => (
                &[10.0, 20.0, 20.0, 30.0, 40.0],
                &[0.0, 100.0, 200.0, 300.0, 400.0],
                // Module doc's VERIFIED DISCREPANCY 1: the report's own
                // printed `lambda` for this fid does not match its own
                // component order or the C at all -- these five rescale
                // factors are derived DIRECTLY from `cf06`'s C body
                // (`fit[i]=K*fit[i]/D` lines), not transcribed from the
                // report.
                &[(EScaffer6, 5.0e-4), (Schwefel, 1.0), (Griewank, 10.0), (Rosenbrock, 1.0), (Rastrigin, 10.0)],
            ),
            27 => (
                &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                &[0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                &[
                    (HGBat, 10.0),
                    (Rastrigin, 10.0),
                    (Schwefel, 2.5),
                    (BentCigar, 1.0e-26),
                    (Ellips, 1.0e-6),
                    (EScaffer6, 5.0e-4),
                ],
            ),
            28 => (
                &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                &[0.0, 100.0, 200.0, 300.0, 400.0, 500.0],
                &[
                    (Ackley, 10.0),
                    (Griewank, 10.0),
                    (Discus, 1.0e-6),
                    (Rosenbrock, 1.0),
                    (HappyCat, 1.0),
                    (EScaffer6, 5.0e-4),
                ],
            ),
            other => unreachable!("composition_spec called with unsupported fid {other}"),
        }
    }

    /// One composition function's total (BEFORE `F_i*`), fid 21-28: each
    /// component's own FULL shift-rotate pipeline (module doc's T8 section:
    /// unlike the hybrids' single shared shift+rotate, EVERY composition
    /// component gets its OWN `(comp_shift[idx], comp_rotation[idx])`) +
    /// base function + its post-eval rescale + its `bias`, weighted by
    /// [`crate::cec_basics::composition_weights`] and summed (`cf_cal`'s own
    /// final loop, module doc's T8 `cf_cal` note).
    fn composition_fitness(&self, xs: &[f64]) -> f64 {
        let (delta, bias, comps) = Self::composition_spec(self.fid);
        let fit: Vec<f64> = comps
            .iter()
            .enumerate()
            .map(|(idx, &(comp, rescale))| {
                let z = crate::cec_basics::shift_scale_rotate(
                    xs,
                    &self.comp_shift[idx],
                    &self.comp_rotation[idx],
                    comp.sh_rate(),
                    true, // every cf01-cf10 component is rotated (module doc's T8 dispatch note).
                );
                comp.eval(&z) * rescale + bias[idx]
            })
            .collect();
        let w = crate::cec_basics::composition_weights(xs, &self.comp_shift, delta);
        fit.iter().zip(&w).map(|(&f, &wi)| f * wi).sum()
    }

    /// `cf09`/`cf10`'s `(delta, bias, hybrid fids)` triple (module doc's T8
    /// "cf09/cf10 -- hybrid-in-composition" section): `hybrid_fids[i]` is
    /// the fid [`Self::hybrid_fitness`] should evaluate component `i` at
    /// (`15,16,17` for `cf09`/fid29; `15,18,19` for `cf10`/fid30, the SAME
    /// `hf0N`<->fid mapping T7's own dispatch section already establishes).
    fn composition_hybrid_spec(fid: u32) -> (&'static [f64], &'static [f64], &'static [u32]) {
        match fid {
            29 => (&[10.0, 30.0, 50.0], &[0.0, 100.0, 200.0], &[15, 16, 17]),
            30 => (&[10.0, 30.0, 50.0], &[0.0, 100.0, 200.0], &[15, 18, 19]),
            other => unreachable!("composition_hybrid_spec called with unsupported fid {other}"),
        }
    }

    /// One composition function's total (BEFORE `F_i*`), fid 29-30
    /// (`cf09`/`cf10`, module doc's T8 "hybrid-in-composition" section):
    /// each component gets its OWN full shift+rotate+shuffle
    /// (`comp_shift[idx]`/`comp_rotation[idx]`/`comp_shuffle[idx]`, ALL
    /// THREE per-component, unlike fid 11-20's single shared triple), then
    /// [`Self::hybrid_fitness`] (T7's helper, reused unmodified) evaluates
    /// that component's own hybrid at the matching hybrid fid. `cf09`/
    /// `cf10` apply NO post-eval rescale (module doc: neither one has a
    /// `fit[i]=K*fit[i]/D` line, unlike `cf01`-`cf08`) -- only `bias[idx]`,
    /// then the SAME shared [`crate::cec_basics::composition_weights`]
    /// weighting. `o` passed to [`Self::hybrid_fitness`] is each
    /// component's own `comp_shift[idx]` -- unused in practice (module
    /// doc's T8 section: none of `hf05..hf09` is `BiRastriginBuggy`, the
    /// only arm that reads it), but kept for signature uniformity with the
    /// fid-11-20 call site.
    fn composition_hybrid_fitness(&self, xs: &[f64]) -> f64 {
        let (delta, bias, hybrid_fids) = Self::composition_hybrid_spec(self.fid);
        let fit: Vec<f64> = hybrid_fids
            .iter()
            .enumerate()
            .map(|(idx, &hfid)| {
                let z = crate::cec_basics::shift_scale_rotate(
                    xs,
                    &self.comp_shift[idx],
                    &self.comp_rotation[idx],
                    1.0,
                    true,
                );
                let y: Vec<f64> = (0..self.dim).map(|i| z[self.comp_shuffle[idx][i]]).collect();
                Self::hybrid_fitness(hfid, self.dim, &y, &self.comp_shift[idx]) + bias[idx]
            })
            .collect();
        let w = crate::cec_basics::composition_weights(xs, &self.comp_shift, delta);
        fit.iter().zip(&w).map(|(&f, &wi)| f * wi).sum()
    }
}

/// One hybrid function's (fid 11-20) sub-component (module doc's T7
/// section): pairs the base function with the `sh_rate` constant the C
/// reference's OWN per-component `sr_func` call uses (`s_flag=0,r_flag=0`
/// branch -- pure scale, no shift, no rotate). `SchafferF7Buggy` and
/// `BiRastriginBuggy` are the two verified-buggy components (module doc's
/// "VERIFIED DISCREPANCY 1/2" sections) -- NOT pure functions of their own
/// segment, so [`Cec2017::hybrid_fitness`] special-cases them directly
/// rather than routing them through [`HybridComponent::eval`].
// sezgi decision (M3-6 T7): mirrors `cec2022::HybridComponent::
// SchafferF7Buggy`'s own established shape (a variant that carries no
// `eval`-callable formula, special-cased by name in the caller) for
// `SchafferF7Buggy`; `BiRastriginBuggy` is a SECOND such variant, new this
// task (no `cec2014`/`cec2022` precedent for a buggy Lunacek component) --
// judged the same shape is the right fit here too, rather than inventing a
// different mechanism (e.g. a closure field or a second enum) for a single
// additional buggy case.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HybridComponent {
    Zakharov,
    Rosenbrock,
    Rastrigin,
    Ellips,
    Schwefel,
    BentCigar,
    Ackley,
    HGBat,
    EScaffer6,
    Katsuura,
    GrieRosen,
    Weierstrass,
    Discus,
    SchafferF7Buggy,
    BiRastriginBuggy,
    /// M3-6 T8, new: `griewank_func`, needed only by `cf02`/`cf04`/`cf06`/
    /// `cf08` (module doc's T8 section) -- fid 1-20 never used it.
    Griewank,
    /// M3-6 T8, new: `happycat_func`, needed only by `cf05`/`cf08` (module
    /// doc's T8 section) -- fid 1-20 never used it.
    HappyCat,
}

impl HybridComponent {
    /// This component's own inner scale constant -- module doc's T7
    /// section, the SAME constant this base function's standalone `sr_func`
    /// call uses (T6's module doc "Reused bases" table, where a standalone
    /// fid dispatches to the same base). `SchafferF7Buggy`/
    /// `BiRastriginBuggy` are never routed through the generic scale-then-
    /// `eval` path (module doc), so their own `sh_rate` here is unused --
    /// still `1.0` (their own C-side `sr_func`/internal scale constant) for
    /// completeness, not read by [`Cec2017::hybrid_fitness`].
    fn sh_rate(self) -> f64 {
        match self {
            Self::Zakharov => 1.0,
            Self::Rosenbrock => 2.048 / 100.0,
            Self::Rastrigin => 5.12 / 100.0,
            Self::Ellips => 1.0,
            Self::Schwefel => 1000.0 / 100.0,
            Self::BentCigar => 1.0,
            Self::Ackley => 1.0,
            Self::HGBat => 5.0 / 100.0,
            Self::EScaffer6 => 1.0,
            Self::Katsuura => 5.0 / 100.0,
            Self::GrieRosen => 5.0 / 100.0,
            Self::Weierstrass => 0.5 / 100.0,
            Self::Discus => 1.0,
            Self::SchafferF7Buggy | Self::BiRastriginBuggy => 1.0,
            // M3-6 T8, new: `griewank_func`'s own `sr_func` call uses
            // `600.0/100.0` (module doc's T8 section, `griewank_func`
            // quoted in full there); `happycat_func` uses `5.0/100.0`.
            Self::Griewank => 600.0 / 100.0,
            Self::HappyCat => 5.0 / 100.0,
        }
    }

    /// Evaluate this component's base function on an ALREADY-scaled segment
    /// (caller applies `sh_rate`, module doc's T7 section -- same pattern
    /// `cec2014::HybridComponent::eval`/`cec2022::HybridComponent::eval`
    /// use). Never called for `SchafferF7Buggy`/`BiRastriginBuggy`
    /// (special-cased in [`Cec2017::hybrid_fitness`] instead).
    fn eval(self, seg: &[f64]) -> f64 {
        use crate::cec_basics as cb;
        match self {
            Self::Zakharov => cb::f1_base(seg),
            Self::Rosenbrock => {
                // `rosenbrock_func`'s own "+1, shift to origin" convention
                // (T6's module doc fid-4 note) -- reuses `f2_base`, which
                // expects the array already `+1`'d.
                let shifted: Vec<f64> = seg.iter().map(|&zi| zi + 1.0).collect();
                cb::f2_base(&shifted)
            }
            Self::Rastrigin => cb::f4_base(seg),
            Self::Ellips => cb::ellips_base(seg),
            Self::Schwefel => cb::schwefel_base(seg),
            Self::BentCigar => cb::bent_cigar_base(seg),
            Self::Ackley => cb::ackley_base(seg),
            Self::HGBat => cb::hgbat_base(seg),
            Self::EScaffer6 => cb::escaffer6_base(seg),
            Self::Katsuura => cb::katsuura_base(seg),
            // `grie_rosen_base` handles its own `+1` internally.
            Self::GrieRosen => cb::grie_rosen_base(seg),
            Self::Weierstrass => cb::weierstrass_base(seg),
            Self::Discus => cb::discus_base(seg),
            Self::SchafferF7Buggy | Self::BiRastriginBuggy => unreachable!(
                "SchafferF7Buggy/BiRastriginBuggy are handled directly in Cec2017::hybrid_fitness"
            ),
            // M3-6 T8, new -- both already exist in `cec_basics` (extracted
            // for `cec2014`'s own T2/T4), no new base-function code needed.
            Self::Griewank => cb::griewank_base(seg),
            Self::HappyCat => cb::happycat_base(seg),
        }
    }
}

/// Lunacek bi-Rastrigin Function's fid-13 hybrid-component form
/// (`bi_rastrigin_func` called from `hf03` with `s_flag=0,r_flag=0`, module
/// doc's "VERIFIED DISCREPANCY 2" section) -- NOT the same code path as fid
/// 7 standalone's [`f7_bi_rastrigin`]: no shift step (`seg` is already the
/// shift-rotate-shuffled `y[G[2]..G[2]+len]` `hf03` computed, and
/// `s_flag=0`'s `y[i]=x[i]` branch takes it as-is), rotation always off
/// (`r_flag=0` for every component call), and (the verified bug) the
/// sign-flip reference `sign_ref` is the CALLER's already-buggy-sliced
/// `o[0..len]`, not `o[G[2]..G[2]+len]`. `bi_rastrigin_func`'s FULL body is
/// quoted at [`f7_bi_rastrigin`]'s own doc; this is the SAME arithmetic
/// with the shift step removed and `rotate` hardcoded false (component
/// calls always pass `r_flag=0`).
// sezgi decision (M3-6 T7): a SEPARATE function rather than refactoring
// `f7_bi_rastrigin` to share an inner helper. The two differ in three
// places (shift step present/absent, rotation toggle vs. hardcoded off,
// and which array aligns with `sign_ref`), so a shared helper would need
// 2-3 extra parameters threading those differences through -- for ~15
// lines of arithmetic, that indirection was judged to cost more
// readability than the duplication it would remove; `f7_bi_rastrigin`
// (T6, already committed and covered by its own passing hand-fixture
// tests) is left completely untouched, avoiding any risk of a refactor
// regression to already-verified code.
fn bi_rastrigin_hybrid_component(seg: &[f64], sign_ref: &[f64]) -> f64 {
    const MU0: f64 = 2.5;
    const D_CONST: f64 = 1.0;
    let pi = std::f64::consts::PI;
    let n = seg.len() as f64;
    let s = 1.0 - 1.0 / (2.0 * (n + 20.0).sqrt() - 8.2);
    let mu1 = -(((MU0 * MU0) - D_CONST) / s).sqrt();

    let y: Vec<f64> = seg.iter().map(|&si| si * (10.0 / 100.0)).collect();
    let z: Vec<f64> = y
        .iter()
        .zip(sign_ref)
        .map(|(&yi, &si)| {
            let t = 2.0 * yi;
            if si < 0.0 { -t } else { t }
        })
        .collect();

    let tmp1: f64 = z.iter().map(|&zi| zi * zi).sum();
    let tmp2: f64 = z.iter().map(|&zi| (zi + MU0 - mu1).powi(2)).sum::<f64>() * s + D_CONST * n;
    // r_flag=0 branch always, for every hybrid component call.
    let cos_sum: f64 = z.iter().map(|&zi| (2.0 * pi * zi).cos()).sum();

    tmp1.min(tmp2) + 10.0 * (n - cos_sum)
}

/// Lunacek bi-Rastrigin Function (CEC 2017 fid 7, `bi_rastrigin_func`).
/// sezgi decision: kept module-local (not promoted to `cec_basics`)
/// because its formula re-reads the raw shift vector for a sign flip,
/// breaking the shift-then-evaluate contract every `cec_basics` base
/// follows. See this module's doc, "Reused
/// bases" table, fid-7 row, for why. `bi_rastrigin_func`'s FULL body,
/// quoted VERBATIM (`s_flag==1,r_flag==1`, the only combination this
/// suite's own dispatch ever calls it with -- the `s_flag==0` `else`
/// branch, `y[i]=x[i]` with no shift, is dead for `sezgi`'s own usage and
/// not replicated here):
/// ```text
/// mu0=2.5, d=1.0;
/// s = 1.0 - 1.0/(2.0*pow(nx+20.0,0.5) - 8.2);
/// mu1 = -pow((mu0*mu0-d)/s, 0.5);
/// shiftfunc(x, y, nx, Os);               // y[i] = x[i] - Os[i]
/// for (i=0;i<nx;i++) y[i] *= 10.0/100.0;
/// for (i=0;i<nx;i++) {
///     tmpx[i] = 2*y[i];
///     if (Os[i] < 0.0) tmpx[i] *= -1.;
/// }
/// for (i=0;i<nx;i++) { z[i] = tmpx[i]; tmpx[i] += mu0; }
/// tmp1 = sum((tmpx[i]-mu0)^2);           // == sum(z[i]^2)
/// tmp2 = sum((tmpx[i]-mu1)^2) * s + d*nx;
/// if (r_flag==1) {
///     rotatefunc(z, y, nx, Mr);
///     tmp = sum(cos(2*PI*y[i]));
/// } else {
///     tmp = sum(cos(2*PI*z[i]));
/// }
/// f = min(tmp1, tmp2) + 10.0*(nx - tmp);
/// ```
/// Note `Os[i]` (the RAW shift vector, not any shift-scaled quantity) is
/// read a SECOND time for the sign flip, independent of the `y[i]*=
/// 10.0/100.0` shift/scale step above it -- this is why this base takes
/// `o` directly rather than a pre-computed `z`, unlike every other base in
/// `cec_basics`.
///
/// Hand-fixture cross-checked against the compiled reference C
/// (`probe2.cpp`, calling `bi_rastrigin_func` directly with hand-supplied
/// `Os`/`M`/`x`, bypassing the file-loading dispatcher -- this task's
/// report has the full `g++`/run transcript): `nx=3`, `Os=[1.0,-2.0,0.5]`,
/// `x=[0.3,-0.1,0.2]`:
/// - `s_flag=1,r_flag=0` (`M` unused): compiled C gives
///   `21.78528151784470168195`.
/// - `s_flag=1,r_flag=1`, `M` = a genuine (non-permutation) 30-degree
///   rotation about the 3rd axis
///   (`[0.8660254037844387,-0.5,0; 0.5,0.8660254037844387,0; 0,0,1]`):
///   compiled C gives `19.84499010291626674984` (a sanity pass with `M` =
///   identity, and separately a pure-permutation rotation, both reproduce
///   the `r_flag=0` value bit-for-bit -- `sum(cos(2*pi*y))` is invariant
///   under permutation/sign-flip of its argument, so only a genuinely
///   non-trivial rotation actually exercises the rotate branch, which is
///   why this fixture uses one).
fn f7_bi_rastrigin(xs: &[f64], o: &[f64], m: &[Vec<f64>], rotate: bool) -> f64 {
    const MU0: f64 = 2.5;
    const D_CONST: f64 = 1.0;
    let pi = std::f64::consts::PI;
    let n = xs.len() as f64;
    let s = 1.0 - 1.0 / (2.0 * (n + 20.0).sqrt() - 8.2);
    let mu1 = -(((MU0 * MU0) - D_CONST) / s).sqrt();

    let y: Vec<f64> = xs.iter().zip(o).map(|(&xi, &oi)| (xi - oi) * (10.0 / 100.0)).collect();
    let z: Vec<f64> = y
        .iter()
        .zip(o)
        .map(|(&yi, &oi)| {
            let t = 2.0 * yi;
            if oi < 0.0 { -t } else { t }
        })
        .collect();

    let tmp1: f64 = z.iter().map(|&zi| zi * zi).sum();
    let tmp2: f64 = z.iter().map(|&zi| (zi + MU0 - mu1).powi(2)).sum::<f64>() * s + D_CONST * n;

    let cos_sum: f64 = if rotate {
        crate::bbob::transform::apply(m, &z).iter().map(|&yi| (2.0 * pi * yi).cos()).sum()
    } else {
        z.iter().map(|&zi| (2.0 * pi * zi).cos()).sum()
    };

    tmp1.min(tmp2) + 10.0 * (n - cos_sum)
}

impl Problem for Cec2017 {
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
// The bi-Rastrigin hand fixtures below carry compiled-C `%.20f` output
// transcribed digit-for-digit (module doc's `f7_bi_rastrigin` doc, "Hand-
// fixture cross-checked" section) so they can be diffed against that
// transcript by eye; clippy's "round to f64 precision" rewrite would
// obscure that traceability without changing the value (same convention
// `cec2014/data.rs`'s own tests module uses).
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use sezgi_core::problem::Evaluator;

    fn g(xs: Vec<f64>) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs)] } }

    /// Relative-difference assertion for the random-point cross-checks
    /// below (module doc, "Reused bases" table's own probe-verification
    /// claims). Measured max relative delta across all 54 (fid,dim,trial)
    /// random probes plus the 18 `x=o` pins, this task: `3.631280409821519e-15`
    /// (fid=9, dim=30, trial=0) -- ~1e-15 class as expected (same order
    /// T2-T5 report for their own fixtures); `1e-9` is comfortably above
    /// it while still catching a genuinely wrong formula.
    fn assert_close(got: f64, want: f64, ctx: &str) {
        let rel = (got - want).abs() / want.abs().max(1e-300);
        assert!(rel < 1e-9, "{ctx}: got {got}, want {want}, rel_diff {rel}");
    }

    /// Every usable fid this module now covers ({1} ∪ {3..=30}, M3-6 T8):
    /// re-used across the construction/space/f_star/data-integrity tests
    /// below so extending the suite's range only requires editing here.
    const ALL_FIDS: [u32; 29] = [
        1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
        27, 28, 29, 30,
    ];

    // ---- construction / errors ----

    #[test]
    fn fid_2_is_withdrawn_not_unknown() {
        assert!(matches!(Cec2017::new(2, 10), Err(Cec2017Error::Withdrawn)));
        assert!(matches!(Cec2017::new(2, 30), Err(Cec2017Error::Withdrawn)));
    }

    #[test]
    fn fid_0_2_and_31_upward_are_unknown_or_withdrawn() {
        // fid 21-30 (this task's own range) are now VALID -- the previous
        // T7-era test's boundary (21 was "upward" back then) moved to 31,
        // module doc's "Struct/impl notes" section.
        for fid in [0u32, 31, 35, 40, 100] {
            assert!(
                matches!(Cec2017::new(fid, 10), Err(Cec2017Error::UnknownFid(f)) if f == fid),
                "fid={fid}"
            );
        }
    }

    #[test]
    fn fid_1_and_3_to_30_all_construct_at_dim_10_and_30() {
        for fid in ALL_FIDS {
            for &dim in &[10usize, 30] {
                assert!(Cec2017::new(fid, dim).is_ok(), "fid={fid} dim={dim}");
            }
        }
    }

    #[test]
    fn unsupported_dims_are_bad_dim_for_every_fid() {
        for fid in ALL_FIDS {
            for &dim in &[2usize, 5, 20, 50, 100] {
                assert!(
                    matches!(Cec2017::new(fid, dim), Err(Cec2017Error::BadDim(d)) if d == dim),
                    "fid={fid} dim={dim}"
                );
            }
        }
    }

    #[test]
    fn space_is_single_float_block_pm100() {
        for fid in ALL_FIDS {
            for &dim in &[10usize, 30] {
                let p = Cec2017::new(fid, dim).unwrap();
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
    fn f_star_matches_dispatch_bias() {
        for fid in ALL_FIDS {
            let expect = 100.0 * f64::from(fid);
            assert_eq!(Cec2017::new(fid, 10).unwrap().f_star(), expect, "fid={fid}");
        }
    }

    // ---- x = o pin: F_i(o_i) == F_i* exactly, fid 1,3-8,10 (module doc's
    // derivation); fid 9 (Levy) is the VERIFIED exception -- pinned to the
    // measured compiled-C constant instead (module doc's fid-9 row). ----

    #[test]
    fn fid_1_3_to_8_and_10_x_eq_o_is_exactly_f_star_dim10_and_dim30() {
        for fid in [1u32, 3, 4, 5, 6, 7, 8, 10] {
            for &dim in &[10usize, 30] {
                let p = Cec2017::new(fid, dim).unwrap();
                let out = p.evaluate_batch(&[g(p.o.clone())])[0];
                assert_eq!(out, p.f_star(), "fid={fid} dim={dim}");
            }
        }
    }

    #[test]
    fn fid_9_levy_x_eq_o_matches_measured_compiled_c_not_f_star() {
        // Compiled C, x=o (this task's probe driver, module doc's fid-9
        // row): 901.4426009870527... at dim=10, 903.2594920693923... at
        // dim=30 -- NOT exactly 900 (f_star), a verified property of the
        // plain Levy base's own minimum location, not a bug.
        let p10 = Cec2017::new(9, 10).unwrap();
        let out10 = p10.evaluate_batch(&[g(p10.o.clone())])[0];
        assert_close(out10, 901.4426009870527, "fid=9 dim=10 x=o");
        assert_ne!(out10, p10.f_star());

        let p30 = Cec2017::new(9, 30).unwrap();
        let out30 = p30.evaluate_batch(&[g(p30.o.clone())])[0];
        assert_close(out30, 903.2594920693923, "fid=9 dim=30 x=o");
        assert_ne!(out30, p30.f_star());
    }

    #[test]
    fn fid_11_to_20_x_eq_o_is_exactly_f_star_dim10_and_dim30() {
        // Module doc's T7 "x = o pin" section: unlike fid 9's own Levy
        // exception, ALL TEN hybrid fids hit `F_i*` exactly at `x=o`
        // (verified against the compiled C, `t7_xo_pin.py`, this task's
        // report has the full transcript).
        for fid in 11u32..=20 {
            for &dim in &[10usize, 30] {
                let p = Cec2017::new(fid, dim).unwrap();
                let out = p.evaluate_batch(&[g(p.o.clone())])[0];
                assert_eq!(out, p.f_star(), "fid={fid} dim={dim}");
            }
        }
    }

    // ---- fid 13/14/20 verified-buggy hybrid components: hand fixtures
    // derived from a targeted compiled-C probe (module doc's "VERIFIED
    // DISCREPANCY 1/2" sections have the full derivation) ----

    #[test]
    fn schaffer_f7_buggy_component_reads_prefix_of_full_y_not_own_segment() {
        // Module doc's "VERIFIED DISCREPANCY 1": fid 14 (`hf04`), component
        // index 2 (`schaffer_F7_func`), segment `G[2]=4, G_nx[2]=2` of a
        // 10-long shuffled `y`. The "correct segment" value
        // (`y[4..6]`) is 519.7396346671704; the ACTUAL compiled-C behavior
        // (`y[0..2]`) is 262.971892911507 -- `hybrid_fitness` must produce
        // the SECOND value, not the first.
        let y = [
            -145.52428725828193,
            22.71207862270197,
            48.49251134825084,
            -12.433687574098899,
            16.315276067523875,
            -166.98226316887443,
            -55.91665381774592,
            -49.29522766284257,
            -135.82586109367236,
            -72.13027964071914,
        ];
        use crate::cec_basics as cb;
        let buggy = cb::f16_schaffer_f7_base(&y[0..2]);
        let correct = cb::f16_schaffer_f7_base(&y[4..6]);
        assert!((buggy - 262.971892911507).abs() < 1e-9, "buggy={buggy}");
        assert!((correct - 519.7396346671704).abs() < 1e-9, "correct={correct}");
        assert_ne!(buggy, correct);
    }

    #[test]
    fn bi_rastrigin_hybrid_component_hand_fixture_matches_compiled_c() {
        // Module doc's "VERIFIED DISCREPANCY 2": fid 13 (`hf03`), component
        // index 2 (`bi_rastrigin_func`), segment `G[2]=6, G_nx[2]=4` of a
        // 10-dim `hf03` evaluation (random-point probe, this task's
        // report). Compiled-C component contribution using the BUGGY
        // `Os[0..4]` sign reference: `669.6574913888649`; using the
        // (hypothetical, NOT what the C does) correctly-offset
        // `Os[6..10]`: `311.712370364593`.
        let seg = [-26.058055713356502, 39.385024412774236, -69.01647034067578, -148.48626825133059];
        let sign_ref_buggy = [4.574812690063766, -71.12311809841688, -6.245880092608687, -14.506716474195343];
        let sign_ref_correct = [-65.84337852236041, -69.66282531137709, 56.894991685121795, 74.12525184178969];
        let buggy = bi_rastrigin_hybrid_component(&seg, &sign_ref_buggy);
        let correct = bi_rastrigin_hybrid_component(&seg, &sign_ref_correct);
        assert!((buggy - 669.6574913888649).abs() < 1e-6, "buggy={buggy}");
        assert!((correct - 311.712370364593).abs() < 1e-6, "correct={correct}");
        assert_ne!(buggy, correct);
    }

    // ---- fid 7 (Lunacek bi-Rastrigin) hand fixtures vs. compiled C
    // (module doc's `f7_bi_rastrigin` own doc has the full derivation and
    // `probe2.cpp` transcript reference) ----

    #[test]
    fn bi_rastrigin_hand_fixture_no_rotation_matches_compiled_c() {
        let o = [1.0_f64, -2.0, 0.5];
        let m: Vec<Vec<f64>> = vec![vec![0.0; 3]; 3]; // unused when rotate=false
        let x = [0.3_f64, -0.1, 0.2];
        let got = f7_bi_rastrigin(&x, &o, &m, false);
        assert_close(got, 21.78528151784470168195, "bi_rastrigin r_flag=0");
    }

    #[test]
    fn bi_rastrigin_hand_fixture_with_rotation_matches_compiled_c() {
        let o = [1.0_f64, -2.0, 0.5];
        let m: Vec<Vec<f64>> = vec![
            vec![0.8660254037844387, -0.5, 0.0],
            vec![0.5, 0.8660254037844387, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let x = [0.3_f64, -0.1, 0.2];
        let got = f7_bi_rastrigin(&x, &o, &m, true);
        assert_close(got, 19.84499010291626674984, "bi_rastrigin r_flag=1");
    }

    // ---- data-integrity spot asserts (mirrors data.rs's own, at the
    // Cec2017 level -- confirms `new` actually wires the parsed data
    // through) ----

    #[test]
    fn constructed_o_and_m_have_the_right_shape_for_every_fid() {
        // Composition fids (21-30) have an EMPTY `m` -- `comp_rotation` is
        // used instead (struct doc); their own shape is checked separately
        // below.
        for fid in [1u32, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20] {
            for &dim in &[10usize, 30] {
                let p = Cec2017::new(fid, dim).unwrap();
                assert_eq!(p.o.len(), dim, "fid={fid} dim={dim}");
                assert_eq!(p.m.len(), dim, "fid={fid} dim={dim}");
                assert!(p.m.iter().all(|row| row.len() == dim), "fid={fid} dim={dim}");
            }
        }
    }

    #[test]
    fn shuffle_is_populated_only_for_fid_11_to_20() {
        for fid in [1u32, 3, 4, 5, 6, 7, 8, 9, 10] {
            let p = Cec2017::new(fid, 10).unwrap();
            assert!(p.shuffle.is_empty(), "fid={fid}");
        }
        for fid in 11u32..=20 {
            for &dim in &[10usize, 30] {
                let p = Cec2017::new(fid, dim).unwrap();
                assert_eq!(p.shuffle.len(), dim, "fid={fid} dim={dim}");
                let mut sorted = p.shuffle.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (0..dim).collect::<Vec<_>>(), "fid={fid} dim={dim}");
            }
        }
        for fid in 21u32..=30 {
            let p = Cec2017::new(fid, 10).unwrap();
            assert!(p.shuffle.is_empty(), "fid={fid} (composition) should have no single shuffle field");
        }
    }

    // ---- M3-6 T8: fid 21-30 (composition functions) ----

    #[test]
    fn composition_cf_num_matches_the_c_sources_local_cf_num() {
        let table = [(21u32, 3usize), (22, 3), (23, 4), (24, 4), (25, 5), (26, 5), (27, 6), (28, 6), (29, 3), (30, 3)];
        for (fid, n) in table {
            assert_eq!(Cec2017::composition_cf_num(fid), n, "fid={fid}");
        }
    }

    // ---- constant-table spot check (module doc's T8 cf01-cf08 table):
    // confirms `composition_spec`'s transcription, INCLUDING the C-derived
    // (not report-transcribed) fid-26/cf06 rescale factors (module doc's
    // VERIFIED DISCREPANCY 1) ----

    #[test]
    fn composition_spec_matches_the_c_sources_constant_tables() {
        use HybridComponent::*;
        let (delta21, bias21, comps21) = Cec2017::composition_spec(21);
        assert_eq!(delta21, &[10.0, 20.0, 30.0]);
        assert_eq!(bias21, &[0.0, 100.0, 200.0]);
        assert_eq!(comps21, &[(Rosenbrock, 1.0), (Ellips, 1.0e-6), (Rastrigin, 1.0)]);

        let (delta26, bias26, comps26) = Cec2017::composition_spec(26);
        assert_eq!(delta26, &[10.0, 20.0, 20.0, 30.0, 40.0]);
        assert_eq!(bias26, &[0.0, 100.0, 200.0, 300.0, 400.0]);
        assert_eq!(
            comps26,
            &[
                (EScaffer6, 5.0e-4),
                (Schwefel, 1.0),
                (Griewank, 10.0),
                (Rosenbrock, 1.0),
                (Rastrigin, 10.0),
            ]
        );

        let (delta28, bias28, comps28) = Cec2017::composition_spec(28);
        assert_eq!(delta28, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        assert_eq!(bias28, &[0.0, 100.0, 200.0, 300.0, 400.0, 500.0]);
        assert_eq!(
            comps28,
            &[
                (Ackley, 10.0),
                (Griewank, 10.0),
                (Discus, 1.0e-6),
                (Rosenbrock, 1.0),
                (HappyCat, 1.0),
                (EScaffer6, 5.0e-4),
            ]
        );

        // Every composition's bias[0] is 0 (module doc's x=comp_shift[0]
        // pin derivation relies on this for every fid 21-30).
        for fid in 21u32..=28 {
            let (_, bias, _) = Cec2017::composition_spec(fid);
            assert_eq!(bias[0], 0.0, "fid={fid}: bias[0] must be 0");
        }
        for fid in [29u32, 30] {
            let (_, bias, _) = Cec2017::composition_hybrid_spec(fid);
            assert_eq!(bias[0], 0.0, "fid={fid}: bias[0] must be 0");
        }
    }

    #[test]
    fn composition_hybrid_spec_maps_cf09_cf10_to_the_right_hf0n_fids() {
        // Module doc's T8 "cf09/cf10" section: `hf05`->fid15, `hf06`->fid16,
        // `hf07`->fid17 for cf09/fid29; `hf05`->fid15, `hf08`->fid18,
        // `hf09`->fid19 for cf10/fid30.
        let (delta29, bias29, hybrids29) = Cec2017::composition_hybrid_spec(29);
        assert_eq!(delta29, &[10.0, 30.0, 50.0]);
        assert_eq!(bias29, &[0.0, 100.0, 200.0]);
        assert_eq!(hybrids29, &[15, 16, 17]);

        let (delta30, bias30, hybrids30) = Cec2017::composition_hybrid_spec(30);
        assert_eq!(delta30, &[10.0, 30.0, 50.0]);
        assert_eq!(bias30, &[0.0, 100.0, 200.0]);
        assert_eq!(hybrids30, &[15, 18, 19]);
    }

    #[test]
    fn constructed_comp_shift_and_rotation_have_cf_num_entries_for_every_fid() {
        for fid in 21u32..=30 {
            let n = Cec2017::composition_cf_num(fid);
            for &dim in &[10usize, 30] {
                let p = Cec2017::new(fid, dim).unwrap();
                assert_eq!(p.comp_shift.len(), n, "fid={fid} dim={dim}: comp_shift row count");
                assert!(
                    p.comp_shift.iter().all(|row| row.len() == dim),
                    "fid={fid} dim={dim}: comp_shift row width"
                );
                assert_eq!(p.comp_rotation.len(), n, "fid={fid} dim={dim}: comp_rotation block count");
                assert!(
                    p.comp_rotation.iter().all(|m| m.len() == dim && m.iter().all(|row| row.len() == dim)),
                    "fid={fid} dim={dim}: comp_rotation block shape"
                );
                // `o` mirrors comp_shift[0] exactly (struct doc).
                assert_eq!(p.o, p.comp_shift[0], "fid={fid} dim={dim}: o must equal comp_shift[0]");
                // `m` (the single-matrix field, fid 1-20's own) stays empty
                // for every composition fid -- comp_rotation is used
                // instead.
                assert!(p.m.is_empty(), "fid={fid} dim={dim}: m must be empty for composition fids");
            }
        }
    }

    #[test]
    fn composition_comp_shift_rows_are_genuinely_distinct_across_components() {
        // Multi-row integrity: confirms `composition_shift_blocks` actually
        // read DIFFERENT rows per component, not the same row `cf_num`
        // times (which would silently pass the shape-only assert above).
        for fid in 21u32..=30 {
            let p = Cec2017::new(fid, 10).unwrap();
            for i in 0..p.comp_shift.len() {
                for j in (i + 1)..p.comp_shift.len() {
                    assert_ne!(
                        p.comp_shift[i], p.comp_shift[j],
                        "fid={fid}: comp_shift rows {i} and {j} must differ"
                    );
                }
            }
        }
    }

    #[test]
    fn comp_shuffle_is_populated_only_for_fid_29_and_30() {
        for fid in 21u32..=28 {
            let p = Cec2017::new(fid, 10).unwrap();
            assert!(p.comp_shuffle.is_empty(), "fid={fid} should have no comp_shuffle");
        }
        for fid in [29u32, 30] {
            for &dim in &[10usize, 30] {
                let p = Cec2017::new(fid, dim).unwrap();
                assert_eq!(p.comp_shuffle.len(), 3, "fid={fid} dim={dim}: 3 hybrid components");
                for block in &p.comp_shuffle {
                    assert_eq!(block.len(), dim, "fid={fid} dim={dim}");
                    let mut sorted = block.clone();
                    sorted.sort_unstable();
                    assert_eq!(sorted, (0..dim).collect::<Vec<_>>(), "fid={fid} dim={dim}: not a permutation of 0..{dim}");
                }
            }
        }
    }

    // ---- x = comp_shift[0] pin: F_i(comp_shift[0]) == F_i* exactly, fid
    // 21-30 (module doc's "INF-sentinel absorption" derivation) ----

    #[test]
    fn fid_21_to_30_x_eq_comp_shift0_is_exactly_f_star_dim10_and_dim30() {
        for fid in 21u32..=30 {
            for &dim in &[10usize, 30] {
                let p = Cec2017::new(fid, dim).unwrap();
                let out = p.evaluate_batch(&[g(p.o.clone())])[0];
                assert_eq!(out, p.f_star(), "fid={fid} dim={dim}");
            }
        }
    }

    // ---- random-point probes vs compiled C reference (fid 1,3-10) ----
    include!("random_probe_tests.rs.fragment");

    // ---- random-point probes vs compiled C reference (fid 11-20 hybrids,
    // M3-6 T7) ----
    include!("hybrid_random_probe_tests.rs.fragment");

    // ---- random-point probes vs compiled C reference (fid 21-30
    // compositions, M3-6 T8) ----
    include!("composition_random_probe_tests.rs.fragment");

    // ---- Evaluator integration + budget counting ----

    #[test]
    fn evaluator_counts_budget_and_tracks_best() {
        let p = Cec2017::new(1, 10).unwrap();
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
        for fid in ALL_FIDS {
            let p = Cec2017::new(fid, 10).unwrap();
            assert_eq!(Problem::optimum(&p), Some(p.f_star()), "fid={fid}");
        }
    }

    // ---- malformed genotype handling ----

    #[test]
    fn wrong_length_float_block_evaluates_to_infinity_not_panic() {
        let p = Cec2017::new(1, 10).unwrap();
        let out = p.evaluate_batch(&[g(vec![0.0; 3])]);
        assert_eq!(out, vec![f64::INFINITY]);
    }

    #[test]
    fn non_float_block_evaluates_to_infinity() {
        let p = Cec2017::new(1, 10).unwrap();
        let bad = Genotype { blocks: vec![BlockValues::Perm(vec![0, 1, 2])] };
        let out = p.evaluate_batch(&[bad]);
        assert_eq!(out, vec![f64::INFINITY]);
    }
}
