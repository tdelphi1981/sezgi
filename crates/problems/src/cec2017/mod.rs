//! The CEC 2017 Special Session and Competition benchmark suite: fid 1
//! (Bent Cigar), fid 3-10 (Zakharov, Rosenbrock, Rastrigin, Schaffer's F7,
//! Lunacek bi-Rastrigin, Non-Continuous Rastrigin, Levy, Modified Schwefel
//! -- M3-6 T6, this task). Fid 2 is PERMANENTLY WITHDRAWN (module doc's "F2
//! ruling" section below); `Cec2017::new(2, _)` returns a dedicated
//! [`Cec2017Error::Withdrawn`] error, distinct from
//! [`Cec2017Error::UnknownFid`]. Fid 11-20 (hybrid functions) and fid 21-30
//! (composition functions) are staged for later tasks (T7, T8 -- mirrors
//! `cec2014`'s own T2 -> T3 -> T4 staging).
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
//! ## Struct/impl notes
//!
//! [`Cec2017::new`] rejects `dim` outside `{10,30}` (vendoring scope) and
//! `fid` outside `{1,3..=10}` (this task's functional scope -- `fid==2`
//! gets its own dedicated [`Cec2017Error::Withdrawn`], everything else
//! outside `{1,3..=10}` gets [`Cec2017Error::UnknownFid`], INCLUDING `fid`
//! in `11..=30` even though those are legitimate C `func_num`s not yet
//! wired here -- T7/T8's own concern, mirrors `cec2014/mod.rs`'s own T2
//! `UnknownFid` scoping to whatever it had wired at the time).

use sezgi_core::problem::Problem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

mod data;

#[derive(Debug, thiserror::Error)]
pub enum Cec2017Error {
    #[error(
        "fid must be 1 or in 3..=10 (this task's functional scope of the CEC 2017 suite: \
         unimodal F1/F3, simple-multimodal F4-F10; fid 11-30 are staged for later tasks), got {0}"
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
    space: SearchSpace,
}

impl Cec2017 {
    /// `fid` must be `1` or in `3..=10` ([`Cec2017Error::UnknownFid`]
    /// otherwise); `fid==2` specifically returns
    /// [`Cec2017Error::Withdrawn`] (module doc's F2 ruling). `dim` must be
    /// one of `{10,30}` ([`Cec2017Error::BadDim`] otherwise, module doc's
    /// "Vendoring scope" section).
    pub fn new(fid: u32, dim: usize) -> Result<Cec2017, Cec2017Error> {
        if fid == 2 {
            return Err(Cec2017Error::Withdrawn);
        }
        if !(fid == 1 || (3..=10).contains(&fid)) {
            return Err(Cec2017Error::UnknownFid(fid));
        }
        if !matches!(dim, 10 | 30) {
            return Err(Cec2017Error::BadDim(dim));
        }
        let o = data::shift_vector(fid, dim);
        let m = data::rotation_matrix(fid, dim);
        let space = SearchSpace::new(vec![Block::Float { lo: -100.0, hi: 100.0, n: dim }])
            .expect("CEC 2017 bounds (-100 < 100) are always valid");
        Ok(Self { fid, dim, o, m, space })
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
            other => unreachable!("Cec2017::new rejects fid outside {{1,3..=10}}, got {other}"),
        };
        value + self.f_star()
    }
}

/// Lunacek bi-Rastrigin Function (CEC 2017 fid 7, `bi_rastrigin_func`).
/// Module-local (not `cec_basics`) -- see this module's doc, "Reused
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

    // ---- construction / errors ----

    #[test]
    fn fid_2_is_withdrawn_not_unknown() {
        assert!(matches!(Cec2017::new(2, 10), Err(Cec2017Error::Withdrawn)));
        assert!(matches!(Cec2017::new(2, 30), Err(Cec2017Error::Withdrawn)));
    }

    #[test]
    fn fid_0_2_and_11_upward_are_unknown_or_withdrawn() {
        for fid in [0u32, 11, 20, 21, 30, 31, 100] {
            assert!(
                matches!(Cec2017::new(fid, 10), Err(Cec2017Error::UnknownFid(f)) if f == fid),
                "fid={fid}"
            );
        }
    }

    #[test]
    fn fid_1_and_3_to_10_all_construct_at_dim_10_and_30() {
        for fid in [1u32, 3, 4, 5, 6, 7, 8, 9, 10] {
            for &dim in &[10usize, 30] {
                assert!(Cec2017::new(fid, dim).is_ok(), "fid={fid} dim={dim}");
            }
        }
    }

    #[test]
    fn unsupported_dims_are_bad_dim_for_every_fid() {
        for fid in [1u32, 3, 4, 5, 6, 7, 8, 9, 10] {
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
        for fid in [1u32, 3, 4, 5, 6, 7, 8, 9, 10] {
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
        let expect =
            [(1u32, 100.0), (3, 300.0), (4, 400.0), (5, 500.0), (6, 600.0), (7, 700.0), (8, 800.0), (9, 900.0), (10, 1000.0)];
        for (fid, fstar) in expect {
            assert_eq!(Cec2017::new(fid, 10).unwrap().f_star(), fstar, "fid={fid}");
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
        for fid in [1u32, 3, 4, 5, 6, 7, 8, 9, 10] {
            for &dim in &[10usize, 30] {
                let p = Cec2017::new(fid, dim).unwrap();
                assert_eq!(p.o.len(), dim, "fid={fid} dim={dim}");
                assert_eq!(p.m.len(), dim, "fid={fid} dim={dim}");
                assert!(p.m.iter().all(|row| row.len() == dim), "fid={fid} dim={dim}");
            }
        }
    }

    // ---- random-point probes vs compiled C reference ----
    include!("random_probe_tests.rs.fragment");

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
        for fid in [1u32, 3, 4, 5, 6, 7, 8, 9, 10] {
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
