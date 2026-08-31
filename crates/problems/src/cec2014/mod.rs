//! The CEC 2014 Special Session and Competition benchmark suite: fid 1-16
//! (unimodal F1-F3, simple multimodal F4-F16) are implemented (M3-6 T2).
//! `Cec2014::new` accepts fid `1..=16` this task; fid 17-22 (hybrid) and
//! 23-30 (composition) are vendored as DATA now (module doc's "Vendoring
//! scope" section below) but not yet wired into `evaluate` -- a later
//! milestone task extends the accepted range, mirroring `cec2022`'s own
//! T5(fid 1-5) -> T6(fid 6-8) -> T7(fid 9-12) staging (`cec2022/mod.rs`'s
//! module doc has that precedent in full).
//!
//! Source (PROVENANCE, fetched and read directly, not from memory): J. J.
//! Liang, B. Y. Qu, P. N. Suganthan, "Problem Definitions and Evaluation
//! Criteria for the CEC 2014 Special Session and Competition on Single
//! Objective Real-Parameter Numerical Optimization", Technical Report
//! 201311 / Nanyang Technological University, December 2013. PDF fetched
//! directly from the authors' own official repository:
//! `https://raw.githubusercontent.com/P-N-Suganthan/CEC2014/master/Definitions%20of%20%20CEC2014%20benchmark%20suite%20Part%20A.pdf`
//! (SHA-256 `a506b9b20d5bfaa63a373be16cd6af991ab3d227de4824e2dadd3c3d9c6552c6`,
//! git blob SHA `a3d6b0839d143a91239e7694bde3f9ac418a0952`; every quote below
//! is transcribed from that fetched PDF via `pdftotext -layout`). The
//! reference C implementation and vendored data were fetched from the SAME
//! repository's `cec14-c-code.zip` ->
//! `cec14-c-code/{cec14_test_func.cpp, input_data/}`
//! (`https://raw.githubusercontent.com/P-N-Suganthan/CEC2014/master/cec14-c-code.zip`,
//! SHA-256 `1a210560398ca7a50be6adf1e5e90602222519ef23b6e31aba8847e109761876`,
//! git blob SHA `26e1e183dddce4203ea97e97c9eed09e345e15e4` -- re-fetched fresh
//! for this task and independently re-hashed both ways, matching a prior
//! research pass's own fetch byte-for-byte; this task's report has the full
//! `curl`/`shasum`/`git hash-object` transcript).
//!
//! ## Licensing finding (mirrors `cec2022/mod.rs`'s milestone scope ruling 3)
//!
//! The repository root (`https://api.github.com/repos/P-N-Suganthan/CEC2014/contents/`,
//! fetched directly) lists the zips/PDFs/results archives described above
//! and NONE of `LICENSE`, `LICENSE.md`, `LICENSE.txt` -- no license file
//! exists at the repo root. Per the milestone's scope ruling, the files
//! below are vendored with prominent attribution (this doc) rather than
//! withheld.
//!
//! ## Vendoring scope (user-approved, dims {10, 30} only)
//!
//! `crates/problems/data/cec2014/` carries EVERY `input_data/` file the
//! reference C's own loader reads for `fid` 1-30 at `dim` 10 and 30 (so
//! T3/T4, which extend `Cec2014` to fid 17-30, need not re-fetch anything):
//! every `shift_data_<fid>.txt` for `fid` 1-30 (dimension-independent, one
//! copy each -- module doc's file-grammar section below), every
//! `M_<fid>_D{10,30}.txt` for `fid` 1-30 (the C's loader unconditionally
//! reads an `M` file for every `func_num`, quoted below), and
//! `shuffle_data_<fid>_D{10,30}.txt` ONLY for the 8 fids the loader's own
//! conditional actually reads shuffle data for -- `fid` in
//! `{17,18,19,20,21,22,29,30}` (quoted below; the vendored upstream
//! `input_data/` directory actually carries a `shuffle_data_<fid>_D*.txt`
//! for EVERY `fid` 1-30, not just these 8 -- VERIFIED by listing the
//! directory directly -- but the other 22 are genuinely dead weight the C
//! never reads for ANY `func_num`, so they are deliberately NOT vendored,
//! same "don't vendor what the C never reads" judgment call
//! `cec2022/data.rs`'s module doc makes for its own dim=2 hybrid gap).
//!
//! **Measured total (this task, `find crates/problems/data/cec2014 -type f
//! -exec stat -f%z {} \; | awk '{s+=$1} END {print s}'`, cross-checked with
//! an independent Python `os.path.getsize` sum during the copy step): 106
//! files, 2,816,016 bytes.** This is the exact figure to use going forward;
//! it does NOT match the brief's own "expected 2,568,564 bytes" figure
//! (carried over from an earlier scratchpad research pass) -- traced
//! directly to source: that earlier estimate was `D10-file-bytes +
//! D30-file-bytes` from a per-dimension breakdown that (per its own text)
//! deliberately excluded the dimension-independent `shift_data_<fid>.txt`
//! files ("outside the per-dim table"), AND it summed `shuffle_data_*_D{10,30}.txt`
//! for ALL 30 fids rather than only the 8 the C loader actually reads. Both
//! omissions/inclusions are corrected here: this module's 2,816,016 B is
//! `shift_data` (30 files, dimension-independent) + `M_*_D{10,30}` (60
//! files) + `shuffle_data_*_D{10,30}` for the 8 hybrid/8-of-composition fids
//! that use it (16 files) = 106 files -- the set the reference C's loader
//! genuinely reads for a complete `fid in 1..=30` sweep at `dim in
//! {10,30}`, no more, no less.
//!
//! ## Supported dims -- `{2,10,20,30,50,100}` per the C, `{10,30}` vendored
//! here (this task's user-approved scope)
//!
//! `cec14_test_func.cpp`'s initializer, quoted verbatim:
//! ```text
//! if (!(nx==2||nx==10||nx==20||nx==30||nx==50||nx==100))
//!     printf("\nError: Test functions are only defined for D=2,10,20,30,50,100.\n");
//! if (nx==2&&((func_num>=17&&func_num<=22)||(func_num>=29&&func_num<=30)))
//!     printf("\nError: hf01,hf02,hf03,hf04,hf05,hf06,cf07&cf08 are NOT defined for D=2.\n");
//! ```
//! Not relevant to fid 1-16 (this task: every fid here supports every
//! listed dim, including dim=2, per this guard -- the dim=2 exclusion is
//! for `func_num` 17-22/29-30 only, this suite's hybrid/composition
//! functions, T3/T4's concern); `Cec2014::new` restricts `dim` to `{10,30}`
//! ONLY because those are the two dims this task vendored (user-approved
//! scope), not because dim=2/20/50/100 are unsupported by the underlying
//! functions.
//!
//! ## Also quoted (the loader's unconditional `M` load for `func_num<23`,
//! confirming EVERY fid 1-22 gets a full `dim`x`dim` `M_<fid>_D<dim>.txt`
//! loaded regardless of whether that fid's own dispatch arm ever rotates
//! with it -- the SAME "M vendored but sometimes unused" shape
//! `cec2022/mod.rs`'s F3 discrepancy note already describes for that suite)
//!
//! ```text
//! if (func_num<23) {
//!     M=(double*)malloc(nx*nx*sizeof(double));
//!     for (i=0; i<nx*nx; i++) fscanf(fpt,"%lf",&M[i]);
//! }
//! ```
//! This matters for fid 8 and fid 10 specifically (module doc's dispatch
//! section below): both load a full `M_8_D<dim>.txt` / `M_10_D<dim>.txt`
//! but their OWN dispatch call passes `r_flag=0` (no rotation) -- vendored,
//! parsed, and stored in `Cec2014::m` for struct/API uniformity, same as
//! every other fid, but never actually multiplied against `z` for those two.
//!
//! ## Portability edits (this task, mirrors `cec2022/mod.rs`'s own note):
//! the vendored C was actually COMPILED (`g++ -O2`) and RUN for every probe
//! and hand fixture below (driver source and exact invocations in this
//! task's report). Two edits were needed, both purely portability (no
//! behavior change):
//! 1. `#include <WINDOWS.H>` and `#include <malloc.h>` removed (WINDOWS.H
//!    does not exist off Windows; `malloc`/`free` come from `<cstdlib>`
//!    instead) -- the SAME edit `cec2022/mod.rs`'s module doc records for
//!    that suite's own compile.
//! 2. Every `fscanf(fpt,"%Lf", &<double*>)` rewritten to `%lf`. The
//!    vendored source's OWN `readme.txt` names this exact fix: *"For Linux
//!    Users: Please change %xx in fscanf and fprintf and do use
//!    "WINDOWS.H"."* -- `%Lf` scans a `long double` (16 bytes on this
//!    platform) into an 8-byte `double*` destination, undefined behavior
//!    that corrupts adjacent heap memory on non-Windows toolchains; `%lf` is
//!    the correct conversion for `double*` and is what the loader's own
//!    comment instructs fixing. `main.cpp` (the reference's own example
//!    driver, not used by this task -- a purpose-built probe driver
//!    replaced it, report has the source) carries the same `%Lf` bug,
//!    unfixed there since it was not compiled.
//!
//! ## 1.2. Summary table (quoted; PDF `pdftotext -layout` extraction,
//! `Fi*=Fi(x*)` column) -- CROSS-CHECKED against `cec14_test_func.cpp`'s
//! `switch(func_num)` dispatch (quoted below), MATCHES exactly, no
//! report-vs-code numbering discrepancy of the kind `cec2022/mod.rs` and
//! this crate's own `cec1417-research.md` scratchpad note found for
//! CEC 2017's F2
//!
//! | No. | Functions | `Fi*` |
//! |---|---|---|
//! | 1 | Rotated High Conditioned Elliptic Function | 100 |
//! | 2 | Rotated Bent Cigar Function | 200 |
//! | 3 | Rotated Discus Function | 300 |
//! | 4 | Shifted and Rotated Rosenbrock's Function | 400 |
//! | 5 | Shifted and Rotated Ackley's Function | 500 |
//! | 6 | Shifted and Rotated Weierstrass Function | 600 |
//! | 7 | Shifted and Rotated Griewank's Function | 700 |
//! | 8 | Shifted Rastrigin's Function | 800 |
//! | 9 | Shifted and Rotated Rastrigin's Function | 900 |
//! | 10 | Shifted Schwefel's Function | 1000 |
//! | 11 | Shifted and Rotated Schwefel's Function | 1100 |
//! | 12 | Shifted and Rotated Katsuura Function | 1200 |
//! | 13 | Shifted and Rotated HappyCat Function | 1300 |
//! | 14 | Shifted and Rotated HGBat Function | 1400 |
//! | 15 | Shifted and Rotated Expanded Griewank's plus Rosenbrock's Function | 1500 |
//! | 16 | Shifted and Rotated Expanded Scaffer's F6 Function | 1600 |
//!
//! (fid 17-30: Hybrid Functions 1-6 (1700-2200), Composition Functions 1-8
//! (2300-3000) -- T3/T4's concern, not re-transcribed here.) "Search Range:
//! `[-100,100]^D`." confirms the brief's sketch domain -- [`Cec2014::new`]
//! builds `Block::Float{lo:-100.0,hi:100.0,n:dim}`.
//!
//! **`Fi* = 100*fid` exactly for every fid 1-16, VERIFIED two ways**: (a)
//! the table above, read literally; (b) the compiled C reference's dispatch
//! switch (quoted next section) -- every `case N: <fn>(...); f[i]+=N*100.0;`.
//! [`Cec2014::f_star`] therefore implements the closed form `100.0 *
//! fid as f64` rather than a 16-arm match table (unlike `cec2022`'s
//! `f_star`, whose table is NOT a clean arithmetic progression) --
//! `f_star_matches_report_table` spot-checks several fids against the
//! table's literal numbers as an independent re-assertion of the formula.
//!
//! ## Dispatch (quoted in full for fid 1-16; `cec14_test_func.cpp`'s
//! `switch(func_num)`) -- confirms function names, `F*` bias, AND each
//! fid's `s_flag`/`r_flag` (rotation on/off) directly from the executable
//! source, not the report's prose
//!
//! ```text
//! case 1:  ellips_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=100.0; break;
//! case 2:  bent_cigar_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=200.0; break;
//! case 3:  discus_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=300.0; break;
//! case 4:  rosenbrock_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=400.0; break;
//! case 5:  ackley_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=500.0; break;
//! case 6:  weierstrass_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=600.0; break;
//! case 7:  griewank_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=700.0; break;
//! case 8:  rastrigin_func(&x[i*nx],&f[i],nx,OShift,M,1,0); f[i]+=800.0; break;
//! case 9:  rastrigin_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=900.0; break;
//! case 10: schwefel_func(&x[i*nx],&f[i],nx,OShift,M,1,0); f[i]+=1000.0; break;
//! case 11: schwefel_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=1100.0; break;
//! case 12: katsuura_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=1200.0; break;
//! case 13: happycat_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=1300.0; break;
//! case 14: hgbat_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=1400.0; break;
//! case 15: grie_rosen_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=1500.0; break;
//! case 16: escaffer6_func(&x[i*nx],&f[i],nx,OShift,M,1,1); f[i]+=1600.0; break;
//! ```
//! `s_flag=1` for every fid 1-16 (all shift); `r_flag=0` ONLY for fid 8
//! (Rastrigin's) and fid 10 (Schwefel's) -- matching the report table's
//! "Shifted Rastrigin's"/"Shifted Schwefel's" naming EXACTLY (no "Rotated"
//! in either name, unlike fid 9/11's "Shifted and Rotated" counterparts) --
//! NO discrepancy here, unlike `cec2022`'s F4 (whose "Non-Continuous"
//! dispatch name turned out to be dead code, `cec2022/mod.rs`'s module doc).
//! A cosmetic-only curiosity worth recording so a reader diffing this
//! module against the C source isn't confused: the function *declaration*
//! comments two lines apart read `bent_cigar_func(...); /* Discus */` and
//! `discus_func(...);  /* Bent_Cigar */` -- the comments are SWAPPED
//! relative to which function they actually document; the dispatch switch
//! itself (quoted above) calls each function by its correct NAME
//! (`bent_cigar_func` for fid 2, `discus_func` for fid 3), so this is
//! harmless stale-comment noise, not a behavioral discrepancy (verified:
//! `bent_cigar_func`'s own body, quoted below, IS the Bent Cigar formula;
//! `discus_func`'s body IS the Discus formula -- the code matches the
//! report's Table I row assignment, only the inline `/* ... */` labels two
//! lines up are transposed).
//!
//! ## `sr_func` (shift-then-scale-then-rotate) -- IDENTICAL to `cec2022`'s
//! own `sr_func`, byte-for-byte (both suites share the same reference-C
//! lineage), so [`crate::cec_basics::shift_scale_rotate`] is reused AS-IS,
//! no new shift/rotate plumbing needed this task
//!
//! ```text
//! void sr_func (double *x, double *sr_x, int nx, double *Os,double *Mr,
//!               double sh_rate, int s_flag,int r_flag)
//! {
//!     if (s_flag==1) {
//!         if (r_flag==1) { shiftfunc(x,y,nx,Os); for(i)y[i]*=sh_rate; rotatefunc(y,sr_x,nx,Mr); }
//!         else { shiftfunc(x,sr_x,nx,Os); for(i) sr_x[i]*=sh_rate; }
//!     } else {
//!         if (r_flag==1) { for(i) y[i]=x[i]*sh_rate; rotatefunc(y,sr_x,nx,Mr); }
//!         else for(i) sr_x[i]=x[i]*sh_rate;
//!     }
//! }
//! ```
//! Confirmed identical to `cec2022/mod.rs`'s own quoted `sr_func` (same
//! branch structure, same shift-scale-rotate order) by direct line
//! comparison of both vendored sources during this task. Every fid 1-16
//! calls its own base function with `s_flag=1` always (module doc's
//! dispatch section above), so [`Cec2014::shift_scale_rotate`]'s `rotate`
//! parameter alone (mapping the dispatch's `r_flag`) is enough to cover
//! this suite's fid 1-16 -- no separate `s_flag=0` path is needed here.
//!
//! ## Basic-function reuse from `crate::cec_basics` -- VERIFIED BY PROBE,
//! not by name (per the task brief's explicit instruction): every fid 1-16
//! base function's C body was read directly (quoted below) and its
//! `sh_rate` constant cross-checked against the SAME-NAMED `cec2022`
//! function's own usage; ALL FIFTEEN already exist in `crate::cec_basics`
//! (extracted there in M3-6 T1) with byte-identical formulas -- confirmed
//! by the random-point cross-checks against the compiled C reference in
//! this module's own test suite (`fidN_random_points_match_compiled_c_reference`,
//! 1e-9 relative tolerance, measured max deviation in this task's report).
//! Only Weierstrass (fid 6) is genuinely new to this suite --
//! [`crate::cec_basics::weierstrass_base`] (added this task, full
//! provenance + a compiled-C hand fixture in its own doc comment).
//!
//! | fid | C function | `sh_rate` | `crate::cec_basics` fn | rotate |
//! |---|---|---|---|---|
//! | 1 | `ellips_func` | `1.0` | `ellips_base` | yes |
//! | 2 | `bent_cigar_func` | `1.0` | `bent_cigar_base` | yes |
//! | 3 | `discus_func` | `1.0` | `discus_base` | yes |
//! | 4 | `rosenbrock_func` | `2.048/100.0` | `f2_base` (on `z+1`) | yes |
//! | 5 | `ackley_func` | `1.0` | `ackley_base` | yes |
//! | 6 | `weierstrass_func` | `0.5/100.0` | `weierstrass_base` (NEW) | yes |
//! | 7 | `griewank_func` | `600.0/100.0` | `griewank_base` | yes |
//! | 8 | `rastrigin_func` | `5.12/100.0` | `f4_base` | **no** |
//! | 9 | `rastrigin_func` | `5.12/100.0` | `f4_base` | yes |
//! | 10 | `schwefel_func` | `1000.0/100.0` | `schwefel_base` | **no** |
//! | 11 | `schwefel_func` | `1000.0/100.0` | `schwefel_base` | yes |
//! | 12 | `katsuura_func` | `5.0/100.0` | `katsuura_base` | yes |
//! | 13 | `happycat_func` | `5.0/100.0` | `happycat_base` | yes |
//! | 14 | `hgbat_func` | `5.0/100.0` | `hgbat_base` | yes |
//! | 15 | `grie_rosen_func` | `5.0/100.0` | `grie_rosen_base` (handles its own `+1`) | yes |
//! | 16 | `escaffer6_func` | `1.0` | `escaffer6_base` | yes |
//!
//! Each C body, quoted (confirms the `sh_rate` column and every formula
//! detail against `crate::cec_basics`'s own doc comments, which already
//! quote the CEC 2022 report's equivalent C -- transcribed here again for
//! this suite's independent provenance trail, byte-identical bodies):
//! `ellips_func`: `f[0] += pow(10.0,6.0*i/(nx-1))*z[i]*z[i]` (i=0..nx-1) --
//! matches `ellips_base`. `bent_cigar_func`: `f[0]=z[0]*z[0]; f[0] +=
//! pow(10.0,6.0)*z[i]*z[i]` (i=1..nx-1) -- matches `bent_cigar_base`.
//! `discus_func`: `f[0]=pow(10.0,6.0)*z[0]*z[0]; f[0]+=z[i]*z[i]`
//! (i=1..nx-1) -- matches `discus_base`. `rosenbrock_func`: `sr_func(...,
//! 2.048/100.0,...); z[0]+=1.0; for(i<nx-1){ z[i+1]+=1.0;
//! tmp1=z[i]*z[i]-z[i+1]; tmp2=z[i]-1.0; f[0]+=100.0*tmp1*tmp1+tmp2*tmp2; }`
//! -- the incremental `+1.0` (module doc's `cec2022/mod.rs` sibling note,
//! same reasoning) is algebraically identical to `+1`'ing the whole vector
//! up front, matching `f2_base` on a `+1`'d copy. `ackley_func`: `sum1 +=
//! z[i]*z[i]; sum2 += cos(2*PI*z[i]); f[0]=E-20*exp(-0.2*sqrt(sum1/nx)) -
//! exp(sum2/nx) + 20.0` -- matches `ackley_base`. `griewank_func`:
//! `sr_func(...,600.0/100.0,...); s+=z[i]*z[i]; p*=cos(z[i]/sqrt(1.0+i));
//! f[0]=1.0+s/4000.0-p` -- matches `griewank_base`. `rastrigin_func`:
//! `sr_func(...,5.12/100.0,...); f[0] += z[i]*z[i]-10*cos(2*PI*z[i])+10` --
//! matches `f4_base` (`cec2022/mod.rs`'s own F4 discrepancy note is
//! irrelevant here: this suite's dispatch never claims a non-continuous
//! variant for fid 8/9 the way `cec2022`'s fid-4 name did -- CEC 2014's
//! actual non-continuous variant, `step_rastrigin_func`, is declared in the
//! C but never dispatched by ANY fid 1-30 case, confirmed by grepping every
//! `case` arm -- genuinely unused dead code in the reference, not a bug
//! this suite's F8/F9 need to replicate). `schwefel_func`: identical body
//! to `crate::cec_basics::schwefel_base`'s own quoted C (byte-for-byte,
//! diffed directly against `cec2022`'s vendored copy during this task) --
//! matches `schwefel_base`. `katsuura_func`, `happycat_func`, `hgbat_func`,
//! `grie_rosen_func`, `escaffer6_func`: each diffed byte-for-byte against
//! `cec2022`'s vendored copy of the same-named function during this task
//! (identical bodies, identical `sh_rate` constants) -- matches
//! `katsuura_base`/`happycat_base`/`hgbat_base`/`grie_rosen_base`/`escaffer6_base`
//! respectively.
//!
//! ## VERIFIED DISCREPANCY: F16's printed equation shows a `+1` shift-to-
//! origin the code does NOT apply (code-over-report ruling)
//!
//! Report eq (30), quoted: `F16(x) = f14(M(x-o16) + 1) + F16*`. But
//! `escaffer6_func`'s body (quoted above, and in
//! `crate::cec_basics::escaffer6_base`'s own doc) never adds `1` anywhere --
//! `sr_func` is called with `sh_rate=1.0` and no offset loop follows it,
//! unlike `rosenbrock_func`/`grie_rosen_func`'s explicit `z[i]+=1.0` lines a
//! few functions above/below it in the SAME source file. This reads as the
//! same class of PDF copy-paste artifact `cec2022/mod.rs`'s module doc
//! documents for that report's own F5 (`5.12` duplicated from the line
//! above) -- eq (29), directly above eq (30) in the PDF, is F15's Expanded
//! Griewank's-plus-Rosenbrock's, whose formula genuinely DOES have a `+1`
//! (`grie_rosen_func` does shift-to-origin, matching eq (29) exactly); eq
//! (30)'s `+1` most plausibly carried over from the adjacent equation
//! during typesetting. Per the milestone's STANDING RULING (code-over-
//! report): this module's fid-16 arm applies NO `+1` offset, matching
//! `escaffer6_func`'s actual, executable behavior -- `f16_x_eq_o_pin_and_random_probes_confirm_no_plus_one_offset`
//! documents this (the compiled C reference itself is the cross-check, not
//! just this module's own Rust).
//!
//! ## `x = o` pin (Step 2): why `F_i(o_i) == F_i*` EXACTLY for every fid
//! 1-16, VERIFIED BOTH algebraically AND against the compiled C reference
//! (this task's report has the `%.20f`-formatted transcript, all sixteen
//! fids, dim 10 AND 30 -- every one printed the exact integer `100*fid`,
//! zero fractional residue)
//!
//! At `x = o`: the shift `x - o = 0` exactly (identical `f64` operands
//! subtract to exact `0.0`), so every fid's `z` (post shift-scale-rotate) is
//! the ALL-ZERO vector regardless of `sh_rate` (`0*sr=0`) or rotation
//! (`M*0=0`), EXCEPT fid 4 and fid 15, whose base functions add `1`
//! afterward (`f2_base`/`grie_rosen_base`'s own internal `+1`), giving the
//! ALL-ONES vector for those two. Per base function at its own pipeline's
//! origin argument: `ellips_base(0)=0`; `bent_cigar_base(0)=0`;
//! `discus_base(0)=0`; `f2_base(1,1,...,1)=sum(100*(1-1)^2+(1-1)^2)=0`;
//! `ackley_base(0)`: `sum1=0,sum2=n`; term reduces to `E - 20*exp(0) -
//! exp(1) + 20 = E - exp(1.0)` -- `0.0` only if `std::f64::consts::E` and a
//! live `1.0_f64.exp()` round to the IDENTICAL `f64`, VERIFIED empirically
//! (this task's report; same argument `cec2022/mod.rs`'s own Ackley pin
//! note makes) rather than claimed as an algebraic identity;
//! `weierstrass_base(0)`: every `i`'s inner `sum` term is
//! `cos(2*pi*b^k*(0+0.5))`, IDENTICAL to `sum2`'s own
//! `cos(2*pi*b^k*0.5)` term-by-term (same floating-point operations, same
//! operands) -- `sum == sum2` bit-exact each outer iteration, so `f`
//! accumulates to exactly `n*sum2`, and the final `f -= n*sum2` cancels to
//! exact `0.0`; `griewank_base(0)`: `s=0,p=prod(cos(0))=1`;
//! `f=1+0-1=0`; `f4_base(0)` [Rastrigin]: `sum(0-10*cos(0)+10)=sum(0)=0`
//! per term (`cos(0)=1.0` exact); `schwefel_base(0)`: `zi=0+
//! 4.209687462275036e+02` (`<=500`, else-branch: `f -= zi*sin(sqrt(zi))`,
//! `f += 4.189828872724338e+02*n`) -- like `cec2022/mod.rs`'s own Schwefel
//! hybrid-component pin note, NOT claimed as an algebraic identity (depends
//! on `sin`/`sqrt` of a specific non-round `f64` cancelling against a
//! separately-truncated-decimal constant); VERIFIED EMPIRICALLY instead:
//! the compiled C reference's fid 10/11 `F(o)` returns EXACTLY
//! `1000.0`/`1100.0` (dim 10 AND 30, `%.20f`, zero fractional digits),
//! matching this module's own `fidN_x_eq_o_is_exactly_f_star` test;
//! `katsuura_base(0)`: same `floor(0+0.5)=floor(0.5)=0` exact reasoning as
//! `cec2022/mod.rs`'s own Katsuura pin -- `f*tmp1-tmp1=0` exact;
//! `happycat_base(0)`: shift-to-origin `zs=-1` for all `i`; `r2=n,
//! sum_z=-n`; `|r2-n|=0`; `0^0.25=0`; remainder `(0.5n+(-n))/n=-0.5`; total
//! `0-0.5+0.5=0`; `hgbat_base(0)`: same shift-to-origin, `r2=n,sum_z=-n`;
//! `r2^2-sum_z^2=n^2-n^2=0` bit-exact (both `pow(x,2.0)` of the same
//! magnitude `n`); remainder identical to HappyCat's, `total=0`;
//! `grie_rosen_base(0)` [internal `+1`, `z1` all-ones]: every cyclic pair
//! `tmp1=1-1=0,tmp2=0,temp=0`; `temp^2/4000-cos(0)+1=0-1+1=0` per pair;
//! `escaffer6_base(0)`: `g(0,0)=0.5+(sin^2(0)-0.5)/1^2=0.5-0.5=0` per cyclic
//! pair (all-zero `z`, no `+1` per the F16 discrepancy note above). Every
//! base function is `0` at its pipeline's origin, so `F_i(o_i) = 0 + F_i* =
//! F_i*` EXACTLY.
//!
//! ## Genotype mapping
//!
//! [`Cec2014::space`] is one [`Block::Float`] of `dim` variables, bounds
//! `[-100.0, 100.0]` (module doc, "Search range" quote above) -- same shape
//! as `cec2022`'s own single-block encoding.

mod data;

use sezgi_core::problem::Problem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

#[derive(Debug, thiserror::Error)]
pub enum Cec2014Error {
    #[error(
        "fid must be in 1..=16 in this build (the CEC 2014 unimodal + simple-multimodal \
         functions -- fid 17..=22 hybrid and 23..=30 composition are vendored as data \
         (input_data D10/D30 for all 30 fids) but not yet wired into evaluate; a later \
         milestone task extends this range, mirroring cec2022's T5->T6->T7 staging), got {0}"
    )]
    UnknownFid(u32),
    #[error(
        "dim must be one of {{10,30}} (this crate vendors only these two dimensions of the \
         CEC 2014 input_data set -- the official suite also supports 2,20,50,100), got {0}"
    )]
    BadDim(usize),
}

/// One instance of a CEC 2014 basic (fid 1-3) or simple-multimodal (fid
/// 4-16) function: embedded official shift vector `o` and rotation matrix
/// `M` for the requested `(fid, dim)`, plus the pinned `F_i* = 100*fid` bias
/// (module doc, section 1.2's table). See the module doc for each fid's
/// exact shift/scale/rotate pipeline, including the verified F16
/// discrepancy between the printed report and the vendored reference C code
/// this module actually follows.
pub struct Cec2014 {
    fid: u32,
    dim: usize,
    o: Vec<f64>,
    m: Vec<Vec<f64>>,
    space: SearchSpace,
}

impl Cec2014 {
    /// `fid` in `1..=16` (module doc's staging note: `Cec2014Error::UnknownFid`
    /// otherwise, including `0`, `17..=30`, and anything `>30`), `dim` in
    /// `{10,30}` (module doc's "Vendoring scope" section: the two dims this
    /// task vendored; `Cec2014Error::BadDim` otherwise).
    pub fn new(fid: u32, dim: usize) -> Result<Cec2014, Cec2014Error> {
        if !(1..=16).contains(&fid) {
            return Err(Cec2014Error::UnknownFid(fid));
        }
        if !matches!(dim, 10 | 30) {
            return Err(Cec2014Error::BadDim(dim));
        }
        let o = data::shift_vector(fid, dim);
        let m = data::rotation_matrix(fid, dim);
        let space = SearchSpace::new(vec![Block::Float { lo: -100.0, hi: 100.0, n: dim }])
            .expect("CEC 2014 bounds (-100 < 100) are always valid");
        Ok(Self { fid, dim, o, m, space })
    }

    pub fn fid(&self) -> u32 { self.fid }
    pub fn dim(&self) -> usize { self.dim }

    /// The report's `F_i* = 100*fid` (module doc, section 1.2's table AND
    /// the dispatch switch's own `f[i]+=<N>00.0` lines, VERIFIED both ways
    /// to agree -- see module doc for the cross-check).
    pub fn f_star(&self) -> f64 { 100.0 * f64::from(self.fid) }

    /// `(x - o) * sr`, optionally rotated by `self.m` (module doc's
    /// `sr_func` quote, byte-identical to `cec2022`'s own -- delegates to
    /// the SAME shared helper, no new shift/rotate arithmetic this task).
    fn shift_scale_rotate(&self, xs: &[f64], sr: f64, rotate: bool) -> Vec<f64> {
        crate::cec_basics::shift_scale_rotate(xs, &self.o, &self.m, sr, rotate)
    }

    /// The full pipeline (shift/scale/rotate + base function + `F_i*` bias)
    /// for one already-flattened decision vector, per this module's doc
    /// (each fid's exact `sr`/rotate choice, module doc's reuse table,
    /// including the verified F16 discrepancy from the printed report).
    fn eval_one(&self, xs: &[f64]) -> f64 {
        use crate::cec_basics as cb;
        let value = match self.fid {
            1 => cb::ellips_base(&self.shift_scale_rotate(xs, 1.0, true)),
            2 => cb::bent_cigar_base(&self.shift_scale_rotate(xs, 1.0, true)),
            3 => cb::discus_base(&self.shift_scale_rotate(xs, 1.0, true)),
            4 => {
                let mut z = self.shift_scale_rotate(xs, 2.048 / 100.0, true);
                for zi in &mut z {
                    *zi += 1.0;
                }
                cb::f2_base(&z)
            }
            5 => cb::ackley_base(&self.shift_scale_rotate(xs, 1.0, true)),
            6 => cb::weierstrass_base(&self.shift_scale_rotate(xs, 0.5 / 100.0, true)),
            7 => cb::griewank_base(&self.shift_scale_rotate(xs, 600.0 / 100.0, true)),
            // No rotation (module doc's dispatch section: r_flag=0).
            8 => cb::f4_base(&self.shift_scale_rotate(xs, 5.12 / 100.0, false)),
            9 => cb::f4_base(&self.shift_scale_rotate(xs, 5.12 / 100.0, true)),
            // No rotation (module doc's dispatch section: r_flag=0).
            10 => cb::schwefel_base(&self.shift_scale_rotate(xs, 1000.0 / 100.0, false)),
            11 => cb::schwefel_base(&self.shift_scale_rotate(xs, 1000.0 / 100.0, true)),
            12 => cb::katsuura_base(&self.shift_scale_rotate(xs, 5.0 / 100.0, true)),
            13 => cb::happycat_base(&self.shift_scale_rotate(xs, 5.0 / 100.0, true)),
            14 => cb::hgbat_base(&self.shift_scale_rotate(xs, 5.0 / 100.0, true)),
            15 => cb::grie_rosen_base(&self.shift_scale_rotate(xs, 5.0 / 100.0, true)),
            // No `+1` offset -- module doc's verified F16 discrepancy note
            // (the printed report's eq (30) shows one; the code does not).
            16 => cb::escaffer6_base(&self.shift_scale_rotate(xs, 1.0, true)),
            other => unreachable!("Cec2014::new rejects fid outside 1..=16, got {other}"),
        };
        value + self.f_star()
    }
}

impl Problem for Cec2014 {
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
    fn fid_0_17_and_31_are_unknown() {
        for fid in [0u32, 17, 22, 23, 30, 31, 100] {
            assert!(
                matches!(Cec2014::new(fid, 10), Err(Cec2014Error::UnknownFid(f)) if f == fid),
                "fid={fid}"
            );
        }
    }

    #[test]
    fn fid_1_to_16_all_construct_at_dim_10_and_30() {
        for fid in 1u32..=16 {
            for &dim in &[10usize, 30] {
                assert!(Cec2014::new(fid, dim).is_ok(), "fid={fid} dim={dim}");
            }
        }
    }

    #[test]
    fn unsupported_dims_are_bad_dim_for_every_fid() {
        for fid in 1u32..=16 {
            for &dim in &[2usize, 5, 20, 50, 100] {
                assert!(
                    matches!(Cec2014::new(fid, dim), Err(Cec2014Error::BadDim(d)) if d == dim),
                    "fid={fid} dim={dim}"
                );
            }
        }
    }

    #[test]
    fn space_is_single_float_block_pm100() {
        for fid in 1u32..=16 {
            for &dim in &[10usize, 30] {
                let p = Cec2014::new(fid, dim).unwrap();
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
        // Module doc's 1.2 summary table, spot-checked directly against the
        // literal numbers (independent re-assertion of the `100*fid`
        // formula `Cec2014::f_star` actually implements).
        let expect = [
            (1u32, 100.0),
            (2, 200.0),
            (3, 300.0),
            (4, 400.0),
            (5, 500.0),
            (6, 600.0),
            (7, 700.0),
            (8, 800.0),
            (9, 900.0),
            (10, 1000.0),
            (11, 1100.0),
            (12, 1200.0),
            (13, 1300.0),
            (14, 1400.0),
            (15, 1500.0),
            (16, 1600.0),
        ];
        for (fid, fstar) in expect {
            assert_eq!(Cec2014::new(fid, 10).unwrap().f_star(), fstar, "fid={fid}");
        }
    }

    // ---- x = o pin: F_i(o_i) == F_i* exactly (module doc's derivation) ----

    #[test]
    fn fid_1_to_16_x_eq_o_is_exactly_f_star_dim10_and_dim30() {
        for fid in 1u32..=16 {
            for &dim in &[10usize, 30] {
                let p = Cec2014::new(fid, dim).unwrap();
                let out = p.evaluate_batch(&[g(p.o.clone())])[0];
                assert_eq!(out, p.f_star(), "fid={fid} dim={dim}");
            }
        }
    }

    // ---- data-integrity spot asserts (mirrors data.rs's own, at the
    // Cec2014 level -- confirms `new` actually wires the parsed data
    // through, not just that data.rs parses in isolation) ----

    #[test]
    fn constructed_o_and_m_have_the_right_shape_for_every_fid() {
        for fid in 1u32..=16 {
            for &dim in &[10usize, 30] {
                let p = Cec2014::new(fid, dim).unwrap();
                assert_eq!(p.o.len(), dim, "fid={fid} dim={dim}");
                assert_eq!(p.m.len(), dim, "fid={fid} dim={dim}");
                assert!(p.m.iter().all(|row| row.len() == dim), "fid={fid} dim={dim}");
            }
        }
    }

    // ---- random-point cross-checks vs. the compiled C reference (this
    // task's report has the full driver transcript and the g++
    // invocation; one random point per fid at dim=10 AND dim=30, `x = o +
    // U(-50,50)^dim`, seeded `Random(20260831)` in the probe script;
    // measured max relative deviation stated in the report).

    fn assert_close(fid: u32, dim: usize, x: &[f64], expect: f64) {
        let p = Cec2014::new(fid, dim).unwrap();
        let got = p.evaluate_batch(&[g(x.to_vec())])[0];
        let rel = ((got - expect) / expect).abs();
        assert!(rel < 1e-9, "fid={fid} dim={dim}: got {got}, expect {expect}, rel diff {rel}");
    }

    #[test]
    fn fid1_random_points_match_compiled_c_reference() {
        assert_close(1, 10, &[20.292119682916706, 33.77949881373659, -58.358919586187675, 61.6150228881862, -21.004604448183855, -58.177787927367305, -6.668568755922905, 25.670303776062596, 79.77203319921188, 65.1633810905091], 262486904.8395523);
        assert_close(1, 30, &[98.84985892995898, 91.92227114147362, -10.135472938715765, 97.0610751958374, 4.159260396945598, -68.84072061273622, -45.193569706591305, 32.223927741449344, 53.1749838401549, 77.99547544493531, -60.32379441348586, 45.9991342229144, 66.0649555363088, -20.043647122848796, 93.55892315111517, -83.09819664412055, -0.3168016774400968, 61.11394464256165, 23.512435919665418, 118.1986554373508, -11.364061571266944, -32.184423862322646, 95.42592381302194, 28.012982849011948, -18.718083054306476, 24.415439000779145, 28.66714780852773, -51.8610401539974, 26.22943803850321, -11.946839950923597], 5401981486.566844);
    }

    #[test]
    fn fid2_random_points_match_compiled_c_reference() {
        assert_close(2, 10, &[-56.22797268688652, 35.40937750393257, 3.993765817617472, -44.368190561438766, -15.690677101895787, -30.924570090686487, -43.66815535103423, 35.26945782285545, 7.874292472234508, 59.4555436363803], 8319236250.372579);
        assert_close(2, 30, &[-54.58394165319574, 24.934920845645298, -33.95006440369281, -30.34200004192583, -42.910747366957615, 5.142511760556147, -52.76150329467487, 38.24569730103649, -0.9585640586066404, 22.238643397288705, -37.73003968643327, -37.39950095340784, 3.761586139222551, 102.46988089198915, -34.78006313476478, 44.763331755950134, -29.307304952666357, 38.721355529803176, -104.51905366535402, -36.95878756629497, -83.79419188604187, 71.91453143504788, -106.22109184028852, 50.64724395567061, 37.30132244156434, 100.72425472950442, -35.837652725969875, 9.792565663479323, -76.93705662270725, 30.821021838648427], 38550410470.619934);
    }

    #[test]
    fn fid3_random_points_match_compiled_c_reference() {
        assert_close(3, 10, &[58.82280470356939, 6.375434727602695, 63.077347958876395, -33.461396636446366, -26.819173622974027, 85.40118240429285, 70.06284226659974, 24.61469597277879, -14.049516593248718, 29.50744508181232], 66067711.48866725);
        assert_close(3, 30, &[-3.5799895536753823, -45.58154633673746, 29.797754822855715, 45.24453189704606, -53.10571053208761, 55.66566291575905, 100.22948374150481, 50.416392889509964, 54.356897220085884, 24.0751166722955, -53.268408011897506, -29.9680787318365, -10.243511876633697, -16.68880698737034, 47.74647624211867, -6.285095396477715, -83.03398504364993, -54.37896757402922, -21.355972235651308, 4.802917739497516, -24.106204224005367, 49.34619379319918, 1.835790873947623, -20.968296134450725, 40.726072922750156, 34.43583281583939, -14.474605840509327, -55.997485115395904, -62.580803484502425, -30.6005906529398], 111811473.95822974);
    }

    #[test]
    fn fid4_random_points_match_compiled_c_reference() {
        assert_close(4, 10, &[-33.03070765500547, 71.98700447498335, -86.12935241171292, -32.23770860123061, 30.95849619948612, -73.29667114228533, 22.451836784749187, 51.032587743628184, 60.863908604117086, -103.73114185997622], 1100.1339753455695);
        assert_close(4, 30, &[-25.87437419886747, 45.15844628532075, -39.48451633812314, -28.485162890767498, -3.920285672219187, -59.03433294409018, 10.821388326764897, 75.2573594947701, 34.6927671653559, -63.843295905714854, 46.44548035359108, 22.17118177574801, 74.68335294108061, -5.190638652318441, 57.0066243385163, -45.943921519330054, -59.51380619302641, 21.861222670412918, -116.61515534265581, -70.19439989369226, -47.39282095598997, -41.366319872896476, 52.400916270883464, 87.43130835810975, -117.17060052243552, -27.859803773883144, 24.701944881918223, -30.73232311469834, -1.620387976048864, -7.265733922604692], 11035.721740456776);
    }

    #[test]
    fn fid5_random_points_match_compiled_c_reference() {
        assert_close(5, 10, &[13.358013587222558, -92.96408610585127, 53.18344429482019, -22.420564506836158, 1.2201849761658963, 13.394901261459609, -0.39018441881622934, -123.75944548780149, -108.65034716575576, -55.950852861276175], 521.9579233431177);
        assert_close(5, 30, &[-41.86250366517946, -109.79879552044233, 37.39873529751608, 2.1798157559654783, 7.330308139178367, 60.80542480277856, -7.54177804850471, -34.46961363790929, -84.3740414347763, -10.448044994607216, 10.652443630389492, 10.736948947770578, -28.73828905166382, 48.92312326455632, -10.449392097363685, 48.76947172323707, 0.5422253502284065, -63.77016094240811, -45.62649215942599, 15.640270197762149, 35.887584329047186, -44.29096054387787, -98.05925100799526, -55.654472614752194, -90.35224228822617, -26.807719550031337, 4.413088608470282, -24.613495968028587, -31.644072160826262, 42.193378784119766], 521.5938314977633);
    }

    #[test]
    fn fid6_random_points_match_compiled_c_reference() {
        assert_close(6, 10, &[18.263577775038122, 16.405868078934404, 107.01887273116084, -23.62451665420572, -63.98137228592973, -100.85126214230898, 84.84473651516821, 22.47082847387931, -48.82689812073097, 46.73843165588438], 615.5701770099816);
        assert_close(6, 30, &[59.5847227547746, -31.004433359559748, 10.515949147366634, -62.15585343422946, -48.13377641349614, -28.753158065757006, 90.02006296634781, 68.30311352918713, 0.30078676106409574, 60.366540506542194, 33.6390671572655, 55.776490729089545, -5.607863347674446, 50.0349362216555, -13.207016572613156, 29.035632660593137, -37.69761027368104, -46.768116713326904, -108.697111095428, -113.01923246707014, -52.702978003860004, -33.92655807648728, 0.5432998514670402, 65.31918264923169, 17.18445671215536, 7.4976474392505, 68.82572704767767, 88.10076542822638, -75.65499673114115, -56.22940236043334], 647.6186164207649);
    }

    #[test]
    fn fid7_random_points_match_compiled_c_reference() {
        assert_close(7, 10, &[9.317380817569841, 22.722081385098832, 57.57833000055226, 25.854123078121987, 28.23629506515897, 93.25498476597271, 23.49183242581961, 122.22822016883782, -87.21080628603485, 21.206453284948125], 828.638065692721);
        assert_close(7, 30, &[58.83370257238995, 3.7609488898052703, 50.244221419259546, 33.80684163440454, 55.02922466198956, 72.12230144986752, 35.343545373539335, 115.36269588238035, -52.39601612376518, 74.2858254822221, 5.156207941115653, 63.0212379960381, -15.412266870902016, -37.69981269485994, 58.1330816002927, -82.49168139275724, 31.931533169749812, 96.21818953162172, 10.757395210333286, 33.88235780674199, 33.59844982011149, -17.177645615997395, -82.17882179211279, -21.183834541949167, -64.46063196497528, -28.953990878956084, -5.67764548504357, -3.4580890888517786, 37.66655869401899, 58.78506884297157], 1068.476127407669);
    }

    #[test]
    fn fid8_random_points_match_compiled_c_reference() {
        assert_close(8, 10, &[-28.953470598990855, 58.165813376003584, -25.542371481245787, -47.40309613030512, -40.27421844356327, -85.98959493167592, 75.91119941943654, -31.73613617793137, 19.125463658449284, -14.394871607656462], 919.7987743130766);
        assert_close(8, 30, &[34.31240404939892, 47.45233135097574, -2.60621719214069, 25.113449687244085, -81.37621418821644, -40.86965049226411, 45.18380012604466, -50.972577149707234, 84.82387399115514, -29.656216501384968, 35.21263387650863, 53.77965758167524, 14.530308455292499, 35.00612554448272, -23.759000582133297, -35.659022458432965, -14.448216710087344, 17.875934559827805, 38.12108479546614, -97.29893614360827, -3.180834956571026, 24.45316785191288, -98.99800294282355, 34.60967196069943, -36.40897480654302, -46.188690769136386, -62.660113278087124, 103.04005661318682, 14.116630205562387, -24.245365314714896], 1217.2236520275771);
    }

    #[test]
    fn fid9_random_points_match_compiled_c_reference() {
        assert_close(9, 10, &[43.58155264265994, -97.08429695274319, 4.216536972804377, -83.5868984279285, -36.43397441081147, 57.08811416610351, 9.397174995879944, -2.376663595394845, -5.409499704896497, -16.773010868720444], 1060.6656555976695);
        assert_close(9, 30, &[-0.6772736813509397, -71.93211329289021, 76.01070331241718, -106.20385988779964, -34.94398166363827, 59.04318174658452, -76.68332000193229, -1.8322370634309095, 64.29121353847728, 56.379046318159965, 8.313904715462147, -30.36336220289801, 32.331108870821566, 40.58670552066723, 16.515058139781793, -28.041978466269924, -42.37715802661567, 21.690784483471077, 61.42643660332175, 59.99683044922362, 65.17517867586206, -57.75261242792483, -47.382142795055856, -56.58269253875632, 67.74790783672398, -62.28993825947808, -58.633113461669794, 3.100630282911382, 1.5004099347549982, 28.158354757053182], 1377.2762304366283);
    }

    #[test]
    fn fid10_random_points_match_compiled_c_reference() {
        assert_close(10, 10, &[74.20764467100598, -55.701002423715074, -45.872326440019314, -59.311537033270724, -24.89913939988729, 51.74101488721925, 2.7753905783233392, 82.34555201541066, 1.5218548083831962, -23.916185827405897], 3880.8642048270144);
        assert_close(10, 30, &[97.83450143386666, -33.09578939235075, -14.148033588740432, -48.81119273095603, -39.042913621076735, -1.382720844202975, -56.15627305324581, 29.44304655019201, -3.655078672498149, 31.505535879546045, -68.6111525614803, 6.006275359789953, -96.99401470624764, 14.213579533366634, 12.945173487016845, -49.62278767121522, 44.783279327640784, -62.231658578082204, 4.143679155168549, 65.39018526866474, -6.791355895216483, -10.605256236149131, -60.724828558349955, -8.128207888130355, -23.940521329019724, 22.71073062747601, 55.00359032773018, 63.625785647232604, -29.942201179787467, 9.162986627156783], 12413.902459624444);
    }

    #[test]
    fn fid11_random_points_match_compiled_c_reference() {
        assert_close(11, 10, &[-12.078635662956373, 16.986891809059117, 113.40988728879591, -93.1442805856659, 5.888034743954755, 41.4473719121178, -33.87654468022631, -80.85028383833497, -123.29296850065722, 32.53800395330586], 4571.3599952032255);
        assert_close(11, 30, &[-4.723445366082288, -3.1338553259837028, 42.362928258729305, -55.16425255811332, -4.991864384050011, 12.221394209631512, 4.580839925455763, -8.37716893018576, -99.13751435017868, -12.534877146350965, -8.432488790746717, 45.21542281050573, -8.070900927528562, 6.554268301915023, 10.453075219279341, -5.941068022836539, -24.625945060007503, 116.08752949887324, -48.00093360553232, 25.6617003387018, -20.416366459585014, -20.384474657081554, -33.714019232323224, 35.54953523120825, -98.04728140205117, -106.1324799828713, 17.477074702733745, 15.942275621272913, -3.2105043080824274, -28.916962697447758], 12800.922036809085);
    }

    #[test]
    fn fid12_random_points_match_compiled_c_reference() {
        assert_close(12, 10, &[-2.7008167205665288, 19.649672271282917, -35.78656334684483, 53.24090936981992, 63.91885483512286, 22.59011602016909, -38.08444113658996, -40.644126496957085, 19.925454763485888, 6.73897465267974], 1204.2339351186513);
        assert_close(12, 30, &[67.89374411413789, 72.29044920128524, -102.57584964370719, 24.206593300426878, 111.2199317975483, 37.64027232293715, -121.15270339793777, 6.039926280400614, 44.44096726525782, 53.69789210815611, -28.7326751103315, -47.75074162854102, 54.30975490122766, -3.4450494453643614, 63.50501165968008, -87.46150962216598, -73.49045085066662, 33.82075476587209, 60.45158185181515, 95.67938195130435, -35.83962944013885, -2.3946059432300757, -66.51238813578343, -23.76569309536238, -89.86068895470214, 103.61172971333373, -95.9145513594577, -84.99677239717727, 4.6708319106117955, 36.20852567630522], 1221.440133690572);
    }

    #[test]
    fn fid13_random_points_match_compiled_c_reference() {
        assert_close(13, 10, &[-4.596592177666878, 10.250802816716075, 28.562752303446253, -39.46002755657802, -4.258065776956755, 55.8401895360049, -121.3131056504267, 75.02861292223776, -64.07149294870055, 33.33230650156403], 1301.654157928757);
        assert_close(13, 30, &[39.557564433080884, 43.03588046509378, 54.23710942250816, 25.68657530069501, -51.39745535608264, 99.03159965093451, -63.03207183959251, 100.50000191685528, -22.21245779885703, 43.84743448172942, 33.65193388567022, 1.3181421415655485, -43.592156867666205, -4.3420608277448665, 72.53480174016295, -62.28726218943419, -21.635158839133055, 102.01860964483316, -24.017411074353333, -75.62349834191113, 8.037455259910928, -46.58756991899442, -71.59842574180581, -86.87401468277396, 18.214337524582405, -83.32386000489991, -26.41036431503906, -105.03484588760558, 24.550401982226163, -39.9397316712861], 1305.2308793320176);
    }

    #[test]
    fn fid14_random_points_match_compiled_c_reference() {
        assert_close(14, 10, &[2.8273547088266753, 66.18728959704187, 52.13477450010174, 96.19908404146635, -52.14203442014383, 66.73826275688289, 23.085289871354973, 124.72199526182285, 44.5045731795983, 95.15112489035765], 1450.2245966161279);
        assert_close(14, 30, &[-62.00549268909653, 54.37052752997501, 108.62483944766909, 62.303209618331614, 21.551420795139194, 23.410346260220926, -11.418428242443646, 92.13544148237172, 101.61839829692897, 58.112099612630395, -39.52681511900279, 38.99885635136674, -80.477591216826, 16.991949979799298, 106.52290109541829, 66.90718749238232, 25.86788991204753, 24.357839938187176, -64.09569731921769, 30.50419610359822, 24.35978182782482, -47.83008887621743, 45.73222843111607, 43.17787930098717, 81.43386890313286, -59.04973630489123, 70.43631578771861, 39.79402391470925, -43.01629582415269, 17.616727119662414], 1516.5335940638822);
    }

    #[test]
    fn fid15_random_points_match_compiled_c_reference() {
        assert_close(15, 10, &[-32.31781005345859, 8.014305056070349, 7.251583417272506, 26.292083851659484, -32.16899793715288, -37.54741614361528, -0.20628353064022775, 59.225880187417275, 66.53962168444811, -3.353839314069482], 9929.009535466175);
        assert_close(15, 30, &[34.15500958881724, 3.954853207999278, 72.62519028595347, -26.96742705283839, -38.734180910346474, -0.319866307196385, 19.01663006137413, 68.73296392277977, -27.311321114520595, 18.835633107458406, 36.37438191019063, -8.98558217542572, 49.05234319576495, -19.673634610843596, -63.504858292620334, 55.23230612127442, 14.027356840629459, -37.78788064799153, 78.70188231317525, 84.81716476055455, 18.86809201035959, -26.090498368146598, -11.989114365284273, 54.08560592399772, 47.03532289901739, 25.42226995806702, 78.49627199005148, 119.066872643053, 37.171899602405546, 49.81246749580583], 1096314.7302577763);
    }

    #[test]
    fn fid16_random_points_match_compiled_c_reference() {
        assert_close(16, 10, &[-66.17551935387122, -70.77259023389352, -51.31666211504242, 5.237516101551918, -66.24730274793025, -66.85260544100458, -1.779163289733063, -44.83034678772697, 42.33296926107343, -1.6216829855607529], 1604.5487345718104);
        assert_close(16, 30, &[-13.856681264772163, -80.31236148331706, -67.25473828754367, 13.076529418370725, -80.27064155592497, -80.44143911324262, 31.741558119411422, -66.97715691110488, 5.836257118945433, 62.78584808228203, -5.082542866069652, 29.774068300345455, -102.03244516697265, -63.496707940558395, -84.1803604353351, -94.38521088317606, -107.84978612452696, -44.733317817675655, -47.35775439150855, -51.733067755274895, -99.04312105175654, -19.36413674810143, 3.68952244753757, 85.99791585419263, 31.00614815267018, -0.7861043347106076, 24.651618849189646, 3.8805055153506274, 3.0967488149392253, -30.484676526968464], 1615.4431966124566);
    }

    // ---- Evaluator integration + budget counting ----

    #[test]
    fn evaluator_counts_budget_and_tracks_best() {
        let p = Cec2014::new(1, 10).unwrap();
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
        for fid in 1u32..=16 {
            let p = Cec2014::new(fid, 10).unwrap();
            assert_eq!(Problem::optimum(&p), Some(p.f_star()), "fid={fid}");
        }
    }

    // ---- malformed genotype handling ----

    #[test]
    fn wrong_length_float_block_evaluates_to_infinity_not_panic() {
        let p = Cec2014::new(1, 10).unwrap();
        let out = p.evaluate_batch(&[g(vec![0.0; 3])]);
        assert_eq!(out, vec![f64::INFINITY]);
    }

    #[test]
    fn non_float_block_evaluates_to_infinity() {
        let p = Cec2014::new(1, 10).unwrap();
        let bad = Genotype { blocks: vec![BlockValues::Perm(vec![0, 1, 2])] };
        let out = p.evaluate_batch(&[bad]);
        assert_eq!(out, vec![f64::INFINITY]);
    }
}
