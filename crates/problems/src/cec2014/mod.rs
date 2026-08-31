//! The CEC 2014 Special Session and Competition benchmark suite: fid 1-16
//! (unimodal F1-F3, simple multimodal F4-F16, M3-6 T2), fid 17-22 (hybrid
//! functions 1-6, M3-6 T3), and fid 23-30 (composition functions 1-8, M3-6
//! T4, this task's addendum, module doc section below, "M3-6 T4") are all
//! implemented. `Cec2014::new` accepts fid `1..=30`, the FULL suite --
//! mirrors `cec2022`'s own T5(fid 1-5) -> T6(fid 6-8) -> T7(fid 9-12)
//! staging (`cec2022/mod.rs`'s module doc has that precedent in full), now
//! complete for this suite.
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
//!
//! ## M3-6 T3: Hybrid Functions (fid 17-22)
//!
//! Source (PROVENANCE, re-verified this task, not reused from memory): the
//! SAME two artifacts T2 fetched (module doc's opening PROVENANCE section,
//! SHA-256 `a506b9b2...` for the report PDF and `1a210560...` for
//! `cec14-c-code.zip`, both re-checked this task with `shasum -a 256`
//! against the scratchpad copy T2's report already pinned -- unchanged).
//! The vendored C was again COMPILED (`g++ -O2`, the SAME two portability
//! edits T2's module doc already records -- re-verified present, not
//! re-applied, since T2's edited `.cpp` was reused directly from the
//! scratchpad) and RUN through the unmodified `cec14_test_func()` entry
//! point for every `x=o` pin and random-point fixture below (a purpose-built
//! probe driver, `argv: fid dim x0 x1 ...` -> `%.20f`-printed `f(x)`, the
//! SAME `probe.cpp` T2's report already describes -- reused unmodified).
//!
//! ### Dispatch: which C function each fid actually calls (quoted, `switch`
//! block immediately following fid 1-16's, module doc's dispatch section
//! above)
//!
//! ```text
//! case 17: hf01(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=1700.0; break;
//! case 18: hf02(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=1800.0; break;
//! case 19: hf03(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=1900.0; break;
//! case 20: hf04(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=2000.0; break;
//! case 21: hf05(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=2100.0; break;
//! case 22: hf06(&x[i*nx],&f[i],nx,OShift,M,SS,1,1); f[i]+=2200.0; break;
//! ```
//! UNLIKE `cec2022`'s hybrid dispatch (whose internal `hf02`/`hf06`/`hf10`
//! names don't track the report's "Hybrid Function 1/2/3" numbering at all,
//! `cec2022/mod.rs`'s module doc's naming curiosity note), CEC 2014's
//! dispatch is a CLEAN 1:1 mapping -- fid 17 calls `hf01` ("Hybrid Function
//! 1"), fid 18 calls `hf02`, ..., fid 22 calls `hf06`, no naming mismatch to
//! record here. `Fi* = 1700..2200` for fid 17-22 continues the SAME
//! `100*fid` closed form fid 1-16 already established (module doc's section
//! 1.2 note) -- [`Cec2014::f_star`]'s existing `100.0 * fid` formula needs no
//! per-fid override for this task's fids either.
//!
//! ### `sr_func`, `G_nx`/`G` segmentation, and the shuffle read -- IDENTICAL
//! shape to `cec2022`'s own hybrids (quoted for this suite's own
//! independent provenance trail; `hf01`'s exact body shown, all six `hf0N`
//! share this identical structure -- verified by reading all six, not just
//! `hf01`)
//!
//! ```text
//! void hf01 (double *x, double *f, int nx, double *Os,double *Mr,int *S,int s_flag,int r_flag)
//! {
//!     int i,tmp,cf_num=3;
//!     double fit[3];
//!     int G[3],G_nx[3];
//!     double Gp[3]={0.3,0.3,0.4};
//!     tmp=0;
//!     for (i=0; i<cf_num-1; i++) { G_nx[i] = ceil(Gp[i]*nx); tmp += G_nx[i]; }
//!     G_nx[cf_num-1]=nx-tmp;
//!     G[0]=0;
//!     for (i=1; i<cf_num; i++) { G[i] = G[i-1]+G_nx[i-1]; }
//!     sr_func (x, z, nx, Os, Mr, 1.0, s_flag, r_flag); /* shift and rotate */
//!     for (i=0; i<nx; i++) { y[i]=z[S[i]-1]; }
//!     i=0; schwefel_func(&y[G[i]],&fit[i],G_nx[i],Os,Mr,0,0);
//!     i=1; rastrigin_func(&y[G[i]],&fit[i],G_nx[i],Os,Mr,0,0);
//!     i=2; ellips_func(&y[G[i]],&fit[i],G_nx[i],Os,Mr,0,0);
//!     f[0]=0.0;
//!     for(i=0;i<cf_num;i++) { f[0] += fit[i]; }
//! }
//! ```
//! Byte-identical `ceil`-based `G_nx`/`G` arithmetic to `cec2022`'s own
//! `hf02`/`hf06`/`hf10` (`cec2022/mod.rs`'s module doc, Eq (21) section,
//! quoted there) -- [`crate::cec_basics::segment_sizes`]/
//! [`crate::cec_basics::segment_starts`] (already suite-agnostic, extracted
//! there in M3-6 T1) are reused AS-IS, no new segmentation arithmetic this
//! task. The outer `sr_func` call is ALWAYS `sh_rate=1.0` (full shift+rotate
//! of the WHOLE `nx`-length vector, once, before shuffling -- same "ONE
//! shared rotation, not one per component" shape `cec2022/mod.rs`'s module
//! doc already documents for that suite's hybrids, re-verified here by
//! reading all six `hf0N` bodies directly: every one's own `sr_func` call is
//! `sh_rate=1.0`). The shuffle read `y[i]=z[S[i]-1]` is BYTE-IDENTICAL
//! syntax to `cec2022`'s `hf02` (`cec2022/mod.rs`'s module doc, "Shuffle"
//! section, quoted there) -- 1-based `S` converted to 0-based ON LOAD by
//! [`data::shuffle_indices`] (`data.rs`'s module doc), so
//! [`Cec2014::eval_one`]'s hybrid arm builds `y` the same way `cec2022`
//! does: `y[i] = z[self.shuffle[i]]`.
//!
//! ### Per-component evaluation: NO shift, NO rotation, ONLY the
//! component's own inner `sh_rate` -- IDENTICAL branch to `cec2022`'s own
//! (module doc's dispatch section above already quotes `sr_func`'s
//! `s_flag==0` branch in full; every `hf0N` component call passes literal
//! `0,0` for `(s_flag,r_flag)`, e.g. `schwefel_func(&y[G[i]],&fit[i],G_nx[i],Os,Mr,0,0)`)
//!
//! **No fid-7-shaped bug found here** -- UNLIKE `cec2022`'s fid 7
//! (`SchafferF7Buggy`, `cec2022/mod.rs`'s module doc, whose component reads
//! the GLOBAL `y` from index `0` instead of its assigned pointer offset
//! `&y[G[5]]`), every `hf0N` component call here correctly passes
//! `&y[G[i]]` (this component's OWN pointer-offset segment) and every base
//! function it calls (`schwefel_func`, `rastrigin_func`, `ellips_func`,
//! `bent_cigar_func`, `hgbat_func`, `griewank_func`, `weierstrass_func`,
//! `rosenbrock_func`, `escaffer6_func`, `discus_func`, `grie_rosen_func`,
//! `katsuura_func`, `happycat_func`, `ackley_func`) reads its OWN `x`
//! parameter inside its own `sr_func` call, not a raw global buffer --
//! VERIFIED by reading each function's own body directly (module doc's
//! "Basic-function reuse" section above already quotes/cross-checks every
//! one of these fourteen for fid 1-16's own standalone dispatch; none of
//! them is `schaffer_F7_func`, the ONLY function `cec2022/mod.rs`'s module
//! doc identifies as having this bug, and `schaffer_F7_func` is never called
//! by any `hf0N` here -- confirmed by grepping every call site in the
//! vendored source). Each component's own `sh_rate` constant (module doc's
//! "Basic-function reuse" table above, re-confirmed directly against every
//! `sr_func` call inside `ellips_func`/`bent_cigar_func`/.../`hgbat_func`'s
//! bodies this task, `awk`-mapped call-site transcript in this task's
//! report) is IDENTICAL to the constant that same function uses standalone
//! for fid 1-16 -- [`HybridComponent::sh_rate`] below reuses those same
//! sixteen literals.
//!
//! ### Hybrid composition tables -- quoted from BOTH the C (`Gp` arrays +
//! component call order, all six `hf0N`) AND the report (section "C. Hybrid
//! Functions"), CROSS-CHECKED: NO discrepancy found for ANY of the six
//! (unlike `cec2022`'s fid 7, whose printed `p` array had one too many
//! entries) -- every `p_i` value and every component NAME/ORDER matches
//! digit-for-digit and word-for-word between the two sources
//!
//! Report (PDF `pdftotext -layout`, section "C. Hybrid Functions",
//! transcribed verbatim):
//! ```text
//! 17) Hybrid Function 1        N=3   p=[0.3,0.3,0.4]
//!     g1: Modified Schwefel's Function f9   g2: Rastrigin's Function f8   g3: High Conditioned Elliptic Function f1
//! 18) Hybrid Function 2        N=3   p=[0.3,0.3,0.4]
//!     g1: Bent Cigar Function f2   g2: HGBat Function f12   g3: Rastrigin's Function f8
//! 19) Hybrid Function 3        N=4   p=[0.2,0.2,0.3,0.3]
//!     g1: Griewank's Function f7   g2: Weierstrass Function f6   g3: Rosenbrock's Function f4   g4: Scaffer's F6 Function f14
//! 20) Hybrid Function 4        N=4   p=[0.2,0.2,0.3,0.3]
//!     g1: HGBat Function f12   g2: Discus Function f3   g3: Expanded Griewank's plus Rosenbrock's Function f13   g4: Rastrigin's Function f8
//! 21) Hybrid Function 5        N=5   p=[0.1,0.2,0.2,0.2,0.3]
//!     g1: Scaffer's F6 Function f14   g2: HGBat Function f12   g3: Rosenbrock's Function f4   g4: Modified Schwefel's Function f9   g5: High Conditioned Elliptic Function f1
//! 22) Hybrid Function 6        N=5   p=[0.1,0.2,0.2,0.2,0.3]
//!     g1: Katsuura Function f10   g2: HappyCat Function f11   g3: Expanded Griewank's plus Rosenbrock's Function f13   g4: Modified Schwefel's Function f9   g5: Ackley's Function f5
//! ```
//! C (`Gp` arrays + component call order, transcribed from all six `hf0N`
//! bodies directly, this task's report has the full six-function
//! transcript):
//!
//! | fid | `hf0N` | `Gp` (C) | Component call order (C) |
//! |---|---|---|---|
//! | 17 | `hf01` | `{0.3,0.3,0.4}` | `schwefel_func, rastrigin_func, ellips_func` |
//! | 18 | `hf02` | `{0.3,0.3,0.4}` | `bent_cigar_func, hgbat_func, rastrigin_func` |
//! | 19 | `hf03` | `{0.2,0.2,0.3,0.3}` | `griewank_func, weierstrass_func, rosenbrock_func, escaffer6_func` |
//! | 20 | `hf04` | `{0.2,0.2,0.3,0.3}` | `hgbat_func, discus_func, grie_rosen_func, rastrigin_func` |
//! | 21 | `hf05` | `{0.1,0.2,0.2,0.2,0.3}` | `escaffer6_func, hgbat_func, rosenbrock_func, schwefel_func, ellips_func` |
//! | 22 | `hf06` | `{0.1,0.2,0.2,0.2,0.3}` | `katsuura_func, happycat_func, grie_rosen_func, schwefel_func, ackley_func` |
//!
//! Every row matches its report counterpart exactly: `p` values identical,
//! `g1..gN` names/order identical (`escaffer6_func` is this suite's
//! "Scaffer's F6"/eq (30) function, module doc's fid-16 discrepancy note
//! above already establishes `escaffer6_func` as the C's actual Scaffer's F6
//! dispatch target; `grie_rosen_func` is "Expanded Griewank's plus
//! Rosenbrock's"; `ellips_func` is "High Conditioned Elliptic", matching
//! fid 1's own name). [`Cec2014::hybrid_spec`] transcribes all six rows
//! verbatim from this table.
//!
//! ### New component enum ([`HybridComponent`]) -- every base function ALREADY
//! exists in `crate::cec_basics` (all sixteen: `ellips_base`,
//! `bent_cigar_base`, `discus_base`, `f2_base` (Rosenbrock, `+1`'d
//! externally, same convention fid 4's own dispatch arm above uses),
//! `ackley_base`, `weierstrass_base`, `griewank_base` (standalone, NOT
//! `grie_rosen_base`), `f4_base` (Rastrigin), `schwefel_base`,
//! `katsuura_base`, `happycat_base`, `hgbat_base`, `grie_rosen_base`
//! (handles its own `+1` internally), `escaffer6_base`) -- NO new base
//! function needed this task (unlike T2's Weierstrass, which was genuinely
//! new to `cec_basics` at the time)
//!
//! ### `x = o` pin (Step 2) for fid 17-22: why `F_i(o_i) == F_i*` EXACTLY,
//! dim=10 AND dim=30 -- derived per-component (generalizing fid 1-16's own
//! zero-argument derivation, module doc above, to an ARBITRARY segment
//! length `n` instead of the full `dim`), then VERIFIED against the
//! compiled reference for all twelve `(fid,dim)` pairs
//!
//! At `x=o`: the outer `sr_func` gives `z=M*(x-o)*1.0=M*0=0` (IDENTICAL
//! all-zero argument to fid 1-16's own pin, module doc above) -- shuffling
//! an all-zero vector yields an all-zero vector regardless of the
//! permutation (`y[i]=z[S[i]-1]=0` for every `i`), so EVERY segment
//! `y[G[j]..G[j]+G_nx[j]]` is all-zero, for every hybrid, every dim, every
//! shuffle. Each component then reads its all-zero segment through its own
//! `s_flag=0,r_flag=0` branch of `sr_func` (`sr_x[i]=x[i]*sh_rate`) -- `0 *
//! sh_rate = 0` exactly regardless of WHICH `sh_rate` constant or HOW LONG
//! the segment is, so every component's own base-function argument is an
//! all-zero vector of length `G_nx[j]` (not necessarily `dim` -- the
//! generalization fid 1-16's pin didn't need). Per base function, at an
//! ALL-ZERO argument of ARBITRARY length `n>=1` (re-derived here since fid
//! 1-16's pin only needed `n=dim`):
//! - **Elliptic, Bent Cigar, Discus, Rastrigin, Griewank (standalone),
//!   Expanded Scaffer's F6**: each is a pure elementwise sum/product of
//!   `zi=0` terms (module doc's fid 1-16 pin section already derives each of
//!   these `=0` termwise; the derivation is per-ELEMENT, so it holds for ANY
//!   `n`, not just `n=dim`) -- **0 exactly for any `n>=1`.**
//! - **Rosenbrock (`+1`'d to all-ones)**: `f2_base(1,1,...,1) =
//!   sum_{i=0}^{n-2}(100*(1-1)^2+(1-1)^2) = 0` for any `n>=1` (empty sum when
//!   `n=1`) -- **0 exactly.**
//! - **Griewank-Rosenbrock (`grie_rosen_base`, internal `+1` to all-ones)**:
//!   every cyclic pair (including the `n=1` self-pair `z1[0],z1[0]`) gives
//!   `tmp1=1-1=0,tmp2=0,temp=0`, `temp^2/4000-cos(0)+1=0-1+1=0` -- **0
//!   exactly for any `n>=1`.**
//! - **Weierstrass**: fid 1-16's own pin note (module doc above) establishes
//!   the inner-sum-vs-`sum2` cancellation is bit-exact PER INDEX `i`,
//!   independent of `n` -- **0 exactly for any `n>=1`.**
//! - **HGBat, HappyCat**: shift-to-origin gives `zs_i=-1` for `n` elements;
//!   `r2=n, sum_z=-n` (both exact integer-valued `f64`, any `n`); HGBat's
//!   `r2^2-sum_z^2=n^2-n^2=0` bit-exact (same `pow(x,2.0)` magnitude);
//!   HappyCat's `|r2-n|=|n-n|=0` exact; both then reduce to remainder
//!   `(0.5n+(-n))/n=-0.5` (exact for any `n!=0`) plus the fixed `+0.5` ->
//!   **0 exactly for any `n>=1`.**
//! - **Katsuura**: `floor(0+0.5)=floor(0.5)=0` exact per `(i,j)` term,
//!   independent of `n` (module doc's fid 1-16 pin note, same reasoning) --
//!   **0 exactly for any `n>=1`.**
//! - **Ackley's**: `sum1=0`; `sum2=sum(cos(0))=n`; `sum1'=-0.2*sqrt(0/n)=0`;
//!   `sum2'=n/n=1` exact for any `n!=0`; `f=E-20*exp(0)-exp(1)+20=E-exp(1.0)`
//!   -- independent of `n` entirely (the `n` cancels via `sum2/n`) --
//!   VERIFIED empirically (same argument fid 1-16's own Ackley pin note
//!   makes, not re-claimed as a from-scratch identity) that
//!   `std::f64::consts::E - 1f64.exp() == 0.0` bit-exact on this platform.
//! - **Modified Schwefel's**: NOT an algebraic identity (module doc's fid
//!   1-16 pin note, same caveat) -- but VERIFIED EMPIRICALLY, and *stronger*
//!   here: `python3` (`term = -zi*sin(sqrt(zi))` at
//!   `zi=4.209687462275036e+02`, `c=4.189828872724338e+02`) confirms
//!   `term+c==0.0` bit-exact PER-ELEMENT (not merely "the `dim`-length sum
//!   happens to cancel") -- so `n*term+c*n = n*(term+c) = n*0.0 = 0.0`
//!   exactly for ANY segment length `n`, including the shorter hybrid
//!   segments this task introduces (fid 1-16's own pin only exercised
//!   `n=dim`; this task's report has the transcript proving the per-element
//!   cancellation directly, closing that generalization gap).
//!
//! Summing three, four, or five exact `0.0` components (fid 17/18: 3 each;
//! fid 19/20: 4 each; fid 21/22: 5 each) gives `F_i(o_i) = 0 + F_i* = F_i*`
//! EXACTLY -- confirmed by BOTH the per-component algebraic derivation above
//! AND the compiled C reference (`%.20f`-format, all twelve `(fid,dim)`
//! pairs in `{17..=22} x {10,30}` print the exact integer `100*fid`, zero
//! fractional residue, this task's report has the transcript) AND this
//! module's own `fid_17_to_22_x_eq_o_is_exactly_f_star_dim10_and_dim30`
//! test.
//!
//! ## M3-6 T4: Composition Functions (fid 23-30)
//!
//! Source (PROVENANCE, re-verified this task): the SAME two artifacts T2/T3
//! fetched (module doc's opening PROVENANCE section) -- re-checked this task
//! with `shasum -a 256` against the scratchpad copies T2's report already
//! pinned (report PDF `a506b9b2...`, `cec14-c-code.zip` extraction's
//! `cec14_test_func.cpp` at `scratchpad/t2_build/src/cec14_test_func.cpp`,
//! SHA-256 `a629bb7b35d24d9e43dea977e097b01972b5495cb0db0cc084c2346b84522c66`
//! -- IDENTICAL to T3's own pinned hash, no re-fetch needed). Re-compiled
//! fresh this task in an isolated scratchpad build dir (`g++ -O2 -o probe
//! cec14_test_func.cpp probe.cpp -lm`, only harmless deprecated-`sprintf`
//! warnings) using the SAME unmodified `probe.cpp` T2/T3 already describe
//! (`argv: fid dim x0 x1 ...` -> `%.20f`-printed `f(x)`, dispatches through
//! the unmodified `cec14_test_func()` entry point for ANY `fid`, so fid
//! 23-30 needed no probe-side change). All `input_data/` for fid 23-30 was
//! ALREADY vendored in `crates/problems/data/cec2014/` from T2's full-30-fid
//! vendoring pass -- confirmed present, byte-identical to the scratchpad
//! probe's own copy, no re-vendoring this task.
//!
//! ### Dispatch: which C function each fid actually calls (quoted, `switch`
//! block immediately following fid 17-22's own, module doc's dispatch
//! section above)
//!
//! ```text
//! case 23: cf01(&x[i*nx],&f[i],nx,OShift,M,1); f[i]+=2300.0; break;
//! case 24: cf02(&x[i*nx],&f[i],nx,OShift,M,1); f[i]+=2400.0; break;
//! case 25: cf03(&x[i*nx],&f[i],nx,OShift,M,1); f[i]+=2500.0; break;
//! case 26: cf04(&x[i*nx],&f[i],nx,OShift,M,1); f[i]+=2600.0; break;
//! case 27: cf05(&x[i*nx],&f[i],nx,OShift,M,1); f[i]+=2700.0; break;
//! case 28: cf06(&x[i*nx],&f[i],nx,OShift,M,1); f[i]+=2800.0; break;
//! case 29: cf07(&x[i*nx],&f[i],nx,OShift,M,SS,1); f[i]+=2900.0; break;
//! case 30: cf08(&x[i*nx],&f[i],nx,OShift,M,SS,1); f[i]+=3000.0; break;
//! ```
//! Clean 1:1 mapping, same shape T3's own fid-17-22 dispatch note already
//! establishes: fid 23 -> `cf01` ("Composition Function 1"), ..., fid 30 ->
//! `cf08` ("Composition Function 8"). `Fi* = 2300..3000` continues the SAME
//! `100*fid` closed form fid 1-22 already established -- `Cec2014::f_star`'s
//! existing formula needs no per-fid override for fid 23-30 either. The
//! outer `r_flag` literal passed to every `cf0N` call is always `1` (never
//! `0`) -- module doc's per-fid tables below record each COMPONENT's own
//! rotate choice, which is sometimes hardcoded `0` INSIDE `cf01`/`cf02`
//! regardless of this outer literal (the divergence note below).
//!
//! ### The `cf_num=10`-hardcoded data-loading quirk (verified directly, NOT
//! assumed from the brief's "one row per component" phrasing -- the actual
//! grammar is subtler)
//!
//! `cec14_test_func`'s initializer (quoted in full, `func_num>=23` branches,
//! `int cf_num=10` is a LOCAL declared at the very top of the function,
//! module doc's `data.rs` sibling doc quotes the same loops):
//! ```text
//! int cf_num=10,i,j;
//! ...
//! if (func_num<23) { M=malloc(nx*nx*...); for(i<nx*nx) fscanf(...); }
//! else { M=malloc(cf_num*nx*nx*...); for(i<cf_num*nx*nx) fscanf(...); }
//! ...
//! if (func_num<23) { OShift=malloc(nx*...); for(i<nx) fscanf(...); }
//! else {
//!     OShift=malloc(nx*cf_num*...);
//!     for(i=0;i<cf_num-1;i++) { for(j<nx) fscanf(...,&OShift[i*nx+j]);
//!                                fscanf(fpt,"%*[^\n]%*c"); }
//!     for(j<nx) fscanf(...,&OShift[(cf_num-1)*nx+j]);
//! }
//! ...
//! else if (func_num==29||func_num==30) {
//!     SS=malloc(nx*cf_num*sizeof(int));
//!     for(i<nx*cf_num) fscanf(fpt,"%d",&SS[i]);
//! }
//! ```
//! `cf_num=10` here is the reference C's own FIXED buffer-sizing constant
//! for the LOADER, unrelated to any individual composition's OWN `cf_num`
//! local (5 for `cf01`, 3 for `cf02`/`cf03`, 5 for `cf04`-`cf06`, 3 for
//! `cf07`/`cf08`, quoted per-fid below) -- EVERY `shift_data_<fid>.txt` (fid
//! 23-30) genuinely carries 10 rows of up to 100 values (verified: `wc -l`
//! on all 8 vendored files gives `10`, not `cf_num_actual`), and EVERY
//! `M_<fid>_D<dim>.txt` genuinely carries `10*dim*dim` values (verified:
//! `wc -w` on `M_23_D10.txt`/`M_23_D30.txt` gives `1000`/`9000`), and
//! `shuffle_data_{29,30}_D<dim>.txt` genuinely carries `10*dim` ints
//! (verified: `wc -w` gives `100`/`300`). Only the FIRST `cf_num_actual`
//! rows/blocks of each are ever read by that composition's own `cf0N` body
//! (`Os[i*nx]`/`Mr[i*nx*nx]`/`SS[i*nx]` for `i` in `0..cf_num_actual`) --
//! the remaining rows/blocks are genuinely dead weight the C loads but never
//! touches for that `fid`. [`data::composition_shift_blocks`] /
//! [`data::composition_rotation_blocks`] / [`data::composition_shuffle_blocks`]
//! (data.rs's own doc comments) take the CALLER's actual `cf_num` and parse
//! only that many rows/blocks, mirroring `cec2022::data`'s own
//! `composition_shift_blocks`/`composition_rotation_blocks` pattern exactly
//! (that suite's own hardcoded loader constant is `cf_num=12`, a DIFFERENT
//! number -- confirms this "loader constant != any individual composition's
//! own cf_num" shape is a recurring reference-C idiom across CEC suites, not
//! specific to either one).
//!
//! ### cf01-cf06 tables (base-function compositions) -- quoted from the C
//! (`cf01`..`cf06` bodies, `cf_num`/`delta`/`bias`/component-call-order/
//! post-eval-rescale-and-r_flag) AND CROSS-CHECKED against the report
//! (section "D. Composition Functions", `pdftotext -layout`)
//!
//! ```text
//! void cf01(...) { cf_num=5; delta={10,20,30,40,50}; bias={0,100,200,300,400};
//!     i=0; rosenbrock_func(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],1,r_flag); fit[i]=10000*fit[i]/1e+4;
//!     i=1; ellips_func(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],1,r_flag);     fit[i]=10000*fit[i]/1e+10;
//!     i=2; bent_cigar_func(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],1,r_flag); fit[i]=10000*fit[i]/1e+30;
//!     i=3; discus_func(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],1,r_flag);     fit[i]=10000*fit[i]/1e+10;
//!     i=4; ellips_func(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],1,0);          fit[i]=10000*fit[i]/1e+10;
//!     cf_cal(x,f,nx,Os,delta,bias,fit,cf_num); }
//! void cf02(...) { cf_num=3; delta={20,20,20}; bias={0,100,200};
//!     i=0; schwefel_func(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],1,0);
//!     i=1; rastrigin_func(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],1,r_flag);
//!     i=2; hgbat_func(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],1,r_flag);
//!     cf_cal(...); }
//! void cf03(...) { cf_num=3; delta={10,30,50}; bias={0,100,200};
//!     i=0; schwefel_func(...,r_flag);   fit[i]=1000*fit[i]/4e+3;
//!     i=1; rastrigin_func(...,r_flag);  fit[i]=1000*fit[i]/1e+3;
//!     i=2; ellips_func(...,r_flag);     fit[i]=1000*fit[i]/1e+10;
//!     cf_cal(...); }
//! void cf04(...) { cf_num=5; delta={10,10,10,10,10}; bias={0,100,200,300,400};
//!     i=0; schwefel_func(...,r_flag);    fit[i]=1000*fit[i]/4e+3;
//!     i=1; happycat_func(...,r_flag);    fit[i]=1000*fit[i]/1e+3;
//!     i=2; ellips_func(...,r_flag);      fit[i]=1000*fit[i]/1e+10;
//!     i=3; weierstrass_func(...,r_flag); fit[i]=1000*fit[i]/400;
//!     i=4; griewank_func(...,r_flag);    fit[i]=1000*fit[i]/100;
//!     cf_cal(...); }
//! void cf05(...) { cf_num=5; delta={10,10,10,20,20}; bias={0,100,200,300,400};
//!     i=0; hgbat_func(...,r_flag);       fit[i]=10000*fit[i]/1000;
//!     i=1; rastrigin_func(...,r_flag);   fit[i]=10000*fit[i]/1e+3;
//!     i=2; schwefel_func(...,r_flag);    fit[i]=10000*fit[i]/4e+3;
//!     i=3; weierstrass_func(...,r_flag); fit[i]=10000*fit[i]/400;
//!     i=4; ellips_func(...,r_flag);      fit[i]=10000*fit[i]/1e+10;
//!     cf_cal(...); }
//! void cf06(...) { cf_num=5; delta={10,20,30,40,50}; bias={0,100,200,300,400};
//!     i=0; grie_rosen_func(...,r_flag);  fit[i]=10000*fit[i]/4e+3;
//!     i=1; happycat_func(...,r_flag);    fit[i]=10000*fit[i]/1e+3;
//!     i=2; schwefel_func(...,r_flag);    fit[i]=10000*fit[i]/4e+3;
//!     i=3; escaffer6_func(...,r_flag);   fit[i]=10000*fit[i]/2e+7;
//!     i=4; ellips_func(...,r_flag);      fit[i]=10000*fit[i]/1e+10;
//!     cf_cal(...); }
//! ```
//! Since every `case 23..28` dispatch call passes the literal `r_flag=1`
//! (module doc's dispatch section above), every `r_flag`-parameter-driven
//! component above IS rotated; only components with a HARDCODED `0` (not the
//! `r_flag` parameter) are ever unrotated for a `cf0N` -- exactly TWO such
//! sites exist in the whole file: `cf01`'s `i=4` (second Ellipsoidal
//! component) and `cf02`'s `i=0` (Schwefel). [`Cec2014::composition_spec`]
//! transcribes all six tables below, `rotate` column reflecting this
//! exactly.
//!
//! | fid | `cf0N` | `cf_num` | `delta` | `bias` | components `(fn, rotate, rescale)` |
//! |---|---|---|---|---|---|
//! | 23 | `cf01` | 5 | `[10,20,30,40,50]` | `[0,100,200,300,400]` | `(Rosenbrock,true,1.0)`, `(Ellips,true,1e-6)`, `(BentCigar,true,1e-26)`, `(Discus,true,1e-6)`, `(Ellips,FALSE,1e-6)` |
//! | 24 | `cf02` | 3 | `[20,20,20]` | `[0,100,200]` | `(Schwefel,FALSE,1.0)`, `(Rastrigin,true,1.0)`, `(HGBat,true,1.0)` |
//! | 25 | `cf03` | 3 | `[10,30,50]` | `[0,100,200]` | `(Schwefel,true,0.25)`, `(Rastrigin,true,1.0)`, `(Ellips,true,1e-7)` |
//! | 26 | `cf04` | 5 | `[10,10,10,10,10]` | `[0,100,200,300,400]` | `(Schwefel,true,0.25)`, `(HappyCat,true,1.0)`, `(Ellips,true,1e-7)`, `(Weierstrass,true,2.5)`, `(Griewank,true,10.0)` |
//! | 27 | `cf05` | 5 | `[10,10,10,20,20]` | `[0,100,200,300,400]` | `(HGBat,true,10.0)`, `(Rastrigin,true,10.0)`, `(Schwefel,true,2.5)`, `(Weierstrass,true,25.0)`, `(Ellips,true,1e-6)` |
//! | 28 | `cf06` | 5 | `[10,20,30,40,50]` | `[0,100,200,300,400]` | `(GrieRosen,true,2.5)`, `(HappyCat,true,10.0)`, `(Schwefel,true,2.5)`, `(EScaffer6,true,5e-4)`, `(Ellips,true,1e-6)` |
//!
//! Every `delta`/`bias`/component-name/component-order value above was
//! CROSS-CHECKED against the report's own transcribed tables (below) and
//! matches EXACTLY -- the report's own `sigma`/`lambda` symbols correspond to
//! the C's `delta`/(rescale-factor implied by the `10000*fit/K` lines), and
//! every rescale factor above was independently derived from the C's own
//! division (e.g. `10000/1e4=1.0`, `10000/1e10=1e-6`, `10000/1e30=1e-26`) and
//! confirmed to match the report's printed `lambda` array digit-for-digit
//! for all six functions.
//!
//! Report (PDF `pdftotext -layout`, section "D. Composition Functions",
//! transcribed verbatim):
//! ```text
//! 23) Composition Function 1: N=5, sigma=[10,20,30,40,50], lambda=[1,1e-6,1e-26,1e-6,1e-6], bias=[0,100,200,300,400]
//!     g1: Rotated Rosenbrock's F4'  g2: High Conditioned Elliptic F1'  g3: Rotated Bent Cigar F2'
//!     g4: Rotated Discus F3'        g5: High Conditioned Elliptic F1'
//! 24) Composition Function 2: N=3, sigma=[20,20,20], lambda=[1,1,1], bias=[0,100,200]
//!     g1: Schwefel's F10'  g2: Rotated Rastrigin's F9'  g3: Rotated HGBat F14'
//! 25) Composition Function 3: N=3, sigma=[10,30,50], lambda=[0.25,1,1e-7], bias=[0,100,200]
//!     g1: Rotated Schwefel's F11'  g2: Rotated Rastrigin's F9'  g3: Rotated High Conditioned Elliptic F1'
//! 26) Composition Function 4: N=5, sigma=[10,10,10,10,10], lambda=[0.25,1,1e-7,2.5,10], bias=[0,100,200,300,400]
//!     g1: Rotated Schwefel's F11'  g2: Rotated HappyCat F13'  g3: Rotated High Conditioned Elliptic F1'
//!     g4: Rotated Weierstrass F6'  g5: Rotated Griewank's F7'
//! 27) Composition Function 5: N=5, sigma=[10,10,10,20,20], lambda=[10,10,2.5,25,1e-6], bias=[0,100,200,300,400]
//!     g1: Rotated HGBat F14'  g2: Rotated Rastrigin's F9'  g3: Rotated Schwefel's F11'
//!     g4: Rotated Weierstrass F6'  g5: Rotated High Conditioned Elliptic F1'
//! 28) Composition Function 6: N=5, sigma=[10,20,30,40,50], lambda=[2.5,10,2.5,5e-4,1e-6], bias=[0,100,200,300,400]
//!     g1: Rotated Expanded Griewank's plus Rosenbrock's F15'  g2: Rotated HappyCat F13'
//!     g3: Rotated Schwefel's F11'  g4: Rotated Expanded Scaffer's F6 F16'  g5: Rotated High Conditioned Elliptic F1'
//! ```
//!
//! ### VERIFIED DISCREPANCY: CF1's g2/g5 (both "High Conditioned Elliptic
//! F1'" in the report's own printed text, IDENTICAL naming for both) are NOT
//! rotated identically -- only g2 is; g5 is hardcoded unrotated (code-over-
//! report ruling)
//!
//! The report prints g2 and g5 with the EXACT SAME name, "High Conditioned
//! Elliptic Function F1'" (neither carries a "Rotated" prefix the way g1/g3/
//! g4 do) -- this reads as ambiguous/under-specified prose, not a clear
//! rotate/no-rotate distinction between the two. The C, however, is
//! unambiguous and treats them DIFFERENTLY: `cf01`'s `i=1` (g2) call passes
//! the `r_flag` PARAMETER (`=1` from the dispatch, i.e. rotated), while
//! `i=4` (g5) passes a HARDCODED literal `0` (never rotated, regardless of
//! `r_flag`) -- quoted above, `ellips_func(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],1,0)`
//! for `i=4` vs. `...,1,r_flag)` for `i=1`. Per the milestone's STANDING
//! RULING (code-over-report): [`Cec2014::composition_spec`]'s fid-23 row
//! marks g2 `rotate=true`, g5 `rotate=FALSE`, matching the C's actual,
//! executable behavior -- `composition_spec_matches_the_c_sources_constant_tables`
//! documents this. No other CF1-CF6 component has this ambiguity: every
//! other report row's "Rotated"/no-prefix naming matches its C `r_flag`
//! choice exactly (CF2's `g1: Schwefel's F10'`, no "Rotated" prefix, matches
//! `cf02`'s hardcoded `i=0` unrotated call -- the ONE other hardcoded-`0`
//! site in the whole file -- and every remaining component in CF2-CF6 is
//! named "Rotated ..." AND passes the `r_flag` parameter, verified by
//! reading all six bodies directly, not merely the two flagged here).
//!
//! ### cf07/cf08 -- hybrid-in-composition (the report's own note, quoted
//! above the per-fid list: "In CEC'14, the hybrid functions are also used as
//! the basic functions for composition functions (Composition Function 7
//! and Composition Function 8)") -- quoted from the C, CROSS-CHECKED against
//! the report
//!
//! ```text
//! void cf07(double *x,double *f,int nx,double *Os,double *Mr,int *SS,int r_flag) {
//!     cf_num=3; delta={10,30,50}; bias={0,100,200};
//!     i=0; hf01(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],&SS[i*nx],1,r_flag);
//!     i=1; hf02(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],&SS[i*nx],1,r_flag);
//!     i=2; hf03(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],&SS[i*nx],1,r_flag);
//!     cf_cal(x,f,nx,Os,delta,bias,fit,cf_num); }
//! void cf08(double *x,double *f,int nx,double *Os,double *Mr,int *SS,int r_flag) {
//!     cf_num=3; delta={10,30,50}; bias={0,100,200};
//!     i=0; hf04(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],&SS[i*nx],1,r_flag);
//!     i=1; hf05(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],&SS[i*nx],1,r_flag);
//!     i=2; hf06(x,&fit[i],nx,&Os[i*nx],&Mr[i*nx*nx],&SS[i*nx],1,r_flag);
//!     cf_cal(x,f,nx,Os,delta,bias,fit,cf_num); }
//! ```
//! Report (transcribed verbatim, matches exactly, no discrepancy):
//! ```text
//! 29) Composition Function 7: N=3, sigma=[10,30,50], lambda=[1,1,1], bias=[0,100,200]
//!     g1: Hybrid Function 1 F17'  g2: Hybrid Function 2 F18'  g3: Hybrid Function 3 F19'
//! 30) Composition Function 8: N=3, sigma=[10,30,50], lambda=[1,1,1], bias=[0,100,200]
//!     g1: Hybrid Function 4 F20'  g2: Hybrid Function 5 F21'  g3: Hybrid Function 6 F22'
//! ```
//! `lambda=[1,1,1]` for both matches the C exactly: NEITHER `cf07` NOR
//! `cf08` post-scales any `fit[i]` (unlike `cf01`-`cf06`'s `fit[i]=K*fit[i]/D`
//! lines) -- `cf_cal` adds `bias[i]` only. `Os[i*nx]`/`Mr[i*nx*nx]`/
//! `SS[i*nx]` give EACH hybrid component its OWN FULL shift+rotate+shuffle
//! (a DIFFERENT `(o,M,S)` triple per component `i`, unlike fid 17-22's own
//! standalone dispatch, which uses ONE shared `(o,M,S)` triple for the
//! WHOLE hybrid) -- [`Cec2014::composition_hybrid_fitness`] threads this
//! through explicitly: for component `idx`, `z = shift_scale_rotate(xs,
//! comp_shift[idx], comp_rotation[idx], 1.0, true)`, `y[i] =
//! z[comp_shuffle[idx][i]]`, then [`Cec2014::hybrid_fitness`] (fid 17-22's
//! own T3 helper, REUSED unmodified here -- same `hf01`..`hf06` component
//! tables, same segmentation) evaluated on that component's own `y`, at the
//! hybrid fid matching this composition slot (`cf07`: hf01/hf02/hf03 ->
//! `hybrid_fitness(17,..)`/`(18,..)`/`(19,..)`; `cf08`: hf04/hf05/hf06 ->
//! `(20,..)`/`(21,..)`/`(22,..)`, since fid 17-22's own dispatch already
//! established the clean `hf0N` <-> `fid` mapping this module doc's earlier
//! T3 section documents).
//!
//! ### `cf_cal` weight formula -- BIT-FOR-BIT IDENTICAL to the shared
//! `crate::cec_basics::composition_weights` (cec2022's own extraction, M3-6
//! T1) -- VERIFIED by direct line comparison, no suite-specific divergence
//!
//! ```text
//! void cf_cal(double *x,double *f,int nx,double *Os,double *delta,double *bias,double *fit,int cf_num) {
//!     for(i<cf_num) { fit[i]+=bias[i]; w[i]=sum_j((x[j]-Os[i*nx+j])^2);
//!         w[i] = (w[i]!=0) ? pow(1.0/w[i],0.5)*exp(-w[i]/2.0/nx/pow(delta[i],2.0)) : INF;
//!         if (w[i]>w_max) w_max=w[i]; }
//!     w_sum=sum(w); if (w_max==0) { w[i]=1 for all i; w_sum=cf_num; }
//!     f[0]=sum_i(w[i]/w_sum*fit[i]); }
//! ```
//! (`#define INF 1.0e99`, confirmed by grepping the vendored source directly
//! -- the SAME finite sentinel `crate::cec_basics::composition_weights`'s own
//! doc comment already documents for `cec2022`.) Byte-for-byte identical
//! structure to `cec2022`'s own `cf_cal` (module doc's own T7 section, this
//! suite's own copy read and diffed directly during this task, zero
//! difference found) -- so [`Cec2014::composition_fitness`] and
//! [`Cec2014::composition_hybrid_fitness`] both call the ALREADY-SHARED
//! [`crate::cec_basics::composition_weights`] directly, no new weight-formula
//! code this task (unlike, e.g., a hypothetical suite whose `cf_cal` used a
//! different sentinel or normalization -- NOT the case here, confirmed by
//! direct comparison, so no suite-specific weight function was written; the
//! brief's suggested "weight-formula unit test if machinery diverged" does
//! not apply -- this module's own tests instead exercise
//! `crate::cec_basics::composition_weights` indirectly through the `x=o_1`
//! pin and random-point cross-checks below, which would fail immediately had
//! any formula divergence existed).
//!
//! ### `x = o_1` pin (Step 2) for fid 23-30: why `F_i(comp_shift[0]) == F_i*`
//! EXACTLY -- SAME "INF-sentinel absorption" argument `cec2022/mod.rs`'s own
//! T7 section already makes for its own fid 9-12 composition pin (re-derived
//! here for this suite's own tables, VERIFIED against the compiled C
//! reference for every fid 23-30 x every dim in {10,30})
//!
//! At `x = comp_shift[0]` (component 1's OWN shift row): component 1's raw
//! squared distance `D_0 = ||x - comp_shift[0]||^2 = 0` EXACTLY (identical
//! `f64` operands subtract to exact `0.0`, same argument every earlier pin
//! in this module doc relies on), so `cf_cal`'s `w_0 = INF = 1e99` -- and
//! `1e99`'s ULP (`~2.2e83`) overwhelmingly dwarfs any realistic OTHER
//! component's finite `w_j`, so `w_sum` (plain `f64` addition) rounds to
//! EXACTLY `1e99` regardless of the other `w_j` (absorption), making
//! `w_0/w_sum` round to EXACTLY `1.0`. Component 1's own shift-rotated
//! argument at `x=comp_shift[0]` is the all-zero vector (`(x-o_1)*sr=0`
//! regardless of `sr`/rotation, module doc's earlier pin sections), so
//! component 1's OWN base-function evaluation is EXACTLY `0` (every CF1-CF6
//! first component -- Rosenbrock/Schwefel x2/HGBat.../GrieRosen -- is one of
//! the fourteen base functions this module doc's fid-1-16/17-22 pin sections
//! already prove `=0` at an all-zero argument of ANY length; for cf07/cf08,
//! component 1 is `hybrid_fitness(17,..)`/`hybrid_fitness(20,..)` on an
//! all-zero `y`, which the fid-17-22 pin section above ALREADY proves is
//! exactly `0`), and `bias[0]==0` for EVERY ONE of the eight tables above
//! (`{0,100,200,...}`) -- so component 1's raw contribution to `cf_cal`'s sum
//! is `0*1.0=0` exactly. The OTHER components' contributions (`w_j/w_sum *
//! fit_j` for `j!=0`) are each a picoscopic-but-not-necessarily-bit-exact-
//! zero double (SAME caveat `cec2022/mod.rs`'s own T7 pin note makes) -- NOT
//! provably bit-exact `0.0` on their own, but VERIFIED EMPIRICALLY (compiled
//! C reference, `%.20f`-format, all sixteen `(fid,dim)` pairs in `{23..=30}
//! x {10,30}` print the exact integer `100*fid`, zero fractional residue,
//! this task's report has the transcript) that the residual is far below
//! `F_i*`'s (order `1e3`) ULP (`~1e-13`), so `Cec2014::eval_one`'s final `+=
//! F_i*` (the SAME `value + self.f_star()` line used for every fid) absorbs
//! it completely -- `F_i(comp_shift[0]) == F_i*` EXACTLY, confirmed by this
//! module's own `fid_23_to_30_x_eq_o1_is_exactly_f_star_dim10_and_dim30`
//! test.

mod data;

use sezgi_core::problem::Problem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

/// One hybrid function's (fid 17-22) sub-component (module doc's T3
/// section): pairs the base function with the `sh_rate` constant the C
/// reference's OWN per-component `sr_func` call uses (`s_flag=0,r_flag=0`
/// branch -- pure scale, no shift, no rotate; module doc's "Per-component
/// evaluation" section). Unlike `cec2022::HybridComponent`, no buggy variant
/// is needed here -- module doc's T3 section verifies every component
/// correctly reads its own pointer-offset segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HybridComponent {
    Ellips,
    BentCigar,
    Discus,
    Rosenbrock,
    Ackley,
    Weierstrass,
    Griewank,
    Rastrigin,
    Schwefel,
    Katsuura,
    HappyCat,
    HGBat,
    GrieRosen,
    EScaffer6,
}

impl HybridComponent {
    /// This component's own inner scale constant -- module doc's T3
    /// "Per-component evaluation" section, the SAME constant this base
    /// function uses in its fid-1-16 standalone dispatch (module doc's
    /// "Basic-function reuse" table above).
    fn sh_rate(self) -> f64 {
        match self {
            Self::Ellips => 1.0,
            Self::BentCigar => 1.0,
            Self::Discus => 1.0,
            Self::Rosenbrock => 2.048 / 100.0,
            Self::Ackley => 1.0,
            Self::Weierstrass => 0.5 / 100.0,
            Self::Griewank => 600.0 / 100.0,
            Self::Rastrigin => 5.12 / 100.0,
            Self::Schwefel => 1000.0 / 100.0,
            Self::Katsuura => 5.0 / 100.0,
            Self::HappyCat => 5.0 / 100.0,
            Self::HGBat => 5.0 / 100.0,
            Self::GrieRosen => 5.0 / 100.0,
            Self::EScaffer6 => 1.0,
        }
    }

    /// Evaluate this component's base function on an ALREADY-scaled segment
    /// (caller applies `sh_rate`, module doc's T3 section -- same pattern
    /// `cec2022::HybridComponent::eval` uses).
    fn eval(self, seg: &[f64]) -> f64 {
        use crate::cec_basics as cb;
        match self {
            Self::Ellips => cb::ellips_base(seg),
            Self::BentCigar => cb::bent_cigar_base(seg),
            Self::Discus => cb::discus_base(seg),
            Self::Rosenbrock => {
                // `rosenbrock_func`'s own "+1, shift to origin" convention
                // (module doc's fid-4 note above) -- reuses `f2_base`, which
                // expects the array already `+1`'d.
                let shifted: Vec<f64> = seg.iter().map(|&zi| zi + 1.0).collect();
                cb::f2_base(&shifted)
            }
            Self::Ackley => cb::ackley_base(seg),
            Self::Weierstrass => cb::weierstrass_base(seg),
            Self::Griewank => cb::griewank_base(seg),
            Self::Rastrigin => cb::f4_base(seg),
            Self::Schwefel => cb::schwefel_base(seg),
            Self::Katsuura => cb::katsuura_base(seg),
            Self::HappyCat => cb::happycat_base(seg),
            Self::HGBat => cb::hgbat_base(seg),
            // `grie_rosen_base` handles its own `+1` internally (module
            // doc's fid-15 note above).
            Self::GrieRosen => cb::grie_rosen_base(seg),
            Self::EScaffer6 => cb::escaffer6_base(seg),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Cec2014Error {
    #[error(
        "fid must be in 1..=30 (the full CEC 2014 suite: unimodal + simple-multimodal \
         functions 1-16, hybrid functions 17-22, composition functions 23-30), got {0}"
    )]
    UnknownFid(u32),
    #[error(
        "dim must be one of {{10,30}} (this crate vendors only these two dimensions of the \
         CEC 2014 input_data set -- the official suite also supports 2,20,50,100), got {0}"
    )]
    BadDim(usize),
}

/// One instance of a CEC 2014 basic (fid 1-3), simple-multimodal (fid
/// 4-16), hybrid (fid 17-22), or composition (fid 23-30) function: embedded
/// official shift vector `o` (`comp_shift[0]` for fid 23-30) and rotation
/// matrix `M` for the requested `(fid, dim)`, plus the pinned `F_i* =
/// 100*fid` bias (module doc, section 1.2's table). See the module doc for
/// each fid's exact shift/scale/rotate pipeline, including the verified F16
/// discrepancy between the printed report and the vendored reference C code
/// this module actually follows.
pub struct Cec2014 {
    fid: u32,
    dim: usize,
    o: Vec<f64>,
    m: Vec<Vec<f64>>,
    /// fid 17-22 only (module doc's T3 section): 0-based shuffle permutation
    /// applied to the shift-rotated vector before segmenting into hybrid
    /// components. Empty for fid 1-16 and fid 23-30 (fid 29-30 use
    /// [`Self::comp_shuffle`] instead, one permutation PER component).
    shuffle: Vec<usize>,
    /// fid 23-30 only (module doc's T4 section): each component's OWN shift
    /// row (`comp_shift[idx]`, `cf_num` rows total -- 3 or 5). Empty for fid
    /// 1-22. `o` mirrors `comp_shift[0]` for API uniformity with the `x = o`
    /// pin tests (same `cec2022::Cec2022`'s own fid-9-12 struct doc
    /// decision, quoted there).
    comp_shift: Vec<Vec<f64>>,
    /// fid 23-30 only: each component's OWN `dim`x`dim` rotation matrix,
    /// parallel to [`Self::comp_shift`]. Empty for fid 1-22.
    comp_rotation: Vec<Vec<Vec<f64>>>,
    /// fid 29-30 only (module doc's T4 section, cf07/cf08's hybrid-in-
    /// composition wiring): each of the 3 hybrid components' OWN 0-based
    /// shuffle permutation, parallel to [`Self::comp_shift`]. Empty for
    /// every other fid (including fid 23-28, whose components are plain
    /// base functions with no shuffle at all).
    comp_shuffle: Vec<Vec<usize>>,
    space: SearchSpace,
}

impl Cec2014 {
    /// `fid` in `1..=30` (the full suite; `Cec2014Error::UnknownFid`
    /// otherwise, including `0` and anything `>30`), `dim` in `{10,30}`
    /// (module doc's "Vendoring scope" section: the two dims this crate
    /// vendored; `Cec2014Error::BadDim` otherwise).
    pub fn new(fid: u32, dim: usize) -> Result<Cec2014, Cec2014Error> {
        if !(1..=30).contains(&fid) {
            return Err(Cec2014Error::UnknownFid(fid));
        }
        if !matches!(dim, 10 | 30) {
            return Err(Cec2014Error::BadDim(dim));
        }
        let (o, m, shuffle, comp_shift, comp_rotation, comp_shuffle) = if (23..=30).contains(&fid) {
            let cf_num = Self::composition_cf_num(fid);
            let comp_shift = data::composition_shift_blocks(fid, dim, cf_num);
            let comp_rotation = data::composition_rotation_blocks(fid, dim, cf_num);
            let comp_shuffle = if (29..=30).contains(&fid) {
                data::composition_shuffle_blocks(fid, dim, cf_num)
            } else {
                Vec::new()
            };
            // `o` mirrors component 1's own shift (module doc's struct doc,
            // same `cec2022::Cec2022` decision for its own fid 9-12) so the
            // `x = o` pin tests can stay uniform across the whole 1..=30
            // range.
            let o = comp_shift[0].clone();
            (o, Vec::new(), Vec::new(), comp_shift, comp_rotation, comp_shuffle)
        } else {
            let o = data::shift_vector(fid, dim);
            let m = data::rotation_matrix(fid, dim);
            let shuffle =
                if (17..=22).contains(&fid) { data::shuffle_indices(fid, dim) } else { Vec::new() };
            (o, m, shuffle, Vec::new(), Vec::new(), Vec::new())
        };
        let space = SearchSpace::new(vec![Block::Float { lo: -100.0, hi: 100.0, n: dim }])
            .expect("CEC 2014 bounds (-100 < 100) are always valid");
        Ok(Self { fid, dim, o, m, shuffle, comp_shift, comp_rotation, comp_shuffle, space })
    }

    /// This composition fid's own `cf_num` (module doc's T4 section,
    /// transcribed from `cf01`..`cf08`'s own local `cf_num` -- NOT the
    /// loader's unrelated hardcoded `cf_num=10`, module doc's dedicated note
    /// on that distinction). `fid` must be `23..=30` (caller's
    /// responsibility).
    fn composition_cf_num(fid: u32) -> usize {
        match fid {
            23 | 26 | 27 | 28 => 5,
            24 | 25 | 29 | 30 => 3,
            other => unreachable!("composition_cf_num called with unsupported fid {other}"),
        }
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

    /// `fid`'s (`17..=22`) proportions `p` and component list, in call order
    /// (module doc's T3 "Hybrid composition tables" section, transcribed
    /// verbatim from the cross-checked C/report table there -- no
    /// discrepancy found for any of the six, unlike `cec2022`'s fid 7).
    fn hybrid_spec(fid: u32) -> (&'static [f64], &'static [HybridComponent]) {
        use HybridComponent::*;
        match fid {
            17 => (&[0.3, 0.3, 0.4], &[Schwefel, Rastrigin, Ellips]),
            18 => (&[0.3, 0.3, 0.4], &[BentCigar, HGBat, Rastrigin]),
            19 => (&[0.2, 0.2, 0.3, 0.3], &[Griewank, Weierstrass, Rosenbrock, EScaffer6]),
            20 => (&[0.2, 0.2, 0.3, 0.3], &[HGBat, Discus, GrieRosen, Rastrigin]),
            21 => (&[0.1, 0.2, 0.2, 0.2, 0.3], &[EScaffer6, HGBat, Rosenbrock, Schwefel, Ellips]),
            22 => (&[0.1, 0.2, 0.2, 0.2, 0.3], &[Katsuura, HappyCat, GrieRosen, Schwefel, Ackley]),
            other => unreachable!("hybrid_spec called with unsupported fid {other}"),
        }
    }

    /// One hybrid function's component-sum (BEFORE `F_i*`), given the
    /// already shift-rotate-shuffled `y` (module doc's T3 section: `y[i] =
    /// z[S[i]-1]`, `z` from [`Self::shift_scale_rotate`]). A separate method
    /// (not inlined into [`Self::eval_one`]) so the segmentation/shuffle
    /// logic can be probed directly on a hand-built `y`, independent of the
    /// shift/rotate/embedded-data plumbing -- mirrors
    /// `cec2022::Cec2022::hybrid_fitness`'s own separation.
    // sezgi decision (M3-6 T3): calls `crate::cec_basics::segment_sizes`/
    // `segment_starts` directly instead of adding `Cec2014::segment_sizes`/
    // `segment_starts` thin-wrapper methods the way `cec2022::Cec2022` keeps
    // (`cec2022/mod.rs`'s own "M3-6 T1" decision note on those methods: kept
    // there ONLY because pre-existing test call sites referenced
    // `Cec2022::segment_sizes` directly before the T1 extraction). No such
    // call site exists in this fresh `cec2014` addition, so the wrapper
    // would add a layer with no consumer -- calling the shared
    // `crate::cec_basics` functions directly here is the simpler choice,
    // same underlying arithmetic either way.
    fn hybrid_fitness(fid: u32, dim: usize, y: &[f64]) -> f64 {
        let (gp, comps) = Self::hybrid_spec(fid);
        let sizes = crate::cec_basics::segment_sizes(gp, dim);
        let starts = crate::cec_basics::segment_starts(&sizes);
        comps
            .iter()
            .enumerate()
            .map(|(idx, &comp)| {
                let start = starts[idx];
                let len = sizes[idx];
                let seg: Vec<f64> =
                    y[start..start + len].iter().map(|&yi| yi * comp.sh_rate()).collect();
                comp.eval(&seg)
            })
            .sum()
    }

    /// One composition function's (fid 23-28, [`Self::composition_spec`])
    /// `(delta, bias, components)` triple -- `components[i]` is `(base
    /// function, this component's own rotate flag, its post-eval rescale)`.
    /// Named to keep the signature readable (clippy's `type_complexity`,
    /// same shape `cec2022::Cec2022`'s own `CompositionSpec` alias uses).
    #[allow(clippy::type_complexity)]
    fn composition_spec(fid: u32) -> (&'static [f64], &'static [f64], &'static [(HybridComponent, bool, f64)]) {
        use HybridComponent::*;
        match fid {
            23 => (
                &[10.0, 20.0, 30.0, 40.0, 50.0],
                &[0.0, 100.0, 200.0, 300.0, 400.0],
                &[
                    (Rosenbrock, true, 1.0),
                    (Ellips, true, 1.0e-6),
                    (BentCigar, true, 1.0e-26),
                    (Discus, true, 1.0e-6),
                    // Module doc's VERIFIED DISCREPANCY note: the C
                    // hardcodes `r_flag=0` here (NOT the `r_flag` parameter)
                    // -- this second Elliptic component is never rotated,
                    // unlike the report's identically-named g2/g5 prose.
                    (Ellips, false, 1.0e-6),
                ],
            ),
            24 => (
                &[20.0, 20.0, 20.0],
                &[0.0, 100.0, 200.0],
                // The C's OTHER hardcoded-unrotated site (module doc's
                // discrepancy note): Schwefel here matches the report's own
                // "Schwefel's F10'" (no "Rotated" prefix) exactly, no
                // ambiguity for this one.
                &[(Schwefel, false, 1.0), (Rastrigin, true, 1.0), (HGBat, true, 1.0)],
            ),
            25 => (
                &[10.0, 30.0, 50.0],
                &[0.0, 100.0, 200.0],
                &[(Schwefel, true, 0.25), (Rastrigin, true, 1.0), (Ellips, true, 1.0e-7)],
            ),
            26 => (
                &[10.0, 10.0, 10.0, 10.0, 10.0],
                &[0.0, 100.0, 200.0, 300.0, 400.0],
                &[
                    (Schwefel, true, 0.25),
                    (HappyCat, true, 1.0),
                    (Ellips, true, 1.0e-7),
                    (Weierstrass, true, 2.5),
                    (Griewank, true, 10.0),
                ],
            ),
            27 => (
                &[10.0, 10.0, 10.0, 20.0, 20.0],
                &[0.0, 100.0, 200.0, 300.0, 400.0],
                &[
                    (HGBat, true, 10.0),
                    (Rastrigin, true, 10.0),
                    (Schwefel, true, 2.5),
                    (Weierstrass, true, 25.0),
                    (Ellips, true, 1.0e-6),
                ],
            ),
            28 => (
                &[10.0, 20.0, 30.0, 40.0, 50.0],
                &[0.0, 100.0, 200.0, 300.0, 400.0],
                &[
                    (GrieRosen, true, 2.5),
                    (HappyCat, true, 10.0),
                    (Schwefel, true, 2.5),
                    (EScaffer6, true, 5.0e-4),
                    (Ellips, true, 1.0e-6),
                ],
            ),
            other => unreachable!("composition_spec called with unsupported fid {other}"),
        }
    }

    /// One composition function's total (BEFORE `F_i*`), fid 23-28: each
    /// component's own FULL shift-rotate pipeline (module doc's T4 section:
    /// unlike the hybrids' single shared shift+rotate, EVERY composition
    /// component gets its OWN `(comp_shift[idx], comp_rotation[idx])`) +
    /// base function + its post-eval rescale + its `bias`, weighted by
    /// [`crate::cec_basics::composition_weights`] and summed (`cf_cal`'s own
    /// final loop, module doc's T4 `cf_cal` quote).
    fn composition_fitness(&self, xs: &[f64]) -> f64 {
        let (delta, bias, comps) = Self::composition_spec(self.fid);
        let fit: Vec<f64> = comps
            .iter()
            .enumerate()
            .map(|(idx, &(comp, rotate, rescale))| {
                let z = crate::cec_basics::shift_scale_rotate(
                    xs,
                    &self.comp_shift[idx],
                    &self.comp_rotation[idx],
                    comp.sh_rate(),
                    rotate,
                );
                comp.eval(&z) * rescale + bias[idx]
            })
            .collect();
        let w = crate::cec_basics::composition_weights(xs, &self.comp_shift, delta);
        fit.iter().zip(&w).map(|(&f, &wi)| f * wi).sum()
    }

    /// `cf07`/`cf08`'s `(delta, bias, hybrid fids)` triple (module doc's T4
    /// section): `hybrid_fids[i]` is the fid `Self::hybrid_fitness` should
    /// evaluate component `i` at (`17,18,19` for `cf07`/fid29; `20,21,22`
    /// for `cf08`/fid30, the SAME clean `hf0N` <-> `fid` mapping T3's own
    /// dispatch section already establishes).
    fn composition_hybrid_spec(fid: u32) -> (&'static [f64], &'static [f64], &'static [u32]) {
        match fid {
            29 => (&[10.0, 30.0, 50.0], &[0.0, 100.0, 200.0], &[17, 18, 19]),
            30 => (&[10.0, 30.0, 50.0], &[0.0, 100.0, 200.0], &[20, 21, 22]),
            other => unreachable!("composition_hybrid_spec called with unsupported fid {other}"),
        }
    }

    /// One composition function's total (BEFORE `F_i*`), fid 29-30
    /// (`cf07`/`cf08`, module doc's T4 "hybrid-in-composition" section):
    /// each component gets its OWN full shift+rotate+shuffle
    /// (`comp_shift[idx]`/`comp_rotation[idx]`/`comp_shuffle[idx]`, ALL
    /// THREE per-component, unlike fid 17-22's single shared triple), then
    /// [`Self::hybrid_fitness`] (T3's helper, reused unmodified) evaluates
    /// that component's own hybrid at the matching hybrid fid. `cf07`/`cf08`
    /// apply NO post-eval rescale (module doc: neither one has a `fit[i]=
    /// K*fit[i]/D` line, unlike `cf01`-`cf06`) -- only `bias[idx]`, then the
    /// SAME shared [`crate::cec_basics::composition_weights`] weighting.
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
                Self::hybrid_fitness(hfid, self.dim, &y) + bias[idx]
            })
            .collect();
        let w = crate::cec_basics::composition_weights(xs, &self.comp_shift, delta);
        fit.iter().zip(&w).map(|(&f, &wi)| f * wi).sum()
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
            17..=22 => {
                // Full shift+rotate ONCE (module doc's T3 section: the outer
                // sr_func is always sh_rate=1.0, shared by every component),
                // then shuffle, then segment + per-component scale-only
                // evaluation (module doc's T3 section).
                let z = self.shift_scale_rotate(xs, 1.0, true);
                let y: Vec<f64> = (0..self.dim).map(|i| z[self.shuffle[i]]).collect();
                Self::hybrid_fitness(self.fid, self.dim, &y)
            }
            23..=28 => self.composition_fitness(xs),
            29 | 30 => self.composition_hybrid_fitness(xs),
            other => unreachable!("Cec2014::new rejects fid outside 1..=30, got {other}"),
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
    fn fid_0_31_and_100_are_unknown() {
        // Renamed from `fid_0_23_and_31_are_unknown` (T3's own precedent:
        // that test itself was renamed from `fid_0_17_and_31_are_unknown`)
        // now that fid 23-30 (composition) construct successfully -- this
        // task's own boundary check.
        for fid in [0u32, 31, 100] {
            assert!(
                matches!(Cec2014::new(fid, 10), Err(Cec2014Error::UnknownFid(f)) if f == fid),
                "fid={fid}"
            );
        }
    }

    #[test]
    fn fid_1_to_30_all_construct_at_dim_10_and_30() {
        for fid in 1u32..=30 {
            for &dim in &[10usize, 30] {
                assert!(Cec2014::new(fid, dim).is_ok(), "fid={fid} dim={dim}");
            }
        }
    }

    #[test]
    fn unsupported_dims_are_bad_dim_for_every_fid() {
        for fid in 1u32..=30 {
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
        for fid in 1u32..=30 {
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
        // formula `Cec2014::f_star` actually implements) -- extended this
        // task to include fid 23-30's own `F*` column (module doc's T4
        // dispatch section quote: `f[i]+=2300.0` .. `f[i]+=3000.0`).
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
            (17, 1700.0),
            (18, 1800.0),
            (19, 1900.0),
            (20, 2000.0),
            (21, 2100.0),
            (22, 2200.0),
            (23, 2300.0),
            (24, 2400.0),
            (25, 2500.0),
            (26, 2600.0),
            (27, 2700.0),
            (28, 2800.0),
            (29, 2900.0),
            (30, 3000.0),
        ];
        for (fid, fstar) in expect {
            assert_eq!(Cec2014::new(fid, 10).unwrap().f_star(), fstar, "fid={fid}");
        }
    }

    // ---- x = o pin: F_i(o_i) == F_i* exactly (module doc's derivation) ----

    #[test]
    fn fid_1_to_22_x_eq_o_is_exactly_f_star_dim10_and_dim30() {
        for fid in 1u32..=22 {
            for &dim in &[10usize, 30] {
                let p = Cec2014::new(fid, dim).unwrap();
                let out = p.evaluate_batch(&[g(p.o.clone())])[0];
                assert_eq!(out, p.f_star(), "fid={fid} dim={dim}");
            }
        }
    }

    // ---- T4: x = o_1 pin for fid 23-30 (module doc's "INF-sentinel
    // absorption" derivation): `p.o` mirrors `comp_shift[0]` for fid 23-30
    // (struct doc), so this is the SAME shape as the fid 1-22 pin test
    // above, just a different `o` source under the hood. ----

    #[test]
    fn fid_23_to_30_x_eq_o1_is_exactly_f_star_dim10_and_dim30() {
        for fid in 23u32..=30 {
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
    fn constructed_o_and_m_have_the_right_shape_for_every_fid_1_to_22() {
        for fid in 1u32..=22 {
            for &dim in &[10usize, 30] {
                let p = Cec2014::new(fid, dim).unwrap();
                assert_eq!(p.o.len(), dim, "fid={fid} dim={dim}");
                assert_eq!(p.m.len(), dim, "fid={fid} dim={dim}");
                assert!(p.m.iter().all(|row| row.len() == dim), "fid={fid} dim={dim}");
            }
        }
    }

    // ---- T4: composition shift/rotation/shuffle multi-row data-integrity
    // asserts (brief's explicit requirement: "composition shift-data
    // multi-row integrity asserts") ----

    #[test]
    fn composition_comp_shift_and_rotation_have_cf_num_entries_for_every_fid() {
        for fid in 23u32..=30 {
            let n = Cec2014::composition_cf_num(fid);
            for &dim in &[10usize, 30] {
                let p = Cec2014::new(fid, dim).unwrap();
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
                // `o` mirrors comp_shift[0] exactly (struct doc, this task's
                // `new()` decision) -- not just "same length", the SAME
                // values.
                assert_eq!(p.o, p.comp_shift[0], "fid={fid} dim={dim}: o must equal comp_shift[0]");
                // `m` (the single-matrix field, fid 1-22's own) stays empty
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
        for fid in 23u32..=30 {
            let p = Cec2014::new(fid, 10).unwrap();
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

    // ---- constant-table spot check (module doc's T4 cf01-cf06 table):
    // confirms `composition_spec`'s transcription, INCLUDING the verified
    // CF1 g2/g5 rotate divergence (module doc's VERIFIED DISCREPANCY note)
    // ----

    #[test]
    fn composition_spec_matches_the_c_sources_constant_tables() {
        use HybridComponent::*;
        let (delta23, bias23, comps23) = Cec2014::composition_spec(23);
        assert_eq!(delta23, &[10.0, 20.0, 30.0, 40.0, 50.0]);
        assert_eq!(bias23, &[0.0, 100.0, 200.0, 300.0, 400.0]);
        assert_eq!(
            comps23,
            &[
                (Rosenbrock, true, 1.0),
                (Ellips, true, 1.0e-6),
                (BentCigar, true, 1.0e-26),
                (Discus, true, 1.0e-6),
                // The verified discrepancy: g5 (second Ellipsoidal) is
                // hardcoded UNROTATED in the C, unlike g2 (also
                // Ellipsoidal, same report name, but rotated).
                (Ellips, false, 1.0e-6),
            ]
        );

        let (delta24, bias24, comps24) = Cec2014::composition_spec(24);
        assert_eq!(delta24, &[20.0, 20.0, 20.0]);
        assert_eq!(bias24, &[0.0, 100.0, 200.0]);
        assert_eq!(comps24, &[(Schwefel, false, 1.0), (Rastrigin, true, 1.0), (HGBat, true, 1.0)]);

        let (delta28, bias28, comps28) = Cec2014::composition_spec(28);
        assert_eq!(delta28, &[10.0, 20.0, 30.0, 40.0, 50.0]);
        assert_eq!(bias28, &[0.0, 100.0, 200.0, 300.0, 400.0]);
        assert_eq!(
            comps28,
            &[
                (GrieRosen, true, 2.5),
                (HappyCat, true, 10.0),
                (Schwefel, true, 2.5),
                (EScaffer6, true, 5.0e-4),
                (Ellips, true, 1.0e-6),
            ]
        );

        // Every composition's bias[0] is 0 (module doc's x=o_1 pin
        // derivation relies on this for every fid 23-30).
        for fid in 23u32..=28 {
            let (_, bias, _) = Cec2014::composition_spec(fid);
            assert_eq!(bias[0], 0.0, "fid={fid}: bias[0] must be 0");
        }
        for fid in [29u32, 30] {
            let (_, bias, _) = Cec2014::composition_hybrid_spec(fid);
            assert_eq!(bias[0], 0.0, "fid={fid}: bias[0] must be 0");
        }
    }

    #[test]
    fn composition_hybrid_spec_maps_cf07_cf08_to_the_right_hf0n_fids() {
        let (delta29, bias29, hybrids29) = Cec2014::composition_hybrid_spec(29);
        assert_eq!(delta29, &[10.0, 30.0, 50.0]);
        assert_eq!(bias29, &[0.0, 100.0, 200.0]);
        assert_eq!(hybrids29, &[17, 18, 19]);

        let (delta30, bias30, hybrids30) = Cec2014::composition_hybrid_spec(30);
        assert_eq!(delta30, &[10.0, 30.0, 50.0]);
        assert_eq!(bias30, &[0.0, 100.0, 200.0]);
        assert_eq!(hybrids30, &[20, 21, 22]);
    }

    #[test]
    fn comp_shuffle_is_populated_only_for_fid_29_and_30() {
        for fid in [23u32, 24, 25, 26, 27, 28] {
            let p = Cec2014::new(fid, 10).unwrap();
            assert!(p.comp_shuffle.is_empty(), "fid={fid} should have no comp_shuffle");
        }
        for fid in [29u32, 30] {
            for &dim in &[10usize, 30] {
                let p = Cec2014::new(fid, dim).unwrap();
                assert_eq!(p.comp_shuffle.len(), 3, "fid={fid} dim={dim}: 3 hybrid components");
                for block in &p.comp_shuffle {
                    assert_eq!(block.len(), dim, "fid={fid} dim={dim}");
                    let mut sorted = block.clone();
                    sorted.sort_unstable();
                    assert_eq!(
                        sorted,
                        (0..dim).collect::<Vec<_>>(),
                        "fid={fid} dim={dim}: not a permutation of 0..{dim}"
                    );
                }
            }
        }
    }

    #[test]
    fn shuffle_is_populated_only_for_fid_17_to_22() {
        for fid in 1u32..=16 {
            let p = Cec2014::new(fid, 10).unwrap();
            assert!(p.shuffle.is_empty(), "fid={fid} should have no shuffle");
        }
        for fid in 23u32..=30 {
            let p = Cec2014::new(fid, 10).unwrap();
            assert!(p.shuffle.is_empty(), "fid={fid} (composition) should have no single shuffle field");
        }
        for fid in 17u32..=22 {
            for &dim in &[10usize, 30] {
                let p = Cec2014::new(fid, dim).unwrap();
                assert_eq!(p.shuffle.len(), dim, "fid={fid} dim={dim}");
                // Module doc's file-grammar note (data.rs): every
                // shuffle_data_<fid>_D<dim>.txt is a 1-based permutation of
                // 1..=dim -- assert the 0-based result is a permutation of
                // 0..dim.
                let mut sorted = p.shuffle.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (0..dim).collect::<Vec<_>>(), "fid={fid} dim={dim}");
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

    // ---- T3: fid 17-22 random-point cross-checks vs. the compiled C
    // reference (this task's report has the full driver transcript; one
    // random point per fid at dim=10 AND dim=30, `x = o + U(-50,50)^dim`,
    // seeded `Random(20260831)` in the probe script -- SAME seed T2 used,
    // continuing the SAME rng stream fid-by-fid, `assert_close`'s 1e-9
    // relative tolerance re-used unchanged; measured max actual relative
    // deviation across all twelve points (temporary `eprintln!` during this
    // task, removed before commit, same convention T2's own report
    // describes): `2.80011838053369e-16`, essentially f64 machine epsilon
    // (`2.22e-16`) -- same class of agreement T2 measured for fid 1-16
    // (`5.64e-16`), most of the twelve points landing EXACTLY `0.0` relative
    // deviation).

    #[test]
    fn fid17_random_points_match_compiled_c_reference() {
        assert_close(17, 10, &[-2.500432146803348, -42.10542185405995, 32.427815356380044, -43.439583970833226, -120.62069986262385, 11.381038642469363, -6.525569334176311, 89.56608526645344, 71.6018621713246, -26.389252385517295], 88032153.08330923);
        assert_close(17, 30, &[52.994716489496994, -52.44621975855776, 61.372365864003115, -6.028186954213496, -55.61955895632137, 23.71581159015618, -53.57748286593011, 44.20159351319494, 113.40196675626089, -40.92079984333431, -62.77150367221094, -7.211325319114245, 77.91146350607748, 63.357050255012076, -107.97203947008211, 23.74067576711245, -14.384997469215126, 62.172732129026485, 78.90118062528931, 32.09409101304847, 84.80039725307468, 6.811187743771605, 62.59976976838179, -30.500099741701632, -87.79352120474037, 72.31525042351609, -44.18148842762021, 39.918772478187336, 53.91965513246884, -46.662168041576415], 4393491922.808026);
    }

    #[test]
    fn fid18_random_points_match_compiled_c_reference() {
        assert_close(18, 10, &[48.57900652406255, -15.740731776652844, 64.61564148080105, -87.93912224770192, 59.056656665724816, -60.182717119092295, 64.63854085407802, -54.84590887610544, -32.4359428574019, 27.625708233298525], 464235024.95850646);
        assert_close(18, 30, &[17.8383205696353, 55.65668664793827, 111.26263253738114, -111.66130812139467, -33.773099294732546, -51.20568818637151, 42.203100660263246, -84.05953276788065, -7.842230437444478, 23.609131018038738, 67.91506350526612, -3.0762169144285636, -41.47475926652625, -74.75639435199719, -8.792845006163788, 29.481247128411297, 51.92925722524386, -60.955705786262776, -69.0302805952785, 6.527900454067584, 48.707703299076776, 0.760957836770956, -52.80721610715284, -68.18732711920838, 71.72339969031093, 12.4764246717328, 1.0482019156936815, 48.176437173782176, -10.841222729801196, 42.2115981837716], 3405835706.9335184);
    }

    #[test]
    fn fid19_random_points_match_compiled_c_reference() {
        assert_close(19, 10, &[-95.8229706450162, -86.35184346377949, 3.3564136742132007, 58.56559723986315, 35.87974029659887, -39.50086031322412, -40.745311200012154, 100.585265192923, 16.732556371558303, 38.98221228705884], 2190.0811301908507);
        assert_close(19, 30, &[-90.32965801530905, -17.32048867992087, -66.04260399514229, 59.428659947282235, -2.942050940408251, 27.824999804949343, -94.44320225529552, 44.157954450923654, -28.342304301493535, 37.3795977752537, 39.232476792045574, 19.603338782983826, 17.822103272166565, 53.038344123058565, 43.250339701216674, 54.366576084603956, 92.13325109826314, 23.80311300489457, -47.279307909163926, -5.790176136402071, 58.48821645908325, 27.268512975323233, 30.025979292284404, 25.632164343873697, 101.33602339637324, -20.31253850817324, -3.279309509610215, 15.602283795414635, -43.22119054919473, -64.72389735383906], 3692.8401848672056);
    }

    #[test]
    fn fid20_random_points_match_compiled_c_reference() {
        assert_close(20, 10, &[-51.22774168755064, 29.031784032627478, 9.601073756812696, -61.95342595399873, -17.27930096507798, 3.7984127133462096, 14.244203577947047, 48.78646952427393, 26.41795928294591, -49.49308600530463], 1770859962.9428875);
        assert_close(20, 30, &[-27.4933935582077, 54.75795251940757, -57.662878878706664, -37.39136286370544, -41.82919746512113, 99.86652044365077, -20.30313578080991, 44.96212191730231, 91.559029225204, -103.64062397325714, 113.23160740231616, 33.81229332666926, 29.134494761505195, 50.138896477722206, 44.129608203107374, -83.13257470797365, 26.939353265625243, 42.463299906659486, 40.37465334145587, 97.53279497085643, 62.73465165066144, -8.947641716894989, 9.495017887025043, 21.880848512272934, -63.83603207925365, 50.49261641386282, -29.599979004140863, -60.58583358359554, 8.150786277027379, -72.6930081343279], 5615721303.685283);
    }

    #[test]
    fn fid21_random_points_match_compiled_c_reference() {
        assert_close(21, 10, &[9.362161551834731, 91.95182868406012, 5.941812949278692, -22.92858081427154, -23.149685029374332, -31.365157602254868, -69.50855649088732, 27.73788201820996, -108.81693717517614, -12.960629897344536], 2215422879.1351266);
        assert_close(21, 30, &[18.320221883732266, 60.773588615216454, -68.14602589699216, -52.602914264163324, -57.45633172251882, 17.735328911465473, -3.41384990432735, 94.22542295636434, -110.30611998812464, -2.3632541642189224, -44.87356956767915, 88.52342855341588, -36.86377598396542, 6.367142111397349, 92.83593328984574, 7.375245721791529, -102.30297539422241, -89.91493069923021, -26.8648141734544, -24.598775743161895, 0.004873590229173885, 44.366880253096724, 21.80216668355375, 102.1585656910766, 40.063871730637445, 110.65009741437572, -118.51396777807592, -20.925138289978094, 87.88000096441706, -55.51507089717613], 121034139.02812998);
    }

    #[test]
    fn fid22_random_points_match_compiled_c_reference() {
        assert_close(22, 10, &[126.36857972718498, 1.6562349756550532, 101.27514160148674, 65.65380564926998, -49.04946920520006, -24.781105314595095, 101.63963797829702, -81.02091115498885, 30.277705314249445, 73.20044018682117], 199188.80366484044);
        assert_close(22, 30, &[65.12002080520298, -16.9396791376041, 20.78186353435987, 83.82754404834698, -12.7098201658129, 12.976341888481059, 62.30784113903372, -3.2047962456268877, -30.635345463816336, 35.202314899015654, 21.81745946326599, 32.1147598562277, 37.12062109921547, 125.19628096458891, -11.331209962229664, -68.17446886010657, -91.38573304526764, 31.612153071081124, -36.375606379543946, 29.172723174476047, 45.11418484102837, -3.681043764402485, -7.806933301035222, -15.515668157850904, 98.69224727389152, 49.649112388737976, -33.62177762414724, 52.7900271084531, -12.707344505620043, -22.138791070516277], 74345.95791923154);
    }

    // ---- T4: fid 23-30 random-point cross-checks vs. the compiled C
    // reference (this task's report has the full driver transcript; one
    // random point per fid at dim=10 AND dim=30, `x = comp_shift[0] +
    // U(-50,50)^dim` -- component 1's own shift row, since composition fids
    // have no single `o` of their own -- freshly seeded
    // `random.Random(20260831)` this task, drawn fid-then-dim in ascending
    // order 23..=30 x {10,30}; `assert_close`'s 1e-9 relative tolerance
    // re-used unchanged; measured max actual relative deviation stated in
    // this task's report).

    #[test]
    fn fid23_random_points_match_compiled_c_reference() {
        assert_close(23, 10, &[-7.8677432362617665, -89.17019019608017, -27.033404976723887, 26.50946206852018, -8.860600212270185, 22.89830496768795, 105.13163730035133, 9.939687421633463, 44.341124665759494, -15.769046094660968], 3651.1538638275088);
        assert_close(23, 30, &[47.627405400038576, -99.51098810057798, 1.9111455308991836, 63.92085908513991, 56.140540694032296, 35.233077915374764, 58.07972376859753, -35.42480433162503, 86.1412292506958, -30.300593552477977, 28.668143502104357, -78.34619162835128, 86.72807496877424, -84.02068749942225, 41.9128535217432, 95.49646094945328, -100.37394845658721, -38.49352361667951, 11.006583382168955, 25.224164047593177, 35.72727065182526, 74.1189770040699, -19.748682104371852, 11.18785019153811, -57.78708765095375, 18.35054917463227, 46.62719586613486, 95.16182146566703, -46.64158301651572, 40.21214609277175], 10654.151383037428);
    }

    #[test]
    fn fid24_random_points_match_compiled_c_reference() {
        assert_close(24, 10, &[10.538574650924353, 21.71657101048359, -6.529874266008379, -12.003649438562961, 119.4068765821656, 38.037836137946066, 84.98012260743377, 60.43956854859941, 53.01043320093193, 54.580769680956784], 6218.086927684563);
        assert_close(24, 30, &[-20.202111303502896, 93.1139894350747, 40.117116790571714, -35.72583531225571, 26.577120621708232, 47.01486507066685, 62.544682413619, 31.22594465682419, 77.60414562088934, 50.564192465697, -21.517091697060245, -96.91544704566391, 47.81256219068379, -48.60927141102055, -14.762517769647019, 53.96927689487353, -22.96117905614536, 65.24476908185983, -76.59322837134059, 28.189840462200028, 41.4347486751381, 15.07244375236235, -11.148641158890932, 40.417627200084674, -19.333868160618096, -29.402654593041923, -59.97110411162333, -45.17387553641588, 75.93828011745745, 48.65842424829842], 12123.693078361453);
    }

    #[test]
    fn fid25_random_points_match_compiled_c_reference() {
        assert_close(25, 10, &[-28.11935019709343, 19.832629114998994, 108.19433173007506, -56.19684852691445, 106.86303785036435, 6.157768170034604, 46.97833021301033, -45.152839255395605, -12.748255097209025, -16.251772693474408], 3435.4493844729154);
        assert_close(25, 30, &[-22.626037567386273, 88.86398389885761, 38.795314060719576, -55.33378581949537, 68.04124661335723, 73.48362828820807, -6.719560842273026, -101.58014999739495, -57.82311577026086, -17.85438720527955, 57.52902338226942, -40.00326043456961, -121.0172460411053, 30.210528830783257, 80.24317919736184, 41.30628330693134, 22.867217950548472, -27.826488483810223, -2.10725554965358, -28.719915068232112, -11.110971691026096, 66.3677575164731, -7.400852072480404, 39.517679116881155, 27.819537615743016, -26.487957926261924, -28.154023734416086, -30.83461220726757, 65.01134792186639, 22.63335212936468], 3119.8702272067935);
    }

    #[test]
    fn fid26_random_points_match_compiled_c_reference() {
        assert_close(26, 10, &[-79.71611590760119, -78.39491422621472, 11.884000995616198, -59.32876353157014, 82.51771172045838, -16.794878806421305, -81.85603686271125, 56.70835313478971, -33.51263279325283, -18.707338162991128], 3610.7629876471406);
        assert_close(26, 30, &[-55.981767778258245, -52.66874573943463, -55.37995163990316, -34.766700441276846, 57.967815220415225, 79.27322892388325, -116.4033762214682, 52.88400552781809, 31.628437149005265, -72.85487613094364, 6.880342187086981, -5.587738391625756, -90.16435595842945, -62.91498507257436, 13.53328451827877, -17.811940311336155, 103.57908462834632, -21.743095176479194, 47.380115664568365, 10.863557025931385, -55.971179288714495, 31.53594708810982, 11.936448553574545, -67.79295406438283, 22.31491552180184, -83.79455116017434, 32.81410448498559, -9.014042100388313, -70.31734819206142, -61.11208567111331], 5080.322149527994);
    }

    #[test]
    fn fid27_random_points_match_compiled_c_reference() {
        assert_close(27, 10, &[-44.74527199525133, -2.044866325544824, 89.42696445048597, 48.8364973759149, -49.56272841122208, 5.714952581831186, -46.28481309030215, 7.641224265392935, -99.02466256429832, 11.213903163050848], 3292.487863198183);
        assert_close(27, 30, &[-35.78721166335379, -33.22310639438849, 15.33912560421511, 19.16216392602312, -83.86937510436657, 54.815439095551525, 19.809893496257814, 74.12876520354732, -100.51384537724682, 21.81127889617646, 31.310760845428653, -32.36672969859296, 82.93102849463861, 75.72543257706411, 7.474598066747879, 100.02778487434688, -5.72704285584966, -45.50975463978993, -87.63121804835146, 58.99300144637027, 43.19794635085112, -83.3770595994491, -10.809277838741087, -30.3175887773862, 119.93195909366165, 108.23621760612772, -103.41859594560972, -1.4651346376267824, -21.693676733220606, -78.72096678159747], 3723.4661100157577);
    }

    #[test]
    fn fid28_random_points_match_compiled_c_reference() {
        assert_close(28, 10, &[57.0468783659007, -69.39247482570593, 41.93800994123095, 41.13659647884014, -111.92545227263582, -76.07035211292828, -37.12956858820222, -46.11150582647872, -4.168638106511466, -70.56053687750892], 10174.985596217308);
        assert_close(28, 30, &[-4.201680556081294, -87.98838893896509, -38.555268125895914, 59.310334877917136, -75.58580323324865, -38.31290490985212, -76.46136542746552, 31.704609082883252, -65.08168888457725, -108.55866216531444, 35.14872644750017, 56.09270305952275, 37.17218347951277, 75.29428805644707, -80.98059747964214, 43.55975089313601, -54.344044619049605, 18.71684824441931, -14.08856091608557, -96.15942706794887, 59.38795379137728, -73.77130846812373, 23.88879213536638, -40.461139774697045, -48.749974719166886, 107.95551096779454, -34.3267304164676, 12.689378117375568, 7.466148588442973, -6.632734657447358], 64297.32805039919);
    }

    #[test]
    fn fid29_random_points_match_compiled_c_reference() {
        assert_close(29, 10, &[-1.507026198170287, -41.288106524032614, -54.57271761272953, -31.75732056851799, -60.53873241035736, 60.40931426796034, 38.28017930151003, 67.60004470131997, -27.874549797118895, 8.729323088926156], 704065758.2204767);
        assert_close(29, 30, &[-70.08879401329713, -82.45459370156368, -26.236901883241316, -20.22446413016266, -106.92924854127855, 58.690491949602716, 67.25883894289142, 87.09123266734687, -31.74616148469419, 42.03212067225116, -53.815525415217465, -16.592674057272376, -28.137975737208116, 46.488342052505374, 37.68767505601551, 11.694152742656811, -44.65774635518946, 27.18702582656813, -2.12891110243298, -3.8484520434210587, 37.726861188531835, 37.388668369801024, 55.76590409397422, 115.02681249949515, 7.270504733070421, -59.86554314301833, -16.749003830349153, 117.59081901529808, 44.790586420383164, 75.58159358041888], 5413900021.400275);
    }

    #[test]
    fn fid30_random_points_match_compiled_c_reference() {
        assert_close(30, 10, &[32.97369530204225, -67.12630862461504, 3.9813862129021746, -36.902365041517484, 29.23771485824784, -23.410332071163225, 29.857319195389593, 5.5632006261328595, 14.40772625072149, 86.87936438155106], 91224587.29388589);
        assert_close(30, 30, &[45.5983000631529, -87.35336599137987, 22.575242113564563, -33.874359774610284, 12.5257633813587, -108.28685806509186, 64.3449876539375, 13.478894877927742, -31.681168693411777, 55.79247006772927, -44.38707917416122, 15.364860564291018, 17.184482101945733, -11.167597641147808, -65.39650668532516, 70.2309350685909, 24.41027505041062, -27.997994422266515, 37.87042856926411, -3.4520779060146083, 30.99319431977716, -88.50518949055623, 42.98115118747468, 30.248259826886308, 21.781262680076836, -32.56372065025978, -19.585858538742954, 64.0634605108763, 66.15076953985927, -54.57627038571515], 1514834951.239448);
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
        for fid in 1u32..=30 {
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
