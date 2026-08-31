//! Embedded CEC 2014 `input_data/` files for fid 1-16 (basic + simple-
//! multimodal functions) -- vendored verbatim (byte-for-byte, `include_str!`)
//! from the official `github.com/P-N-Suganthan/CEC2014` repository's
//! `cec14-c-code.zip` -> `cec14-c-code/input_data/` (fetched directly,
//! PROVENANCE recorded in this crate's `cec2014/mod.rs` module doc and this
//! task's report). Files live at `crates/problems/data/cec2014/`, named
//! exactly as the upstream repo names them (`M_<fid>_D<dim>.txt`,
//! `shift_data_<fid>.txt`, `shuffle_data_<fid>_D<dim>.txt`), mirroring the
//! `cec2022/data.rs` convention this module is siblings with.
//!
//! ## Vendoring scope (M3-6 T2, user-approved): dims {10, 30}, ALL 30 fids'
//! files copied to `crates/problems/data/cec2014/` (so T3/T4, which extend
//! `Cec2014` to fid 17-30, do not need to re-fetch anything). This module's
//! `include_str!`/parser functions now cover fid 1-22 (T3, this task, adds
//! fid 17-22's `shift_data_<fid>.txt` + `M_<fid>_D{10,30}.txt` +
//! `shuffle_data_<fid>_D{10,30}.txt` consts and [`shuffle_indices`],
//! mirroring `cec2022/data.rs`'s own fid-6-8 shuffle-data section); fid
//! 23-30 (composition) remains vendored-but-unreferenced until a later
//! milestone task, per `mod.rs`'s module doc staging note.
//! `crates/problems/tests/cec2014_data_inventory.rs` asserts the full 30-fid
//! vendored set is present and totals the exact byte count (T2's report has
//! the measured number, 2,816,016 bytes across 106 files -- every
//! `shift_data_<fid>.txt` for `fid` 1-30 (dimension-independent, one copy),
//! every `M_<fid>_D{10,30}.txt` for `fid` 1-30, and
//! `shuffle_data_<fid>_D{10,30}.txt` ONLY for the 8 fids the reference C's
//! own loader ever reads shuffle data for -- `fid` in `{17..=22, 29, 30}`,
//! verified directly against `cec14_test_func.cpp`'s loader, quoted in
//! `mod.rs`'s module doc).
//!
//! ## File grammar (verified by reading `cec14_test_func.cpp`'s
//! `cec14_test_func` initializer, `func_num<23` branch -- the branch that
//! applies to every fid this module serves, 1..=16, byte-identical grammar to
//! `cec2022/data.rs`'s fid-1-8 section since both suites share the same
//! reference-C loader lineage)
//!
//! - **`shift_data_<fid>.txt`**: ONE line of up to 100 whitespace-separated
//!   `f64` values (`%lf`-scanned in the fixed-up C, module doc's `mod.rs`
//!   portability-edit note -- the vendored source itself reads `%Lf`, a
//!   verified bug) -- MORE than any supported `dim` needs; only the first
//!   `dim` values are the shift vector `o` for that `dim` (the C loader does
//!   exactly this: `for(i=0;i<nx;i++) fscanf(fpt,"%lf",&OShift[i]);`, i.e. it
//!   stops after `nx` values and simply never reads the rest of the line).
//!   [`shift_vector`] mirrors that: parse ALL values, then take the first
//!   `dim`.
//! - **`M_<fid>_D<dim>.txt`**: `dim` lines of `dim` whitespace-separated
//!   `f64` values each -- a `dim`x`dim` ROW-MAJOR matrix (row `i`, column
//!   `j`; the C loader reads `nx*nx` values in row-major order into a flat
//!   buffer indexed `M[i*nx+j]` by `rotatefunc`). [`rotation_matrix`]
//!   reshapes the same flat, whitespace-split stream into `Vec<Vec<f64>>`
//!   rows, matching that indexing.
//! - **`shuffle_data_<fid>_D<dim>.txt`** (fid 17-22 only, T3): ONE line of
//!   `dim` whitespace-separated 1-BASED integers, a permutation of
//!   `1..=dim` -- `cec14_test_func.cpp`'s loader (`func_num>=17&&func_num<=22`
//!   branch, `mod.rs`'s module doc quotes the surrounding conditional) reads
//!   exactly `dim` `int`s with `fscanf(fpt,"%d",&S[i])`, then each `hf0N`
//!   indexes `y[i]=z[S[i]-1]` (1-based -> 0-based at USE time, not load
//!   time). [`shuffle_indices`] converts to 0-based ON LOAD instead (matches
//!   `cec2022/data.rs`'s own [`crate::cec2022::data::shuffle_indices`]
//!   convention exactly, same reasoning: callers then index directly,
//!   `y[i]=z[shuffle[i]]`, no per-use `-1`).

const SHIFT_1: &str = include_str!("../../data/cec2014/shift_data_1.txt");
const SHIFT_2: &str = include_str!("../../data/cec2014/shift_data_2.txt");
const SHIFT_3: &str = include_str!("../../data/cec2014/shift_data_3.txt");
const SHIFT_4: &str = include_str!("../../data/cec2014/shift_data_4.txt");
const SHIFT_5: &str = include_str!("../../data/cec2014/shift_data_5.txt");
const SHIFT_6: &str = include_str!("../../data/cec2014/shift_data_6.txt");
const SHIFT_7: &str = include_str!("../../data/cec2014/shift_data_7.txt");
const SHIFT_8: &str = include_str!("../../data/cec2014/shift_data_8.txt");
const SHIFT_9: &str = include_str!("../../data/cec2014/shift_data_9.txt");
const SHIFT_10: &str = include_str!("../../data/cec2014/shift_data_10.txt");
const SHIFT_11: &str = include_str!("../../data/cec2014/shift_data_11.txt");
const SHIFT_12: &str = include_str!("../../data/cec2014/shift_data_12.txt");
const SHIFT_13: &str = include_str!("../../data/cec2014/shift_data_13.txt");
const SHIFT_14: &str = include_str!("../../data/cec2014/shift_data_14.txt");
const SHIFT_15: &str = include_str!("../../data/cec2014/shift_data_15.txt");
const SHIFT_16: &str = include_str!("../../data/cec2014/shift_data_16.txt");
const SHIFT_17: &str = include_str!("../../data/cec2014/shift_data_17.txt");
const SHIFT_18: &str = include_str!("../../data/cec2014/shift_data_18.txt");
const SHIFT_19: &str = include_str!("../../data/cec2014/shift_data_19.txt");
const SHIFT_20: &str = include_str!("../../data/cec2014/shift_data_20.txt");
const SHIFT_21: &str = include_str!("../../data/cec2014/shift_data_21.txt");
const SHIFT_22: &str = include_str!("../../data/cec2014/shift_data_22.txt");

const M_1_D10: &str = include_str!("../../data/cec2014/M_1_D10.txt");
const M_1_D30: &str = include_str!("../../data/cec2014/M_1_D30.txt");
const M_2_D10: &str = include_str!("../../data/cec2014/M_2_D10.txt");
const M_2_D30: &str = include_str!("../../data/cec2014/M_2_D30.txt");
const M_3_D10: &str = include_str!("../../data/cec2014/M_3_D10.txt");
const M_3_D30: &str = include_str!("../../data/cec2014/M_3_D30.txt");
const M_4_D10: &str = include_str!("../../data/cec2014/M_4_D10.txt");
const M_4_D30: &str = include_str!("../../data/cec2014/M_4_D30.txt");
const M_5_D10: &str = include_str!("../../data/cec2014/M_5_D10.txt");
const M_5_D30: &str = include_str!("../../data/cec2014/M_5_D30.txt");
const M_6_D10: &str = include_str!("../../data/cec2014/M_6_D10.txt");
const M_6_D30: &str = include_str!("../../data/cec2014/M_6_D30.txt");
const M_7_D10: &str = include_str!("../../data/cec2014/M_7_D10.txt");
const M_7_D30: &str = include_str!("../../data/cec2014/M_7_D30.txt");
const M_8_D10: &str = include_str!("../../data/cec2014/M_8_D10.txt");
const M_8_D30: &str = include_str!("../../data/cec2014/M_8_D30.txt");
const M_9_D10: &str = include_str!("../../data/cec2014/M_9_D10.txt");
const M_9_D30: &str = include_str!("../../data/cec2014/M_9_D30.txt");
const M_10_D10: &str = include_str!("../../data/cec2014/M_10_D10.txt");
const M_10_D30: &str = include_str!("../../data/cec2014/M_10_D30.txt");
const M_11_D10: &str = include_str!("../../data/cec2014/M_11_D10.txt");
const M_11_D30: &str = include_str!("../../data/cec2014/M_11_D30.txt");
const M_12_D10: &str = include_str!("../../data/cec2014/M_12_D10.txt");
const M_12_D30: &str = include_str!("../../data/cec2014/M_12_D30.txt");
const M_13_D10: &str = include_str!("../../data/cec2014/M_13_D10.txt");
const M_13_D30: &str = include_str!("../../data/cec2014/M_13_D30.txt");
const M_14_D10: &str = include_str!("../../data/cec2014/M_14_D10.txt");
const M_14_D30: &str = include_str!("../../data/cec2014/M_14_D30.txt");
const M_15_D10: &str = include_str!("../../data/cec2014/M_15_D10.txt");
const M_15_D30: &str = include_str!("../../data/cec2014/M_15_D30.txt");
const M_16_D10: &str = include_str!("../../data/cec2014/M_16_D10.txt");
const M_16_D30: &str = include_str!("../../data/cec2014/M_16_D30.txt");
const M_17_D10: &str = include_str!("../../data/cec2014/M_17_D10.txt");
const M_17_D30: &str = include_str!("../../data/cec2014/M_17_D30.txt");
const M_18_D10: &str = include_str!("../../data/cec2014/M_18_D10.txt");
const M_18_D30: &str = include_str!("../../data/cec2014/M_18_D30.txt");
const M_19_D10: &str = include_str!("../../data/cec2014/M_19_D10.txt");
const M_19_D30: &str = include_str!("../../data/cec2014/M_19_D30.txt");
const M_20_D10: &str = include_str!("../../data/cec2014/M_20_D10.txt");
const M_20_D30: &str = include_str!("../../data/cec2014/M_20_D30.txt");
const M_21_D10: &str = include_str!("../../data/cec2014/M_21_D10.txt");
const M_21_D30: &str = include_str!("../../data/cec2014/M_21_D30.txt");
const M_22_D10: &str = include_str!("../../data/cec2014/M_22_D10.txt");
const M_22_D30: &str = include_str!("../../data/cec2014/M_22_D30.txt");

const SHIFT_23: &str = include_str!("../../data/cec2014/shift_data_23.txt");
const SHIFT_24: &str = include_str!("../../data/cec2014/shift_data_24.txt");
const SHIFT_25: &str = include_str!("../../data/cec2014/shift_data_25.txt");
const SHIFT_26: &str = include_str!("../../data/cec2014/shift_data_26.txt");
const SHIFT_27: &str = include_str!("../../data/cec2014/shift_data_27.txt");
const SHIFT_28: &str = include_str!("../../data/cec2014/shift_data_28.txt");
const SHIFT_29: &str = include_str!("../../data/cec2014/shift_data_29.txt");
const SHIFT_30: &str = include_str!("../../data/cec2014/shift_data_30.txt");

const M_23_D10: &str = include_str!("../../data/cec2014/M_23_D10.txt");
const M_23_D30: &str = include_str!("../../data/cec2014/M_23_D30.txt");
const M_24_D10: &str = include_str!("../../data/cec2014/M_24_D10.txt");
const M_24_D30: &str = include_str!("../../data/cec2014/M_24_D30.txt");
const M_25_D10: &str = include_str!("../../data/cec2014/M_25_D10.txt");
const M_25_D30: &str = include_str!("../../data/cec2014/M_25_D30.txt");
const M_26_D10: &str = include_str!("../../data/cec2014/M_26_D10.txt");
const M_26_D30: &str = include_str!("../../data/cec2014/M_26_D30.txt");
const M_27_D10: &str = include_str!("../../data/cec2014/M_27_D10.txt");
const M_27_D30: &str = include_str!("../../data/cec2014/M_27_D30.txt");
const M_28_D10: &str = include_str!("../../data/cec2014/M_28_D10.txt");
const M_28_D30: &str = include_str!("../../data/cec2014/M_28_D30.txt");
const M_29_D10: &str = include_str!("../../data/cec2014/M_29_D10.txt");
const M_29_D30: &str = include_str!("../../data/cec2014/M_29_D30.txt");
const M_30_D10: &str = include_str!("../../data/cec2014/M_30_D10.txt");
const M_30_D30: &str = include_str!("../../data/cec2014/M_30_D30.txt");

const SHUFFLE_29_D10: &str = include_str!("../../data/cec2014/shuffle_data_29_D10.txt");
const SHUFFLE_29_D30: &str = include_str!("../../data/cec2014/shuffle_data_29_D30.txt");
const SHUFFLE_30_D10: &str = include_str!("../../data/cec2014/shuffle_data_30_D10.txt");
const SHUFFLE_30_D30: &str = include_str!("../../data/cec2014/shuffle_data_30_D30.txt");

const SHUFFLE_17_D10: &str = include_str!("../../data/cec2014/shuffle_data_17_D10.txt");
const SHUFFLE_17_D30: &str = include_str!("../../data/cec2014/shuffle_data_17_D30.txt");
const SHUFFLE_18_D10: &str = include_str!("../../data/cec2014/shuffle_data_18_D10.txt");
const SHUFFLE_18_D30: &str = include_str!("../../data/cec2014/shuffle_data_18_D30.txt");
const SHUFFLE_19_D10: &str = include_str!("../../data/cec2014/shuffle_data_19_D10.txt");
const SHUFFLE_19_D30: &str = include_str!("../../data/cec2014/shuffle_data_19_D30.txt");
const SHUFFLE_20_D10: &str = include_str!("../../data/cec2014/shuffle_data_20_D10.txt");
const SHUFFLE_20_D30: &str = include_str!("../../data/cec2014/shuffle_data_20_D30.txt");
const SHUFFLE_21_D10: &str = include_str!("../../data/cec2014/shuffle_data_21_D10.txt");
const SHUFFLE_21_D30: &str = include_str!("../../data/cec2014/shuffle_data_21_D30.txt");
const SHUFFLE_22_D10: &str = include_str!("../../data/cec2014/shuffle_data_22_D10.txt");
const SHUFFLE_22_D30: &str = include_str!("../../data/cec2014/shuffle_data_22_D30.txt");

/// Parse a whitespace-separated stream of `f64` values. Embedded-data parse
/// failure panics with a clear message (acceptable for `include_str!`-baked
/// constants that never vary at runtime -- same convention `cec2022/data.rs`
/// uses).
fn parse_floats(text: &str, source: &str) -> Vec<f64> {
    text.split_whitespace()
        .map(|tok| {
            tok.parse::<f64>().unwrap_or_else(|e| {
                panic!("cec2014 embedded data {source}: malformed float {tok:?}: {e}")
            })
        })
        .collect()
}

/// `fid`'s shift vector `o`, truncated to the first `dim` values (module
/// doc: every `shift_data_<fid>.txt` carries up to 100 values -- only `dim`
/// of them are the actual shift for that `dim`). `fid` must be `1..=22`
/// (caller's responsibility -- [`crate::cec2014::Cec2014::new`] validates
/// before calling this).
pub(crate) fn shift_vector(fid: u32, dim: usize) -> Vec<f64> {
    let (text, name) = match fid {
        1 => (SHIFT_1, "shift_data_1.txt"),
        2 => (SHIFT_2, "shift_data_2.txt"),
        3 => (SHIFT_3, "shift_data_3.txt"),
        4 => (SHIFT_4, "shift_data_4.txt"),
        5 => (SHIFT_5, "shift_data_5.txt"),
        6 => (SHIFT_6, "shift_data_6.txt"),
        7 => (SHIFT_7, "shift_data_7.txt"),
        8 => (SHIFT_8, "shift_data_8.txt"),
        9 => (SHIFT_9, "shift_data_9.txt"),
        10 => (SHIFT_10, "shift_data_10.txt"),
        11 => (SHIFT_11, "shift_data_11.txt"),
        12 => (SHIFT_12, "shift_data_12.txt"),
        13 => (SHIFT_13, "shift_data_13.txt"),
        14 => (SHIFT_14, "shift_data_14.txt"),
        15 => (SHIFT_15, "shift_data_15.txt"),
        16 => (SHIFT_16, "shift_data_16.txt"),
        17 => (SHIFT_17, "shift_data_17.txt"),
        18 => (SHIFT_18, "shift_data_18.txt"),
        19 => (SHIFT_19, "shift_data_19.txt"),
        20 => (SHIFT_20, "shift_data_20.txt"),
        21 => (SHIFT_21, "shift_data_21.txt"),
        22 => (SHIFT_22, "shift_data_22.txt"),
        other => unreachable!("shift_vector called with unsupported fid {other}"),
    };
    let all = parse_floats(text, name);
    assert!(
        all.len() >= dim,
        "cec2014 embedded data {name}: only {} values, need at least {dim}",
        all.len()
    );
    all[..dim].to_vec()
}

/// `fid`'s `dim`x`dim` rotation matrix `M`, row-major (module doc): row `i`
/// is `matrix[i]`, `matrix[i][j]` is the C reference's `M[i*nx+j]`. `fid`
/// must be `1..=22`, `dim` one of `{10,30}` (caller's responsibility --
/// `Cec2014::new` validates first).
pub(crate) fn rotation_matrix(fid: u32, dim: usize) -> Vec<Vec<f64>> {
    let (text, name) = match (fid, dim) {
        (1, 10) => (M_1_D10, "M_1_D10.txt"),
        (1, 30) => (M_1_D30, "M_1_D30.txt"),
        (2, 10) => (M_2_D10, "M_2_D10.txt"),
        (2, 30) => (M_2_D30, "M_2_D30.txt"),
        (3, 10) => (M_3_D10, "M_3_D10.txt"),
        (3, 30) => (M_3_D30, "M_3_D30.txt"),
        (4, 10) => (M_4_D10, "M_4_D10.txt"),
        (4, 30) => (M_4_D30, "M_4_D30.txt"),
        (5, 10) => (M_5_D10, "M_5_D10.txt"),
        (5, 30) => (M_5_D30, "M_5_D30.txt"),
        (6, 10) => (M_6_D10, "M_6_D10.txt"),
        (6, 30) => (M_6_D30, "M_6_D30.txt"),
        (7, 10) => (M_7_D10, "M_7_D10.txt"),
        (7, 30) => (M_7_D30, "M_7_D30.txt"),
        (8, 10) => (M_8_D10, "M_8_D10.txt"),
        (8, 30) => (M_8_D30, "M_8_D30.txt"),
        (9, 10) => (M_9_D10, "M_9_D10.txt"),
        (9, 30) => (M_9_D30, "M_9_D30.txt"),
        (10, 10) => (M_10_D10, "M_10_D10.txt"),
        (10, 30) => (M_10_D30, "M_10_D30.txt"),
        (11, 10) => (M_11_D10, "M_11_D10.txt"),
        (11, 30) => (M_11_D30, "M_11_D30.txt"),
        (12, 10) => (M_12_D10, "M_12_D10.txt"),
        (12, 30) => (M_12_D30, "M_12_D30.txt"),
        (13, 10) => (M_13_D10, "M_13_D10.txt"),
        (13, 30) => (M_13_D30, "M_13_D30.txt"),
        (14, 10) => (M_14_D10, "M_14_D10.txt"),
        (14, 30) => (M_14_D30, "M_14_D30.txt"),
        (15, 10) => (M_15_D10, "M_15_D10.txt"),
        (15, 30) => (M_15_D30, "M_15_D30.txt"),
        (16, 10) => (M_16_D10, "M_16_D10.txt"),
        (16, 30) => (M_16_D30, "M_16_D30.txt"),
        (17, 10) => (M_17_D10, "M_17_D10.txt"),
        (17, 30) => (M_17_D30, "M_17_D30.txt"),
        (18, 10) => (M_18_D10, "M_18_D10.txt"),
        (18, 30) => (M_18_D30, "M_18_D30.txt"),
        (19, 10) => (M_19_D10, "M_19_D10.txt"),
        (19, 30) => (M_19_D30, "M_19_D30.txt"),
        (20, 10) => (M_20_D10, "M_20_D10.txt"),
        (20, 30) => (M_20_D30, "M_20_D30.txt"),
        (21, 10) => (M_21_D10, "M_21_D10.txt"),
        (21, 30) => (M_21_D30, "M_21_D30.txt"),
        (22, 10) => (M_22_D10, "M_22_D10.txt"),
        (22, 30) => (M_22_D30, "M_22_D30.txt"),
        (other_fid, other_dim) => {
            unreachable!("rotation_matrix called with unsupported (fid={other_fid}, dim={other_dim})")
        }
    };
    let flat = parse_floats(text, name);
    assert_eq!(
        flat.len(),
        dim * dim,
        "cec2014 embedded data {name}: expected {} values ({dim}x{dim}), got {}",
        dim * dim,
        flat.len()
    );
    (0..dim).map(|i| flat[i * dim..(i + 1) * dim].to_vec()).collect()
}

/// `fid`'s (`17..=22`) shuffle permutation, 0-based (module doc's file
/// grammar section: `S[i]-1` conversion applied ON LOAD, mirroring
/// `cec2022::data::shuffle_indices`). `dim` must be one of `{10,30}`.
pub(crate) fn shuffle_indices(fid: u32, dim: usize) -> Vec<usize> {
    let (text, name) = match (fid, dim) {
        (17, 10) => (SHUFFLE_17_D10, "shuffle_data_17_D10.txt"),
        (17, 30) => (SHUFFLE_17_D30, "shuffle_data_17_D30.txt"),
        (18, 10) => (SHUFFLE_18_D10, "shuffle_data_18_D10.txt"),
        (18, 30) => (SHUFFLE_18_D30, "shuffle_data_18_D30.txt"),
        (19, 10) => (SHUFFLE_19_D10, "shuffle_data_19_D10.txt"),
        (19, 30) => (SHUFFLE_19_D30, "shuffle_data_19_D30.txt"),
        (20, 10) => (SHUFFLE_20_D10, "shuffle_data_20_D10.txt"),
        (20, 30) => (SHUFFLE_20_D30, "shuffle_data_20_D30.txt"),
        (21, 10) => (SHUFFLE_21_D10, "shuffle_data_21_D10.txt"),
        (21, 30) => (SHUFFLE_21_D30, "shuffle_data_21_D30.txt"),
        (22, 10) => (SHUFFLE_22_D10, "shuffle_data_22_D10.txt"),
        (22, 30) => (SHUFFLE_22_D30, "shuffle_data_22_D30.txt"),
        (other_fid, other_dim) => {
            unreachable!("shuffle_indices called with unsupported (fid={other_fid}, dim={other_dim})")
        }
    };
    let ints: Vec<i64> = text
        .split_whitespace()
        .map(|tok| {
            tok.parse::<i64>()
                .unwrap_or_else(|e| panic!("cec2014 embedded data {name}: malformed int {tok:?}: {e}"))
        })
        .collect();
    assert_eq!(
        ints.len(),
        dim,
        "cec2014 embedded data {name}: expected {dim} 1-based shuffle indices, got {}",
        ints.len()
    );
    ints.iter()
        .map(|&one_based| {
            assert!(
                (1..=dim as i64).contains(&one_based),
                "cec2014 embedded data {name}: shuffle index {one_based} out of range 1..={dim}"
            );
            (one_based - 1) as usize
        })
        .collect()
}

/// `fid`'s (`23..=30`) first `cf_num` component shift rows, each truncated
/// to the first `dim` values (M3-6 T4: `cec14_test_func.cpp`'s
/// `cec14_test_func` initializer hardcodes a LOCAL `cf_num=10` for loading
/// `OShift`/`M`/`SS` for EVERY composition `func_num` (`>=23`), regardless of
/// that composition's OWN `cf_num` (5 for `cf01`, 3 for `cf02`/`cf03`, 5 for
/// `cf04`/`cf05`/`cf06`, 3 for `cf07`/`cf08` -- `mod.rs`'s module doc T4
/// section quotes the cf01..cf08 bodies and each one's own local `cf_num`)
/// -- so every `shift_data_<fid>.txt` for `fid` 23-30 vendors exactly 10
/// rows (verified: `wc -l` on every one of the 8 files gives `10`), of which
/// only the first `cf_num` (caller's, this composition's own, always `<=5`)
/// are ever read by that composition's own dispatch. Quoted load loop
/// (`func_num>=23` branch):
/// ```text
/// OShift=(double *)malloc(nx*cf_num*sizeof(double));
/// for(i=0;i<cf_num-1;i++) {
///     for (j=0;j<nx;j++) fscanf(fpt,"%lf",&OShift[i*nx+j]);
///     fscanf(fpt,"%*[^\n]%*c");   // skip rest of this line, incl. newline
/// }
/// for (j=0;j<nx;j++) fscanf(fpt,"%lf",&OShift[(cf_num-1)*nx+j]);
/// ```
/// The `%*[^\n]%*c` skip is why each vendored row must be split BY LINE
/// (unlike [`shift_vector`]'s flat `split_whitespace`, which is safe there
/// only because fid 1-22's files are single-row): every row of
/// `shift_data_<fid>.txt` (`fid` 23-30) genuinely carries 100 tokens (module
/// doc's file-grammar section for fid 1-22 already notes shift files carry
/// "MORE than any supported `dim` needs"), and taking `dim` tokens from the
/// FLATTENED stream instead of per-line would misalign at `dim<100` -- this
/// function splits on `.lines()` first, matching the C's per-row skip
/// exactly. `fid` must be `23..=30` (caller's responsibility -- `Cec2014::new`
/// validates).
pub(crate) fn composition_shift_blocks(fid: u32, dim: usize, cf_num: usize) -> Vec<Vec<f64>> {
    let (text, name) = match fid {
        23 => (SHIFT_23, "shift_data_23.txt"),
        24 => (SHIFT_24, "shift_data_24.txt"),
        25 => (SHIFT_25, "shift_data_25.txt"),
        26 => (SHIFT_26, "shift_data_26.txt"),
        27 => (SHIFT_27, "shift_data_27.txt"),
        28 => (SHIFT_28, "shift_data_28.txt"),
        29 => (SHIFT_29, "shift_data_29.txt"),
        30 => (SHIFT_30, "shift_data_30.txt"),
        other => unreachable!("composition_shift_blocks called with unsupported fid {other}"),
    };
    let rows: Vec<&str> = text.lines().filter(|line| !line.trim().is_empty()).collect();
    assert!(
        rows.len() >= cf_num,
        "cec2014 embedded data {name}: only {} shift rows, need at least {cf_num}",
        rows.len()
    );
    rows[..cf_num]
        .iter()
        .map(|line| {
            let vals = parse_floats(line, name);
            assert!(
                vals.len() >= dim,
                "cec2014 embedded data {name}: shift row has only {} values, need at least {dim}",
                vals.len()
            );
            vals[..dim].to_vec()
        })
        .collect()
}

/// `fid`'s (`23..=30`) first `cf_num` component `dim`x`dim` rotation
/// matrices, row-major each block (M3-6 T4: unlike [`composition_shift_blocks`],
/// `M_<fid>_D<dim>.txt` is loaded with NO per-row skip -- the C reads
/// `cf_num*nx*nx` doubles in one flat loop (`func_num>=23` branch):
/// ```text
/// M=(double*)malloc(cf_num*nx*nx*sizeof(double));
/// for (i=0; i<cf_num*nx*nx; i++) fscanf(fpt,"%lf",&M[i]);
/// ```
/// so every `M_<fid>_D<dim>.txt` (`fid` 23-30) is dimension-SPECIFIC and
/// EXACTLY `10*dim*dim` values (verified: `wc -w` on `M_23_D10.txt` gives
/// `1000` = `10*10*10`, `M_23_D30.txt` gives `9000` = `10*30*30`, no padding
/// -- unlike the shift files' fixed 100-wide rows) -- a plain flat parse and
/// contiguous `dim*dim` chunking (same shape [`rotation_matrix`] already
/// uses for fid 1-22, just chunked `cf_num` times instead of once). `fid`
/// must be `23..=30`, `dim` one of `{10,30}` (caller's responsibility --
/// `Cec2014::new` validates).
pub(crate) fn composition_rotation_blocks(fid: u32, dim: usize, cf_num: usize) -> Vec<Vec<Vec<f64>>> {
    let (text, name) = match (fid, dim) {
        (23, 10) => (M_23_D10, "M_23_D10.txt"),
        (23, 30) => (M_23_D30, "M_23_D30.txt"),
        (24, 10) => (M_24_D10, "M_24_D10.txt"),
        (24, 30) => (M_24_D30, "M_24_D30.txt"),
        (25, 10) => (M_25_D10, "M_25_D10.txt"),
        (25, 30) => (M_25_D30, "M_25_D30.txt"),
        (26, 10) => (M_26_D10, "M_26_D10.txt"),
        (26, 30) => (M_26_D30, "M_26_D30.txt"),
        (27, 10) => (M_27_D10, "M_27_D10.txt"),
        (27, 30) => (M_27_D30, "M_27_D30.txt"),
        (28, 10) => (M_28_D10, "M_28_D10.txt"),
        (28, 30) => (M_28_D30, "M_28_D30.txt"),
        (29, 10) => (M_29_D10, "M_29_D10.txt"),
        (29, 30) => (M_29_D30, "M_29_D30.txt"),
        (30, 10) => (M_30_D10, "M_30_D10.txt"),
        (30, 30) => (M_30_D30, "M_30_D30.txt"),
        (other_fid, other_dim) => unreachable!(
            "composition_rotation_blocks called with unsupported (fid={other_fid}, dim={other_dim})"
        ),
    };
    let flat = parse_floats(text, name);
    let block_len = dim * dim;
    assert!(
        flat.len() >= cf_num * block_len,
        "cec2014 embedded data {name}: only {} values, need at least {} ({cf_num} blocks of {dim}x{dim})",
        flat.len(),
        cf_num * block_len
    );
    (0..cf_num)
        .map(|b| {
            let block = &flat[b * block_len..(b + 1) * block_len];
            (0..dim).map(|i| block[i * dim..(i + 1) * dim].to_vec()).collect()
        })
        .collect()
}

/// `fid`'s (`29..=30` only -- cf07/cf08, the only two composition functions
/// whose components are themselves hybrids, module doc's T4 section) first
/// `cf_num` component 0-based shuffle permutations, `dim` entries each (M3-6
/// T4: the C's `func_num==29||func_num==30` branch reads `nx*cf_num` ints
/// FLAT, no per-row skip -- same shape as [`composition_rotation_blocks`],
/// not [`composition_shift_blocks`]):
/// ```text
/// SS=(int *)malloc(nx*cf_num*sizeof(int));
/// for(i=0;i<nx*cf_num;i++) fscanf(fpt,"%d",&SS[i]);
/// ```
/// verified: `wc -w` on `shuffle_data_29_D10.txt` gives `100` = `10*10`,
/// `shuffle_data_29_D30.txt` gives `300` = `10*30` -- each `dim`-wide chunk
/// independently a permutation of `1..=dim` (spot-checked in this module's
/// own tests). `fid` must be `29..=30`, `dim` one of `{10,30}` (caller's
/// responsibility -- `Cec2014::new` validates).
pub(crate) fn composition_shuffle_blocks(fid: u32, dim: usize, cf_num: usize) -> Vec<Vec<usize>> {
    let (text, name) = match (fid, dim) {
        (29, 10) => (SHUFFLE_29_D10, "shuffle_data_29_D10.txt"),
        (29, 30) => (SHUFFLE_29_D30, "shuffle_data_29_D30.txt"),
        (30, 10) => (SHUFFLE_30_D10, "shuffle_data_30_D10.txt"),
        (30, 30) => (SHUFFLE_30_D30, "shuffle_data_30_D30.txt"),
        (other_fid, other_dim) => unreachable!(
            "composition_shuffle_blocks called with unsupported (fid={other_fid}, dim={other_dim})"
        ),
    };
    let ints: Vec<i64> = text
        .split_whitespace()
        .map(|tok| {
            tok.parse::<i64>()
                .unwrap_or_else(|e| panic!("cec2014 embedded data {name}: malformed int {tok:?}: {e}"))
        })
        .collect();
    assert!(
        ints.len() >= cf_num * dim,
        "cec2014 embedded data {name}: only {} values, need at least {} ({cf_num} blocks of {dim})",
        ints.len(),
        cf_num * dim
    );
    (0..cf_num)
        .map(|b| {
            ints[b * dim..(b + 1) * dim]
                .iter()
                .map(|&one_based| {
                    assert!(
                        (1..=dim as i64).contains(&one_based),
                        "cec2014 embedded data {name}: shuffle index {one_based} out of range 1..={dim}"
                    );
                    (one_based - 1) as usize
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
// Literals below are transcribed digit-for-digit from the vendored .txt
// files (module doc's data-integrity spot checks) so they can be diffed
// against the raw text by eye; clippy's "round to f64 precision" rewrite
// would obscure that traceability without changing the value.
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    // ---- data-integrity: spot-assert embedded values against the raw
    // vendored text (module doc: the exact files copied into
    // crates/problems/data/cec2014/, byte-for-byte from the official repo).

    #[test]
    fn shift_vector_1_dim10_matches_vendored_file() {
        // data/cec2014/shift_data_1.txt, first two whitespace-separated
        // tokens: "5.0355789822908633e+001  6.4926709932099072e+001 ...".
        let o = shift_vector(1, 10);
        assert_eq!(o.len(), 10);
        assert_eq!(o[0], 5.0355789822908633e+001);
        assert_eq!(o[1], 6.4926709932099072e+001);
    }

    #[test]
    fn shift_vector_6_dim30_takes_only_first_30_of_the_100_column_file() {
        // data/cec2014/shift_data_6.txt carries 100 values (module doc); at
        // dim=30 only the first 30 are the shift vector.
        let o = shift_vector(6, 30);
        assert_eq!(o.len(), 30);
    }

    #[test]
    fn rotation_matrix_1_dim10_row_major_spot_checks() {
        // data/cec2014/M_1_D10.txt row 1 starts
        // "1.4706067109255289e-001  0.0...  7.2108895223116487e-001 ..."
        // (columns 0 and 5); row 2 starts "0.0  7.2941502719915285e-002
        // ..." (column 1). Confirms row-major reshape (module doc):
        // matrix[0][0] is the file's very first token, matrix[1][1] is the
        // SECOND line's second token, not e.g. the 11th flat value
        // misindexed.
        let m = rotation_matrix(1, 10);
        assert_eq!(m.len(), 10);
        assert_eq!(m[0].len(), 10);
        assert_eq!(m[0][0], 1.4706067109255289e-001);
        assert_eq!(m[0][5], 7.2108895223116487e-001);
        assert_eq!(m[1][1], 7.2941502719915285e-002);
    }

    #[test]
    fn all_fid_1_to_16_dims_10_30_parse_without_panic() {
        for fid in 1u32..=16 {
            for &dim in &[10usize, 30] {
                let o = shift_vector(fid, dim);
                assert_eq!(o.len(), dim, "fid={fid} dim={dim}");
                let m = rotation_matrix(fid, dim);
                assert_eq!(m.len(), dim, "fid={fid} dim={dim}");
                assert!(m.iter().all(|row| row.len() == dim), "fid={fid} dim={dim}");
            }
        }
    }

    // ---- T3: fid 17-22 hybrid data (shift/rotation/shuffle) ----

    #[test]
    fn all_fid_17_to_22_dims_10_30_parse_without_panic() {
        for fid in 17u32..=22 {
            for &dim in &[10usize, 30] {
                let o = shift_vector(fid, dim);
                assert_eq!(o.len(), dim, "fid={fid} dim={dim}");
                let m = rotation_matrix(fid, dim);
                assert_eq!(m.len(), dim, "fid={fid} dim={dim}");
                assert!(m.iter().all(|row| row.len() == dim), "fid={fid} dim={dim}");
                let s = shuffle_indices(fid, dim);
                assert_eq!(s.len(), dim, "fid={fid} dim={dim}");
            }
        }
    }

    #[test]
    fn shuffle_indices_17_dim10_matches_vendored_file_0_based() {
        // data/cec2014/shuffle_data_17_D10.txt: "7 8 5 10 3 6 9 4 2 1"
        // (1-based); shuffle_indices subtracts 1 from each.
        let s = shuffle_indices(17, 10);
        assert_eq!(s, vec![6, 7, 4, 9, 2, 5, 8, 3, 1, 0]);
    }

    #[test]
    fn shuffle_indices_22_dim30_matches_vendored_file_0_based() {
        // data/cec2014/shuffle_data_22_D30.txt: "12 10 4 11 2 15 26 20 6 5
        // 27 30 17 7 29 13 9 25 1 3 21 19 16 24 23 28 8 22 18 14" (1-based).
        let s = shuffle_indices(22, 30);
        assert_eq!(
            s,
            vec![
                11, 9, 3, 10, 1, 14, 25, 19, 5, 4, 26, 29, 16, 6, 28, 12, 8, 24, 0, 2, 20, 18, 15,
                23, 22, 27, 7, 21, 17, 13
            ]
        );
    }

    #[test]
    fn every_shuffle_file_is_a_valid_permutation_of_0_dim() {
        // module doc's file-grammar note: every shuffle_data_<fid>_D<dim>.txt
        // is a 1-based permutation of 1..=dim; shuffle_indices converts to
        // 0-based -- assert the 0-based result is a permutation of 0..dim
        // (all values distinct, full coverage), not just "dim values in
        // range" (which the parser itself already asserts per-element).
        for fid in 17u32..=22 {
            for &dim in &[10usize, 30] {
                let mut s = shuffle_indices(fid, dim);
                s.sort_unstable();
                let expect: Vec<usize> = (0..dim).collect();
                assert_eq!(s, expect, "fid={fid} dim={dim}: not a permutation of 0..{dim}");
            }
        }
    }

    // ---- T4: fid 23-30 composition data (multi-row shift/rotation/shuffle
    // blocks) ----

    /// This composition fid's own `cf_num` (`mod.rs`'s module doc T4 section,
    /// transcribed from `cf01`..`cf08`'s own local `cf_num` -- 5/3/3/5/5/5/3/3),
    /// duplicated here (not imported from `mod.rs`, which is a sibling module
    /// with its own private `composition_cf_num`) purely so `data.rs`'s own
    /// tests can drive the parsers without depending on `Cec2014` at all.
    fn cf_num(fid: u32) -> usize {
        match fid {
            23 | 26 | 27 | 28 => 5,
            24 | 25 | 29 | 30 => 3,
            other => unreachable!("cf_num called with unsupported fid {other}"),
        }
    }

    #[test]
    fn every_composition_shift_data_file_has_exactly_10_rows() {
        // Module doc's `composition_shift_blocks` note: the C hardcodes a
        // LOCAL cf_num=10 for the shift-data LOAD LOOP regardless of this
        // composition's own (smaller) cf_num -- assert the raw vendored file
        // really does carry 10 non-empty lines for every fid 23-30 (not just
        // "at least cf_num", the stronger claim the module doc makes).
        for fid in 23u32..=30 {
            let text = match fid {
                23 => SHIFT_23,
                24 => SHIFT_24,
                25 => SHIFT_25,
                26 => SHIFT_26,
                27 => SHIFT_27,
                28 => SHIFT_28,
                29 => SHIFT_29,
                30 => SHIFT_30,
                other => unreachable!("fid {other}"),
            };
            let rows = text.lines().filter(|l| !l.trim().is_empty()).count();
            assert_eq!(rows, 10, "fid={fid}: expected 10 shift rows (C's hardcoded cf_num=10 load)");
        }
    }

    #[test]
    fn composition_shift_blocks_returns_cf_num_rows_of_dim_values() {
        for fid in 23u32..=30 {
            let n = cf_num(fid);
            for &dim in &[10usize, 30] {
                let rows = composition_shift_blocks(fid, dim, n);
                assert_eq!(rows.len(), n, "fid={fid} dim={dim}");
                assert!(rows.iter().all(|r| r.len() == dim), "fid={fid} dim={dim}");
            }
        }
    }

    #[test]
    fn composition_shift_blocks_23_dim10_first_row_matches_vendored_file() {
        // data/cec2014/shift_data_23.txt, first line, first two tokens:
        // "2.2195926903730161e+001 -5.8022979077717679e+001 ...".
        let rows = composition_shift_blocks(23, 10, 5);
        assert_eq!(rows[0][0], 2.2195926903730161e+001);
        assert_eq!(rows[0][1], -5.8022979077717679e+001);
    }

    #[test]
    fn composition_shift_blocks_23_dim10_second_row_is_a_different_component() {
        // Confirms the per-line skip: row 1 (component 2's own shift) must
        // start with the SECOND line's own first token, not e.g. a
        // misaligned offset into row 0's 100-wide line.
        let rows = composition_shift_blocks(23, 10, 5);
        assert_eq!(rows[1][0], -6.9900980251695813e+001);
    }

    #[test]
    fn composition_rotation_blocks_returns_cf_num_dim_by_dim_matrices() {
        for fid in 23u32..=30 {
            let n = cf_num(fid);
            for &dim in &[10usize, 30] {
                let blocks = composition_rotation_blocks(fid, dim, n);
                assert_eq!(blocks.len(), n, "fid={fid} dim={dim}");
                for b in &blocks {
                    assert_eq!(b.len(), dim, "fid={fid} dim={dim}");
                    assert!(b.iter().all(|row| row.len() == dim), "fid={fid} dim={dim}");
                }
            }
        }
    }

    #[test]
    fn composition_shuffle_blocks_returns_cf_num_valid_permutations() {
        for &(fid, n) in &[(29u32, 3usize), (30, 3)] {
            for &dim in &[10usize, 30] {
                let blocks = composition_shuffle_blocks(fid, dim, n);
                assert_eq!(blocks.len(), n, "fid={fid} dim={dim}");
                for block in &blocks {
                    assert_eq!(block.len(), dim, "fid={fid} dim={dim}");
                    let mut sorted = block.clone();
                    sorted.sort_unstable();
                    assert_eq!(sorted, (0..dim).collect::<Vec<_>>(), "fid={fid} dim={dim}: not a permutation");
                }
            }
        }
    }

    #[test]
    fn composition_shuffle_blocks_29_dim10_first_block_matches_vendored_file() {
        // data/cec2014/shuffle_data_29_D10.txt, first 10 (of 100) tokens:
        // "1 10 4 5 2 6 3 7 8 9" (1-based) -> 0-based [0,9,3,4,1,5,2,6,7,8].
        let blocks = composition_shuffle_blocks(29, 10, 3);
        assert_eq!(blocks[0], vec![0, 9, 3, 4, 1, 5, 2, 6, 7, 8]);
    }
}
