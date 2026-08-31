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
}
