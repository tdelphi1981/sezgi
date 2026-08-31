//! Embedded CEC 2017 `input_data/` files for fid 1, 3-10 (this task's
//! functional scope: unimodal F1/F3, simple-multimodal F4-F10) -- vendored
//! verbatim (byte-for-byte, `include_str!`) from the official
//! `github.com/P-N-Suganthan/CEC2017-BoundContrained` repository's
//! `codes.rar` -> `codes/C version/input_data/` (fetched directly,
//! PROVENANCE recorded in this crate's `cec2017/mod.rs` module doc and this
//! task's report). Files live at `crates/problems/data/cec2017/`, named
//! exactly as the upstream repo names them (`M_<fid>_D<dim>.txt`,
//! `shift_data_<fid>.txt`), mirroring `cec2014/data.rs`'s convention this
//! module is siblings with.
//!
//! ## Vendoring scope (M3-6 T6, user-approved): dims {10, 30}, the FULL
//! usable suite's files copied to `crates/problems/data/cec2017/` -- every
//! `fid` in `{1,3..=30}` (fid 2 excluded: permanently withdrawn, module
//! doc's F2 ruling; its `input_data` files are dead weight the reference C
//! never usefully reads, so they are not vendored at all) -- so T7 (fid
//! 11-20) and T8 (fid 21-30) do not need to re-fetch or re-copy anything.
//! Measured total (independently cross-checked with both `find -exec stat`
//! and a Python `os.path.getsize` sum during the copy step): **111 files,
//! 3,285,318 bytes** -- every `shift_data_<fid>.txt` for `fid` in
//! `{1,3..=30}` (dimension-independent, 29 files), every
//! `M_<fid>_D{10,30}.txt` for `fid` in `{1,3..=30}` (58 files), and
//! `shuffle_data_<fid>_D{10,30}.txt` ONLY for the fids the reference C's own
//! loader ever reads shuffle data for -- `fid` in `{11..=20, 29, 30}` (24
//! files; `cec17_test_func.cpp`'s loader, quoted in `mod.rs`'s module doc,
//! reads shuffle data exactly for `func_num>=11&&func_num<=20` or
//! `func_num==29||func_num==30`). This module's `include_str!`/parser
//! functions cover ONLY fid `{1,3..=10}` (this task's functional scope);
//! fid 11-30's already-vendored `.txt` files sit unreferenced on disk until
//! T7/T8 add their own `include_str!` consts and parsers here, mirroring
//! `cec2014/data.rs`'s own T2 -> T3 -> T4 staging note.
//!
//! ## File grammar (verified by reading `cec17_test_func.cpp`'s
//! `cec17_test_func` initializer, `func_num<20` branch -- the branch that
//! applies to every fid this module serves, byte-identical grammar to
//! `cec2014/data.rs`'s fid-1-16 section since both suites share the same
//! reference-C loader lineage)
//!
//! - **`shift_data_<fid>.txt`**: ONE line of 100 whitespace-separated `f64`
//!   values (`%lf`-scanned in the fixed-up C, module doc's `mod.rs`
//!   portability-edit note -- the vendored source itself reads `%Lf`, a
//!   verified bug) -- MORE than any supported `dim` needs; only the first
//!   `dim` values are the shift vector `o` for that `dim` (the C loader does
//!   exactly this: `for(i=0;i<nx;i++) fscanf(fpt,"%lf",&OShift[i]);`).
//!   [`shift_vector`] mirrors that: parse ALL values, then take the first
//!   `dim`.
//! - **`M_<fid>_D<dim>.txt`**: `dim` lines of `dim` whitespace-separated
//!   `f64` values each -- a `dim`x`dim` ROW-MAJOR matrix (row `i`, column
//!   `j`; the C loader reads `nx*nx` values in row-major order into a flat
//!   buffer indexed `M[i*nx+j]` by `rotatefunc`). [`rotation_matrix`]
//!   reshapes the same flat, whitespace-split stream into `Vec<Vec<f64>>`
//!   rows, matching that indexing.

const SHIFT_1: &str = include_str!("../../data/cec2017/shift_data_1.txt");
const SHIFT_3: &str = include_str!("../../data/cec2017/shift_data_3.txt");
const SHIFT_4: &str = include_str!("../../data/cec2017/shift_data_4.txt");
const SHIFT_5: &str = include_str!("../../data/cec2017/shift_data_5.txt");
const SHIFT_6: &str = include_str!("../../data/cec2017/shift_data_6.txt");
const SHIFT_7: &str = include_str!("../../data/cec2017/shift_data_7.txt");
const SHIFT_8: &str = include_str!("../../data/cec2017/shift_data_8.txt");
const SHIFT_9: &str = include_str!("../../data/cec2017/shift_data_9.txt");
const SHIFT_10: &str = include_str!("../../data/cec2017/shift_data_10.txt");

const M_1_D10: &str = include_str!("../../data/cec2017/M_1_D10.txt");
const M_1_D30: &str = include_str!("../../data/cec2017/M_1_D30.txt");
const M_3_D10: &str = include_str!("../../data/cec2017/M_3_D10.txt");
const M_3_D30: &str = include_str!("../../data/cec2017/M_3_D30.txt");
const M_4_D10: &str = include_str!("../../data/cec2017/M_4_D10.txt");
const M_4_D30: &str = include_str!("../../data/cec2017/M_4_D30.txt");
const M_5_D10: &str = include_str!("../../data/cec2017/M_5_D10.txt");
const M_5_D30: &str = include_str!("../../data/cec2017/M_5_D30.txt");
const M_6_D10: &str = include_str!("../../data/cec2017/M_6_D10.txt");
const M_6_D30: &str = include_str!("../../data/cec2017/M_6_D30.txt");
const M_7_D10: &str = include_str!("../../data/cec2017/M_7_D10.txt");
const M_7_D30: &str = include_str!("../../data/cec2017/M_7_D30.txt");
const M_8_D10: &str = include_str!("../../data/cec2017/M_8_D10.txt");
const M_8_D30: &str = include_str!("../../data/cec2017/M_8_D30.txt");
const M_9_D10: &str = include_str!("../../data/cec2017/M_9_D10.txt");
const M_9_D30: &str = include_str!("../../data/cec2017/M_9_D30.txt");
const M_10_D10: &str = include_str!("../../data/cec2017/M_10_D10.txt");
const M_10_D30: &str = include_str!("../../data/cec2017/M_10_D30.txt");

/// Parse a whitespace-separated stream of `f64` values. Embedded-data parse
/// failure panics with a clear message (acceptable for `include_str!`-baked
/// constants that never vary at runtime -- same convention `cec2014/data.rs`
/// uses).
fn parse_floats(text: &str, source: &str) -> Vec<f64> {
    text.split_whitespace()
        .map(|tok| {
            tok.parse::<f64>().unwrap_or_else(|e| {
                panic!("cec2017 embedded data {source}: malformed float {tok:?}: {e}")
            })
        })
        .collect()
}

/// `fid`'s shift vector `o`, truncated to the first `dim` values (module
/// doc: every `shift_data_<fid>.txt` carries 100 values -- only `dim` of
/// them are the actual shift for that `dim`). `fid` must be `1` or `3..=10`
/// (caller's responsibility -- [`crate::cec2017::Cec2017::new`] validates
/// before calling this).
pub(crate) fn shift_vector(fid: u32, dim: usize) -> Vec<f64> {
    let (text, name) = match fid {
        1 => (SHIFT_1, "shift_data_1.txt"),
        3 => (SHIFT_3, "shift_data_3.txt"),
        4 => (SHIFT_4, "shift_data_4.txt"),
        5 => (SHIFT_5, "shift_data_5.txt"),
        6 => (SHIFT_6, "shift_data_6.txt"),
        7 => (SHIFT_7, "shift_data_7.txt"),
        8 => (SHIFT_8, "shift_data_8.txt"),
        9 => (SHIFT_9, "shift_data_9.txt"),
        10 => (SHIFT_10, "shift_data_10.txt"),
        other => unreachable!("shift_vector called with unsupported fid {other}"),
    };
    let all = parse_floats(text, name);
    assert!(
        all.len() >= dim,
        "cec2017 embedded data {name}: only {} values, need at least {dim}",
        all.len()
    );
    all[..dim].to_vec()
}

/// `fid`'s `dim`x`dim` rotation matrix `M`, row-major (module doc): row `i`
/// is `matrix[i]`, `matrix[i][j]` is the C reference's `M[i*nx+j]`. `fid`
/// must be `1` or `3..=10`, `dim` one of `{10,30}` (caller's responsibility
/// -- `Cec2017::new` validates first).
pub(crate) fn rotation_matrix(fid: u32, dim: usize) -> Vec<Vec<f64>> {
    let (text, name) = match (fid, dim) {
        (1, 10) => (M_1_D10, "M_1_D10.txt"),
        (1, 30) => (M_1_D30, "M_1_D30.txt"),
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
        (other_fid, other_dim) => {
            unreachable!("rotation_matrix called with unsupported (fid={other_fid}, dim={other_dim})")
        }
    };
    let flat = parse_floats(text, name);
    assert_eq!(
        flat.len(),
        dim * dim,
        "cec2017 embedded data {name}: expected {} values ({dim}x{dim}), got {}",
        dim * dim,
        flat.len()
    );
    (0..dim).map(|i| flat[i * dim..(i + 1) * dim].to_vec()).collect()
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
    // crates/problems/data/cec2017/, byte-for-byte from the official repo).

    #[test]
    fn shift_vector_1_dim10_matches_vendored_file() {
        // data/cec2017/shift_data_1.txt, first two whitespace-separated
        // tokens: "-5.5276398498228005e+01  -7.0429559718086182e+01 ...".
        let o = shift_vector(1, 10);
        assert_eq!(o.len(), 10);
        assert_eq!(o[0], -5.5276398498228005e+01);
        assert_eq!(o[1], -7.0429559718086182e+01);
    }

    #[test]
    fn shift_vector_6_dim30_takes_only_first_30_of_the_100_column_file() {
        // data/cec2017/shift_data_6.txt carries 100 values (module doc); at
        // dim=30 only the first 30 are the shift vector.
        let o = shift_vector(6, 30);
        assert_eq!(o.len(), 30);
    }

    #[test]
    fn rotation_matrix_1_dim10_row_major_spot_checks() {
        // data/cec2017/M_1_D10.txt row 1 (index 0) starts
        // -6.0130701301896017e-01; row 2 (index 1) second token (index 1) is
        // 9.2845625214477390e-01. Confirms row-major reshape (module doc):
        // matrix[1][1] is the SECOND line's SECOND token, not e.g. an
        // 11th-flat-value misindex.
        let m = rotation_matrix(1, 10);
        assert_eq!(m.len(), 10);
        assert_eq!(m[0].len(), 10);
        assert_eq!(m[0][0], -6.0130701301896017e-01);
        assert_eq!(m[1][1], 9.2845625214477390e-01);
    }

    #[test]
    fn all_fid_1_3_to_10_dims_10_30_parse_without_panic() {
        for fid in [1u32, 3, 4, 5, 6, 7, 8, 9, 10] {
            for &dim in &[10usize, 30] {
                let o = shift_vector(fid, dim);
                assert_eq!(o.len(), dim, "fid={fid} dim={dim}");
                let m = rotation_matrix(fid, dim);
                assert_eq!(m.len(), dim, "fid={fid} dim={dim}");
                assert!(m.iter().all(|row| row.len() == dim), "fid={fid} dim={dim}");
            }
        }
    }
}
