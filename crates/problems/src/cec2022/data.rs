//! Embedded CEC 2022 `input_data/` files for the basic functions (fid 1-5) --
//! vendored verbatim (byte-for-byte, `include_str!`) from the official
//! `github.com/P-N-Suganthan/2022-SO-BO` repository's `CEC2022.zip` ->
//! `CEC2022/C-Code/input_data/` (fetched directly, PROVENANCE recorded in
//! this crate's `cec2022/mod.rs` module doc and this task's report). Files
//! live at `crates/problems/data/cec2022/`, named exactly as the upstream
//! repo names them (`M_<fid>_D<dim>.txt`, `shift_data_<fid>.txt`) so they
//! stay traceable to their origin.
//!
//! ## File grammar (verified by reading `cec22_test_func.cpp`'s
//! `cec22_test_func` initializer, `func_num<9` branch -- the branch that
//! applies to every fid this module serves, 1..=5)
//!
//! - **`shift_data_<fid>.txt`**: ONE line of up to 100 whitespace-separated
//!   `f64` values (`%lf`-scanned) -- MORE than any supported `dim` needs;
//!   only the first `dim` values are the shift vector `o` for that `dim`
//!   (the C loader does exactly this: `for(i=0;i<nx;i++)
//!   fscanf(fpt,"%lf",&OShift[i]);`, i.e. it stops after `nx` values and
//!   simply never reads the rest of the line). [`shift_vector`] mirrors that:
//!   parse ALL values, then take the first `dim`.
//! - **`M_<fid>_D<dim>.txt`**: `dim` lines of `dim` whitespace-separated
//!   `f64` values each -- a `dim`x`dim` ROW-MAJOR matrix (row `i`, column
//!   `j`; the C loader reads `nx*nx` values in row-major order into a flat
//!   buffer indexed `M[i*nx+j]` by `rotatefunc`). [`rotation_matrix`]
//!   reshapes the same flat, whitespace-split stream into `Vec<Vec<f64>>`
//!   rows, matching that indexing.
//!
//! ## Extension points for T6/T7 (one line each -- do NOT extend the
//! functions below for these; the file grammar itself changes)
//!
//! - **Hybrid shuffle indices (fid 6-8)**: add `shuffle_data_<fid>_D<dim>.txt`
//!   embeds (present in the vendored `input_data/` for fid 6/7/8, dim 10/20
//!   only -- fid 6-8 have NO dim=2 data at all, see `cec2022/mod.rs`'s doc)
//!   plus a new `shuffle_indices(fid, dim) -> Vec<usize>` parser (one `int`
//!   per line via `%d`, 0-based per the C `SS` array).
//! - **Composition data (fid 9-12)**: `shift_data_9..12.txt` and
//!   `M_9..12_D<dim>.txt` use a DIFFERENT layout -- `cf_num=12` STACKED
//!   shift rows (`cf_num*nx` values total, `nx` per sub-function) and
//!   `cf_num*nx*nx` M values (one `nx`x`nx` block per sub-function) -- per
//!   `cec22_test_func.cpp`'s `else` branch (`func_num>=9`) of the same
//!   initializer this module's doc quotes above. This needs its OWN parser
//!   (e.g. `composition_shift_blocks`/`composition_rotation_blocks`), not a
//!   generalization of [`shift_vector`]/[`rotation_matrix`], which assume
//!   exactly one shift row / one M block per fid.

const SHIFT_1: &str = include_str!("../../data/cec2022/shift_data_1.txt");
const SHIFT_2: &str = include_str!("../../data/cec2022/shift_data_2.txt");
const SHIFT_3: &str = include_str!("../../data/cec2022/shift_data_3.txt");
const SHIFT_4: &str = include_str!("../../data/cec2022/shift_data_4.txt");
const SHIFT_5: &str = include_str!("../../data/cec2022/shift_data_5.txt");

const M_1_D2: &str = include_str!("../../data/cec2022/M_1_D2.txt");
const M_1_D10: &str = include_str!("../../data/cec2022/M_1_D10.txt");
const M_1_D20: &str = include_str!("../../data/cec2022/M_1_D20.txt");
const M_2_D2: &str = include_str!("../../data/cec2022/M_2_D2.txt");
const M_2_D10: &str = include_str!("../../data/cec2022/M_2_D10.txt");
const M_2_D20: &str = include_str!("../../data/cec2022/M_2_D20.txt");
const M_3_D2: &str = include_str!("../../data/cec2022/M_3_D2.txt");
const M_3_D10: &str = include_str!("../../data/cec2022/M_3_D10.txt");
const M_3_D20: &str = include_str!("../../data/cec2022/M_3_D20.txt");
const M_4_D2: &str = include_str!("../../data/cec2022/M_4_D2.txt");
const M_4_D10: &str = include_str!("../../data/cec2022/M_4_D10.txt");
const M_4_D20: &str = include_str!("../../data/cec2022/M_4_D20.txt");
const M_5_D2: &str = include_str!("../../data/cec2022/M_5_D2.txt");
const M_5_D10: &str = include_str!("../../data/cec2022/M_5_D10.txt");
const M_5_D20: &str = include_str!("../../data/cec2022/M_5_D20.txt");

/// Parse a whitespace-separated stream of `f64` values. Embedded-data parse
/// failure panics with a clear message (acceptable for `include_str!`-baked
/// constants that never vary at runtime -- same convention this crate's
/// `bbob` module implicitly relies on for its own generated data; there is
/// no user input path that can trigger this).
fn parse_floats(text: &str, source: &str) -> Vec<f64> {
    text.split_whitespace()
        .map(|tok| {
            tok.parse::<f64>().unwrap_or_else(|e| {
                panic!("cec2022 embedded data {source}: malformed float {tok:?}: {e}")
            })
        })
        .collect()
}

/// `fid`'s shift vector `o`, truncated to the first `dim` values (module
/// doc: every `shift_data_<fid>.txt` carries up to 100 values; only `dim`
/// of them are the actual shift for that `dim`). `fid` must be `1..=5`
/// (caller's responsibility -- [`crate::cec2022::Cec2022::new`] validates
/// before calling this).
pub(crate) fn shift_vector(fid: u32, dim: usize) -> Vec<f64> {
    let (text, name) = match fid {
        1 => (SHIFT_1, "shift_data_1.txt"),
        2 => (SHIFT_2, "shift_data_2.txt"),
        3 => (SHIFT_3, "shift_data_3.txt"),
        4 => (SHIFT_4, "shift_data_4.txt"),
        5 => (SHIFT_5, "shift_data_5.txt"),
        other => unreachable!("shift_vector called with unsupported fid {other}"),
    };
    let all = parse_floats(text, name);
    assert!(
        all.len() >= dim,
        "cec2022 embedded data {name}: only {} values, need at least {dim}",
        all.len()
    );
    all[..dim].to_vec()
}

/// `fid`'s `dim`x`dim` rotation matrix `M`, row-major (module doc): row `i`
/// is `matrix[i]`, `matrix[i][j]` is the C reference's `M[i*nx+j]`. `fid`
/// must be `1..=5` and `dim` one of `{2,10,20}` (caller's responsibility).
pub(crate) fn rotation_matrix(fid: u32, dim: usize) -> Vec<Vec<f64>> {
    let (text, name) = match (fid, dim) {
        (1, 2) => (M_1_D2, "M_1_D2.txt"),
        (1, 10) => (M_1_D10, "M_1_D10.txt"),
        (1, 20) => (M_1_D20, "M_1_D20.txt"),
        (2, 2) => (M_2_D2, "M_2_D2.txt"),
        (2, 10) => (M_2_D10, "M_2_D10.txt"),
        (2, 20) => (M_2_D20, "M_2_D20.txt"),
        (3, 2) => (M_3_D2, "M_3_D2.txt"),
        (3, 10) => (M_3_D10, "M_3_D10.txt"),
        (3, 20) => (M_3_D20, "M_3_D20.txt"),
        (4, 2) => (M_4_D2, "M_4_D2.txt"),
        (4, 10) => (M_4_D10, "M_4_D10.txt"),
        (4, 20) => (M_4_D20, "M_4_D20.txt"),
        (5, 2) => (M_5_D2, "M_5_D2.txt"),
        (5, 10) => (M_5_D10, "M_5_D10.txt"),
        (5, 20) => (M_5_D20, "M_5_D20.txt"),
        (other_fid, other_dim) => {
            unreachable!("rotation_matrix called with unsupported (fid={other_fid}, dim={other_dim})")
        }
    };
    let flat = parse_floats(text, name);
    assert_eq!(
        flat.len(),
        dim * dim,
        "cec2022 embedded data {name}: expected {} values ({dim}x{dim}), got {}",
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
    // crates/problems/data/cec2022/, byte-for-byte from the official repo).

    #[test]
    fn shift_vector_1_dim2_matches_vendored_file() {
        // data/cec2022/shift_data_1.txt, first two whitespace-separated
        // tokens: "-5.5938326705218444e+01   4.5430653935964642e+00 ...".
        let o = shift_vector(1, 2);
        assert_eq!(o, vec![-5.5938326705218444e+01, 4.5430653935964642e+00]);
    }

    #[test]
    fn shift_vector_3_dim2_matches_vendored_file() {
        // data/cec2022/shift_data_3.txt, first two tokens:
        // "7.9089392944746379e+01  -2.4572777647791789e+01 ...".
        let o = shift_vector(3, 2);
        assert_eq!(o, vec![7.9089392944746379e+01, -2.4572777647791789e+01]);
    }

    #[test]
    fn shift_vector_1_dim10_takes_only_first_ten_of_the_100_column_file() {
        // data/cec2022/shift_data_1.txt carries 100 values (module doc); at
        // dim=10 only the first 10 are the shift vector -- spot-check the
        // 10th (last taken) value against the raw file's 10th token:
        // "... -6.1053557089089210e+01  -4.7133996322347784e+01  ..." (9th,
        // 10th tokens).
        let o = shift_vector(1, 10);
        assert_eq!(o.len(), 10);
        assert_eq!(o[8], -6.1053557089089210e+01);
        assert_eq!(o[9], -4.7133996322347784e+01);
    }

    #[test]
    fn rotation_matrix_1_dim2_matches_vendored_file() {
        // data/cec2022/M_1_D2.txt, two rows of two values each:
        //    9.9965877537837600e-01  -2.6121500894966159e-02
        //   -2.6121500894966111e-02  -9.9965877537837589e-01
        let m = rotation_matrix(1, 2);
        assert_eq!(
            m,
            vec![
                vec![9.9965877537837600e-01, -2.6121500894966159e-02],
                vec![-2.6121500894966111e-02, -9.9965877537837589e-01],
            ]
        );
    }

    #[test]
    fn rotation_matrix_1_dim10_row_major_spot_checks() {
        // data/cec2022/M_1_D10.txt: row 1 starts
        // "5.5875251571865003e-01  -3.4947173920977476e-01 ..."; row 10
        // (last) starts "-3.4170279585135427e-01  -4.9858688714161425e-01
        // ...". Confirms row-major reshape (module doc): matrix[0][0] is
        // the file's very first token, matrix[9][0] is the first token of
        // the file's LAST line, not e.g. the 91st flat value misindexed.
        let m = rotation_matrix(1, 10);
        assert_eq!(m.len(), 10);
        assert_eq!(m[0].len(), 10);
        assert_eq!(m[0][0], 5.5875251571865003e-01);
        assert_eq!(m[0][1], -3.4947173920977476e-01);
        assert_eq!(m[9][0], -3.4170279585135427e-01);
        assert_eq!(m[9][1], -4.9858688714161425e-01);
    }

    #[test]
    fn all_fid_1_to_5_dims_2_10_20_parse_without_panic() {
        for fid in 1u32..=5 {
            for &dim in &[2usize, 10, 20] {
                let o = shift_vector(fid, dim);
                assert_eq!(o.len(), dim, "fid={fid} dim={dim}");
                let m = rotation_matrix(fid, dim);
                assert_eq!(m.len(), dim, "fid={fid} dim={dim}");
                assert!(m.iter().all(|row| row.len() == dim), "fid={fid} dim={dim}");
            }
        }
    }
}
