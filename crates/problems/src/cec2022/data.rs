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
//! ## T6 addendum: hybrid data (fid 6-8, dim 10/20 only)
//!
//! `shift_data_{6,7,8}.txt` follow the SAME single-shift-vector grammar as
//! fid 1-5 above (only the first `dim` of up to 100 whitespace-separated
//! values matter) -- [`shift_vector`] therefore needed NO changes, just new
//! `fid` match arms. One file-shape surprise worth recording precisely:
//! `shift_data_8.txt` is laid out as 10 lines of 100 values each (`wc -l` ->
//! 10, vs. `shift_data_6.txt`/`shift_data_7.txt`'s single line) -- but since
//! the C loader's `fscanf(fpt,"%lf",&OShift[i])` loop (quoted above) reads
//! `nx` whitespace-delimited tokens off the stream regardless of newlines,
//! and `dim` is always `<=20 < 100`, only line 1 is EVER read for fid 8 at
//! any supported dim; lines 2-10 are vendored (byte-for-byte, unmodified)
//! but dead weight, exactly mirroring the reference's own read pattern.
//! `M_{6,7,8}_D{10,20}.txt` follow the SAME `dim`x`dim` row-major grammar as
//! fid 1-5's -- [`rotation_matrix`] needed only new `fid` match arms too.
//! `M_*_D2.txt` for fid 6-8 are deliberately NOT vendored (`cec2022/mod.rs`'s
//! doc: fid 6-8 reject `dim==2` uniformly regardless of which per-fid M file
//! happens to exist at dim=2, so no dim=2 data is needed here at all).
//!
//! - **Hybrid shuffle indices (fid 6-8)**: `shuffle_data_<fid>_D<dim>.txt`
//!   embeds (vendored for fid 6/7/8, dim 10/20 only -- dim=2 shuffle data is
//!   uniformly ABSENT upstream for fid 6-8, see `cec2022/mod.rs`'s doc) --
//!   ONE line of `dim` whitespace-separated 1-based `int`s (`%d`-scanned by
//!   the C loader, `for(i=0;i<nx;i++) fscanf(fpt,"%d",&SS[i]);`), a
//!   permutation of `1..=dim`. [`shuffle_indices`] parses this and converts
//!   to 0-based (subtracting 1 per element) so callers can index directly
//!   (matches the C's own `y[i]=z[S[i]-1]`, `cec22_test_func.cpp`'s `hf02`/
//!   `hf06`/`hf10`, quoted in `cec2022/mod.rs`'s doc).
//!
//! ## T7 addendum: composition data (fid 9-12) -- a DIFFERENT layout from
//! fid 1-8's, needing its own parsers ([`composition_shift_blocks`] /
//! [`composition_rotation_blocks`], not a generalization of
//! [`shift_vector`]/[`rotation_matrix`], which assume exactly one shift row
//! / one M block per fid)
//!
//! `cec22_test_func.cpp`'s loader (`func_num>=9` branch, this module's T5
//! sibling doc quotes the surrounding `else`) allocates and attempts to fill
//! a HARDCODED `cf_num=12`-sized buffer for BOTH `OShift`
//! (`nx*cf_num` doubles, one row of up to 100 raw tokens per slot, module
//! doc's fid-1-8 "up to 100 values, only the first `dim` matter" grammar
//! extended per-row) and `M` (`cf_num*nx*nx` doubles) -- `12` being a
//! single hardcoded local (`int cf_num=12,i,j;`, the function's very first
//! line) shared by ALL FOUR composition dispatch cases, NOT any individual
//! composition function's own `cf_num` (5 for `cf01`/fid9, 3 for
//! `cf02`/fid10, 5 for `cf06`/fid11, 6 for `cf07`/fid12 -- each hardcoded
//! locally INSIDE that function, this module doc's T7 section, and
//! genuinely different from the loader's `12`).
//!
//! **Verified directly against the vendored files (not assumed from the
//! loader's `12`)**: every `shift_data_9..12.txt` carries exactly 10 lines
//! of 100 tokens each (`wc`/`awk` counted, this task's report), and every
//! `M_9..12_D<dim>.txt` carries exactly 10 stacked `dim`x`dim` blocks
//! (`M_9_D10.txt`: 100 lines of 10 values = 1000 = 10*10*10; `M_9_D2.txt`:
//! 16 lines of 2 values = 32 = 8*2*2, i.e. 8 blocks at dim=2, still >= the
//! max `cf_num` any fid 9-12 function needs) -- NEITHER file provides the
//! full 12 slots/blocks the loader's hardcoded `12` would read. This is
//! harmless in practice: the C loader's 11th/12th-slot reads run past EOF
//! (`fscanf` silently fails, leaving unread `malloc` memory untouched) but
//! that garbage is NEVER accessed -- `cf01`/`cf02`/`cf06`/`cf07` each index
//! only `Os[0..cf_num_actual*nx]` / `Mr[0..cf_num_actual*nx*nx]` with their
//! OWN small `cf_num` (max 6, for `cf07`/fid12), always well inside the 8-10
//! blocks the files actually provide. [`composition_shift_blocks`] /
//! [`composition_rotation_blocks`] below therefore take the CALLER's actual
//! `cf_num` (module doc's `Cec2022::composition_spec` table, T7) and parse
//! only that many rows/blocks -- not a fixed 12 -- matching what the C
//! genuinely reads (as opposed to what it over-allocates for).
//!
//! Grammar per block, once the block boundary is known: shift rows follow
//! fid-1-8's exact per-row grammar (`dim` of up to 100 whitespace-separated
//! values per LINE, one line per component -- [`composition_shift_blocks`]
//! therefore parses line-by-line, unlike [`shift_vector`]'s single
//! whitespace-flattened stream, so row boundaries are preserved); `M` blocks
//! are `dim`x`dim` row-major, stacked with NO padding between blocks (no
//! per-row skip needed, unlike the shift file's 100-column dead-weight
//! columns) -- [`composition_rotation_blocks`] slices the flat parsed stream
//! directly into `cf_num` contiguous `dim*dim`-sized chunks.

const SHIFT_1: &str = include_str!("../../data/cec2022/shift_data_1.txt");
const SHIFT_2: &str = include_str!("../../data/cec2022/shift_data_2.txt");
const SHIFT_3: &str = include_str!("../../data/cec2022/shift_data_3.txt");
const SHIFT_4: &str = include_str!("../../data/cec2022/shift_data_4.txt");
const SHIFT_5: &str = include_str!("../../data/cec2022/shift_data_5.txt");
const SHIFT_6: &str = include_str!("../../data/cec2022/shift_data_6.txt");
const SHIFT_7: &str = include_str!("../../data/cec2022/shift_data_7.txt");
const SHIFT_8: &str = include_str!("../../data/cec2022/shift_data_8.txt");

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
const M_6_D10: &str = include_str!("../../data/cec2022/M_6_D10.txt");
const M_6_D20: &str = include_str!("../../data/cec2022/M_6_D20.txt");
const M_7_D10: &str = include_str!("../../data/cec2022/M_7_D10.txt");
const M_7_D20: &str = include_str!("../../data/cec2022/M_7_D20.txt");
const M_8_D10: &str = include_str!("../../data/cec2022/M_8_D10.txt");
const M_8_D20: &str = include_str!("../../data/cec2022/M_8_D20.txt");

const SHIFT_9: &str = include_str!("../../data/cec2022/shift_data_9.txt");
const SHIFT_10: &str = include_str!("../../data/cec2022/shift_data_10.txt");
const SHIFT_11: &str = include_str!("../../data/cec2022/shift_data_11.txt");
const SHIFT_12: &str = include_str!("../../data/cec2022/shift_data_12.txt");

const M_9_D2: &str = include_str!("../../data/cec2022/M_9_D2.txt");
const M_9_D10: &str = include_str!("../../data/cec2022/M_9_D10.txt");
const M_9_D20: &str = include_str!("../../data/cec2022/M_9_D20.txt");
const M_10_D2: &str = include_str!("../../data/cec2022/M_10_D2.txt");
const M_10_D10: &str = include_str!("../../data/cec2022/M_10_D10.txt");
const M_10_D20: &str = include_str!("../../data/cec2022/M_10_D20.txt");
const M_11_D2: &str = include_str!("../../data/cec2022/M_11_D2.txt");
const M_11_D10: &str = include_str!("../../data/cec2022/M_11_D10.txt");
const M_11_D20: &str = include_str!("../../data/cec2022/M_11_D20.txt");
const M_12_D2: &str = include_str!("../../data/cec2022/M_12_D2.txt");
const M_12_D10: &str = include_str!("../../data/cec2022/M_12_D10.txt");
const M_12_D20: &str = include_str!("../../data/cec2022/M_12_D20.txt");

const SHUFFLE_6_D10: &str = include_str!("../../data/cec2022/shuffle_data_6_D10.txt");
const SHUFFLE_6_D20: &str = include_str!("../../data/cec2022/shuffle_data_6_D20.txt");
const SHUFFLE_7_D10: &str = include_str!("../../data/cec2022/shuffle_data_7_D10.txt");
const SHUFFLE_7_D20: &str = include_str!("../../data/cec2022/shuffle_data_7_D20.txt");
const SHUFFLE_8_D10: &str = include_str!("../../data/cec2022/shuffle_data_8_D10.txt");
const SHUFFLE_8_D20: &str = include_str!("../../data/cec2022/shuffle_data_8_D20.txt");

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
/// doc: every `shift_data_<fid>.txt` carries up to 100 values -- or, for
/// fid 8, up to 100 values PER LINE across 10 lines, module doc's T6
/// addendum -- only `dim` of them are the actual shift for that `dim`).
/// `fid` must be `1..=8` (caller's responsibility --
/// [`crate::cec2022::Cec2022::new`] validates before calling this).
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
/// must be `1..=8` and `dim` one of `{2,10,20}` for fid 1-5, `{10,20}` for
/// fid 6-8 (caller's responsibility -- `Cec2022::new` rejects `(6..=8, 2)`
/// before this is ever called for those fids, module doc's T6 addendum: no
/// `M_{6,7,8}_D2.txt` is vendored).
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
        (6, 10) => (M_6_D10, "M_6_D10.txt"),
        (6, 20) => (M_6_D20, "M_6_D20.txt"),
        (7, 10) => (M_7_D10, "M_7_D10.txt"),
        (7, 20) => (M_7_D20, "M_7_D20.txt"),
        (8, 10) => (M_8_D10, "M_8_D10.txt"),
        (8, 20) => (M_8_D20, "M_8_D20.txt"),
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

/// `fid`'s (`6..=8`) shuffle permutation for `dim` (`{10,20}` only, module
/// doc's T6 addendum), already converted to 0-based indices (the vendored
/// file's `int`s are 1-based, module doc: mirrors the C reference's own
/// `y[i]=z[S[i]-1]` -- subtracting 1 here means callers can write
/// `y[i] = z[shuffle[i]]` directly, no further `-1` needed). Caller's
/// responsibility to only call this for `fid in 6..=8`, `dim in {10,20}`
/// ([`crate::cec2022::Cec2022::new`] validates first).
pub(crate) fn shuffle_indices(fid: u32, dim: usize) -> Vec<usize> {
    let (text, name) = match (fid, dim) {
        (6, 10) => (SHUFFLE_6_D10, "shuffle_data_6_D10.txt"),
        (6, 20) => (SHUFFLE_6_D20, "shuffle_data_6_D20.txt"),
        (7, 10) => (SHUFFLE_7_D10, "shuffle_data_7_D10.txt"),
        (7, 20) => (SHUFFLE_7_D20, "shuffle_data_7_D20.txt"),
        (8, 10) => (SHUFFLE_8_D10, "shuffle_data_8_D10.txt"),
        (8, 20) => (SHUFFLE_8_D20, "shuffle_data_8_D20.txt"),
        (other_fid, other_dim) => {
            unreachable!("shuffle_indices called with unsupported (fid={other_fid}, dim={other_dim})")
        }
    };
    let ints: Vec<i64> = text
        .split_whitespace()
        .map(|tok| {
            tok.parse::<i64>().unwrap_or_else(|e| {
                panic!("cec2022 embedded data {name}: malformed int {tok:?}: {e}")
            })
        })
        .collect();
    assert_eq!(
        ints.len(),
        dim,
        "cec2022 embedded data {name}: expected {dim} 1-based shuffle indices, got {}",
        ints.len()
    );
    ints.iter()
        .map(|&one_based| {
            assert!(
                (1..=dim as i64).contains(&one_based),
                "cec2022 embedded data {name}: shuffle index {one_based} out of range 1..={dim}"
            );
            (one_based - 1) as usize
        })
        .collect()
}

/// `fid`'s (`9..=12`) first `cf_num` component shift rows, each truncated to
/// the first `dim` values (module doc's T7 addendum: the vendored file has
/// 10 stacked 100-wide rows -- `cf_num` (caller's, 3/5/5/6 -- always `<=6`)
/// is the number this composition function itself actually uses). `fid`
/// must be `9..=12` (caller's responsibility -- `Cec2022::new` validates).
pub(crate) fn composition_shift_blocks(fid: u32, dim: usize, cf_num: usize) -> Vec<Vec<f64>> {
    let (text, name) = match fid {
        9 => (SHIFT_9, "shift_data_9.txt"),
        10 => (SHIFT_10, "shift_data_10.txt"),
        11 => (SHIFT_11, "shift_data_11.txt"),
        12 => (SHIFT_12, "shift_data_12.txt"),
        other => unreachable!("composition_shift_blocks called with unsupported fid {other}"),
    };
    let rows: Vec<&str> = text.lines().filter(|line| !line.trim().is_empty()).collect();
    assert!(
        rows.len() >= cf_num,
        "cec2022 embedded data {name}: only {} shift rows, need at least {cf_num}",
        rows.len()
    );
    rows[..cf_num]
        .iter()
        .map(|line| {
            let vals = parse_floats(line, name);
            assert!(
                vals.len() >= dim,
                "cec2022 embedded data {name}: shift row has only {} values, need at least {dim}",
                vals.len()
            );
            vals[..dim].to_vec()
        })
        .collect()
}

/// `fid`'s (`9..=12`) first `cf_num` component `dim`x`dim` rotation
/// matrices, row-major each block (module doc's T7 addendum: the vendored
/// file stacks 10 `dim`x`dim` blocks with no padding between them --
/// `cf_num` (caller's, always `<=6`) is the number this composition
/// function itself actually uses). `fid` must be `9..=12`, `dim` one of
/// `{2,10,20}` (caller's responsibility -- `Cec2022::new` validates).
pub(crate) fn composition_rotation_blocks(fid: u32, dim: usize, cf_num: usize) -> Vec<Vec<Vec<f64>>> {
    let (text, name) = match (fid, dim) {
        (9, 2) => (M_9_D2, "M_9_D2.txt"),
        (9, 10) => (M_9_D10, "M_9_D10.txt"),
        (9, 20) => (M_9_D20, "M_9_D20.txt"),
        (10, 2) => (M_10_D2, "M_10_D2.txt"),
        (10, 10) => (M_10_D10, "M_10_D10.txt"),
        (10, 20) => (M_10_D20, "M_10_D20.txt"),
        (11, 2) => (M_11_D2, "M_11_D2.txt"),
        (11, 10) => (M_11_D10, "M_11_D10.txt"),
        (11, 20) => (M_11_D20, "M_11_D20.txt"),
        (12, 2) => (M_12_D2, "M_12_D2.txt"),
        (12, 10) => (M_12_D10, "M_12_D10.txt"),
        (12, 20) => (M_12_D20, "M_12_D20.txt"),
        (other_fid, other_dim) => unreachable!(
            "composition_rotation_blocks called with unsupported (fid={other_fid}, dim={other_dim})"
        ),
    };
    let flat = parse_floats(text, name);
    let block_len = dim * dim;
    assert!(
        flat.len() >= cf_num * block_len,
        "cec2022 embedded data {name}: only {} values, need at least {} ({cf_num} blocks of {dim}x{dim})",
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

    // ---- T6 addendum: fid 6-8 (hybrid) embedded data ----

    #[test]
    fn shift_vector_6_dim2_matches_vendored_file() {
        // data/cec2022/shift_data_6.txt, first two tokens:
        // "1.8154450402667024e+01  1.8479176246784164e+00 ...".
        let o = shift_vector(6, 2);
        assert_eq!(o, vec![1.8154450402667024e+01, 1.8479176246784164e+00]);
    }

    #[test]
    fn shift_vector_7_dim2_matches_vendored_file() {
        // data/cec2022/shift_data_7.txt, first two tokens:
        // "-3.0590574643849664e+01  -5.5788039181282763e+01 ...".
        let o = shift_vector(7, 2);
        assert_eq!(o, vec![-3.0590574643849664e+01, -5.5788039181282763e+01]);
    }

    #[test]
    fn shift_vector_8_dim10_reads_only_line1_of_the_10x100_file() {
        // data/cec2022/shift_data_8.txt has 10 LINES of 100 values each
        // (module doc's T6 addendum -- a file-shape surprise unique to fid
        // 8) but the C loader's fscanf loop reads `dim` tokens off the
        // stream regardless of newlines, so at dim=10 only line 1's first
        // 10 values are ever used. Spot-check against line 1's own first
        // and 10th tokens (transcribed from the raw file).
        let o = shift_vector(8, 10);
        assert_eq!(o.len(), 10);
        assert_eq!(o[0], -1.4386804758751207e+01);
        // 10th token of line 1 (index 9), NOT line 2's first token --
        // confirms the "line breaks don't matter, only token count does"
        // reading this module doc describes.
        assert_eq!(o[9], 4.1910300289238947e+01);
    }

    #[test]
    fn rotation_matrix_6_dim10_row_major_spot_check() {
        // data/cec2022/M_6_D10.txt row 1: "-3.0784412248344223e-001  0 ...
        // -5.5680434188808448e-001"; row 2 starts "0 5.0035979531831165e-001
        // ...".
        let m = rotation_matrix(6, 10);
        assert_eq!(m.len(), 10);
        assert_eq!(m[0][0], -3.0784412248344223e-001);
        assert_eq!(m[0][9], -5.5680434188808448e-001);
        assert_eq!(m[1][1], 5.0035979531831165e-001);
    }

    #[test]
    fn shuffle_indices_6_dim10_matches_vendored_file_0_based() {
        // data/cec2022/shuffle_data_6_D10.txt: "4 7 9 3 5 2 10 8 6 1"
        // (1-based); [`shuffle_indices`] subtracts 1 from each.
        let s = shuffle_indices(6, 10);
        assert_eq!(s, vec![3, 6, 8, 2, 4, 1, 9, 7, 5, 0]);
    }

    #[test]
    fn shuffle_indices_7_dim10_matches_vendored_file_0_based() {
        // data/cec2022/shuffle_data_7_D10.txt: "10 9 7 6 3 2 8 5 4 1".
        let s = shuffle_indices(7, 10);
        assert_eq!(s, vec![9, 8, 6, 5, 2, 1, 7, 4, 3, 0]);
    }

    #[test]
    fn shuffle_indices_6_dim20_matches_vendored_file_0_based() {
        // data/cec2022/shuffle_data_6_D20.txt:
        // "5 11 12 20 14 19 1 13 18 6 17 16 4 15 3 2 8 9 10 7".
        let s = shuffle_indices(6, 20);
        assert_eq!(
            s,
            vec![4, 10, 11, 19, 13, 18, 0, 12, 17, 5, 16, 15, 3, 14, 2, 1, 7, 8, 9, 6]
        );
    }

    #[test]
    fn shuffle_indices_is_a_permutation_of_0_dot_dot_dim_for_every_fid_6_to_8_and_dim() {
        for fid in 6u32..=8 {
            for &dim in &[10usize, 20] {
                let mut s = shuffle_indices(fid, dim);
                s.sort_unstable();
                assert_eq!(s, (0..dim).collect::<Vec<_>>(), "fid={fid} dim={dim}");
            }
        }
    }

    #[test]
    fn all_fid_6_to_8_dims_10_20_parse_without_panic() {
        for fid in 6u32..=8 {
            for &dim in &[10usize, 20] {
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

    // ---- T7 addendum: fid 9-12 (composition) embedded data ----

    #[test]
    fn composition_shift_blocks_9_dim2_rows_0_and_1_match_vendored_file() {
        // data/cec2022/shift_data_9.txt, first two tokens of line 1 and
        // line 2:
        //   line1: "5.2898156995371991e+01  -2.5374962177767859e+00 ..."
        //   line2: "6.7324621217010829e+01   5.7101767141885020e+01 ..."
        let blocks = composition_shift_blocks(9, 2, 5);
        assert_eq!(blocks.len(), 5);
        assert_eq!(blocks[0], vec![5.2898156995371991e+01, -2.5374962177767859e+00]);
        assert_eq!(blocks[1], vec![6.7324621217010829e+01, 5.7101767141885020e+01]);
    }

    #[test]
    fn composition_shift_blocks_10_11_12_dim2_row0_matches_vendored_file() {
        // First token of line 1 of shift_data_10/11/12.txt.
        assert_eq!(composition_shift_blocks(10, 2, 3)[0][0], 6.1770164235388009e+01);
        assert_eq!(composition_shift_blocks(11, 2, 5)[0][0], 4.9178999700201210e+01);
        assert_eq!(composition_shift_blocks(12, 2, 6)[0][0], -6.2391809693720106e+01);
    }

    #[test]
    fn composition_rotation_blocks_9_dim2_block_boundaries_match_vendored_file() {
        // data/cec2022/M_9_D2.txt, first two 2x2 blocks (4 lines):
        //   block0 row0: "-1.2107125975322155e+000  -5.1761149145075969e-001"
        //   block0 row1: " 2.5080932042014417e+000  -1.4056037240749819e+000"
        //   block1 row0: "-9.8943637273985763e-001   1.4496780435460016e-001"
        //   block1 row1: "-1.4496780435460044e-001  -9.8943637273985763e-001"
        // Confirms blocks are read CONTIGUOUSLY (no padding between them,
        // unlike the shift file's 100-column dead weight) -- block 1 starts
        // exactly at flat index 4, not e.g. 100 (a wrong "each block is
        // still 100-wide" reading would misindex here).
        let blocks = composition_rotation_blocks(9, 2, 5);
        assert_eq!(blocks.len(), 5);
        assert_eq!(blocks[0], vec![
            vec![-1.2107125975322155e+000, -5.1761149145075969e-001],
            vec![2.5080932042014417e+000, -1.4056037240749819e+000],
        ]);
        assert_eq!(blocks[1], vec![
            vec![-9.8943637273985763e-001, 1.4496780435460016e-001],
            vec![-1.4496780435460044e-001, -9.8943637273985763e-001],
        ]);
    }

    #[test]
    fn all_fid_9_to_12_dims_2_10_20_parse_without_panic() {
        // cf_num per fid (module doc's T7 `composition_spec` table): fid 9
        // -> 5, fid 10 -> 3, fid 11 -> 5, fid 12 -> 6.
        for &(fid, cf_num) in &[(9u32, 5usize), (10, 3), (11, 5), (12, 6)] {
            for &dim in &[2usize, 10, 20] {
                let shifts = composition_shift_blocks(fid, dim, cf_num);
                assert_eq!(shifts.len(), cf_num, "fid={fid} dim={dim}");
                assert!(shifts.iter().all(|row| row.len() == dim), "fid={fid} dim={dim}");
                let rots = composition_rotation_blocks(fid, dim, cf_num);
                assert_eq!(rots.len(), cf_num, "fid={fid} dim={dim}");
                assert!(
                    rots.iter().all(|m| m.len() == dim && m.iter().all(|row| row.len() == dim)),
                    "fid={fid} dim={dim}"
                );
            }
        }
    }
}
