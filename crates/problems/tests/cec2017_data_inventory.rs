//! Data-integrity check for `crates/problems/data/cec2017/`: confirms the
//! FULL usable-suite vendoring (M3-6 T6's "vendor now so T7/T8 do not
//! re-vendor" scope, `cec2017/mod.rs`'s module doc "Vendoring scope"
//! section) is actually present on disk with the exact measured file count
//! and byte total this task's report records -- 111 files, 3,280,439
//! bytes: every `shift_data_<fid>.txt` for `fid` in `{1,3..=30}`
//! (dimension-independent, fid 2 excluded -- permanently withdrawn, module
//! doc's F2 ruling), every `M_<fid>_D{10,30}.txt` for `fid` in
//! `{1,3..=30}`, and `shuffle_data_<fid>_D{10,30}.txt` for the fids the
//! reference C's own loader reads shuffle data for (`{11..=20, 29, 30}`).
//! This is a filesystem-level check independent of `include_str!` (all
//! `{1,3..=30}` files are also embedded into the compiled crate,
//! `cec2017/data.rs`'s own module doc) so a future change accidentally
//! deleting or renaming an already-vendored file fails CI here, not
//! silently.

use std::fs;
use std::path::PathBuf;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/cec2017")
}

fn wanted_fids() -> Vec<u32> {
    let mut v = vec![1u32];
    v.extend(3..=30);
    v
}

fn expected_files() -> Vec<String> {
    let mut files = Vec::new();
    for fid in wanted_fids() {
        files.push(format!("shift_data_{fid}.txt"));
    }
    for fid in wanted_fids() {
        for dim in [10, 30] {
            files.push(format!("M_{fid}_D{dim}.txt"));
        }
    }
    let shuffle_fids: Vec<u32> = (11..=20).chain([29, 30]).collect();
    for fid in shuffle_fids {
        for dim in [10, 30] {
            files.push(format!("shuffle_data_{fid}_D{dim}.txt"));
        }
    }
    files
}

#[test]
fn every_expected_fid_file_is_vendored() {
    let dir = data_dir();
    for name in expected_files() {
        let path = dir.join(&name);
        assert!(path.is_file(), "missing vendored file: {name} (looked in {dir:?})");
    }
}

#[test]
fn no_fid_2_file_is_vendored() {
    // Module doc's F2 ruling: fid 2 was never implemented and its dead
    // input_data files were deliberately never vendored.
    let dir = data_dir();
    assert!(!dir.join("shift_data_2.txt").is_file());
    assert!(!dir.join("M_2_D10.txt").is_file());
    assert!(!dir.join("M_2_D30.txt").is_file());
}

#[test]
fn vendored_set_totals_the_exact_measured_byte_count() {
    let dir = data_dir();
    let expected = expected_files();
    let mut total: u64 = 0;
    for name in &expected {
        let meta = fs::metadata(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        total += meta.len();
    }
    assert_eq!(expected.len(), 111, "expected file count changed -- update this test's derivation");
    assert_eq!(total, 3_280_439, "vendored byte total changed -- update this task's report too");
}

#[test]
fn no_extraneous_files_sit_in_the_cec2017_data_directory() {
    // Every file actually present in crates/problems/data/cec2017/ must be
    // one of the expected 111 -- guards against an accidental extra/stray
    // file (e.g. a duplicate vendored under the wrong name, or a fid-2
    // file) inflating the byte total silently past what the two tests
    // above check.
    let dir = data_dir();
    let expected: std::collections::HashSet<String> = expected_files().into_iter().collect();
    let mut on_disk: Vec<String> = fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{dir:?}: {e}"))
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    on_disk.sort();
    for name in &on_disk {
        assert!(expected.contains(name), "unexpected file in {dir:?}: {name}");
    }
    assert_eq!(on_disk.len(), expected.len());
}
