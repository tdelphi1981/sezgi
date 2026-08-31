//! Data-integrity check for `crates/problems/data/cec2014/`: confirms the
//! FULL 30-function vendoring (M3-6 T2's "vendor now so T3/T4 do not
//! re-vendor" scope, `cec2014/mod.rs`'s module doc "Vendoring scope"
//! section) is actually present on disk with the exact measured file count
//! and byte total this task's report records -- 106 files, 2,811,834
//! bytes: every `shift_data_<fid>.txt` for `fid` 1-30 (dimension-
//! independent), every `M_<fid>_D{10,30}.txt` for `fid` 1-30, and
//! `shuffle_data_<fid>_D{10,30}.txt` for the 8 fids the reference C's own
//! loader reads shuffle data for (`{17,18,19,20,21,22,29,30}`). This is a
//! filesystem-level check independent of `include_str!` (all fid 1-30
//! files are also embedded into the compiled crate, `cec2014/data.rs`'s own
//! module doc) so a future change accidentally deleting or renaming an
//! already-vendored file fails CI here, not silently.

use std::fs;
use std::path::PathBuf;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/cec2014")
}

fn expected_files() -> Vec<String> {
    let mut files = Vec::new();
    for fid in 1..=30 {
        files.push(format!("shift_data_{fid}.txt"));
    }
    for fid in 1..=30 {
        for dim in [10, 30] {
            files.push(format!("M_{fid}_D{dim}.txt"));
        }
    }
    for fid in [17, 18, 19, 20, 21, 22, 29, 30] {
        for dim in [10, 30] {
            files.push(format!("shuffle_data_{fid}_D{dim}.txt"));
        }
    }
    files
}

#[test]
fn every_expected_fid_1_to_30_file_is_vendored() {
    let dir = data_dir();
    for name in expected_files() {
        let path = dir.join(&name);
        assert!(path.is_file(), "missing vendored file: {name} (looked in {dir:?})");
    }
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
    assert_eq!(expected.len(), 106, "expected file count changed -- update this test's derivation");
    assert_eq!(total, 2_811_834, "vendored byte total changed -- update this task's report too");
}

#[test]
fn no_extraneous_files_sit_in_the_cec2014_data_directory() {
    // Every file actually present in crates/problems/data/cec2014/ must be
    // one of the expected 106 -- guards against an accidental extra/stray
    // file (e.g. a duplicate vendored under the wrong name) inflating the
    // byte total silently past what the two tests above check.
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
