//! CEC 2014 independent numeric cross-validation (M3-6 Task 5).
//!
//! MIRRORS the CEC 2022 cross-validation precedent set by M3-3 Task 8
//! (`cec2022_crosscheck.rs` / `tests/data/cec2022_reference_values.json`):
//! the COMPILED official C reference (`cec14_test_func.cpp`,
//! `github.com/P-N-Suganthan/CEC2014`) is the PRIMARY reference (1e-8
//! relative tolerance, see `TOLERANCE`'s doc comment), and `opfunu==1.0.4`
//! is a SECONDARY cross-check, asserted against on every point NOT tagged
//! with an exclusion in the fixture.
//!
//! Fixture: `tests/data/cec2014_reference_values.json`, generated in the
//! scratchpad (never committed from inside the repo) by compiling the
//! vendored C reference (the SAME portability-patched `cec14_test_func.cpp`
//! M3-6 T2 fixed and T3/T4 reused unmodified) and running `opfunu==1.0.4` at
//! four fixed deterministic probe points per (fid, dim) -- `o`, `zeros`,
//! `ones10`, and a seeded `random` point -- for every `fid` in `1..=30` and
//! `dim` in `{10,30}` (this milestone's vendored scope). See the fixture's
//! own `header` block (source URLs, SHAs, exact probe-point definitions,
//! tolerance rationale, and every divergence class's full quoted evidence)
//! and `task-5-report.md` for the complete generation transcript.
//!
//! ## opfunu agreement summary
//!
//! Unlike CEC 2022 (where opfunu diverged on nearly every fid), opfunu's
//! CEC 2014 implementation agrees with the compiled C on fid 1-16 (all 64
//! points, no exclusions -- CEC2014 has no Zakharov-shaped bug the way
//! CEC2022's fid 1 did) and on fid 28 (Composition Function 6, all 8
//! points -- the ONE composition class opfunu implements with fully correct
//! per-component shift AND rotation-matrix slicing). Every other fid in
//! 17-27 and 29-30 has AT LEAST ONE tagged divergence class, each with full
//! quoted source evidence in the fixture's `header.divergence_classes` map:
//! a shuffle-before-rotate ordering bug for every hybrid fid (17-22, same
//! bug CLASS M3-3 T8 already found in opfunu's CEC2022 hybrids), a
//! missing-shift-and-rotation bug specific to Composition Function 1's two
//! ambiguously-named Elliptic components (fid 23 only), a wrong-rotation-
//! submatrix bug for four composition functions that reuse standalone
//! sub-benchmark instances without overriding their default matrix (fid
//! 24-27), and all three of the above compounding for the two
//! hybrid-of-hybrids composition functions (fid 29-30).
//!
//! Every exclusion tag below is defined, with its full quoted adjudicating
//! evidence, in the fixture's `header.divergence_classes` map -- named, not
//! just numbered, so a reader of a failing assertion can look the tag up
//! directly in the JSON.

use std::collections::BTreeMap;

use sezgi_core::problem::Problem;
use sezgi_core::space::{BlockValues, Genotype};
use sezgi_problems::Cec2014;

/// Tolerance rationale: 1e-8 RELATIVE, same value and same rationale
/// `cec2022_crosscheck.rs`'s own `TOLERANCE` uses (M3-3 T8 precedent,
/// mirrored here) -- independent summation order across sezgi (Rust), the
/// compiled C reference (C++), and opfunu (numpy) means bit-exactness is
/// not expected from any of these pairings. In practice sezgi vs. the
/// compiled C measures far tighter than this in this task's own generation
/// run: max relative delta 2.906961657986538e-15 over all 240 probe points
/// (near f64 machine epsilon), consistent with T2-T4's own ~1e-15 findings
/// for their own, smaller, random-point fixtures.
const TOLERANCE: f64 = 1e-8;

fn g(xs: Vec<f64>) -> Genotype { Genotype { blocks: vec![BlockValues::Float(xs)] } }

fn rel_diff(got: f64, want: f64) -> f64 { (got - want).abs() / want.abs().max(1e-300) }

#[derive(Debug)]
struct Point {
    x: Vec<f64>,
    c_value: f64,
    opfunu_value: f64,
    exclude: Vec<String>,
}

#[derive(Debug)]
struct FidDim {
    fid: u32,
    dim: usize,
    points: BTreeMap<String, Point>,
}

fn load_fixture() -> Vec<FidDim> {
    let raw = include_str!("data/cec2014_reference_values.json");
    let v: serde_json::Value = serde_json::from_str(raw).expect("fixture must be valid JSON");
    let points_obj = v["points"].as_object().expect("fixture must have a `points` object");
    let mut out = Vec::new();
    for (_key, entry) in points_obj {
        let fid = entry["fid"].as_u64().unwrap() as u32;
        let dim = entry["dim"].as_u64().unwrap() as usize;
        let mut points = BTreeMap::new();
        for (label, p) in entry["points"].as_object().unwrap() {
            let x: Vec<f64> =
                p["x"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let c_value = p["c_value"].as_f64().unwrap();
            let opfunu_value = p["opfunu_value"].as_f64().unwrap();
            let exclude: Vec<String> = match &p["exclude"] {
                serde_json::Value::Null => Vec::new(),
                serde_json::Value::Array(a) => {
                    a.iter().map(|t| t.as_str().unwrap().to_string()).collect()
                }
                other => panic!("unexpected `exclude` shape: {other:?}"),
            };
            points.insert(label.clone(), Point { x, c_value, opfunu_value, exclude });
        }
        assert_eq!(
            points.len(),
            4,
            "fid {fid} dim {dim}: expected 4 probe points (o, zeros, ones10, random)"
        );
        out.push(FidDim { fid, dim, points });
    }
    assert_eq!(out.len(), 60, "expected 60 (fid,dim) combinations in the fixture (30 fids x {{10,30}})");
    out
}

/// Every point must match the compiled C (primary, always). Every point NOT
/// carrying an exclusion tag must ALSO match opfunu (secondary) -- exclusion
/// only ever waives the opfunu comparison, never the primary one.
#[test]
fn sezgi_matches_compiled_c_and_non_excluded_opfunu_points() {
    let fixture = load_fixture();
    let mut checked_c = 0usize;
    let mut checked_opfunu = 0usize;
    let mut excluded = 0usize;
    for fd in &fixture {
        let p = Cec2014::new(fd.fid, fd.dim)
            .unwrap_or_else(|e| panic!("fid {} dim {}: {e}", fd.fid, fd.dim));
        for (label, pt) in &fd.points {
            let got = p.evaluate_batch(&[g(pt.x.clone())])[0];

            // Primary reference: the compiled official C. Always asserted,
            // exclusion tags never waive this one.
            let rel_c = rel_diff(got, pt.c_value);
            assert!(
                rel_c < TOLERANCE,
                "fid {} dim {} [{label}]: sezgi={got:?} vs compiled-C={:?} (rel={rel_c:.3e}) -- \
                 exceeds {TOLERANCE:e} relative tolerance against the PRIMARY reference",
                fd.fid, fd.dim, pt.c_value
            );
            checked_c += 1;

            // Secondary reference: opfunu, unless this point is tagged with
            // an adjudicated divergence (fixture header's
            // `divergence_classes` names the evidence for every tag).
            if pt.exclude.is_empty() {
                let rel_o = rel_diff(got, pt.opfunu_value);
                assert!(
                    rel_o < TOLERANCE,
                    "fid {} dim {} [{label}]: sezgi={got:?} vs opfunu={:?} (rel={rel_o:.3e}) -- \
                     exceeds {TOLERANCE:e} relative tolerance against opfunu, and this point carries \
                     NO exclusion tag in the fixture (an untagged opfunu disagreement is either a new \
                     sezgi bug or a new divergence that needs a fixture tag + header entry, not a \
                     silent pass)",
                    fd.fid, fd.dim, pt.opfunu_value
                );
                checked_opfunu += 1;
            } else {
                excluded += 1;
            }
        }
    }
    assert_eq!(checked_c, 240, "expected 240 total probe points (60 combos x 4 points)");
    assert_eq!(checked_opfunu + excluded, 240);
    assert_eq!(excluded, 64, "expected 64 excluded (opfunu-diverges) points, see task-5-report.md's disagreement table");
    assert_eq!(
        checked_opfunu, 176,
        "expected 176 points asserted against opfunu: fid 1-16's 128 points + fid 28's 8 points \
         (full agreement, no exclusions) + fid 17-22's 12 `o` points + fid 23-27/29-30's 28 `o`+`zeros` \
         points (128+8+12+28=176)"
    );
}

/// Sanity check on the fixture's own internal consistency: every exclusion
/// tag used by a point must be defined (with its adjudicating evidence) in
/// the fixture's `header.divergence_classes` map -- catches a typo'd tag
/// silently meaning "no evidence recorded" instead of failing loudly.
#[test]
fn every_exclusion_tag_is_documented_in_the_fixture_header() {
    let raw = include_str!("data/cec2014_reference_values.json");
    let v: serde_json::Value = serde_json::from_str(raw).unwrap();
    let classes = v["header"]["divergence_classes"].as_object().unwrap();
    let fixture = load_fixture();
    let mut seen = std::collections::BTreeSet::new();
    for fd in &fixture {
        for pt in fd.points.values() {
            for tag in &pt.exclude {
                seen.insert(tag.clone());
            }
        }
    }
    assert!(!seen.is_empty(), "expected at least one exclusion tag to be exercised");
    for tag in &seen {
        assert!(
            classes.contains_key(tag),
            "exclusion tag {tag:?} is used by a point but has no entry in header.divergence_classes"
        );
    }
    // Every documented class should also actually be used somewhere --
    // otherwise the header is carrying dead documentation.
    for tag in classes.keys() {
        assert!(seen.contains(tag), "header.divergence_classes has {tag:?} but no point uses it");
    }
}

/// Every `o` point in the fixture must be exactly (bit-for-bit) `100*fid` --
/// the same algebraic pin T2-T4's own `mod.rs` tests already assert from a
/// different angle (`Cec2014::f_star()` directly); this test re-derives it
/// from the fixture's OWN independently-generated `c_value`s, catching a
/// hypothetical fixture-generation bug that a pure code-side test couldn't.
#[test]
fn fixture_o_points_are_exactly_f_star() {
    let fixture = load_fixture();
    for fd in &fixture {
        let o = &fixture.iter().find(|f| f.fid == fd.fid && f.dim == fd.dim).unwrap().points["o"];
        assert_eq!(
            o.c_value,
            100.0 * f64::from(fd.fid),
            "fid {} dim {}: fixture's own compiled-C `o` value is not exactly 100*fid",
            fd.fid,
            fd.dim
        );
    }
}
