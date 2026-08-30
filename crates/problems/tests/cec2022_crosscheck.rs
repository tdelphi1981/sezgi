//! CEC 2022 independent numeric cross-validation (Task 8).
//!
//! PROTOCOL AMENDMENT (controller ruling, supersedes the milestone brief's
//! original "the report is the authority" discrepancy protocol): where the
//! printed CEC 2022 technical report and the official C reference code
//! (`github.com/P-N-Suganthan/2022-SO-BO`) diverge, sezgi follows the C
//! code -- the code scored the competition. This test therefore treats the
//! COMPILED official C as the PRIMARY reference (1e-8 relative tolerance,
//! see `TOLERANCE`'s doc comment below) and `opfunu` (a widely used,
//! MIT-licensed, independent Python CEC-benchmark package) as a SECONDARY
//! cross-check, asserted against on every point NOT tagged with an
//! exclusion in the fixture.
//!
//! Fixture: `tests/data/cec2022_reference_values.json`, generated in the
//! scratchpad (never committed from inside the repo) by compiling the
//! vendored C reference and running `opfunu==1.0.4` at three fixed
//! deterministic probe points per (fid, dim) plus the official shift vector
//! `o` -- see the fixture's own `header` block (source URLs, SHA, exact
//! probe-point definitions, tolerance rationale, and every divergence
//! class's full quoted evidence) and `task-8-report.md` for the complete
//! generation transcript.
//!
//! ## Why most opfunu points are excluded here
//!
//! This task found opfunu (1.0.4) disagrees with the compiled C reference
//! (and therefore with sezgi, which matches the compiled C to ~1e-15
//! relative at every probe point in this fixture -- see the report) on
//! EVERY fid except fid 2 (Rosenbrock), for every probe point except `o`
//! (which is trivially `F_i*` for any correct implementation, C-following
//! or report-following alike). Four of these are the milestone's
//! PRE-ADJUDICATED report-vs-code divergences (F3, F4, F5, fid-7 hybrid --
//! `cec2022/mod.rs`'s module doc, quoted C lines). The rest are NEW opfunu
//! bugs found and root-caused this task (not sezgi bugs -- verified by
//! matching sezgi against the compiled C independently of opfunu at every
//! point): a missing per-index weight in opfunu's Zakharov (fid 1), a
//! shuffle-before-rotate ordering bug affecting every hybrid fid (6-8), and
//! a "every composition component shares component-0's shift vector" bug
//! plus two smaller per-fid issues affecting every composition fid (9-12).
//! Every exclusion tag below is defined, with its full quoted adjudicating
//! evidence (C source lines and/or opfunu source lines), in the fixture's
//! `header.divergence_classes` map -- named, not just numbered, so a reader
//! of a failing assertion can look the tag up directly in the JSON.

use std::collections::BTreeMap;

use sezgi_core::problem::Problem;
use sezgi_core::space::{BlockValues, Genotype};
use sezgi_problems::Cec2022;

/// Tolerance rationale: 1e-8 RELATIVE, because independent summation order
/// across sezgi (Rust), the compiled C
/// reference (C++), and opfunu (numpy) means bit-exactness is not expected
/// from any of these pairings. In practice sezgi vs. the compiled C
/// measured far tighter than this in this task's own generation run: max
/// relative delta 5.7e-15 over all 132 probe points (near f64 machine
/// epsilon) -- consistent with T5-T7's own reviews (~1e-10). 1e-8 leaves
/// comfortable headroom above both measured maxima while still catching a
/// genuine formula-level bug (every real bug found this task showed
/// relative deltas of 1e-3 or larger, module doc's divergence classes).
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
    let raw = include_str!("data/cec2022_reference_values.json");
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
        assert_eq!(points.len(), 4, "fid {fid} dim {dim}: expected 4 probe points (o, zeros, ones10, random)");
        out.push(FidDim { fid, dim, points });
    }
    assert_eq!(out.len(), 33, "expected 33 (fid,dim) combinations in the fixture");
    out
}

/// Every non-excluded point must match BOTH the compiled C (primary,
/// always) and opfunu (secondary, only where not excluded) within
/// [`TOLERANCE`] relative. Excluded points still assert against the
/// compiled C -- exclusion only ever waives the opfunu comparison, never
/// the primary one.
#[test]
fn sezgi_matches_compiled_c_and_non_excluded_opfunu_points() {
    let fixture = load_fixture();
    let mut checked_c = 0usize;
    let mut checked_opfunu = 0usize;
    let mut excluded = 0usize;
    for fd in &fixture {
        let p = Cec2022::new(fd.fid, fd.dim)
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
            // an adjudicated or newly-found divergence (fixture header's
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
    assert_eq!(checked_c, 132, "expected 132 total probe points (33 combos x 4 points)");
    // 42 points agree with opfunu (fid 2's 12 points at every dim/label, plus
    // the 33 `o` points minus fid 2's own `o` already counted -- see the
    // fixture: only fid 2 and every fid's `o` point carry no exclusion tag).
    assert_eq!(checked_opfunu + excluded, 132);
    assert_eq!(excluded, 90, "expected 90 excluded (opfunu-diverges) points, see task-8-report.md's disagreement table");
    assert_eq!(checked_opfunu, 42, "expected 42 points asserted against opfunu (fid 2's 12 points + 33 `o` points, minus fid 2's own `o` counted once)");
}

/// Sanity check on the fixture's own internal consistency: every exclusion
/// tag used by a point must be defined (with its adjudicating evidence) in
/// the fixture's `header.divergence_classes` map -- catches a typo'd tag
/// silently meaning "no evidence recorded" instead of failing loudly.
#[test]
fn every_exclusion_tag_is_documented_in_the_fixture_header() {
    let raw = include_str!("data/cec2022_reference_values.json");
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
