//! CEC 2017 independent numeric cross-validation (M3-6 Task 8).
//!
//! MIRRORS the CEC 2014 cross-validation precedent set by M3-6 Task 5
//! (`cec2014_crosscheck.rs` / `tests/data/cec2014_reference_values.json`,
//! itself mirroring CEC 2022's own M3-3 Task 8 fixture): the COMPILED
//! official C reference (`cec17_test_func.cpp`,
//! `github.com/P-N-Suganthan/CEC2017-BoundContrained`) is the PRIMARY
//! reference (1e-8 relative tolerance, see `TOLERANCE`'s doc comment), and
//! `opfunu==1.0.4` is a SECONDARY cross-check, asserted against on every
//! point NOT tagged with an exclusion in the fixture.
//!
//! Fixture: `tests/data/cec2017_reference_values.json`, generated in the
//! scratchpad (never committed from inside the repo) by compiling the
//! vendored C reference (the SAME portability-patched `cec17_test_func.cpp`
//! M3-6 T6 fixed and T7/T8 reused unmodified) and running `opfunu==1.0.4` at
//! four fixed deterministic probe points per (fid, dim) -- `o`, `zeros`,
//! `ones10`, and a seeded `random` point -- for every `fid` in `{1,3..=30}`
//! (fid 2 permanently withdrawn, never in sezgi) and `dim` in `{10,30}`
//! (this milestone's vendored scope): 58 combos x 4 points = 232 total. See
//! the fixture's own `header` block (source URLs, SHAs, exact probe-point
//! definitions, tolerance rationale, the opfunu remap rule + derivation,
//! and every divergence class's full quoted evidence) and
//! `task-8-report.md` for the complete generation transcript.
//!
//! ## opfunu remap + agreement summary (VERY different shape from CEC 2014's
//! own fixture -- read this before assuming the mechanics are identical)
//!
//! opfunu 1.0.4 numbers CEC2017 CONTIGUOUSLY (`F1`..`F29`, no gap for the
//! withdrawn fid 2), so its own class index `k` does not equal sezgi's own
//! (gapped) `fid` -- the fixture's `header.opfunu_remap` documents the
//! empirically-derived rule (`opfunu_class_index(fid) = 1 if fid==1 else
//! fid-1`) and the diagnostic transcripts that established it (a naive "no
//! remap" or "always fid-1" guess was NOT assumed; both were probed against
//! every opfunu class x every sezgi fid at 3 independent points before
//! settling on this rule -- see `header.opfunu_remap.derivation`).
//!
//! Unlike CEC 2014 (where opfunu agreed on 17 of 30 fids, `cec2014_
//! crosscheck.rs`'s own doc comment), CEC2017's remapped opfunu agrees on
//! **only fid 1** (8 of 232 points: `o`/`zeros`/`ones10`/`random` x
//! `{10,30}`) -- EVERY other fid (3-30) has at least one verified,
//! source-quoted opfunu bug making its own remapped class never
//! numerically correspond to the real suite at all, not merely
//! occasionally disagree at hard points:
//! - fid 3 (opfunu's own "F2"): the class whose Table-row position and
//!   bias corresponds to fid 3 is built from the WITHDRAWN fid 2's own dead
//!   shift/rotation data (`opfunu-f2-bogus-withdrawn-data`) -- this
//!   milestone's own explicit scope ruling (SKIP, expected-disagreement-
//!   by-construction).
//! - fid 4-20: every remapped opfunu class reads the shift/rotation data
//!   belonging to the PREVIOUS real function slot (an unapplied +1 gap
//!   adjustment, `opfunu-unimodal-hybrid-data-source-offset`) -- a
//!   formula/data mismatch, not any real CEC2017 instance.
//! - fid 21-28 (composition, base-function): the remapped opfunu class DOES
//!   use the correct per-fid data, but drops at least one component's own
//!   internal scale constant (verified concretely for fid 21's Rastrigin
//!   component, `opfunu-composition-missing-component-scale`).
//! - fid 29-30 (composition, hybrid-in-composition): the remapped opfunu
//!   class shares ONE component's shift vector across all three
//!   sub-hybrids instead of giving each its own
//!   (`opfunu-composition-hybrid-shared-shift`).
//!
//! Every exclusion tag below is defined, with its full quoted adjudicating
//! evidence, in the fixture's `header.divergence_classes` map -- named, not
//! just numbered, so a reader of a failing assertion can look the tag up
//! directly in the JSON.

use std::collections::BTreeMap;

use sezgi_core::problem::Problem;
use sezgi_core::space::{BlockValues, Genotype};
use sezgi_problems::Cec2017;

/// Tolerance rationale: 1e-8 RELATIVE, same value and same rationale every
/// earlier crosscheck fixture in this crate uses (`cec2014_crosscheck.rs`'s
/// own `TOLERANCE`, M3-6 T5 precedent, itself following `cec2022_
/// crosscheck.rs`'s M3-3 T8 precedent) -- independent summation order
/// across sezgi (Rust) and the compiled C reference (C++) means
/// bit-exactness is not expected. In practice sezgi vs. the compiled C
/// measures far tighter than this in this task's own generation run: max
/// relative delta `6.066832304353314e-15` over all 232 probe points (near
/// f64 machine epsilon), consistent with T2-T8's own ~1e-15 findings for
/// their own fixtures (see
/// `sezgi_matches_compiled_c_and_the_one_agreeing_opfunu_fid`'s own
/// max-measured-deviation assertion below, and this task's report has the
/// full transcript).
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
    let raw = include_str!("data/cec2017_reference_values.json");
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
    assert_eq!(
        out.len(),
        58,
        "expected 58 (fid,dim) combinations in the fixture (29 usable fids x {{10,30}})"
    );
    out
}

/// Every point must match the compiled C (primary, always). Every point NOT
/// carrying an exclusion tag must ALSO match opfunu (secondary) -- exclusion
/// only ever waives the opfunu comparison, never the primary one.
#[test]
fn sezgi_matches_compiled_c_and_the_one_agreeing_opfunu_fid() {
    let fixture = load_fixture();
    let mut checked_c = 0usize;
    let mut checked_opfunu = 0usize;
    let mut excluded = 0usize;
    let mut max_c_rel = 0.0f64;
    for fd in &fixture {
        let p = Cec2017::new(fd.fid, fd.dim)
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
            if rel_c > max_c_rel {
                max_c_rel = rel_c;
            }
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
    assert_eq!(checked_c, 232, "expected 232 total probe points (58 combos x 4 points)");
    assert_eq!(checked_opfunu + excluded, 232);
    assert_eq!(
        checked_opfunu, 8,
        "expected exactly 8 points asserted against opfunu: fid 1's own o/zeros/ones10/random x \
         {{10,30}} -- opfunu's CEC2017 remap agrees on NO other fid (module doc's opfunu remap + \
         agreement summary section)"
    );
    assert_eq!(
        excluded, 224,
        "expected 224 excluded (opfunu-diverges) points: 232 total - 8 agreeing = 224, see \
         task-8-report.md's disagreement table"
    );
    // Measured (this task's own generation run, stated in the report):
    // max relative deviation sezgi-vs-compiled-C well under 1e-9 for every
    // one of the 232 points -- 1e-8 is comfortably above it, same pattern
    // T2-T8's own smaller fixtures already established.
    assert!(
        max_c_rel < 1e-9,
        "measured max relative deviation vs compiled C ({max_c_rel:.3e}) unexpectedly exceeds the \
         ~1e-15-class figure this task's report states -- investigate before trusting the 1e-8 gate"
    );
}

/// Sanity check on the fixture's own internal consistency: every exclusion
/// tag used by a point must be defined (with its adjudicating evidence) in
/// the fixture's `header.divergence_classes` map -- catches a typo'd tag
/// silently meaning "no evidence recorded" instead of failing loudly.
#[test]
fn every_exclusion_tag_is_documented_in_the_fixture_header() {
    let raw = include_str!("data/cec2017_reference_values.json");
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
/// the same algebraic pin T6-T8's own `mod.rs` tests already assert from a
/// different angle (`Cec2017::f_star()` directly, plus fid 9's own verified
/// exception -- see below); this test re-derives it from the fixture's OWN
/// independently-generated `c_value`s, catching a hypothetical
/// fixture-generation bug that a pure code-side test couldn't.
#[test]
fn fixture_o_points_are_exactly_f_star_except_fid_9() {
    let fixture = load_fixture();
    for fd in &fixture {
        let o = &fd.points["o"];
        if fd.fid == 9 {
            // Module doc's fid-9 (Levy) VERIFIED exception (cec2017/mod.rs):
            // x=o does NOT pin to F_i* exactly for this one fid -- confirm
            // the fixture's own independently-generated value matches the
            // ALREADY-documented measured constants instead.
            let expect = if fd.dim == 10 { 901.4426009870527 } else { 903.2594920693923 };
            let rel = rel_diff(o.c_value, expect);
            assert!(
                rel < 1e-9,
                "fid 9 dim {}: fixture's own compiled-C `o` value {} does not match the \
                 module-doc-documented measured constant {expect} (rel={rel:.3e})",
                fd.dim,
                o.c_value
            );
            assert_ne!(o.c_value, 100.0 * f64::from(fd.fid), "fid 9: `o` must NOT pin to F_i* exactly");
            continue;
        }
        assert_eq!(
            o.c_value,
            100.0 * f64::from(fd.fid),
            "fid {} dim {}: fixture's own compiled-C `o` value is not exactly 100*fid",
            fd.fid,
            fd.dim
        );
    }
}
