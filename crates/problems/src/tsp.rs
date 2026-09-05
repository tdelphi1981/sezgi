//! TSPLIB95 `EUC_2D` reader and the scalar Traveling Salesman `Problem`.
//!
//! Source (PROVENANCE, fetched and read directly, not from memory): Gerhard
//! Reinelt, "TSPLIB 95", Universitat Heidelberg, Institut fur Angewandte
//! Mathematik. PDF fetched from a university mirror (the canonical
//! `comopt.ifi.uni-heidelberg.de` host did not resolve at fetch time):
//! `https://www.or.uni-bonn.de/lectures/ws17/co_exercises/programming/tsp/tsp95.pdf`
//! (16 pages; every quote below is transcribed from that fetched PDF, page
//! numbers as printed in the document).
//!
//! ## File grammar (Sec. 1, pp. 2-5, quoted)
//!
//! "Each file consists of a **specification part** and of a **data part**.
//! [...] All entries in this section are of the form `<keyword> : <value>`."
//! The keywords this module consumes:
//! - **`NAME`** (1.1.1): "Identifies the data file."
//! - **`TYPE`** (1.1.2): "Specifies the type of the data." (`TSP` = "Data for
//!   a symmetric traveling salesman problem" -- this module targets that
//!   case only; the field is parsed but not enforced, since the tests this
//!   task pins are about `EDGE_WEIGHT_TYPE`, not `TYPE`).
//! - **`DIMENSION`** (1.1.4): "For a TSP or ATSP, the dimension is the
//!   number of its nodes."
//! - **`EDGE_WEIGHT_TYPE`** (1.1.6): "Specifies how the edge weights (or
//!   distances) are given." Of the listed values, this module (v1) supports
//!   only `EUC_2D` ("Weights are Euclidean distances in 2-D"); every other
//!   value (`GEO`, `ATT`, `CEIL_2D`, ...) is a documented, explicit
//!   [`TspError::UnsupportedEdgeWeightType`].
//! - **`NODE_COORD_SECTION`** (1.2.1): "Node coordinates are given in this
//!   section. Each line is of the form `<integer> <real> <real>` if
//!   `NODE_COORD_TYPE` is `TWOD_COORDS` [the default, and the only case this
//!   module handles]. [...] The integers give the number of the respective
//!   nodes." Every vendored `EUC_2D` instance in this crate numbers nodes
//!   `1..=DIMENSION` (see the vendored files under `data/tsplib/`), matching
//!   the wider TSPLIB corpus convention; this module does not assume the
//!   lines appear in that order (it indexes by the printed node number, not
//!   by line position) but does require every node `1..=DIMENSION` to
//!   appear exactly once.
//! - **`EOF`** (1.1.11): "Terminates the input data. **This entry is
//!   optional.**" -- pinned here because it is the one keyword the spec
//!   explicitly marks optional; [`Tsp::from_tsplib`] must therefore accept
//!   input with no trailing `EOF` line (tested below), unlike a missing
//!   `DIMENSION`/`EDGE_WEIGHT_TYPE`/`NODE_COORD_SECTION`, which this module
//!   treats as required because nothing else in the grammar can supply the
//!   coordinate count or the distance rule.
//!
//! ## `EUC_2D` distance -- THE rounding trap (Sec. 2.1 + intro, p. 6, quoted)
//!
//! "Since distances are required to be integral, we round to the nearest
//! integer (in most cases). Below we have used the rounding function `nint`
//! (`nint(x)` can be replaced by `(int) (x+0.5)`)." Then, Sec. 2.1
//! ("Euclidean distance (`L2`-metric)"), the reference C snippet, quoted
//! verbatim:
//! ```text
//! xd = x[i] - x[j];
//! yd = y[i] - y[j];
//! dij = nint( sqrt( xd*xd + yd*yd) );
//! ```
//! This module pins EXACTLY that form -- `(int) (x + 0.5)`, i.e. truncate
//! `x + 0.5` toward zero -- as [`nint`]. Note (per the brief): Rust's
//! `f64::round` (round-half-away-from-zero) and `(int)(x+0.5)`-truncation
//! disagree for negative `x` (e.g. `round(-0.5) = -1.0` vs.
//! `(int)(-0.5+0.5) = (int)(0.0) = 0`), but every `EUC_2D` distance here is
//! `sqrt(...) >= 0.0`, where truncation-toward-zero and `floor` coincide --
//! so `nint` is implemented as `(x + 0.5).floor()`, the pinned formula, not
//! `x.round()`.
//!
//! Tour length = the sum of the `nint`-rounded leg distances around the
//! CLOSED tour (last city back to the first) -- this is what
//! [`Tsp::evaluate_batch`] computes, and what the golden test below pins
//! against a published-optimal tour.
//!
//! ## Vendored instances and their published optima (Sec. 3.1, Table 1, p.
//! 9-11 of the same PDF, quoted rows)
//!
//! | Name | #cities | Type | Bounds |
//! |---|---|---|---|
//! | berlin52 | 52 | EUC_2D | **7542** |
//! | eil51 | 51 | EUC_2D | **426** |
//! | st70 | 70 | EUC_2D | **675** |
//!
//! "Table 1 gives the problem names along with number of cities, problem
//! type, and known lower and upper bounds for the optimal tour length (a
//! single number indicating that the optimal length is known)." All three
//! rows above show a single (bold) number, i.e. a PROVEN optimum, not a
//! bound gap.
//!
//! The `.tsp` data files and their published-optimal `.opt.tour` files were
//! fetched verbatim from two independent GitHub mirrors of TSPLIB (the
//! canonical Heidelberg host was unreachable at fetch time):
//! `https://raw.githubusercontent.com/mastqe/tsplib/master/` (provided the
//! three `.tsp` files) and
//! `https://raw.githubusercontent.com/pdrozdowski/TSPLib.Net/master/TSPLIB95/tsp/`
//! (provided the same three `.tsp` files -- confirmed BYTE-IDENTICAL to the
//! `mastqe` copies by diff -- plus all three `.opt.tour` files, which
//! `mastqe`'s repo does not carry).
//!
//! ## Licensing finding (mirrors `cec2014/mod.rs`'s and `cec2022/mod.rs`'s
//! scope rulings)
//!
//! The canonical Heidelberg site
//! (`comopt.ifi.uni-heidelberg.de/software/TSPLIB95/`, checked directly, not
//! assumed) carries NO copyright notice, license statement, terms of use or
//! redistribution policy of any kind -- it does not describe the data as
//! "public domain" or as anything else. The data is nonetheless
//! universally redistributed and parsed by TSP-adjacent software without a
//! license grant from Reinelt, e.g. CRAN's `TSP` package and PyPI's
//! `tsplib95`. Per the same scope ruling the CEC suites follow, the three
//! `EUC_2D` instances vendored here (`berlin52`, `eil51`, `st70`) are kept
//! on that established community-norm basis -- vendored with prominent
//! attribution (this doc) rather than withheld -- and this status is
//! documented rather than asserted away. The files are copied verbatim,
//! byte-for-byte, with no reformatting.
//!
//! Independent scratch verification (outside this crate, Python, not part
//! of this task's deliverable) parsed each vendored `.opt.tour` file,
//! summed its `nint`-rounded closed-tour legs against the matching `.tsp`
//! coordinates, and reproduced 7542 / 426 / 675 exactly, before any Rust
//! code below was written -- the golden tests re-derive the same numbers
//! through this module's own pipeline.
//!
//! ## Genotype mapping
//!
//! [`Tsp::space`] is one [`Block::Permutation`] of `n_cities()` elements.
//! [`Block::Permutation`]'s [`BlockValues::Perm`] is `Vec<u32>`, each entry
//! in `0..n` (0-based, [`sezgi_core::space::SearchSpace::validate`]'s own
//! contract). This module maps genotype index `c` (0-based) to TSPLIB node
//! number `c + 1` (1-based) -- i.e. `coords[c]` (0-based storage) holds the
//! coordinates TSPLIB printed for node `c + 1`. A permutation genotype is
//! read as a CLOSED tour: city order `[p0, p1, ..., p_{n-1}]` visits `p0 ->
//! p1 -> ... -> p_{n-1} -> p0`, and [`Tsp::evaluate_batch`] returns the sum
//! of the `n` `nint`-rounded leg distances, including the closing leg
//! `p_{n-1} -> p0`.
//!
//! Genotype validity (each value `< n`, no repeats) is
//! [`sezgi_core::space::SearchSpace::validate`]'s job, not this module's:
//! [`Tsp::evaluate_batch`] receives `&[Genotype]` built by construction
//! (the type system's contract, per [`sezgi_core::problem::Problem`]'s own
//! doc), so it defends only against a genotype whose first block is not a
//! same-length `BlockValues::Perm` or contains an out-of-range index --
//! returning `f64::INFINITY` for that one individual (same sentinel
//! convention as [`crate::zdt::Zdt::evaluate_batch`] /
//! [`sezgi_core::problem::SphereShifted`]) rather than panicking.

use sezgi_core::problem::Problem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};

#[derive(Debug, thiserror::Error)]
pub enum TspError {
    #[error("missing required TSPLIB field {0}")]
    MissingField(&'static str),
    #[error("malformed TSPLIB field {field}: {value:?}")]
    MalformedField { field: &'static str, value: String },
    #[error(
        "unsupported EDGE_WEIGHT_TYPE {0:?}: only EUC_2D is supported in v1 \
         (see this module's doc)"
    )]
    UnsupportedEdgeWeightType(String),
    #[error("malformed NODE_COORD_SECTION line: {0:?}")]
    MalformedCoordLine(String),
    #[error(
        "NODE_COORD_SECTION node index {id} out of range for DIMENSION {n} \
         (TSPLIB node numbers are 1-based, 1..={n})"
    )]
    NodeIndexOutOfRange { id: usize, n: usize },
    #[error("NODE_COORD_SECTION node {id} listed more than once")]
    DuplicateNode { id: usize },
    #[error(
        "DIMENSION says {expected} cities but NODE_COORD_SECTION only supplied \
         coordinates for {actual} of them"
    )]
    CoordCountMismatch { expected: usize, actual: usize },
    #[error(
        "unknown vendored TSPLIB instance {0:?} (expected one of \"berlin52\", \
         \"eil51\", \"st70\")"
    )]
    UnknownVendored(String),
}

const BERLIN52_TSP: &str = include_str!("../data/tsplib/berlin52.tsp");
const EIL51_TSP: &str = include_str!("../data/tsplib/eil51.tsp");
const ST70_TSP: &str = include_str!("../data/tsplib/st70.tsp");

/// Section keywords this parser recognizes as ending the current
/// specification-part scan / the `NODE_COORD_SECTION` data block (Sec. 1.2
/// of the module doc's pinned spec -- every data-part section keyword, so a
/// `NODE_COORD_SECTION` immediately followed by, say, a `DISPLAY_DATA_SECTION`
/// in some other TSPLIB file stops cleanly rather than being mis-parsed as
/// coordinate data).
const SECTION_KEYWORDS: &[&str] = &[
    "NODE_COORD_SECTION",
    "DEPOT_SECTION",
    "DEMAND_SECTION",
    "EDGE_DATA_SECTION",
    "FIXED_EDGES_SECTION",
    "DISPLAY_DATA_SECTION",
    "TOUR_SECTION",
    "EDGE_WEIGHT_SECTION",
];

/// TSPLIB95's pinned rounding rule (module doc): `nint(x) = (int)(x + 0.5)`.
/// Implemented as `floor(x + 0.5)` -- equal to C's truncate-toward-zero cast
/// for the non-negative `x` every `EUC_2D` distance produces (see module
/// doc for why `f64::round` is NOT used here).
fn nint(x: f64) -> f64 { (x + 0.5).floor() }

/// `EUC_2D` distance between two coordinate pairs (module doc, Sec. 2.1
/// quoted): `dij = nint( sqrt( xd*xd + yd*yd) )`.
fn euc_2d(a: (f64, f64), b: (f64, f64)) -> f64 {
    let xd = a.0 - b.0;
    let yd = a.1 - b.1;
    nint((xd * xd + yd * yd).sqrt())
}

fn split_keyword(line: &str) -> Option<(&str, &str)> {
    let idx = line.find(':')?;
    Some((line[..idx].trim(), line[idx + 1..].trim()))
}

/// One TSPLIB `EUC_2D` symmetric-TSP instance: `n_cities()` 2-D city
/// coordinates plus (for vendored instances) the published-optimal tour
/// length. Implements the scalar [`Problem`] trait: [`Tsp::evaluate_batch`]
/// scores a [`Block::Permutation`] genotype as a closed-tour length (module
/// doc).
#[derive(Debug)]
pub struct Tsp {
    name: String,
    coords: Vec<(f64, f64)>,
    space: SearchSpace,
    known_optimum: Option<f64>,
}

impl Tsp {
    /// Parse a TSPLIB-format string. Supports `EDGE_WEIGHT_TYPE: EUC_2D`
    /// only (v1, module doc); `known_optimum()` is `None` for an instance
    /// parsed this way (there is no field in the TSPLIB format itself that
    /// carries the optimum -- see [`Tsp::vendored`] for the published
    /// values pinned separately for this crate's three vendored instances).
    pub fn from_tsplib(text: &str) -> Result<Tsp, TspError> {
        let mut name = String::new();
        let mut dimension: Option<usize> = None;
        let mut edge_weight_type: Option<String> = None;
        let mut slots: Option<Vec<Option<(f64, f64)>>> = None;

        let mut lines = text.lines().peekable();
        while let Some(raw) = lines.next() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if line == "EOF" {
                break;
            }
            if line == "NODE_COORD_SECTION" {
                // sezgi simplification: DIMENSION must precede NODE_COORD_SECTION here; TSPLIB95 says keyword order is "arbitrary in principle", but every distributed instance uses this order.
                let n = dimension.ok_or(TspError::MissingField("DIMENSION"))?;
                let mut section: Vec<Option<(f64, f64)>> = vec![None; n];
                while let Some(&next_raw) = lines.peek() {
                    let next = next_raw.trim();
                    if next.is_empty() {
                        lines.next();
                        continue;
                    }
                    if next == "EOF" || SECTION_KEYWORDS.contains(&next) {
                        break;
                    }
                    lines.next();
                    let parts: Vec<&str> = next.split_whitespace().collect();
                    if parts.len() != 3 {
                        return Err(TspError::MalformedCoordLine(next.to_string()));
                    }
                    let id: usize = parts[0]
                        .parse()
                        .map_err(|_| TspError::MalformedCoordLine(next.to_string()))?;
                    let x: f64 = parts[1]
                        .parse()
                        .map_err(|_| TspError::MalformedCoordLine(next.to_string()))?;
                    let y: f64 = parts[2]
                        .parse()
                        .map_err(|_| TspError::MalformedCoordLine(next.to_string()))?;
                    if id == 0 || id > n {
                        return Err(TspError::NodeIndexOutOfRange { id, n });
                    }
                    if section[id - 1].is_some() {
                        return Err(TspError::DuplicateNode { id });
                    }
                    section[id - 1] = Some((x, y));
                }
                slots = Some(section);
                continue;
            }
            if let Some((keyword, value)) = split_keyword(line) {
                match keyword {
                    "NAME" => name = value.to_string(),
                    "DIMENSION" => {
                        dimension = Some(value.parse().map_err(|_| TspError::MalformedField {
                            field: "DIMENSION",
                            value: value.to_string(),
                        })?);
                    }
                    "EDGE_WEIGHT_TYPE" => edge_weight_type = Some(value.to_string()),
                    // NAME/TYPE/COMMENT and any other keyword this v1
                    // reader does not need are accepted and ignored --
                    // the grammar (module doc) allows unrecognized-but-
                    // well-formed keywords.
                    _ => {}
                }
            }
        }

        let n = dimension.ok_or(TspError::MissingField("DIMENSION"))?;
        let ewt = edge_weight_type.ok_or(TspError::MissingField("EDGE_WEIGHT_TYPE"))?;
        if ewt != "EUC_2D" {
            return Err(TspError::UnsupportedEdgeWeightType(ewt));
        }
        let slots = slots.ok_or(TspError::MissingField("NODE_COORD_SECTION"))?;
        let actual = slots.iter().filter(|c| c.is_some()).count();
        if actual != n {
            return Err(TspError::CoordCountMismatch { expected: n, actual });
        }
        let coords: Vec<(f64, f64)> = slots.into_iter().map(|c| c.expect("counted above")).collect();

        let space = SearchSpace::new(vec![Block::Permutation { n }])
            .expect("Block::Permutation has no bounds to validate");
        Ok(Tsp { name, coords, space, known_optimum: None })
    }

    /// A vendored instance by name: `"berlin52"`, `"eil51"`, or `"st70"`
    /// (module doc: byte-for-byte vendored `.tsp` text, embedded via
    /// `include_str!`). Sets [`Tsp::known_optimum`] to the published
    /// TSPLIB-optima value (module doc, Table 1 quote) -- unlike
    /// [`Tsp::from_tsplib`] on arbitrary external text, which cannot know
    /// it.
    pub fn vendored(name: &str) -> Result<Tsp, TspError> {
        let (text, optimum) = match name {
            "berlin52" => (BERLIN52_TSP, 7542.0),
            "eil51" => (EIL51_TSP, 426.0),
            "st70" => (ST70_TSP, 675.0),
            other => return Err(TspError::UnknownVendored(other.to_string())),
        };
        let mut tsp = Self::from_tsplib(text)?;
        tsp.known_optimum = Some(optimum);
        Ok(tsp)
    }

    pub fn name(&self) -> &str { &self.name }
    pub fn n_cities(&self) -> usize { self.coords.len() }

    /// Published optimal tour length, if known for the vendored instance
    /// (module doc); `None` for an instance parsed via
    /// [`Tsp::from_tsplib`] directly.
    pub fn known_optimum(&self) -> Option<f64> { self.known_optimum }

    /// The 0-based-index-to-TSPLIB-node coordinate this instance stores for
    /// genotype index `c` (module doc's mapping): `coords()[c]` is the pair
    /// TSPLIB printed for node `c + 1`.
    pub fn coords(&self) -> &[(f64, f64)] { &self.coords }

    /// Closed-tour length for one already-validated permutation (module
    /// doc): sum of the `n` `nint`-rounded `EUC_2D` legs, including the
    /// closing leg back to the start. Returns `f64::INFINITY` if `order`'s
    /// length doesn't match [`Tsp::n_cities`] or any entry is out of range
    /// (module doc: defensive, not a genotype-validity check).
    fn tour_length(&self, order: &[u32]) -> f64 {
        let n = self.coords.len();
        if order.len() != n {
            return f64::INFINITY;
        }
        let at = |i: usize| -> Option<(f64, f64)> {
            self.coords.get(*order.get(i)? as usize).copied()
        };
        let mut total = 0.0;
        for i in 0..n {
            let (Some(a), Some(b)) = (at(i), at((i + 1) % n)) else {
                return f64::INFINITY;
            };
            total += euc_2d(a, b);
        }
        total
    }
}

impl Problem for Tsp {
    fn space(&self) -> &SearchSpace { &self.space }

    fn optimum(&self) -> Option<f64> { self.known_optimum() }

    fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<f64> {
        pop.iter()
            .map(|g| match g.blocks.first() {
                Some(BlockValues::Perm(order)) => self.tour_length(order),
                _ => f64::INFINITY,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_core::problem::Evaluator;

    fn perm(order: &[u32]) -> Genotype { Genotype { blocks: vec![BlockValues::Perm(order.to_vec())] } }

    // ---- parser ----

    #[test]
    fn berlin52_parses_with_expected_dimension_and_spot_checked_coords() {
        let tsp = Tsp::vendored("berlin52").unwrap();
        assert_eq!(tsp.n_cities(), 52);
        assert_eq!(tsp.name(), "berlin52");
        // Vendored data/tsplib/berlin52.tsp, first line of NODE_COORD_SECTION:
        // "1 565.0 575.0" -> genotype index 0.
        assert_eq!(tsp.coords()[0], (565.0, 575.0));
        // ... last line: "52 1740.0 245.0" -> genotype index 51.
        assert_eq!(tsp.coords()[51], (1740.0, 245.0));
    }

    #[test]
    fn missing_eof_still_parses() {
        // TSPLIB95 Sec. 1.1.11 (module doc, quoted): "EOF ... This entry is
        // optional." -- a well-formed file with no trailing EOF line must
        // still parse.
        let text = "NAME: tiny\nTYPE: TSP\nDIMENSION: 2\nEDGE_WEIGHT_TYPE: EUC_2D\n\
                     NODE_COORD_SECTION\n1 0.0 0.0\n2 3.0 4.0\n";
        let tsp = Tsp::from_tsplib(text).unwrap();
        assert_eq!(tsp.n_cities(), 2);
        assert_eq!(tsp.coords(), &[(0.0, 0.0), (3.0, 4.0)]);
    }

    #[test]
    fn non_euc2d_edge_weight_type_is_unsupported_error() {
        let text = "NAME: g\nTYPE: TSP\nDIMENSION: 2\nEDGE_WEIGHT_TYPE: GEO\n\
                     NODE_COORD_SECTION\n1 0.0 0.0\n2 1.0 1.0\nEOF\n";
        let err = Tsp::from_tsplib(text).unwrap_err();
        assert!(
            matches!(&err, TspError::UnsupportedEdgeWeightType(ewt) if ewt == "GEO"),
            "{err:?}"
        );
        assert!(err.to_string().contains("GEO"), "{err}");
        assert!(err.to_string().contains("unsupported"), "{err}");
    }

    #[test]
    fn dimension_coord_count_mismatch_is_error() {
        // DIMENSION says 3 but only 2 coordinate lines are given.
        let text = "NAME: m\nTYPE: TSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: EUC_2D\n\
                     NODE_COORD_SECTION\n1 0.0 0.0\n2 1.0 1.0\nEOF\n";
        let err = Tsp::from_tsplib(text).unwrap_err();
        assert!(
            matches!(err, TspError::CoordCountMismatch { expected: 3, actual: 2 }),
            "{err:?}"
        );
    }

    #[test]
    fn missing_dimension_is_error() {
        let text = "NAME: m\nTYPE: TSP\nEDGE_WEIGHT_TYPE: EUC_2D\n\
                     NODE_COORD_SECTION\n1 0.0 0.0\nEOF\n";
        assert!(matches!(Tsp::from_tsplib(text), Err(TspError::MissingField("DIMENSION"))));
    }

    #[test]
    fn node_index_out_of_range_is_error() {
        let text = "NAME: m\nTYPE: TSP\nDIMENSION: 2\nEDGE_WEIGHT_TYPE: EUC_2D\n\
                     NODE_COORD_SECTION\n1 0.0 0.0\n3 1.0 1.0\nEOF\n";
        assert!(matches!(
            Tsp::from_tsplib(text),
            Err(TspError::NodeIndexOutOfRange { id: 3, n: 2 })
        ));
    }

    #[test]
    fn vendored_unknown_name_is_error() {
        let err = Tsp::vendored("nope").unwrap_err();
        assert!(matches!(&err, TspError::UnknownVendored(n) if n == "nope"), "{err:?}");
    }

    #[test]
    fn eil51_and_st70_vendored_parse_with_expected_dimension() {
        let eil51 = Tsp::vendored("eil51").unwrap();
        assert_eq!(eil51.n_cities(), 51);
        assert_eq!(eil51.known_optimum(), Some(426.0));
        let st70 = Tsp::vendored("st70").unwrap();
        assert_eq!(st70.n_cities(), 70);
        assert_eq!(st70.known_optimum(), Some(675.0));
    }

    // ---- EUC_2D nint distance: hand fixtures from vendored berlin52 ----
    // TSPLIB95 (module doc, Sec. 2.1 quoted): dij = nint(sqrt(xd*xd+yd*yd)),
    // nint(x) = (int)(x+0.5). Coordinates below are read straight from
    // data/tsplib/berlin52.tsp (genotype index = TSPLIB node - 1).

    #[test]
    fn euc2d_fixture_cities_1_2_ordinary_rounding() {
        // node 1 (565.0,575.0), node 2 (25.0,185.0):
        // xd=540.0, yd=390.0, xd^2+yd^2=291600+152100=443700
        // sqrt(443700) = 666.1080993352356...
        // nint: (int)(666.1080993352356 + 0.5) = (int)(666.6080993352356) = 666
        let tsp = Tsp::vendored("berlin52").unwrap();
        let d = euc_2d(tsp.coords()[0], tsp.coords()[1]);
        assert_eq!(d, 666.0);
    }

    #[test]
    fn euc2d_fixture_cities_33_38_fractional_sqrt_just_below_half() {
        // node 33 (1150.0,1160.0), node 38 (795.0,645.0):
        // xd=355.0, yd=515.0, xd^2+yd^2=126025+265225=391250
        // sqrt(391250) = 625.4998001598402... (fractional part 0.49980...,
        // i.e. JUST BELOW the .5 boundary)
        // nint: (int)(625.4998001598402 + 0.5) = (int)(625.9998001598402) = 625
        // -- rounds DOWN despite being within 0.0002 of the boundary.
        let tsp = Tsp::vendored("berlin52").unwrap();
        let d = euc_2d(tsp.coords()[32], tsp.coords()[37]);
        assert_eq!(d, 625.0);
    }

    #[test]
    fn euc2d_fixture_cities_47_49_fractional_sqrt_just_above_half() {
        // node 47 (1170.0,65.0), node 49 (605.0,625.0):
        // xd=565.0, yd=-560.0, xd^2+yd^2=319225+313600=632825
        // sqrt(632825) = 795.5029855380808... (fractional part 0.50299...,
        // i.e. JUST ABOVE the .5 boundary)
        // nint: (int)(795.5029855380808 + 0.5) = (int)(796.0029855380808) = 796
        // -- rounds UP, the mirror case of the 33/38 pair above.
        let tsp = Tsp::vendored("berlin52").unwrap();
        let d = euc_2d(tsp.coords()[46], tsp.coords()[48]);
        assert_eq!(d, 796.0);
    }

    // ---- THE GOLDEN: published-optimal tour length, through Tsp's own
    // pipeline (parses the vendored .opt.tour TOUR_SECTION per the pinned
    // grammar: 1-based node numbers, one per line, terminated by -1). ----

    /// TOUR_SECTION parser (module doc, Sec. 1.2.7, quoted): "Each tour is
    /// given by a list of integers giving the sequence in which the nodes
    /// are visited in this tour. Every such tour is terminated by a -1."
    /// This is a TEST-ONLY helper (not part of Tsp's public surface, which
    /// only reads `.tsp` instance files, per the brief) that converts the
    /// 1-based `.opt.tour` node list into the 0-based `BlockValues::Perm`
    /// genotype [`Tsp::evaluate_batch`] expects (module doc mapping).
    fn parse_opt_tour(text: &str) -> Vec<u32> {
        let mut lines = text.lines().map(str::trim);
        while let Some(line) = lines.next() {
            if line == "TOUR_SECTION" {
                let mut order = Vec::new();
                for l in lines {
                    let l = l.trim();
                    if l.is_empty() {
                        continue;
                    }
                    let v: i64 = l.parse().expect("TOUR_SECTION line must be an integer");
                    if v == -1 {
                        break;
                    }
                    order.push((v - 1) as u32); // 1-based TSPLIB -> 0-based genotype
                }
                return order;
            }
        }
        panic!("no TOUR_SECTION found");
    }

    const BERLIN52_OPT_TOUR: &str = include_str!("../data/tsplib/berlin52.opt.tour");
    const EIL51_OPT_TOUR: &str = include_str!("../data/tsplib/eil51.opt.tour");
    const ST70_OPT_TOUR: &str = include_str!("../data/tsplib/st70.opt.tour");

    #[test]
    fn berlin52_published_optimal_tour_evaluates_to_exactly_7542() {
        let tsp = Tsp::vendored("berlin52").unwrap();
        let order = parse_opt_tour(BERLIN52_OPT_TOUR);
        assert_eq!(order.len(), 52);
        let out = tsp.evaluate_batch(&[perm(&order)]);
        assert_eq!(out, vec![7542.0]);
        assert_eq!(tsp.known_optimum(), Some(7542.0));
    }

    #[test]
    fn eil51_published_optimal_tour_evaluates_to_exactly_426() {
        // data/tsplib/eil51.opt.tour carries "COMMENT : Optimal tour for
        // eil51.tsp  (426)" -- matching the module doc's Table 1 quote.
        let tsp = Tsp::vendored("eil51").unwrap();
        let order = parse_opt_tour(EIL51_OPT_TOUR);
        assert_eq!(order.len(), 51);
        let out = tsp.evaluate_batch(&[perm(&order)]);
        assert_eq!(out, vec![426.0]);
    }

    #[test]
    fn st70_published_optimal_tour_evaluates_to_exactly_675() {
        // data/tsplib/st70.opt.tour carries "COMMENT : Optimal tour for
        // st70 (675)" -- matching the module doc's Table 1 quote.
        let tsp = Tsp::vendored("st70").unwrap();
        let order = parse_opt_tour(ST70_OPT_TOUR);
        assert_eq!(order.len(), 70);
        let out = tsp.evaluate_batch(&[perm(&order)]);
        assert_eq!(out, vec![675.0]);
    }

    // ---- rotation / reversal invariance (closed, symmetric tour) ----

    #[test]
    fn tour_length_is_rotation_invariant() {
        let tsp = Tsp::vendored("berlin52").unwrap();
        let order = parse_opt_tour(BERLIN52_OPT_TOUR);
        let base = tsp.evaluate_batch(&[perm(&order)])[0];
        for k in [1usize, 17, 51] {
            let mut rotated = order.clone();
            rotated.rotate_left(k);
            let rotated_len = tsp.evaluate_batch(&[perm(&rotated)])[0];
            assert_eq!(rotated_len, base, "rotation by {k}");
        }
    }

    #[test]
    fn tour_length_is_reversal_invariant() {
        // EUC_2D is symmetric (dij == dji), so reversing visit order must
        // not change the closed-tour length.
        let tsp = Tsp::vendored("berlin52").unwrap();
        let order = parse_opt_tour(BERLIN52_OPT_TOUR);
        let base = tsp.evaluate_batch(&[perm(&order)])[0];
        let mut reversed = order.clone();
        reversed.reverse();
        let reversed_len = tsp.evaluate_batch(&[perm(&reversed)])[0];
        assert_eq!(reversed_len, base);
    }

    // ---- Evaluator integration ----

    #[test]
    fn evaluator_counts_budget_and_tracks_best() {
        let tsp = Tsp::vendored("berlin52").unwrap();
        let order = parse_opt_tour(BERLIN52_OPT_TOUR);
        let mut worse = order.clone();
        worse.swap(0, 1); // any transposition of the optimal tour is >= optimal
        let mut ev = Evaluator::new(&tsp, 10);
        ev.evaluate(&[perm(&order), perm(&worse)]).unwrap();
        assert_eq!(ev.used(), 2);
        assert_eq!(ev.best_so_far(), Some(7542.0));
    }

    #[test]
    fn problem_optimum_maps_to_known_optimum() {
        let tsp = Tsp::vendored("berlin52").unwrap();
        assert_eq!(Problem::optimum(&tsp), tsp.known_optimum());
        assert_eq!(Problem::optimum(&tsp), Some(7542.0));
        // from_tsplib on arbitrary text has no known optimum.
        let text = "NAME: m\nTYPE: TSP\nDIMENSION: 2\nEDGE_WEIGHT_TYPE: EUC_2D\n\
                     NODE_COORD_SECTION\n1 0.0 0.0\n2 1.0 1.0\nEOF\n";
        let parsed = Tsp::from_tsplib(text).unwrap();
        assert_eq!(Problem::optimum(&parsed), None);
    }

    // ---- malformed genotype handling (documented, module doc) ----

    #[test]
    fn wrong_length_permutation_evaluates_to_infinity_not_panic() {
        let tsp = Tsp::vendored("berlin52").unwrap();
        let out = tsp.evaluate_batch(&[perm(&[0, 1, 2])]);
        assert_eq!(out, vec![f64::INFINITY]);
    }

    #[test]
    fn out_of_range_index_evaluates_to_infinity_not_panic() {
        let tsp = Tsp::vendored("eil51").unwrap();
        let mut order: Vec<u32> = (0..51).collect();
        order[0] = 999; // out of range for n=51
        let out = tsp.evaluate_batch(&[perm(&order)]);
        assert_eq!(out, vec![f64::INFINITY]);
    }

    #[test]
    fn non_permutation_block_evaluates_to_infinity() {
        let tsp = Tsp::vendored("eil51").unwrap();
        let bad = Genotype { blocks: vec![BlockValues::Float(vec![0.0; 51])] };
        assert_eq!(tsp.evaluate_batch(&[bad]), vec![f64::INFINITY]);
    }

    #[test]
    fn space_is_one_permutation_block_of_n_cities() {
        let tsp = Tsp::vendored("st70").unwrap();
        assert_eq!(tsp.space().blocks(), &[Block::Permutation { n: 70 }]);
        assert_eq!(tsp.space().dim(), 70);
    }
}
