//! Archive-first multi-objective (MO) run logging: format **"sezgi-moa
//! v1"** (M3-7 Task 9).
//!
//! ## Provenance (cited, no compatibility claim)
//!
//! This format's PHILOSOPHY -- log every nondominated point ever found
//! during a run, not merely the final population, so post-hoc analysis at
//! ANY evaluation budget is possible without re-running the algorithm -- is
//! the "archive-first" practice of two published sources, cited here for
//! the design idea alone:
//!
//! - **COCO bbob-biobj's `.adat` archive files**: Brockhoff, Auger, Hansen,
//!   Tušar, "Biobjective Performance Assessment with the COCO Platform",
//!   arXiv:1605.01746. The `.adat` file records every point that ever
//!   entered the algorithm's unbounded nondominated archive, tagged with
//!   its evaluation count, so indicators (hypervolume relative to a
//!   reference point) can be recomputed at any budget from the archive
//!   trajectory alone.
//! - **MO-IOHinspector's "unbounded archiving"**: de Nobel et al.,
//!   arXiv:2412.07444. Indicators are computed POST-HOC from the archive
//!   trajectory, not logged inline at runtime -- the same split this crate
//!   pins (module doc below, "Indicators are out of scope").
//!
//! **No compatibility claim.** "sezgi-moa v1" is NOT a `.adat` file and is
//! NOT an IOHinspector-family format: it shares neither COCO's whitespace
//! grammar nor MO-IOHinspector's JSON/CSV layout, and no reader for either
//! upstream format can parse a "sezgi-moa v1" file (nor the reverse). Only
//! the ARCHIVE-FIRST DESIGN IDEA is borrowed, cited above; the on-disk
//! grammar below is this crate's own, designed to match this workspace's
//! own existing conventions (`crate::ioh`'s writer style: plain
//! whitespace-delimited text, LF endings, one file per logical unit).
//!
//! ## Indicators are out of scope (scope rulings 5+6)
//!
//! This module logs coordinates only (eval index, objective vector,
//! genotype) -- it never computes hypervolume, IGD, or any other indicator
//! at write time. Indicators are computed POST-HOC, against an EXPLICIT
//! reference point the caller supplies at analysis time (mirroring
//! MO-IOHinspector's own split, cited above) -- consistent with this
//! crate's existing `sezgi_stats::hypervolume_2d`/`igd`, which already take
//! their reference point/front as explicit arguments rather than reading
//! one from a log file.
//!
//! ## Format grammar
//!
//! One file per run. Text, LF line endings (`writeln!`'s own convention on
//! every platform -- never platform-dependent CRLF). Exactly 7 header
//! lines, in this FIXED order, followed by zero or more record lines (one
//! per archive INSERTION, in eval order):
//!
//! ```text
//! sezgi-moa v1
//! algo <algo>
//! problem <problem label>
//! m <n_objectives>
//! seed <seed>
//! budget <budget>
//! kind <float|binary>
//! <eval_index> <obj_0> <obj_1> ... <obj_{m-1}> | <genotype>
//! <eval_index> <obj_0> <obj_1> ... <obj_{m-1}> | <genotype>
//! ...
//! ```
//!
//! `<genotype>` is `kind`-dependent (see "Genotype serialization" below).
//! Every f64 (`<obj_i>` and, for `kind=float`, every genotype value) is
//! written via `{v}` (`f64`'s own `Display`) and read back via
//! `str::parse::<f64>()` -- the SAME convention `crate::ioh`/`crate::ioh_read`
//! already pin (see `ioh_read.rs`'s own module doc: "a correctly-rounded
//! parse", matching this workspace's `float_roundtrip`
//! `serde_json` feature used elsewhere): Rust's `Display` for `f64`
//! produces the SHORTEST decimal string that round-trips back to the exact
//! same bit pattern, and `str::parse` is its correctly-rounded inverse --
//! together, bit-faithful (`%.17g`-class: never decimal-truncated,
//! per the M3-2 precedent this crate already follows) without this module
//! needing to hand-roll a fixed-precision formatter.
//!
//! ## `// sezgi decision:` design choices
//!
//! **Trajectory, not compaction.** A new point is inserted into the archive
//! (and thus written to the file) iff it is not dominated by any CURRENT
//! archive member at the time it is considered; once written, a record is
//! NEVER deleted or rewritten, even after a later point dominates it and
//! removes it from the CURRENT (in-memory) archive. The file is a
//! trajectory log of every point ever admitted, not a live snapshot of "the
//! archive" at end-of-run -- exactly COCO's `.adat` philosophy (cited
//! above): a reader reconstructs the CURRENT archive at any evaluation
//! budget by dominance-filtering the file's prefix up to that budget
//! ([`MoArchiveRun::archive_at`]). This is provably equivalent to replaying
//! the writer's own incremental insert-and-prune algorithm on that prefix:
//! a point `x` that was dominated at consideration time by an
//! already-archived `y` is NEVER written at all (so it can never
//! resurface), and a point `x` that WAS written but is later dominated by a
//! `y` written at a later eval index is correctly excluded by
//! `archive_at(evals)` for any `evals >= y`'s eval index (since `y` is
//! present in that prefix and pairwise-dominates `x`), while still correctly
//! INCLUDED for any earlier `evals < y`'s eval index (since `y` has not yet
//! entered the prefix) -- i.e. `archive_at` reconstructs the archive's
//! HISTORICAL state at that budget, not merely its final state clipped.
//!
//! **Timestamp: dropped, in favor of byte-determinism.** The header carries
//! NO timestamp field. T11's cross-language byte-identity requirement (same
//! `(algo, label, m, seed, budget)` run must produce byte-identical files
//! across implementations/re-runs) is fundamentally incompatible with a
//! genuine wall-clock timestamp (it is never reproducible), and a FAKE
//! fixed-format placeholder (`"1970-01-01T00:00:00Z"` or similar) would be
//! actively misleading -- a reader could reasonably assume a `timestamp`
//! field means "when this run happened". Dropping it entirely is the
//! honest choice: [`MoArchiveWriter::create`]'s header is a pure function
//! of `(algo, problem_label, m, seed, budget, kind)`, which is exactly what
//! [`same_seed_produces_byte_identical_files`] (this module's test) checks.
//!
//! **Feasibility gate: archive membership is feasible-only.** COCO
//! bbob-biobj (cited above) is an UNCONSTRAINED suite -- it has no
//! constraint-violation concept to gate on. `sezgi_components::nsga2`'s own
//! constrained-domination support (M3-7 Task 1) makes an infeasible point
//! incomparable to a feasible one under the SAME total order a feasible-only
//! archive already assumes (a feasible point always dominates an infeasible
//! one outright, per that module's `dominates_constrained`), so admitting
//! infeasible points into this UNCONSTRAINED-dominance archive would let an
//! infeasible point with merely "good" raw objectives sit in the archive
//! forever, never displaced by the (feasible) points NSGA-II's own
//! constrained selection actually prefers. Feasible-only membership is the
//! defensible choice: this module's archive machinery itself
//! ([`MoArchiveWriter::consider`]) knows nothing about constraints (it only
//! ever sees an objective vector) -- the gate is applied by the CALLER,
//! [`nsga2_run_logged`] below, which skips `consider()` entirely for any
//! individual whose [`sezgi_components::nsga2`] constraint-violation scalar
//! is `< 0.0` (infeasible; `MoProblem::evaluate_constraints_batch`'s own
//! `g_j >= 0` feasible convention, threaded through as
//! `Nsga2::MoBatchObserver`'s `Option<&[f64]>` violations slice). An
//! unconstrained problem (`violations` slice absent) offers every
//! individual, unchanged.
//!
//! **Genotype serialization.** Both this crate's `nsga2_run`/`nsga2_run_observed`
//! restrict a run's search space to ALL-`Block::Float` or ALL-`Block::Binary`
//! (`sezgi_components::nsga2`'s own module doc, "Space classification and
//! error design") -- so a single run's genotype KIND never varies record to
//! record, and the header's `kind` field is enough context to parse every
//! record's genotype column with no further lookup. `kind=float`: every
//! `Block::Float` block's values, flattened in `SearchSpace::blocks()`
//! order (block-major, variable-minor -- this module's own established
//! convention, matching `sezgi_components::nsga2`'s binary-path loop
//! nesting), space-separated, `f64` `Display`-formatted (same bit-faithful
//! convention as the objective columns). `kind=binary`: every
//! `Block::Binary` block's bits, flattened in the SAME block-major/bit-minor
//! order, as a single contiguous token of `'0'`/`'1'` characters (no
//! separator needed -- a bit has no precision to preserve, unlike a float,
//! so a compact fixed-width token is unambiguous and needs no
//! `str::parse::<f64>` machinery at all).
//!
//! ## File naming
//!
//! `<log_dir>/<label>-s<seed>.moa` ([`nsga2_run_logged`]) -- mirrors
//! `crate::ioh::IohLogger`'s own convention of embedding every identifying
//! field directly in a human-greppable path (`IOHprofiler_f{fid}_DIM{dim}.dat`)
//! rather than a content hash or opaque id; `.moa` names the format
//! ("sezgi-moa"), matching `.dat`/`.adat`'s own precedent of a
//! format-specific extension.

use sezgi_components::nsga2::{
    dominates, nsga2_run_observed, MoBatchObserver, MoRunResult, Nsga2Config, Nsga2Error,
};
use sezgi_core::mo::MoProblem;
use sezgi_core::space::{Block, BlockValues, Genotype, SearchSpace};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

/// This format's own version tag -- the header's first line, verbatim.
pub const MOA_FORMAT_TAG: &str = "sezgi-moa v1";

/// A run's genotype representation (see the module doc's "Genotype
/// serialization" section) -- fixed for the whole file, carried in the
/// header's `kind` line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenoKind {
    Float,
    Binary,
}

impl GenoKind {
    fn as_str(self) -> &'static str {
        match self {
            GenoKind::Float => "float",
            GenoKind::Binary => "binary",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MoArchiveError {
    #[error("{path}: {source}")]
    Io { path: PathBuf, source: io::Error },
    #[error("{path}: {reason}")]
    Format { path: PathBuf, reason: String },
}

/// A streaming "sezgi-moa v1" writer: one file per run, header written by
/// [`Self::create`], one record line per [`Self::consider`] call that
/// results in an archive insertion. Uses a `BufWriter` (this module's own
/// streaming requirement: write on insertion, no end-of-run buffering of
/// the WHOLE run beyond the normal `BufWriter` block size); call
/// [`Self::finish`] to flush and surface any pending IO error explicitly
/// (relying on `BufWriter`'s `Drop` alone would silently swallow a flush
/// failure).
pub struct MoArchiveWriter {
    file: BufWriter<File>,
    path: PathBuf,
    kind: GenoKind,
    m: usize,
    /// The CURRENT (in-memory) nondominated archive's objective vectors --
    /// used only for the next candidate's dominance check; never itself
    /// written to disk (only records that pass [`Self::consider`]'s
    /// insertion test are written). See the module doc's "Trajectory, not
    /// compaction" decision for why this working set intentionally diverges
    /// from the file's own contents over the run.
    working: Vec<Vec<f64>>,
}

impl MoArchiveWriter {
    /// Creates `path` (truncating any existing file, matching
    /// `std::fs::File::create`'s own convention) and writes the 7-line
    /// header. See the module doc's "Timestamp: dropped" decision for why
    /// there is no 8th header line.
    pub fn create(
        path: impl AsRef<Path>,
        algo: &str,
        problem_label: &str,
        m: usize,
        seed: u64,
        budget: u64,
        kind: GenoKind,
    ) -> Result<Self, MoArchiveError> {
        let path = path.as_ref().to_path_buf();
        let file = File::create(&path).map_err(|e| MoArchiveError::Io { path: path.clone(), source: e })?;
        let mut file = BufWriter::new(file);
        let write_header = |file: &mut BufWriter<File>| -> io::Result<()> {
            writeln!(file, "{MOA_FORMAT_TAG}")?;
            writeln!(file, "algo {algo}")?;
            writeln!(file, "problem {problem_label}")?;
            writeln!(file, "m {m}")?;
            writeln!(file, "seed {seed}")?;
            writeln!(file, "budget {budget}")?;
            writeln!(file, "kind {}", kind.as_str())?;
            Ok(())
        };
        write_header(&mut file).map_err(|e| MoArchiveError::Io { path: path.clone(), source: e })?;
        Ok(Self { file, path, kind, m, working: Vec::new() })
    }

    /// Considers one evaluated individual for archive insertion: `obj` is
    /// admitted (and written, along with `eval_index` and `geno`'s
    /// serialization) iff it is not dominated by any point currently in the
    /// working archive (plain unconstrained Pareto dominance --
    /// [`sezgi_components::nsga2::dominates`], reused rather than
    /// duplicated). On admission, every working-archive member `obj` itself
    /// now dominates is pruned from the IN-MEMORY working set (never from
    /// the file -- module doc's "Trajectory, not compaction"). Returns
    /// `Ok(true)` iff `obj` was admitted (and thus written).
    ///
    /// Feasibility is NOT this method's concern -- see the module doc's
    /// "Feasibility gate" decision: the caller decides which individuals to
    /// offer.
    pub fn consider(&mut self, eval_index: u64, geno: &Genotype, obj: &[f64]) -> Result<bool, MoArchiveError> {
        debug_assert_eq!(obj.len(), self.m, "consider: objective vector length must equal the header's own m");
        if self.working.iter().any(|w| dominates(w, obj)) {
            return Ok(false);
        }
        self.working.retain(|w| !dominates(obj, w));
        self.working.push(obj.to_vec());

        let geno_str = serialize_genotype(geno, self.kind);
        let write_record = |file: &mut BufWriter<File>| -> io::Result<()> {
            write!(file, "{eval_index}")?;
            for o in obj {
                write!(file, " {o}")?;
            }
            write!(file, " | {geno_str}")?;
            writeln!(file)?;
            Ok(())
        };
        write_record(&mut self.file).map_err(|e| MoArchiveError::Io { path: self.path.clone(), source: e })?;
        Ok(true)
    }

    /// Flushes the underlying `BufWriter` and surfaces any pending IO
    /// error. See the struct doc for why this is not left to `Drop` alone.
    pub fn finish(mut self) -> Result<(), MoArchiveError> {
        self.file.flush().map_err(|e| MoArchiveError::Io { path: self.path.clone(), source: e })
    }
}

/// `geno`'s serialized genotype column, per the module doc's "Genotype
/// serialization" section. A block whose `BlockValues` variant does not
/// match `kind` is silently skipped -- unreachable in practice, since
/// [`nsga2_run_logged`] derives `kind` from the SAME `SearchSpace`
/// `nsga2_run_observed` itself validates as all-Float or all-Binary before
/// any record is ever written (a space that is neither, or genuinely
/// mixed, errors out of `nsga2_run_observed` before `consider` is ever
/// called).
fn serialize_genotype(geno: &Genotype, kind: GenoKind) -> String {
    match kind {
        GenoKind::Float => {
            let mut parts: Vec<String> = Vec::new();
            for block in &geno.blocks {
                if let BlockValues::Float(xs) = block {
                    parts.extend(xs.iter().map(f64::to_string));
                }
            }
            parts.join(" ")
        }
        GenoKind::Binary => {
            let mut s = String::new();
            for block in &geno.blocks {
                if let BlockValues::Bin(bits) = block {
                    s.extend(bits.iter().map(|&b| if b { '1' } else { '0' }));
                }
            }
            s
        }
    }
}

/// One decoded genotype column (module doc: `kind`-dependent). Unlike the
/// original [`Genotype`], this carries no [`SearchSpace`] block structure
/// (the reader has no `SearchSpace` to reconstruct it against) -- just the
/// flattened values/bits, in the SAME order [`serialize_genotype`] wrote
/// them.
#[derive(Debug, Clone, PartialEq)]
pub enum MoArchiveGenotype {
    Float(Vec<f64>),
    Binary(Vec<bool>),
}

/// One record: one archive insertion.
#[derive(Debug, Clone, PartialEq)]
pub struct MoArchiveRecord {
    pub eval_index: u64,
    pub objectives: Vec<f64>,
    pub genotype: MoArchiveGenotype,
}

/// A fully-read "sezgi-moa v1" file: header fields plus every record, in
/// file order (== eval order, [`MoArchiveWriter::consider`]'s own writing
/// order).
#[derive(Debug, Clone, PartialEq)]
pub struct MoArchiveRun {
    pub algo: String,
    pub problem: String,
    pub m: usize,
    pub seed: u64,
    pub budget: u64,
    pub kind: GenoKind,
    pub records: Vec<MoArchiveRecord>,
}

impl MoArchiveRun {
    /// Reconstructs the CURRENT nondominated archive at evaluation budget
    /// `evals`: every record with `eval_index <= evals`, dominance-filtered
    /// down to its own mutually-nondominated subset (plain Pareto
    /// dominance -- this file's archive is feasible-only by construction,
    /// module doc's "Feasibility gate" decision, so no constraint channel
    /// is needed here). See the module doc's "Trajectory, not compaction"
    /// decision for why this batch re-filter of the whole prefix is
    /// PROVABLY equivalent to the writer's own incremental
    /// insert-and-prune. Order: the surviving records' own file order
    /// (stable, deterministic for a given file). Duplicate objective
    /// vectors (two records with bit-identical objectives, neither
    /// dominating the other) are NOT deduplicated -- each survives
    /// independently if it survives at all, mirroring the writer's own
    /// "insert iff not dominated" rule (a duplicate is never dominated by
    /// its own twin).
    pub fn archive_at(&self, evals: u64) -> Vec<Vec<f64>> {
        let prefix: Vec<&Vec<f64>> =
            self.records.iter().filter(|r| r.eval_index <= evals).map(|r| &r.objectives).collect();
        prefix
            .iter()
            .enumerate()
            .filter(|&(i, obj)| !prefix.iter().enumerate().any(|(j, other)| j != i && dominates(other, obj)))
            .map(|(_, obj)| (*obj).clone())
            .collect()
    }
}

/// Reads a "sezgi-moa v1" file written by [`MoArchiveWriter`]. See the
/// module doc's format grammar for the exact header/record layout this
/// parses.
pub fn read_moa(path: impl AsRef<Path>) -> Result<MoArchiveRun, MoArchiveError> {
    let path = path.as_ref().to_path_buf();
    let text = std::fs::read_to_string(&path).map_err(|e| MoArchiveError::Io { path: path.clone(), source: e })?;
    let fmt_err = |reason: String| MoArchiveError::Format { path: path.clone(), reason };

    let mut lines = text.lines();
    let tag = lines.next().ok_or_else(|| fmt_err("empty file: expected a \"sezgi-moa v1\" header".into()))?;
    if tag != MOA_FORMAT_TAG {
        return Err(fmt_err(format!("unexpected format tag {tag:?}, expected {MOA_FORMAT_TAG:?}")));
    }

    let mut next_kv = |key: &str| -> Result<String, MoArchiveError> {
        let line = lines.next().ok_or_else(|| fmt_err(format!("missing {key:?} header line")))?;
        let prefix = format!("{key} ");
        line.strip_prefix(prefix.as_str())
            .map(str::to_string)
            .ok_or_else(|| fmt_err(format!("expected a {key:?} header line, got {line:?}")))
    };

    let algo = next_kv("algo")?;
    let problem = next_kv("problem")?;
    let m: usize = next_kv("m")?.parse().map_err(|_| fmt_err("invalid m header value".into()))?;
    let seed: u64 = next_kv("seed")?.parse().map_err(|_| fmt_err("invalid seed header value".into()))?;
    let budget: u64 = next_kv("budget")?.parse().map_err(|_| fmt_err("invalid budget header value".into()))?;
    let kind = match next_kv("kind")?.as_str() {
        "float" => GenoKind::Float,
        "binary" => GenoKind::Binary,
        other => return Err(fmt_err(format!("unknown kind {other:?}, expected \"float\" or \"binary\""))),
    };

    let mut records = Vec::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        let (left, right) = line
            .split_once('|')
            .ok_or_else(|| fmt_err(format!("record line missing '|' separator: {line:?}")))?;
        let mut tokens = left.split_whitespace();
        let eval_index: u64 = tokens
            .next()
            .ok_or_else(|| fmt_err(format!("record line missing eval index: {line:?}")))?
            .parse()
            .map_err(|_| fmt_err(format!("invalid eval index in record: {line:?}")))?;
        let mut objectives = Vec::with_capacity(m);
        for tok in tokens {
            let v: f64 = tok.parse().map_err(|_| fmt_err(format!("invalid objective value in record: {line:?}")))?;
            objectives.push(v);
        }
        if objectives.len() != m {
            return Err(fmt_err(format!(
                "record has {} objectives, expected m={m}: {line:?}",
                objectives.len()
            )));
        }
        let geno_str = right.trim();
        let genotype = match kind {
            GenoKind::Float => {
                let mut xs = Vec::new();
                for tok in geno_str.split_whitespace() {
                    let v: f64 =
                        tok.parse().map_err(|_| fmt_err(format!("invalid genotype value in record: {line:?}")))?;
                    xs.push(v);
                }
                MoArchiveGenotype::Float(xs)
            }
            GenoKind::Binary => {
                let mut bits = Vec::with_capacity(geno_str.len());
                for c in geno_str.chars() {
                    match c {
                        '0' => bits.push(false),
                        '1' => bits.push(true),
                        other => {
                            return Err(fmt_err(format!(
                                "invalid binary genotype character {other:?} in record: {line:?}"
                            )))
                        }
                    }
                }
                MoArchiveGenotype::Binary(bits)
            }
        };
        records.push(MoArchiveRecord { eval_index, objectives, genotype });
    }

    Ok(MoArchiveRun { algo, problem, m, seed, budget, kind, records })
}

/// [`nsga2_run_logged`]'s own hardcoded `algo` header field. **sezgi
/// decision:** `nsga2_run_logged` has no `algo` parameter (the PINNED
/// interface, task brief) -- this wrapper is NSGA-II-specific by
/// construction (it calls `nsga2_run_observed` directly), so the header's
/// `algo` field is simply this fixed literal rather than a caller-supplied
/// string that could disagree with what actually ran.
const NSGA2_ALGO_NAME: &str = "nsga2";

#[derive(Debug, thiserror::Error)]
pub enum MoRunLoggedError {
    #[error(transparent)]
    Nsga2(#[from] Nsga2Error),
    #[error(transparent)]
    Archive(#[from] MoArchiveError),
}

/// This run's [`GenoKind`], derived from `space` alone: all-`Block::Binary`
/// is [`GenoKind::Binary`], anything else (including all-`Block::Float`,
/// mixed, or another block kind) is [`GenoKind::Float`]. This is a
/// PRE-VALIDATION heuristic only, used to pick the archive header's `kind`
/// line before [`nsga2_run_observed`]'s OWN space classification runs (the
/// archive file must exist, header written, before the observer can be
/// wired in) -- an invalid space (mixed, or neither Float nor Binary)
/// always errors out of `nsga2_run_observed` before a single record is
/// ever considered, leaving a header-only file behind (not deleted; see
/// [`nsga2_run_logged`]'s own doc).
fn geno_kind_of(space: &SearchSpace) -> GenoKind {
    if !space.blocks().is_empty() && space.blocks().iter().all(|b| matches!(b, Block::Binary { .. })) {
        GenoKind::Binary
    } else {
        GenoKind::Float
    }
}

/// Thin archive-logging wrapper around [`nsga2_run_observed`]: runs NSGA-II
/// exactly as [`sezgi_components::nsga2::nsga2_run`] would (same RNG draws,
/// same result -- see this module's [`logged_run_matches_unlogged_run`]
/// test), while streaming every feasible archive insertion to
/// `<log_dir>/<label>-s<seed>.moa` (module doc's "File naming" section).
///
/// Feasibility gating happens HERE (module doc's "Feasibility gate"
/// decision), not inside [`MoArchiveWriter`]: an individual is offered to
/// the archive iff the problem is unconstrained (`violations` absent) or
/// its own violation scalar is `== 0.0` (fully feasible, per
/// `sezgi_components::nsga2::constraint_violation`'s own `<= 0.0`, `0.0` =
/// feasible convention).
pub fn nsga2_run_logged(
    problem: &dyn MoProblem,
    cfg: &Nsga2Config,
    log_dir: &Path,
    label: &str,
) -> Result<MoRunResult, MoRunLoggedError> {
    std::fs::create_dir_all(log_dir)
        .map_err(|e| MoArchiveError::Io { path: log_dir.to_path_buf(), source: e })?;
    let kind = geno_kind_of(problem.space());
    let path = log_dir.join(format!("{label}-s{seed}.moa", seed = cfg.seed));
    let mut writer =
        MoArchiveWriter::create(&path, NSGA2_ALGO_NAME, label, problem.n_objectives(), cfg.seed, cfg.budget, kind)?;

    let mut write_err: Option<MoArchiveError> = None;
    let result = {
        let mut observe: Box<MoBatchObserver> = Box::new(
            |first_eval_index: u64, genos: &[Genotype], objs: &[Vec<f64>], viol: Option<&[f64]>| {
                if write_err.is_some() {
                    return;
                }
                for i in 0..genos.len() {
                    let feasible = viol.map(|v| v[i] == 0.0).unwrap_or(true);
                    if !feasible {
                        continue;
                    }
                    if let Err(e) = writer.consider(first_eval_index + i as u64, &genos[i], &objs[i]) {
                        write_err = Some(e);
                        return;
                    }
                }
            },
        );
        nsga2_run_observed(problem, cfg, &mut observe)?
    };
    if let Some(e) = write_err {
        return Err(e.into());
    }
    writer.finish()?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sezgi_components::nsga2::nsga2_run;
    use sezgi_core::space::{Block, BlockValues, SearchSpace};
    use sezgi_problems::Zdt;

    fn base_cfg(seed: u64) -> Nsga2Config {
        Nsga2Config {
            pop_size: 8,
            budget: 80,
            seed,
            eta_c: 20.0,
            eta_m: 20.0,
            p_c: 0.9,
            p_m: None,
            p_c_bin: 0.9,
            p_m_bin: None,
        }
    }

    fn g_float(xs: &[f64]) -> Genotype {
        Genotype { blocks: vec![BlockValues::Float(xs.to_vec())] }
    }

    // ---- write/read round-trip: bit-faithful f64 -------------------------

    #[test]
    fn write_read_round_trip_bit_faithful_floats() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("run.moa");
        let mut w = MoArchiveWriter::create(&path, "algoX", "ProbY", 2, 42, 100, GenoKind::Float).unwrap();

        // Tricky values: a value whose shortest round-trip repr is long,
        // negative zero, a negative value, and a very small magnitude.
        let tricky: [f64; 5] = [
            f64::from_bits(0x3FD5555555555555), // not a "nice" decimal
            -0.0,
            -123.456,
            1e-300,
            f64::MAX,
        ];
        for (i, &v) in tricky.iter().enumerate() {
            w.consider(i as u64 + 1, &g_float(&[v]), &[v, -v]).unwrap();
        }
        w.finish().unwrap();

        let run = read_moa(&path).unwrap();
        assert_eq!(run.algo, "algoX");
        assert_eq!(run.problem, "ProbY");
        assert_eq!(run.m, 2);
        assert_eq!(run.seed, 42);
        assert_eq!(run.budget, 100);
        assert_eq!(run.kind, GenoKind::Float);
        assert_eq!(run.records.len(), tricky.len(), "every tricky value must be non-dominated by the others' pair");

        for (rec, &v) in run.records.iter().zip(tricky.iter()) {
            assert_eq!(rec.objectives[0].to_bits(), v.to_bits(), "objective bits must round-trip exactly");
            assert_eq!(rec.objectives[1].to_bits(), (-v).to_bits());
            match &rec.genotype {
                MoArchiveGenotype::Float(xs) => {
                    assert_eq!(xs.len(), 1);
                    assert_eq!(xs[0].to_bits(), v.to_bits(), "genotype bits must round-trip exactly");
                }
                other => panic!("expected Float genotype, got {other:?}"),
            }
        }
    }

    // ---- header round-trip -------------------------------------------------

    #[test]
    fn header_fields_round_trip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("run.moa");
        let w = MoArchiveWriter::create(&path, "nsga2", "ZDT1", 2, 7, 200, GenoKind::Binary).unwrap();
        w.finish().unwrap();
        let run = read_moa(&path).unwrap();
        assert_eq!(run.algo, "nsga2");
        assert_eq!(run.problem, "ZDT1");
        assert_eq!(run.m, 2);
        assert_eq!(run.seed, 7);
        assert_eq!(run.budget, 200);
        assert_eq!(run.kind, GenoKind::Binary);
        assert!(run.records.is_empty());

        let raw = std::fs::read_to_string(&path).unwrap();
        assert_eq!(raw.lines().next().unwrap(), MOA_FORMAT_TAG);
        assert_eq!(raw.matches('\n').count(), raw.lines().count(), "LF endings only, no CR");
    }

    // ---- archive dominance invariant at every prefix ----------------------

    #[test]
    fn archive_dominance_invariant_over_seeded_run() {
        let tmp = tempfile::tempdir().unwrap();
        let problem = Zdt::new(1, 6).unwrap();
        let cfg = base_cfg(11);
        let result = nsga2_run_logged(&problem, &cfg, tmp.path(), "zdt1").unwrap();
        let path = tmp.path().join(format!("zdt1-s{}.moa", cfg.seed));
        let run = read_moa(&path).unwrap();
        assert!(!run.records.is_empty());
        assert!(
            run.records.iter().all(|r| r.eval_index <= result.evals_used),
            "every archived eval index must fall within the run's own charged budget"
        );

        // Property: at every eval index that appears in the file, the
        // reconstructed archive must be mutually nondominated (no member
        // dominates another).
        for evals in run.records.iter().map(|r| r.eval_index) {
            let front = run.archive_at(evals);
            for i in 0..front.len() {
                for j in 0..front.len() {
                    if i != j {
                        assert!(
                            !dominates(&front[i], &front[j]),
                            "archive_at({evals}) is not mutually nondominated: {:?} dominates {:?}",
                            front[i],
                            front[j]
                        );
                    }
                }
            }
        }
    }

    // ---- archive_at correctness on a hand-built trajectory ----------------

    #[test]
    fn archive_at_hand_built_trajectory() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("hand.moa");
        let mut w = MoArchiveWriter::create(&path, "test", "Hand", 2, 0, 100, GenoKind::Float).unwrap();

        // Eval 1: (1.0, 5.0) -- admitted (archive empty).
        assert!(w.consider(1, &g_float(&[0.0]), &[1.0, 5.0]).unwrap());
        // Eval 2: (5.0, 1.0) -- incomparable to eval 1, admitted.
        assert!(w.consider(2, &g_float(&[1.0]), &[5.0, 1.0]).unwrap());
        // Eval 3: (3.0, 3.0) -- incomparable to both, admitted.
        assert!(w.consider(3, &g_float(&[2.0]), &[3.0, 3.0]).unwrap());
        // Eval 4: (2.0, 2.0) -- dominates eval 3's (3.0, 3.0); admitted,
        // eval 3 pruned from the WORKING set (but stays in the file).
        assert!(w.consider(4, &g_float(&[3.0]), &[2.0, 2.0]).unwrap());
        // Eval 5: (4.0, 4.0) -- dominated by eval 4's (2.0, 2.0); rejected,
        // never written.
        assert!(!w.consider(5, &g_float(&[4.0]), &[4.0, 4.0]).unwrap());
        w.finish().unwrap();

        let run = read_moa(&path).unwrap();
        // The file itself holds 4 records (eval 5 was never written).
        assert_eq!(run.records.iter().map(|r| r.eval_index).collect::<Vec<_>>(), vec![1, 2, 3, 4]);

        // Hand-derived expected archives at several cut points.
        let mut at1 = run.archive_at(1);
        at1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(at1, vec![vec![1.0, 5.0]]);

        let mut at2 = run.archive_at(2);
        at2.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(at2, vec![vec![1.0, 5.0], vec![5.0, 1.0]]);

        let mut at3 = run.archive_at(3);
        at3.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(at3, vec![vec![1.0, 5.0], vec![3.0, 3.0], vec![5.0, 1.0]]);

        // At budget 4: eval 3's (3.0, 3.0) is now dominated by eval 4's
        // (2.0, 2.0), so it must be filtered OUT even though its record is
        // still physically present in the file.
        let mut at4 = run.archive_at(4);
        at4.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(at4, vec![vec![1.0, 5.0], vec![2.0, 2.0], vec![5.0, 1.0]]);

        // Querying past the last written eval index is the same as
        // querying at the last one (no eval-5 record exists to include).
        assert_eq!(run.archive_at(100), run.archive_at(4));
    }

    // ---- determinism: same seed -> identical file bytes -------------------

    #[test]
    fn same_seed_produces_byte_identical_files() {
        let tmp1 = tempfile::tempdir().unwrap();
        let tmp2 = tempfile::tempdir().unwrap();
        let problem = Zdt::new(2, 6).unwrap();
        let cfg = base_cfg(99);
        nsga2_run_logged(&problem, &cfg, tmp1.path(), "zdt2").unwrap();
        nsga2_run_logged(&problem, &cfg, tmp2.path(), "zdt2").unwrap();

        let name = format!("zdt2-s{}.moa", cfg.seed);
        let bytes1 = std::fs::read(tmp1.path().join(&name)).unwrap();
        let bytes2 = std::fs::read(tmp2.path().join(&name)).unwrap();
        assert_eq!(bytes1, bytes2, "identical (algo, label, m, seed, budget) must produce byte-identical files");
        assert!(!bytes1.is_empty());
    }

    // ---- unlogged path frozen: logged run matches the plain nsga2_run -----

    #[test]
    fn logged_run_matches_unlogged_run() {
        let tmp = tempfile::tempdir().unwrap();
        let problem = Zdt::new(1, 6).unwrap();
        let cfg = base_cfg(7);

        // Plain, unlogged path (byte-identical to every pre-task caller --
        // this task's own `nsga2.rs` edits never touch this call).
        let unlogged = nsga2_run(&problem, &cfg).unwrap();
        let logged = nsga2_run_logged(&problem, &cfg, tmp.path(), "zdt1").unwrap();

        assert_eq!(unlogged, logged, "attaching an archive observer must not perturb the RNG-driven run at all");
    }

    // ---- constrained problems: feasible-only archive membership -----------

    /// f1 = x0, f2 = -x0 on [-5, 5]; ONE constraint g0 = x0 - 1 (feasible
    /// region x0 >= 1, `g_j >= 0` feasible convention) -- same shape as
    /// `sezgi_core::mo`'s own `OneConstraint` test fixture.
    struct OneConstraint {
        space: SearchSpace,
    }
    impl OneConstraint {
        fn new() -> Self {
            Self { space: SearchSpace::new(vec![Block::Float { lo: -5.0, hi: 5.0, n: 1 }]).unwrap() }
        }
    }
    impl MoProblem for OneConstraint {
        fn space(&self) -> &SearchSpace {
            &self.space
        }
        fn n_objectives(&self) -> usize {
            2
        }
        fn evaluate_batch(&self, pop: &[Genotype]) -> Vec<Vec<f64>> {
            pop.iter()
                .map(|g| {
                    let BlockValues::Float(xs) = &g.blocks[0] else { return vec![f64::INFINITY; 2] };
                    vec![xs[0], -xs[0]]
                })
                .collect()
        }
        fn evaluate_constraints_batch(&self, pop: &[Genotype]) -> Option<Vec<Vec<f64>>> {
            Some(
                pop.iter()
                    .map(|g| {
                        let BlockValues::Float(xs) = &g.blocks[0] else { return vec![f64::NEG_INFINITY] };
                        vec![xs[0] - 1.0]
                    })
                    .collect(),
            )
        }
    }

    #[test]
    fn constrained_problem_archive_admits_feasible_points_only() {
        let tmp = tempfile::tempdir().unwrap();
        let problem = OneConstraint::new();
        let cfg = base_cfg(3);
        nsga2_run_logged(&problem, &cfg, tmp.path(), "onecon").unwrap();
        let run = read_moa(tmp.path().join(format!("onecon-s{}.moa", cfg.seed))).unwrap();
        assert!(!run.records.is_empty(), "the feasible region x0>=1 is non-empty, some individual must qualify");

        for rec in &run.records {
            let MoArchiveGenotype::Float(xs) = &rec.genotype else { panic!("expected Float genotype") };
            let geno = g_float(xs);
            let cons = problem.evaluate_constraints_batch(std::slice::from_ref(&geno)).unwrap();
            let g0 = cons[0][0];
            assert!(g0 >= 0.0, "archived record has x0={:?}, g0={g0} < 0 (infeasible) -- must never be logged", xs);
        }
    }

    // ---- binary genotype: ZDT5 logs correctly ------------------------------

    #[test]
    fn binary_genotype_zdt5_logs_correctly() {
        let tmp = tempfile::tempdir().unwrap();
        let problem = sezgi_problems::zdt::Zdt5::new();
        let cfg = base_cfg(5);
        nsga2_run_logged(&problem, &cfg, tmp.path(), "zdt5").unwrap();
        let run = read_moa(tmp.path().join(format!("zdt5-s{}.moa", cfg.seed))).unwrap();
        assert_eq!(run.kind, GenoKind::Binary);
        assert!(!run.records.is_empty());

        let expected_bits = problem.space().dim();
        for rec in &run.records {
            let MoArchiveGenotype::Binary(bits) = &rec.genotype else { panic!("expected Binary genotype") };
            assert_eq!(bits.len(), expected_bits, "flattened bit count must equal the space's total dim");

            // Re-derive the genotype's blocks (30-bit x1, then ten 5-bit
            // groups -- Zdt5::new's own fixed layout) and re-evaluate: the
            // logged objectives must be bit-exact against a fresh
            // evaluation of the decoded genotype.
            let mut blocks = Vec::new();
            let mut rest = bits.as_slice();
            let (head, tail) = rest.split_at(30);
            blocks.push(BlockValues::Bin(head.to_vec()));
            rest = tail;
            for _ in 0..10 {
                let (grp, tail) = rest.split_at(5);
                blocks.push(BlockValues::Bin(grp.to_vec()));
                rest = tail;
            }
            let geno = Genotype { blocks };
            let recomputed = problem.evaluate_batch(std::slice::from_ref(&geno));
            assert_eq!(recomputed.len(), 1);
            for (a, b) in recomputed[0].iter().zip(rec.objectives.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "re-evaluating the decoded genotype must match the logged objectives bit-exactly");
            }
        }
    }
}
