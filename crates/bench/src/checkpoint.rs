//! Journal-based checkpoint and resume for experiment runs.
//!
//! Implements a JSONL-based run journal that allows resuming an experiment
//! from the last completed run. Each run is recorded as a JSON line, and the
//! journal is validated against the experiment spec's hash (FNV-1a 64-bit) to
//! detect if the spec changed between runs.

use crate::experiment::{
    ExperimentSpec, RunKey, RunRecord, ExperimentError, PlannedRun, enumerate,
    execute_run, build_ioh_observers,
};
use crate::ioh::IohRunObserver;
use serde::{Deserialize, Serialize};
use sezgi_core::problem::EvalObserver;
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

/// Header line in the JSONL journal, identifying the experiment and spec.
#[derive(Debug, Serialize, Deserialize, Clone)]
struct JournalHeader {
    experiment: String,
    spec_hash: String,
}

/// FNV-1a 64-bit hash of the canonical TOML string representation of an
/// [`ExperimentSpec`]. Used to detect spec changes when resuming.
///
/// FNV-1a algorithm: for each byte, XOR with hash then multiply by prime.
/// Constants (pinned):
/// - Offset basis: 0xcbf29ce484222325
/// - Prime: 0x100000001b3
pub fn fnv1a_64(data: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    let mut hash = OFFSET_BASIS;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// Compute the FNV-1a 64-bit hash of an experiment spec's canonical TOML
/// string, returning it as a hex string.
pub fn spec_hash(spec: &ExperimentSpec) -> String {
    let toml = spec.to_toml();
    let hash = fnv1a_64(toml.as_bytes());
    format!("{:016x}", hash)
}

/// Load a JSONL run journal from disk.
///
/// The journal format is:
/// - Line 0: Header JSON `{"experiment": <name>, "spec_hash": <hash>}`
/// - Lines 1+: RunRecord JSON, one per line, appended with flush after each run.
///
/// A final partial/torn line (incomplete JSON at EOF, or a line without a
/// trailing newline) is tolerated and skipped; its count is returned along
/// with the byte offset of the end of the last VALID line (header or record,
/// including its newline). That offset lets a resumer truncate the torn tail
/// away before appending, so a re-executed record is never glued onto the
/// garbage. An unparseable line NOT at the end is a hard error.
/// Empty journal file or file that doesn't exist are treated as "no records".
pub fn load_journal(
    path: &Path,
    expected_name: &str,
    expected_hash: &str,
) -> Result<(Vec<RunRecord>, usize, u64), ExperimentError> {
    if !path.exists() {
        return Ok((Vec::new(), 0, 0));
    }

    let content = fs::read_to_string(path)
        .map_err(|e| ExperimentError::JournalLoad(format!("could not read journal file: {}", e)))?;

    if content.is_empty() {
        return Ok((Vec::new(), 0, 0));
    }

    // Split preserving newlines, so torn (unterminated) final lines are
    // distinguishable and byte offsets can be tracked exactly.
    let segments: Vec<&str> = content.split_inclusive('\n').collect();

    // Read and validate header.
    let header: JournalHeader = serde_json::from_str(segments[0].trim_end())
        .map_err(|e| ExperimentError::JournalLoad(format!("header JSON parse failed: {}", e)))?;

    if header.experiment != expected_name {
        return Err(ExperimentError::JournalLoad(format!(
            "journal experiment name mismatch: expected '{}', got '{}'",
            expected_name, header.experiment
        )));
    }

    if header.spec_hash != expected_hash {
        return Err(ExperimentError::ExperimentHashMismatch {
            expected: expected_hash.to_string(),
            got: header.spec_hash,
        });
    }

    // Read records; tolerate a final torn line. A record line is VALID only
    // if it parses AND is newline-terminated: a parseable final line without
    // a trailing newline is treated as torn too (appending after it would
    // glue the next record onto it). Its run is simply re-executed on resume;
    // the engine is deterministic, so the record comes back bit-identical.
    let mut records = Vec::new();
    let mut torn_count = 0usize;
    let mut valid_end = segments[0].len() as u64;

    for (i, seg) in segments.iter().enumerate().skip(1) {
        let terminated = seg.ends_with('\n');
        match serde_json::from_str::<RunRecord>(seg.trim_end()) {
            Ok(record) if terminated => {
                records.push(record);
                valid_end += seg.len() as u64;
            }
            Ok(_) => {
                // Unterminated (necessarily last) segment: torn.
                torn_count = 1;
            }
            Err(_) => {
                if i == segments.len() - 1 {
                    torn_count = 1;
                } else {
                    return Err(ExperimentError::JournalLoad(
                        "unparseable line in middle of journal (likely corruption)".to_string(),
                    ));
                }
            }
        }
    }

    Ok((records, torn_count, valid_end))
}

/// Run an experiment with checkpoint/resume support.
///
/// If a journal exists at `journal_path`:
/// - Loads it, validating the spec hash.
/// - Collects the set of already-completed RunKeys.
/// - Executes only the remaining planned runs.
///
/// If no journal exists:
/// - Creates one with the header line.
///
/// Execution:
/// - **Sequential** (`parallel = false`): appends each completed run to the
///   journal with flush immediately (safe for crash recovery).
/// - **Parallel** (`parallel = true`): runs via rayon with `threads` threads;
///   each rayon task appends its record (with flush) through a shared
///   `Mutex<File>` AS IT COMPLETES, so parallel runs get the same incremental
///   crash durability as sequential ones. File order may differ from
///   enumeration order (resume is unaffected — the loader keys records by
///   [`RunKey`]), but the returned Vec is always in enumeration order.
///
/// If the journal ends in a torn line (crash mid-append), the file is
/// truncated back to the end of the last valid line before any new record is
/// appended, so the garbage tail never merges with fresh records.
///
/// Returns the FULL result set in ENUMERATION order (journal records + new
/// records merged by key).
///
/// The spec hash is computed HERE, internally, from `spec.to_toml()` (the
/// canonical serialization), never from whatever raw TOML text a caller may
/// have parsed `spec` out of. Since M2d this is the *only* place a spec hash
/// is produced for the checkpoint path: both the Python and R bindings parse
/// their raw user TOML into an `ExperimentSpec` and hand that (already-
/// canonicalized-on-read) spec here, so formatting-only differences in the
/// original TOML text (whitespace, comments, key order) can never affect the
/// hash and can never cause the two bindings to diverge from each other or
/// from a fresh run of the same spec.
///
/// `log_dir`: when `Some`, every run this call actually EXECUTES (i.e. not
/// resumed from the journal) is also logged in IOH-profiler format under
/// that directory, exactly as [`crate::experiment::run_experiment_logged`]
/// would log it (same grouping, same deterministic upfront-observer-creation
/// ordering). NOTE: a run resumed from the journal was executed in a PRIOR
/// process and is never re-run here, so it is never (re-)logged — the IOH
/// tree written by a given call only ever reflects runs that call actually
/// executed. Keep `log_dir` tied to a single, fresh, uninterrupted run of an
/// experiment if the IOH tree must reflect the experiment in full; `None`
/// disables IOH logging entirely (both bindings pass `None` until they wire
/// this through). A multi-budget `spec` is fully supported with `log_dir`
/// (M2d-3 lifted the earlier single-budget restriction): each budget's runs
/// land as distinct archive entries. What a later read does with those
/// entries depends on the reader — see `crate::ioh_read`'s module doc,
/// "Two canonicalization policies": [`crate::ioh_read::ioh_records`] prefers
/// the genuine run at each queried budget (falling back to a curtailed view
/// only for budgets the archive lacks), while
/// [`crate::ioh_read::canonical_anytime`] (used by `ecdf`/`ecdf_per_algo`/
/// `coco_export`) always canonicalizes to one run per `(instance, seed)` —
/// the largest-budget one.
pub fn run_experiment_with_checkpoint(
    spec: &ExperimentSpec,
    journal_path: &Path,
    parallel: bool,
    threads: Option<usize>,
    log_dir: Option<&Path>,
) -> Result<Vec<RunRecord>, ExperimentError> {
    use rayon::prelude::*;
    use sezgi_core::component::Registry;

    let hash_str = spec_hash(spec);

    // Load existing journal or create new one.
    let (journal_records, torn_count, valid_end) = load_journal(journal_path, &spec.name, &hash_str)?;
    let done_keys: HashSet<RunKey> = journal_records.iter().map(|r| r.key.clone()).collect();

    // A torn tail (crash mid-append) has no trailing newline; appending after
    // it would glue the next record onto the garbage. Truncate the file back
    // to the end of the last valid line before doing anything else.
    if torn_count > 0 {
        let file = OpenOptions::new()
            .write(true)
            .open(journal_path)
            .map_err(|e| ExperimentError::JournalWrite(format!("could not open journal to truncate torn tail: {}", e)))?;
        file.set_len(valid_end)
            .map_err(|e| ExperimentError::JournalWrite(format!("could not truncate torn tail: {}", e)))?;
    }

    // Enumerate all planned runs.
    let all_planned = enumerate(spec)?;

    // Partition: done (from journal) and todo (to execute).
    let mut planned_to_execute = Vec::new();
    for run in &all_planned {
        if !done_keys.contains(&run.key) {
            planned_to_execute.push(run.clone());
        }
    }

    // If nothing to execute, return journal records in enumeration order.
    if planned_to_execute.is_empty() {
        let mut result = Vec::new();
        for run in &all_planned {
            if let Some(record) = journal_records.iter().find(|r| r.key == run.key) {
                result.push(record.clone());
            }
        }
        return Ok(result);
    }

    // Ensure journal file and header exist.
    if !journal_path.exists() {
        let mut file = File::create(journal_path)
            .map_err(|e| ExperimentError::JournalWrite(format!("could not create journal: {}", e)))?;
        let header = JournalHeader {
            experiment: spec.name.clone(),
            spec_hash: hash_str.clone(),
        };
        let header_json = serde_json::to_string(&header)
            .map_err(|e| ExperimentError::JournalWrite(format!("could not serialize header: {}", e)))?;
        writeln!(file, "{}", header_json)
            .map_err(|e| ExperimentError::JournalWrite(format!("could not write header: {}", e)))?;
    }

    // Build the registry once for all runs.
    let mut reg = Registry::new();
    sezgi_components::register_builtins(&mut reg);

    // If IOH logging is requested, build one observer per run this call is
    // actually about to execute — upfront, in (the todo subsequence of)
    // enumeration order, before either execution arm runs. See
    // `build_ioh_observers`'s doc comment for why creating them upfront,
    // before any parallel dispatch, is what keeps the .dat/meta run order
    // deterministic between `parallel = true` and `parallel = false`.
    let (loggers, observers): (Vec<_>, Vec<Option<IohRunObserver>>) = match log_dir {
        Some(dir) => {
            let (loggers, observers) = build_ioh_observers(dir, &planned_to_execute)?;
            (loggers, observers.into_iter().map(Some).collect())
        }
        None => (Vec::new(), planned_to_execute.iter().map(|_| None).collect()),
    };

    // Execute runs.
    let new_records = if parallel {
        // Parallel: run via rayon; each task appends its record (with flush)
        // through the Mutex AS IT COMPLETES, giving incremental crash
        // durability. File order is nondeterministic, but the loader keys
        // records by RunKey, so resume is unaffected; the returned Vec is in
        // enumeration order (indexed par_iter().map().collect() preserves it).
        let file = Mutex::new(
            OpenOptions::new()
                .append(true)
                .open(journal_path)
                .map_err(|e| ExperimentError::JournalWrite(format!("could not open journal for append: {}", e)))?,
        );

        let reg = &reg;
        let pairs: Vec<(&PlannedRun, Option<IohRunObserver>)> =
            planned_to_execute.iter().zip(observers).collect();

        let run_one = |(planned_run, obs): (&PlannedRun, Option<IohRunObserver>)| -> Result<RunRecord, ExperimentError> {
            let observer: Option<Box<dyn EvalObserver>> = obs.map(|o| Box::new(o) as Box<dyn EvalObserver>);
            let record = execute_run(reg, planned_run, observer)?;

            // Serialize OUTSIDE the Mutex critical section to avoid holding
            // the lock during serialization.
            let record_json = serde_json::to_string(&record)
                .map_err(|e| ExperimentError::JournalWrite(format!("could not serialize record: {}", e)))?;
            {
                let mut file_guard = file.lock().map_err(|_| ExperimentError::JournalWrite(
                    "could not acquire file lock".to_string()))?;
                writeln!(file_guard, "{}", record_json)
                    .map_err(|e| ExperimentError::JournalWrite(format!("could not write record: {}", e)))?;
                file_guard.flush().map_err(|e| ExperimentError::JournalWrite(format!("could not flush journal: {}", e)))?;
            }
            Ok(record)
        };

        let slotted: Vec<Result<RunRecord, ExperimentError>> = match threads {
            Some(n) => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| ExperimentError::Engine(e.to_string()))?;
                pool.install(|| pairs.into_par_iter().map(run_one).collect())
            }
            None => pairs.into_par_iter().map(run_one).collect(),
        };

        let mut new_records = Vec::with_capacity(slotted.len());
        for r in slotted {
            new_records.push(r?);
        }
        new_records
    } else {
        // Sequential: execute and append inline with flush.
        let mut file = OpenOptions::new()
            .append(true)
            .open(journal_path)
            .map_err(|e| ExperimentError::JournalWrite(format!("could not open journal for append: {}", e)))?;

        let mut new_records = Vec::with_capacity(planned_to_execute.len());
        for (planned_run, obs) in planned_to_execute.iter().zip(observers) {
            let observer: Option<Box<dyn EvalObserver>> = obs.map(|o| Box::new(o) as Box<dyn EvalObserver>);
            let record = execute_run(&reg, planned_run, observer)?;

            let record_json = serde_json::to_string(&record)
                .map_err(|e| ExperimentError::JournalWrite(format!("could not serialize record: {}", e)))?;
            writeln!(file, "{}", record_json)
                .map_err(|e| ExperimentError::JournalWrite(format!("could not write record: {}", e)))?;
            file.flush().map_err(|e| ExperimentError::JournalWrite(format!("could not flush journal: {}", e)))?;

            new_records.push(record);
        }
        new_records
    };

    // Now that every run this call was going to execute has completed,
    // write each IOH logger's .dat/meta files (a no-op Vec when
    // `log_dir` was `None`).
    for logger in loggers {
        logger.finish().map_err(|e| ExperimentError::IohWrite(e.to_string()))?;
    }

    // Merge journal records and new records in enumeration order.
    let mut result = Vec::new();
    let journal_with_new: Vec<RunRecord> = {
        let mut merged = journal_records.clone();
        merged.extend(new_records);
        merged
    };

    for run in &all_planned {
        if let Some(record) = journal_with_new.iter().find(|r| r.key == run.key) {
            result.push(record.clone());
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experiment::SUITE_BBOB;
    use std::fs;
    use tempfile::TempDir;

    fn demo_spec() -> ExperimentSpec {
        ExperimentSpec {
            name: "test-checkpoint".into(),
            seeds: vec![1, 2],
            budgets: vec![500],
            algorithms: vec![crate::experiment::AlgoEntry {
                name: "de".into(),
                source: crate::experiment::AlgoSource::Preset {
                    kind: "de_rand_1".into(),
                    pop_size: Some(10),
                },
            }],
            problems: vec![crate::experiment::ProblemEntry {
                suite: "bbob".into(),
                fid: 1,
                dim: 5,
                instances: vec![1],
            }],
        }
    }

    #[test]
    fn fnv1a_hash_deterministic() {
        let data = b"hello world";
        let h1 = fnv1a_64(data);
        let h2 = fnv1a_64(data);
        assert_eq!(h1, h2, "FNV-1a must be deterministic");
    }

    #[test]
    fn fnv1a_different_inputs_different_hashes() {
        let h1 = fnv1a_64(b"hello");
        let h2 = fnv1a_64(b"world");
        assert_ne!(h1, h2, "different inputs must produce different hashes");
    }

    #[test]
    fn spec_hash_is_hex_string() {
        let spec = demo_spec();
        let hash = spec_hash(&spec);
        assert_eq!(hash.len(), 16, "spec hash must be 16 hex digits");
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()), "spec hash must be hex");
    }

    #[test]
    fn load_journal_nonexistent_returns_empty() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nonexistent.jsonl");
        let (records, torn, _) = load_journal(&path, "test", "0000000000000000").unwrap();
        assert_eq!(records.len(), 0);
        assert_eq!(torn, 0);
    }

    #[test]
    fn load_journal_empty_file_returns_empty() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.jsonl");
        fs::write(&path, "").unwrap();
        let (records, torn, _) = load_journal(&path, "test", "0000000000000000").unwrap();
        assert_eq!(records.len(), 0);
        assert_eq!(torn, 0);
    }

    #[test]
    fn load_journal_hash_mismatch_errors() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("journal.jsonl");
        let header = JournalHeader {
            experiment: "test".to_string(),
            spec_hash: "0000000000000000".to_string(),
        };
        let header_json = serde_json::to_string(&header).unwrap();
        fs::write(&path, header_json).unwrap();

        let result = load_journal(&path, "test", "1111111111111111");
        assert!(matches!(result, Err(ExperimentError::ExperimentHashMismatch { .. })));
    }

    #[test]
    fn load_journal_name_mismatch_errors() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("journal.jsonl");
        let header = JournalHeader {
            experiment: "old-name".to_string(),
            spec_hash: "0000000000000000".to_string(),
        };
        let header_json = serde_json::to_string(&header).unwrap();
        fs::write(&path, header_json).unwrap();

        let result = load_journal(&path, "new-name", "0000000000000000");
        assert!(matches!(result, Err(ExperimentError::JournalLoad(_))));
    }

    #[test]
    fn load_journal_with_records() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("journal.jsonl");

        let header = JournalHeader {
            experiment: "test".to_string(),
            spec_hash: "aaaaaaaaaaaaaaaa".to_string(),
        };
        let header_json = serde_json::to_string(&header).unwrap();

        let record1 = RunRecord {
            key: RunKey {
                algo: "de".into(),
                fid: 1,
                dim: 5,
                instance: 1,
                seed: 123,
                budget: 500,
                suite: SUITE_BBOB.into(),
            },
            best_f: 1.5,
            f_opt: 0.0,
            evals_used: 450,
            wall_secs: 1.23,
        };
        let record1_json = serde_json::to_string(&record1).unwrap();

        let content = format!("{}\n{}\n", header_json, record1_json);
        fs::write(&path, content).unwrap();

        let (records, torn, _) = load_journal(&path, "test", "aaaaaaaaaaaaaaaa").unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].best_f, 1.5);
        assert_eq!(torn, 0);
    }

    #[test]
    fn load_journal_tolerates_torn_line() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("journal.jsonl");

        let header = JournalHeader {
            experiment: "test".to_string(),
            spec_hash: "bbbbbbbbbbbbbbbb".to_string(),
        };
        let header_json = serde_json::to_string(&header).unwrap();

        let record1 = RunRecord {
            key: RunKey {
                algo: "de".into(),
                fid: 1,
                dim: 5,
                instance: 1,
                seed: 123,
                budget: 500,
                suite: SUITE_BBOB.into(),
            },
            best_f: 1.5,
            f_opt: 0.0,
            evals_used: 450,
            wall_secs: 1.23,
        };
        let record1_json = serde_json::to_string(&record1).unwrap();

        // Torn line: incomplete JSON
        let content = format!("{}\n{}\n{{\"incomplete", header_json, record1_json);
        fs::write(&path, content).unwrap();

        let (records, torn, _) = load_journal(&path, "test", "bbbbbbbbbbbbbbbb").unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(torn, 1);
    }

    #[test]
    fn load_journal_mid_file_corruption_errors() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("journal.jsonl");

        let header = JournalHeader {
            experiment: "test".to_string(),
            spec_hash: "cccccccccccccccc".to_string(),
        };
        let header_json = serde_json::to_string(&header).unwrap();

        let record1 = RunRecord {
            key: RunKey {
                algo: "de".into(),
                fid: 1,
                dim: 5,
                instance: 1,
                seed: 123,
                budget: 500,
                suite: SUITE_BBOB.into(),
            },
            best_f: 1.5,
            f_opt: 0.0,
            evals_used: 450,
            wall_secs: 1.23,
        };
        let record1_json = serde_json::to_string(&record1).unwrap();

        let record2 = RunRecord {
            key: RunKey {
                algo: "de".into(),
                fid: 1,
                dim: 5,
                instance: 1,
                seed: 456,
                budget: 500,
                suite: SUITE_BBOB.into(),
            },
            best_f: 2.0,
            f_opt: 0.0,
            evals_used: 400,
            wall_secs: 0.99,
        };
        let record2_json = serde_json::to_string(&record2).unwrap();

        // Valid record, bad record, valid record
        let content = format!("{}\n{}\n{{\"bad\n{}\n", header_json, record1_json, record2_json);
        fs::write(&path, content).unwrap();

        let result = load_journal(&path, "test", "cccccccccccccccc");
        assert!(matches!(result, Err(ExperimentError::JournalLoad(_))));
    }

    #[test]
    fn fresh_run_creates_journal() {
        let dir = TempDir::new().unwrap();
        let journal_path = dir.path().join("journal.jsonl");

        let spec = demo_spec();

        let result = run_experiment_with_checkpoint(&spec, &journal_path, false, None, None);
        assert!(result.is_ok(), "fresh run must succeed");

        assert!(journal_path.exists(), "journal must be created");
        let (records, _, _) = load_journal(&journal_path, &spec.name, &format_hash(&spec)).unwrap();
        assert!(!records.is_empty(), "journal must contain records after fresh run");
    }

    #[test]
    fn resume_skips_done_runs() {
        let dir = TempDir::new().unwrap();
        let journal_path = dir.path().join("journal.jsonl");

        let spec = demo_spec();

        // First run: complete the full experiment
        let result1 = run_experiment_with_checkpoint(&spec, &journal_path, false, None, None)
            .expect("first run should succeed");
        let count1 = result1.len();
        assert_eq!(count1, 2, "1 algo x 1 problem x 1 instance x 2 seeds x 1 budget = 2 runs");

        // Count journal lines after first run (header + N records)
        let (records_after_first, _, _) = load_journal(&journal_path, &spec.name, &format_hash(&spec))
            .expect("should load journal");
        let lines_after_first = 1 + records_after_first.len(); // header + records

        // Second run: resume with same journal (should execute nothing new)
        let result2 = run_experiment_with_checkpoint(&spec, &journal_path, false, None, None)
            .expect("resume should succeed");
        let count2 = result2.len();

        // Count journal lines after second run (should not increase if nothing executed)
        let (records_after_second, _, _) = load_journal(&journal_path, &spec.name, &format_hash(&spec))
            .expect("should load journal");
        let lines_after_second = 1 + records_after_second.len(); // header + records

        assert_eq!(count1, count2, "resume should return all records (no new executions)");
        assert_eq!(lines_after_first, lines_after_second, "journal should not grow on resume of completed experiment");

        // Content equality (excluding wall_secs which legitimately differs).
        // Bit-exact: the engine is deterministic and serde_json f64 round-trips
        // are shortest-repr correct, so fresh and journal-loaded records must
        // be identical to the bit, index by index in enumeration order.
        assert_eq!(result1.len(), result2.len(), "record counts must match");
        for (i, (r1, r2)) in result1.iter().zip(result2.iter()).enumerate() {
            let t1 = (r1.key.to_string(), r1.best_f.to_bits(), r1.f_opt.to_bits(), r1.evals_used);
            let t2 = (r2.key.to_string(), r2.best_f.to_bits(), r2.f_opt.to_bits(), r2.evals_used);
            assert_eq!(
                t1, t2,
                "record {}: fresh vs resumed record must be bit-identical \
                 (best_f: {} bits={:016x} vs {} bits={:016x})",
                i, r1.best_f, r1.best_f.to_bits(), r2.best_f, r2.best_f.to_bits()
            );
        }
    }

    #[test]
    fn resume_from_half() {
        let dir = TempDir::new().unwrap();
        let journal_path = dir.path().join("journal.jsonl");

        let spec = demo_spec();

        // First full run
        let full_result = run_experiment_with_checkpoint(&spec, &journal_path, false, None, None)
            .expect("full run should succeed");
        let full_count = full_result.len();

        // Create a fresh journal with only the first half of records
        let (all_records, _, _) = load_journal(&journal_path, &spec.name, &format_hash(&spec))
            .expect("should load journal");
        let half_count = all_records.len().div_ceil(2);
        let missing_count = all_records.len() - half_count;

        let half_records = &all_records[..half_count];

        // Write journal with half records
        fs::remove_file(&journal_path).unwrap();
        let mut file = File::create(&journal_path).unwrap();
        let header = JournalHeader {
            experiment: spec.name.clone(),
            spec_hash: format_hash(&spec),
        };
        let header_json = serde_json::to_string(&header).unwrap();
        writeln!(file, "{}", header_json).unwrap();
        for record in half_records {
            let record_json = serde_json::to_string(record).unwrap();
            writeln!(file, "{}", record_json).unwrap();
        }
        drop(file);

        // Count journal lines before resume (header + half_count records)
        let lines_before_resume = 1 + half_count;

        // Resume from half
        let resume_result = run_experiment_with_checkpoint(&spec, &journal_path, false, None, None)
            .expect("resume should succeed");

        // Count journal lines after resume (should be header + all records)
        let (records_after_resume, _, _) = load_journal(&journal_path, &spec.name, &format_hash(&spec))
            .expect("should load journal");
        let lines_after_resume = 1 + records_after_resume.len();

        // Verify journal grew by exactly the number of missing runs
        assert_eq!(
            lines_after_resume - lines_before_resume,
            missing_count,
            "journal should grow by exactly {} missing runs", missing_count
        );

        // Verify total record count
        assert_eq!(resume_result.len(), full_count, "resume should produce same total record count");

        // Content equality: enumeration-order comparison proves journal merge order is correct.
        // Bit-exact: the engine is deterministic and serde_json f64 round-trips
        // are shortest-repr correct, so the resumed result (journal half + freshly
        // executed half) must equal the uninterrupted baseline to the bit.
        assert_eq!(full_result.len(), resume_result.len(), "record counts must match");
        for (i, (full_r, resume_r)) in full_result.iter().zip(resume_result.iter()).enumerate() {
            let tf = (full_r.key.to_string(), full_r.best_f.to_bits(), full_r.f_opt.to_bits(), full_r.evals_used);
            let tr = (resume_r.key.to_string(), resume_r.best_f.to_bits(), resume_r.f_opt.to_bits(), resume_r.evals_used);
            assert_eq!(
                tf, tr,
                "record {}: baseline vs resumed record must be bit-identical \
                 (best_f: {} bits={:016x} vs {} bits={:016x})",
                i, full_r.best_f, full_r.best_f.to_bits(), resume_r.best_f, resume_r.best_f.to_bits()
            );
        }
    }

    #[test]
    fn spec_hash_mismatch_errors() {
        let dir = TempDir::new().unwrap();
        let journal_path = dir.path().join("journal.jsonl");

        let spec = demo_spec();

        // Create journal with first spec
        run_experiment_with_checkpoint(&spec, &journal_path, false, None, None).unwrap();

        // Modify spec (change a seed)
        let mut modified_spec = spec.clone();
        modified_spec.seeds[0] = 999;

        // Try to resume with modified spec
        let result = run_experiment_with_checkpoint(&modified_spec, &journal_path, false, None, None);
        assert!(matches!(result, Err(ExperimentError::ExperimentHashMismatch { .. })));
    }

    fn format_hash(spec: &ExperimentSpec) -> String {
        spec_hash(spec)
    }

    #[test]
    fn resume_after_torn_line_truncates_and_second_resume_is_stable() {
        let dir = TempDir::new().unwrap();
        let journal_path = dir.path().join("journal.jsonl");

        let spec = demo_spec();

        // Full baseline run.
        let baseline = run_experiment_with_checkpoint(&spec, &journal_path, false, None, None)
            .expect("baseline run should succeed");
        assert_eq!(baseline.len(), 2);

        // Simulate a torn write: chop the journal mid-way through its last
        // record, leaving a partial line with NO trailing newline.
        let bytes = fs::read(&journal_path).unwrap();
        let torn_len = bytes.len() - 10;
        let file = OpenOptions::new().write(true).open(&journal_path).unwrap();
        file.set_len(torn_len as u64).unwrap();
        drop(file);
        let (_, torn, _) = load_journal(&journal_path, &spec.name, &format_hash(&spec)).unwrap();
        assert_eq!(torn, 1, "chopped journal must present a torn line");

        // Resume 1: must truncate the torn tail, re-execute the lost run, and
        // leave a fully parseable journal.
        let resume1 = run_experiment_with_checkpoint(&spec, &journal_path, false, None, None)
            .expect("first resume after torn line should succeed");
        assert_eq!(resume1.len(), 2);
        let (records1, torn1, _) = load_journal(&journal_path, &spec.name, &format_hash(&spec))
            .expect("journal must be fully parseable after first resume");
        assert_eq!(torn1, 0, "no torn line may remain after first resume");
        assert_eq!(records1.len(), 2, "journal must hold all records after first resume");
        let len_after_resume1 = fs::metadata(&journal_path).unwrap().len();

        // Resume 2: nothing to execute, journal must not grow, and must stay
        // fully parseable. (A glued torn tail only detonates on the SECOND
        // resume — this is the regression the fix is pinned against.)
        let resume2 = run_experiment_with_checkpoint(&spec, &journal_path, false, None, None)
            .expect("second resume should succeed");
        assert_eq!(resume2.len(), 2);
        let (records2, torn2, _) = load_journal(&journal_path, &spec.name, &format_hash(&spec))
            .expect("journal must be fully parseable after second resume");
        assert_eq!(torn2, 0, "no torn line may appear on second resume");
        assert_eq!(records2.len(), 2);
        let len_after_resume2 = fs::metadata(&journal_path).unwrap().len();
        assert_eq!(
            len_after_resume1, len_after_resume2,
            "journal must not grow on a second resume of a completed experiment"
        );

        // Records stay bit-identical to the uninterrupted baseline.
        for (i, (b, r)) in baseline.iter().zip(resume2.iter()).enumerate() {
            let tb = (b.key.to_string(), b.best_f.to_bits(), b.f_opt.to_bits(), b.evals_used);
            let tr = (r.key.to_string(), r.best_f.to_bits(), r.f_opt.to_bits(), r.evals_used);
            assert_eq!(tb, tr, "record {}: baseline vs post-torn-resume must be bit-identical", i);
        }
    }

    #[test]
    fn parallel_journal_written_incrementally_and_resumable() {
        let dir = TempDir::new().unwrap();
        let journal_path = dir.path().join("parallel.jsonl");
        let seq_path = dir.path().join("sequential.jsonl");

        let spec = demo_spec();

        // Fresh parallel run: every record must land in the journal.
        let parallel_result =
            run_experiment_with_checkpoint(&spec, &journal_path, true, Some(2), None)
                .expect("parallel run should succeed");
        assert_eq!(parallel_result.len(), 2);
        let (records, torn, _) = load_journal(&journal_path, &spec.name, &format_hash(&spec))
            .expect("parallel journal must be parseable");
        assert_eq!(torn, 0);
        assert_eq!(records.len(), 2, "parallel run must append every record to the journal");

        // Resume from the parallel-written journal (file order may differ from
        // enumeration order; the loader keys by RunKey): nothing re-executes,
        // no growth, results identical.
        let len_before = fs::metadata(&journal_path).unwrap().len();
        let resumed =
            run_experiment_with_checkpoint(&spec, &journal_path, true, Some(2), None)
                .expect("resume from parallel journal should succeed");
        let len_after = fs::metadata(&journal_path).unwrap().len();
        assert_eq!(len_before, len_after, "resume of complete parallel journal must not grow it");
        assert_eq!(resumed.len(), 2);

        // Parallel checkpoint results must be bit-identical to sequential
        // checkpoint results, in enumeration order.
        let sequential_result =
            run_experiment_with_checkpoint(&spec, &seq_path, false, None, None)
                .expect("sequential run should succeed");
        for (i, (s, p)) in sequential_result.iter().zip(parallel_result.iter()).enumerate() {
            let ts = (s.key.to_string(), s.best_f.to_bits(), s.f_opt.to_bits(), s.evals_used);
            let tp = (p.key.to_string(), p.best_f.to_bits(), p.f_opt.to_bits(), p.evals_used);
            assert_eq!(ts, tp, "record {}: sequential vs parallel checkpoint must be bit-identical", i);
        }
        for (i, (p, r)) in parallel_result.iter().zip(resumed.iter()).enumerate() {
            let tp = (p.key.to_string(), p.best_f.to_bits(), p.f_opt.to_bits(), p.evals_used);
            let tr = (r.key.to_string(), r.best_f.to_bits(), r.f_opt.to_bits(), r.evals_used);
            assert_eq!(tp, tr, "record {}: fresh vs resumed parallel run must be bit-identical", i);
        }
    }

    #[test]
    fn json_f64_roundtrip_is_bitexact() {
        // Verify that serde_json preserves f64 bit patterns exactly
        let test_values = vec![
            1.5f64,
            0.123456789012345,
            1e-100,
            1e100,
            f64::MIN_POSITIVE,
            f64::MAX,
            // Regression: serde_json's DEFAULT float parser (without the
            // `float_roundtrip` feature) parses this shortest repr of
            // 0xc05f57581a40c97a one ULP off, to 0xc05f57581a40c97b.
            // The bench crate enables `float_roundtrip` precisely for this.
            f64::from_bits(0xc05f57581a40c97a),
        ];

        for orig in test_values {
            let json = serde_json::json!(orig);
            let loaded: f64 = serde_json::from_value(json).unwrap();
            assert_eq!(
                orig.to_bits(),
                loaded.to_bits(),
                "JSON round-trip failed for {}: orig_bits={:x}, loaded_bits={:x}",
                orig,
                orig.to_bits(),
                loaded.to_bits()
            );
        }
    }

    #[test]
    fn journal_roundtrip_runrecord_bitexact() {
        // Verify that writing and reading a RunRecord to/from JSON preserves f64 bits
        let dir = TempDir::new().unwrap();
        let _path = dir.path().join("test_record.jsonl");

        let record = RunRecord {
            key: RunKey {
                algo: "test".into(),
                fid: 1,
                dim: 5,
                instance: 1,
                seed: 42,
                budget: 1000,
                suite: SUITE_BBOB.into(),
            },
            best_f: 1.234_567_890_123_456_7,
            f_opt: 0.0,
            evals_used: 999,
            wall_secs: 1.5,
        };

        // Write to JSON string (as done in checkpoint)
        let json_str = serde_json::to_string(&record).unwrap();

        // Read back (as done in load_journal)
        let loaded: RunRecord = serde_json::from_str(&json_str).unwrap();

        assert_eq!(
            record.best_f.to_bits(),
            loaded.best_f.to_bits(),
            "RunRecord JSON round-trip lost precision: orig={} bits={:x}, loaded={} bits={:x}",
            record.best_f, record.best_f.to_bits(),
            loaded.best_f, loaded.best_f.to_bits()
        );
        assert_eq!(
            record.f_opt.to_bits(),
            loaded.f_opt.to_bits(),
            "RunRecord f_opt JSON round-trip lost precision"
        );
    }

    /// M2d-3: the multi-budget restriction was LIFTED — `log_dir` with a
    /// multi-budget spec now works end to end through the checkpoint path
    /// too, and a disk read back canonicalizes to unique, non-conflicting
    /// keys the same way `run_experiment_logged`'s does (see
    /// `crates/bench/src/experiment.rs`'s repurposed
    /// `run_experiment_logged_allows_multiple_budgets` test).
    #[test]
    fn checkpoint_allows_multiple_budgets_with_log_dir() {
        let dir = TempDir::new().unwrap();
        let journal_path = dir.path().join("journal.jsonl");
        let log_dir = dir.path().join("logs");

        let mut spec = demo_spec();
        spec.budgets = vec![300, 600];

        let records = run_experiment_with_checkpoint(
            &spec, &journal_path, false, None, Some(&log_dir),
        ).unwrap();
        // demo_spec: 1 algo x 1 problem x 1 instance x 2 seeds x 2 budgets
        assert_eq!(records.len(), 4);
        assert!(journal_path.exists());

        let scenarios = crate::ioh_read::read_ioh_root(&log_dir).unwrap();
        let disk_records = crate::ioh_read::ioh_records(&scenarios, &[300, 600]).unwrap();
        assert_eq!(disk_records.len(), 4);
        let mut seen: std::collections::HashSet<RunKey> = std::collections::HashSet::new();
        for r in &disk_records {
            assert!(seen.insert(r.key.clone()), "duplicate key {} in disk records", r.key);
        }
    }

    #[test]
    fn checkpoint_allows_multiple_budgets_without_log_dir() {
        // The restriction is specific to log_dir: a multi-budget spec must
        // still run fine through the checkpoint path when not logging.
        let dir = TempDir::new().unwrap();
        let journal_path = dir.path().join("journal.jsonl");

        let mut spec = demo_spec();
        spec.budgets = vec![300, 600];

        let result = run_experiment_with_checkpoint(&spec, &journal_path, false, None, None);
        assert!(result.is_ok(), "multi-budget spec without log_dir must still succeed");
    }

    #[test]
    fn journal_file_roundtrip_bitexact() {
        // Verify that writing RunRecords to a journal file and reading them back preserves f64 bits
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test_journal.jsonl");

        let record = RunRecord {
            key: RunKey {
                algo: "test".into(),
                fid: 1,
                dim: 5,
                instance: 1,
                seed: 42,
                budget: 1000,
                suite: SUITE_BBOB.into(),
            },
            best_f: 1.234_567_890_123_456_7,
            f_opt: 0.0,
            evals_used: 999,
            wall_secs: 1.5,
        };

        // Write journal with header and one record
        let mut file = File::create(&path).unwrap();
        let header = JournalHeader {
            experiment: "test".to_string(),
            spec_hash: "aaaaaaaaaaaaaaaa".to_string(),
        };
        let header_json = serde_json::to_string(&header).unwrap();
        writeln!(file, "{}", header_json).unwrap();
        let record_json = serde_json::to_string(&record).unwrap();
        writeln!(file, "{}", record_json).unwrap();
        drop(file);

        // Read back
        let (loaded_records, _, _) = load_journal(&path, "test", "aaaaaaaaaaaaaaaa").unwrap();
        assert_eq!(loaded_records.len(), 1, "should have loaded one record");
        let loaded = &loaded_records[0];

        assert_eq!(
            record.best_f.to_bits(),
            loaded.best_f.to_bits(),
            "Journal file round-trip lost precision: orig={} bits={:x}, loaded={} bits={:x}",
            record.best_f, record.best_f.to_bits(),
            loaded.best_f, loaded.best_f.to_bits()
        );
    }

}
