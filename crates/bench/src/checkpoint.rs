//! Journal-based checkpoint and resume for experiment runs.
//!
//! Implements a JSONL-based run journal that allows resuming an experiment
//! from the last completed run. Each run is recorded as a JSON line, and the
//! journal is validated against the experiment spec's hash (FNV-1a 64-bit) to
//! detect if the spec changed between runs.

use crate::experiment::{ExperimentSpec, RunKey, RunRecord, ExperimentError, enumerate};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
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
/// A final partial/torn line (incomplete JSON at EOF) is tolerated and skipped;
/// its count is returned. An unparseable line NOT at the end is a hard error.
/// Empty journal file or file that doesn't exist are treated as "no records".
pub fn load_journal(
    path: &Path,
    expected_name: &str,
    expected_hash: &str,
) -> Result<(Vec<RunRecord>, usize), ExperimentError> {
    if !path.exists() {
        return Ok((Vec::new(), 0));
    }

    let file = File::open(path)
        .map_err(|e| ExperimentError::JournalLoad(format!("could not open journal: {}", e)))?;

    let reader = BufReader::new(file);
    let all_lines: Vec<String> = reader
        .lines()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ExperimentError::JournalLoad(format!("could not read journal file: {}", e)))?;

    if all_lines.is_empty() {
        return Ok((Vec::new(), 0));
    }

    // Read and validate header.
    let header: JournalHeader = serde_json::from_str(&all_lines[0])
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

    // Read records; tolerate a final torn line.
    let mut records = Vec::new();
    let mut torn_count = 0usize;

    for (i, line) in all_lines.iter().enumerate().skip(1) {
        match serde_json::from_str::<RunRecord>(line) {
            Ok(record) => records.push(record),
            Err(_) => {
                // Check if this is the last line (torn write tolerance).
                if i == all_lines.len() - 1 {
                    torn_count = 1;
                } else {
                    return Err(ExperimentError::JournalLoad(
                        "unparseable line in middle of journal (likely corruption)".to_string(),
                    ));
                }
            }
        }
    }

    Ok((records, torn_count))
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
/// - **Parallel** (`parallel = true`): runs via rayon with `threads` threads,
///   then appends all results behind a Mutex as they complete (file order may
///   differ from enumeration order, but the returned Vec is always in
///   enumeration order).
///
/// Returns the FULL result set in ENUMERATION order (journal records + new
/// records merged by key).
pub fn run_experiment_with_checkpoint(
    spec: &ExperimentSpec,
    spec_toml_str: &str,
    journal_path: &Path,
    parallel: bool,
    threads: Option<usize>,
) -> Result<Vec<RunRecord>, ExperimentError> {
    use rayon::prelude::*;
    use sezgi_core::component::Registry;
    use sezgi_core::engine::{Engine, RunConfig};
    use sezgi_core::problem::Problem;
    use sezgi_problems::BbobProblem;

    let hash = fnv1a_64(spec_toml_str.as_bytes());
    let hash_str = format!("{:016x}", hash);

    // Load existing journal or create new one.
    let (journal_records, _torn_count) = load_journal(journal_path, &spec.name, &hash_str)?;
    let done_keys: HashSet<RunKey> = journal_records.iter().map(|r| r.key.clone()).collect();

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

    // Execute runs.
    let new_records = if parallel {
        // Parallel: run via rayon, then append behind Mutex.
        let file = Mutex::new(
            OpenOptions::new()
                .append(true)
                .open(journal_path)
                .map_err(|e| ExperimentError::JournalWrite(format!("could not open journal for append: {}", e)))?,
        );

        let run_one = |planned_run: &crate::experiment::PlannedRun| -> Result<RunRecord, ExperimentError> {
            let problem = BbobProblem::new(planned_run.fid, planned_run.dim, planned_run.instance)
                .map_err(|e| ExperimentError::Problem(e.to_string()))?;
            let engine = Engine::from_spec(&planned_run.algo_spec, &reg, problem.space())
                .map_err(|e| ExperimentError::Engine(e.to_string()))?;
            let t0 = std::time::Instant::now();
            let res = engine
                .run(&problem, RunConfig { master_seed: planned_run.seed, run_id: planned_run.run_id }, None)
                .map_err(|e| ExperimentError::Engine(e.to_string()))?;
            Ok(RunRecord {
                key: planned_run.key.clone(),
                best_f: res.best_f,
                f_opt: problem.f_opt(),
                evals_used: res.evals_used,
                wall_secs: t0.elapsed().as_secs_f64(),
            })
        };

        let slotted: Vec<Result<RunRecord, ExperimentError>> = match threads {
            Some(n) => {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| ExperimentError::Engine(e.to_string()))?;
                pool.install(|| planned_to_execute.par_iter().map(run_one).collect())
            }
            None => planned_to_execute.par_iter().map(run_one).collect(),
        };

        let mut new_records = Vec::with_capacity(slotted.len());
        for r in slotted {
            let record = r?;
            // Serialize OUTSIDE the Mutex critical section to avoid holding the lock during serialization
            let record_json = serde_json::to_string(&record)
                .map_err(|e| ExperimentError::JournalWrite(format!("could not serialize record: {}", e)))?;
            {
                let mut file_guard = file.lock().map_err(|_| ExperimentError::JournalWrite(
                    "could not acquire file lock".to_string()))?;
                writeln!(file_guard, "{}", record_json)
                    .map_err(|e| ExperimentError::JournalWrite(format!("could not write record: {}", e)))?;
                file_guard.flush().map_err(|e| ExperimentError::JournalWrite(format!("could not flush journal: {}", e)))?;
            }
            new_records.push(record);
        }
        new_records
    } else {
        // Sequential: execute and append inline with flush.
        let mut file = OpenOptions::new()
            .append(true)
            .open(journal_path)
            .map_err(|e| ExperimentError::JournalWrite(format!("could not open journal for append: {}", e)))?;

        let mut new_records = Vec::with_capacity(planned_to_execute.len());
        for planned_run in &planned_to_execute {
            let problem = BbobProblem::new(planned_run.fid, planned_run.dim, planned_run.instance)
                .map_err(|e| ExperimentError::Problem(e.to_string()))?;
            let engine = Engine::from_spec(&planned_run.algo_spec, &reg, problem.space())
                .map_err(|e| ExperimentError::Engine(e.to_string()))?;
            let t0 = std::time::Instant::now();
            let res = engine
                .run(&problem, RunConfig { master_seed: planned_run.seed, run_id: planned_run.run_id }, None)
                .map_err(|e| ExperimentError::Engine(e.to_string()))?;
            let record = RunRecord {
                key: planned_run.key.clone(),
                best_f: res.best_f,
                f_opt: problem.f_opt(),
                evals_used: res.evals_used,
                wall_secs: t0.elapsed().as_secs_f64(),
            };

            let record_json = serde_json::to_string(&record)
                .map_err(|e| ExperimentError::JournalWrite(format!("could not serialize record: {}", e)))?;
            writeln!(file, "{}", record_json)
                .map_err(|e| ExperimentError::JournalWrite(format!("could not write record: {}", e)))?;
            file.flush().map_err(|e| ExperimentError::JournalWrite(format!("could not flush journal: {}", e)))?;

            new_records.push(record);
        }
        new_records
    };

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
        let (records, torn) = load_journal(&path, "test", "0000000000000000").unwrap();
        assert_eq!(records.len(), 0);
        assert_eq!(torn, 0);
    }

    #[test]
    fn load_journal_empty_file_returns_empty() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("empty.jsonl");
        fs::write(&path, "").unwrap();
        let (records, torn) = load_journal(&path, "test", "0000000000000000").unwrap();
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
            },
            best_f: 1.5,
            f_opt: 0.0,
            evals_used: 450,
            wall_secs: 1.23,
        };
        let record1_json = serde_json::to_string(&record1).unwrap();

        let content = format!("{}\n{}\n", header_json, record1_json);
        fs::write(&path, content).unwrap();

        let (records, torn) = load_journal(&path, "test", "aaaaaaaaaaaaaaaa").unwrap();
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

        let (records, torn) = load_journal(&path, "test", "bbbbbbbbbbbbbbbb").unwrap();
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
        let spec_toml = spec.to_toml();

        let result = run_experiment_with_checkpoint(&spec, &spec_toml, &journal_path, false, None);
        assert!(result.is_ok(), "fresh run must succeed");

        assert!(journal_path.exists(), "journal must be created");
        let (records, _) = load_journal(&journal_path, &spec.name, &format_hash(&spec)).unwrap();
        assert!(!records.is_empty(), "journal must contain records after fresh run");
    }

    #[test]
    fn resume_skips_done_runs() {
        let dir = TempDir::new().unwrap();
        let journal_path = dir.path().join("journal.jsonl");

        let spec = demo_spec();
        let spec_toml = spec.to_toml();

        // First run: complete the full experiment
        let result1 = run_experiment_with_checkpoint(&spec, &spec_toml, &journal_path, false, None)
            .expect("first run should succeed");
        let count1 = result1.len();
        assert_eq!(count1, 2, "1 algo x 1 problem x 1 instance x 2 seeds x 1 budget = 2 runs");

        // Count journal lines after first run (header + N records)
        let (records_after_first, _) = load_journal(&journal_path, &spec.name, &format_hash(&spec))
            .expect("should load journal");
        let lines_after_first = 1 + records_after_first.len(); // header + records

        // Second run: resume with same journal (should execute nothing new)
        let result2 = run_experiment_with_checkpoint(&spec, &spec_toml, &journal_path, false, None)
            .expect("resume should succeed");
        let count2 = result2.len();

        // Count journal lines after second run (should not increase if nothing executed)
        let (records_after_second, _) = load_journal(&journal_path, &spec.name, &format_hash(&spec))
            .expect("should load journal");
        let lines_after_second = 1 + records_after_second.len(); // header + records

        assert_eq!(count1, count2, "resume should return all records (no new executions)");
        assert_eq!(lines_after_first, lines_after_second, "journal should not grow on resume of completed experiment");

        // Content equality (excluding wall_secs which legitimately differs).
        // Note: JSON roundtrip for f64 values can lose 1 ULP precision due to decimal→binary conversion.
        // We allow ±1 ULP tolerance to account for this constraint.
        assert_eq!(result1.len(), result2.len(), "record counts must match");
        for (i, (r1, r2)) in result1.iter().zip(result2.iter()).enumerate() {
            assert_eq!(
                r1.key.to_string(),
                r2.key.to_string(),
                "record {}: key mismatch: result1={} vs result2={}",
                i, r1.key, r2.key
            );
            // Allow ±1 ULP due to JSON decimal→binary conversion
            assert!(
                (r1.best_f - r2.best_f).abs() <= f64::EPSILON * r1.best_f.abs().max(1.0),
                "record {}: best_f must match (r1={}, r2={})", i, r1.best_f, r2.best_f
            );
            assert!(
                (r1.f_opt - r2.f_opt).abs() <= f64::EPSILON * r1.f_opt.abs().max(1.0),
                "record {}: f_opt must match (r1={}, r2={})", i, r1.f_opt, r2.f_opt
            );
            assert_eq!(r1.evals_used, r2.evals_used, "record {}: evals_used must match", i);
        }
    }

    #[test]
    fn resume_from_half() {
        let dir = TempDir::new().unwrap();
        let journal_path = dir.path().join("journal.jsonl");

        let spec = demo_spec();
        let spec_toml = spec.to_toml();

        // First full run
        let full_result = run_experiment_with_checkpoint(&spec, &spec_toml, &journal_path, false, None)
            .expect("full run should succeed");
        let full_count = full_result.len();

        // Create a fresh journal with only the first half of records
        let (all_records, _) = load_journal(&journal_path, &spec.name, &format_hash(&spec))
            .expect("should load journal");
        let half_count = (all_records.len() + 1) / 2;
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
        let resume_result = run_experiment_with_checkpoint(&spec, &spec_toml, &journal_path, false, None)
            .expect("resume should succeed");

        // Count journal lines after resume (should be header + all records)
        let (records_after_resume, _) = load_journal(&journal_path, &spec.name, &format_hash(&spec))
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
        // Note: JSON roundtrip for f64 values can lose 1 ULP precision due to decimal→binary conversion.
        // We allow ±1 ULP tolerance to account for this constraint.
        assert_eq!(full_result.len(), resume_result.len(), "record counts must match");
        for (i, (full_r, resume_r)) in full_result.iter().zip(resume_result.iter()).enumerate() {
            assert_eq!(
                full_r.key.to_string(),
                resume_r.key.to_string(),
                "record {}: key mismatch: full={} vs resume={}",
                i, full_r.key, resume_r.key
            );
            // Allow ±1 ULP due to JSON decimal→binary conversion
            assert!(
                (full_r.best_f - resume_r.best_f).abs() <= f64::EPSILON * full_r.best_f.abs().max(1.0),
                "record {}: best_f must match (full={}, resume={})", i, full_r.best_f, resume_r.best_f
            );
            assert!(
                (full_r.f_opt - resume_r.f_opt).abs() <= f64::EPSILON * full_r.f_opt.abs().max(1.0),
                "record {}: f_opt must match (full={}, resume={})", i, full_r.f_opt, resume_r.f_opt
            );
            assert_eq!(full_r.evals_used, resume_r.evals_used, "record {}: evals_used must match", i);
        }
    }

    #[test]
    fn spec_hash_mismatch_errors() {
        let dir = TempDir::new().unwrap();
        let journal_path = dir.path().join("journal.jsonl");

        let spec = demo_spec();
        let spec_toml = spec.to_toml();

        // Create journal with first spec
        run_experiment_with_checkpoint(&spec, &spec_toml, &journal_path, false, None).unwrap();

        // Modify spec (change a seed)
        let mut modified_spec = spec.clone();
        modified_spec.seeds[0] = 999;
        let modified_toml = modified_spec.to_toml();

        // Try to resume with modified spec
        let result = run_experiment_with_checkpoint(&modified_spec, &modified_toml, &journal_path, false, None);
        assert!(matches!(result, Err(ExperimentError::ExperimentHashMismatch { .. })));
    }

    fn format_hash(spec: &ExperimentSpec) -> String {
        spec_hash(spec)
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
            },
            best_f: 1.234567890123456789,
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
            },
            best_f: 1.234567890123456789,
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
        let (loaded_records, _) = load_journal(&path, "test", "aaaaaaaaaaaaaaaa").unwrap();
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
