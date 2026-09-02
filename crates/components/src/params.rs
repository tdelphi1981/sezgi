//! Shared parameter-name-aliasing helper, closing the crossover-probability
//! spelling split `compound.rs`'s module doc and README's compound example
//! used to document: the legacy generators (`gen/ga-real`, `gen/ox`,
//! `gen/ga-perm`) parsed only `pc`, while every M3-8 typed family
//! (`gen/bin-2pt`/`gen/ga-bin`, `gen/int-sbx`/`gen/ga-int`, `gen/cat-ux`/
//! `gen/ga-cat`) parsed only `p_c` -- since every `from_params` here is
//! permissive (unknown keys silently ignored), writing the "wrong" spelling
//! for a given family silently fell back to the default instead of erroring.
//!
//! Every family now accepts BOTH spellings via [`resolve`]. Canonical
//! `p_c`/`p_m` is the spelling documented and recommended going forward;
//! `pc`/`pm` (bare, no underscore) is accepted as a legacy alias so every
//! EXISTING spec (either spelling) keeps parsing identically. Checked: no
//! `pm`/`p_m` split actually exists anywhere in this crate today (every
//! family that has a per-gene/per-variable mutation-probability key already
//! spells it `p_m`; `ga.rs`'s `pm_per_gene` is a distinct, unrelated key
//! name, not a `pm`/`p_m` spelling variant) -- `eta_c`/`eta_m`/
//! `tournament_k` are likewise spelled identically everywhere already -- so
//! [`resolve`] is wired up only where a real split existed: every family's
//! crossover-probability key.
//!
//! **Canonical-wins rule**: if a spec sets BOTH the canonical and the legacy
//! key on the same block, the canonical key's value is used and the legacy
//! key is ignored outright (not merged, not validated, not warned about) --
//! see [`resolve`]'s own doc for the precise lookup order this implements.

use serde_json::Value;

/// Looks up `canonical` first; if absent, falls back to `legacy`; if
/// neither is present, returns `None` (the caller's own `.unwrap_or(...)`
/// default then applies, exactly as before this alias existed). When BOTH
/// keys are present on `p`, `canonical` wins outright -- `legacy`'s value is
/// never even inspected in that case.
pub(crate) fn resolve<'a>(p: &'a Value, canonical: &str, legacy: &str) -> Option<&'a Value> {
    p.get(canonical).or_else(|| p.get(legacy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn falls_back_to_legacy_when_canonical_absent() {
        let p = serde_json::json!({"pc": 0.7});
        assert_eq!(resolve(&p, "p_c", "pc").and_then(|v| v.as_f64()), Some(0.7));
    }

    #[test]
    fn canonical_wins_when_both_present() {
        let p = serde_json::json!({"p_c": 0.3, "pc": 0.9});
        assert_eq!(resolve(&p, "p_c", "pc").and_then(|v| v.as_f64()), Some(0.3));
    }

    #[test]
    fn none_when_neither_present() {
        let p = serde_json::json!({});
        assert!(resolve(&p, "p_c", "pc").is_none());
    }
}
