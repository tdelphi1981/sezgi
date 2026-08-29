# Example triplets: 5 labeled-metaphor algorithms x 3 forms

Each algorithm below ships as three parallel artifacts, all implementing the
SAME pinned update equations documented in the corresponding Rust component
(`crates/components/src/{gwo,woa,hs,cs,goa}.rs`):

1. **`python/<algo>.py`** -- a pure-Python teaching implementation, stdlib
   only (`random`, `math` -- no numpy), driven through `sezgi.EvalSession`
   (an ask/tell session that gives an EXTERNAL algorithm evaluation
   counting, budget enforcement, and optional IOH logging for free, with no
   Rust algorithm code involved).
2. **`r/<algo>.R`** -- the same algorithm in base R, via `sz_eval_session`.
3. **`specs/<algo>.toml`** -- an `ExperimentSpec` TOML running the SAME
   algorithm as sezgi's own built-in, RNG-stream-pinned preset (via
   `sezgi.run_experiment` / `sz_run_experiment`), with a comment block
   explaining its component decomposition.

None of the three forms is RNG-stream-identical to either of the others --
a pure Python `random.Random` and R's RNG do not draw the same sequence as
the Rust component's `RngStream`, and the pure scripts do not attempt to
replicate the Rust preset's exact per-draw contract. What IS identical
across all three is the underlying **mathematical update rule**: read the
Rust module doc first, then the pure script, and the correspondence between
"paper equation" and "code" should be direct.

All three forms use the same fixed scenario for a live run: BBOB function 1
(sphere) at dimension 5, budget 2000 evaluations, seed 42. Every script
prints `evals_used`, `best_f`, and `gap` (`best_f - f_opt`). **No
cross-algorithm quality claims are made anywhere in this catalog or its
scripts** -- these are single-seed, single-problem runs, reported as a gap
against the problem's known optimum, never as a ranking between algorithms.
Comparing algorithms properly needs `sezgi.per_budget_packages`/
`sz_per_budget_packages` over many seeds and problems (see the main
`README.md`'s "Experiments & Statistics" section).

## Catalog

| Algorithm | Primary reference | Equivalence / critique reference | Files | What the pure version teaches vs. the preset |
|---|---|---|---|---|
| Grey Wolf Optimizer (GWO) | Mirjalili, Mirjalili & Lewis (2014), "Grey Wolf Optimizer", *Advances in Engineering Software* | Camacho-Villalón, Dorigo & Stützle (ANTS 2020, three-algorithm study); Camacho-Villalón, Dorigo & Stützle (*International Transactions in Operational Research*, six-algorithm journal extension) | `python/gwo.py`, `r/gwo.R`, `specs/gwo.toml` | The leader-centroid update rule itself (mean of alpha/beta/delta contributions), stdlib-clear; the preset adds the pinned per-`(wolf,dim)` 6-draw RNG-stream contract and IOH logging. |
| Whale Optimization Algorithm (WOA) | Mirjalili & Lewis (2016), "The Whale Optimization Algorithm", *Advances in Engineering Software* | Camacho-Villalón, Dorigo & Stützle (*International Transactions in Operational Research*), which names WOA ("whale") explicitly among six metaphor-based algorithms shown to be, component for component, relabeled special cases of older operators | `python/woa.py`, `r/woa.R`, `specs/woa.toml` | The encircle/search/spiral branch structure, with `p`/`r1`/`r2`/`l` drawn once per whale (matching the reference MATLAB `WOA.m` exactly; only the random-leader index is drawn per dimension); the preset adds the exact branch-dependent draw-order contract and IOH logging. |
| Harmony Search (HS) | Geem, Kim & Loganathan (2001), "A new heuristic optimization algorithm: harmony search", *Simulation* | Weyland (2010), whose analysis shows HS's harmony-memory-consideration/pitch-adjustment mechanism is, component for component, a special case of evolution strategies | `python/hs.py`, `r/hs.R`, `specs/hs.toml` | The memory-consideration/pitch-adjustment/random-selection branch and its single-offspring-per-call shape; the preset adds the pinned per-dimension 2/3/4-draw RNG-stream contract and IOH logging. |
| Cuckoo Search (CS) | Yang, X.-S. & Deb, S. (2009), "Cuckoo Search via Lévy Flights", *2009 World Congress on Nature & Biologically Inspired Computing (NaBIC)*, pp. 210-214, IEEE | -- (not mandated for CS by the task brief) | `python/cs.py`, `r/cs.R`, `specs/cs.toml` | The Lévy-flight step toward the best nest (Mantegna's algorithm, via `math.gamma`/base-R `gamma()`) plus the worst-fraction abandonment; the preset adds the pinned Lévy-sample-call-boundary RNG contract and IOH logging. |
| Grasshopper Optimisation Algorithm (GOA) | Saremi, Mirjalili & Lewis (2017), "Grasshopper optimisation algorithm: theory and application", *Advances in Engineering Software*, 105, 30-47 | -- (none cited in the Rust module's doc) | `python/goa.py`, `r/goa.R`, `specs/goa.toml` | The pairwise social-force update (Eq. 2.3/2.7/2.8) and its VERIFIED `[2,4)`-bounded distance mapping (confirmed against the reference MATLAB via `mealpy`'s `OriginalGOA`), fully RNG-free; the preset adds the pinned j-ascending summation-order contract (float addition is not associative) and IOH logging. |

## Running the examples

Pure Python (stdlib only, from the repo root, using the project's own venv):

    ./py-sezgi/.venv/bin/python examples/python/gwo.py

Pure R (base R only, `sezgi` package installed):

    Rscript examples/r/gwo.R

Spec TOMLs (any of the 5, via `sezgi.run_experiment`):

    import sezgi
    with open("examples/specs/gwo.toml") as f:
        records = sezgi.run_experiment(f.read(), parallel=False)
    print(records[0])

or in R:

    library(sezgi)
    records <- sz_run_experiment(paste(readLines("examples/specs/gwo.toml"), collapse = "\n"))
    print(records)
