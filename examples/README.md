# Example triplets: 17 labeled-metaphor algorithms x 3 forms

Each algorithm below ships as three parallel artifacts, all implementing the
SAME pinned update equations documented in the corresponding Rust component
(`crates/components/src/{gwo,woa,hs,cs,goa,sca,jaya,mfo,ssa,fa,ba,fpa,tlbo,hho,alo,abc,gsa}.rs`):

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
| Sine Cosine Algorithm (SCA) | Mirjalili, S. (2016), "SCA: A Sine Cosine Algorithm for Solving Optimization Problems", *Knowledge-Based Systems*, 96, 120-133 | -- (no established equivalence critique covers SCA) | `python/sca.py`, `r/sca.R`, `specs/sca.toml` | The pinned sin/cos position update (Eq. 3.1/3.2), verified against the author's reference `SCA.m` (3 draws -- r2, r3, r4 -- per (agent, dimension), always all three regardless of branch); the preset adds the exact per-(agent,dimension) RNG-stream contract and IOH logging. |
| JAYA | Rao, R.V. (2016), "Jaya: A Simple and New Optimization Algorithm for Solving Constrained and Unconstrained Optimization Problems", *International Journal of Industrial Engineering Computations*, 7(1), 19-34 | -- (no established equivalence critique covers JAYA) | `python/jaya.py`, `r/jaya.R`, `specs/jaya.toml` | Eq. (1)'s best-minus-worst pull, verified bit-for-bit against the paper's own worked numerical example, including the finding that `r1[d]`/`r2[d]` are drawn ONCE PER DIMENSION PER GENERATION and shared across every candidate (not fresh per agent); the preset adds the exact 2*dim-draws-per-generation RNG-stream contract and IOH logging. |
| Moth-Flame Optimization (MFO) | Mirjalili, S. (2015), "Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm", *Knowledge-Based Systems*, 89, 228-249 | Camacho-Villalón, Dorigo & Stützle (*International Transactions in Operational Research*, six-algorithm critique: grey wolf, moth-flame, whale, firefly, bat, antlion) | `python/mfo.py`, `r/mfo.R`, `specs/mfo.toml` | The verified two-index move (distance always reads the moth's own unclamped row `i`; only the additive target flame is clamped to `min(i, flame_count-1)`) and the elitist merge-sort-truncate flame memory, carried here as a plain script-local variable instead of Rust blackboard state; the preset adds the exact RNG-stream contract, the flame memory as real `adapter/mfo-flame-update`-owned state, and IOH logging. |
| Salp Swarm Algorithm (SSA) | Mirjalili, S., Gandomi, A.H., Mirjalili, S.Z., Saremi, S., Faris, H. & Mirjalili, S.M. (2017), "Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems", *Advances in Engineering Software*, 114, 163-191 | -- (no established equivalence critique covers SSA) | `python/ssa.py`, `r/ssa.R`, `specs/ssa.toml` | The verified half-population leader/follower split (a fixed positional split of the array, not a fitness-based single leader) and the in-place follower chain (each follower reads its ALREADY-COMPUTED predecessor entry from the same sweep); the preset adds the exact RNG-stream contract and IOH logging. |
| Firefly Algorithm (FA) | Yang, X.-S., *Nature-Inspired Metaheuristic Algorithms*, 2nd ed., Luniver Press, 2010 | Camacho-Villalón, Dorigo & Stützle (*International Transactions in Operational Research*, six-algorithm critique: grey wolf, moth-flame, whale, firefly, bat, antlion) | `python/fa.py`, `r/fa.R`, `specs/fa.toml` | The floored attractiveness formula (`beta` floors at `betamin`, never vanishes), the closed-form `alpha` decay, and the verified hybrid live/frozen double loop (distance and a firefly's own accumulating term read the LIVE array; the additive move target reads a FROZEN start-of-generation snapshot), fully reproduced with nothing but stdlib math; the preset adds the exact per-firing-pair RNG-stream contract and IOH logging. |
| Bat Algorithm (BA) | Yang, X.-S. (2010), "A new metaheuristic bat-inspired algorithm", in: *Nature Inspired Cooperative Strategies for Optimization (NICSO 2010)*, Studies in Computational Intelligence, vol. 284, Springer, 65-74 | Camacho-Villalón, Dorigo & Stützle (*International Transactions in Operational Research*, six-algorithm critique: grey wolf, moth-flame, whale, firefly, bat, antlion) | `python/ba.py`, `r/ba.R`, `specs/ba.toml` | The two verified composing sign-inversion quirks (an inverted `[-2,0]` frequency range and a `(X-X_best)` velocity term, which together produce net attraction toward best, not repulsion), fixed loudness/pulse-rate constants (the source's own demo never implements their decay), and the persisted velocity, carried here as a plain script-local variable instead of Rust blackboard state; the preset adds the exact RNG-stream contract, `ba/velocity` as real blackboard state, and IOH logging. |
| Flower Pollination Algorithm (FPA) | Yang, X.-S. (2012), "Flower Pollination Algorithm for Global Optimization", in: *Unconventional Computation and Natural Computation* (UCNC 2012), LNCS vol. 7445, Springer, 240-249 | -- (not mandated for FPA by the task brief) | `python/fpa.py`, `r/fpa.R`, `specs/fpa.toml` | The verified branch orientation (`u > p` selects GLOBAL, not `u < p` -- matching the "slight bias towards local pollination" prose) and the local branch's `j`/`k` indices, distinct FROM EACH OTHER ONLY, never excluded from equaling `i`; the global step reuses the SAME Mantegna Lévy step as `cs.py`/`cs.R`; the preset adds the exact per-flower RNG-stream contract and IOH logging. |
| Teaching-Learning-Based Optimization (TLBO) | Rao, R.V., Savsani, V.J., Vakharia, D.P. (2011), "Teaching-learning-based optimization: A novel method for constrained mechanical design optimization problems", *Computer-Aided Design*, 43(3), 303-315 | -- (a DIFFERENT critique targets TLBO's own accounting/framing: Črepinšek, Liu & Mernik (2012), *Information Sciences*, 212, 79-93, "A note on teaching-learning-based optimization algorithm") | `python/tlbo.py`, `r/tlbo.R`, `specs/tlbo.toml` | sezgi's FIRST multi-stage algorithm, modeled here as two EXPLICIT passes per generation (teacher then learner, each with its own greedy accept, 2*pop_size evals/gen) -- verified against Yarpiz's `tlbo.m` (explicitly labeled third-party, not Rao's own code), including `TF=randi{1,2}` drawn per learner and the learner phase's partner drawn UNCONDITIONALLY distinct from `i`; the preset adds the exact per-stage RNG-stream contract and IOH logging. |
| Harris Hawks Optimization (HHO) | Heidari, A.A., Mirjalili, S., Faris, H., Aljarah, I., Mafarja, M. & Chen, H. (2019), "Harris hawks optimization: Algorithm and applications", *Future Generation Computer Systems*, 97, 849-872 | -- (no established equivalence critique covers HHO) | `python/hho.py`, `r/hho.R`, `specs/hho.toml` | The full explore/exploit escape-energy branch tree, verified against the paper AUTHOR's own `HHO.m`, including the CORRECTED soft-dive `Y` formula (`Rabbit - E*\|J*Rabbit - X_i\|`, a genuinely different equation from the no-dive soft besiege) and the rapid dives' mid-loop evaluation (0/1/2 extra evals per diving hawk); the preset adds the exact branch-tree RNG-stream contract, including the interleaved per-dim dive draws, and IOH logging. |
| Ant Lion Optimizer (ALO) | Mirjalili, S. (2015), "The Ant Lion Optimizer", *Advances in Engineering Software*, 83, 80-98 | Camacho-Villalón, Dorigo & Stützle (*International Transactions in Operational Research*, six-algorithm critique: grey wolf, moth-flame, whale, firefly, bat, antlion) | `python/alo.py`, `r/alo.R`, `specs/alo.toml` | The full-horizon cumsum random walk, min-max normalized into a SHRINKING `I`-ratio bound, verified against the author's own `ALO.m`/`Random_walk_around_antlion.m`/`RouletteWheelSelection.m`; the 1/fitness roulette WITH a floor-shift for sezgi's (possibly negative) raw fitness, and the antlion population's own merge-sort-truncate elitism (`replace/mu-plus-lambda`, no blackboard state needed); the preset adds the exact per-ant RNG-stream contract and IOH logging. |
| Artificial Bee Colony (ABC) | Karaboga, D. (2005), "An Idea Based On Honey Bee Swarm For Numerical Optimization", TR-06, Erciyes University; Karaboga, D., Basturk, B. (2007), "A Powerful and Efficient Algorithm for Numerical Function Optimization: ABC Algorithm", *Journal of Global Optimization*, 39(3), 459-471 | -- (no established equivalence critique covers ABC) | `python/abc.py`, `r/abc.R`, `specs/abc.toml` | The three-phase decomposition (employed one-dimension-move + trial counters, onlooker's repeated `0.9*fit/max+0.1` linear scan -- not literal roulette-wheel sampling, scout's single-argmax-with-last-index-tiebreak abandonment), verified against Karaboga & Basturk's OWN `ABCorig.m`; the `calculateFitness` transform (`f>=0 -> 1/(f+1)`, `f<0 -> 1+\|f\|`); the preset adds the exact per-phase RNG-stream contract, real blackboard-owned trial counters, and IOH logging. |
| Gravitational Search Algorithm (GSA) | Rashedi, E., Nezamabadi-pour, H. & Saryazdi, S. (2009), "GSA: A Gravitational Search Algorithm", *Information Sciences*, 179(13), 2232-2248 | -- (no established equivalence critique covers GSA) | `python/gsa.py`, `r/gsa.R`, `specs/gsa.toml` | The wave's LAST stateful algorithm: mass normalization (with a degenerate all-equal-fitness guard), exponential `G` decay, a Kbest elite set that SHRINKS with progress (no floor -- can reach 0 agents), a per-`(i,j,d)` force draw with NO `M_i` term (`Gfield.m`'s own "Mp(i)/Mi(i)=1" comment), and `v'=rand*v+accel` (multiplicative, not additive, unlike `ba.py`'s velocity term), carried here as a plain script-local variable; the preset adds the exact RNG-stream contract, `gsa/velocity` as real blackboard state, and IOH logging. |

## Running the examples

Pure Python (stdlib only, from the repo root, using the project's own venv):

    ./py-sezgi/.venv/bin/python examples/python/gwo.py

Pure R (base R only, `sezgi` package installed):

    Rscript examples/r/gwo.R

Spec TOMLs (any of the 17, via `sezgi.run_experiment`):

    import sezgi
    with open("examples/specs/gwo.toml") as f:
        records = sezgi.run_experiment(f.read(), parallel=False)
    print(records[0])

or in R:

    library(sezgi)
    records <- sz_run_experiment(paste(readLines("examples/specs/gwo.toml"), collapse = "\n"))
    print(records)

## Multi-objective: NSGA-II on ZDT1

A separate, matched PAIR (not a triplet like the 17 above): NSGA-II (Deb,
Pratap, Agarwal & Meyarivan 2002) on ZDT1 (Zitzler, Deb & Thiele 2000),
run directly through `sezgi.mo.nsga2()` / `sz_nsga2()` -- both bindings
call the SAME Rust `nsga2_run` core, so this pair's reported numbers are
bit-identical across languages, not merely statistically comparable like
the pure-Python/pure-R algorithm scripts above.

| Algorithm | Primary reference | Files | What it demonstrates |
|---|---|---|---|
| NSGA-II (on ZDT1) | Deb, Pratap, Agarwal & Meyarivan (2002), "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II", *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197; problem: Zitzler, Deb & Thiele (2000), "Comparison of Multiobjective Evolutionary Algorithms: Empirical Results", *Evolutionary Computation*, 8(2), 173-195 | `python/nsga2_zdt1.py`, `r/nsga2_zdt1.R` | A one-call, seeded, small-budget NSGA-II run, reporting final non-dominated front size, `evals_used`, 2-objective hypervolume, and IGD against a 200-point analytic Pareto-front sample, plus plot-ready `(f1, f2)` front points. |

**No `specs/nsga2_zdt1.toml` exists in this catalog.** NSGA-II ships as a
self-contained, seeded Rust runner over a parallel `MoProblem`/
`MoEvaluator` surface, not as a composable `ExperimentSpec` component
graph -- the MO spec-graph integration this catalog's `specs/*.toml` path
depends on is deferred to v2 (see the main `README.md`'s "Multi-objective
optimization" section for the full spec-tension ruling).

Run:

    ./py-sezgi/.venv/bin/python examples/python/nsga2_zdt1.py
    Rscript examples/r/nsga2_zdt1.R

Live output (`dim=10`, `pop_size=40`, `budget=4000`, `seed=20260830`,
measured by running both scripts from the repo root):

    problem: zdt1  dim=10  pop_size=40  budget=4000  seed=20260830
    front size: 40
    evals_used: 4000
    hypervolume_2d (ref_point=[1.1, 1.1]): 0.8580576535101335
    igd (vs pareto_front(200)): 0.01254540902919091
    front points (f1, f2), sorted by f1, first 5 (plot-ready -- x=f1, y=f2):
      0.000000  1.003359
      0.000911  0.972881
      0.003438  0.944386
      0.007097  0.917851
      0.015445  0.878219

## CEC 2022 and TSPLIB/permutation quickstarts

Two more matched Python/R PAIRS (not triplets -- no `specs/*.toml` for
either, see each script's own header for why), both going through
`sezgi.solve()` / r-sezgi's established `sz_solve_*`/`sz_preset_*` path:

| Pair | Files | What it demonstrates |
|---|---|---|
| SHADE on CEC 2022 f3 | `python/cec2022_shade.py`, `r/cec2022_shade.R` | `sezgi.presets.shade` / `sz_preset_shade` (Tanabe & Fukunaga 2013) solved against `sezgi.problems.cec2022(fid, dim)` / `sz_solve_cec2022` via `sezgi.solve()` / r-sezgi's own `sz_solve_cec2022`, printing `best_f` and the gap to the report's pinned `F*`. **Both scripts now run SHADE** (`sz_solve_cec2022` was added later than the rest of the CEC surface; r-sezgi previously had direct evaluation only, `sz_cec2022_evaluate`/`sz_cec2022_f_star`, and no `solve()`-integrated path). |
| ga-perm on TSPLIB berlin52 | `python/tsp_ga_perm.py`, `r/tsp_ga_perm.R` | `sezgi.presets.ga_perm` / `sz_preset_ga_perm` (a fused OX1-crossover + swap-mutation permutation GA) solved against the vendored `berlin52` TSPLIB instance (`sezgi.problems.tsp` / `sz_solve_tsp`), printing the best tour length against berlin52's published TSPLIB optimum (7542.0). The Python script uses 0-based tour indices; the R script uses 1-based (r-sezgi's own established indexing convention, matching TSPLIB's own node numbering). |

Both pairs are single-seed, single-problem SMOKE demonstrations of the
binding surface, reported as a gap against a known optimum -- never a
cross-algorithm or cross-language quality claim. **Both pairs now run the
SAME algorithm through the SAME Rust core in both languages** (the CEC pair
closed its two-different-algorithms gap later on, see the table row
above): their Python/R `best_f` numbers agree BIT-FOR-BIT, verified via a
`writeBin`/`struct.pack` byte comparison, not a decimal-literal
eyeball-match (`r-sezgi/tests/testthat/test-cec-tsp.R`'s "R sz_solve_cec2022
(SHADE) is bit-identical to the Python/Rust golden" test). See each
script's own header comment for full provenance and the exact numbers from
a real run.

Run:

    ./py-sezgi/.venv/bin/python examples/python/cec2022_shade.py
    Rscript examples/r/cec2022_shade.R
    ./py-sezgi/.venv/bin/python examples/python/tsp_ga_perm.py
    Rscript examples/r/tsp_ga_perm.R

Live output (measured by running all four scripts from the repo root;
`seed=20260830` throughout):

    CEC 2022 f3 (dim=10), SHADE, pop_size=20 budget=5000 seed=20260830 -- SMOKE DEMO, single seed
    F* (report's pinned optimum): 600.0
    best_f: 600.0040181437115
    gap (best_f - F*): 0.004018143711505218
    evals_used: 5000  iterations: 249

    CEC 2022 f3 (dim=10), SHADE, pop_size=20 budget=5000 seed=20260830 -- SMOKE DEMO, single seed
    F* (report's pinned optimum): 600
    best_f: 600.004018143712
    gap (best_f - F*): 0.00401814371150522
    evals: 5000

    TSP berlin52 (52 cities), ga-perm, pop_size=32 budget=5000 seed=20260830 -- SMOKE DEMO, single seed
    known optimum (TSPLIB): 7542.0
    best tour length: 11771.0
    gap (best - optimum): 4229.0  ratio: 1.5607
    evals_used: 4992  iterations: 155

The R `cec2022_shade.R` run prints `best_f: 600.004018143712` against
Python's `600.0040181437115` -- these are the SAME IEEE-754 double, just
printed with different default precision (R's `cat`/`sprintf("%s", .)`
shows ~15 significant digits, Python's `repr` shows the shortest
round-tripping representation); a `writeBin`/`struct.pack` byte comparison
confirms bit-identical bytes (`4082c0083aaa1ea7` on both sides). The R
`tsp_ga_perm.R` run prints the same tour length (`11771`) -- the TSP pair
runs the same Rust core in both languages, so the numbers agree exactly
(R's default printing drops the trailing `.0`).

## OOP twins

`examples/python/oop/` holds a fourth artifact per algorithm: the SAME 17
algorithms as the catalog above (gwo, woa, hs, cs, goa, sca, jaya, mfo,
ssa, fa, ba, fpa, tlbo, hho, alo, abc, gsa), each ported onto
`sezgi.AskTellAlgorithm` (named `sezgi.Algorithm` before it was renamed --
`sezgi.Algorithm` now names an unrelated, engine-hosted class-first base,
see "Class-first authoring + a data recipe" below) -- a subclass
implementing `setup(ctx)`/`step(ctx)` over `AlgoContext`, driven by the
same `sezgi.EvalSession` core the pure `python/<algo>.py` scripts already
use, but expressed as an OOP template method instead of a bare script.
See the main `README.md`'s "Write your own algorithm, ask/tell style
(Python)" section for the authoring guide these twins demonstrate.

**Bit-exact parity, not merely statistical equivalence.** Each twin is a
verbatim RNG-draw-order port of its pure script sibling -- same
`random.Random` seed, same per-generation draw sequence, same
`ctx.evaluate`/`BudgetExhausted` boundary reproducing the pure script's own
`while used + N <= BUDGET` guard -- so at the shared scenario (BBOB f1,
dim 5, budget 2000, seed 42) the twin's printed `evals_used`/`best_f`/`gap`
fields are STRING-IDENTICAL to the pure script's, not just close. This is
gated by `py-sezgi/tests/test_examples_oop_parity.py`: for each of the 17
algorithms it runs both `examples/python/<algo>.py` and
`examples/python/oop/<algo>.py` as subprocesses and compares their printed
`evals_used=... best_f=... gap=...` fields -- each formatted `%.6g` by the
scripts themselves (pure Python, untouched by this project) -- for exact
string equality. The gate therefore enforces agreement to 6 significant
digits, not full IEEE-754 precision; the stronger claim holds too, but was
verified separately, not by this gate: the 2026-08-30 final whole-branch
review additionally captured the raw f64 bit patterns of `best_f`, `gap`,
`evals_used`, and every `best_x` coordinate for all 17 pairs and found them
byte-identical on all four, confirming the twins are bit-exact and not
merely 6-digit-equal.

**The pure scripts remain the pedagogical/provenance originals** --
`examples/python/<algo>.py` files were not touched by this port (verified
empty `git diff` against all 17 at every porting task's gate); they stay
the primary teaching artifact this catalog's table above describes ("what
the pure version teaches vs. the preset"). The OOP twins are a SEPARATE,
additional artifact demonstrating the `sezgi.AskTellAlgorithm` authoring
surface on already-understood algorithms, not a replacement for the pure
scripts.

Run (any of the 17; using gwo here):

    ./py-sezgi/.venv/bin/python examples/python/oop/gwo.py

Live output (same scenario/seed as the pure script above it):

    gwo (oop): evals_used=1980 best_f=-125.949 gap=0.000659831

which matches `./py-sezgi/.venv/bin/python examples/python/gwo.py`'s own
`gwo: evals_used=1980 best_f=-125.949 gap=0.000659831` field-for-field.

## R authoring example

`examples/r/oop/gwo.R` is the R-side counterpart to the Python OOP twins
above — ONE worked twin (not a full 17-algorithm wave, per this scope's
own ruling: the 17 pure-R scripts under `examples/r/` already teach
the algorithms; the pure-R authoring surface itself is what needed a
worked proof), porting `examples/r/gwo.R` onto `sz_algorithm`/
`sz_algo_solve` (r-sezgi's pure-R mirror of `sezgi.AskTellAlgorithm`,
base-R closures/environments/condition classes only — see the main
`README.md`'s "Write your own algorithm (R)" section for the
authoring guide this twin demonstrates).

**Bit-exact parity, achieved by draw-order identity, not restated
statistical closeness.** `sz_algo_solve()` calls R's `set.seed(seed)`
exactly once, up front, and R's global RNG is the ONE stream both
`ctx$random_point()` and any direct `runif()` call inside `setup()`/
`step()` draw from — the SAME stream and SAME per-call shape the pure
`examples/r/gwo.R` script already uses. The twin's `setup()` reproduces the
pure script's init draws verbatim (`ctx$random_point()` == `runif(dim, lo,
hi)`, same `lapply` order); `step()` reproduces one generation's `for (i)
for (d) runif(6)` nesting verbatim; `ctx$evaluate`'s `sz_budget_exhausted`
condition reproduces the pure script's own `while used + pop_size <=
budget` guard as a boundary condition instead of a loop precondition (see
`examples/r/oop/gwo.R`'s own header comment for the one structural
difference this introduces — a final, discarded burst of RNG draws on a
generation whose batch never reaches the session — and why it provably
does not change the printed output). Because of this, at the shared
scenario (BBOB f1, dim 5, budget 2000, seed 42) the twin's printed
`evals_used`/`best_f`/`gap` fields are STRING-IDENTICAL to the pure
script's, verified live, not merely argued — gated by
`r-sezgi/tests/testthat/test-algo.R`'s "R OOP gwo twin matches the pure
gwo.R script" test, which runs both `examples/r/gwo.R` and
`examples/r/oop/gwo.R` as `Rscript` subprocesses and compares their printed
`evals_used=... best_f=... gap=...` fields for exact string equality
(mirroring `py-sezgi/tests/test_examples_oop_parity.py`'s own subprocess/
regex mechanics).

**The pure script remains the pedagogical/provenance original** —
`examples/r/gwo.R` was not touched by this port (verified empty `git diff`
against it at this task's gate); it stays the primary teaching artifact the
catalog table above describes. The twin is a SEPARATE, additional artifact
demonstrating the `sz_algorithm` authoring surface on an already-understood
algorithm, not a replacement for the pure script.

Run:

    Rscript examples/r/gwo.R
    Rscript examples/r/oop/gwo.R

Live output (same scenario/seed as the pure script, measured by running
both from the repo root):

    gwo: evals_used=1980 best_f=-125.949 gap=0.000431711
    gwo (oop): evals_used=1980 best_f=-125.949 gap=0.000431711

field-for-field identical (R's own RNG stream differs from Python's — the
R pair's `best_f`/`gap` are not expected to match the Python pair's; only
each language's pure/twin pair is compared).

## CEC 2014: L-SHADE quickstart

`python/cec2014_lshade.py` / `r/cec2014_lshade.R` — one more matched
Python/R PAIR (no `specs/cec2014_lshade.toml`, same reasoning as the CEC
2022/TSP pairs above), going through `sezgi.solve()` / `sz_solve_cec2014`:
L-SHADE (Tanabe & Fukunaga 2014, the CEC 2014 competition's own 1st-place
algorithm — `presets.lshade` / `sz_preset_lshade`, parameterized by `dim`
rather than `pop_size`) against `sezgi.problems.cec2014(fid, dim)` /
`sz_solve_cec2014`, on f1 ("Rotated High Conditioning Elliptic Function",
dim=10). Single seed, single problem, small budget — a SMOKE demonstration
of the CEC 2014 binding, not a claim about L-SHADE's quality or
convergence rate; f1's ill-conditioned scale keeps `best_f` visibly above
`F*` at this budget by design, not by defect. See `README.md`'s "CEC
2014/CEC 2017 benchmark suites" section for the suite's full
provenance and every report-vs-official-C divergence found.

**Bit-identical between languages**, verified via the same
`writeBin`/`struct.pack` byte comparison the CEC 2022 pair above uses, not
a decimal-literal eyeball match
(`r-sezgi/tests/testthat/test-cec1417.R`'s "R sz_solve_cec2014 (L-SHADE)
is bit-identical to the Python/Rust golden" test, anchored on this pair's
own live scenario).

Run:

    ./py-sezgi/.venv/bin/python examples/python/cec2014_lshade.py
    Rscript examples/r/cec2014_lshade.R

Live output (measured by running both scripts from the repo root;
`fid=1`, `dim=10`, `budget=9000`, `seed=20260830`):

    CEC 2014 f1 (dim=10), L-SHADE, budget=9000 seed=20260830 -- SMOKE DEMO, single seed
    F* (report's pinned optimum): 100.0
    best_f: 180.23000508877507
    gap (best_f - F*): 80.23000508877507
    evals_used: 9000  iterations: 192

    CEC 2014 f1 (dim=10), L-SHADE, budget=9000 seed=20260830 -- SMOKE DEMO, single seed
    F* (report's pinned optimum): 100
    best_f: 180.230005088775
    gap (best_f - F*): 80.2300050887751
    evals: 9000

The R run prints `best_f: 180.230005088775` against Python's
`180.23000508877507` — the SAME IEEE-754 double, just printed with
different default precision (R's `cat`/`sprintf("%s", .)` shows ~15
significant digits, Python's `repr` shows the shortest round-tripping
representation); a `writeBin`/`struct.pack` byte comparison confirms
bit-identical bytes (`4066875c33a1c67b` on both sides).

## WFG4 + sezgi-moa quickstart

`python/wfg4_nsga2.py` / `r/wfg4_nsga2.R` — a matched Python/R PAIR for
the multi-objective remainders (no `specs/wfg4_nsga2.toml`, same
reasoning as `nsga2_zdt1`'s own pair above: NSGA-II is a self-contained
Rust runner, not a composable component graph), calling
`sezgi.mo.nsga2()` / `sz_nsga2()` directly rather than through
`sezgi.solve()`. Runs NSGA-II on WFG4 (Huband, Barone, While & Hingston,
EMO 2005; later Huband, Hingston, Barone & While, IEEE TEC 2006), `m=2`,
`k`/`l` omitted so they resolve to the toolkit's own recommended defaults
(`k=4`, `l=20`, `dim=24`), with `log_dir`/`label` set so the run streams
to a sezgi-moa v1 archive file — read back via `mo.read_moa()` /
`sz_mo_read_moa()` and reported through the general-`M` `mo.hypervolume()`
/ `sz_mo_hypervolume()` (not the frozen 2-objective
`hypervolume_2d`/`hypervolume_2d`, deliberately, to exercise the new
entry point). Reference point: the WFG4/`m=2` analytic front's nadir
`(2.0, 4.0)` scaled by `1.1` — `(2.2, 4.4)` — per
`crates/stats/src/moo_indicators.rs`'s own reference-point convention
(see that script's own header comment for the full derivation). See
`README.md`'s "Multi-objective optimization" section for the full
binding surface.

**Bit-identical between languages**, verified via the same
`writeBin`/`struct.pack` byte comparison the other pairs above use, not a
decimal-literal eyeball match (`r-sezgi/tests/testthat/test-mo.R`'s "R
wfg4_nsga2 example scenario is bit-identical to the Python/Rust golden"
test, anchored on this pair's own live scenario).

Run:

    ./py-sezgi/.venv/bin/python examples/python/wfg4_nsga2.py
    Rscript examples/r/wfg4_nsga2.R

Live output (measured by running both scripts from the repo root;
`problem=wfg4`, `m=2`, `pop_size=40`, `budget=4000`, `seed=20260830`):

    problem: wfg4  m=2  k=4 l=20 (toolkit defaults, dim=24)  pop_size=40  budget=4000  seed=20260830
    evals_used: 4000
    archive size: 217
    hypervolume (ref_point=[2.2, 4.4]): 3.0620256270488344
    front0 points (f1, f2), sorted by f1, first 5 (plot-ready -- x=f1, y=f2):
      0.095382  4.046673
      0.162711  4.039493
      0.243813  4.032702
      0.327011  3.995469
      0.498196  3.950213

    problem: wfg4  m=2  k=4 l=20 (toolkit defaults, dim=24)  pop_size=40  budget=4000  seed=20260830
    evals_used: 4000
    archive size: 217
    hypervolume (ref_point=[2.2,4.4]): 3.06202562704883
    front0 points (f1, f2), sorted by f1, first 5 (plot-ready -- x=f1, y=f2):
      0.095382  4.046673
      0.162711  4.039493
      0.243813  4.032702
      0.327011  3.995469
      0.498196  3.950213

The R run prints `hypervolume ... : 3.06202562704883` against Python's
`3.0620256270488344` — the SAME IEEE-754 double, just printed with
different default precision (same R-vs-Python printing gap as the CEC
pairs above); a `writeBin`/`struct.pack` byte comparison confirms
bit-identical bytes (`40087f074abd8254` on both sides), and `archive
size`/`evals_used` (plain integers) print identically in both languages.

## Class-first authoring + a data recipe

Three more worked examples over the NEW engine-hosted class-first surface
(`sezgi.Algorithm`/`PopulationAlgorithm`/`LocalSearch` — NOT the ask/tell
`AskTellAlgorithm` the OOP twins above use) plus one data recipe. See the
main `README.md`'s "Author your own algorithm (Python, class-first)" and
"Data recipes: feature selection" sections for the full walkthroughs.

| Example | Base | Files | What it demonstrates |
|---|---|---|---|
| DE/rand/1-shaped mutation | `PopulationAlgorithm` (overrides only `vary()`) | `python/oop/custom_de_variant.py` | A ~14-line `vary()` override producing DE/rand/1-shaped offspring (`r1 + F*(r2-r3)`, `F=0.5`, three independent donor draws per offspring slot via `ctx.rng`, no separate crossover step); the default `select()` (seeded binary tournament) is left in place. Runs on `sezgi.bbob(1, 10, 1)`. |
| Perturbation local search | `LocalSearch` (overrides only `neighbor()`) | `python/oop/custom_local_search.py` | A per-coordinate uniform-perturbation `neighbor()` override (`[-0.3, 0.3]` via `ctx.rng.next_f64()`); the default `accept()` (greedy) is left in place, `pop_size=1`. Runs on `sezgi.bbob(1, 10, 1)`. |
| Feature selection | `Problem` subclass (`sezgi.recipes.FeatureSelection`) + `GeneticAlgorithm` Binary auto-dispatch | `python/oop/feature_selection.py` | A binary-mask feature-selection objective (`scorer(X[:, mask], y) + penalty*popcount/n_features`) recovering a known 3-column informative subset from a fixed synthetic 20x8 dataset (an OLS-residual-sum-of-squares `scorer`, pure numpy, no sklearn), via `GeneticAlgorithm`'s Binary space-kind auto-dispatch — no manual preset choice. |

Like `tsp_two_opt.py` above, `custom_de_variant.py`/`custom_local_search.py`
sit OUTSIDE the 17-pair OOP-twin parity gate (no pure-script counterpart
exists to reproduce) — each gets its own anchored pytest in
`py-sezgi/tests/test_oop_families.py`; `feature_selection.py` is gated by
`py-sezgi/tests/test_feature_selection_example.py` (including a full
`2**8 = 256`-mask brute-force cross-check that the recovered mask is the
unique global minimum).

Run:

    ./py-sezgi/.venv/bin/python examples/python/oop/custom_de_variant.py
    ./py-sezgi/.venv/bin/python examples/python/oop/custom_local_search.py
    ./py-sezgi/.venv/bin/python examples/python/oop/feature_selection.py

Live output (measured by running all three scripts from the repo root):

    custom_de_variant (oop): evals_used=2000 best_f=-59.3946 gap=25.0094
    custom_local_search (oop): evals_used=2000 best_f=-84.3232 gap=0.0807033
    feature_selection (oop): evals_used=200 best_f=0.3607495483 popcount=3 mask=01010010 recovered=True

**`tsp_two_opt.py` now imports `sezgi.AskTellAlgorithm`.** This work renamed
the ask/tell `Algorithm` ABC to `AskTellAlgorithm` (`sezgi.Algorithm` now
names the new engine-hosted class-first base used by the three examples
above); `examples/python/oop/tsp_two_opt.py` and its 17
`examples/python/oop/<algo>.py` siblings in the "OOP twins" section
above updated their import accordingly — an import-name-only change, no
assertion or printed number affected. `tsp_two_opt.py`'s own output
(`evals_used=2000 best_f=9077 gap=1535 tour_length=9077`, see the main
`README.md`'s "Write your own algorithm, ask/tell style (Python)" and
"Typed operators, mixed spaces, and diagnostic problems" sections)
is unchanged.

## Class-first authoring + a data recipe (R)

Three more worked examples, the R twins of the three above, over the NEW
engine-hosted class-first surface (`sezgi::Algorithm`/`PopulationAlgorithm`/
`LocalSearch`, R6 classes — NOT the `sz_algorithm`/`sz_algo_solve` ask/tell
surface the "R authoring example" section above uses) plus one data
recipe. See the main `README.md`'s "Author your own algorithm (R,
class-first)" and "Data recipes: feature selection (R)" sections for the
full walkthroughs.

| Example | Base | Files | What it demonstrates |
|---|---|---|---|
| DE/rand/1-shaped mutation | `PopulationAlgorithm` (overrides only `vary()`) | `r/oop/custom_de_variant.R` | A `vary()` override producing DE/rand/1-shaped offspring (`r1 + F*(r2-r3)`, `F=0.5`, three independent donor draws per offspring slot via `ctx$rng$next_below()`, no separate crossover step); the default `select()` (seeded binary tournament) is left in place. Runs on `sz_builtin_bbob(1, 10, 1)`. |
| Perturbation local search | `LocalSearch` (overrides only `neighbor()`) | `r/oop/custom_local_search.R` | A per-coordinate uniform-perturbation `neighbor()` override (`[-0.3, 0.3]` via `ctx$rng$next_f64()`); the default `accept()` (greedy) is left in place, `pop_size = 1`. Runs on `sz_builtin_bbob(1, 10, 1)`. |
| Feature selection | `Problem` subclass (`FeatureSelection`) + `GeneticAlgorithm` Binary auto-dispatch | `r/oop/feature_selection.R` | A binary-mask feature-selection objective (`scorer(X[, mask, drop = FALSE], y) + penalty*popcount/n_features`) recovering a known 3-column informative subset from a fixed synthetic 20x8 dataset (an OLS-residual-sum-of-squares `scorer`, pure base R, no external package), via `GeneticAlgorithm`'s Binary space-kind auto-dispatch — no manual preset choice. The dataset is an INDEPENDENT R derivation (`set.seed()`/`rnorm()`), not the Python twin's own numpy fixture, so its `best_f` differs from `feature_selection.py`'s. |

Like `tsp_two_opt.R` above, `custom_de_variant.R`/`custom_local_search.R`
sit OUTSIDE the 17-pair OOP-twin parity gate (no pure-script counterpart
exists to reproduce) — each gets its own anchored `stopifnot()` at the end
of the script; `feature_selection.R` is gated by
`r-sezgi/tests/testthat/test-oop-recipes.R` (including a full `2^8 =
256`-mask brute-force cross-check that the recovered mask is the unique
global minimum) and its own `stopifnot(recovered)`.

Run:

    Rscript examples/r/oop/custom_de_variant.R
    Rscript examples/r/oop/custom_local_search.R
    Rscript examples/r/oop/feature_selection.R

Live output (measured by running all three scripts from the repo root,
against the INSTALLED package):

    custom_de_variant (oop): evals_used=2000 best_f=-59.3946
    custom_local_search (oop): evals_used=2000 best_f=-84.3232
    feature_selection (oop): evals_used=200 best_f=0.6562182758 popcount=3 mask=01010010 recovered=TRUE

**`gwo.R`/`tsp_two_opt.R` above are unaffected by the class-first
addition** — they exercise the SEPARATE, unchanged `sz_algorithm`/
`sz_algo_solve` ask/tell surface (no rename, no name collision to resolve,
unlike Python's `AskTellAlgorithm` rename). No existing R example's
printed numbers changed.
