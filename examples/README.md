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
