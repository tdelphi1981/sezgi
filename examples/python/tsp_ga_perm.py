"""ga-perm (a fused permutation GA -- OX1 crossover + swap mutation in one
generator, see crates/components/src/presets.rs::ga_perm's doc) on
berlin52, the classic TSPLIB (Reinelt) benchmark instance -- a matched
Python/R example pair for sezgi's TSPLIB/permutation-problem suite.
See examples/r/tsp_ga_perm.R for the R counterpart (1-based tour indices
there vs. this script's 0-based, per r-sezgi's own indexing convention).

Runs through the same `sezgi.solve()` spec-JSON path every other preset
uses: `sezgi.presets.ga_perm(pop_size, budget)` + `sezgi.problems.tsp(name)`.

berlin52's published-optimal tour length is EXACTLY 7542.0 (TSPLIB, Reinelt
1991) -- `ga-perm` is a baseline permutation GA with no local search
(no 2-opt), so it is not expected to reach the optimum; the point of this
example is a valid, budget-respecting solve() run and an honest gap report.

Single seed, single problem, small budget -- a SMOKE demonstration, not a
claim about ga-perm's quality relative to any other algorithm or to a
proper multi-seed benchmark. Comparing algorithms properly needs many seeds
via `sezgi.per_budget_packages`/statistical tests (see README.md's
"Experiments & Statistics" section) -- never a single-seed gap number.
"""
import sezgi

INSTANCE = "berlin52"
POP_SIZE = 32
BUDGET = 5000
SEED = 20260830

info = sezgi.problems.tsp_load(INSTANCE)
known_optimum = info["known_optimum"]
problem = sezgi.problems.tsp(INSTANCE)
spec = sezgi.presets.ga_perm(pop_size=POP_SIZE, budget=BUDGET)

result = sezgi.solve(spec, problem, master_seed=SEED, run_id=0)
best_length = result["best_f"]
best_tour = result["best_x"]

# Independent cross-check: recompute the reported best tour's length directly.
recomputed = sezgi.problems.tsp_tour_length(INSTANCE, best_tour)
assert recomputed == best_length, "solve()'s best_f must match an independent tour_length recomputation"

print(f"TSP {INSTANCE} ({info['n_cities']} cities), ga-perm, pop_size={POP_SIZE} budget={BUDGET} seed={SEED} -- SMOKE DEMO, single seed")
print(f"known optimum (TSPLIB): {known_optimum}")
print(f"best tour length: {best_length}")
print(f"gap (best - optimum): {best_length - known_optimum}  ratio: {best_length / known_optimum:.4f}")
print(f"evals_used: {result['evals_used']}  iterations: {result['iterations']}")
