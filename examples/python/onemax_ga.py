"""ga_bin (KanGAL two-point binary crossover + bit-flip mutation, Deb,
Pratap, Agarwal & Meyarivan 2002, Sec. IV.A -- p_c=0.9, p_m=1/L) on OneMax
(Goldberg 1989's classic "maximize the number of set bits" toy, re-expressed
in this crate as minimizing the ZERO-bit count -- see
crates/problems/src/diagnostics.rs's module doc) -- a matched Python/R
example pair for sezgi's typed-operator support. See
examples/r/onemax_ga.R for the R counterpart: same scenario, same Rust core
underneath (both bindings run the identical `AlgorithmSpec`/`Engine` path,
like every other matched pair in this project), so this pair's numbers are
bit-identical across languages -- only the report formatting differs
(numbers bit-identical, formatting divergence acceptable across every
matched pair in this project).

Runs through the same `sezgi.solve()` spec-JSON path every other preset
uses: `sezgi.presets.ga_bin(pop_size, budget)` + `sezgi.problems.onemax(n_bits)`.

Small budget, seeded, deliberately chosen so the run does NOT reach the
known optimum (`best_f == 0.0`, all bits set) -- a fully-converged run would
be a trivial, uninformative anchor (any correctly terminating optimal run
gives the identical 0.0 regardless of path taken, the same reasoning
r-sezgi's own cross-language anchor report gives for avoiding
its own OneMax scenario's easy exact-optimum case). At n_bits=100,
pop_size=20, budget=400 (== 20 generations past init), seed=7, `ga_bin`
lands on a genuine partial result (16 of 100 bits still zero) that pins the
run's actual trajectory, not just "did it solve the toy."

Single seed, single problem, small budget -- a SMOKE demonstration of the
binding (typed operators + diagnostic problems reachable end-to-end from
Python), not a claim about ga_bin's quality relative to any other algorithm
or to a proper multi-seed benchmark. Comparing algorithms properly needs
many seeds via `sezgi.per_budget_packages`/statistical tests (see
README.md's "Experiments & Statistics" section) -- never a single-seed gap
number.
"""
import sezgi

N_BITS = 100
POP_SIZE = 20
BUDGET = 400
SEED = 7

problem = sezgi.problems.onemax(N_BITS)
spec = sezgi.presets.ga_bin(pop_size=POP_SIZE, budget=BUDGET)

result = sezgi.solve(spec, problem, master_seed=SEED, run_id=0)
best_f = result["best_f"]
best_x = result["best_x"]

# Independent cross-check: recompute the reported best_f directly from
# best_x (OneMax's own zero-bit-count formula) -- the same
# "recompute, don't just trust" pattern tsp_ga_perm.py's own
# tsp_tour_length cross-check uses.
ones = sum(best_x)
zeros = len(best_x) - ones
assert float(zeros) == best_f, "solve()'s best_f must match an independent zero-bit-count recomputation"

known_optimum = 0.0  # OneMax's known optimum: all bits set (Goldberg 1989)
gap = best_f - known_optimum

print(f"OneMax n_bits={N_BITS}, ga_bin, pop_size={POP_SIZE} budget={BUDGET} seed={SEED} -- SMOKE DEMO, single seed")
print(f"known optimum: {known_optimum}")
print(f"evals_used={result['evals_used']} best_f={best_f:.6g} gap={gap:.6g} ones={ones}")
