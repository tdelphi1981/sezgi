"""SHADE (Tanabe & Fukunaga 2013) on CEC 2022 (Kumar, Price, Mohamed, Hadi &
Suganthan 2021) function f3, dim=10 -- a matched Python/R example pair for
sezgi's M3-3 CEC 2022 benchmark suite. See examples/r/cec2022_shade.R for
the R counterpart.

f3 is the CEC 2022 report's "Shifted and full Rotated Expanded Schaffer's
f6 Function" BY NAME -- but the vendored official C reference actually
dispatches problem 3 to plain Schaffer's F7 with no scale and (due to a
verified buffer-reuse bug in the reference code) no effective rotation;
sezgi follows the C code, which is what scored the competition, not the
report's printed formula. See docs/DECISIONS.md's M3-3 record and
crates/problems/src/cec2022/mod.rs's module doc ("F3: report says...") for
the full, quoted discrepancy.

Runs through `sezgi.solve()` with the `presets.shade` reference-tier preset
(Tanabe & Fukunaga's Success-History based Adaptive DE) against
`sezgi.problems.cec2022(fid, dim)`, the same solve()-integrated path
`sezgi.presets.ga_perm`/`sezgi.problems.tsp` use in tsp_ga_perm.py.

Single seed, single problem, small budget -- a SMOKE demonstration of the
CEC 2022 binding, not a claim about SHADE's quality or convergence rate.
Comparing algorithms properly needs many seeds via
`sezgi.per_budget_packages`/statistical tests (see README.md's "Experiments
& Statistics" section) -- never a single-seed gap number.
"""
import sezgi

FID = 3
DIM = 10
POP_SIZE = 20
BUDGET = 5000
SEED = 20260830

f_star = sezgi.problems.cec2022_f_star(FID)
problem = sezgi.problems.cec2022(FID, DIM)
spec = sezgi.presets.shade(pop_size=POP_SIZE, budget=BUDGET)

result = sezgi.solve(spec, problem, master_seed=SEED, run_id=0)

print(f"CEC 2022 f{FID} (dim={DIM}), SHADE, pop_size={POP_SIZE} budget={BUDGET} seed={SEED} -- SMOKE DEMO, single seed")
print(f"F* (report's pinned optimum): {f_star}")
print(f"best_f: {result['best_f']}")
print(f"gap (best_f - F*): {result['best_f'] - f_star}")
print(f"evals_used: {result['evals_used']}  iterations: {result['iterations']}")
