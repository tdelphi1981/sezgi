"""L-SHADE (Tanabe & Fukunaga 2014) on CEC 2014 (Liang, Qu & Suganthan 2013)
function f1, dim=10 -- a matched Python/R example pair for sezgi's M3-6 CEC
2014 benchmark suite. See examples/r/cec2014_lshade.R for the R
counterpart.

L-SHADE (SHADE with linear population-size reduction) placed 1st in the
CEC 2014 competition itself, making it the natural pairing for this suite --
the same relationship `examples/python/cec2022_shade.py` has to CEC 2022
(SHADE, the CEC 2013 winner and L-SHADE's own ancestor).

f1 is CEC 2014's "Rotated High Conditioning Elliptic Function" -- a
unimodal function with an ill-conditioned, widely spread per-coordinate
scale, which is why even a competitive optimizer's `best_f` stays visibly
above F* at a small smoke-demo budget; this is expected for f1's shape, not
a defect. See docs/DECISIONS.md's M3-6 record for the CEC 2014 suite's
provenance, vendored-data totals, and every report-vs-official-C divergence
found across the whole suite (f1 itself has none).

Runs through `sezgi.solve()` with the `presets.lshade` reference-tier
preset against `sezgi.problems.cec2014(fid, dim)`, the same solve()-
integrated path `presets.shade`/`problems.cec2022` use in
cec2022_shade.py. `presets.lshade` is parameterized by `dim` (its initial
population size is `18 * dim`, then shrinks per Tanabe & Fukunaga's own
linear reduction schedule), not `pop_size` directly -- unlike `presets.shade`.

Single seed, single problem, small budget -- a SMOKE demonstration of the
CEC 2014 binding, not a claim about L-SHADE's quality or convergence rate.
Comparing algorithms properly needs many seeds via
`sezgi.per_budget_packages`/statistical tests (see README.md's "Experiments
& Statistics" section) -- never a single-seed gap number.
"""
import sezgi

FID = 1
DIM = 10
BUDGET = 9000
SEED = 20260830

f_star = sezgi.problems.cec2014_f_star(FID)
problem = sezgi.problems.cec2014(FID, DIM)
spec = sezgi.presets.lshade(dim=DIM, budget=BUDGET)

result = sezgi.solve(spec, problem, master_seed=SEED, run_id=0)

print(f"CEC 2014 f{FID} (dim={DIM}), L-SHADE, budget={BUDGET} seed={SEED} -- SMOKE DEMO, single seed")
print(f"F* (report's pinned optimum): {f_star}")
print(f"best_f: {result['best_f']}")
print(f"gap (best_f - F*): {result['best_f'] - f_star}")
print(f"evals_used: {result['evals_used']}  iterations: {result['iterations']}")
