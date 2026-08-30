"""NSGA-II (Deb, Pratap, Agarwal & Meyarivan 2002) on ZDT1 (Zitzler, Deb &
Thiele 2000) -- a matched Python/R example pair for sezgi's M3-2
multi-objective baseline. See examples/r/nsga2_zdt1.R for the R
counterpart: same scenario, same Rust core underneath (NSGA-II is a
self-contained, seeded Rust runner in BOTH bindings, unlike the
pure-Python/pure-R algorithm scripts elsewhere in this directory), so this
pair's numbers are bit-identical across languages -- only the report
formatting differs.

Calls `sezgi.mo.nsga2()` directly rather than going through an ask/tell
`EvalSession` or a component spec: NSGA-II ships as a self-contained,
seeded reference runner over the parallel `MoProblem`/`MoEvaluator`
surface, not as a composable component graph -- MO component-graph specs
(the ExperimentSpec/spec-JSON path the rest of this catalog uses) are
deferred to v2 (see README.md's "Multi-objective optimization (M3-2)"
section and docs/DECISIONS.md's M3-2 record for the full spec-tension
ruling). Accordingly, there is no `specs/nsga2_zdt1.toml` in this catalog.

Small budget, seeded, for a fast illustrative run -- no cross-algorithm or
cross-run quality claims are made here (single seed, single problem).
`pop_size` must be a multiple of 4 (KanGAL-faithful tightening of the
naive "even, >= 4" rule; see crates/components/src/nsga2.rs).
"""
import sezgi

PROBLEM = "zdt1"
DIM = 10
POP_SIZE = 40
BUDGET = 4000
SEED = 20260830
REF_POINT = [1.1, 1.1]  # ZDT1 objectives lie in [0,1]x[0,1]; 1.1 dominates the whole front

result = sezgi.mo.nsga2(PROBLEM, dim=DIM, pop_size=POP_SIZE, budget=BUDGET, seed=SEED)
front0_objectives = [result["objectives"][i] for i in result["front0"]]
reference_front = sezgi.mo.pareto_front(PROBLEM, dim=DIM, n=200)

hv = sezgi.mo.hypervolume_2d(front0_objectives, REF_POINT)
igd_value = sezgi.mo.igd(front0_objectives, reference_front)

print(f"problem: {PROBLEM}  dim={DIM}  pop_size={POP_SIZE}  budget={BUDGET}  seed={SEED}")
print(f"front size: {len(front0_objectives)}")
print(f"evals_used: {result['evals_used']}")
print(f"hypervolume_2d (ref_point={REF_POINT}): {hv}")
print(f"igd (vs pareto_front(200)): {igd_value}")

print("front points (f1, f2), sorted by f1, first 5 (plot-ready -- x=f1, y=f2):")
for f1, f2 in sorted(front0_objectives, key=lambda p: p[0])[:5]:
    print(f"  {f1:.6f}  {f2:.6f}")
