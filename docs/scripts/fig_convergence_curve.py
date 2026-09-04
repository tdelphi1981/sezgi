"""Figure: a single seeded run's convergence curve (best-f vs. evaluations
used), for the Quickstart page.

Mechanism: 12 INDEPENDENT `.run()` calls, one per budget in BUDGETS below,
same `(problem, algorithm, seed)` every time -- not a single logged run
truncated/replayed after the fact (that is a DIFFERENT technique,
`read_ioh_records`'s curtailed-view reconstruction, narrated in
`docs/site/learn/06-reading-convergence-curves.md` and drawn by
`fig_budget_anytime_curve.py`). `GeneticAlgorithm`'s generator
(`crates/components/src/ga.rs`'s `GaRealGenerator::generate`) never reads
the total budget while it runs -- unlike a budget-adaptive preset such as
`lshade`, whose population-shrinking schedule reads `termination.budget`
to plan its FULL run --
so an independent run at a smaller budget lands on the same point a
longer run with the same seed would have reached, WHENEVER the two runs
complete the same number of generations. That is true at 11 of this
script's own 12 BUDGETS; `budget=450` is the one exception (the engine
only evaluates whole generation batches, so a `budget=450` request stops
at `evals_used=440`, one generation short of where a longer run's own
trajectory actually stood at evaluation #450) -- confirmed by comparing
this script's own independent-run output against a curtailed-view replay
of a single logged budget=2000 run at the same seed. The resulting curve
is still a faithful, real convergence curve (every plotted point is a
genuine `.run()` result); only the "exact reproduction at every budget"
claim would be false, and is not made here.
"""
import sezgi
from _figutil import png_path, write_sidecar

NAME = "convergence_curve"
SEED = 1
PROBLEM_ARGS = (1, 5, 1)  # bbob(fid=1 Sphere, dim=5, instance=1)
POP_SIZE = 20
BUDGETS = [20, 40, 80, 120, 200, 300, 450, 650, 900, 1200, 1600, 2000]


def compute_data():
    problem = sezgi.bbob(*PROBLEM_ARGS)
    best_f = []
    gap = []
    for budget in BUDGETS:
        result = sezgi.GeneticAlgorithm(pop_size=POP_SIZE).run(
            problem, budget=budget, seed=SEED)
        best_f.append(result.best_f)
        gap.append(result.gap)
    return {
        "algo": "GeneticAlgorithm",
        "problem": f"bbob{PROBLEM_ARGS}",
        "seed": SEED,
        "pop_size": POP_SIZE,
        "budgets": BUDGETS,
        "best_f": best_f,
        "gap": gap,
    }


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(data["budgets"], data["gap"], marker="o", color="#3b5bdb")
    ax.set_yscale("log")
    ax.set_xlabel("evaluations used (budget)")
    ax.set_ylabel("gap = best_f - f_opt  (log scale)")
    ax.set_title("GeneticAlgorithm on bbob(1, 5, 1), seed=1")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path(NAME), dpi=150)
    plt.close(fig)


def main():
    data = compute_data()
    render(data)
    write_sidecar(NAME, data)


if __name__ == "__main__":
    main()
