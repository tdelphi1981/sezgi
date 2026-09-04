"""Figure: the budget/anytime curve, for
`docs/site/learn/06-reading-convergence-curves.md`.

Exactly the scenario that page's own prose table walks through by hand
(`sezgi.DifferentialEvolution(pop_size=20)` on `bbob(1, 5, 1)`, seed=1,
budgets 100/300/900/2700 -- the four numbers quoted in that page's text),
densified into enough budget points to draw a real curve, plotted BOTH on
a linear scale (absolute best_f) and a log-gap scale, side by side -- the
same "two honest ways to view the same run" distinction that page's own
text draws (a third, ECDF, needs many runs/an on-disk log and is described
in prose only, not plotted here).
"""
import sezgi
from _figutil import png_path, write_sidecar

NAME = "budget_anytime_curve"
PROBLEM_ARGS = (1, 5, 1)
SEED = 1
POP_SIZE = 20
BUDGETS = [50, 100, 150, 200, 300, 450, 650, 900, 1300, 1800, 2300, 2700, 3200]


def compute_data():
    problem = sezgi.bbob(*PROBLEM_ARGS)
    best_f = []
    gap = []
    for budget in BUDGETS:
        result = sezgi.DifferentialEvolution(pop_size=POP_SIZE).run(
            problem, budget=budget, seed=SEED)
        best_f.append(result.best_f)
        gap.append(result.gap)
    return {
        "algo": "DifferentialEvolution",
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(data["budgets"], data["best_f"], marker="o", color="#3b5bdb")
    ax1.set_xlabel("evaluations used (budget)")
    ax1.set_ylabel("best_f (linear scale)")
    ax1.set_title("Raw best-so-far")
    ax1.grid(True, alpha=0.3)

    ax2.plot(data["budgets"], data["gap"], marker="o", color="#e8590c")
    ax2.set_yscale("log")
    ax2.set_xlabel("evaluations used (budget)")
    ax2.set_ylabel("gap = best_f - f_opt (log scale)")
    ax2.set_title("Log-gap")
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle("DifferentialEvolution on bbob(1, 5, 1), seed=1 -- same run, two views")
    fig.tight_layout()
    fig.savefig(png_path(NAME), dpi=150)
    plt.close(fig)


def main():
    data = compute_data()
    render(data)
    write_sidecar(NAME, data)


if __name__ == "__main__":
    main()
