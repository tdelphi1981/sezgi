"""Figure: a single seeded run's convergence curve (best-f vs. evaluations
used), for the Quickstart page.

Mechanism: sezgi runs are deterministic under a fixed `(problem, budget,
seed)` (see `docs/site/concepts/rng-and-determinism.md`), so truncating the
SAME seeded run at successively larger budgets reproduces exactly what that
one run's own trajectory looked like at each point -- an anytime view built
entirely from real `.run()` calls, no internal engine hooks needed. This is
the same truncation trick `docs/site/learn/06-reading-convergence-curves.md`
narrates in prose; this script is its figure-producing twin, plus a second,
denser problem/algorithm pair for the Quickstart page itself.
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
