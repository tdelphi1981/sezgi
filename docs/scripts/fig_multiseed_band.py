"""Figure: multi-seed convergence bands for two algorithms on the same
problem, for `docs/site/learn/01-what-are-metaheuristics.md`.

Same budget-truncation mechanism as `fig_convergence_curve.py`, repeated
over several seeds per algorithm so the plotted line is a MEAN with a
min/max band, not one lucky (or unlucky) run -- the point of this page's
own text: "why gradient methods do not apply" needs no proof, but "does
this algorithm actually do better here" needs more than one seed, which is
exactly what Learn page 5 later makes explicit with a real statistical
test. This figure is intentionally the FIRST place in the site a reader
sees run-to-run spread drawn, before any test is introduced.
"""
import sezgi
from _figutil import png_path, write_sidecar

NAME = "multiseed_band"
PROBLEM_ARGS = (1, 5, 1)  # bbob(fid=1 Sphere, dim=5, instance=1)
POP_SIZE = 20
BUDGETS = [50, 100, 200, 400, 700, 1000]
SEEDS = list(range(8))

ALGOS = {
    "random_search": sezgi.RandomSearch,
    "diff_evolution": sezgi.DifferentialEvolution,
}


def compute_data():
    problem = sezgi.bbob(*PROBLEM_ARGS)
    out = {"problem": f"bbob{PROBLEM_ARGS}", "budgets": BUDGETS, "seeds": SEEDS}
    for label, cls in ALGOS.items():
        per_seed = []
        for seed in SEEDS:
            row = [cls(pop_size=POP_SIZE).run(problem, budget=b, seed=seed).gap
                   for b in BUDGETS]
            per_seed.append(row)
        mean = [sum(col) / len(col) for col in zip(*per_seed)]
        lo = [min(col) for col in zip(*per_seed)]
        hi = [max(col) for col in zip(*per_seed)]
        out[label] = {"per_seed_gap": per_seed, "mean_gap": mean,
                       "min_gap": lo, "max_gap": hi}
    return out


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"random_search": "#e8590c", "diff_evolution": "#2f9e44"}
    for label in ALGOS:
        row = data[label]
        ax.plot(data["budgets"], row["mean_gap"], marker="o",
                color=colors[label], label=label)
        ax.fill_between(data["budgets"], row["min_gap"], row["max_gap"],
                         color=colors[label], alpha=0.15)
    ax.set_yscale("log")
    ax.set_xlabel("evaluations used (budget)")
    ax.set_ylabel("gap (log scale) -- mean over 8 seeds, band = min/max")
    ax.set_title("RandomSearch vs DifferentialEvolution on bbob(1, 5, 1)")
    ax.legend()
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
