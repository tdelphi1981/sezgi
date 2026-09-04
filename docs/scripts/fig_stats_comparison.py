"""Figure: a Friedman mean-rank summary across three algorithms and five
BBOB instances, for Tutorial 8 (comparing algorithms + statistics).

Mechanism: `sezgi.run_experiment` runs a small `ExperimentSpec` (TOML) --
3 algorithms x 5 BBOB f1 instances x 5 seeds, one budget -- and
`sezgi.results_matrix` aggregates the per-seed gaps into one
`sezgi.stats.friedman`-shaped matrix (rows = problems, columns =
algorithms). `mean_ranks` (lower = better, per-problem ranks averaged
across problems) is the plotted summary -- exactly the same statistic
`sezgi.stats.paper_package`'s own `friedman` sub-result reports, at a
scale small enough to build in well under a second.
"""
import sezgi
from _figutil import png_path, write_sidecar

NAME = "stats_comparison"
SEEDS = [1, 2, 3, 4, 5]
BUDGET = 500
INSTANCES = [1, 2, 3, 4, 5]

SPEC_TOML = f"""
name = "docs-fig-stats-comparison"
seeds = {SEEDS}
budgets = [{BUDGET}]

[[algorithms]]
name = "de_rand_1"
preset = {{ kind = "de_rand_1", pop_size = 20 }}

[[algorithms]]
name = "cmaes"
preset = {{ kind = "cmaes", pop_size = 20 }}

[[algorithms]]
name = "random_search"
preset = {{ kind = "random_search", pop_size = 20 }}

[[problems]]
suite = "bbob"
fid = 1
dim = 5
instances = {INSTANCES}
"""


def compute_data():
    records = sezgi.run_experiment(SPEC_TOML, parallel=True)
    algo_names, problem_labels, matrix = sezgi.results_matrix(records, BUDGET)
    friedman = sezgi.stats.friedman(matrix)
    return {
        "budget": BUDGET,
        "seeds": SEEDS,
        "instances": INSTANCES,
        "n_records": len(records),
        "algo_names": algo_names,
        "problem_labels": problem_labels,
        "gap_matrix": matrix,
        "friedman_statistic": friedman["statistic"],
        "friedman_p_value": friedman["p_value"],
        "mean_ranks": friedman["mean_ranks"],
    }


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = data["algo_names"]
    ranks = data["mean_ranks"]
    order = sorted(range(len(names)), key=lambda i: ranks[i])

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2f9e44", "#1971c2", "#e8590c"]
    ax.bar([names[i] for i in order], [ranks[i] for i in order],
           color=[colors[i % len(colors)] for i in range(len(order))])
    ax.set_ylabel("mean rank across 5 BBOB f1 instances (lower = better)")
    ax.set_title(
        f"Friedman test: statistic={data['friedman_statistic']:.3g}  "
        f"p={data['friedman_p_value']:.3g}\n"
        f"({len(data['seeds'])} seeds x {len(data['instances'])} instances, "
        f"budget={data['budget']})")
    fig.tight_layout()
    fig.savefig(png_path(NAME), dpi=150)
    plt.close(fig)


def main():
    data = compute_data()
    render(data)
    write_sidecar(NAME, data)


if __name__ == "__main__":
    main()
