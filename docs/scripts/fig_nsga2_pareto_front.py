"""Figure: NSGA-II's found front0 vs. ZDT1's known analytic Pareto front,
for Tutorial 7 (multi-objective optimization with NSGA-II).

Same scenario as `examples/python/nsga2_zdt1.py` (see that file's own
docstring for the full provenance/citation), reproduced here rather than
imported. `sezgi.NSGA2` (a wrapper class, `sezgi.builtins.NSGA2`) delegates
verbatim to `sezgi.mo.nsga2` -- same call, same result dict -- see that
class's own docstring for exactly why this is the ONE builtin wrapper class
whose `.run()` returns a different dict shape (multi-objective, no single
`best_x`/`best_f`) than every other wrapper class's `SolveResult`.
"""
import sezgi
from _figutil import png_path, write_sidecar

NAME = "nsga2_pareto_front"
PROBLEM = "zdt1"
DIM = 10
POP_SIZE = 40
BUDGET = 4000
SEED = 20260830
REF_POINT = [1.1, 1.1]


def compute_data():
    result = sezgi.NSGA2(pop_size=POP_SIZE).run(PROBLEM, DIM, BUDGET, seed=SEED)
    front0 = [result["objectives"][i] for i in result["front0"]]
    reference_front = sezgi.mo.pareto_front(PROBLEM, DIM, 200)

    hv = sezgi.mo.hypervolume_2d(front0, REF_POINT)
    igd_value = sezgi.mo.igd(front0, reference_front)

    front0_sorted = sorted(front0, key=lambda p: p[0])
    ref_sorted = sorted(reference_front, key=lambda p: p[0])
    return {
        "problem": PROBLEM,
        "dim": DIM,
        "pop_size": POP_SIZE,
        "budget": BUDGET,
        "seed": SEED,
        "ref_point": REF_POINT,
        "evals_used": result["evals_used"],
        "front_size": len(front0),
        "hypervolume_2d": hv,
        "igd": igd_value,
        "front0_f1": [p[0] for p in front0_sorted],
        "front0_f2": [p[1] for p in front0_sorted],
        "reference_f1": [p[0] for p in ref_sorted],
        "reference_f2": [p[1] for p in ref_sorted],
    }


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(data["reference_f1"], data["reference_f2"], color="#adb5bd",
             linewidth=2, label="known analytic front (200-pt sample)")
    ax.scatter(data["front0_f1"], data["front0_f2"], color="#e8590c",
                s=18, zorder=3, label=f"NSGA-II front0 ({data['front_size']} pts)")
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_title(
        f"NSGA-II on ZDT1 (dim={data['dim']}, seed={data['seed']})\n"
        f"hypervolume_2d={data['hypervolume_2d']:.4g}  igd={data['igd']:.4g}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path(NAME), dpi=150)
    plt.close(fig)


def main():
    data = compute_data()
    render(data)
    write_sidecar(NAME, data)


if __name__ == "__main__":
    main()
