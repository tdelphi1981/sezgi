"""Figure: recovered feature mask vs. the dataset's true informative
columns, for Tutorial 5 (feature selection recipe).

Same dataset/config as `examples/python/oop/feature_selection.py`
(reproduced here rather than imported, since that file lives outside the
package `sys.path` normally reaches -- see that file's own docstring for
the full generation story): a fixed 20x8 design matrix with exactly 3
informative columns (indices 1, 3, 6), searched via
`sezgi.recipes.FeatureSelection` + `sezgi.GeneticAlgorithm`'s Binary
auto-dispatch.
"""
import numpy as np

import sezgi
from sezgi.recipes import FeatureSelection
from _figutil import png_path, write_sidecar

NAME = "feature_mask_recovery"
SEED = 98
N_SAMPLES, N_FEATURES = 20, 8
INFORMATIVE = (1, 3, 6)
COEFS = np.array([3.0, -2.0, 1.5])
NOISE_STD = 0.2
PENALTY = 0.3
POP_SIZE = 20
BUDGET = 200
GA_SEED = 42


def _make_dataset():
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))
    noise = rng.normal(scale=NOISE_STD, size=N_SAMPLES)
    y = X[:, INFORMATIVE] @ COEFS + noise
    return X, y


def _scorer(X_sub, y):
    design = np.column_stack([np.ones(X_sub.shape[0]), X_sub])
    coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coefs
    return float(np.sum(resid ** 2))


def compute_data():
    X, y = _make_dataset()
    problem = FeatureSelection(X, y, _scorer, penalty=PENALTY)
    result = sezgi.GeneticAlgorithm(pop_size=POP_SIZE).run(
        problem, budget=BUDGET, seed=GA_SEED)

    mask = [bool(v) for v in result.best_x]
    informative_mask = [i in INFORMATIVE for i in range(N_FEATURES)]
    return {
        "n_features": N_FEATURES,
        "informative_indices": list(INFORMATIVE),
        "penalty": PENALTY,
        "budget": BUDGET,
        "seed": GA_SEED,
        "best_f": result.best_f,
        "evals_used": result.evals_used,
        "mask": mask,
        "informative_mask": informative_mask,
        "recovered": mask == informative_mask,
    }


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = data["n_features"]
    fig, ax = plt.subplots(figsize=(6, 2.5))
    for i in range(n):
        selected = data["mask"][i]
        informative = data["informative_mask"][i]
        if selected and informative:
            color = "#2f9e44"  # correctly selected
        elif selected and not informative:
            color = "#e03131"  # false positive
        elif not selected and informative:
            color = "#f08c00"  # false negative (missed)
        else:
            color = "#ced4da"  # correctly excluded
        ax.bar(i, 1.0, color=color, edgecolor="white")
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"x{i}" for i in range(n)])
    ax.set_yticks([])
    ax.set_title(
        f"FeatureSelection recovered mask (recovered={data['recovered']})")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#2f9e44", label="selected + informative"),
        plt.Rectangle((0, 0), 1, 1, color="#ced4da", label="excluded + noise"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path(NAME), dpi=150)
    plt.close(fig)


def main():
    data = compute_data()
    render(data)
    write_sidecar(NAME, data)


if __name__ == "__main__":
    main()
