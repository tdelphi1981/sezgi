"""Figure: three block kinds' own sampled points, side by side, for
`docs/site/learn/04-encodings-and-search-spaces.md`.

Mechanism: a tiny `sezgi.Algorithm` subclass (`_CapturePopulation`) that
captures `pop.individuals` on its FIRST `generate()` call (i.e. the
population the engine's own `init/uniform` initializer just produced --
Rust-side, one call per block kind here) and then returns it unchanged as
this generation's offspring, so the run finishes normally. This is the
REAL engine-sampled initial population for each space -- not a
reimplementation of the sampling math -- captured through the same
`Algorithm.generate(pop, ctx)` hook Tutorial 3 teaches, at
`budget = pop_size + 1` so exactly one `generate()` call happens after
initialization.

Three block kinds, three panels: `Float(-5, 5, 2)` (a continuous 2D box,
scatter), `Categorical(4, 1)` (a discrete choice among 4 options, sampled
100 times as a count histogram), `Permutation(8)` (one sampled ordering of
8 positions, shown as a heatmap row -- the encoding a TSP tour or an
ordering problem uses).
"""
import sezgi
from _figutil import png_path, write_sidecar

NAME = "encoding_space"


class _CapturePopulation(sezgi.Algorithm):
    def __init__(self):
        self.captured = None

    def generate(self, pop, ctx):
        if self.captured is None:
            self.captured = [list(x) for x in pop.individuals]
        return pop.individuals


class _FloatBox(sezgi.Problem):
    def space(self):
        return sezgi.Float(-5.0, 5.0, 2)

    def evaluate(self, x):
        return sum(v * v for v in x)


class _CatChoice(sezgi.Problem):
    def space(self):
        return sezgi.Categorical(4, 1)

    def evaluate(self, x):
        return 0.0


class _PermOrder(sezgi.Problem):
    def space(self):
        return sezgi.Permutation(8)

    def evaluate(self, x):
        return 0.0


def compute_data():
    float_cap = _CapturePopulation()
    float_cap.run(_FloatBox(), budget=41, seed=3, pop_size=40)

    cat_cap = _CapturePopulation()
    cat_cap.run(_CatChoice(), budget=101, seed=5, pop_size=100)
    counts = [0, 0, 0, 0]
    for (v,) in cat_cap.captured:
        counts[v] += 1

    perm_cap = _CapturePopulation()
    perm_cap.run(_PermOrder(), budget=2, seed=5, pop_size=1)

    return {
        "float_space": {"lo": -5.0, "hi": 5.0, "n": 2, "seed": 3,
                         "points": float_cap.captured},
        "categorical_space": {"k": 4, "n_samples": 100, "seed": 5,
                               "counts": counts},
        "permutation_space": {"n": 8, "seed": 5,
                               "tour": perm_cap.captured[0]},
    }


def render(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    fpts = data["float_space"]["points"]
    xs = [p[0] for p in fpts]
    ys = [p[1] for p in fpts]
    axes[0].scatter(xs, ys, color="#3b5bdb", alpha=0.7)
    axes[0].set_title("Float(-5, 5, 2)\ncontinuous 2D box")
    axes[0].set_xlim(-5.5, 5.5)
    axes[0].set_ylim(-5.5, 5.5)
    axes[0].set_aspect("equal")

    counts = data["categorical_space"]["counts"]
    axes[1].bar(range(len(counts)), counts, color="#2f9e44")
    axes[1].set_title("Categorical(4, 1)\n100 samples")
    axes[1].set_xticks(range(len(counts)))
    axes[1].set_xlabel("category index")
    axes[1].set_ylabel("count")

    tour = data["permutation_space"]["tour"]
    axes[2].imshow([tour], cmap="viridis", aspect="auto")
    axes[2].set_title("Permutation(8)\none sampled ordering")
    axes[2].set_yticks([])
    axes[2].set_xticks(range(len(tour)))
    for i, v in enumerate(tour):
        axes[2].text(i, 0, str(v), ha="center", va="center", color="white")

    fig.tight_layout()
    fig.savefig(png_path(NAME), dpi=150)
    plt.close(fig)


def main():
    data = compute_data()
    render(data)
    write_sidecar(NAME, data)


if __name__ == "__main__":
    main()
