"""0.1.3 compound generalization, Task 3: mixed-space hybrid auto-dispatch
for compound-eligible presets (probe protocol R-G).

Mixed space: Float(4, [-5,5]) + Binary(6) + Int(3, [0,9]); objective =
sphere(float) + onemax-complement(binary) + sum |int - 5|. Per eligible
preset: (1) seed-determinism (two runs, seed=7, budget=4000 -> identical
best_f and best_x); (2) strictly better than random-search at the same
budget/seed. Ineligible-preset error tests live in test_oop_builtins.py.
"""
import pytest
import sezgi

SEED = 7
BUDGET = 4000


class MixedProbe(sezgi.Problem):
    def space(self):
        return sezgi.Space(
            sezgi.Float(-5.0, 5.0, 4),
            sezgi.Binary(6),
            sezgi.Int(0, 9, 3),
        )

    def evaluate(self, x):
        f, b, i = x
        return (sum(v * v for v in f)
                + sum(1 for v in b if not v)
                + sum(abs(v - 5) for v in i))


ELIGIBLE = [
    sezgi.DifferentialEvolution,
    lambda **kw: sezgi.DifferentialEvolution(variant="best1", **kw),
    sezgi.GreyWolfOptimizer,
    sezgi.WhaleOptimization,
    sezgi.SineCosineAlgorithm,
    sezgi.JAYA,
    sezgi.GrasshopperOptimization,
    sezgi.SalpSwarm,
    sezgi.FireflyAlgorithm,
    sezgi.FlowerPollination,
    sezgi.TLBO,
    sezgi.CuckooSearch,
    sezgi.AntLion,
    sezgi.EvolutionStrategy,
    sezgi.GravitationalSearch,
    sezgi.BatAlgorithm,
]
IDS = ["de-rand1", "de-best1", "gwo", "woa", "sca", "jaya", "goa", "ssa",
       "fa", "fpa", "tlbo", "cuckoo", "alo", "es", "gsa", "bat"]


@pytest.fixture(scope="module")
def baseline():
    return sezgi.RandomSearch().run(MixedProbe(), budget=BUDGET, seed=SEED)


@pytest.mark.parametrize("make", ELIGIBLE, ids=IDS)
def test_hybrid_deterministic_and_beats_random_search(make, baseline):
    a = make().run(MixedProbe(), budget=BUDGET, seed=SEED)
    b = make().run(MixedProbe(), budget=BUDGET, seed=SEED)
    assert a.best_f == b.best_f
    assert a.best_x == b.best_x
    assert a.best_f < baseline.best_f


@pytest.mark.parametrize("make", ELIGIBLE, ids=IDS)
def test_hybrid_keeps_block_structure(make):
    r = make().run(MixedProbe(), budget=400, seed=SEED)
    f, b, i = r.best_x
    assert len(f) == 4 and len(b) == 6 and len(i) == 3
    assert all(0 <= v <= 9 for v in i)


def test_docstrings_state_hybrid():
    for cls in (sezgi.GreyWolfOptimizer, sezgi.TLBO, sezgi.CuckooSearch,
                sezgi.DifferentialEvolution, sezgi.AntLion,
                sezgi.EvolutionStrategy, sezgi.GravitationalSearch,
                sezgi.BatAlgorithm):
        assert "HYBRID" in cls.__doc__
        assert "gen/ga-bin" in cls.__doc__


def test_docstrings_warn_no_float_block_means_pure_ga():
    for cls in (sezgi.GreyWolfOptimizer, sezgi.TLBO, sezgi.CuckooSearch,
                sezgi.DifferentialEvolution, sezgi.AntLion,
                sezgi.EvolutionStrategy, sezgi.GravitationalSearch,
                sezgi.BatAlgorithm):
        assert "NO float block" in cls.__doc__


class _Discrete(sezgi.Problem):
    def space(self):
        return sezgi.Space(sezgi.Binary(10))

    def evaluate(self, x):
        return float(sum(1 for v in x if not v))


class _TwoFloat(sezgi.Problem):
    def space(self):
        return sezgi.Space(sezgi.Float(-5.0, 5.0, 3), sezgi.Float(-5.0, 5.0, 2))

    def evaluate(self, x):
        return sum(v * v for blk in x for v in blk)


def test_pure_discrete_space_runs_deterministically():
    a = sezgi.GreyWolfOptimizer().run(_Discrete(), budget=600, seed=SEED)
    b = sezgi.GreyWolfOptimizer().run(_Discrete(), budget=600, seed=SEED)
    assert a.best_f == b.best_f and a.best_x == b.best_x
    assert len(a.best_x) == 10


def test_multi_float_only_space_runs_deterministically():
    a = sezgi.DifferentialEvolution().run(_TwoFloat(), budget=1000, seed=SEED)
    b = sezgi.DifferentialEvolution().run(_TwoFloat(), budget=1000, seed=SEED)
    assert a.best_f == b.best_f and a.best_x == b.best_x
    assert len(a.best_x[0]) == 3 and len(a.best_x[1]) == 2
