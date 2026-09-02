"""M4-1 Task 5: built-in algorithm wrapper classes (`sezgi.builtins`) +
`NSGA2` + result unification.

Covers the brief's own pinned test list:
(a) table-driven smoke over ALL 28 preset-backed wrappers (GeneticAlgorithm
    and DifferentialEvolution included -- both fold multiple presets into
    one class, see sezgi/builtins.py's own module doc) -- tiny budget,
    fixed seed, result is a sane SolveResult. NSGA2 is deliberately NOT
    part of this generic table (see its own dedicated smoke test below):
    its run() has a different call convention (a problem NAME string plus
    a `dim` argument, mirroring mo.nsga2 exactly) and returns mo.nsga2's
    own dict shape, not SolveResult -- documented in NSGA2's own docstring,
    not a "cannot run" skip.
(b) GeneticAlgorithm auto-dispatch: one test per space kind (Float,
    Permutation, Binary, Int, Categorical), the Mixed honest error, and the
    representation= override.
(c) Anchored equivalence (load-bearing): wrapper.run(...) ==
    sezgi.solve(sezgi.presets.X(...), ...) BIT-IDENTICALLY (best_f AND
    best_x) at the same seed, for ga_real, de_rand_1, pso, and hho (a
    fourth representative beyond the brief's pinned three) -- proves the
    wrapper adds nothing over the compat internals it calls.
(d) NSGA2 class == mo.nsga2 direct-call equivalence on one seeded run,
    including a WFG-specific anchor (fix round 1) proving k/l flow through
    the wrapper unchanged.
"""
import pytest
import sezgi


# ---------------------------------------------------------------------------
# (a) table-driven smoke over every preset-backed wrapper class.
#
# One shared smoke problem (Sphere, bbob fid=1 dim=3 instance=1) for every
# scalar wrapper -- pop_size=6 satisfies every preset's own min_pop
# requirement in this crate (the strictest is gwo's min_pop=3), budgets
# chosen per class's own population-cost shape: dim_budget presets derive
# their own population from dim=3 (LSHADE: 18*3=54, CMAESIpop: 4+floor(3*ln
# (3))=7, NelderMead: 3+1=4) so their budgets are sized to cover at least
# one full generation; TLBO/ArtificialBeeColony (multi-eval-per-cycle
# presets) get the same budget=60 headroom as every other pop_size=6 class.
_SMOKE_PROBLEM = sezgi.bbob(1, 3, 1)

_SMOKE_TABLE = [
    ("EvolutionStrategy", {"pop_size": 6}, 60),
    ("ParticleSwarm", {"pop_size": 6}, 60),
    ("SimulatedAnnealing", {}, 50),
    ("SHADE", {"pop_size": 6}, 60),
    ("LSHADE", {}, 200),
    ("CMAES", {"pop_size": 6}, 60),
    ("CMAESIpop", {}, 100),
    ("NelderMead", {}, 40),
    ("RandomSearch", {"pop_size": 6}, 60),
    ("GreyWolfOptimizer", {"pop_size": 6}, 60),
    ("WhaleOptimization", {"pop_size": 6}, 60),
    ("HarmonySearch", {"pop_size": 6}, 60),
    ("CuckooSearch", {"pop_size": 6}, 60),
    ("GrasshopperOptimization", {"pop_size": 6}, 60),
    ("SineCosineAlgorithm", {"pop_size": 6}, 60),
    ("JAYA", {"pop_size": 6}, 60),
    ("MothFlameOptimization", {"pop_size": 6}, 60),
    ("SalpSwarm", {"pop_size": 6}, 60),
    ("FireflyAlgorithm", {"pop_size": 6}, 60),
    ("BatAlgorithm", {"pop_size": 6}, 60),
    ("FlowerPollination", {"pop_size": 6}, 60),
    ("TLBO", {"pop_size": 6}, 60),
    ("HarrisHawks", {"pop_size": 6}, 60),
    ("AntLion", {"pop_size": 6}, 60),
    ("ArtificialBeeColony", {"pop_size": 6}, 60),
    ("GravitationalSearch", {"pop_size": 6}, 60),
    ("GeneticAlgorithm", {"pop_size": 6}, 60),
    ("DifferentialEvolution", {"pop_size": 6}, 60),
]

# Every entry in _SMOKE_TABLE names a class this module actually exports --
# a drift guard: if builtins.py's __all__ ever gains/loses a preset-backed
# class without this table being updated, this assertion (not a silent
# skip) catches it at collection time.
_EXPECTED_PRESET_CLASSES = set(sezgi.builtins.__all__) - {"NSGA2"}
assert {row[0] for row in _SMOKE_TABLE} == _EXPECTED_PRESET_CLASSES, (
    "_SMOKE_TABLE is out of sync with sezgi.builtins.__all__ "
    f"(missing: {_EXPECTED_PRESET_CLASSES - {r[0] for r in _SMOKE_TABLE}}, "
    f"extra: {{r[0] for r in _SMOKE_TABLE}} - _EXPECTED_PRESET_CLASSES)")


@pytest.mark.parametrize("name,init_kwargs,budget", _SMOKE_TABLE,
                          ids=[row[0] for row in _SMOKE_TABLE])
def test_wrapper_smoke_runs_and_returns_sane_solve_result(name, init_kwargs, budget):
    cls = getattr(sezgi, name)
    instance = cls(**init_kwargs)
    result = instance.run(_SMOKE_PROBLEM, budget=budget, seed=1)
    assert isinstance(result, sezgi.algo.SolveResult)
    assert 0 < result.evals_used <= budget
    assert result.budget == budget
    assert result.seed == 1
    assert isinstance(result.best_f, float)
    assert isinstance(result.best_x, list) and len(result.best_x) == 3
    assert all(isinstance(v, float) for v in result.best_x)
    # bbob's own known optimum -> f_opt/gap populated.
    assert result.f_opt is not None
    assert result.gap == pytest.approx(result.best_f - result.f_opt)


def test_wrapper_smoke_run_is_seed_deterministic():
    """Same class, same seed -> identical SolveResult (dataclass equality
    over every field) -- a spot check (not exhaustive over every class,
    which would be redundant with the anchored-equivalence tests below,
    themselves proof of determinism)."""
    r1 = sezgi.GreyWolfOptimizer(pop_size=6).run(_SMOKE_PROBLEM, budget=60, seed=3)
    r2 = sezgi.GreyWolfOptimizer(pop_size=6).run(_SMOKE_PROBLEM, budget=60, seed=3)
    assert r1 == r2


# ---------------------------------------------------------------------------
# (b) GeneticAlgorithm auto-dispatch.

class _CustomFloat(sezgi.Problem):
    """A sezgi.Problem subclass (not just a native handle) to prove
    auto-dispatch also works over the M4-1 Task 1 Problem ABC path (not
    just bbob(...))."""

    def space(self):
        return sezgi.Float(-5.0, 5.0, 3)

    def evaluate(self, x):
        return sum(v * v for v in x)


def test_ga_auto_dispatch_float_engages_ga_real():
    ga = sezgi.GeneticAlgorithm(pop_size=6)
    assert ga.dispatched_representation is None
    result = ga.run(_SMOKE_PROBLEM, budget=60, seed=1)
    assert ga.dispatched_representation == "real"
    # Anchored equivalence proof that ga_real (not some other preset) ran:
    # bit-identical to a direct presets.ga_real(...) + solve() call.
    direct_spec = sezgi.presets.ga_real(6, 60)
    direct = sezgi.solve(direct_spec, _SMOKE_PROBLEM, master_seed=1)
    assert result.best_f == direct["best_f"]
    assert result.best_x == direct["best_x"]


def test_ga_auto_dispatch_works_over_problem_subclass_too():
    ga = sezgi.GeneticAlgorithm(pop_size=6)
    result = ga.run(_CustomFloat(), budget=60, seed=1)
    assert ga.dispatched_representation == "real"
    assert isinstance(result.best_f, float)


def test_ga_auto_dispatch_permutation_engages_ga_perm():
    tsp = sezgi.problems.tsp("berlin52")
    ga = sezgi.GeneticAlgorithm(pop_size=6)
    result = ga.run(tsp, budget=60, seed=1)
    assert ga.dispatched_representation == "perm"
    direct_spec = sezgi.presets.ga_perm(6, 60)
    direct = sezgi.solve(direct_spec, tsp, master_seed=1)
    assert result.best_f == direct["best_f"]
    assert result.best_x == direct["best_x"]


def test_ga_auto_dispatch_binary_engages_ga_bin():
    onemax = sezgi.problems.onemax(10)
    ga = sezgi.GeneticAlgorithm(pop_size=6)
    result = ga.run(onemax, budget=60, seed=1)
    assert ga.dispatched_representation == "bin"
    direct_spec = sezgi.presets.ga_bin(6, 60)
    direct = sezgi.solve(direct_spec, onemax, master_seed=1)
    assert result.best_f == direct["best_f"]
    assert result.best_x == direct["best_x"]


def test_ga_auto_dispatch_int_engages_ga_int():
    iq = sezgi.problems.int_quadratic(-5, 5, 3)
    ga = sezgi.GeneticAlgorithm(pop_size=6)
    result = ga.run(iq, budget=60, seed=1)
    assert ga.dispatched_representation == "int"
    direct_spec = sezgi.presets.ga_int(6, 60)
    direct = sezgi.solve(direct_spec, iq, master_seed=1)
    assert result.best_f == direct["best_f"]
    assert result.best_x == direct["best_x"]


def test_ga_auto_dispatch_categorical_engages_ga_cat():
    cm = sezgi.problems.cat_match(4, 3, 1)
    ga = sezgi.GeneticAlgorithm(pop_size=6)
    result = ga.run(cm, budget=60, seed=1)
    assert ga.dispatched_representation == "cat"
    direct_spec = sezgi.presets.ga_cat(6, 60)
    direct = sezgi.solve(direct_spec, cm, master_seed=1)
    assert result.best_f == direct["best_f"]
    assert result.best_x == direct["best_x"]


def test_ga_auto_dispatch_mixed_space_raises_honest_error():
    mixed = sezgi.problems.mixed_diagnostic(2, 2, 3, 2, 2)
    ga = sezgi.GeneticAlgorithm(pop_size=6)
    with pytest.raises(NotImplementedError, match="gen/compound"):
        ga.run(mixed, budget=60, seed=1)
    # No dispatch happened -- attribute stays at its pre-run default.
    assert ga.dispatched_representation is None


def test_ga_representation_override_skips_auto_dispatch():
    """representation= forces a specific preset regardless of the
    problem's own space kind -- here forcing "cat" onto a Float problem,
    which auto-dispatch would never choose on its own, proving the
    override genuinely bypasses block-kind introspection (ga_cat treats
    a Float genotype as if it were Categorical indices, which is nonsense
    numerically but still a valid, deterministic engine run -- the point
    of this test is dispatch control, not landscape sanity)."""
    ga = sezgi.GeneticAlgorithm(pop_size=6, representation="cat")
    assert ga.representation == "cat"
    result = ga.run(sezgi.problems.cat_match(4, 3, 1), budget=60, seed=1)
    assert ga.dispatched_representation == "cat"
    direct_spec = sezgi.presets.ga_cat(6, 60)
    direct = sezgi.solve(direct_spec, sezgi.problems.cat_match(4, 3, 1), master_seed=1)
    assert result.best_f == direct["best_f"]


def test_ga_invalid_representation_rejected():
    with pytest.raises(ValueError, match="representation"):
        sezgi.GeneticAlgorithm(representation="not-a-real-kind")


# ---------------------------------------------------------------------------
# (c) Anchored equivalence -- the load-bearing proof that each wrapper
# adds nothing over sezgi.solve(sezgi.presets.X(...), ...).

_ANCHOR_PROBLEM = sezgi.bbob(2, 5, 1)


def test_anchored_equivalence_ga_real():
    seed = 11
    wrapper = sezgi.GeneticAlgorithm(pop_size=12).run(_ANCHOR_PROBLEM, budget=400, seed=seed)
    direct_spec = sezgi.presets.ga_real(12, 400)
    direct = sezgi.solve(direct_spec, _ANCHOR_PROBLEM, master_seed=seed)
    assert wrapper.best_f == direct["best_f"]
    assert wrapper.best_x == direct["best_x"]
    assert wrapper.evals_used == direct["evals_used"]


def test_anchored_equivalence_de_rand_1():
    seed = 11
    wrapper = sezgi.DifferentialEvolution(pop_size=12, variant="rand1").run(
        _ANCHOR_PROBLEM, budget=400, seed=seed)
    direct_spec = sezgi.presets.de_rand_1(12, 400)
    direct = sezgi.solve(direct_spec, _ANCHOR_PROBLEM, master_seed=seed)
    assert wrapper.best_f == direct["best_f"]
    assert wrapper.best_x == direct["best_x"]
    assert wrapper.evals_used == direct["evals_used"]


def test_anchored_equivalence_de_best_1_and_jde_variants():
    """The non-default DifferentialEvolution variants get the same
    anchored proof (not just rand1)."""
    seed = 11
    for variant, preset_attr in (("best1", "de_best_1"), ("jde", "jde")):
        wrapper = sezgi.DifferentialEvolution(pop_size=12, variant=variant).run(
            _ANCHOR_PROBLEM, budget=400, seed=seed)
        direct_spec = getattr(sezgi.presets, preset_attr)(12, 400)
        direct = sezgi.solve(direct_spec, _ANCHOR_PROBLEM, master_seed=seed)
        assert wrapper.best_f == direct["best_f"], variant
        assert wrapper.best_x == direct["best_x"], variant


def test_anchored_equivalence_pso():
    seed = 11
    wrapper = sezgi.ParticleSwarm(pop_size=12).run(_ANCHOR_PROBLEM, budget=400, seed=seed)
    direct_spec = sezgi.presets.pso(12, 400)
    direct = sezgi.solve(direct_spec, _ANCHOR_PROBLEM, master_seed=seed)
    assert wrapper.best_f == direct["best_f"]
    assert wrapper.best_x == direct["best_x"]
    assert wrapper.evals_used == direct["evals_used"]


def test_anchored_equivalence_hho():
    """Fourth representative beyond the brief's pinned ga_real/de_rand_1/
    pso trio -- HarrisHawks, one of the more structurally complex presets
    (multi-branch escape-energy tree, per presets.rs:622-648's own doc)."""
    seed = 11
    wrapper = sezgi.HarrisHawks(pop_size=12).run(_ANCHOR_PROBLEM, budget=400, seed=seed)
    direct_spec = sezgi.presets.hho(12, 400)
    direct = sezgi.solve(direct_spec, _ANCHOR_PROBLEM, master_seed=seed)
    assert wrapper.best_f == direct["best_f"]
    assert wrapper.best_x == direct["best_x"]
    assert wrapper.evals_used == direct["evals_used"]


def test_anchored_equivalence_dim_budget_preset_lshade():
    """A dim_budget-mode class gets its own anchor too: pop_size is
    derived from the problem's dim (18*dim, see LSHADE's own docstring),
    so the direct comparison call must use the SAME derived value."""
    seed = 11
    dim = _ANCHOR_PROBLEM.dim()
    wrapper = sezgi.LSHADE().run(_ANCHOR_PROBLEM, budget=400, seed=seed)
    direct_spec = sezgi.presets.lshade(dim, 400)
    direct = sezgi.solve(direct_spec, _ANCHOR_PROBLEM, master_seed=seed)
    assert wrapper.best_f == direct["best_f"]
    assert wrapper.best_x == direct["best_x"]


# ---------------------------------------------------------------------------
# (d) NSGA2 class == mo.nsga2 equivalence.

def test_nsga2_smoke_runs():
    n = sezgi.NSGA2(pop_size=8)
    result = n.run("zdt1", 3, 40, seed=1)
    assert "individuals" in result and "objectives" in result
    assert "front0" in result and "evals_used" in result
    assert 0 < result["evals_used"] <= 40


def test_nsga2_class_equals_mo_nsga2_direct_call():
    n = sezgi.NSGA2(pop_size=8, eta_c=15.0, eta_m=15.0, p_c=0.8)
    wrapper_result = n.run("zdt1", 3, 40, seed=5)
    direct_result = sezgi.mo.nsga2(
        "zdt1", 3, 8, 40, seed=5, eta_c=15.0, eta_m=15.0, p_c=0.8)
    assert wrapper_result == direct_result


def test_nsga2_class_equals_mo_nsga2_with_defaults():
    """Same equivalence with every operator kwarg left at its own
    default, to prove the class's own __init__ defaults match mo.nsga2's
    own defaults exactly."""
    n = sezgi.NSGA2(pop_size=8)
    wrapper_result = n.run("dtlz2", 5, 40, m=3, seed=2)
    direct_result = sezgi.mo.nsga2("dtlz2", 5, 8, 40, m=3, seed=2)
    assert wrapper_result == direct_result


def test_nsga2_class_equals_mo_nsga2_on_wfg_with_explicit_k_l():
    """WFG-specific anchor (fix round 1, coordinator review): zdt1/dtlz2
    above never exercise k/l (both REJECTED for zdt/dtlz per mo.nsga2's
    own doc -- only wfg1-9 accept them, dim itself must be None, derived
    from k+l). Uses non-default k/l (6, 10) rather than the toolkit's own
    recommended defaults (k=4 for m=2, l=20) so this test cannot pass by
    coincidentally ignoring them."""
    n = sezgi.NSGA2(pop_size=8)
    wrapper_result = n.run("wfg1", None, 40, m=2, seed=3, k=6, l=10)
    direct_result = sezgi.mo.nsga2("wfg1", None, 8, 40, m=2, seed=3, k=6, l=10)
    assert wrapper_result == direct_result


def test_nsga2_wfg_k_l_actually_change_the_result():
    """Proves k/l genuinely flow through the wrapper into the run (not
    silently dropped): the SAME seed with different k/l must NOT produce
    the same objectives."""
    n = sezgi.NSGA2(pop_size=8)
    result_a = n.run("wfg1", None, 40, m=2, seed=3, k=6, l=10)
    result_b = n.run("wfg1", None, 40, m=2, seed=3, k=4, l=20)
    assert result_a["objectives"] != result_b["objectives"]
