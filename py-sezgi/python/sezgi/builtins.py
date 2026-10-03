"""Built-in algorithm wrapper classes (M4-1 Task 5) -- one class per preset
builder in `crates/components/src/presets.rs` (34 `pub fn` builders,
verified against `sezgi.presets`'s own `SimpleNamespace`, which exposes all
34 -- see this task's report for the full reconciliation table: every
preset has exactly one class, no orphan on either side), plus `NSGA2` (a
thin `sezgi.mo.nsga2` skin, NOT one of the 34 presets -- see its own
docstring).

TABLE-DRIVEN (ruling: "no hand-divergence"): `_PRESET_TABLE` below is the
single source of truth; `_make_preset_class` generates every uniformly
-shaped wrapper class from it. `GeneticAlgorithm` (auto-dispatch over 5
presets) and `DifferentialEvolution` (variant= over 3 presets) are
hand-written because their dispatch logic genuinely differs from the
uniform "one class, one preset" shape -- but both funnel through the SAME
`_run_spec` helper every generated class uses, so the actual
spec-building-then-`solve()`-then-wrap-in-`SolveResult` machinery has
exactly one call site in this file, not N. The wrap-in-`SolveResult` step
itself (`_wrap_result`) is a single shared helper in `sezgi.algo`
(final-review fix 3), imported here rather than reimplemented -- the SAME
helper `sezgi.algorithm.Algorithm.run` uses, so that translation has
exactly one implementation across the whole crate, not two.

Every wrapper (GeneticAlgorithm/DifferentialEvolution/NSGA2 included, with
NSGA2's own documented exception): `__init__(pop_size=<preset's own
documented/conventional default>, **preset_kwargs)`,
`run(problem, budget, seed=0, run_id=0, log_dir=None) ->
sezgi.algo.SolveResult`. `run()` builds the preset's `AlgorithmSpec` via
`sezgi.presets.<name>(...)` (UNCHANGED) and calls `sezgi.solve(...)`
(UNCHANGED) internally -- literally ruling 1's "solve()/presets.* stay as
compat internals": these wrapper classes add a constructor + a result-shape
translation, nothing else. Each class's docstring names its preset function
and Rust source (`presets.rs:<lines>`), which carries the algorithm's own
academic citation (unchanged, in the Rust doc comment).
"""
import sezgi
from sezgi.algo import _wrap_result
from sezgi.problem import as_native_problem


def _run_spec(class_name, spec, native, budget, seed, run_id, log_dir):
    """Shared `spec -> sezgi.solve() -> SolveResult` tail, used by every
    wrapper class in this module (generated and hand-written alike) -- the
    ONE place `sezgi.solve` is actually called from this file."""
    result = sezgi.solve(spec, native, master_seed=seed, run_id=run_id,
                          log_dir=log_dir, algo_name=class_name.lower())
    return _wrap_result(class_name.lower(), seed, budget, result, native.optimum())


# ---------------------------------------------------------------------------
# Mixed-space hybrid auto-dispatch (0.1.3 compound generalization).
#
# gen/compound (crates/components/src/compound.rs) accepts any ELIGIBLE
# registered sub-generator per block: stateless (empty requires/provides),
# pop-to-pop (OffspringCount::PopLen), no internal evaluation, no nesting.
# A preset therefore auto-dispatches on a mixed (or non-float) space iff
# every stage's generator is eligible AND every stage's replacer/adapter is
# SupportedBlocks::All and stateless (verified per preset against
# crates/components/src/presets.rs; citations in the 0.1.3 task report).
#
# HYBRID semantics: float blocks keep the preset's OWN float generator
# (same kind and params the preset uses single-block); every non-float
# block gets the fused GA variation default (gen/ga-bin, gen/ga-int,
# gen/ga-cat, gen/ga-perm, registered default params). Parent selection
# and replacement still follow the host preset's pipeline.

_HYBRID_DISCRETE_GEN = {
    "binary": "gen/ga-bin",
    "int": "gen/ga-int",
    "categorical": "gen/ga-cat",
    "permutation": "gen/ga-perm",
}

# preset_attr -> short label used in docstrings.
_HYBRID_PRESETS = {
    "de_rand_1": "DE", "de_best_1": "DE", "gwo": "GWO", "woa": "WOA",
    "sca": "SCA", "jaya": "JAYA", "goa": "GOA", "ssa": "SSA",
    "firefly": "FA", "fpa": "FPA", "tlbo": "TLBO",
    "cuckoo_search": "Cuckoo Search",
    "alo": "ALO", "es_mu_plus_lambda": "ES",
    "gsa": "GSA", "bat": "BA",
}

_WHY_STATEFUL = (
    "it keeps engine-level blackboard state ({what}) and needs its paired "
    "replacer/adapter, which gen/compound cannot route per block")

# preset_attr -> reason the preset has no mixed-space auto-dispatch.
_HYBRID_INELIGIBLE = {
    "harmony_search": (
        "gen/hs emits ONE offspring per generation (OffspringCount::One), "
        "but gen/compound requires every sub-generator to produce "
        "pop.len() offspring"),
    "nelder_mead": (
        "gen/nelder-mead emits ONE offspring per generation "
        "(OffspringCount::One) and is stateful: " + _WHY_STATEFUL.format(
            what="the simplex")),
    "hho": (
        "gen/hho evaluates single-block genotypes against the full "
        "problem inside generate() (internal evaluation), which is "
        "unsound on a block slice"),
    "sa": (
        "SimulatedAnnealing has a fixed population of 1 (single "
        "trajectory), which cannot host the GA discrete sub-generators "
        "(gen/compound requires population size >= 2)"),
    "pso": "gen/pso is stateful: " + _WHY_STATEFUL.format(what="velocity/pbest"),
    "cmaes": "gen/cma is stateful: " + _WHY_STATEFUL.format(what="CMA distribution"),
    "cmaes_ipop": "gen/cma is stateful: " + _WHY_STATEFUL.format(what="CMA distribution"),
    "shade": "gen/de-shade is stateful: " + _WHY_STATEFUL.format(what="success history"),
    "lshade": "gen/de-shade is stateful: " + _WHY_STATEFUL.format(what="success history"),
    "mfo": "gen/mfo is stateful: " + _WHY_STATEFUL.format(what="flame memory"),
    "abc": "gen/abc-employed is stateful: " + _WHY_STATEFUL.format(what="abc/trials"),
}

_HYBRID_DOC = (
    "\n\nMixed spaces: auto-dispatches to a HYBRID via gen/compound -- "
    "{label} on continuous (float) blocks, GA variation (gen/ga-bin / "
    "gen/ga-int / gen/ga-cat / gen/ga-perm, default params) on "
    "binary/int/categorical/permutation blocks; parent selection and "
    "replacement follow the {label} pipeline. A space with NO float block "
    "gets zero {label}-specific variation (all variation is GA).")


def _block_kinds(native):
    return [b["kind"] for b in native.blocks()]


def _needs_compound(kinds):
    return len(kinds) > 1 or any(k != "float" for k in kinds)


def _check_hybrid_ineligible(class_name, preset_attr, native):
    """Raises NotImplementedError naming WHY when an ineligible preset is
    asked to run on a mixed/non-float space (no-op for float-only spaces)."""
    reason = _HYBRID_INELIGIBLE.get(preset_attr)
    if reason is None:
        return
    kinds = _block_kinds(native)
    if _needs_compound(kinds):
        raise NotImplementedError(
            f"{class_name} cannot run on a mixed or non-float space "
            f"(block kinds {kinds}): {reason}. Only compound-eligible "
            "presets auto-dispatch (sezgi.presets: de, gwo, woa, sca, "
            "jaya, goa, ssa, firefly, fpa, tlbo, cuckoo_search, alo, "
            "es_mu_plus_lambda; see "
            "gen/compound in crates/components/src/compound.rs).")


def _compound_wrap_stages(spec, kinds, hybrid):
    """Wraps EVERY stage's generator in its own gen/compound entry with one
    sub-generator per block. hybrid=False copies the stage's generator onto
    every block (GeneticAlgorithm's same-kind multi-block wrap); hybrid=True
    keeps it only on float blocks and uses the fused GA default
    (_HYBRID_DISCRETE_GEN) on every other block."""
    for stage in spec["stages"]:
        gen = stage["generator"]
        blocks = []
        for k in kinds:
            if hybrid and k != "float":
                blocks.append({"kind": _HYBRID_DISCRETE_GEN[k]})
            else:
                blocks.append(dict(gen))
        stage["generator"] = {"kind": "gen/compound", "blocks": blocks}
    return spec


def _maybe_hybrid_wrap(spec, preset_attr, native):
    if preset_attr in _HYBRID_PRESETS:
        kinds = _block_kinds(native)
        if _needs_compound(kinds):
            _compound_wrap_stages(spec, kinds, hybrid=True)
    return spec


# ---------------------------------------------------------------------------
# Table-driven generation for every UNIFORMLY-shaped preset wrapper (i.e.
# every preset except the 5 ga_* variants folded into GeneticAlgorithm's own
# auto-dispatch, and the 3 de_*/jde variants folded into
# DifferentialEvolution's own variant= dispatch -- both handled below).
#
# Each row: (class_name, preset_attr, mode, default_pop_size, source, doc).
#
# mode:
#   "pop_budget"   -- sezgi.presets.<preset_attr>(pop_size, budget, **kwargs)
#                     (covers es_mu_plus_lambda's dist=/mean=/... kwargs too
#                     -- they flow through **preset_kwargs unchanged, no
#                     special-casing needed: preset_es_mu_plus_lambda's own
#                     pyo3 signature already accepts them as kwargs).
#   "budget_only"  -- sezgi.presets.<preset_attr>(budget, **kwargs); the
#                     preset's own Rust signature has NO pop_size parameter
#                     at all (sa: pop_size is hard-coded to 1 inside the
#                     preset itself, presets.rs:743-759) -- pop_size is
#                     accepted at the wrapper's own __init__ ONLY for
#                     interface uniformity with every other class, fixed at
#                     1, and rejected (ValueError) if given any other value.
#   "dim_budget"   -- sezgi.presets.<preset_attr>(dim, budget, **kwargs),
#                     dim read from the problem at run() time
#                     (native.dim()); the preset's own population size is a
#                     DERIVED formula (lshade: 18*dim, presets.rs:776-791;
#                     cmaes_ipop: 4+floor(3*ln(dim)), presets.rs:813-834;
#                     nelder_mead: dim+1, presets.rs:836-853), so an explicit
#                     pop_size at __init__ is REJECTED (ValueError) -- there
#                     is no "pop_size" knob these presets expose to override.
#
# default_pop_size: the class's own `pop_size` default. Where
# presets.rs's own doc comment states a canonical value (the metaphor
# algorithms' own source-paper convention, e.g. "canonical is 30 per the
# source paper"), that value is used verbatim. Where presets.rs states NO
# canonical value ("the caller picks it, no canonical value from a single
# source" -- de_rand_1/de_best_1/jde, ga_real/perm/bin/int/cat,
# es_mu_plus_lambda, pso, shade, cmaes, random_search), 20 is used --
# matching this repo's own dominant convention for these exact presets
# (py-sezgi/tests/test_solve.py, README.md's own top-line example). None
# for "budget_only"/"dim_budget" rows (pop_size is fixed/derived, not a
# free default -- see mode's own doc above).
_PRESET_TABLE = [
    ("EvolutionStrategy", "es_mu_plus_lambda", "pop_budget", 20,
     "presets.rs:56-70",
     "(mu+lambda)-Evolution Strategy -- gen/step over a caller-selected "
     "mutation distribution (dist=, plus its own params: mean=/sigma= for "
     "gaussian (default), loc=/scale= for cauchy/laplace, alpha= for levy, "
     "nu= for student_t) paired with replace/mu-plus-lambda."),
    ("ParticleSwarm", "pso", "pop_budget", 20, "presets.rs:243-257",
     "Particle Swarm Optimization (Clerc & Kennedy constriction variant, "
     "spec name \"pso/clerc-kennedy\") -- gen/pso (w=0.7298, "
     "c1=c2=1.49618) paired with replace/pso-commit."),
    ("SimulatedAnnealing", "sa", "budget_only", None, "presets.rs:743-759",
     "Simulated Annealing (Metropolis acceptance, geometric cooling "
     "t0=1.0, alpha=0.999) -- gen/step (gaussian, sigma=0.5) paired with "
     "replace/metropolis. Single-trajectory: pop_size is fixed at 1 by "
     "the preset itself."),
    ("SHADE", "shade", "pop_budget", 20, "presets.rs:761-774",
     "SHADE (Success-History-based Adaptive DE, Tanabe & Fukunaga 2013) "
     "-- gen/de-shade (h=6, p=0.11) paired with replace/shade and "
     "adapter/shade-history."),
    ("LSHADE", "lshade", "dim_budget", None, "presets.rs:776-791",
     "L-SHADE (Linear-population-size-reduction SHADE, Tanabe & Fukunaga "
     "2014) -- same gen/de-shade + replace/shade as SHADE, plus "
     "adapter/shade-lshade for the linear population shrink. pop_size is "
     "DERIVED as 18*dim (the preset's own formula) -- not a free "
     "parameter of this wrapper."),
    ("CMAES", "cmaes", "pop_budget", 20, "presets.rs:798-811",
     "(mu/mu_w,lambda)-CMA-ES (Hansen's tutorial form, positive-weights "
     "variant) -- gen/cma paired with replace/cma-update. pop_size is "
     "lambda; Hansen's own guideline is 4+floor(3*ln(dim)) (see "
     "CMAESIpop, which computes this automatically) -- this class leaves "
     "the choice to the caller, matching the Rust preset's own signature."),
    ("CMAESIpop", "cmaes_ipop", "dim_budget", None, "presets.rs:813-834",
     "CMA-ES with IPOP-style stagnation restarts (M2b Task 12) -- same "
     "gen/cma + replace/cma-update stage as CMAES, plus restart/stagnation "
     "(patience=2000, sizing=ipop, factor=2.0, max_pop=512). pop_size is "
     "DERIVED as 4+floor(3*ln(dim)) (Hansen's default, computed by the "
     "preset itself since IPOP restarts scale from this starting "
     "population) -- not a free parameter of this wrapper."),
    ("NelderMead", "nelder_mead", "dim_budget", None, "presets.rs:836-853",
     "Nelder-Mead simplex (M2b Task 13) -- gen/nelder-mead paired with "
     "replace/nelder-mead. pop_size is DERIVED as dim+1 (the population "
     "IS the simplex) -- not a free parameter of this wrapper."),
    ("RandomSearch", "random_search", "pop_budget", 20,
     "presets.rs:855-869",
     "Uniform random resampling (gen/uniform-resample) paired with "
     "replace/mu-plus-lambda -- the baseline every other algorithm should "
     "beat."),
    ("GreyWolfOptimizer", "gwo", "pop_budget", 30, "presets.rs:259-279",
     "Grey Wolf Optimizer (Mirjalili, Mirjalili & Lewis 2014) -- gen/gwo "
     "paired with replace/generational (non-elitist by construction). "
     "pop_size is the pack size; canonical is 30 per the source paper."),
    ("WhaleOptimization", "woa", "pop_budget", 30, "presets.rs:281-302",
     "Whale Optimization Algorithm (Mirjalili & Lewis 2016) -- gen/woa "
     "paired with replace/generational. pop_size is the school size; "
     "canonical is 30 per the source paper."),
    ("HarmonySearch", "harmony_search", "pop_budget", 30,
     "presets.rs:304-327",
     "Harmony Search (Geem, Kim & Loganathan 2001) -- gen/hs paired with "
     "replace/worst-if-better. pop_size is HMS (Harmony Memory Size); "
     "canonical is 30 per the source paper."),
    ("CuckooSearch", "cuckoo_search", "pop_budget", 25,
     "presets.rs:329-359",
     "Cuckoo Search (Yang & Deb 2009) -- gen/cuckoo_levy paired with "
     "replace/one-to-one-greedy and adapter/abandon-worst-fraction "
     "(pa=0.25). pop_size is the nest count; canonical is 25 per the "
     "source paper."),
    ("GrasshopperOptimization", "goa", "pop_budget", 30,
     "presets.rs:361-386",
     "Grasshopper Optimisation Algorithm (Saremi, Mirjalili & Lewis 2017) "
     "-- gen/goa paired with replace/generational. pop_size is the swarm "
     "size; canonical is 30 per the source paper."),
    ("SineCosineAlgorithm", "sca", "pop_budget", 30, "presets.rs:388-411",
     "Sine Cosine Algorithm (Mirjalili 2016) -- gen/sca paired with "
     "replace/generational. pop_size is the number of search agents; "
     "canonical is 30 per the source paper."),
    ("JAYA", "jaya", "pop_budget", 30, "presets.rs:413-438",
     "JAYA (Rao 2016) -- gen/jaya paired with replace/one-to-one-greedy. "
     "pop_size is the candidate count; canonical is 30 per this wave's "
     "own convention (the paper itself demonstrates with 5)."),
    ("MothFlameOptimization", "mfo", "pop_budget", 30, "presets.rs:440-464",
     "Moth-Flame Optimization (Mirjalili 2015) -- gen/mfo paired with "
     "replace/generational and adapter/mfo-flame-update (the flame "
     "memory). pop_size is the number of search agents; canonical is 30 "
     "per the source paper."),
    ("SalpSwarm", "ssa", "pop_budget", 30, "presets.rs:466-491",
     "Salp Swarm Algorithm (Mirjalili et al. 2017) -- gen/ssa paired with "
     "replace/generational. pop_size is the number of salps; canonical is "
     "30 per the source paper."),
    ("FireflyAlgorithm", "firefly", "pop_budget", 25, "presets.rs:493-521",
     "Firefly Algorithm (Yang, X.-S., Nature-Inspired Metaheuristic "
     "Algorithms, 2nd ed., Luniver Press, 2010) -- gen/fa paired with "
     "replace/generational. pop_size is the number of fireflies; "
     "canonical is 25 per this wave's own convention (the source's own "
     "demo uses 20)."),
    ("BatAlgorithm", "bat", "pop_budget", 30, "presets.rs:523-550",
     "Bat Algorithm (Yang, X.-S. 2010, NICSO) -- gen/ba paired with the "
     "new replace/bat-loudness-greedy. pop_size is the number of bats; "
     "canonical is 30 per this wave's own convention (the source's own "
     "demo uses 20)."),
    ("FlowerPollination", "fpa", "pop_budget", 25, "presets.rs:552-580",
     "Flower Pollination Algorithm (Yang, X.-S. 2012, UCNC) -- gen/fpa "
     "paired with replace/one-to-one-greedy. pop_size is the "
     "flower/pollen-gamete count; canonical is 25 per the source's demo."),
    ("TLBO", "tlbo", "pop_budget", 30, "presets.rs:582-620",
     "Teaching-Learning-Based Optimization (Rao, Savsani & Vakharia 2011) "
     "-- sezgi's first MULTI-STAGE preset: gen/tlbo-teacher then "
     "gen/tlbo-learner, each paired with replace/one-to-one-greedy. "
     "pop_size is the class size; canonical is 30 per the source paper. A "
     "full generation costs 2*pop_size evaluations."),
    ("HarrisHawks", "hho", "pop_budget", 30, "presets.rs:622-648",
     "Harris Hawks Optimization (Heidari, Mirjalili, Faris, Aljarah, "
     "Mafarja & Chen 2019) -- gen/hho paired with replace/generational. "
     "pop_size is the hawk count; canonical is 30 per the source's own "
     "demo."),
    ("AntLion", "alo", "pop_budget", 25, "presets.rs:650-675",
     "Ant Lion Optimizer (Mirjalili 2015) -- gen/alo paired with "
     "replace/mu-plus-lambda. pop_size is the ant/antlion count; "
     "canonical is 25 per this wave's own convention."),
    ("ArtificialBeeColony", "abc", "pop_budget", 20, "presets.rs:677-708",
     "Artificial Bee Colony (Karaboga 2005, TR-06 / Karaboga & Basturk "
     "2007) -- gen/abc-employed paired with the new "
     "replace/abc-trial-greedy and adapter/abc-onlooker-scout. pop_size "
     "IS SN (the food-source count), NOT Karaboga's colony size NP=2*SN; "
     "canonical is 20 per this module's own resolved convention. A full "
     "cycle costs 2*pop_size evaluations (+1 when a scout fires)."),
    ("GravitationalSearch", "gsa", "pop_budget", 30, "presets.rs:710-741",
     "Gravitational Search Algorithm (Rashedi, Nezamabadi-pour & "
     "Saryazdi 2009) -- gen/gsa paired with replace/generational. "
     "pop_size is the agent count; canonical is 30 per this wave's own "
     "convention."),
]


def _make_preset_class(class_name, preset_attr, mode, default_pop_size,
                        source, citation):
    """Generates one uniformly-shaped preset wrapper class from a
    `_PRESET_TABLE` row. See the table's own header comment for `mode`'s
    three variants."""

    def __init__(self, pop_size=None, **preset_kwargs):
        if mode == "budget_only":
            if pop_size is not None and pop_size != 1:
                raise ValueError(
                    f"{class_name} has a fixed population of 1 (a single "
                    f"search trajectory -- sezgi.presets.{preset_attr}'s "
                    "own Rust signature has no pop_size parameter at all, "
                    f"{source}) -- pop_size is not a settable parameter "
                    f"of this class, got pop_size={pop_size!r}")
            self.pop_size = 1
        elif mode == "dim_budget":
            if pop_size is not None:
                raise ValueError(
                    f"{class_name} derives its population size from the "
                    f"problem's own dimensionality at run() time "
                    f"(sezgi.presets.{preset_attr}'s own formula, "
                    f"{source}) -- pop_size is not a settable parameter "
                    f"of this class, got pop_size={pop_size!r}")
            self.pop_size = None
        else:
            self.pop_size = default_pop_size if pop_size is None else pop_size
        self.preset_kwargs = preset_kwargs

    if mode == "pop_budget":
        __init__.__doc__ = (
            f"Constructs a {class_name} instance; see the class docstring "
            "for the algorithm itself.\n"
            f"pop_size: population size for sezgi.presets.{preset_attr}, "
            f"defaults to {default_pop_size} (see the class docstring for "
            "where this default comes from).\n"
            "**preset_kwargs: forwarded UNCHANGED to "
            f"sezgi.presets.{preset_attr}(pop_size, budget, **preset_kwargs) "
            f"at run() time -- any keyword that preset's own Rust signature "
            f"accepts beyond pop_size/budget ({source}).")
    elif mode == "budget_only":
        __init__.__doc__ = (
            f"Constructs a {class_name} instance; see the class docstring "
            "for the algorithm itself.\n"
            f"pop_size: NOT a free parameter -- {class_name} is a "
            "single-trajectory search, fixed at 1 by "
            f"sezgi.presets.{preset_attr}'s own Rust signature ({source}); "
            "passing anything other than None or 1 raises ValueError.\n"
            "**preset_kwargs: forwarded UNCHANGED to "
            f"sezgi.presets.{preset_attr}(budget, **preset_kwargs) at "
            "run() time.")
    else:  # dim_budget
        __init__.__doc__ = (
            f"Constructs a {class_name} instance; see the class docstring "
            "for the algorithm itself.\n"
            f"pop_size: NOT a free parameter -- {class_name} DERIVES its "
            "population from the problem's own dimensionality at run() "
            f"time (sezgi.presets.{preset_attr}'s own formula, {source} -- "
            "see the class docstring for the exact formula); passing "
            "anything other than None raises ValueError.\n"
            "**preset_kwargs: forwarded UNCHANGED to "
            f"sezgi.presets.{preset_attr}(dim, budget, **preset_kwargs) at "
            "run() time.")

    def run(self, problem, budget, seed=0, run_id=0, log_dir=None):
        native = as_native_problem(problem)
        _check_hybrid_ineligible(class_name, preset_attr, native)
        preset_fn = getattr(sezgi.presets, preset_attr)
        if mode == "pop_budget":
            spec = preset_fn(self.pop_size, budget, **self.preset_kwargs)
        elif mode == "budget_only":
            spec = preset_fn(budget, **self.preset_kwargs)
        else:  # dim_budget
            spec = preset_fn(native.dim(), budget, **self.preset_kwargs)
        _maybe_hybrid_wrap(spec, preset_attr, native)
        return _run_spec(class_name, spec, native, budget, seed, run_id, log_dir)

    run.__doc__ = (
        f"Runs sezgi.presets.{preset_attr}(...) via sezgi.solve() and "
        "wraps the result in sezgi.algo.SolveResult. See the class "
        "docstring for pop_size's own semantics under this preset.")

    doc = (
        f"{citation}\n\n"
        f"Delegates to sezgi.presets.{preset_attr} "
        f"(crates/components/src/{source}).")
    if preset_attr in _HYBRID_PRESETS:
        doc += _HYBRID_DOC.format(label=_HYBRID_PRESETS[preset_attr])
    elif preset_attr in _HYBRID_INELIGIBLE:
        doc += ("\n\nMixed spaces: NOT supported -- "
                + _HYBRID_INELIGIBLE[preset_attr] + ".")

    return type(class_name, (object,), {
        "__init__": __init__,
        "run": run,
        "__doc__": doc,
        "_preset_attr": preset_attr,
        "_mode": mode,
    })


# One module-level class per table row -- this IS the mechanical
# "~30 classes" deliverable; nothing below this point hand-diverges from
# `_make_preset_class`'s single implementation.
for _row in _PRESET_TABLE:
    globals()[_row[0]] = _make_preset_class(*_row)
del _row


# ---------------------------------------------------------------------------
# GeneticAlgorithm: auto-dispatching over the 5 ga_* presets (Task 5's
# pinned deliverable 2). Hand-written (not table-generated) because its
# dispatch axis is the PROBLEM's own space, not a constructor kwarg alone --
# genuinely different shape from every other class in this file -- but its
# run() tail still funnels through the same `_run_spec` helper.

_GA_KIND_TO_REPRESENTATION = {
    "float": "real",
    "permutation": "perm",
    "binary": "bin",
    "int": "int",
    "categorical": "cat",
}

_GA_REPRESENTATION_TO_PRESET = {
    "real": "ga_real", "perm": "ga_perm", "bin": "ga_bin",
    "int": "ga_int", "cat": "ga_cat",
}


class GeneticAlgorithm(object):
    """Genetic Algorithm (real-coded SBX / permutation OX / binary /
    integer / categorical) -- auto-dispatches to one of
    sezgi.presets.ga_real / ga_perm / ga_bin / ga_int / ga_cat
    (crates/components/src/presets.rs:72-241) based on the problem's own
    space block kinds, read via the native Problem handle's `.blocks()`
    accessor (M4-1 Task 5: a minimal read-only introspection accessor added
    to `PyProblem` in py-sezgi/src/lib.rs -- `dim()`/`bounds()`/`optimum()`
    alone cannot distinguish an all-Float space from an all-Permutation
    one; see this task's report):

        all-Float        -> presets.ga_real
        all-Permutation   -> presets.ga_perm
        all-Binary        -> presets.ga_bin
        all-Int           -> presets.ga_int
        all-Categorical   -> presets.ga_cat

    A multi-block space whose blocks all share one kind is dispatched the
    same way, but run() wraps one copy of the preset's generator per block
    in a gen/compound generator (init/boundary/replacer/termination stay the
    preset's own), so every block is recombined independently. Other flat
    presets used on a multi-block space fail fast with the engine's
    GenotypeShapeMismatch guard.

    A space MIXING block kinds (e.g. Float + Int together) has no ga_*
    preset in this milestone -- run() raises NotImplementedError naming the
    gen/compound generator (crates/components/src/compound.rs) via a
    hand-written AlgorithmSpec dict/TOML passed directly to sezgi.solve() as
    this milestone's documented workaround (see
    sezgi.problems.mixed_diagnostic's own doc for a worked mixed-space
    scaffold problem built for exactly this path).

    `representation=` (one of "real"/"perm"/"bin"/"int"/"cat", a
    constructor kwarg) OVERRIDES auto-dispatch entirely -- block-kind
    introspection is skipped whenever it is given.

    After a run() call, `self.dispatched_representation` holds the
    representation actually engaged (e.g. "real") -- an introspectable
    accessor proving which preset ran, independent of `representation=`
    itself (which stays None under auto-dispatch).
    """

    def __init__(self, pop_size=20, representation=None, **preset_kwargs):
        """pop_size: population size, forwarded to whichever ga_* preset
        run() ultimately dispatches to (default 20, matching this module's
        own convention for the ga_* presets -- see _PRESET_TABLE's header
        comment). representation: overrides auto-dispatch entirely when
        given (one of "real"/"perm"/"bin"/"int"/"cat"); None (default)
        means auto-dispatch from the problem's own space at run() time --
        see the class docstring for the full dispatch table and the Mixed-
        space error. **preset_kwargs: forwarded UNCHANGED to the dispatched
        sezgi.presets.ga_*(pop_size, budget, **preset_kwargs) at run() time."""
        if representation is not None and representation not in _GA_REPRESENTATION_TO_PRESET:
            raise ValueError(
                "representation must be one of "
                f"{sorted(_GA_REPRESENTATION_TO_PRESET)} or None "
                f"(auto-dispatch), got {representation!r}")
        self.pop_size = pop_size
        self.representation = representation
        self.preset_kwargs = preset_kwargs
        self.dispatched_representation = None

    def _resolve_representation(self, native):
        if self.representation is not None:
            return self.representation
        blocks = native.blocks()
        if not blocks:
            raise ValueError("GeneticAlgorithm auto-dispatch: empty search space")
        kinds = {b["kind"] for b in blocks}
        if len(kinds) != 1:
            raise NotImplementedError(
                "GeneticAlgorithm auto-dispatch requires every block in "
                f"the space to share one kind; got a Mixed space with "
                f"kinds {sorted(kinds)}. This milestone has no ga_* preset "
                "for a mixed space -- build the AlgorithmSpec by hand "
                "around the gen/compound generator "
                "(crates/components/src/compound.rs) instead, passed "
                "directly to sezgi.solve() (see "
                "sezgi.problems.mixed_diagnostic's own doc for a worked "
                "mixed-space scaffold problem), or pass representation= "
                "to force a single-kind preset.")
        kind = next(iter(kinds))
        representation = _GA_KIND_TO_REPRESENTATION.get(kind)
        if representation is None:
            raise NotImplementedError(
                f"GeneticAlgorithm auto-dispatch has no preset for block "
                f"kind {kind!r}")
        return representation

    def run(self, problem, budget, seed=0, run_id=0, log_dir=None):
        """Resolves the representation (auto-dispatch from problem.space(),
        or self.representation if forced), runs the matching
        sezgi.presets.ga_* preset via sezgi.solve(), and wraps the result
        in sezgi.algo.SolveResult. Sets self.dispatched_representation as a
        side effect (see the class docstring). Raises NotImplementedError
        for a Mixed-kind space unless representation= was given."""
        native = as_native_problem(problem)
        representation = self._resolve_representation(native)
        self.dispatched_representation = representation
        preset_attr = _GA_REPRESENTATION_TO_PRESET[representation]
        preset_fn = getattr(sezgi.presets, preset_attr)
        spec = preset_fn(self.pop_size, budget, **self.preset_kwargs)
        n_blocks = len(native.blocks())
        if n_blocks > 1:
            # The ga_* presets' flat generators are single-block; wrap one
            # per-block copy in gen/compound so every block is recombined
            # independently and the genotype keeps its block structure.
            stages = spec["stages"]
            if len(stages) != 1:
                raise RuntimeError(
                    "GeneticAlgorithm: expected a single-stage ga_* preset "
                    f"spec, got {len(stages)} stages")
            _compound_wrap_stages(
                spec, [b["kind"] for b in native.blocks()], hybrid=False)
        return _run_spec("GeneticAlgorithm", spec, native, budget, seed,
                          run_id, log_dir)


# ---------------------------------------------------------------------------
# DifferentialEvolution: variant= over the 3 de_rand_1/de_best_1/jde
# presets (Task 5's pinned deliverable, "variant= for de_rand_1/de_best_1/
# jde"). Hand-written for the same reason as GeneticAlgorithm (dispatch is
# not a uniform "one preset" shape), but note the axis here is a
# CONSTRUCTOR kwarg (variant=), not the problem's own space -- all three DE
# presets share the identical (pop_size, budget) signature
# (presets.rs:8-54), so no problem introspection is needed at run() time.

_DE_VARIANT_TO_PRESET = {"rand1": "de_rand_1", "best1": "de_best_1", "jde": "jde"}


class DifferentialEvolution(object):
    """Differential Evolution -- delegates to sezgi.presets.de_rand_1
    (default, DE/rand/1/bin) / de_best_1 (DE/best/1/bin) / jde
    (self-adaptive jDE) (crates/components/src/presets.rs:8-54), selected
    by `variant=` ("rand1" | "best1" | "jde") at construction time. All
    three presets share the identical (pop_size, budget) signature, so
    variant= is the only dispatch axis -- no problem introspection needed
    (unlike GeneticAlgorithm's space-driven auto-dispatch).

    Mixed spaces: variant="rand1"/"best1" auto-dispatch to a HYBRID via
    gen/compound -- DE on continuous (float) blocks, GA variation
    (gen/ga-bin / gen/ga-int / gen/ga-cat / gen/ga-perm, default params) on
    discrete blocks; parent selection and replacement follow the DE
    pipeline. A space with NO float block gets zero DE-specific variation
    (all variation is GA). variant="jde" is stateful and raises NotImplementedError on
    mixed spaces."""

    def __init__(self, pop_size=20, variant="rand1", **preset_kwargs):
        """pop_size: population size, forwarded to whichever de_*/jde preset
        run() dispatches to (default 20, matching this module's own
        convention -- all three variants share the same (pop_size, budget)
        signature, presets.rs:8-54). variant: one of "rand1" (default,
        sezgi.presets.de_rand_1) / "best1" (de_best_1) / "jde" (jde) -- see
        the class docstring for what each variant means. **preset_kwargs:
        forwarded UNCHANGED to the dispatched preset at run() time (none of
        the three currently accept any beyond pop_size/budget)."""
        if variant not in _DE_VARIANT_TO_PRESET:
            raise ValueError(
                f"variant must be one of {sorted(_DE_VARIANT_TO_PRESET)}, "
                f"got {variant!r}")
        self.pop_size = pop_size
        self.variant = variant
        self.preset_kwargs = preset_kwargs

    def run(self, problem, budget, seed=0, run_id=0, log_dir=None):
        """Runs the preset self.variant selected at construction time
        (sezgi.presets.de_rand_1 / de_best_1 / jde) via sezgi.solve(), and
        wraps the result in sezgi.algo.SolveResult."""
        native = as_native_problem(problem)
        preset_attr = _DE_VARIANT_TO_PRESET[self.variant]
        preset_fn = getattr(sezgi.presets, preset_attr)
        if self.variant == "jde":
            kinds = _block_kinds(native)
            if _needs_compound(kinds):
                raise NotImplementedError(
                    "DifferentialEvolution(variant='jde') cannot run on a "
                    f"mixed or non-float space (block kinds {kinds}): "
                    "gen/de-jde is stateful: " + _WHY_STATEFUL.format(
                        what="per-individual jde_f/jde_cr") + ". Use "
                    "variant='rand1' or 'best1' for the compound hybrid.")
        spec = preset_fn(self.pop_size, budget, **self.preset_kwargs)
        _maybe_hybrid_wrap(spec, preset_attr, native)
        return _run_spec("DifferentialEvolution", spec, native, budget,
                          seed, run_id, log_dir)


# ---------------------------------------------------------------------------
# NSGA2: a thin class skin over sezgi.mo.nsga2 -- NOT one of the 34
# presets.rs builders (NSGA-II is not built on the scalar
# Engine/Registry/Generator machinery those presets target).
# ZERO new MO capability: run() delegates to mo.nsga2 verbatim.

class NSGA2(object):
    """Class skin over sezgi.mo.nsga2 (crates/components/src/nsga2.rs:
    2370-2377, sezgi.mo.nsga2's own docstring in
    py-sezgi/python/sezgi/__init__.py for the full parameter contract) --
    ZERO new MO capability: run() delegates to mo.nsga2 VERBATIM, same
    positional/keyword arguments, same return dict. Authoring an NSGA-II
    variant as an engine-hosted Python callback (the way sezgi.Algorithm
    lets a scalar algorithm's generate() be authored in Python) is
    explicitly OUT OF SCOPE this milestone: NSGA-II's own
    selection/crossover/replacement loop is hard-coded Rust
    (nsga2_run_float_impl/nsga2_run_binary_impl/nsga2_run_mixed_impl), not
    routed through the Registry/AlgorithmSpec/Engine/Generator machinery
    sezgi.Algorithm targets -- see the research doc's §B7 for the full
    reasoning and what a future milestone would need.

    Constructor kwargs mirror mo.nsga2's own "algorithm configuration"
    parameters -- pop_size plus every VARIATION-OPERATOR knob (eta_c,
    eta_m, p_c, p_m, p_c_bin, p_m_bin, p_c_cat, p_m_cat), the ones that
    describe the algorithm instance itself, independent of which problem
    it is pointed at. run()'s own arguments mirror mo.nsga2's remaining
    "this particular problem/run" parameters (problem, dim, budget, m, k,
    l, seed, log_dir, label) -- m/k/l describe the PROBLEM being solved
    (m = objective count for dtlz/wfg; k/l = WFG's own shape parameters),
    not the algorithm, so they belong with run() rather than __init__,
    exactly mirroring mo.nsga2's own per-problem-family validation (a
    parameter given where it does not apply, or omitted where required,
    raises ValueError the same way calling mo.nsga2 directly would).
    Together, calling NSGA2(pop_size=P, **op_kwargs).run(problem, dim,
    budget, m=M, seed=S, ...) is IDENTICAL to calling
    mo.nsga2(problem, dim, P, budget, m=M, seed=S, **op_kwargs, ...)
    directly (see this task's anchored-equivalence test).

    run()'s return value is mo.nsga2's OWN dict shape (individuals,
    objectives, front0, evals_used, violations when constrained) -- NOT
    sezgi.algo.SolveResult: that dataclass's fields (best_x/best_f/f_opt/
    gap) assume a single-objective run with one best point, which does not
    fit NSGA-II's multi-objective Pareto-front result.
    """

    def __init__(self, pop_size, eta_c=20.0, eta_m=20.0, p_c=0.9,
                 p_m=None, p_c_bin=0.9, p_m_bin=None, p_c_cat=0.9,
                 p_m_cat=None):
        """The algorithm-configuration half of mo.nsga2's parameters (see
        the class docstring for why the split is here and not at run()).
        pop_size: population size -- REQUIRED, no default (mo.nsga2 itself
        requires >= 4 and a multiple of 4; ValueError at run() time
        otherwise). eta_c/eta_m/p_c: NSGA-II paper's own pinned experimental
        settings (Deb et al. 2002, Sec. IV.A) as the defaults. p_m: None
        (default) resolves on the Rust side to 1/n_variables. p_c_bin/
        p_m_bin/p_c_cat/p_m_cat: the Binary/Categorical-genotype
        counterparts, consulted only when the problem's space contains that
        block kind -- see mo.nsga2's own docstring for each default's
        provenance."""
        self.pop_size = pop_size
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.p_c = p_c
        self.p_m = p_m
        self.p_c_bin = p_c_bin
        self.p_m_bin = p_m_bin
        self.p_c_cat = p_c_cat
        self.p_m_cat = p_m_cat

    def run(self, problem, dim, budget, m=None, seed=0, k=None, l=None,
            log_dir=None, label=None):
        """Delegates to sezgi.mo.nsga2(problem, dim, self.pop_size, budget,
        ...) verbatim -- see the class docstring for the full parameter
        mirroring and the return-value shape (mo.nsga2's own dict, not
        SolveResult)."""
        return sezgi.mo.nsga2(
            problem, dim, self.pop_size, budget, m=m, seed=seed,
            eta_c=self.eta_c, eta_m=self.eta_m, p_c=self.p_c, p_m=self.p_m,
            p_c_bin=self.p_c_bin, p_m_bin=self.p_m_bin,
            p_c_cat=self.p_c_cat, p_m_cat=self.p_m_cat, k=k, l=l,
            log_dir=log_dir, label=label)


__all__ = [row[0] for row in _PRESET_TABLE] + [
    "GeneticAlgorithm", "DifferentialEvolution", "NSGA2",
]
