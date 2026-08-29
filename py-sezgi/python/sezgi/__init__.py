"""sezgi — Rust-core, component-based metaheuristic optimization."""
import json
from types import SimpleNamespace

from sezgi._sezgi import Problem, bbob, from_callable
from sezgi import _sezgi


def solve(spec, problem, master_seed=0, run_id=0, log_dir=None, algo_name=None):
    """Run an algorithm spec against a problem.

    spec: dict (JSON-compatible algorithm spec) or a JSON string.
    """
    if isinstance(spec, dict):
        spec = json.dumps(spec)
    return _sezgi.solve(spec, problem, master_seed=master_seed, run_id=run_id,
                        log_dir=log_dir, algo_name=algo_name)


def _preset(fn):
    def wrapper(*args, **kwargs):
        return json.loads(fn(*args, **kwargs))
    return wrapper


# All presets from crates/components/src/presets.rs are exposed here except
# `es_mu_plus_lambda`: its Rust signature takes a `Distribution` (an enum
# with nested params), which doesn't have a clean pyfunction argument
# mapping. Bridging it is deferred to M2d.
presets = SimpleNamespace(
    de_rand_1=_preset(_sezgi.preset_de_rand_1),
    de_best_1=_preset(_sezgi.preset_de_best_1),
    jde=_preset(_sezgi.preset_jde),
    shade=_preset(_sezgi.preset_shade),
    lshade=_preset(_sezgi.preset_lshade),
    ga_real=_preset(_sezgi.preset_ga_real),
    pso=_preset(_sezgi.preset_pso),
    sa=_preset(_sezgi.preset_sa),
    random_search=_preset(_sezgi.preset_random_search),
    nelder_mead=_preset(_sezgi.preset_nelder_mead),
    cmaes=_preset(_sezgi.preset_cmaes),
    cmaes_ipop=_preset(_sezgi.preset_cmaes_ipop),
)

__all__ = ["Problem", "bbob", "from_callable", "solve", "presets"]
