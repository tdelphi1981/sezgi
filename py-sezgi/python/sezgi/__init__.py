"""sezgi — Rust çekirdekli, bileşen-tabanlı meta-sezgisel optimizasyon."""
import json
from types import SimpleNamespace

from sezgi._sezgi import Problem, bbob, from_callable
from sezgi import _sezgi


def solve(spec, problem, master_seed=0, run_id=0, log_dir=None, algo_name=None):
    """Bir algoritma spec'ini problem üzerinde koştur.

    spec: dict (JSON-uyumlu algoritma spec'i) veya JSON dizgisi.
    """
    if isinstance(spec, dict):
        spec = json.dumps(spec)
    return _sezgi.solve(spec, problem, master_seed=master_seed, run_id=run_id,
                        log_dir=log_dir, algo_name=algo_name)


def _preset(fn):
    def wrapper(pop_size, budget):
        return json.loads(fn(pop_size, budget))
    return wrapper


presets = SimpleNamespace(
    de_rand_1=_preset(_sezgi.preset_de_rand_1),
    ga_real=_preset(_sezgi.preset_ga_real),
    pso=_preset(_sezgi.preset_pso),
)

__all__ = ["Problem", "bbob", "from_callable", "solve", "presets"]
