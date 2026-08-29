# sezgi

**sezgi** (Turkish for "intuition") is a Rust-core, component-based metaheuristic
optimization library with Python and R frontends. Design doc: `docs/superpowers/specs/2026-08-27-sezgi-design.md`.

## Quickstart (Python)

    import sezgi

    problem = sezgi.bbob(fid=1, dim=10, instance=1)
    spec = sezgi.presets.de_rand_1(pop_size=50, budget=20_000)
    result = sezgi.solve(spec, problem, master_seed=42, log_dir="logs/")
    print(result["best_f"])   # IOH-format log under logs/

Your own problem (batch evaluation — a single call per population):

    import numpy as np

    def f(X):                       # X: np.ndarray (n, d) float64
        return ((X - 1.0) ** 2).sum(axis=1)

    problem = sezgi.from_callable(f, lo=-5.0, hi=5.0, dim=10)

## Development

    cargo test --workspace --release        # Rust tests
    cd py-sezgi && maturin develop && pytest # Python tests

## Status

M2b (reference algorithms) **complete** — 13 algorithm presets delivered; Adapter + Restart engine families; 24-fid BBOB pins; Best1 dead-draw fix. Next: M2c (experiment runner + statistics), M2d (R frontend + examples). License: MIT.

## Algorithms

sezgi M2b ships 13 reference algorithm presets (with Rust function names):

| Algorithm | Preset Function |
|-----------|-----------------|
| Differential Evolution (rand/1) | `presets::de_rand_1` |
| Differential Evolution (best/1) | `presets::de_best_1` |
| jDE (self-adaptive DE) | `presets::jde` |
| SHADE | `presets::shade` |
| L-SHADE | `presets::lshade` |
| CMA-ES | `presets::cmaes` |
| CMA-ES with IPOP restarts | `presets::cmaes_ipop` |
| Particle Swarm Optimization | `presets::pso` |
| Genetic Algorithm (real-coded) | `presets::ga_real` |
| (μ+λ)-Evolution Strategy | `presets::es_mu_plus_lambda` |
| Simulated Annealing | `presets::sa` |
| Nelder–Mead Simplex | `presets::nelder_mead` |
| Random Search (baseline) | `presets::random_search` |

Python bindings currently expose every preset above except `es_mu_plus_lambda` (its `Distribution` argument crosses the FFI boundary awkwardly; deferred to M2d).
