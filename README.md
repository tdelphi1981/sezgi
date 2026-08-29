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

## Experiments & Statistics (M2c)

Run multiple algorithms across multiple problems, seeds, and budgets with a TOML grid; get comparative statistics:

    import sezgi
    
    spec_toml = """
        name = "example"
        seeds = [1, 2, 3]
        budgets = [1000, 5000]
        
        [[algorithms]]
        name = "de"
        preset = { kind = "de_rand_1", pop_size = 20 }
        
        [[algorithms]]
        name = "cmaes"
        preset = { kind = "cmaes", pop_size = 20 }
        
        [[problems]]
        suite = "bbob"
        fid = 1
        dim = 10
        instances = [1, 2, 3, 4, 5]
    """
    
    # Run experiment (TOML grid × seeds × budgets × instances)
    records = sezgi.run_experiment(spec_toml, parallel=True)

    # One paper_package PER DISTINCT BUDGET present in records, in ascending
    # budget order — no hand-rolled filtering/aggregation needed. The
    # Wilcoxon tests inside each package need at least 5 problem rows, hence
    # 5 instances above.
    packages = sezgi.per_budget_packages(records, rope=0.01, samples=10000, seed=42)

    for budget, pkg in packages:
        print(f"--- budget={budget} ---")
        print(pkg["latex_summary"])  # Friedman ranks + pairwise Wilcoxon–Holm

`per_budget_packages` reports statistics per budget, never pooled across
budgets: rankings can flip between small and large budgets (Piotrowski et al.
2025), so each budget gets its own package rather than one arbitrarily-chosen
budget standing in for all of them. `sezgi.results_matrix(records, budget)`
is available separately if you need just the `(algo_names, problem_labels,
matrix)` triple for one budget (e.g. to feed a custom analysis).

## Development

    cargo test --workspace --release        # Rust tests
    cd py-sezgi && maturin develop && pytest # Python tests

## Status

M2c (experiment runner + statistics) **complete** — TOML grids with parallel execution, multi-budget checkpoint/resume, Friedman/Holm/Hochberg/Nemenyi tests, Wilcoxon-Pratt/Cliff's δ, Bayesian signed-rank with ROPE, Plackett–Luce rankings, LaTeX paper package generation. Next: M2d (R frontend + examples + ECDF/anytime analysis). License: MIT.

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

Python bindings expose every preset above except `es_mu_plus_lambda` (its `Distribution` argument lacks clean FFI mapping; deferred to M2d). All other algorithm families are ready for experiment-driven research.
