# sezgi

**sezgi** (Turkish for "intuition") is a Rust-core, component-based
metaheuristic optimization library with Python and R frontends, built
for deterministic, cross-language, bit-exact reproducibility of
algorithm runs.

- **29 built-in algorithm classes** (genetic/evolutionary, swarm,
  physics-inspired, local search), each citing its source in its own
  docstring.
- **A subclassable class hierarchy** (`Algorithm`, `PopulationAlgorithm`,
  `LocalSearch`, `Problem`): your Python `generate()`/`vary()`/`neighbor()`
  hook runs as a callback inside the Rust engine loop.
- **Deterministic by construction**: a run is fully determined by
  `(spec, problem, seed, budget)`, and the same seed reproduces
  byte-identical results across the Python and R frontends.
- **Benchmarking built in**: BBOB and CEC 2014/2017/2022 suites,
  IOH-format logging, ECDF curves, COCO export, and a
  statistical-comparison toolkit (Friedman, Wilcoxon+Holm, Cliff's
  delta, Bayesian signed-rank, Plackett-Luce).
- **Multi-objective optimization** via NSGA-II over ZDT/DTLZ/WFG, with
  hypervolume and IGD indicators.

## Quickstart

```python
import sezgi

problem = sezgi.bbob(1, 5, 1)  # BBOB f1 (Sphere), dim=5, instance=1
result = sezgi.GeneticAlgorithm(pop_size=20).run(problem, budget=1000, seed=1)
print(result.evals_used, result.best_f, result.gap)
```

## Documentation

Full documentation — installation, a six-page introductory Learn track,
eight executed tutorials, architecture pages, and the complete API
reference — is at <https://tdelphi1981.github.io/sezgi/>.

## License

MIT. Developed at the Department of Computer Science, Karadeniz
Technical University. Source code:
<https://github.com/tdelphi1981/sezgi>.
