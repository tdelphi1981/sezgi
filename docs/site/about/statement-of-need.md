# Statement of need

This page states, in the form JOSS and SoftwareX reviewers expect, what
problem sezgi is designed to solve, who it is for, and how it relates to
existing tools in the same space. The [Home](../index.md) page carries a
shorter version of the same statement for a general reader; this page is
the fuller, citable one.

## Summary

**sezgi** (Turkish for "intuition") is a Rust-core, component-based
metaheuristic optimization library with Python and R frontends. One Rust
engine drives both frontends: a user-authored algorithm hook (Python
`Algorithm.generate()` / R6 `Algorithm$generate()`) runs as a callback
*into* that same engine loop rather than as a separate per-language
reimplementation, so a shared seed reproduces byte-identical runs across
languages. Built-in algorithms are exposed as classes (29 in Python, 29
in R6), each citing its source; user-authored problems and algorithms are
ordinary subclasses.

## The problem

Metaheuristic-optimization research has two related, well-documented
reproducibility problems that motivated sezgi's design:

1. **Under-specified randomness.** Published algorithm comparisons rarely
   pin the RNG draw order, seeding scheme, or exact operator sequence
   precisely enough for a third party to reproduce a run byte-for-byte.
   A "same algorithm, same parameters, same seed" claim is frequently not
   actually checkable against the paper alone.
2. **Metaphor inflation without equivalence checking.** A large and
   growing body of "novel" nature-inspired algorithms turns out, on
   careful component-by-component analysis, to be a known method wearing
   new vocabulary — a critique made explicitly and repeatedly in the
   literature (Camacho-Villalón, Dorigo & Stützle, ANTS 2020 and its
   *International Transactions in Operational Research* journal
   extension, covering six metaphor-named algorithms; Weyland (2010) on
   Harmony Search as a special case of evolution strategies; Črepinšek,
   Liu & Mernik (2012) on accounting/framing issues in
   Teaching-Learning-Based Optimization). sezgi's own algorithm catalog
   (`examples/README.md`) cites this literature per algorithm, honestly,
   rather than presenting each preset as an independent invention.

sezgi addresses both: every run is fully determined by
`(spec, problem, seed, budget)` with a single pinned Rust engine behind
every frontend, and every built-in algorithm's class docstring states its
primary source and, where one exists, the equivalence critique that
applies to it.

## Target audience

- **Optimization researchers** who need cross-language, bit-exact
  reproducibility for algorithm comparisons — running the same
  configuration from Python or R and getting the identical trajectory,
  not merely a statistically similar one.
- **Students learning metaheuristics** who want a library that exposes
  the field's actual structure (selection / variation / replacement,
  exploration vs. exploitation, encodings and search spaces — see the
  [Learn track](../learn/01-what-are-metaheuristics.md)) rather than
  hiding it behind an opaque `.optimize()` call.
- **Practitioners** who want a dependable, class-first, documented
  library — subclass `Algorithm`, `PopulationAlgorithm`, `LocalSearch`,
  or `Problem` — rather than a loose collection of reference scripts
  copied out of a paper's appendix.

## State of the field

sezgi is not the first metaheuristic-optimization library, and does not
claim to be the most feature-complete one. Naming where it sits relative
to established tools, honestly:

- **[pymoo](https://pymoo.org/)** (Apache-2.0) is the most feature-rich
  Python multi-objective optimization library, with a broad algorithm and
  problem-suite catalog and a mature, widely-adopted class hierarchy.
  sezgi's own `Algorithm` / `PopulationAlgorithm` hook taxonomy (the
  `initialize()` / `select()` / `vary()` / `generate()` split) is
  explicitly modeled after pymoo's public API shape — one-line
  attribution in the base-class module docstring — but sezgi's engine is
  Rust, not Python: a subclassed hook runs as a callback *into* a
  compiled loop that owns budget tracking, boundary repair, and
  evaluation counting, rather than a pure-Python loop owning all of it.
  This is a different tradeoff (a smaller surface, a build step, but a
  single implementation shared bit-exactly across two host languages),
  not a claim of superiority.
- **[jMetal](https://github.com/jMetal/jMetal)** (MIT) is the closest
  Java analogue in template-method structure — its
  `AbstractEvolutionaryAlgorithm`'s
  `createInitialPopulation` / `evaluatePopulation` / `selection` /
  `reproduction` / `replacement` split is the other model sezgi's hook
  taxonomy cites directly, alongside pymoo, for the same reason: it is a
  well-tested public shape for exactly this kind of subclass hierarchy.
  No code is copied from either project; only the taxonomy shape is
  credited.
- **[ecr](https://cran.r-project.org/package=ecr)** (CRAN, S3/closure
  design) is the closest domain analogue on the R side: an
  evolutionary-computation toolbox aimed at researchers, without a class
  system. sezgi's R6-based `Algorithm` hierarchy is a deliberate
  departure from that precedent — the project judged that a true class
  system with inheritance and partial hook override (needed for the
  `Algorithm` → `PopulationAlgorithm` / `LocalSearch` → user-subclass
  chain) is better served by R6's `obj$method()` ergonomics, in the idiom
  the target academic audience already reads from mlr3 and torch, than
  by ecr's closure-based convention.

sezgi's own distinguishing claim, relative to all three, is narrower and
more specific than "more algorithms" or "better performance": **one Rust
engine, two host-language frontends that are bit-exact against each
other for shared algorithms**, with every built-in algorithm's
provenance and any known simplification stated in its own docstring
rather than left implicit.

## Research purpose

sezgi was built as infrastructure for reproducible metaheuristic-
comparison research: every built-in algorithm documents its problem
framing, the design tradeoffs weighed, and the honest scope of every
deferred or simplified piece in its own docstring and in the
documentation pages that accompany the code. That discipline is itself
part of the project's answer to "why does this exist": the goal was
never to add one more algorithm implementation to an already crowded
field, but to make a specific class of comparison — "does this seeded run reproduce
byte-for-byte, in the language of the reader's choice" — checkable rather
than assumed.

## See also

- [Install](../install.md) and [Quickstart](../quickstart.md) for the
  fastest path to a running example.
- [Examples gallery](../examples.md) for real (not toy) usage, including
  a feature-selection recipe and a TSP local search.
- `examples/README.md` in the repository root for the full per-algorithm
  provenance table this page's "State of the field" section draws from.
