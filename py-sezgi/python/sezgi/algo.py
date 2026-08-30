"""Subclassable algorithm authoring over the EvalSession ask/tell core.

M3-4 Task 2: `sezgi.Algorithm` is the pure-Python ABC that later M3-4 tasks
port 17 example algorithms onto (with bit-exact RNG parity against existing
pure scripts). A subclass implements `setup(ctx)`/`step(ctx)`; `solve()`
drives the setup/step loop over an `AlgoContext` until the budget is
exhausted and returns a `SolveResult`.
"""
import abc
import random
from dataclasses import dataclass

import sezgi


class BudgetExhausted(Exception):
    """Raised by AlgoContext.evaluate when a batch does not fit the
    remaining budget. The driver catches it to end the run cleanly; user
    code may also catch it to trigger its own finalization."""


@dataclass
class SolveResult:
    algo: str
    seed: int
    budget: int
    evals_used: int
    best_x: list
    best_f: float
    f_opt: float | None
    gap: float | None


class AlgoContext:
    """Everything a subclass touches during a run. Wraps the EvalSession
    (sole keeper of counting/best/logging) plus a seeded random.Random."""

    def __init__(self, session, dim, bounds, seed):
        self._session = session
        self.dim = dim
        self.bounds = bounds          # (lo, hi)
        self.rng = random.Random(seed)

    @property
    def budget(self):
        return self._session.budget()

    @property
    def evals_used(self):
        return self._session.evals_used()

    @property
    def remaining(self):
        return self._session.budget() - self._session.evals_used()

    @property
    def f_opt(self):
        return self._session.f_opt()

    def best(self):
        return self._session.best()

    def random_point(self):
        lo, hi = self.bounds
        return [self.rng.uniform(lo, hi) for _ in range(self.dim)]

    def evaluate(self, points):
        if len(points) > self.remaining:
            raise BudgetExhausted(
                f"batch of {len(points)} exceeds remaining budget {self.remaining}")
        return self._session.evaluate(points)


class Algorithm(abc.ABC):
    """Subclass, implement setup() and step(), call solve()."""

    name = None  # default resolves to cls.__name__.lower()

    @abc.abstractmethod
    def setup(self, ctx): ...

    @abc.abstractmethod
    def step(self, ctx): ...

    def solve(self, problem, budget, seed, log_dir=None):
        algo_name = self.name or type(self).__name__.lower()
        session = sezgi.EvalSession.for_problem(
            problem, budget, log_dir=log_dir, algo_name=algo_name, seed=seed)
        ctx = AlgoContext(session, problem.dim(), problem.bounds(), seed)
        try:
            self.setup(ctx)
            while ctx.remaining > 0:
                before = ctx.evals_used
                self.step(ctx)
                if ctx.evals_used == before:
                    raise RuntimeError(
                        "step() consumed no budget; a step must evaluate at "
                        "least one point or raise BudgetExhausted")
        except BudgetExhausted:
            pass
        best = ctx.best()
        if best is None:
            raise RuntimeError("run ended with no evaluations at all")
        best_x, best_f = best
        f_opt = ctx.f_opt
        result = SolveResult(
            algo=algo_name, seed=seed, budget=budget,
            evals_used=ctx.evals_used, best_x=list(best_x), best_f=best_f,
            f_opt=f_opt, gap=None if f_opt is None else best_f - f_opt)
        session.finish()
        return result
