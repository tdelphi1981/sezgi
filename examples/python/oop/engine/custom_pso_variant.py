"""OOP example: a `sezgi.Algorithm` subclass with a FULL `generate()`
override (no family base -- unlike `custom_de.py`'s `PopulationAlgorithm`
or `local_search_2opt.py`'s `LocalSearch`) implementing a simplified,
PSO-flavored velocity update. Teaches TWO things `custom_de.py` does not:
(1) `generate()` itself, authored end to end, with no `select()`/`vary()`
split to lean on; (2) PER-INSTANCE state kept on `self` and CARRIED across
`generate()` calls (velocities, personal bests, the global best) -- as
opposed to `ctx.rng`, which is per-call/per-stage state the engine itself
owns (see `sezgi.Algorithm`'s own class docstring).

HONESTY NOTE (read before treating this as textbook PSO): `Algorithm.run`'s
default replacer is `replace/mu-plus-lambda` (drains {current pop, this
call's offspring}, sorts by fitness, keeps the best `pop_size`) -- so slot
index `i` in `pop.individuals` on one `generate()` call is NOT guaranteed
to be "the same particle" as slot `i` on the next call, unlike a textbook
PSO's own stable per-particle identity. This class tracks velocity/
personal-best PER SLOT INDEX anyway (an accepted approximation, not a bug
-- see also `sezgi.LocalSearch`'s own ACCEPT() DESIGN docstring for a
similar documented approximation elsewhere in this family). This is
exactly what makes the class "PSO-flavored" rather than PSO: it is still
fully deterministic under a fixed seed (the slot ordering itself is a
deterministic function of the seed), which is all a teaching example
needs, but a rigorous port of PSO onto this hook shape would need a
stable particle-identity tag threaded through the population, which
`sezgi.Algorithm`'s hooks do not provide.
"""
import sezgi


class SimplePSOVariant(sezgi.Algorithm):
    """generate(): a simplified, gbest+pbest PSO-flavored velocity update
    (see the module docstring's HONESTY NOTE for exactly what "simplified"
    means here). Constriction/inertia weight `w`, cognitive coefficient
    `c1`, social coefficient `c2` -- the classic three PSO knobs.

    First call (`self.velocities is None`): velocities start at all-zero,
    one vector per slot; personal bests are seeded from the initial
    population itself (whatever `initialize()` path is in effect -- this
    class does not override `initialize()`, so the engine's own
    `init/uniform` runs, no Python call involved).

    Every call: updates `self.pbest_x`/`self.pbest_f` per slot and
    `self.gbest_x`/`self.gbest_f` globally from the CURRENT population
    (this call's own `pop.individuals`/`pop.fitness`, i.e. whatever
    `replace/mu-plus-lambda` already decided), then draws two independent
    `ctx.rng.next_f64()` coefficients (`r1`, `r2`) PER SLOT PER DIMENSION
    and applies the standard PSO velocity/position update:

        v[j] = w*v[j] + c1*r1*(pbest[j] - x[j]) + c2*r2*(gbest[j] - x[j])
        x[j] = x[j] + v[j]

    Returns exactly `len(pop.individuals)` offspring (one per slot),
    matching `Algorithm.run`'s default `pop_size` behavior.
    """

    def __init__(self, w=0.7, c1=1.5, c2=1.5):
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.velocities = None
        self.pbest_x = None
        self.pbest_f = None
        self.gbest_x = None
        self.gbest_f = None

    def generate(self, pop, ctx):
        n = len(pop.individuals)
        dim = len(pop.individuals[0])

        if self.velocities is None:
            self.velocities = [[0.0] * dim for _ in range(n)]
            self.pbest_x = [list(x) for x in pop.individuals]
            self.pbest_f = [float(f) for f in pop.fitness]

        for i in range(n):
            f = float(pop.fitness[i])
            if f < self.pbest_f[i]:
                self.pbest_f[i] = f
                self.pbest_x[i] = list(pop.individuals[i])
            if self.gbest_f is None or f < self.gbest_f:
                self.gbest_f = f
                self.gbest_x = list(pop.individuals[i])

        offspring = []
        for i in range(n):
            x = pop.individuals[i]
            v = self.velocities[i]
            new_v, new_x = [], []
            for j in range(dim):
                r1 = ctx.rng.next_f64()
                r2 = ctx.rng.next_f64()
                vj = (self.w * v[j]
                      + self.c1 * r1 * (self.pbest_x[i][j] - x[j])
                      + self.c2 * r2 * (self.gbest_x[j] - x[j]))
                new_v.append(vj)
                new_x.append(x[j] + vj)
            self.velocities[i] = new_v
            offspring.append(new_x)
        return offspring


def main():
    res = SimplePSOVariant().run(sezgi.bbob(1, 10, 1), budget=2000, seed=42, pop_size=20)
    print(f"custom_pso_variant (oop engine): evals_used={res.evals_used} "
          f"best_f={res.best_f:.6g} gap={res.gap:.6g}")
    # ANCHORED (per this project's convention): measured once at this exact
    # config (sezgi.bbob(1, 10, 1), budget=2000, seed=42, pop_size=20), then
    # pinned as literal values below.
    assert res.evals_used == 2000
    assert res.best_f == -76.95674124930342
    assert res.gap == 7.447197681757615


if __name__ == "__main__":
    main()
