"""OOP twin of examples/python/alo.py -- same math, same RNG draw order.

Port of the pure-Python Ant Lion Optimizer script onto sezgi.Algorithm. The
pure script (and the Rust module doc it cites) remains the provenance for
the update equations and the three helpers (full-horizon random walk,
floor-shifted roulette, merge-sort-truncate elitism); this file re-derives
NOTHING and must reproduce the pure script's evals_used/best_f/gap output
bit-for-bit at the same seed -- enforced by
py-sezgi/tests/test_examples_oop_parity.py.

The antlion population persists across generations in self.pop/self.fitness
(replace/mu-plus-lambda elitism happens every step).
"""
import sezgi

DIM = 5
BUDGET = 2000
POP_SIZE = 25  # ant/antlion count
SEED = 42


def alo_i_ratio(progress):
    """Eq. (2.10)/(2.11): five SEQUENTIAL unconditional overwrites (not
    elif) -- reproduced literally; since each threshold implies the lower
    ones, the last that fires wins, matching a descending-elseif chain."""
    i = 1.0
    if progress > 0.1:
        i = 1.0 + 100.0 * progress
    if progress > 0.5:
        i = 1.0 + 1000.0 * progress
    if progress > 0.75:
        i = 1.0 + 10_000.0 * progress
    if progress > 0.9:
        i = 1.0 + 100_000.0 * progress
    if progress > 0.95:
        i = 1.0 + 1_000_000.0 * progress
    return i


def alo_shift_bound(scaled, antlion_d, use_plus):
    return scaled + antlion_d if use_plus else -scaled + antlion_d


def alo_cumsum_walk(steps):
    """Eq. (2.1): X[0]=0, X[k]=X[k-1]+(+1 if step else -1)."""
    x = [0.0]
    acc = 0.0
    for s in steps:
        acc += 1.0 if s else -1.0
        x.append(acc)
    return x


def alo_normalize_walk(x, row, c, d):
    """Eq. (2.7): min-max normalize the WHOLE walk into [c,d], read at row."""
    a, b = min(x), max(x)
    return (x[row] - a) * (d - c) / (b - a) + c


def alo_roulette_weights(fitness):
    """1/fitness (minimization), with a floor-shift (f - min(f) + 1) applied
    ONLY when some fitness is <= 0 -- keeps the reciprocal well-defined and
    monotonic (sezgi's raw fitness can be negative; ALO.m's own literal
    1./fitness formula assumes it never is)."""
    min_f = min(fitness)
    if min_f > 0.0:
        return [1.0 / f for f in fitness]
    shift = -min_f + 1.0
    return [1.0 / (f + shift) for f in fitness]


def alo_roulette_select(weights, u):
    total = sum(weights)
    target = u * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if acc > target:
            return i
    return 0  # fallback, matching ALO.m's chosen_index==-1 -> index 1 (1-based)


def alo_walk_around(dim, t_max, row, lo, hi, i_ratio, antlion, rng):
    lb_scaled = lo / i_ratio
    ub_scaled = hi / i_ratio
    use_plus_lo = rng.random() < 0.5
    use_plus_hi = rng.random() >= 0.5
    out = []
    for d in range(dim):
        c = alo_shift_bound(lb_scaled, antlion[d], use_plus_lo)
        dd = alo_shift_bound(ub_scaled, antlion[d], use_plus_hi)
        steps = [rng.random() > 0.5 for _ in range(t_max)]
        x = alo_cumsum_walk(steps)
        out.append(alo_normalize_walk(x, row, c, dd))
    return out


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class Alo(sezgi.Algorithm):
    name = "alo"

    def setup(self, ctx):
        self.pop = [ctx.random_point() for _ in range(POP_SIZE)]
        self.fitness = ctx.evaluate(self.pop)

        # t_max = ALO.m's Max_iter, derived once from (budget, pop_size) --
        # both constant across the run, so this is identical every call.
        self.t_max = max(1, BUDGET // POP_SIZE)
        self.generation = 0  # ALO.m's Current_iter = generation + 2 (1-based, iteration 1 = setup)

    def step(self, ctx):
        lo, hi = ctx.bounds
        current_iter = self.generation + 2
        progress = current_iter / self.t_max
        i_ratio = alo_i_ratio(progress)
        row = min(self.generation + 1, self.t_max)

        elite_idx = min(range(POP_SIZE), key=lambda i: (self.fitness[i], i))
        elite = self.pop[elite_idx]
        weights = alo_roulette_weights(self.fitness)

        offspring = []
        for _ in range(POP_SIZE):
            u_sel = ctx.rng.random()
            sel_idx = alo_roulette_select(weights, u_sel)
            selected = self.pop[sel_idx]

            ra = alo_walk_around(DIM, self.t_max, row, lo, hi, i_ratio, selected, ctx.rng)
            re = alo_walk_around(DIM, self.t_max, row, lo, hi, i_ratio, elite, ctx.rng)
            xs = [clamp((ra[d] + re[d]) / 2.0, lo, hi) for d in range(DIM)]
            offspring.append(xs)

        new_fitness = ctx.evaluate(offspring)

        # replace/mu-plus-lambda: pop first, offspring second, stable sort
        # ascending by fitness, truncate to POP_SIZE.
        combined = list(zip(self.fitness, self.pop)) + list(zip(new_fitness, offspring))
        combined.sort(key=lambda item: item[0])
        combined = combined[:POP_SIZE]
        self.fitness = [f for f, _ in combined]
        self.pop = [p for _, p in combined]

        self.generation += 1


def main():
    res = Alo().solve(sezgi.bbob(1, DIM, 1), budget=BUDGET, seed=SEED)
    print(f"alo (oop): evals_used={res.evals_used} best_f={res.best_f:.6g} "
          f"gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
