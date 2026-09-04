"""OOP example: a custom `sezgi.LocalSearch` variant (M4-1 Task 4) --
overrides ONLY `neighbor()`, keeping the base's default greedy `accept()`.
Same new-style precedent as `custom_de_variant.py`: authored directly
against the engine-hosted family base, no pure-script twin, so it sits
OUTSIDE `test_examples_oop_parity.py`'s 17-pair gate; its own gate is
`py-sezgi/tests/test_oop_families.py`.
"""
import sezgi


class CustomPerturbationSearch(sezgi.LocalSearch):
    """neighbor(): perturbs every coordinate of x by an independent
    uniform draw in [-step, step] via ctx.rng.next_f64(). The base's
    default greedy accept() (f_new <= f_old) tracks whichever of
    {current, neighbor} the engine's own replace/mu-plus-lambda replacer
    already keeps at pop_size=1 -- see sezgi.LocalSearch's own class
    docstring (ACCEPT() DESIGN) for exactly what accept() does and does
    not control here. `step`: perturbation half-width."""

    def __init__(self, step=0.3):
        self.step = step

    def neighbor(self, x, ctx):
        return [xi + (ctx.rng.next_f64() - 0.5) * 2 * self.step for xi in x]


def main():
    res = CustomPerturbationSearch().run(sezgi.bbob(1, 10, 1), budget=2000, seed=42)
    print(f"custom_local_search (oop): evals_used={res.evals_used} "
          f"best_f={res.best_f:.6g} gap={res.gap:.6g}")


if __name__ == "__main__":
    main()
