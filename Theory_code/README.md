# Optimal sample selection for a misclassification-gated update

Code reproducing the figures. Stream-based active learning with the Adatron
update, in two settings: minimizing the final generalization error under a
query budget, and minimizing the cumulative cost when each label has a price.

Requires numpy, scipy, matplotlib and jax (CPU is fine).

## Layout

    adatron.py              model: moments, ODEs, policies, finite-d simulator
    optimize.py             the six policy families, solved in both settings
    plotting.py             figure style and drawing helpers
    01_concentration.ipynb  ODEs against a single finite-d run
    02_budget.ipynb         budget setting
    03_cost.ipynb           cost setting
    04_appendix.ipynb       selected sets across the constraint range
    figures/                the output PDFs
    results/                cached .npz, so figures redraw without re-solving

Each notebook has a `RECOMPUTE` flag. With `RECOMPUTE = False` and the cached
files present the figures redraw in seconds. Setting it to `True`, or deleting
`results/`, reruns everything.

`Params` holds the defaults used throughout: learning rate 0.5, horizon 20, and
a student starting orthogonal to the teacher, so the error begins at 1/2.

## The policy families

| key    | policy                                             |
|--------|----------------------------------------------------|
| `local`| myopic rule                                        |
| `cmar` | inner margin, \|w\| < kappa, constant              |
| `smar` | inner margin, \|w\| < kappa_t, schedule            |
| `cann` | intermediate margin, a < \|w\| < b, constant       |
| `sann` | intermediate margin, a_t < \|w\| < b_t, schedule   |
| `rand` | query at a constant probability, ignoring the margin |

Each is a policy object in `adatron.py`: `Margin`, `Annulus`, `MarginSchedule`,
`Schedule`, `LocalBudget`, `LocalCost`, `Random`. The deterministic families are
nested, since a constant policy is a flat schedule and an inner margin is the
a = 0 intermediate margin. The sweeps use that twice: a smaller family's
solution seeds the larger ones, and no family is reported worse than one it
contains.

## Two things to know about the numerics

**Quadrature.** Every deterministic policy is a union of intervals in |w| and
the integrands are even, so each expectation becomes
2 * int_a^b dw P(w) (.), evaluated by Gauss-Legendre mapped onto [a, b]. The
integrand is analytic, so eight nodes already match forty-eight to 5e-16. More
importantly, the policy enters through the integration limits rather than
through the integrand, which is what makes the objective differentiable in the
radii despite the policy being a hard indicator. On a fixed grid in w the
indicator would make it piecewise constant, with no usable gradient.

**One set, two views.** A policy exposes its selected set both as intervals,
for the quadrature, and as a pointwise test, for the simulator. The same object
is passed to `integrate` and to `simulate`, so theory and simulation cannot end
up comparing different sets.

## Reported values

Schedules are optimized by L-BFGS-B on exact JAX gradients, from several
starts, so the reported values are the best local optimum found. Under the
budget the constraint is enforced by a quadratic penalty during optimization and
then exactly, by rescaling the interval widths by one scalar, so every reported
value is attained by a policy within 2e-4 of the nominal budget.

Simulations are one SGD run per point at d = 1000, with no averaging and no
error bars, so the scatter is the real finite-d correction.

## Runtime

From scratch, on a laptop CPU: about 10 s, 22 min, 15 min and 3 min for the
four notebooks. Dominated by the schedule optimizations and by the myopic rule,
whose selected set is located by root finding at every Euler step.
