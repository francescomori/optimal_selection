"""
Optimal selection policies, for two objectives.

    budget   minimize eps_g(t_F) subject to a cap gamma on the query rate
    cost     minimize R(t_F) = Delta Q + M, unconstrained, where a query costs
             Delta and an unqueried mistake costs 1

Five families in each, keyed as in the saved results and the figure legends:

    local   myopic rule
    cmar    inner margin         |omega| < kappa,       constant
    smar    inner margin         |omega| < kappa_t,     schedule
    cann    intermediate margin  a < |omega| < b,       constant
    sann    intermediate margin  a_t < |omega| < b_t,   schedule
    rand    query at a constant probability, ignoring the margin

Constant policies are solved by root finding or bounded scalar minimization.
Schedules are discretized on N_CTRL uniform blocks and optimized by L-BFGS-B on
exact JAX gradients.  

"""
import numpy as np
from dataclasses import replace

from scipy.optimize import minimize, minimize_scalar
import jax
import jax.numpy as jnp

import adatron as A

jax.config.update("jax_enable_x64", True)

N_CTRL = 50         # control points per schedule
N_OPT = 500         # Euler steps inside the optimizer (N_CTRL must divide it)
RHO = 300.0         # weight of the quadratic budget penalty
OUTER_MAX = 10.0    # stand-in for an unbounded outer radius
KEYS = ('local', 'cmar', 'smar', 'cann', 'sann', 'rand')

# Each family contains the ones listed after it, so the optimum of a smaller
# family is a feasible point of the larger one: a constant policy is a flat
# schedule, and an inner margin is the a = 0 intermediate margin.  Reported
# values are made to respect this, and the smaller family's solution is offered
# to the larger one as a starting point.
CONTAINS = {'smar': ('cmar',), 'cann': ('cmar',), 'sann': ('cann', 'smar')}

# the JAX view of the model, so optimize.py and adatron.py share one set of
# formulas rather than keeping parallel copies
JAX = A.Backend(jnp, jax.scipy.special.ndtr)


# ------------------------------------------------------- differentiable ODE
def make_fwd(p: A.Params, n_ctrl=N_CTRL, family='annulus', n_steps=N_OPT):
    """Build fwd(par) -> (eps_g(t_F), query rate, Q, M), differentiable in par.

    par is the schedule, piecewise constant on n_ctrl uniform blocks:

        'margin'    par = kappa_t      (n_ctrl,)    support [0, kappa_t]
        'annulus'   par = [a_t; b_t]   (2 n_ctrl,)  support [a_t, b_t]

    The quadrature nodes sit at the interval endpoints, so the endpoints are
    ordinary differentiable parameters and autodiff through lax.scan is exact.

    In:  a Params, the number of control blocks, the family, and the number of
         Euler steps, which n_ctrl must divide.
    Out: the function fwd, mapping a parameter array of shape (n_ctrl,) or
         (2 n_ctrl,) to four JAX scalars (eps_g(t_F), query rate, Q, M).
    """
    if n_steps % n_ctrl:
        raise ValueError('n_ctrl must divide n_steps')
    dt = p.t_F / n_steps
    per_block = n_steps // n_ctrl

    def terms(edges, m, q):
        """Everything one Euler step needs, at one state and one interval.

        `edges` is the pair (lower, upper) radius in effect at this step.  The
        two radii are independent parameters, so the optimizer can propose an
        inverted or empty interval; clamping the upper one keeps the quadrature
        well posed, at the cost of a kink that L-BFGS-B handles through its
        bounds rather than through smoothness.

        The last return is the *unqueried* mistake rate, eps_g minus the
        queried part, matching what adatron.integrate accumulates into M.

        In:  edges, the (lower, upper) pair for this step, and the state m, q.
        Out: four scalars, (dm/dt, dq/dt, E[p], E[(1-p) 1_mistake]).
        """
        lo = edges[0]
        hi = jnp.maximum(edges[1], lo + 1e-9)
        dm, dq, rate, hit = A.segment_terms(lo, hi, m, q, p.eta, bk=JAX)
        return dm, dq, rate, A.gen_error(m, q, JAX) - hit

    def step(state, edges):
        """One forward-Euler step, in the form lax.scan wants.

        The carry is the state (m, q) and the per-step output is the pair of
        integrands.  Both are evaluated *before* the update, so the emitted
        sequence sits at t = 0, dt, ..., t_F - dt, which is what the endpoint
        correction in fwd then has to repair.

        In:  the carry (m, q) and this step's edges.
        Out: ((m, q) after the step, the two integrands before it).
        """
        m, q = state
        dm, dq, rate, missed = terms(edges, m, q)
        return (m + dt * dm, q + dt * dq), (rate, missed)

    def fwd(par):
        # expand the control vector to one (lower, upper) pair per Euler step:
        # each block's radii are repeated over the steps that fall inside it.
        # An inner margin is the same thing with the lower radius pinned to 0,
        # which is also how adatron.MarginSchedule represents it.
        """In:  the control vector, shape (n_ctrl,) or (2 n_ctrl,).
        Out: four JAX scalars, (eps_g(t_F), query rate, Q, M)."""
        if family == 'margin':
            edges = jnp.stack([jnp.zeros(n_steps),
                               jnp.repeat(par, per_block)], axis=1)
        else:
            edges = jnp.stack([jnp.repeat(par[:n_ctrl], per_block),
                               jnp.repeat(par[n_ctrl:], per_block)], axis=1)

        # integrate the trajectory, collecting the two integrands as we go.
        # Reverse-mode autodiff runs back through this scan, which is where the
        # gradients with respect to the radii come from: they enter through the
        # quadrature node positions, so no relaxation of the indicator is used.
        (m_F, q_F), (rate, missed) = jax.lax.scan(
            step, (jnp.float64(p.m0), jnp.float64(p.q0)), edges)

        # the scan left us a left Riemann sum, with samples at t = 0 to
        # t_F - dt.  The trapezoid rule wants half weight at each end plus the
        # missing sample at t_F, which is dt * (sum + (f(t_F) - f(0)) / 2).
        # Evaluating at edges[-1] is right because a piecewise-constant
        # schedule holds its last block's radii at t_F, exactly as
        # adatron.Schedule.set_at does.  Matching the rule matters: with a left
        # Riemann sum the optimizer minimizes a functional up to 5e-3 away from
        # the one that gets reported.
        rate_F, missed_F = terms(edges[-1], m_F, q_F)[2:]
        Q = dt * (jnp.sum(rate) + 0.5 * (rate_F - rate[0]))
        M = dt * (jnp.sum(missed) + 0.5 * (missed_F - missed[0]))

        # Q / t_F is the time-averaged query rate, the quantity the budget
        # constraint caps, defined as in adatron.integrate
        return A.gen_error(m_F, q_F, JAX), Q / p.t_F, Q, M

    return fwd


def best_of_lbfgs(objective, starts, bounds):
    """Best of several L-BFGS-B runs on a jitted value-and-grad objective.

    In:  objective, a scalar JAX function of one parameter array.  starts, a
         list of initial arrays.  bounds, one (lo, hi) pair per parameter.
    Out: one parameter array, the best minimizer found.
    """
    value_grad = jax.jit(jax.value_and_grad(objective))

    def f(z):
        """In:  a parameter array.
        Out: (value, gradient) as a float and a numpy array, which is what
             scipy.optimize.minimize expects from jac=True."""
        value, grad = value_grad(jnp.asarray(z))
        return float(value), np.asarray(grad, float)

    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    best = None
    for z0 in starts:
        result = minimize(f, np.clip(np.asarray(z0, float), lo, hi), jac=True,
                          method='L-BFGS-B', bounds=bounds,
                          options={'maxiter': 800, 'ftol': 1e-14,
                                   'gtol': 1e-11})
        if best is None or result.fun < best.fun:
            best = result
    return best.x


# ============================================================ budget setting
def _penalized(fwd, gamma):
    """eps_g(t_F) plus a quadratic penalty pulling the query rate to gamma.

    In:  a fwd from make_fwd and the target budget gamma.
    Out: a scalar JAX function of the parameter array.
    """
    def objective(z):
        """In: the control vector.  Out: one JAX scalar."""
        eps, rate, _Q, _M = fwd(z)
        return eps + RHO * (rate - gamma) ** 2
    return objective


def solve_local_budget(p, gamma, tol=2e-4):
    """Bisect the shadow price nu so the realized rate equals gamma.  The rate
    decreases in nu, since a dearer query means a smaller selected set.

    In:  a Params and the budget gamma.
    Out: (nu, result), the shadow price and the integrate dict at that nu.
    """
    rate = lambda nu: A.integrate(A.LocalBudget(nu, p.eta), p)['budget']
    nu = 0.0
    if rate(0.0) > gamma + tol:
        lo, hi = 0.0, 1.0
        while rate(hi) > gamma:
            hi *= 2.0
        for _ in range(40):
            nu = 0.5 * (lo + hi)
            realized = rate(nu)
            if abs(realized - gamma) < tol:
                break
            lo, hi = (nu, hi) if realized > gamma else (lo, nu)
    return nu, A.integrate(A.LocalBudget(nu, p.eta), p)


def solve_cmar_budget(p, gamma):
    """Constant inner margin.  One parameter, and the budget pins it, so there
    is nothing left to optimize.

    In:  a Params and the budget gamma.
    Out: (kappa, result), the radius and its integrate dict.
    """
    rate = lambda kappa: A.integrate(A.Margin(kappa), p)['budget']
    kappa = A.bisect(rate, gamma, 1e-4, OUTER_MAX)
    return kappa, A.integrate(A.Margin(kappa), p)


def solve_cann_budget(p, gamma, a_tol=2e-4, search_steps=400):
    """Constant intermediate margin: minimize over the inner radius, with the
    outer one pinned by the budget at each inner radius.

    Only inner radii below a feasibility limit can reach the budget at all:
    past it, even an unbounded outer radius queries less than gamma.  Searching
    inside that limit is what makes this work, since the limit is 0.34 at
    gamma = 0.2 and 0.04 at gamma = 0.9, so a fixed grid over a wide range of a
    puts almost no points in the feasible set and at large budgets can miss it
    entirely, collapsing this family onto a plain inner margin.

    The search runs on a coarser Euler grid, which does not move the optimal
    inner radius measurably; the outer radius and the reported value are
    recomputed at full resolution.

    In:  a Params and the budget gamma.
    Out: (a, b, result), the radii and the integrate dict at full resolution.
    """
    coarse = replace(p, n_steps=search_steps)
    rate = lambda a, b, pp: A.integrate(A.Annulus(a, b), pp)['budget']
    pin_outer = lambda a, pp: A.bisect(lambda b: rate(a, b, pp), gamma,
                                       a + 1e-9, OUTER_MAX)
    # the widest interval available at inner radius a is (a, OUTER_MAX), so the
    # feasibility limit is where even that hits the budget exactly
    a_max = A.bisect(lambda a: -rate(a, OUTER_MAX, coarse), -gamma,
                     0.0, OUTER_MAX)

    def eps_at(a):
        """In:  the inner radius a.
        Out: eps_g(t_F) with the outer radius pinned by the budget."""
        return A.integrate(A.Annulus(a, pin_outer(a, coarse)), coarse)['eps']

    if a_max <= a_tol:                       # budget so large that a must be 0
        a = 0.0
    else:
        found = minimize_scalar(eps_at, bounds=(0.0, a_max - a_tol),
                                method='bounded', options={'xatol': a_tol})
        a = float(found.x)
    b = pin_outer(a, p)
    return a, b, A.integrate(A.Annulus(a, b), p)


def solve_rand_budget(p, gamma):
    """Random benchmark.  Querying every sample with probability gamma spends
    exactly the budget, so there is nothing to optimize.

    In:  a Params and the budget gamma.
    Out: (gamma, result).
    """
    return gamma, A.integrate(A.Random(gamma), p)


def _pin_budget(build, p, gamma, tol=2e-4):
    """Rescale a schedule's interval widths by one scalar so the realized rate
    equals gamma.

    The optimizer enforces the budget only by a quadratic penalty, which leaves
    the rate off by a few times 1e-3.  Rescaling restores the constraint
    exactly, and the reported value is recomputed for the rescaled policy, so
    it stays an upper bound.

    In:  build, a function from a width scale to a policy; a Params; gamma.
    Out: (scale, result), the scalar applied and the integrate dict after it.
    """
    realized = lambda s: A.integrate(build(s), p)['budget']
    if abs(realized(1.0) - gamma) < tol:
        return 1.0, A.integrate(build(1.0), p)
    hi = 1.0
    while realized(hi) < gamma and hi < 128.0:
        hi *= 2.0
    s = A.bisect(realized, gamma, 0.0, hi)
    return s, A.integrate(build(s), p)


def solve_smar_budget(p, gamma, kappa0, warm=None):
    """Inner-margin schedule under the budget.

    In:  a Params, gamma, a starting radius kappa0, and optionally a warm
         start of shape (N_CTRL,).
    Out: (kappa_t, result), the schedule of shape (N_CTRL,) and its integrate
         dict.
    """
    objective = _penalized(make_fwd(p, family='margin'), gamma)
    frac = np.linspace(0.0, 1.0, N_CTRL)
    starts = [np.full(N_CTRL, kappa0),                      # flat
              np.full(N_CTRL, kappa0) * (2.0 - 1.6 * frac),  # decreasing
              np.where(frac > 0.5, 2.0, 1e-4),               # defer then query
              2.0 * np.exp(-3.0 * frac) + 0.15]              # front-loaded
    if warm is not None:
        starts.append(warm)
    kappa = best_of_lbfgs(objective, starts, [(1e-4, OUTER_MAX)] * N_CTRL)
    s, result = _pin_budget(lambda s: A.MarginSchedule(s * kappa, p.t_F),
                            p, gamma)
    return s * kappa, result


def solve_sann_budget(p, gamma, a0, b0, kappa_sched=None, warm=None):
    """Intermediate-margin schedule under the budget.

    Started from the constant intermediate margin, from the inner-margin
    schedule embedded at a_t = 0, and from the previous point of the sweep.

    In:  a Params, gamma, starting radii a0 and b0, optionally the inner-margin
         schedule, and optionally a warm start of shape (2 N_CTRL,).
    Out: (a_t, b_t, result), two arrays of shape (N_CTRL,) and the integrate
         dict.
    """
    objective = _penalized(make_fwd(p, family='annulus'), gamma)
    starts = [np.r_[np.full(N_CTRL, a0), np.full(N_CTRL, b0)]]
    if kappa_sched is not None:
        starts.append(np.r_[np.zeros(N_CTRL), kappa_sched])
    if warm is not None:
        starts.append(warm)
    bounds = [(0.0, 3.0)] * N_CTRL + [(1e-4, OUTER_MAX)] * N_CTRL
    z = best_of_lbfgs(objective, starts, bounds)
    a = z[:N_CTRL]
    width = np.maximum(z[N_CTRL:] - a, 1e-9)
    s, result = _pin_budget(lambda s: A.Schedule(a, a + s * width, p.t_F),
                            p, gamma)
    return a, a + s * width, result


def sweep_budget(p, gammas, verbose=True):
    """Solve all five families at each budget, from the largest down, warm-
    starting each schedule from the previous solution so that the schedules
    stay in the same basin along the curve.

    In:  a Params and the array of budgets.
    Out: one dict ready for np.savez: 'gammas', 'nu', 'kappa', 'a_const',
         'b_const' of shape (n,), 'kap_sched', 'a_sched', 'b_sched' of shape
         (n, N_CTRL), and 'eps_<key>' and 'bud_<key>' of shape (n,) per family.
    """
    gammas = np.asarray(gammas, float)
    n = len(gammas)
    eps = {k: np.zeros(n) for k in KEYS}
    realized = {k: np.zeros(n) for k in KEYS}
    nu = np.zeros(n)
    kappa = np.zeros(n)
    p_rand = np.zeros(n)
    a_const, b_const = np.zeros(n), np.zeros(n)
    kappa_sched = np.zeros((n, N_CTRL))
    a_sched, b_sched = np.zeros((n, N_CTRL)), np.zeros((n, N_CTRL))
    warm_margin = warm_annulus = None

    for i in range(n - 1, -1, -1):
        gamma = float(gammas[i])

        nu[i], r = solve_local_budget(p, gamma)
        eps['local'][i], realized['local'][i] = r['eps'], r['budget']

        kappa[i], r = solve_cmar_budget(p, gamma)
        eps['cmar'][i], realized['cmar'][i] = r['eps'], r['budget']

        p_rand[i], r = solve_rand_budget(p, gamma)
        eps['rand'][i], realized['rand'][i] = r['eps'], r['budget']

        a_const[i], b_const[i], r = solve_cann_budget(p, gamma)
        eps['cann'][i], realized['cann'][i] = r['eps'], r['budget']

        kappa_sched[i], r = solve_smar_budget(p, gamma, kappa[i], warm_margin)
        eps['smar'][i], realized['smar'][i] = r['eps'], r['budget']
        if eps['smar'][i] > eps['cmar'][i]:          # smar contains cmar
            eps['smar'][i] = eps['cmar'][i]
            realized['smar'][i] = realized['cmar'][i]
            kappa_sched[i] = np.full(N_CTRL, kappa[i])
        warm_margin = kappa_sched[i]

        a_sched[i], b_sched[i], r = solve_sann_budget(
            p, gamma, a_const[i], b_const[i], kappa_sched[i], warm_annulus)
        eps['sann'][i], realized['sann'][i] = r['eps'], r['budget']
        # sann contains both cann and smar, so it may fall back to either
        for smaller in CONTAINS['sann']:
            if eps['sann'][i] > eps[smaller][i]:
                eps['sann'][i] = eps[smaller][i]
                realized['sann'][i] = realized[smaller][i]
                if smaller == 'cann':
                    a_sched[i] = np.full(N_CTRL, a_const[i])
                    b_sched[i] = np.full(N_CTRL, b_const[i])
                else:
                    a_sched[i] = np.zeros(N_CTRL)
                    b_sched[i] = kappa_sched[i]
        warm_annulus = np.r_[a_sched[i], b_sched[i]]

        if verbose:
            print('gamma=%.2f  ' % gamma
                  + '  '.join('%s %.4f' % (k, eps[k][i]) for k in KEYS),
                  flush=True)

    out = {'gammas': gammas, 'nu': nu, 'kappa': kappa, 'p_rand': p_rand,
           'a_const': a_const, 'b_const': b_const,
           'kap_sched': kappa_sched, 'a_sched': a_sched, 'b_sched': b_sched}
    out.update({'eps_' + k: eps[k] for k in KEYS})
    out.update({'bud_' + k: realized[k] for k in KEYS})
    return out


# ============================================================== cost setting
def _cost(fwd, delta):
    """R(t_F) = Delta Q + M, straight from the forward pass.

    In:  a fwd from make_fwd and the query price delta.
    Out: a scalar JAX function of the parameter array.
    """
    def objective(z):
        """In: the control vector.  Out: one JAX scalar."""
        _eps, _rate, Q, M = fwd(z)
        return delta * Q + M
    return objective


def solve_cmar_cost(p, delta):
    """Constant inner margin.  R(kappa) flattens for large kappa, so grid first
    and refine inside the best bracket.

    In:  a Params and the query price delta.
    Out: (kappa, result), the radius and its integrate dict.
    """
    cost = lambda kappa: A.integrate(A.Margin(kappa), p, delta=delta)['R']
    grid = np.geomspace(0.02, 5.0, 60)
    values = [cost(k) for k in grid]
    j = int(np.argmin(values))
    refined = minimize_scalar(
        cost, bounds=(grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]),
        method='bounded', options={'xatol': 1e-6})
    kappa = refined.x if refined.fun < values[j] else grid[j]
    return kappa, A.integrate(A.Margin(kappa), p, delta=delta)


def solve_cann_cost(p, delta, kappa0):
    """Constant intermediate margin, over both radii.  Nelder-Mead on (inner
    radius, width) from three starts.  The inner radius is left free rather
    than assumed: that it comes back at zero is the result.

    In:  a Params, delta, and a starting radius kappa0.
    Out: (a, b, result), the two radii and the integrate dict.
    """
    cost = lambda z: A.integrate(
        A.Annulus(abs(z[0]), abs(z[0]) + abs(z[1])), p, delta=delta)['R']
    best = None
    for a0 in (0.0, 0.1, 0.4):
        found = minimize(cost, [a0, kappa0], method='Nelder-Mead',
                         options={'xatol': 1e-5, 'fatol': 1e-10,
                                  'maxiter': 800})
        if best is None or found.fun < best.fun:
            best = found
    a, width = abs(best.x[0]), abs(best.x[1])
    return a, a + width, A.integrate(A.Annulus(a, a + width), p, delta=delta)


def solve_rand_cost(p, delta):
    """Random benchmark: the one query probability that minimizes the cost.

    Grid first and then refine, as for the constant margin, since the cost is
    flat in the probability over part of the range.

    In:  a Params and the query price delta.
    Out: (p_query, result).
    """
    cost = lambda pr: A.integrate(A.Random(pr), p, delta=delta)['R']
    grid = np.linspace(0.02, 1.0, 50)
    values = [cost(pr) for pr in grid]
    j = int(np.argmin(values))
    refined = minimize_scalar(
        cost, bounds=(grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]),
        method='bounded', options={'xatol': 1e-6})
    pr = refined.x if refined.fun < values[j] else grid[j]
    return pr, A.integrate(A.Random(pr), p, delta=delta)


def local_cost_kappa(p, delta, n=N_CTRL):
    """kappa_t of the myopic cost rule, on a control grid.  Used to
    warm-start the schedules, since at small Delta the constant policies sit on
    a flat stretch of R and make a poor starting point.

    In:  a Params, delta, and the number of grid points.
    Out: one array of shape (n,), the radius on a uniform grid of [0, t_F).
    """
    policy = A.LocalCost(delta)
    traj = A.integrate(policy, p, delta=delta, trace=True)
    t = np.linspace(0.0, p.t_F, n, endpoint=False)
    m = np.interp(t, traj['t'], traj['m'])
    q = np.interp(t, traj['t'], traj['q'])
    return np.maximum(policy.kappa(m, q), 1e-4)


def solve_smar_cost(p, delta, kappa0, kappa_local, warm=None):
    """Inner-margin schedule for the cumulative cost.

    In:  a Params, delta, a starting radius, the myopic radii of shape
         (N_CTRL,), and optionally a warm start.
    Out: (kappa_t, result), the schedule and its integrate dict.
    """
    objective = _cost(make_fwd(p, family='margin'), delta)
    starts = [kappa_local, np.full(N_CTRL, kappa0)]
    if warm is not None:
        starts.append(warm)
    kappa = best_of_lbfgs(objective, starts, [(1e-4, OUTER_MAX)] * N_CTRL)
    return kappa, A.integrate(A.MarginSchedule(kappa, p.t_F), p, delta=delta)


def solve_sann_cost(p, delta, a0, b0, kappa_local, kappa_sched=None,
                    warm=None):
    """Intermediate-margin schedule for the cumulative cost.

    In:  a Params, delta, starting radii a0 and b0, the myopic radii,
         optionally the inner-margin schedule, and optionally a warm start.
    Out: (a_t, b_t, result), two arrays of shape (N_CTRL,) and the integrate
         dict.
    """
    objective = _cost(make_fwd(p, family='annulus'), delta)
    starts = [np.r_[np.zeros(N_CTRL), kappa_local],
              np.r_[np.full(N_CTRL, a0), np.full(N_CTRL, b0)]]
    if kappa_sched is not None:
        starts.append(np.r_[np.zeros(N_CTRL), kappa_sched])
    if warm is not None:
        starts.append(warm)
    bounds = [(0.0, 3.0)] * N_CTRL + [(1e-4, OUTER_MAX)] * N_CTRL
    z = best_of_lbfgs(objective, starts, bounds)
    a = z[:N_CTRL]
    b = np.maximum(z[N_CTRL:], a + 1e-9)
    return a, b, A.integrate(A.Schedule(a, b, p.t_F), p, delta=delta)


def sweep_cost(p, deltas, verbose=True):
    """Solve all five families at each query price.

    In:  a Params and the array of query prices.
    Out: one dict ready for np.savez: 'deltas', 'kappa', 'a_const', 'b_const' of
         shape (n,), 'kap_sched', 'a_sched', 'b_sched' of shape (n, N_CTRL), and
         'R_<key>' of shape (n,) per family.
    """
    deltas = np.asarray(deltas, float)
    n = len(deltas)
    cost = {k: np.zeros(n) for k in KEYS}
    kappa = np.zeros(n)
    p_rand = np.zeros(n)
    a_const, b_const = np.zeros(n), np.zeros(n)
    kappa_sched = np.zeros((n, N_CTRL))
    a_sched, b_sched = np.zeros((n, N_CTRL)), np.zeros((n, N_CTRL))
    warm_margin = warm_annulus = None

    for i in range(n):
        delta = float(deltas[i])
        kappa_local = local_cost_kappa(p, delta)

        cost['local'][i] = A.integrate(A.LocalCost(delta), p,
                                       delta=delta)['R']

        kappa[i], r = solve_cmar_cost(p, delta)
        cost['cmar'][i] = r['R']

        p_rand[i], r = solve_rand_cost(p, delta)
        cost['rand'][i] = r['R']

        a_const[i], b_const[i], r = solve_cann_cost(p, delta, kappa[i])
        cost['cann'][i] = r['R']

        if cost['cann'][i] > cost['cmar'][i]:        # cann contains cmar
            cost['cann'][i] = cost['cmar'][i]
            a_const[i], b_const[i] = 0.0, kappa[i]

        kappa_sched[i], r = solve_smar_cost(p, delta, kappa[i],
                                            kappa_local, warm_margin)
        cost['smar'][i] = r['R']
        if cost['smar'][i] > cost['cmar'][i]:        # smar contains cmar
            cost['smar'][i] = cost['cmar'][i]
            kappa_sched[i] = np.full(N_CTRL, kappa[i])
        warm_margin = kappa_sched[i]

        a_sched[i], b_sched[i], r = solve_sann_cost(
            p, delta, a_const[i], b_const[i], kappa_local, kappa_sched[i],
            warm_annulus)
        cost['sann'][i] = r['R']
        # sann contains both cann and smar, so it may fall back to either
        for smaller in CONTAINS['sann']:
            if cost['sann'][i] > cost[smaller][i]:
                cost['sann'][i] = cost[smaller][i]
                if smaller == 'cann':
                    a_sched[i] = np.full(N_CTRL, a_const[i])
                    b_sched[i] = np.full(N_CTRL, b_const[i])
                else:
                    a_sched[i] = np.zeros(N_CTRL)
                    b_sched[i] = kappa_sched[i]
        warm_annulus = np.r_[a_sched[i], b_sched[i]]

        if verbose:
            print('Delta=%.2f  ' % delta
                  + '  '.join('%s %.3f' % (k, cost[k][i]) for k in KEYS),
                  flush=True)

    out = {'deltas': deltas, 'kappa': kappa, 'p_rand': p_rand,
           'a_const': a_const,
           'b_const': b_const, 'kap_sched': kappa_sched,
           'a_sched': a_sched, 'b_sched': b_sched}
    out.update({'R_' + k: cost[k] for k in KEYS})
    return out


# ------------------------------------------------- data for the selection maps
def local_support_trace(p, nu, n_rec=400):
    """Integrate the myopic budget rule, recording its selected set as
    (t, lower edge, upper edge).  The support is a single interval throughout,
    so one pair of edges describes it, and entries are NaN when nothing is
    selected.

    In:  a Params, the shadow price nu, and the number of recorded times.
    Out: three arrays of shape (n_rec + 1,): t, and the two edges, NaN where
         nothing is selected.
    """
    policy = A.LocalBudget(nu, p.eta)
    dt = p.t_F / n_rec
    m, q = p.m0, p.q0
    ts, lower, upper = [], [], []

    for k in range(n_rec + 1):
        intervals = policy.intervals(m, q, k * dt)
        ts.append(k * dt)
        lower.append(intervals[0][0] if intervals else np.nan)
        upper.append(intervals[0][1] if intervals else np.nan)
        if k == n_rec:
            break
        dm, dq, _rate, _hit = A.drift(intervals, m, q, p.eta)
        m, q = m + dt * dm, q + dt * dq

    return np.array(ts), np.array(lower), np.array(upper)


def refine_schedule(p, coarse_a, coarse_b, objective, n_fine=200,
                    smooth=3e-3, n_steps=600, gamma=None, delta=None):
    """Re-optimize a schedule on a finer control grid, for the selection maps.

    With 2 n_fine free parameters the raw optimum is visibly jittery at early
    times while costing the same objective to about 1e-5, so a light penalty on
    the squared differences of the radii picks the smooth representative of that
    near-degenerate family.  `objective` is 'budget' or 'cost'.  Pass
    coarse_b = None for an inner-margin schedule, which returns (kappa_t, None).

    In:  a Params, the coarse radii, 'budget' or 'cost', the fine grid size, the
         smoothing weight, the Euler steps, and whichever of gamma or delta the
         objective needs.
    Out: (a_t, b_t) of shape (n_fine,), or (kappa_t, None) for an inner margin.
    """
    is_margin = coarse_b is None
    family = 'margin' if is_margin else 'annulus'
    per_block = n_fine // len(coarse_a)
    fwd = make_fwd(p, n_ctrl=n_fine, family=family, n_steps=n_steps)

    def penalized(z):
        """In: the control vector.  Out: one JAX scalar, objective plus penalty."""
        eps, rate, Q, M = fwd(z)
        base = (eps + RHO * (rate - gamma) ** 2 if objective == 'budget'
                else delta * Q + M)
        jitter = jnp.sum(jnp.diff(z[:n_fine]) ** 2)
        if not is_margin:
            jitter = jitter + jnp.sum(jnp.diff(z[n_fine:]) ** 2)
        return base + smooth * jitter

    if is_margin:
        z0 = np.repeat(coarse_a, per_block)
        bounds = [(1e-4, OUTER_MAX)] * n_fine
    else:
        z0 = np.r_[np.repeat(coarse_a, per_block),
                   np.repeat(coarse_b, per_block)]
        bounds = [(0.0, 3.0)] * n_fine + [(1e-4, OUTER_MAX)] * n_fine
    z = best_of_lbfgs(penalized, [z0], bounds)

    if is_margin:
        kappa = z
        if objective == 'budget':
            s, _ = _pin_budget(lambda s: A.MarginSchedule(s * kappa, p.t_F),
                               p, gamma)
            kappa = s * kappa
        return kappa, None

    a = z[:n_fine]
    width = np.maximum(z[n_fine:] - a, 1e-9)
    if objective == 'budget':
        s, _ = _pin_budget(lambda s: A.Schedule(a, a + s * width, p.t_F),
                           p, gamma)
        width = s * width
    return a, a + width
