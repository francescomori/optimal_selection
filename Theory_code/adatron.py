"""
Stream-based active learning with a misclassification-gated (Adatron) update.

A student w in R^d learns a sign teacher beta from inputs x ~ N(0, I_d/d) with
||beta||^2 = d.  Writing omega = <w,x> for the student field and omega* =
<beta,x> for the teacher field, the learner sees omega, queries the label
y = sign(omega*) with probability p_t(omega), and on a query takes

    w <- w - eta * ell'(omega y) y x,        ell'(z) = min(z, 0),

so the update fires only on misclassified samples, with magnitude |omega|.  In
high dimension the dynamics closes on m = <w,beta>/d and q = ||w||^2/d at
t = i/d, and eps_g = arccos(m/sqrt(q))/pi.

Quadrature.  Every policy here is a union of intervals in |omega| and every
conditional moment is even in omega, so expectations are evaluated as
2 int_a^b dw P(w) (.) by Gauss-Legendre mapped onto [a,b].  The integrand is
analytic, so a few dozen nodes are exact, and the rule is differentiable in the
endpoints, which the schedule optimizers need.

Policies.  A policy gives its selected set both as intervals, for that
quadrature, and as a pointwise test, for the simulator, so the ODEs and the
simulation cannot end up comparing different sets.
"""
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, erfinv

SQRT_2PI = np.sqrt(2.0 * np.pi)
_trapz = getattr(np, 'trapezoid', None) or np.trapz

GL_NODES = 48       # Gauss-Legendre nodes per interval
OMEGA_MAX = 8.0     # outer cutoff for interval search, in units of sqrt(q)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(GL_NODES)


@dataclass(frozen=True)
class Params:
    """Model and discretization parameters."""
    eta: float = 0.5             # learning rate
    m0: float = 0.0              # initial alignment
    q0: float = 0.15             # initial squared norm
    t_F: float = 20.0            # horizon
    n_steps: int = 2000          # forward-Euler steps for reported values


# Array namespace and normal CDF, so the formulas below are written once and
# evaluated either with numpy, for everything reported, or with jax.numpy,
# inside the optimizers that differentiate through them.
Backend = namedtuple('Backend', 'xp ndtr')
NUMPY = Backend(np, ndtr)


# ------------------------------------------------------------------ moments
def interval_nodes(a, b, q, bk=NUMPY):
    """Nodes and weights for  E[f] = 2 int_a^b dw N(w; 0, q) f(w).

    The map from [-1,1] to [a,b] puts the Jacobian (b-a)/2 into the weights,
    along with the Gaussian density.  The factor 2 covers [-b,-a].

    In:  scalars a < b, the interval edges on omega > 0, and the variance q.
    Out: (nodes, weights), each of shape (GL_NODES,).  The weights already carry
         the density, the Jacobian and the factor 2.
    """
    xp = bk.xp
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    omega = mid + half * _GL_X
    density = xp.exp(-0.5 * omega * omega / q) / xp.sqrt(2.0 * np.pi * q)
    return omega, 2.0 * half * _GL_W * density


def update_moments(omega, m, q, bk=NUMPY):
    """E[g^2|omega], E[g omega|omega], E[g omega*|omega] for the update
    g = ell'(omega y) y.

    Given omega, omega* ~ N(mu, s^2) with mu = (m/q) omega and s^2 = 1 - m^2/q,
    so y = +1 with probability Phi(mu/s).  With ell'(z) = min(z,0) the update is
    min(omega,0) when y = +1 and -min(-omega,0) when y = -1.  The third moment
    uses E[X 1_{X>0}] = mu Phi(mu/s) + s phi(mu/s).

    In:  omega of any shape; m, q scalars, or arrays broadcasting with omega, as
         the simulator passes one pair per run.
    Out: three arrays shaped like omega.
    """
    xp = bk.xp
    s = xp.sqrt(xp.maximum(1.0 - m * m / q, 1e-12))
    mu = (m / q) * omega
    z = mu / s
    prob_pos, prob_neg = bk.ndtr(z), bk.ndtr(-z)
    phi = xp.exp(-0.5 * z * z) / SQRT_2PI
    g_pos, g_neg = xp.minimum(omega, 0.0), xp.minimum(-omega, 0.0)
    return (prob_pos * g_pos * g_pos + prob_neg * g_neg * g_neg,
            omega * (prob_pos * g_pos - prob_neg * g_neg),
            g_pos * (mu * prob_pos + s * phi)
            - g_neg * (mu * prob_neg - s * phi))


def mistake_prob(omega, m, q, bk=NUMPY):
    """P(sign(omega) != sign(omega*) | omega).

    Its omega-average is exactly eps_g, which is what lets the unqueried
    mistakes be counted as eps_g - E[p 1_mistake].

    In:  omega of any shape; m, q scalars or arrays broadcasting with it.
    Out: one array shaped like omega, valued in [0, 1].
    """
    xp = bk.xp
    s = xp.sqrt(xp.maximum(1.0 - m * m / q, 1e-12))
    return bk.ndtr(-(m / q) * xp.abs(omega) / s)


def gen_error(m, q, bk=NUMPY):
    """eps_g = arccos(m/sqrt(q))/pi.

    In:  m, q scalars or arrays.
    Out: same shape, valued in [0, 1/2] for m >= 0.
    """
    xp = bk.xp
    return xp.arccos(xp.clip(m / xp.sqrt(q), -1.0 + 1e-12, 1.0 - 1e-12)) / np.pi


def slope(omega, m, q, eta, bk=NUMPY):
    """S_t(omega): marginal effect on d eps_g/dt of querying mass at omega.

    Chain rule through the order parameters, using
    d eps_g/dm = -1/(pi sqrt(q-m^2)) and d eps_g/dq = m/(2 pi q sqrt(q-m^2)).
    S < 0 means the query helps, so the myopic rule queries where
    S + nu < 0 with nu the shadow price of a query.

    In:  omega of any shape; m, q, eta scalars.
    Out: array shaped like omega, in units of d eps_g/dt per unit of probability
         mass at omega.
    """
    xp = bk.xp
    Egg, Egw, Egws = update_moments(omega, m, q, bk)
    root = xp.sqrt(xp.maximum(q - m * m, 1e-12))
    d_dm = -1.0 / (np.pi * root)
    d_dq = m / (2.0 * np.pi * q * root)
    return d_dm * (-eta * Egws) + d_dq * (-2.0 * eta * Egw + eta ** 2 * Egg)


def segment_terms(a, b, m, q, eta, bk=NUMPY):
    """(dm/dt, dq/dt, E[p], E[p 1_mistake]) from one interval a < |omega| < b.

    The reporting integrator and the JAX forward pass both call this, so they
    integrate the same equations by construction.

    In:  scalars a < b, m, q, eta.
    Out: four scalars, the contribution of this interval alone.
    """
    omega, w = interval_nodes(a, b, q, bk)
    Egg, Egw, Egws = update_moments(omega, m, q, bk)
    return (-eta * (w @ Egws),
            eta * (-2.0 * (w @ Egw) + eta * (w @ Egg)),
            w.sum(),
            w @ mistake_prob(omega, m, q, bk))


def drift(intervals, m, q, eta, prob=1.0):
    """Sum segment_terms over a policy's intervals.

    `prob` is the probability of querying a sample that falls in the set, so
    every term scales with it.  It is 1 for the deterministic policies.

    In:  intervals, a list of (lo, hi) pairs; m, q, eta scalars; prob.
    Out: the same four scalars as segment_terms, summed over the list.
    """
    dm = dq = rate = hit = 0.0
    for a, b in intervals:
        s_dm, s_dq, s_rate, s_hit = segment_terms(a, b, m, q, eta)
        dm += s_dm
        dq += s_dq
        rate += s_rate
        hit += s_hit
    return prob * dm, prob * dq, prob * rate, prob * hit


# ----------------------------------------------------------------- policies
def _bisect_edge(accepts, x_in, x_out, n=40):
    """Locate an edge of the accepted set from a bracket with x_in accepted and
    x_out not.  Bisecting the boolean test works for any policy.

    In:  accepts, a boolean function of one scalar, and a bracket with x_in
         accepted and x_out not.
    Out: one scalar, the edge position.
    """
    for _ in range(n):
        mid = 0.5 * (x_in + x_out)
        if accepts(mid):
            x_in = mid
        else:
            x_out = mid
    return 0.5 * (x_in + x_out)


class Policy:
    """A selection policy, as two views of one set, symmetric in omega.

        intervals(m, q, t)       -> [(lo, hi), ...] on omega > 0, for the ODEs
        accepts(omega, m, q, t)  -> boolean array, for the simulator

    Intervals are given on omega > 0 only; the factor 2 lives in interval_nodes.

    Shapes: intervals returns a list of (lo, hi) pairs of scalars, empty when
    nothing is selected.  accepts returns a boolean array shaped like omega, and
    its m, q may be arrays of one entry per run.
    """

    uses_state = False   # does accepts() depend on (m, q)?
    prob = 1.0           # probability of querying a sample inside the set

    def intervals(self, m, q, t):
        """In: m, q, t scalars.  Out: a list of (lo, hi) pairs on omega > 0."""
        raise NotImplementedError

    def accepts(self, omega, m, q, t):
        """In: omega of any shape, m, q, t.  Out: a boolean array like omega."""
        raise NotImplementedError


class FixedPolicy(Policy):
    """Selected set depends on t alone.  Subclasses give set_at(t) and inherit
    the pointwise test from it.

    In (set_at): the time t.
    Out: a list of (lo, hi) pairs, empty when nothing is selected.
    """

    def set_at(self, t):
        """In: the time t.  Out: a list of (lo, hi) pairs on omega > 0."""
        raise NotImplementedError

    def intervals(self, m, q, t):
        """In: m, q (unused), t.  Out: set_at(t)."""
        return self.set_at(t)

    def accepts(self, omega, m, q, t):
        """In: omega, m, q (unused), t.  Out: a boolean array like omega."""
        a_omega = np.abs(omega)
        inside = np.zeros(np.shape(a_omega), dtype=bool)
        for lo, hi in self.set_at(t):
            inside |= (a_omega > lo) & (a_omega < hi)
        return inside


class StatePolicy(Policy):
    """Selected set moves with (m, q): the myopic rules.

    These write out both views separately, because the simulator needs the test
    at a different (m, q) per run while the quadrature needs endpoints at one
    (m, q).  The two have been checked to agree on a fine grid in omega along
    whole trajectories, for every policy here.
    """

    uses_state = True


class Margin(FixedPolicy):
    """Inner margin  |omega| < kappa.

    In: kappa, a scalar.
    """

    def __init__(self, kappa):
        self.kappa = float(kappa)

    def set_at(self, t):
        """In: t (unused).  Out: [(0, kappa)], empty if kappa is not positive."""
        return [(0.0, self.kappa)] if self.kappa > 0.0 else []


class Annulus(FixedPolicy):
    """Intermediate margin  a < |omega| < b.

    The inner hole exists because the update magnitude |omega| vanishes at the
    decision boundary: a label at omega = 0 is a coin flip and moves nothing.

    In: scalars a < b.
    """

    def __init__(self, a, b):
        self.a, self.b = float(a), float(b)

    def set_at(self, t):
        """In: t (unused).  Out: [(a, b)], empty if the interval is."""
        return [(self.a, self.b)] if self.b > self.a else []


class Schedule(FixedPolicy):
    """Intermediate margin a_t < |omega| < b_t, piecewise constant on len(a)
    uniform blocks of [0, t_F].  Pass a = 0 for an inner margin.

    In: a, b arrays of equal length, one pair per block, and the horizon t_F.
    """

    def __init__(self, a, b, t_F):
        self.a = np.asarray(a, float)
        self.b = np.asarray(b, float)
        self.block = t_F / len(self.a)

    def set_at(self, t):
        # the nudge keeps a t landing on a block boundary in the block it
        # starts, which floating-point division does not guarantee, and which
        # otherwise splits one Euler step between the ODEs and the optimizer
        """In: the time t.  Out: that block's [(a, b)], empty if the interval is."""
        k = min(int(t / self.block + 1e-9), len(self.a) - 1)
        return ([(self.a[k], self.b[k])] if self.b[k] > self.a[k] else [])


def MarginSchedule(kappa, t_F):
    """Inner margin |omega| < kappa_t, piecewise constant.

    In:  kappa, an array of one radius per block, and the horizon t_F.
    Out: a Schedule with a = 0, not a new class.
    """
    kappa = np.asarray(kappa, float)
    return Schedule(np.zeros_like(kappa), kappa, t_F)


class Random(Policy):
    """Query every sample with the same probability, whatever its margin.

    The benchmark with no margin information at all.  It is the only policy
    here that is stochastic rather than a hard indicator, which is why Policy
    carries `prob`.

    In: p, the query probability.
    """

    def __init__(self, p):
        self.prob = float(p)

    def intervals(self, m, q, t):
        """In: m, q, t.  Out: the whole support, as one pair."""
        return [(0.0, OMEGA_MAX * np.sqrt(q))]

    def accepts(self, omega, m, q, t):
        """In: omega, m, q, t.  Out: all true; the thinning by `prob` is done
        by the simulator, which owns the random stream."""
        return np.ones(np.shape(omega), dtype=bool)


class LocalBudget(StatePolicy):
    """Myopic rule under a budget: query where S_t(omega) + nu < 0.

    Minimizes the instantaneous d eps_g/dt + nu E[p], so it is myopic in time
    but unrestricted in omega.  Its support comes out as a single interval with
    a positive inner edge, which is what motivates optimizing over annuli.

    In: nu, the shadow price of a query, and the learning rate eta.
    """

    def __init__(self, nu, eta):
        self.nu, self.eta = float(nu), float(eta)

    def accepts(self, omega, m, q, t):
        """In: omega of any shape, m, q scalars or arrays, t (unused).
                Out: a boolean array like omega."""
        return slope(omega, m, q, self.eta) + self.nu < 0.0

    def intervals(self, m, q, t):
        """Located without a grid: S_t has one minimum on omega > 0, so the set
        is one interval around it, found by bisecting outward.

        In:  m, q, t scalars.
        Out: one (lo, hi) pair in a list, empty once nothing clears nu.
        """
        outer = OMEGA_MAX * np.sqrt(q)
        shifted = lambda w: (float(slope(np.array([w]), m, q, self.eta)[0])
                             + self.nu)
        deepest = minimize_scalar(shifted, bounds=(1e-12, outer),
                                  method='bounded', options={'xatol': 1e-13})
        if deepest.fun >= 0.0:            # nothing is worth its price
            return []
        selected = lambda w: shifted(w) < 0.0
        lo = (1e-12 if selected(1e-12)
              else _bisect_edge(selected, deepest.x, 1e-12))
        hi = (outer if selected(outer)
              else _bisect_edge(selected, deepest.x, outer))
        return [(lo, hi)] if hi > lo else []


class LocalCost(StatePolicy):
    """Myopic rule for the cumulative cost: query where an immediate
    mistake is likelier than the query price, P(mistake|omega) > Delta.

    The cost rate Delta E[p] + E[(1-p) 1_mistake] has no terminal term, so this
    rule places no value on the information a label carries.  It reduces to an
    inner margin with kappa_t = sqrt(q) sqrt(2(q/m^2 -1)) |erfinv(2 Delta -1)|,
    and has no inner hole: a query also avoids a mistake now, and that benefit
    is largest at omega = 0.

    In: delta, the price of a query.
    """

    def __init__(self, delta):
        self.delta = float(delta)
        self.c = abs(erfinv(2.0 * self.delta - 1.0))

    def kappa(self, m, q):
        """The margin radius of this rule, capped at the support cutoff.

        The radius diverges as m -> 0, where every sample is a coin flip and
        so worth querying at any price below 1/2.  An interval that wide makes
        the quadrature weights underflow to zero, which would read as querying
        nothing at all, the exact opposite.  Clipping costs nothing at finite
        m, since the density beyond the cutoff is ~1e-15.

        In:  m, q scalars or arrays.
        Out: the radius, shaped like the inputs.
        """
        spread = np.maximum(2.0 * (q / np.maximum(m * m, 1e-300) - 1.0), 0.0)
        return np.minimum(np.sqrt(q) * np.sqrt(spread) * self.c,
                          OMEGA_MAX * np.sqrt(q))

    def accepts(self, omega, m, q, t):
        """In: omega of any shape, m, q scalars or arrays, t (unused).
                Out: a boolean array like omega."""
        return np.abs(omega) < self.kappa(m, q)

    def intervals(self, m, q, t):
        """In: m, q scalars, t (unused).  Out: [(0, kappa)], empty if kappa is 0."""
        kap = float(self.kappa(m, q))
        return [(0.0, kap)] if kap > 0.0 else []


# -------------------------------------------------------------- integration
def integrate(policy, p: Params, delta=None, trace=False):
    """Forward-Euler the ODEs under `policy`.

    Returns eps_g(t_F) as 'eps', the time-averaged query rate as 'budget', the
    normalized query and unqueried-mistake counts as 'Q' and 'M', plus
    'R' = Delta Q + M if delta is given and the trajectory if trace.

    M uses E[(1-p) 1_mistake] = eps_g - E[p 1_mistake].  Q and M are trapezoid
    sums.  The trajectory is first-order Euler, with an error of ~1e-4 in eps_g
    at the default n_steps.

    In:  a policy, a Params, optionally the query price delta, and trace.
    Out: one dict.  'eps', 'budget', 'Q', 'M' are always floats.  'R' is a
         float present only with delta, and 't', 'm', 'q' are arrays of
         length n_steps + 1 present only with trace.
    """
    dt = p.t_F / p.n_steps
    m, q = p.m0, p.q0
    rate = np.empty(p.n_steps + 1)       # E[p_t]
    missed = np.empty(p.n_steps + 1)     # E[(1 - p_t) 1_mistake]
    traj_m, traj_q = [m], [q]

    for k in range(p.n_steps + 1):
        dm, dq, r, hit = drift(policy.intervals(m, q, k * dt), m, q, p.eta,
                               policy.prob)
        rate[k] = r
        missed[k] = gen_error(m, q) - hit
        if k == p.n_steps:
            break
        m, q = m + dt * dm, q + dt * dq
        if trace:
            traj_m.append(m)
            traj_q.append(q)

    t = np.linspace(0.0, p.t_F, p.n_steps + 1)
    Q, M = float(_trapz(rate, t)), float(_trapz(missed, t))
    out = {'eps': gen_error(m, q), 'budget': Q / p.t_F, 'Q': Q, 'M': M}
    if delta is not None:
        out['R'] = delta * Q + M
    if trace:
        out['t'], out['m'], out['q'] = t, np.array(traj_m), np.array(traj_q)
    return out


def bisect(f, target, lo, hi, xtol=1e-8, n_max=60):
    """Solve f(x) = target for increasing f, returning the nearer endpoint if
    the target lies outside [f(lo), f(hi)].

    In:  f increasing, the target value, and a bracket lo < hi.
    Out: one scalar x.
    """
    if f(hi) < target:
        return hi
    for _ in range(n_max):
        if hi - lo < xtol:
            break
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < target else (lo, mid)
    return 0.5 * (lo + hi)


# ------------------------------------------------- finite-d SGD simulation
def simulate(policy, p: Params, d=1000, n_runs=1, seed=0, delta=None,
             record=0):
    """One pass of SGD on a fresh Gaussian stream, n_runs independent students.

    The initial condition is exact.  Taking beta = sqrt(d) e_1 costs no
    generality, and then the student puts its whole aligned component in
    coordinate 1 while the transverse part is a random direction rescaled to
    the prescribed norm, so every run starts at exactly (m0, q0).

    The figures use a single run, so the scatter around the ODE curves is the
    actual finite-d correction rather than a shrinkable standard error.
    record > 0 also returns the (m, q) trajectory every `record` steps.

    In:  a policy, a Params, the dimension d, the number of runs, a seed,
         optionally delta, and record.
    Out: one dict, everything per run: 'eps', 'Q', 'M', 'budget', 'm', 'q' of
         shape (n_runs,), 'R' likewise with delta, and with record > 0 a 'traj'
         triple of times (n_rec,) and m, q of shape (n_rec, n_runs).  These stay
         arrays at n_runs = 1.
    """
    rng = np.random.default_rng(seed)
    n = int(p.t_F * d)
    sqrt_d = np.sqrt(d)

    W = np.zeros((n_runs, d))
    W[:, 0] = p.m0 * sqrt_d
    transverse = max(0.0, p.q0 - p.m0 ** 2)
    if transverse > 0.0:
        z = rng.standard_normal((n_runs, d - 1))
        W[:, 1:] = z * (np.sqrt(transverse * d)
                        / np.linalg.norm(z, axis=1, keepdims=True))

    # beta = sqrt(d) e_1, so m = W @ beta / d is the first column over sqrt(d)
    align = lambda: W[:, 0] / sqrt_d
    norm2 = lambda: np.einsum('ij,ij->i', W, W) / d

    n_query = np.zeros(n_runs)
    n_missed = np.zeros(n_runs)
    m, q = align(), norm2()
    ts, traj_m, traj_q = [], [], []

    for i in range(n):
        if record and i % record == 0:
            ts.append(i / d)
            traj_m.append(align())
            traj_q.append(norm2())
        x = rng.standard_normal((n_runs, d)) / sqrt_d
        omega = np.einsum('ij,ij->i', W, x)
        omega_star = x[:, 0] * sqrt_d
        y = np.sign(omega_star)
        if policy.uses_state:
            m, q = align(), norm2()
        hit = np.asarray(policy.accepts(omega, m, q, i / d), dtype=bool)
        if policy.prob < 1.0:
            hit &= rng.random(n_runs) < policy.prob
        n_query += hit
        n_missed += (~hit) & (omega * omega_star < 0)
        g = np.where(hit, np.minimum(omega * y, 0.0) * y, 0.0)
        W -= p.eta * g[:, None] * x

    m, q = align(), norm2()
    out = {'eps': gen_error(m, q), 'Q': n_query / d, 'M': n_missed / d,
           'budget': n_query / n, 'm': m, 'q': q}
    if delta is not None:
        out['R'] = delta * n_query / d + n_missed / d
    if record:
        ts.append(n / d)
        traj_m.append(m)
        traj_q.append(q)
        out['traj'] = (np.array(ts), np.array(traj_m), np.array(traj_q))
    return out