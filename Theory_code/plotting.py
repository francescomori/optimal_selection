"""Figure style and drawing helpers shared by the notebooks."""
from collections import namedtuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# One entry per policy family, in the order of optimize.KEYS.  Colour
# identifies the family, and line style and marker repeat that information so
# the series stay distinguishable in greyscale.
STYLES = [('-', 'o', 'C0'), ('--', 's', 'C1'), (':', '^', 'C2'),
          ('-.', 'D', 'C3'), ((0, (3, 1, 1, 1, 1, 1)), 'v', 'C4'),
          ((0, (7, 3)), '*', 'C5')]
LABELS = [('local', 'myopic'),
          ('cmar', 'inner margin (constant)'),
          ('smar', 'inner margin (schedule)'),
          ('cann', 'intermediate margin (constant)'),
          ('sann', 'intermediate margin (schedule)'),
          ('rand', 'random')]

# Okabe-Ito blue and orange: separable under deuteranopia, protanopia and
# tritanopia, and far enough apart in luminance to survive greyscale printing.
CB_LOCAL, CB_GLOBAL = '#0072B2', '#E69F00'


def style():
    """Set the shared rcParams.

    In:  nothing.
    Out: nothing; sets matplotlib rcParams in place.
    """
    matplotlib.rcParams.update({
        'axes.labelsize': 20, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
        'legend.fontsize': 15, 'lines.linewidth': 2.2, 'lines.markersize': 7,
        'axes.grid': False, 'axes.spines.top': False, 'axes.spines.right': False,
        'figure.dpi': 120, 'savefig.bbox': 'tight', 'font.size': 16,
        'mathtext.fontset': 'cm'})


def _clear_legends(fig, ax, curves, max_pad=0.4, step=0.02):
    """Raise the top of the y axis until no curve runs through a legend.

    The legends here sit inside the axes, which only works if the corner they
    occupy is empty.  Rather than assume it, measure it: transform each legend
    box into data coordinates and grow the top in small steps until every box
    is clear.  Curves are densified first, so a crossing between two plotted
    points is not missed.

    In:  the figure, its axes, and curves as a list of (x, y) arrays.
    Out: nothing; adjusts the y limit in place.
    """
    dense = []
    for x, y in curves:
        x, y = np.asarray(x, float), np.asarray(y, float)
        fine = np.linspace(x.min(), x.max(), 500)
        dense.append((fine, np.interp(fine, x, y)))

    lo, hi = ax.get_ylim()
    span = hi - lo
    pad = 0.0
    while True:
        ax.set_ylim(lo, hi + pad * span)
        fig.canvas.draw()
        to_data = ax.transData.inverted()
        clash = False
        for legend in [a for a in ax.get_children() if isinstance(a, Legend)]:
            box = legend.get_window_extent()
            (x0, y0), (x1, y1) = to_data.transform([[box.x0, box.y0],
                                                    [box.x1, box.y1]])
            for x, y in dense:
                if np.any((x >= min(x0, x1)) & (x <= max(x0, x1))
                          & (y >= min(y0, y1)) & (y <= max(y0, y1))):
                    clash = True
                    break
        if not clash or pad >= max_pad:
            return
        pad += step


def curve_figure(x, ode, sim, ode_prefix, sim_prefix, xlabel, ylabel, loc,
                 halo=(), labelsize=None):
    """One family per series: theory as lines, one finite-d run as symbols.

    `ode` and `sim` are the loaded result dicts, read as ode[ode_prefix + key]
    and sim[sim_prefix + key].  A second legend in the opposite corner says
    which is which.

    `halo` names families that coincide with another exactly, as the
    intermediate margins do in the cost setting, where their optimal inner
    radius is zero.  Those are drawn as a thick translucent band underneath, so
    a curve hidden beneath another is not mistaken for missing data.

    In:  x of shape (n,), the ode and sim dicts or sim = None, the two key
         prefixes, the axis labels, the legend corner, halo, a tuple of keys,
         and labelsize to override the shared axis-label size.
    Out: (fig, ax).
    """
    style()
    fig, ax = plt.subplots(figsize=(7.0, 5.4))

    for (key, label), (dash, marker, colour) in zip(LABELS, STYLES):
        if key in halo:
            ax.plot(x, ode[ode_prefix + key], '-', color=colour, lw=7.0,
                    alpha=0.28, solid_capstyle='round', zorder=1)
            ax.plot([], [], '-', color=colour, lw=7.0, alpha=0.28, label=label)
            continue
        # an empty artist carries line style and marker together into the legend
        ax.plot([], [], ls=dash, marker=marker, color=colour, mfc='none',
                mew=1.6, label=label)
        ax.plot(x, ode[ode_prefix + key], ls=dash, color=colour, zorder=3)
        if sim is not None:
            ax.plot(x, sim[sim_prefix + key], ls='none', marker=marker,
                    color=colour, mfc='none', mew=1.6, ms=8, zorder=4)

    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    families = ax.legend(frameon=False, loc=loc, handlelength=2.4,
                         borderaxespad=0.5, labelspacing=0.30, fontsize=13,
                         handletextpad=0.6)
    if sim is not None:
        ax.add_artist(families)
        opposite = {'upper right': 'lower left',
                    'upper left': 'lower right'}[loc]
        ax.legend([Line2D([], [], ls='-', color='0.35', lw=2.2),
                   Line2D([], [], ls='none', marker='o', color='0.35',
                          mfc='none', mew=1.6, ms=8)],
                  ['Theory', 'Simulations'], frameon=False, loc=opposite,
                  handlelength=2.4, borderaxespad=0.5, labelspacing=0.30,
                  fontsize=13, handletextpad=0.6)
    _clear_legends(fig, ax, [(x, ode[ode_prefix + k]) for k, _ in LABELS])
    return fig, ax


# One shaded region of a selection map.
#   t, lower, upper : the edges against time
#   step            : draw as a staircase (piecewise-constant schedules) rather
#                     than interpolating between control points
#   mirror          : the edges are given on omega > 0 and the negative half is
#                     to be drawn too.  An inner margin is instead passed whole,
#                     as lower = -kappa, upper = +kappa, so that the two halves
#                     are one polygon and no hairline seam appears at omega = 0.
Band = namedtuple('Band', 't lower upper step mirror')


def draw_band(ax, band, colour='0.15', alpha=1.0, seam=1e-6):
    """Shade one selected set.  Non-finite or inverted edges are left blank,
    which is how a time with nothing selected appears.

    A mirrored band whose inner edge vanishes everywhere is drawn as a single
    polygon spanning both signs.  Drawn as two touching halves it would show a
    seam at omega = 0 from the anti-aliasing of their shared edge, which reads
    as a gap that is not there.

    In:  an axes, a Band, a colour, an alpha, and the inner-edge tolerance
         below which the two halves are merged.
    Out: nothing; shades the axes in place.
    """
    lower = np.asarray(band.lower, float)
    upper = np.asarray(band.upper, float)
    if band.mirror and np.nanmax(np.abs(lower)) <= seam:
        halves = [(-upper, upper)]
    else:
        halves = [(lower, upper)] + ([(-upper, -lower)] if band.mirror else [])
    for lo, hi in halves:
        ok = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        ax.fill_between(band.t, np.where(ok, lo, np.nan),
                        np.where(ok, hi, np.nan),
                        step='post' if band.step else None, color=colour,
                        alpha=alpha, linewidth=0.0)


def selection_overlay(local, optimal, t_F, omega_max=1.9):
    """The myopic and horizon-optimized selected sets in one axes, as two Band
    instances.  The fills are translucent, so the overlap reads as "both".

    In:  the myopic and horizon-optimized Bands, the horizon t_F, and the
         y limit.
    Out: (fig, ax).
    """
    style()
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for band, colour in ((local, CB_LOCAL), (optimal, CB_GLOBAL)):
        draw_band(ax, band, colour=colour, alpha=0.60)
    ax.set_xlabel(r'Training time $t$')
    ax.set_ylabel(r'Student margin $\omega$')
    ax.set_xlim(0.0, t_F)
    ax.set_ylim(-omega_max, omega_max)
    ax.legend([Patch(facecolor=CB_LOCAL, alpha=0.60),
               Patch(facecolor=CB_GLOBAL, alpha=0.60)],
              ['myopic', 'horizon-optimized'], frameon=False,
              loc='upper right', handlelength=2.2, borderaxespad=0.5)
    return fig, ax

def selection_grid(budget_panels, cost_panels, t_F, omega_max=1.9,
                   figsize=(7.1, 3.3)):
    """Appendix grid of selection maps: one row per objective, one column per
    constraint level, tightening to the right.

    Laid out for a full-width appendix page, with the label and legend sizes set
    in points to sit alongside 10 pt body text when the figure is included at
    its natural width.  The panels are wider than tall so that the full horizon
    fits comfortably, and the time axis is ticked at its ends and midpoint.

    Axes are shared, so the panels are directly comparable, and each is
    annotated with its own constraint value, which also says which row it is
    in, so the figure carries no titles.

    In:  budget_panels and cost_panels, each a list of (label, local Band,
         optimal Band) ordered left to right; the horizon t_F; the y limit.
    Out: (fig, axes) with axes of shape (2, n_columns).
    """
    style()
    n = len(budget_panels)
    fig, axes = plt.subplots(2, n, figsize=figsize, sharex=True, sharey=True)
    for row, panels in enumerate((budget_panels, cost_panels)):
        for col, (label, local, optimal) in enumerate(panels):
            ax = axes[row, col]
            draw_band(ax, local, colour=CB_LOCAL, alpha=0.60)
            draw_band(ax, optimal, colour=CB_GLOBAL, alpha=0.60)
            ax.text(0.5, 1.03, label, transform=ax.transAxes, ha='center',
                    va='bottom', fontsize=10)
            ax.tick_params(labelsize=9)
            ax.set_xlim(0.0, t_F)
            ax.set_ylim(-omega_max, omega_max)
            ax.set_xticks([0.0, 0.5 * t_F, t_F])
            ax.set_yticks([-1.0, 0.0, 1.0])
    fig.supxlabel(r'Training time $t$', fontsize=11, y=0.02)
    fig.supylabel(r'Student margin $\omega$', fontsize=11, x=0.008)
    fig.legend([Patch(facecolor=CB_LOCAL, alpha=0.60),
                Patch(facecolor=CB_GLOBAL, alpha=0.60)],
               ['myopic', 'horizon-optimized'], frameon=False,
               loc='upper center', bbox_to_anchor=(0.55, 1.01), ncol=2,
               handlelength=1.8, fontsize=10, columnspacing=1.4)
    fig.subplots_adjust(left=0.068, right=0.992, top=0.845, bottom=0.155,
                        hspace=0.42, wspace=0.26)
    return fig, axes

def concentration_figure(ode, traj, t_F, order_loc='center right',
                         key_loc='lower right'):
    """The order parameters from the ODEs against one finite-d run.

    In:  ode, the integrate dict obtained with trace=True; traj, the 'traj'
         triple from simulate; the horizon; and the two legend corners.
    Out: (fig, ax).
    """
    style()
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(ode['t'], ode['m'], '-', color='C0', zorder=3)
    ax.plot(ode['t'], ode['q'], '--', color='C1', zorder=3)
    ts, ms, qs = traj
    ax.plot(ts, ms[:, 0], ls='none', marker='o', color='C0', mfc='none',
            mew=1.6, ms=8, zorder=4)
    ax.plot(ts, qs[:, 0], ls='none', marker='s', color='C1', mfc='none',
            mew=1.6, ms=8, zorder=4)
    ax.set_xlabel(r'Training time $t$')
    ax.set_xlim(0.0, t_F)

    order = ax.legend(
        [Line2D([], [], ls='-', color='C0', marker='o', mfc='none', mew=1.6),
         Line2D([], [], ls='--', color='C1', marker='s', mfc='none', mew=1.6)],
        [r'$m$', r'$q$'], frameon=False, loc=order_loc,
        handlelength=2.4, borderaxespad=0.5, fontsize=18)
    ax.add_artist(order)
    ax.legend([Line2D([], [], ls='-', color='0.35', lw=2.2),
               Line2D([], [], ls='none', marker='o', color='0.35',
                      mfc='none', mew=1.6, ms=8)],
              ['Theory', 'Simulations'], frameon=False, loc=key_loc,
              handlelength=2.0, borderaxespad=0.5, fontsize=15)
    return fig, ax
