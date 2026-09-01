"""Reproduce the Pithoprakta velocity sketch, and the caustics that make its loops.

Xenakis's construction: 46 parts, each given a glissando velocity drawn from the
Maxwell-Boltzmann speed law he wrote in the margin of the sheet,

    f(s) = 2 / (alpha sqrt(pi)) * exp(-s^2 / alpha^2),      alpha = 35,

held for one time segment, then redrawn.  Pitch is the integral of velocity, so
within a segment every part is a straight line and the ensemble is a family of
straight lines.  A family of straight lines has an envelope: where the map from
part index to velocity runs against the map from part index to starting pitch,
the family folds at t* = -x0'(xi) / v'(xi), the line density diverges, and the
fold shows up on the page as a lens with bright edges.  Those are the loops.

Equation (M) cannot produce them: it redraws the jump direction at every event,
so tau = 0 and no line stays straight long enough to cross.  Run this with
--tau 0.2 to watch the lenses dissolve into the diffusive wash.

    python pithoprakta.py                  # out/fig_pithoprakta.png
    python pithoprakta.py --tau 0.25       # the overdamped limit, no caustics
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURF, PAGE = "#fcfcfb", "#f4f2ea"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#cfcbba"

ALPHA = 35.0        # the sheet's parameter, in its own units
NPART = 46          # strings in Pithoprakta
TMAX = 24.0         # sheet-widths of time
REG = 48.0          # semitones of register


def velocities(rng, n, alpha):
    """Maxwell-Boltzmann speeds with a random sign: a half-normal of width
    alpha/sqrt(2), signed.  <s> = alpha/sqrt(pi), <s^2> = alpha^2/2."""
    return np.abs(rng.normal(0.0, alpha / np.sqrt(2.0), n)) * rng.choice([-1.0, 1.0], n)


def run(rng, npart=NPART, tau=3.0, alpha=ALPHA, tmax=TMAX, reg=REG, scale=0.055):
    """Velocity-jump paths with reflecting register edges.

    `scale` converts the sheet's velocity units into semitones per unit time;
    tau is the mean time a part keeps its velocity (tau -> inf: one straight
    line each; tau -> 0 at fixed sigma^2 tau: the diffusion of equation (M))."""
    x = rng.uniform(0.0, reg, npart)
    v = velocities(rng, npart, alpha) * scale
    ts, xs = [np.zeros(npart)], [x.copy()]
    t = 0.0
    while t < tmax:
        dt = rng.exponential(tau / npart)          # next tumble, any part
        step = min(dt, tmax - t)
        x = x + v * step
        over = x > reg; x[over] = 2.0 * reg - x[over]; v[over] *= -1.0
        under = x < 0.0; x[under] = -x[under]; v[under] *= -1.0
        t += step
        ts.append(np.full(npart, t)); xs.append(x.copy())
        if t < tmax:
            v[rng.integers(npart)] = velocities(rng, 1, alpha)[0] * scale
    return np.array(ts), np.array(xs)


def draw(panels, out):
    """One panel per velocity memory, same diffusion constant D = sigma^2 tau."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(13.5, 4.3 * len(panels)),
                             dpi=150, sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(PAGE)
    for ax, (ts, xs, title) in zip(axes, panels):
        ax.set_facecolor(SURF)
        for k in np.arange(0, TMAX + .001, TMAX / 24):      # the graph paper
            ax.axvline(k, color=GRID, lw=.5, zorder=0)
        for k in np.arange(0, REG + .001, 4):
            ax.axhline(k, color=GRID, lw=.5, zorder=0)
        for k in np.arange(0, REG + .001, 12):
            ax.axhline(k, color=MUTED, lw=.8, zorder=0)
        for j in range(xs.shape[1]):
            ax.plot(ts[:, j], xs[:, j], color=INK, lw=.62, alpha=.72,
                    solid_joinstyle="miter", zorder=2)
        ax.set_xlim(0, TMAX); ax.set_ylim(0, REG)
        ax.set_ylabel("pitch — semitones", color=INK2, fontsize=8)
        ax.set_title(title, color=INK, fontsize=9.5, loc="left", pad=8)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_color(MUTED); sp.set_linewidth(.7)
    axes[-1].set_xlabel("ΧΡΟΝΟΣ — time", color=INK2, fontsize=8)
    fig.tight_layout()
    os.makedirs("out", exist_ok=True)
    fig.savefig(out); plt.close(fig)
    print(f"wrote {out}  ({os.path.getsize(out)/1e3:.0f} kB)")


def spreading(rng, tau, alpha=ALPHA, scale=0.055, n=20000, tmax=TMAX):
    """<x^2>(t) for the free process, against 2 sigma^2 tau^2 (t/tau - 1 + e^-t/tau)."""
    grid = np.linspace(0, tmax, 60)
    x = np.zeros(n); v = velocities(rng, n, alpha) * scale
    cur = np.zeros(n); nxt = rng.exponential(tau, n)   # each part its own clock
    out = [0.0]
    for g in grid[1:]:
        while True:
            due = nxt < g
            k = int(due.sum())
            if k == 0:
                break
            x[due] += v[due] * (nxt[due] - cur[due])
            cur[due] = nxt[due]
            v[due] = velocities(rng, k, alpha) * scale
            nxt[due] += rng.exponential(tau, k)
        x += v * (g - cur); cur[:] = g
        out.append(np.mean(x ** 2))
    sig2 = (alpha * scale) ** 2 / 2.0
    pred = 2 * sig2 * tau ** 2 * (grid / tau - 1 + np.exp(-grid / tau))
    return grid, np.array(out), pred


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=3.0,
                    help="velocity memory; small tau is the overdamped limit of (M)")
    ap.add_argument("--tau-fast", type=float, default=None,
                    help="memory for the second panel (default: tau/12)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true",
                    help="check <x^2> against the velocity-jump prediction")
    a = ap.parse_args()
    tau_b = a.tau
    tau_d = a.tau_fast if a.tau_fast else tau_b / 12.0
    # hold D = sigma^2 tau fixed, so only the memory differs between panels
    alpha_d = ALPHA * np.sqrt(tau_b / tau_d)
    ts_b, xs_b = run(np.random.default_rng(a.seed), tau=tau_b, alpha=ALPHA)
    ts_d, xs_d = run(np.random.default_rng(a.seed), tau=tau_d, alpha=alpha_d)
    D = (ALPHA * 0.055) ** 2 / 2 * tau_b
    draw([(ts_b, xs_b, f"τ = {tau_b:g}  —  ballistic between redraws: the family of "
                       f"straight lines folds, and every fold is a lens (α = {ALPHA:g})"),
          (ts_d, xs_d, f"τ = {tau_d:g}  —  same diffusion constant D = {D:.2f}, no memory: "
                       f"the overdamped limit of equation (M), and the lenses are gone")],
         a.out or "out/fig_pithoprakta.png")
    if a.check:
        g, emp, pred = spreading(np.random.default_rng(a.seed + 1), a.tau)
        rel = np.abs(emp[1:] - pred[1:]) / pred[1:]
        print(f"<x^2> vs 2 s^2 t^2 (t/t - 1 + e^-t/t): max rel. dev "
              f"{rel.max():.3f}, mean {rel.mean():.3f}")
        for k in (1, len(g) // 4, len(g) // 2, len(g) - 1):
            print(f"   t={g[k]:5.2f}   empirical {emp[k]:8.3f}   predicted {pred[k]:8.3f}")
