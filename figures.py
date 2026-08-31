"""Static figures: sound graphs, chord space, and the classical/quantum contrast."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from concurrent.futures import ProcessPoolExecutor
import xen

# ---- palette (dataviz reference instance, light surface) --------------------
SURF, PAGE = "#fcfcfb", "#f9f9f7"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"          # validated slots 1-3
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", [SURF] + BLUES)
ORANGES = ["#fbe0d3", "#f6c3a9", "#f2a67f", "#ee8b58", "#eb6834", "#d95926",
           "#b8481d", "#953916", "#722b10"]
SEQ2 = LinearSegmentedColormap.from_list("seq_orange", [SURF] + ORANGES)

plt.rcParams.update({
    "figure.facecolor": PAGE, "axes.facecolor": SURF, "savefig.facecolor": PAGE,
    "font.family": ["DejaVu Sans"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
})
NAMES = "C C# D D# E F F# G G# A A# B".split()


def tidy(ax, grid="y"):
    ax.grid(True, axis=grid, zorder=0)
    ax.set_axisbelow(True)


# =============================================================== figure 1
GLISS = dict(beta=0.6, gamma=0.7, drift=(0.8, 0.0, -0.8), start=(24, 24, 24))
TONAL = dict(beta=4.0, gamma=1.0, drift=(0.0, 0.0, 0.0), start=(21, 25, 28))


def _paths(cfg, T, NREG, seed):
    return xen.gillespie(N=NREG, T=T, rng=np.random.default_rng(seed), **cfg)


def fig_soundgraph():
    print("fig 1: sound graphs")
    T, NREG = 22.0, 48
    fig = plt.figure(figsize=(13.4, 4.5))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.05, 1.1, 1.0], wspace=0.30,
                          left=0.048, right=0.99, top=0.78, bottom=0.135)

    # (a) drift-dominated: the Metastaseis fan
    ax = fig.add_subplot(gs[0, 0])
    for r in range(18):
        ts, xs = _paths(GLISS, T, NREG, 300 + r)
        for k, c in enumerate((S1, S2, S3)):
            ax.step(ts, xs[k], where="post", color=c, lw=0.9, alpha=0.30)
    ts, xs = _paths(GLISS, T, NREG, 300)
    for k, (c, lab) in enumerate(zip((S1, S2, S3), ("voice 1  a = +0.8",
                                                    "voice 2  a = 0",
                                                    "voice 3  a = -0.8"))):
        ax.step(ts, xs[k], where="post", color=c, lw=1.9)
        ax.annotate(lab, (0.4, xs[k][-1]), xytext=(0, 0), textcoords="offset points",
                    color=c, fontsize=7.5, va="center", fontweight="bold",
                    bbox=dict(fc=SURF, ec="none", pad=1.2))
    ax.set_xlim(0, T); ax.set_ylim(0, NREG)
    ax.set_xlabel("time"); ax.set_ylabel("pitch (semitones)")
    ax.set_title(r"a   drift regime  $\beta$ = 0.6:  glissandi", loc="left", fontsize=10)
    tidy(ax)

    # (b) tension-dominated: the trio locks onto consonant chords
    ax = fig.add_subplot(gs[0, 1])
    ts, xs = _paths(TONAL, T, NREG, 11)
    for k, c in enumerate((S1, S2, S3)):
        ax.step(ts, xs[k], where="post", color=c, lw=1.9)
        ax.annotate(f"voice {k+1}", (T, xs[k][-1]), xytext=(3, 0),
                    textcoords="offset points", color=c, fontsize=7.5, va="center",
                    fontweight="bold")
    ax.set_xlim(0, T * 1.20); ax.set_ylim(14, 36)
    ax.set_xlabel("time"); ax.set_ylabel("pitch (semitones)")
    ax.set_title(r"b   tension regime  $\beta$ = 4:  the trio binds", loc="left",
                 fontsize=10)
    tidy(ax)

    # (c) the density the master equation actually propagates
    ax = fig.add_subplot(gs[0, 2])
    NT, tg = 220, np.linspace(0, T, 220)
    hist = np.zeros((NREG, NT))
    for r in range(500):
        ts, xs = _paths(GLISS, T, NREG, 1000 + r)
        for k in range(3):
            v = xs[k][np.searchsorted(ts, tg, side="right") - 1]
            np.add.at(hist, (v, np.arange(NT)), 1.0)
    hist /= hist.max()
    ax.imshow(hist, origin="lower", aspect="auto", cmap=SEQ,
              extent=[0, T, 0, NREG], interpolation="bilinear")
    ax.set_xlabel("time"); ax.set_ylabel("pitch (semitones)")
    ax.set_title("c   P(pitch, t):  500 realisations of a", loc="left", fontsize=10)
    ax.text(0.96, 0.05, "the cloud, not the path,\nis what the equation evolves",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)

    # (d) temperature controls the texture
    ax = fig.add_subplot(gs[0, 3])
    for beta, c, lab in ((0.0, S3, r"$\beta=0$  Brownian cloud"),
                         (2.0, S1, r"$\beta=2$  tonal drift"),
                         (6.0, S2, r"$\beta=6$  frozen")):
        L = xen.ChordLattice(beta=beta, drift=(0, 0, 0))
        pb = np.exp(-beta * L.V); pb /= pb.sum()
        ic = np.zeros(12)
        d = np.minimum(np.mod(L.n[0] - L.n[1], 12), np.mod(L.n[1] - L.n[0], 12))
        for k in range(7):
            ic[k] = pb[d == k].sum()
        ax.plot(np.arange(7), ic[:7] / ic[:7].sum(), color=c, lw=2, marker="o", ms=5,
                label=lab, clip_on=False)
    ax.set_xticks(range(7))
    ax.set_xticklabels(["uni", "m2", "M2", "m3", "M3", "P4", "TT"])
    ax.set_xlabel("interval class between voices 1 and 2")
    ax.set_ylabel("stationary probability")
    ax.set_title("d   stationary state vs temperature", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    tidy(ax)

    fig.suptitle("The Xenakis master equation: two regimes, the density it propagates, "
                 "and its stationary state", x=0.048, ha="left", fontsize=12.5, y=0.955)
    fig.savefig("out/fig1_sound_graphs.png", dpi=190)
    plt.close(fig)


# =============================================================== figure 2
def cube_axes(ax):
    ax.set_xlabel("voice 1", labelpad=-6); ax.set_ylabel("voice 2", labelpad=-6)
    ax.set_zlabel("voice 3", labelpad=-6)
    ax.set_xlim(-.5, 11.5); ax.set_ylim(-.5, 11.5); ax.set_zlim(-.5, 11.5)
    ax.set_xticks([0, 4, 8]); ax.set_yticks([0, 4, 8]); ax.set_zticks([0, 4, 8])
    ax.set_xticklabels([NAMES[i] for i in (0, 4, 8)], fontsize=7)
    ax.set_yticklabels([NAMES[i] for i in (0, 4, 8)], fontsize=7)
    ax.set_zticklabels([NAMES[i] for i in (0, 4, 8)], fontsize=7)
    ax.tick_params(pad=-2)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor(SURF); a.pane.set_edgecolor(GRID); a.pane.set_alpha(1.0)
        a._axinfo["grid"]["color"] = GRID; a._axinfo["grid"]["linewidth"] = 0.5
    ax.set_box_aspect((1, 1, 1))


def fig_chordspace():
    print("fig 2: chord space")
    L = xen.ChordLattice(beta=2.0, drift=(0, 0, 0))
    n1, n2, n3 = L.n
    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.06, left=0.01, right=0.99,
                          top=0.82, bottom=0.06)

    # (a) tension landscape
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    v = (L.V - L.V.min()) / (L.V.max() - L.V.min())
    keep = v < 0.40
    ax.scatter(n1[keep], n2[keep], n3[keep], c=1 - v[keep], cmap=SEQ,
               s=24 * (1 - v[keep]) ** 2 + 2, alpha=0.5, linewidths=0, vmin=0, vmax=1)
    cube_axes(ax); ax.view_init(22, 38)
    ax.set_title("a   tension landscape  V(n)\n(darker = more consonant)",
                 loc="left", fontsize=10, y=0.99)

    # (b) ground manifold
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    g = L.V < L.V.min() + 1e-9
    ax.scatter(n1[~g], n2[~g], n3[~g], c=GRID, s=2, alpha=0.30, linewidths=0)
    ax.scatter(n1[g], n2[g], n3[g], c=S2, s=34, linewidths=0, depthshade=False)
    cube_axes(ax); ax.view_init(22, 38)
    ax.set_title(f"b   ground manifold: {int(g.sum())} chords\n"
                 "= the 12 major triads x 6 voicings", loc="left", fontsize=10, y=0.99)

    # (c) Boltzmann state at beta = 3
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    beta = 3.0
    pb = np.exp(-beta * L.V); pb /= pb.max()
    keep = pb > 0.04
    ax.scatter(n1[keep], n2[keep], n3[keep], c=pb[keep], cmap=SEQ,
               s=90 * pb[keep] + 2, alpha=0.8, linewidths=0, vmin=0, vmax=1)
    cube_axes(ax); ax.view_init(22, 38)
    ax.set_title(r"c   stationary state $\propto e^{-\beta V}$" + f",  $\\beta$ = {beta:.0f}",
                 loc="left", fontsize=10, y=0.99)

    fig.suptitle("Chord space: the 12x12x12 torus of three-voice chords",
                 x=0.02, ha="left", fontsize=12.5, y=0.96)
    fig.savefig("out/fig2_chord_space.png", dpi=190)
    plt.close(fig)


# =============================================================== figure 3
def _lind_run(args):
    kappa, times = args
    L = xen.ChordLattice(beta=2.0, drift=(0, 0, 0))
    p0 = np.zeros(L.M); p0[xen.index_of((0, 4, 7))] = 1.0
    rho0 = np.diag(p0).astype(complex)
    pq, _ = L.evolve_lindblad(rho0, times, J=1.0, vscale=1.0, kappa=kappa)
    return kappa, pq


def fig_quantum():
    print("fig 3: classical vs quantum")
    L = xen.ChordLattice(beta=2.0, drift=(0, 0, 0))
    n1, n2, n3 = L.n
    i0 = xen.index_of((0, 4, 7))
    tt = np.linspace(0, 6, 61)
    p0 = np.zeros(L.M); p0[i0] = 1.0
    psi0 = np.zeros(L.M, complex); psi0[i0] = 1.0
    Pc = L.evolve_classical(p0, tt)
    Pq, _ = L.evolve_schrodinger(psi0, tt, J=1.0, vscale=1.0)

    n0 = np.array([0, 4, 7])[:, None]
    d = np.minimum(np.mod(L.n - n0, 12), np.mod(n0 - L.n, 12))
    r2 = (d ** 2).sum(0)
    Rc, Rq = np.sqrt(Pc @ r2), np.sqrt(Pq @ r2)

    fig = plt.figure(figsize=(11.5, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0], hspace=0.30, wspace=0.28,
                          left=0.06, right=0.985, top=0.86, bottom=0.08)

    snaps = [10, 25]
    for col, k in enumerate(snaps):
        for row, (P, name, cm) in enumerate(((Pc, "classical", SEQ), (Pq, "quantum", SEQ2))):
            ax = fig.add_subplot(gs[0, col * 2 + row], projection="3d")
            p = P[k] / P[k].max()
            keep = p > 0.02
            ax.scatter(n1[keep], n2[keep], n3[keep], c=p[keep], cmap=cm,
                       s=70 * p[keep] + 1.5, alpha=0.75, linewidths=0, vmin=0, vmax=1)
            cube_axes(ax); ax.view_init(22, 38)
            ax.set_title(f"{name},  t = {tt[k]:.1f}", loc="left", fontsize=9.5, y=0.97,
                         color=S1 if row == 0 else S2)

    # spreading law
    ax = fig.add_subplot(gs[1, :2])
    ax.loglog(tt[1:], Rc[1:], color=S1, lw=2.2, label="classical  (master equation)")
    ax.loglog(tt[1:], Rq[1:], color=S2, lw=2.2, label="quantum  (unitary, same lattice)")
    tref = tt[3:20]
    ax.loglog(tref, 1.35 * tref ** 0.5, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.loglog(tref, 1.35 * tref ** 1.0, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.annotate(r"$t^{1/2}$  diffusive", (tref[-1], 1.35 * tref[-1] ** .5),
                xytext=(7, -10), textcoords="offset points", fontsize=8, color=MUTED)
    ax.annotate(r"$t^{1}$  ballistic", (tref[-1], 1.35 * tref[-1] ** 1.0),
                xytext=(7, 2), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlim(tt[1] * 0.85, tt[-1] * 1.5)
    ax.set_xlabel("time"); ax.set_ylabel("r.m.s. distance travelled\nin chord space")
    ax.set_title("c   how fast the two explore chord space", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    tidy(ax, grid="both")

    # decoherence crossover
    ax = fig.add_subplot(gs[1, 2:])
    times = np.linspace(0, 1.0, 6)
    pc_short = L.evolve_classical(p0, times)
    kappas = [0.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
    if os.path.exists("out/traces.npz"):                 # reuse the cached sweep
        z = np.load("out/traces.npz")
        kappas, dev = list(z["kappas"]), list(z["dev"])
        print("  (kappa sweep loaded from cache)")
    else:
        with ProcessPoolExecutor(max_workers=2) as ex:
            res = dict(ex.map(_lind_run, [(k, times) for k in kappas]))
        dev = [np.abs(res[k] - pc_short).max() for k in kappas]
    ax.semilogx([max(k, 0.5) for k in kappas], dev, color=S2, lw=2.2, marker="o", ms=6,
                clip_on=False)
    ax.annotate("purely coherent\n(quantum Xenakis)", (0.5, dev[0]), xytext=(12, -14),
                textcoords="offset points", fontsize=8, color=INK2, va="top")
    ax.annotate("classical Xenakis\nchain recovered", (128, dev[-1]), xytext=(-6, 26),
                textcoords="offset points", fontsize=8, color=INK2, ha="right")
    ax.set_xlabel(r"dephasing rate $\kappa$   (log scale)")
    ax.set_ylabel(r"max$_n$ |$P_{\rm quantum}(n)-P_{\rm classical}(n)$|")
    ax.set_title(r"d   decoherence collapses the quantum model onto the classical one",
                 loc="left", fontsize=10)
    ax.set_ylim(0, max(dev) * 1.15)
    tidy(ax)

    fig.suptitle("Quantum vs classical Xenakis on the same chord lattice, from the same "
                 "chord C-E-G", x=0.06, ha="left", fontsize=12.5, y=0.965)
    fig.savefig("out/fig3_quantum.png", dpi=190)
    plt.close(fig)
    np.savez("out/traces.npz", tt=tt, Pc=Pc, Pq=Pq, Rc=Rc, Rq=Rq,
             kappas=np.array(kappas), dev=np.array(dev), V=L.V)


if __name__ == "__main__":
    import os, sys
    os.makedirs("out", exist_ok=True)
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which: fig_soundgraph()
    if "2" in which: fig_chordspace()
    if "3" in which: fig_quantum()
    print("figures done")
