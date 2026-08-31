"""3D animation: the same initial chord evolving classically and quantum-mechanically
over the 12x12x12 chord lattice."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import io, os
import xen
from figures import (SURF, PAGE, INK, INK2, MUTED, GRID, S1, S2, SEQ, SEQ2,
                     cube_axes, tidy, NAMES)

T_END, NFRAMES = 8.0, 150
J, BETA = 1.0, 2.0
CHORD = (0, 4, 7)


def build():
    L = xen.ChordLattice(beta=BETA, drift=(0, 0, 0))
    tt = np.linspace(0, T_END, NFRAMES)
    i0 = xen.index_of(CHORD)
    p0 = np.zeros(L.M); p0[i0] = 1.0
    psi0 = np.zeros(L.M, complex); psi0[i0] = 1.0
    print("  classical ..."); Pc = L.evolve_classical(p0, tt)
    print("  quantum ...");   Pq, _ = L.evolve_schrodinger(psi0, tt, J=J, vscale=1.0)
    n0 = np.array(CHORD)[:, None]
    d = np.minimum(np.mod(L.n - n0, 12), np.mod(n0 - L.n, 12))
    r2 = (d ** 2).sum(0)
    np.savez("out/anim_data.npz", tt=tt, Pc=Pc, Pq=Pq,
             Rc=np.sqrt(Pc @ r2), Rq=np.sqrt(Pq @ r2), n=L.n, V=L.V)
    return L, tt, Pc, Pq, np.sqrt(Pc @ r2), np.sqrt(Pq @ r2)


def render():
    os.makedirs("out", exist_ok=True)
    L, tt, Pc, Pq, Rc, Rq = build()
    n1, n2, n3 = L.n
    ground = L.V < L.V.min() + 1e-9

    fig = plt.figure(figsize=(9.6, 5.9), dpi=110)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.30], hspace=-0.04,
                          wspace=-0.04, left=0.0, right=1.0, top=0.86,
                          bottom=0.11)
    axc = fig.add_subplot(gs[0, 0], projection="3d")
    axq = fig.add_subplot(gs[0, 1], projection="3d")
    axr = fig.add_subplot(gs[1, :])

    title = fig.text(0.015, 0.955, "One chord, two dynamics on the same lattice",
                     fontsize=13, color=INK, ha="left")
    sub = fig.text(0.015, 0.912,
                   "starting from C-E-G  ·  12x12x12 chord torus  ·  "
                   f"beta = {BETA:g}, J = {J:g}",
                   fontsize=8.5, color=INK2, ha="left")
    clock = fig.text(0.985, 0.955, "", fontsize=11, color=INK, ha="right",
                     family="DejaVu Sans")
    fig.text(0.265, 0.865, "classical   master equation", fontsize=10, color=S1,
             ha="center", fontweight="bold")
    fig.text(0.735, 0.865, "quantum   unitary evolution", fontsize=10, color=S2,
             ha="center", fontweight="bold")

    frames = []
    for k in range(NFRAMES):
        az = 30 + 40 * np.sin(2 * np.pi * k / NFRAMES)
        for ax, P, cm, col, name in ((axc, Pc, SEQ, S1, "classical   master equation"),
                                     (axq, Pq, SEQ2, S2, "quantum   unitary evolution")):
            ax.clear()
            p = P[k] / max(P[k].max(), 1e-12)
            keep = p > 0.05
            ax.scatter(n1[keep], n2[keep], n3[keep], c=p[keep], cmap=cm,
                       s=95 * p[keep] ** 0.8 + 1.5, alpha=0.72, linewidths=0,
                       vmin=0, vmax=1, depthshade=True)
            ax.scatter(*np.array(CHORD)[:, None], c="none", edgecolors=INK2,
                       s=52, linewidths=0.9)
            cube_axes(ax); ax.view_init(21, az)
            ax.set_box_aspect((1, 1, 1), zoom=1.06)

        axr.clear()
        axr.plot(tt, Rc, color=S1, lw=1.6, alpha=0.35)
        axr.plot(tt, Rq, color=S2, lw=1.6, alpha=0.35)
        axr.plot(tt[:k + 1], Rc[:k + 1], color=S1, lw=2.2)
        axr.plot(tt[:k + 1], Rq[:k + 1], color=S2, lw=2.2)
        axr.plot([tt[k]], [Rc[k]], "o", color=S1, ms=5)
        axr.plot([tt[k]], [Rq[k]], "o", color=S2, ms=5)
        axr.set_xlim(0, T_END); axr.set_ylim(0, max(Rq.max(), Rc.max()) * 1.12)
        axr.set_xlabel("time", fontsize=8.5)
        axr.set_ylabel("distance from\nthe initial chord", fontsize=8)
        axr.set_position([0.075, 0.085, 0.885, 0.20])
        axr.annotate("quantum", (tt[-1], Rq[-1]), xytext=(-4, 6),
                     textcoords="offset points", color=S2, fontsize=8,
                     ha="right", fontweight="bold")
        axr.annotate("classical", (tt[-1], Rc[-1]), xytext=(-4, -12),
                     textcoords="offset points", color=S1, fontsize=8,
                     ha="right", fontweight="bold")
        axr.tick_params(labelsize=8)
        tidy(axr)
        clock.set_text(f"t = {tt[k]:5.2f}")

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img = imageio.imread(buf)[..., :3]          # drop alpha
        h, w = img.shape[:2]
        frames.append(img[:h - h % 2, :w - w % 2])  # even dims for libx264
        if k % 25 == 0:
            print(f"  frame {k}/{NFRAMES}")
    plt.close(fig)

    imageio.mimsave("out/chord_lattice.mp4", frames, fps=20, quality=8,
                    macro_block_size=1)
    small = [f[::2, ::2] for f in frames[::2]]
    imageio.mimsave("out/chord_lattice.gif", small, fps=10, loop=0)
    print("wrote out/chord_lattice.mp4 and .gif")


if __name__ == "__main__":
    render()
