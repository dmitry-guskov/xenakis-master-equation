"""
Xenakis master equation on the 3-note chord lattice, and its quantum lift.

MODEL
-----
State: a chord  n = (n1,n2,n3) in Z_N^3   (N = 12 pitch classes, 3 voices).
Elementary event: one voice slides one semitone,  n -> n + s e_i,  s = +-1.

Master equation (the generator of sound graphs):

    d/dt P(n,t) = sum_{i,s} [ w_i^s(n - s e_i) P(n - s e_i, t) - w_i^s(n) P(n,t) ]

    w_i^s(n) = gamma (1 + s a_i) / ( 1 + exp( beta [ V(n + s e_i) - V(n) ] ) )

  gamma : tempo (events per unit time per voice)
  a_i   : glissando drift of voice i (Xenakis's signed velocity), |a_i| < 1
  beta  : inverse temperature. beta = 0 is a pure Brownian pitch cloud
          (Pithoprakta); beta -> inf freezes onto the consonant manifold.
  V     : chord tension (below).

These are Glauber / heat-bath rates: bounded by gamma(1+|a|), and for a = 0 they
satisfy detailed balance w.r.t. exp(-beta V) exactly, since
    w(n->m)/w(m->n) = (1 + e^{-beta dV}) / (1 + e^{+beta dV}) = e^{-beta dV}.
A nonzero drift a multiplies the ratio by (1+a)/(1-a): detailed balance breaks
and a steady probability current runs around the pitch circle.  That current IS
the glissando.

TENSION
-------
V(n) = - lambda sum_{i<j} c(n_i - n_j)          two-body consonance
       + U      sum_{i<j} delta(n_i, n_j)       unison / doubling penalty
       - mu     max_r sum_{pc in set(n)} h(pc - r)   three-body root support

c: ratio-simplicity of the interval class, 1/sqrt(pq), symmetrised over
   inversion.  h: weight of a pitch class in the harmonic series over a root
   (virtual pitch).  With mu = 0 the ground state is a quintal stack; the
   three-body term is what makes the major triad the ground state.
"""
import numpy as np
from scipy import sparse

NPC = 12

RATIOS = {0: (1, 1), 1: (16, 15), 2: (9, 8), 3: (6, 5), 4: (5, 4), 5: (4, 3),
          6: (45, 32), 7: (3, 2), 8: (8, 5), 9: (5, 3), 10: (9, 5), 11: (15, 8)}


def consonance_kernel():
    raw = np.array([1.0 / np.sqrt(p * q) for d, (p, q) in sorted(RATIOS.items())])
    c = np.maximum(raw, np.roll(raw[::-1], 1))    # c(d) <- max(c(d), c(12-d))
    c[0] = 0.0                                    # unison's physics is the U term
    return c


def harmonic_template():
    """harmonics 1..8 over a root land on pitch classes 0,0,7,0,4,7,10,0."""
    h = np.zeros(NPC)
    for k, pc in enumerate([0, 0, 7, 0, 4, 7, 10, 0], start=1):
        h[pc] += 1.0 / k
    return h


C = consonance_kernel()
HTEMPL = harmonic_template()


def tension(n1, n2, n3, lam=1.0, U=1.5, mu=1.0):
    v = 0.0
    for a, b in ((n1, n2), (n1, n3), (n2, n3)):
        d = np.mod(a - b, NPC)
        v = v - lam * C[d] + U * (d == 0)
    if mu:
        roots = np.arange(NPC).reshape(-1, *([1] * np.ndim(n1)))
        new2 = np.mod(n2 - n1, NPC) != 0
        new3 = (np.mod(n3 - n1, NPC) != 0) & (np.mod(n3 - n2, NPC) != 0)
        supp = (HTEMPL[np.mod(n1 - roots, NPC)]
                + HTEMPL[np.mod(n2 - roots, NPC)] * new2
                + HTEMPL[np.mod(n3 - roots, NPC)] * new3)
        v = v - mu * supp.max(axis=0)
    return v


def lattice(N=NPC):
    g = np.arange(N)
    n1, n2, n3 = np.meshgrid(g, g, g, indexing="ij")
    return n1.ravel(), n2.ravel(), n3.ravel()


def index_of(n, N=NPC):
    return n[0] * N * N + n[1] * N + n[2]


class ChordLattice:
    """Everything the dynamics needs: shift permutations, tension, jump rates."""

    def __init__(self, N=NPC, beta=2.0, gamma=1.0, drift=(0.0, 0.0, 0.0),
                 lam=1.0, U=1.5, mu=1.0):
        self.N, self.beta, self.gamma, self.drift = N, beta, gamma, tuple(drift)
        self.n = np.stack(lattice(N))
        self.M = self.n.shape[1]
        self.V = tension(self.n[0], self.n[1], self.n[2], lam, U, mu)
        # perm[(i, s)][n] = index of chord n + s e_i
        self.perm = {}
        for i in range(3):
            for s in (+1, -1):
                m = self.n.copy()
                m[i] = np.mod(m[i] + s, N)
                self.perm[(i, s)] = index_of(m, N)
        # Glauber rates w_i^s(n)
        self.w = {}
        for (i, s), p in self.perm.items():
            dV = self.V[p] - self.V
            self.w[(i, s)] = gamma * (1 + s * self.drift[i]) / (1.0 + np.exp(beta * dV))
        # cached pieces for the Lindblad right-hand side
        self._Vrow = self.V.reshape(N, N, N, 1, 1, 1)
        self._wtot = sum(self.w.values())
        self._grow = {(i, s): np.sqrt(self.w[(i, s)][self.perm[(i, -s)]])
                      for i in range(3) for s in (+1, -1)}

    # -- classical ---------------------------------------------------------
    def generator(self):
        """H in Baez-Biamonte form: sum_tau r_tau ( |T(tau)><S(tau)| - |S><S| )."""
        rows, cols, vals = [], [], []
        idx = np.arange(self.M)
        for (i, s), p in self.perm.items():
            r = self.w[(i, s)]
            rows += [p, idx]; cols += [idx, idx]; vals += [r, -r]
        return sparse.coo_matrix((np.concatenate(vals),
                                  (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(self.M, self.M)).tocsr()

    def Hp(self, p):
        """H @ p without forming H (gain minus loss)."""
        out = np.zeros_like(p)
        for (i, s), perm in self.perm.items():
            f = self.w[(i, s)] * p            # flux leaving each chord
            np.add.at(out, perm, f)           # perm is a permutation -> safe
            out -= f
        return out

    def evolve_classical(self, p0, times):
        p = np.asarray(p0, float).copy()
        out = np.empty((len(times), self.M)); out[0] = p
        fast = max(w.max() for w in self.w.values()) * 6
        for k in range(1, len(times)):
            dt = times[k] - times[k - 1]
            ns = max(1, int(np.ceil(dt * fast / 0.5))); h = dt / ns
            for _ in range(ns):
                k1 = self.Hp(p); k2 = self.Hp(p + .5 * h * k1)
                k3 = self.Hp(p + .5 * h * k2); k4 = self.Hp(p + h * k3)
                p = p + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            p = np.maximum(p, 0); p /= p.sum(); out[k] = p
        return out

    # -- quantum -----------------------------------------------------------
    def Hpsi(self, psi, J=1.0, vscale=1.0):
        """Hq = -J sum_i (S_i + S_i^dag) + vscale * Vhat  applied to a vector."""
        out = (vscale * self.V) * psi
        for (i, s), perm in self.perm.items():
            out -= J * psi[perm]              # (S_i^{-s} psi)[n] = psi[n + s e_i]
        return out

    def evolve_schrodinger(self, psi0, times, J=1.0, vscale=1.0):
        psi = np.asarray(psi0, complex).copy()
        out = np.empty((len(times), self.M)); out[0] = np.abs(psi) ** 2
        amps = np.empty((len(times), self.M), complex); amps[0] = psi
        fast = vscale * np.abs(self.V).max() + 6 * J
        for k in range(1, len(times)):
            dt = times[k] - times[k - 1]
            ns = max(1, int(np.ceil(dt * fast / 0.08))); h = dt / ns
            for _ in range(ns):
                f = lambda v: -1j * self.Hpsi(v, J, vscale)
                k1 = f(psi); k2 = f(psi + .5 * h * k1)
                k3 = f(psi + .5 * h * k2); k4 = f(psi + h * k3)
                psi = psi + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            psi /= np.linalg.norm(psi)
            out[k] = np.abs(psi) ** 2; amps[k] = psi
        return out, amps

    def drho(self, r, J, vscale, kappa, dissip):
        """Lindbladian applied to a density matrix (index-based, no sparse mats).

        d rho = -i[Hq, rho]
                + sum_tau ( L_tau rho L_tau^dag - 1/2 {L_tau^dag L_tau, rho} )
                - kappa * offdiag(rho),
        with L_tau = sum_n sqrt(w_i^s(n)) |n + s e_i><n|  (so L^dag L = diag w).
        """
        # Every operator here is a cyclic shift in one lattice coordinate, so
        # each index permutation is an np.roll on the 6-index view of rho.
        N, M = self.N, self.M
        r6 = r.reshape(N, N, N, N, N, N)
        # Hq @ rho   (rho Hermitian => rho @ Hq = (Hq @ rho)^dag)
        A6 = (vscale * self._Vrow) * r6
        for i in range(3):
            for s in (+1, -1):
                A6 = A6 - J * np.roll(r6, -s, axis=i)   # (S^{-s} rho)[n] = rho[n+s e_i]
        A = A6.reshape(M, M)
        d = -1j * (A - A.conj().T)
        if dissip:
            for i in range(3):
                for s in (+1, -1):
                    g = self._grow[(i, s)]                # sqrt(w) gathered at src
                    d += ((g.reshape(N, N, N, 1, 1, 1) * g.reshape(1, 1, 1, N, N, N))
                          * np.roll(np.roll(r6, s, axis=i), s, axis=3 + i)
                          ).reshape(M, M)
            tot = self._wtot
            d -= 0.5 * (tot[:, None] * r + r * tot[None, :])
        if kappa:
            off = r.copy()
            np.fill_diagonal(off, 0.0)
            d -= kappa * off
        return d

    def evolve_lindblad(self, rho0, times, J=1.0, vscale=1.0, kappa=0.0,
                        dissip=True, keep_rho=False):
        rho = np.asarray(rho0, complex).copy()
        out = np.empty((len(times), self.M)); out[0] = np.real(np.diag(rho))
        wmax = max(w.max() for w in self.w.values())
        fast = max(vscale * np.abs(self.V).max() + 6 * J, kappa, 6 * wmax, 1.0)
        for k in range(1, len(times)):
            dt = times[k] - times[k - 1]
            ns = max(1, int(np.ceil(dt * fast / 0.4))); h = dt / ns
            for _ in range(ns):
                f = lambda r: self.drho(r, J, vscale, kappa, dissip)
                k1 = f(rho); k2 = f(rho + .5 * h * k1)
                k3 = f(rho + .5 * h * k2); k4 = f(rho + h * k3)
                rho += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            rho = 0.5 * (rho + rho.conj().T)
            rho /= np.real(np.trace(rho))
            out[k] = np.real(np.diag(rho))
        return out, (rho if keep_rho else None)


# ------------------------------------------------------- sample sound graphs
def gillespie(N=48, beta=2.0, gamma=1.0, drift=(0.55, 0.0, -0.55), lam=1.0,
              U=1.5, mu=1.0, start=(21, 25, 28), T=24.0, rng=None):
    """Exact sample path of the same master equation on a wide register Z_N
    (real pitches; V still depends on intervals mod 12).  A sound graph."""
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.array(start, int); t = 0.0
    ts, xs = [0.0], [x.copy()]
    Vx = lambda v: tension(v[0], v[1], v[2], lam, U, mu)
    while t < T:
        moves, rates = [], []
        v0 = Vx(x)
        for i in range(3):
            for s in (+1, -1):
                y = x.copy(); y[i] += s
                if not (0 <= y[i] < N):
                    continue
                moves.append(y)
                rates.append(gamma * (1 + s * drift[i])
                             / (1.0 + np.exp(beta * (Vx(y) - v0))))
        rates = np.array(rates); R = rates.sum()
        t += rng.exponential(1.0 / R)
        x = moves[rng.choice(len(moves), p=rates / R)]
        ts.append(min(t, T)); xs.append(x.copy())
    return np.array(ts), np.array(xs).T
