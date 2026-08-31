"""Checks on the master equation, its Baez-Biamonte form, and the quantum lift."""
import itertools as it, time
import numpy as np
from scipy import sparse
import xen

def ok(name, val, tol):
    print(f"  {'PASS' if val < tol else 'FAIL'}  {name:<56} {val:.3e}")

print("\n[1] the generator is infinitesimal stochastic (Baez-Biamonte, Def. of a "
      "stochastic Hamiltonian)")
L = xen.ChordLattice(beta=2.0, drift=(0.3, 0.0, -0.3))
H = L.generator()
ok("max |column sum|   (<-|H = 0, probability conserved)",
   np.abs(np.asarray(H.sum(0)).ravel()).max(), 1e-10)
off = H.copy(); off.setdiag(0); off.eliminate_zeros()
ok("max(0, -min off-diagonal)   (off-diagonals >= 0)", max(0.0, -off.data.min()), 1e-14)
p = np.random.default_rng(0).random(L.M); p /= p.sum()
ok("|| H p  -  matrix-free Hp ||_inf", np.abs(H @ p - L.Hp(p)).max(), 1e-12)

print("\n[2] detailed balance: drift = 0  =>  stationary state is Boltzmann e^{-beta V}")
for beta in (0.5, 2.0, 5.0):
    L0 = xen.ChordLattice(beta=beta, drift=(0, 0, 0))
    pb = np.exp(-beta * L0.V); pb /= pb.sum()
    ok(f"|| H p_Boltzmann ||_inf  at beta = {beta}", np.abs(L0.generator() @ pb).max(), 1e-12)
pb = np.exp(-2.0 * L.V); pb /= pb.sum()
print(f"        drift != 0 breaks it:  ||H p_B|| = {np.abs(H @ pb).max():.3e}"
      f"   (a steady current -- that is the glissando)")

print("\n[3] Baez-Biamonte Fock-space assembly vs. the Markov chain it encodes")
# BB stochastic conventions:  a^dag|n> = |n+1>,  a|n> = n|n-1>,  N = a^dag a.
NS, K = 4, 2
states = [s for s in it.product(range(K + 1), repeat=NS) if sum(s) == K]
index = {s: i for i, s in enumerate(states)}; D = len(states)
rng = np.random.default_rng(1)
rates = {(k, sg): float(rng.uniform(.5, 2.)) for k in range(NS) for sg in (+1, -1)}

H_bb = np.zeros((D, D))                 # sum_tau r ( a^dag^T - a^dag^S ) a^S
for (k, sg), r in rates.items():
    for s in states:
        if s[k] == 0: continue
        c = float(s[k])                                  # a|n> = n|n-1>
        mid = list(s); mid[k] -= 1
        tgt = list(mid); tgt[(k + sg) % NS] += 1         # a^dag^{T}
        src = list(mid); src[k] += 1                     # a^dag^{S}
        H_bb[index[tuple(tgt)], index[s]] += r * c
        H_bb[index[tuple(src)], index[s]] -= r * c
H_mc = np.zeros((D, D))                 # each token hops independently
for s in states:
    for (k, sg), r in rates.items():
        if s[k] == 0: continue
        t = list(s); t[k] -= 1; t[(k + sg) % NS] += 1
        H_mc[index[tuple(t)], index[s]] += r * s[k]
        H_mc[index[s], index[s]] -= r * s[k]
ok("|| H_BaezBiamonte - H_MarkovChain ||_inf", np.abs(H_bb - H_mc).max(), 1e-12)
ok("max |column sum| of H_BaezBiamonte", np.abs(H_bb.sum(0)).max(), 1e-12)

print("\n[4] the chord generator equals its brute-force Petri-net assembly")
N = xen.NPC; n1, n2, n3 = xen.lattice(N); Vv = xen.tension(n1, n2, n3)
Hb = sparse.lil_matrix((N**3, N**3)); drift = (0.3, 0.0, -0.3)
for a in range(N):
    for b in range(N):
        for c in range(N):
            src = a * N * N + b * N + c
            for i, s in it.product(range(3), (+1, -1)):
                v = [a, b, c]; v[i] = (v[i] + s) % N
                tgt = v[0] * N * N + v[1] * N + v[2]
                w = (1 + s * drift[i]) / (1 + np.exp(2.0 * (Vv[tgt] - Vv[src])))
                Hb[tgt, src] += w; Hb[src, src] -= w
ok("|| H_vectorised - H_bruteforce ||_inf", abs(Hb.tocsr() - H).max(), 1e-11)

print("\n[5] quantum lift, J = 0: populations follow the classical chain exactly")
times = np.linspace(0, 1.0, 6)
Lq = xen.ChordLattice(beta=2.0, drift=(0, 0, 0))
p0 = np.zeros(Lq.M); p0[xen.index_of((0, 4, 7))] = 1.0        # start on C-E-G
pc = Lq.evolve_classical(p0, times)
rho0 = np.diag(p0).astype(complex)
pq, _ = Lq.evolve_lindblad(rho0, times, J=0.0, vscale=1.0, kappa=0.0)
ok("max |P_quantum(J=0) - P_classical|", np.abs(pq - pc).max(), 1e-7)

print("\n[6] decoherence limit: kappa -> infinity recovers the classical chain")
t0 = time.time()
for kappa in (0.0, 2.0, 8.0, 32.0, 128.0):
    pq, _ = Lq.evolve_lindblad(rho0, times, J=1.0, vscale=1.0, kappa=kappa)
    tag = "  <- coherent transport dominates" if kappa == 0 else ""
    print(f"      kappa = {kappa:6.1f}    max |P_q - P_cl| = {np.abs(pq - pc).max():.4f}{tag}")
print(f"      ({time.time()-t0:.0f}s)")

print("\n[7] unitary sector: norm and energy conserved; spreading is ballistic")
psi0 = np.zeros(Lq.M, complex); psi0[xen.index_of((0, 4, 7))] = 1.0
tt = np.linspace(0, 4, 41)
P, amps = Lq.evolve_schrodinger(psi0, tt, J=1.0, vscale=1.0)
ok("max |1 - norm|", np.abs(np.linalg.norm(amps, axis=1) - 1).max(), 1e-6)
E = np.array([np.vdot(a, Lq.Hpsi(a, 1.0, 1.0)).real for a in amps])
ok("max |E(t) - E(0)| / |E(0)|", np.abs((E - E[0]) / E[0]).max(), 1e-6)


def spread(P):
    """rms lattice distance from the initial chord, on the 3-torus."""
    n = Lq.n; n0 = np.array([0, 4, 7])[:, None]
    d = np.minimum(np.mod(n - n0, 12), np.mod(n0 - n, 12))
    r2 = (d ** 2).sum(0)
    return np.sqrt(P @ r2)


pcl = Lq.evolve_classical(p0, tt)
sq, sc = spread(P), spread(pcl)
i1, i2 = 4, 12
print(f"      quantum  R(t): {sq[i1]:.3f} -> {sq[i2]:.3f}   slope in log-log = "
      f"{np.log(sq[i2]/sq[i1])/np.log(tt[i2]/tt[i1]):.2f}  (ballistic ~ 1)")
print(f"      classical R(t): {sc[i1]:.3f} -> {sc[i2]:.3f}   slope in log-log = "
      f"{np.log(sc[i2]/sc[i1])/np.log(tt[i2]/tt[i1]):.2f}  (diffusive ~ 0.5)")
print("\nDone.")
