"""Turn both dynamics into sound.

classical: a Gillespie realisation of the master equation on the chord lattice.
quantum:   evolve |psi> unitarily for tau, measure in the chord basis (Born rule),
           collapse, repeat.  The score IS the measurement record.
"""
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
import xen

SR = 44100
REG = (24, 36, 48)          # each voice sits in its own octave: C1, C2, C3 offsets
BETA, J = 2.5, 1.0


def synth(times, pitches, dur, gliss_ms=45.0, amp=0.26):
    """pitches: (3, K) semitone values held from times[k] to times[k+1]."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    left = np.zeros(n); right = np.zeros(n)
    pan = (0.30, 0.5, 0.70)
    for v in range(3):
        idx = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 1)
        semis = pitches[v][idx] + REG[v]
        k = max(1, int(gliss_ms * 1e-3 * SR))
        semis = uniform_filter1d(semis.astype(float), k)      # portamento
        f = 8.1758 * 2 ** (semis / 12.0)                      # MIDI -> Hz
        phase = 2 * np.pi * np.cumsum(f) / SR
        tone = np.zeros(n)
        for h, a in ((1, 1.0), (2, 0.42), (3, 0.22), (4, 0.13), (5, 0.07)):
            tone += a * np.sin(h * phase)
        tone /= 1.85
        # gentle articulation at each event boundary
        env = np.ones(n)
        ev = np.searchsorted(t, times)
        ev = ev[(ev > 0) & (ev < n)]
        dip = np.zeros(n); dip[ev] = 1.0
        dip = uniform_filter1d(dip, max(1, int(0.012 * SR))) * (0.012 * SR)
        env *= 1.0 - 0.35 * np.clip(dip, 0, 1)
        env *= np.clip(t / 0.25, 0, 1) * np.clip((dur - t) / 0.6, 0, 1)
        left += amp * tone * env * (1 - pan[v])
        right += amp * tone * env * pan[v]
    x = np.stack([left, right], 1)
    x /= max(1e-9, np.abs(x).max() / 0.93)
    return x


def classical_score(dur=22.0, seed=3):
    L = xen.ChordLattice(beta=BETA, drift=(0, 0, 0), gamma=1.6)
    rng = np.random.default_rng(seed)
    idx = xen.index_of((0, 4, 7))
    tot = L._wtot
    ts, chords = [0.0], [np.array([0, 4, 7])]
    t = 0.0
    keys = list(L.perm.keys())
    while t < dur:
        rates = np.array([L.w[k][idx] for k in keys])
        R = rates.sum()
        t += rng.exponential(1.0 / R)
        k = keys[rng.choice(len(keys), p=rates / R)]
        idx = L.perm[k][idx]
        ts.append(min(t, dur)); chords.append(L.n[:, idx].copy())
    return np.array(ts), np.array(chords).T


def quantum_score(dur=22.0, tau=0.55, seed=3):
    """Repeated Born-rule measurement of the coherently evolving chord state."""
    L = xen.ChordLattice(beta=BETA, drift=(0, 0, 0))
    rng = np.random.default_rng(seed)
    idx = xen.index_of((0, 4, 7))
    ts, chords = [0.0], [np.array([0, 4, 7])]
    t = 0.0
    grid = np.array([0.0, tau])
    while t < dur:
        psi = np.zeros(L.M, complex); psi[idx] = 1.0
        P, _ = L.evolve_schrodinger(psi, grid, J=J, vscale=1.0)
        p = np.maximum(P[-1], 0); p /= p.sum()
        idx = rng.choice(L.M, p=p)                 # collapse
        t += tau
        ts.append(min(t, dur)); chords.append(L.n[:, idx].copy())
    return np.array(ts), np.array(chords).T


if __name__ == "__main__":
    import os
    os.makedirs("out", exist_ok=True)
    for name, fn in (("classical", classical_score), ("quantum", quantum_score)):
        ts, ch = fn()
        x = synth(ts, ch, 22.0)
        wavfile.write(f"out/xenakis_{name}.wav", SR, (x * 32767).astype(np.int16))
        print(f"out/xenakis_{name}.wav   {len(ts)} events")
