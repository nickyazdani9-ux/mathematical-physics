"""GTOR Audit 13: gauge-invariant discrete projector-loop check.

Controls only: standard CP^1 geometric phase. Not a GTOR-specific model.
Python 3 / numpy; run: python gtor_audit13_projector_holonomy.py
"""
import numpy as np

THETA = np.pi / 3
TARGET = np.pi * (1 - np.cos(THETA))


def loop(n, seed=20261115, reverse=False):
    phi = 2 * np.pi * np.arange(n) / n
    psi = np.column_stack((np.full(n, np.cos(THETA / 2), complex),
                           np.sin(THETA / 2) * np.exp(1j * phi)))
    if reverse:
        psi = psi[::-1]
    # Rank-one Hermitian projectors are invariant under local U(1) rephasing.
    P = np.einsum('ni,nj->nij', psi, psi.conj())
    idem = max(np.linalg.norm(p @ p - p) for p in P)
    # Forward Bargmann overlap equals the cyclic projector trace in the
    # corresponding orientation; reverse matrix product flips its phase.
    w = np.prod([np.vdot(psi[j], psi[(j+1) % n]) for j in range(n)])
    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, n)
    psi_g = psi * np.exp(1j * phases)[:, None]
    w_g = np.prod([np.vdot(psi_g[j], psi_g[(j+1) % n]) for j in range(n)])
    return float(np.angle(w)), float(abs(w)), float(abs(w_g - w)), float(idem)


if __name__ == '__main__':
    print('N phase_rad absW gauge_delta max_idempotence analytic_error')
    for n in (16, 64, 128, 256):
        phase, mag, gauge_delta, idem = loop(n)
        print(n, f'{phase:.12f}', f'{mag:.12f}', f'{gauge_delta:.3e}',
              f'{idem:.3e}', f'{abs(phase - TARGET):.6e}')
        assert gauge_delta < 1e-12
        assert idem < 1e-12
    forward = loop(256)[0]
    backward = loop(256, reverse=True)[0]
    assert abs(forward + backward) < 1e-12
    assert abs(forward - TARGET) < 1e-4
    print('PASS: rephasing invariant; orientation reversed; convergent to standard Berry phase magnitude.')
