#!/usr/bin/env python3
"""GTOR audit 12: independent null-cone inverse + projective probability baseline.

Reproduce: python -m pip install numpy && python physics/gtor/scripts/gtor_audit12_arrival_fisher.py
No cloud/API calls, no GTOR-specific equations assumed.
"""
import numpy as np

H = np.array([[1.7, 0.14, 0.05],
              [0.14, 1.4, -0.06],
              [0.05, -0.06, 1.2]], dtype=float)
SHIFT = np.array([0.12, -0.08, 0.07])
LAPSE = 1.35
DELTA = LAPSE**2 - SHIFT @ H @ SHIFT
assert DELTA > 0
E = np.eye(3)
DIRECTIONS = np.concatenate((
    E, -E, np.array([E[0] + E[1], E[0] + E[2], E[1] + E[2]])
))

def arrival(v):
    b = H @ SHIFT
    return (b @ v + np.sqrt((b @ v)**2 + DELTA * (v @ H @ v))) / DELTA

def reconstruct(measured):
    """Nine directed arrivals in a fixed 3D observer frame, up to conformal scale."""
    plus, minus, pairs = measured[:3], measured[3:6], measured[6:9]
    omega = (plus - minus) / 2
    a = np.diag(((plus + minus) / 2)**2)
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2))):
        val = ((pairs[k] - omega[i] - omega[j])**2 - a[i, i] - a[j, j]) / 2
        a[i, j] = a[j, i] = val
    r = a - np.outer(omega, omega)
    beta = np.linalg.solve(r, omega)
    lapse_over_sqrt_delta = np.sqrt(1 + omega @ beta)
    return r, beta, lapse_over_sqrt_delta

readings = np.array([arrival(v) for v in DIRECTIONS])
r, beta, lapse_ratio = reconstruct(readings)
err_h = float(np.max(np.abs(r - H / DELTA)))
err_beta = float(np.max(np.abs(beta - SHIFT)))
err_N = float(abs(lapse_ratio - LAPSE / np.sqrt(DELTA)))
assert max(err_h, err_beta, err_N) < 1e-12

rng = np.random.default_rng(20261114)
noise_errors = []
for _ in range(1000):
    noisy = readings + rng.normal(0, 1e-5, size=9)
    rn, _, _ = reconstruct(noisy)
    noise_errors.append(float(np.max(np.abs(rn - H / DELTA))))
med, p95 = np.quantile(noise_errors, [0.5, 0.95])
assert p95 < 1e-3

# The six-outcome Pauli POVM E_(i,±)=(I±sigma_i)/6 has a
# normalized classical Fisher metric sum_i u_i^2/[3(1-n_i^2)]
# on unit-Bloch pure-state tangents u ⟂ n. At the z-pole, a transverse
# tangent has ratio 1/3, saturating the cited d=2, M=6 outcome bound.
n = np.array([1., 1., 1.]) / np.sqrt(3)
u = np.array([1., -1., 0.]) / np.sqrt(2)
fisher_ratio = float(np.sum(u**2 / (3 * (1 - n**2))))
assert abs(fisher_ratio - 0.5) < 1e-12
pole_ratio = 1.0 / 3
outcome_upper_bound = 0.5 * (1 - 2 / 6)
assert abs(pole_ratio - outcome_upper_bound) < 1e-15

# Projective Born probability is invariant under independent nonzero
# complex rescalings of each ray representative.
psi = np.array([1 + 0j, 1j])
phi = np.array([1 + 0j, 1 + 0j])
def born(x, y):
    return float(abs(np.vdot(x, y))**2 /
                 (np.vdot(x, x).real * np.vdot(y, y).real))
err_projective = abs(born(phi, psi) - born((1+2j)*phi, (2-3j)*psi))
assert err_projective < 1e-12

print("PASS: 3+3+3 directed-arrival cone inverse, projective controls")
print(f"max h/Delta, beta, lapse-ratio errors: {err_h:.3e}, {err_beta:.3e}, {err_N:.3e}")
print(f"1000 noise trials (sigma=1e-5): median {med:.9e}; p95 {p95:.9e}")
print(f"Pauli-POVM Fisher: generic {fisher_ratio:.9f}; pole {pole_ratio:.9f}; M=6 bound {outcome_upper_bound:.9f}")
print(f"complex-rescaling Born residual: {err_projective:.3e}")
