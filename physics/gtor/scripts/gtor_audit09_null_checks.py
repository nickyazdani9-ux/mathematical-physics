#!/usr/bin/env python3
"""GTOR Audit 09: small analytic null controls, stdlib only, not model replication."""
from math import isclose, sqrt

# Lu–Yu Eq. (9), resonant harmonic case, omega = Omega = 1.
def naive_qfi(g, theta):
    return 2*(1 + g*g)/(2 + g*g)**2 + 4*(theta*theta - theta)/(2 + g*g)

assert isclose(naive_qfi(1, 0), 4/9, abs_tol=1e-14)
assert isclose(naive_qfi(1, 1), 4/9, abs_tol=1e-14)
assert isclose(naive_qfi(1, .5), 1/9, abs_tol=1e-14)
print("QFI F(0),F(.5),F(1):", *(f"{naive_qfi(1,t):.9f}" for t in (0,.5,1)))

# Arean et al.: merely the dS stationary-point algebra, not a phase-diagram solution.
def H2(v):
    return -1 + 1/(6*v)
assert H2(1/12) == 1 and isclose(H2(1/6), 0, abs_tol=1e-14) and H2(1/4)<0
print("dS H^2(v=1/12,1/6,1/4):", *(f"{H2(v):.9f}" for v in (1/12,1/6,1/4)))

# Schwarzschild toy, M=1, BL time T=t: N=sqrt(1-2/r).
# Invariant acceleration of normalized preferred normal: a_mu a^mu = (f')^2/(4f).
# Curvature K = 48 M^2/r^6; finite at horizon.
def scalars(eps):
    r=2 + eps
    f=1-2/r
    return (2/r**2)**2/(4*f), 48/r**6
for eps in (1e-2,1e-3,1e-4,1e-5):
    a2, K = scalars(eps)
    print(f"BL eps={eps:.0e}: accel_sq={a2:.8f}, eps*accel_sq={eps*a2:.8f}, K={K:.8f}")
a2, K = scalars(1e-5)
assert isclose(1e-5*a2,1/8,rel_tol=3e-4)
assert isclose(K,.75,rel_tol=5e-5)

# Bohmer–Fiorini limiting radius/curvature algebra, lambda=-9.
ac=2/sqrt(9)
K_lim=4/ac**4
assert isclose(ac,2/3) and isclose(K_lim,20.25)
print(f"BI a_c={ac:.9f}, limiting K={K_lim:.9f}")
print("All elementary null checks passed. No numerical gravity/QM simulation performed.")
