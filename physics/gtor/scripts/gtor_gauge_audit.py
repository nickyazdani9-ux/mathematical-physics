#!/usr/bin/env python3
"""SymPy checks for GTOR 2026-11-06 helical gauge/geometry audit.

Algebraic counterexamples only, not experimental or physical validation.
Run: python physics/gtor/scripts/gtor_gauge_audit.py
Requires: sympy
"""
import sympy as sp

u, k, phi = sp.symbols("u k phi", real=True)
a = 2 * sp.pi * k
r2 = (1 + u)**2

# Metric induced by R=(r*cos(v+a*u),r*sin(v+a*u),u).
g_uv = sp.Matrix([[2 + a*a*r2, a*r2], [a*r2, r2]])
# phi=v+a*u => (du,dv)^T = J (du,dphi)^T.
J = sp.Matrix([[1, 0], [-a, 1]])
g = sp.simplify(J.T * g_uv * J)
assert g == sp.diag(2, r2)
assert sp.simplify(g_uv.det() - 2*r2) == 0

# Compute the intrinsic scalar curvature from the Levi-Civita connection.
x = (u, phi)
g_inv = sp.simplify(g.inv())
gamma = [
    [[sp.simplify(sum(
        g_inv[i, m] * (sp.diff(g[m, j], x[l]) +
                       sp.diff(g[m, l], x[j]) -
                       sp.diff(g[j, l], x[m])) / 2
        for m in range(2)))
      for l in range(2)] for j in range(2)]
    for i in range(2)
]
ric = sp.Matrix(2, 2, lambda i, j: sp.simplify(sum(
    sp.diff(gamma[m][i][j], x[m]) -
    sp.diff(gamma[m][i][m], x[j]) +
    sum(gamma[m][m][n] * gamma[n][i][j] -
        gamma[m][j][n] * gamma[n][i][m] for n in range(2))
    for m in range(2)
)))
scalar_R = sp.simplify(sp.trace(g_inv * ric))
assert scalar_R == 0

# A=(g_uv/g_vv)du=2*pi*k du is exact on the interval-times-circle.
# The open-cell integral is 2*pi*k, not gauge-invariant loop holonomy.
A_u = sp.simplify(g_uv[0,1] / g_uv[1,1])
assert sp.simplify(A_u - a) == 0
assert sp.simplify(sp.integrate(A_u, (u, 0, 1)) - a) == 0

# If a separate unit-speed geodesic has angular momentum L=k,
# radial (du/ds)^2 must remain nonnegative for real trajectories.
p_squared = (1 - k**2/(1+u)**2)/2
assert sp.simplify(sp.integrate(p_squared, (u, 0, 1))
                   - (sp.Rational(1, 2) - k**2/4)) == 0
assert p_squared.subs({k: 2, u: 0}) < 0
assert p_squared.subs({k: 2, u: 1}) == 0

print("PASS: k-independent intrinsic metric diag(2,(1+u)^2)")
print("PASS: scalar curvature R=0 on the cone frustum")
print("PASS: the connection is exact and the cell integral is open-path")
print("PASS: unit-speed L=k=2 invalid over the full interval")
