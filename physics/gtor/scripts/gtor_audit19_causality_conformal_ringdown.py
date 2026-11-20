"""GTOR Audit 19: independent mathematical null-model checks (standard library).

References: arXiv:2611.26560, 2611.24944, 2611.26591.
These are limited algebraic controls, not paper replication or evidence for GTOR.
"""
import math


def bisect(f, lo, hi, n=120):
    flo, fhi = f(lo), f(hi)
    assert flo * fhi < 0, (flo, fhi)
    for _ in range(n):
        mid = (lo + hi) / 2
        if f(mid) == 0:
            return mid
        if flo * f(mid) < 0:
            hi, fhi = mid, f(mid)
        else:
            lo, flo = mid, f(mid)
    return (lo + hi) / 2


# Extremal Dymnikova magnetic-NED reconstruction; all radii in P=1 units.
u_star = bisect(lambda u: math.exp(u) - 1 - 3*u, 1.5, 2.5)
r_e = u_star**(1/3)
r_lum = (7/3)**(1/3)
r_dec = (4/3)**(1/3)
def v_transverse(r):
    return math.sqrt((3*r**3 - 5)/2)
assert r_dec < r_e < r_lum
assert abs(v_transverse(r_lum) - 1) < 2e-15
assert v_transverse(r_e) < 1 < v_transverse(1.5)
print(f'NED u*={u_star:.10f}, r_e/P={r_e:.10f}, r_lum/P={r_lum:.10f}, '
      f'r_DEC/P={r_dec:.10f}, v_perp(1.5P)={v_transverse(1.5):.10f}')

# g_tilde = S(r) g_Schw, S=(1+L^2/r^2)^2; M=1, r>2.
# Null impact parameter b^2=C/A=r^2/(1-2/r), independent of L.
def photon_b2(r, L):
    f = 1-2/r
    S = (1+L**2/r**2)**2
    return (S*r*r)/(S*f)
for L in (0, 0.5, 1, 2):
    assert abs(photon_b2(3, L) - 27) < 1e-13
    assert photon_b2(2.5, L) > photon_b2(3, L)
    assert photon_b2(3.5, L) > photon_b2(3, L)

# Spinless timelike ISCO: d(J^2)/dr = 0. The numerator, for M=1:
# r^5 - 6r^4 + L^4(4r^2 - 21r + 30).
def isco_polynomial(r, L):
    return r**5 - 6*r**4 + L**4*(4*r*r - 21*r + 30)
results = {L:bisect(lambda r:isco_polynomial(r,L),5,7) for L in (0, 0.5, 1, 2)}
assert abs(results[0]-6) < 1e-12
assert 5 < results[2] < results[1] < results[0]
print('Conformal b_crit/M=%.10f; spinless ISCO radii/M=%s' %
      (math.sqrt(photon_b2(3,2)), ', '.join(f'L={L}: {r:.9f}' for L,r in results.items())))

# Forced-mode convolution toy: y'+gamma*y = exp(-kappa*t), y(0)=0.
# Relative weights of free vs driven exponentials at late time.
kappa, t = 0.25, 100.0
for name,gamma in [('fundamental-like',0.09),('overtone-like',0.40)]:
    driven = math.exp(-kappa*t)
    free = math.exp(-gamma*t)
    response=(driven-free)/(gamma-kappa)
    ratio=free/driven
    assert response > 0
    if gamma < kappa: assert ratio > 1000
    else: assert ratio < 1e-4
    print(f'{name}: gamma={gamma}, free/driven={ratio:.8g}, response={response:.8g}')
print('PASS: 3 limited analytical control families')
