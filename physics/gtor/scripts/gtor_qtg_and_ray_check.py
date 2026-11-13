"""Offline GTOR comparator checks (9 November 2026). Python stdlib only.

Reproduces published baseline equations; no GTOR prediction or cloud usage.
D=5, h(psi)=psi/(1-alpha*psi), Mbar=1, alpha=.1.
"""
import cmath
import math

M_BAR = 1.0
ALPHA = 0.1

def h(psi):
    return psi / (1 - ALPHA * psi)

def h_prime(psi):
    return 1 / (1 - ALPHA * psi)**2

def f5(r):
    return 1 - 2 * M_BAR * r * r / (r**4 + 2 * M_BAR * ALPHA)

for r in (0.08, 0.2, 0.5, 1, 4):
    psi = (1 - f5(r)) / r**2
    assert abs(h(psi) / (2 * M_BAR / r**4) - 1) < 1e-10
    assert h_prime(psi) > 0

# D=5 horizon equation: y**2 - 2*Mbar*y + 2*Mbar*alpha = 0, y=r**2.
assert M_BAR >= 2 * ALPHA
r_minus = math.sqrt(M_BAR - math.sqrt(M_BAR**2 - 2*M_BAR*ALPHA))
r_plus = math.sqrt(M_BAR + math.sqrt(M_BAR**2 - 2*M_BAR*ALPHA))
for root in (r_minus, r_plus):
    assert abs(f5(root)) < 1e-12

D = 5
K0 = 2 * D * (D - 1) / ALPHA**2
assert abs((1 - f5(1e-3)) / 1e-6 - 1 / ALPHA) < 1e-8

def p_plus(phi, scale=1 + 0j):
    state = [scale, scale * cmath.exp(1j * phi)]
    norm = sum(abs(x)**2 for x in state)
    amplitude_plus = (state[0] + state[1]) / math.sqrt(2)
    return abs(amplitude_plus)**2 / norm

for c in (1 + 0j, 3 * cmath.exp(1.1j), -2 + 4j):
    assert abs(p_plus(math.pi / 3, c) - 0.75) < 1e-12
    assert abs(p_plus(2 * math.pi / 3, c) - 0.25) < 1e-12

print(f"D=5 horizons: r-={r_minus:.12f}, r+={r_plus:.12f}")
print(f"D=5 de Sitter core K(0)={K0:.1f} for alpha={ALPHA}")
print("qubit P(+): relative phases pi/3 -> .75; 2pi/3 -> .25")
print("PASS: common complex rescaling invariance; model baselines only")
