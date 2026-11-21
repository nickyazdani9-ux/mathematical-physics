#!/usr/bin/env python3
"""GTOR audit 15: standard-model null controls, not simulations of GTOR or papers.
Deterministic, offline, Python 3 standard library only.
"""
import cmath
import math

def inner(a, b):
    return sum(x.conjugate() * y for x, y in zip(a, b))

def qubit(theta, scaled=False):
    base = (math.cos(theta / 2), math.sin(theta / 2))
    deriv = (-0.5 * math.sin(theta / 2), 0.5 * math.cos(theta / 2))
    if not scaled:
        return base, deriv
    z = cmath.exp((0.2 + 5j) * theta)
    dz = (0.2 + 5j) * z
    return tuple(z * x for x in base), tuple(dz * x + z * dx for x, dx in zip(base, deriv))

def pure_qfi(state, derivative):
    norm = inner(state, state).real
    return 4 * (inner(derivative, derivative).real / norm -
                abs(inner(state, derivative)) ** 2 / norm ** 2)

def fidelity(a, b):
    return abs(inner(a, b)) ** 2 / (inner(a, a).real * inner(b, b).real)

def trapezoidal(values, dx):
    return dx * (sum(values) - 0.5 * (values[0] + values[-1]))

def ab_holonomy(n=4096, kappa=1 / 3):
    step = 2 * math.pi / n
    phi = [step * i for i in range(n + 1)]
    # Locally flat but globally non-exact on punctured plane.
    # Single-valued periodic gauge chi(phi)=0.3 sin(2phi)+0.2 cos(3phi).
    dchi = [0.6 * math.cos(2*p) - 0.6 * math.sin(3*p) for p in phi]
    phase = trapezoidal([kappa + a for a in dchi], step)
    exact = trapezoidal(dchi, step)
    return phase, exact

def toy_eigenvalues(detuning, coupling):
    # Illustrative Hermitian two-mode matrix, NOT a black-hole QNM problem.
    radius = math.sqrt((detuning / 2)**2 + coupling**2)
    return -radius, radius

def main():
    theta = 0.7
    psi, dpsi = qubit(theta)
    psi_scaled, dpsi_scaled = qubit(theta, True)
    f1, f2 = pure_qfi(psi, dpsi), pure_qfi(psi_scaled, dpsi_scaled)
    assert abs(f1 - 1) < 1e-12 and abs(f2 - 1) < 1e-11
    for eps in (0.02, 0.01, 0.005):
        a, _ = qubit(theta, True)
        b, _ = qubit(theta + eps, True)
        f_bures = 8 * (1 - math.sqrt(fidelity(a, b))) / eps**2
        f_exact = 8 * (1 - math.cos(eps / 2)) / eps**2
        assert abs(f_bures - f_exact) < 1e-9

    phase, exact = ab_holonomy()
    assert abs(phase - 2 * math.pi / 3) < 1e-10
    assert abs(exact) < 1e-10
    p_plus = 0.5 * (1 + math.cos(phase))
    assert abs(p_plus - 0.25) < 1e-10

    detuning = 0.2
    shift1 = toy_eigenvalues(detuning, 0.01)[1] - detuning / 2
    shift2 = toy_eigenvalues(detuning, 0.005)[1] - detuning / 2
    ratio = shift1 / shift2
    assert 3.9 < ratio < 4.1
    split = toy_eigenvalues(0, 0.01)[1] - toy_eigenvalues(0, 0.01)[0]
    assert abs(split - 0.02) < 1e-12

    print("QFI normalized:", f"{f1:.12f}")
    print("QFI under theta-dependent complex scaling:", f"{f2:.12f}")
    print("Closed-loop flat-connection phase:", f"{phase:.12f}", "rad")
    print("Periodic exact-gauge phase:", f"{exact:.3e}", "rad")
    print("Interference probability:", f"{p_plus:.12f}")
    print("Off-resonant g-vs-g/2 shift ratio:", f"{ratio:.8f}")
    print("Resonant splitting:", f"{split:.12f}")

if __name__ == "__main__":
    main()
