"""GTOR Research Audit 17: independent null-model checks (stdlib only).

These are narrow algebraic controls, not replications of any field equations,
QFT protocol, nonlinear perturbation study or GTOR predictions.
"""
import cmath
import math


def assert_close(actual, expected, atol=1e-12):
    assert abs(actual - expected) < atol, (actual, expected)


def q_abab(b, n, m, np, mp):
    """Tr(A_n B_m A_n' B_m') for rank-one A computational and B columns b."""
    return b[n][m] * b[np][m].conjugate() * b[np][mp] * b[n][mp].conjugate()


def main():
    # Relational scalar-field observable: Phi_f - (int f/int f_R) Phi_R + u*k.
    fint, rint, phi, ref_phi, u, shift = 2.3, 0.7, 1.41, -0.83, 0.11, 3.37
    k = fint / rint
    original = phi - k * ref_phi + k*u
    transformed = phi + shift*fint - k*(ref_phi+shift*rint) + k*u
    qrf_residual = abs(original-transformed)
    assert qrf_residual < 1e-12
    resolution = 0.2
    damping = math.exp(-0.5*(resolution*k)**2)
    assert 0 < damping <= 1

    # Three-outcome mutually unbiased bases: Q=omega^[(n-n')(m-m')]/9.
    omega = cmath.exp(2j*math.pi/3)
    b3 = [[omega**(n*m)/math.sqrt(3) for m in range(3)] for n in range(3)]
    q3 = q_abab(b3, 0, 0, 1, 1)
    q3_theory = omega/9
    assert_close(q3,q3_theory)
    assert_close(q3.imag,math.sqrt(3)/18)
    # Column-wise rephasing of B basis must cancel from the trace.
    phases = [0.21, -0.51, 1.37]
    b3_rephased = [[b3[n][m]*cmath.exp(1j*phases[m]) for m in range(3)] for n in range(3)]
    assert_close(q_abab(b3_rephased, 0, 0, 1, 1),q3)
    # Two-outcome projective measurement produces real extended KD Q.
    theta, phase=0.47,0.81
    b2 = [[math.cos(theta),cmath.exp(1j*phase)*math.sin(theta)],
          [-cmath.exp(-1j*phase)*math.sin(theta), math.cos(theta)]]
    q2 = q_abab(b2,0,0,1,1)
    assert abs(q2.imag)<1e-12

    # Hidden-wormhole near-throat expansion f(r)=-f_T*(r-r_T).
    rT,fT=0.7,2.0
    throat_errors=[]
    for tau in [-2.0,-1.0,-0.3,0.3,1.0,2.0]:
        r=rT+fT*tau*tau/4
        dr_dtau=fT*tau/2
        f=-fT*(r-rT)
        throat_errors.append(abs(dr_dtau**2+f))
    assert max(throat_errors)<1e-12

    # For ds^2=-dτ^2 + a_parallel^2 dρ^2, pρ = a_parallel^2 dρ/dσ = E.
    # Normalization epsilon = -τ'^2 + E^2/a_parallel^2.
    # The preprint's equation (17) prints E rather than E^2 in that second term.
    alpha,H0,a0,E,tau=7.0,1.5,1.4,2.0,0.4
    a=a0*math.exp(alpha*H0*tau/2)
    expected_tauprime_sq=E*E/(a*a)
    printed_tauprime_sq=E/(a*a)
    assert_close(expected_tauprime_sq/printed_tauprime_sq, E)
    print(f"QRF shift residual: {qrf_residual:.3e}")
    print(f"QRF finite-frame Gaussian visibility: {damping:.12f}")
    print(f"three-outcome KD Q: {q3.real:.12f} + {q3.imag:.12f} i")
    print(f"three-outcome rephasing residual: {abs(q_abab(b3_rephased,0,0,1,1)-q3):.3e}")
    print(f"two-outcome imaginary part: {q2.imag:.3e}")
    print(f"near-throat expansion maximum residual: {max(throat_errors):.3e}")
    print(f"null geodesic equation E^2/E ratio for E=2: {expected_tauprime_sq/printed_tauprime_sq:.6f}")
    print("PASS: limited null-model controls only")


if __name__=='__main__':
    main()
