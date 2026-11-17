#!/usr/bin/env python3
"""GTOR Audit 16: offline black-bounce and curvature null-model checks.

Limited independent checks of arXiv:2611.18697 and arXiv:2611.18711.
Not a full paper reproduction or a GTOR-specific prediction.
"""
from math import pi, sqrt, log, isclose

def f_schwarzschild(R, M=1.0):
    return 1.0 - 2.0 * M / R

def photon_potential(R, M=1.0):
    return f_schwarzschild(R, M) / R**2

def second_derivative(func, x, h=0.001):
    return (-func(x+2*h)+16*func(x+h)-30*func(x)
            +16*func(x-h)-func(x-2*h))/(12*h*h)

def lyapunov_from_potential(R, throat, M=1.0):
    h = 1.0-throat*throat/R**2
    return sqrt(-h*f_schwarzschild(R, M)*R**2
                *second_derivative(lambda r: photon_potential(r, M), R)/2)

def kretschmann_local(r, mass_excess, c):
    # Lapse A(r)=1-2*mass_excess/r-2*c, in geometric units.
    A = 1.0-2*mass_excess/r-2*c
    Ap = 2*mass_excess/r**2
    App = -4*mass_excess/r**3
    return ((r*r*App)**2+4*(r*Ap)**2+4*(A-1)**2)/r**4

def logarithmic_slope(fn, r1, r2):
    return log(fn(r2)/fn(r1))/log(r2/r1)

def main():
    M, Rph = 1.0, 3.0
    b = Rph/sqrt(f_schwarzschild(Rph, M))
    assert isclose(b, 3*sqrt(3)*M, rel_tol=1e-13)
    for a in [0, 0.5, 1.5]:
        lnum = lyapunov_from_potential(Rph, a, M)
        lexact = sqrt(1-a*a/(9*M*M))/(3*sqrt(3)*M)
        gamma = pi*lnum*b
        gamma_analytic = pi*sqrt(1-a*a/(9*M*M))
        assert isclose(lnum, lexact, rel_tol=1e-7)
        assert isclose(gamma, gamma_analytic, rel_tol=1e-7)
        print(f'a={a:.1f}, b_c={b:.12f}, lambda_L={lnum:.12f}, gamma={gamma:.12f}')
    tuned = lambda r: kretschmann_local(r, 0.0, 0.1)
    generic = lambda r: kretschmann_local(r, 0.1, 0.1)
    slope_tuned = logarithmic_slope(tuned, 1e-5, 2e-5)
    slope_generic = logarithmic_slope(generic, 1e-5, 2e-5)
    assert isclose(slope_tuned, -4, abs_tol=1e-9)
    assert isclose(slope_generic, -6, abs_tol=2e-4)
    print(f'K tuned exponent: {slope_tuned:.9f} (expected -4)')
    print(f'K generic exponent: {slope_generic:.9f} (expected asymptotically -6)')
    print('PASS: null controls only; no GTOR-specific dynamics evaluated.')

if __name__ == '__main__':
    main()
