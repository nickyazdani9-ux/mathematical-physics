"""GTOR audit 10: offline controls for inverse-scattering duality and Frolov core.

Dependencies: numpy, scipy, sympy. No internet or cloud services.
Not a reproduction of complete black-hole perturbations or cosmological reconstruction.
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

EPSILON = 0.3

def potential(eta, inverted=False):
    """For D=3, V = a''/a, and dual scale factor a_tilde = 1/a."""
    a = 1 + EPSILON * np.tanh(eta)
    ap = EPSILON / np.cosh(eta) ** 2
    app = -2 * EPSILON * np.tanh(eta) / np.cosh(eta) ** 2
    W = ap / a
    Wp = app / a - W * W
    return W * W - Wp if inverted else W * W + Wp

def bogoliubov(k, inverted=False, T=14):
    # Mode equation psi'' + (k^2 - V) psi = 0; incoming e^{-ik eta}.
    u0 = np.exp(1j * k * T)
    def rhs(eta, z):
        return [z[1], (potential(eta, inverted) - k*k) * z[0]]
    sol = solve_ivp(rhs, (-T, T), [u0, -1j*k*u0],
                    rtol=2e-11, atol=1e-12, method='DOP853')
    assert sol.success, sol.message
    u, up = sol.y[:, -1]
    alpha = np.exp(1j*k*T) * (1j*k*u - up) / (2j*k)
    beta = np.exp(-1j*k*T) * (up + 1j*k*u) / (2j*k)
    return alpha, beta

def verify_scattering():
    for k in (0.4, 0.8, 1.3, 2.2):
        a1, b1 = bogoliubov(k)
        a2, b2 = bogoliubov(k, inverted=True)
        n1, n2 = abs(b1)**2, abs(b2)**2
        assert abs(n1 - n2) < 1e-9
        assert abs(b1 + b2) < 2e-8
        assert abs(abs(a1)**2 - n1 - 1) < 1e-8
        assert abs(abs(a2)**2 - n2 - 1) < 1e-8
        print(f'k={k:.1f} N[a]={n1:.12g} N[1/a]={n2:.12g} '
              f'delta_N={abs(n1-n2):.2e} phase_shift={np.angle(b2/b1):+.8f} rad')

def verify_frolov_core():
    r, M, q, alpha = sp.symbols('r M q alpha', positive=True)
    f = 1 - (2*M*r-q**2)*r**2 / (r**4 + (2*M*r+q**2)*alpha**2)
    R = -sp.diff(f,r,2) - 4*sp.diff(f,r)/r + 2*(1-f)/r**2
    K = sp.diff(f,r,2)**2 + 4*sp.diff(f,r)**2/r**2 + 4*(f-1)**2/r**4
    charged_f2 = sp.simplify(sp.limit((f-1)/r**2,r,0))
    neutral_f2 = sp.simplify(sp.limit((f.subs(q,0)-1)/r**2,r,0))
    assert sp.simplify(charged_f2 - 1/alpha**2) == 0
    assert sp.simplify(neutral_f2 + 1/alpha**2) == 0
    charged_R = sp.simplify(sp.limit(R,r,0))
    neutral_R = sp.simplify(sp.limit(R.subs(q,0),r,0))
    charged_K = sp.simplify(sp.limit(K,r,0))
    neutral_K = sp.simplify(sp.limit(K.subs(q,0),r,0))
    assert sp.simplify(charged_R + 12/alpha**2) == 0
    assert sp.simplify(neutral_R - 12/alpha**2) == 0
    assert sp.simplify(charged_K - 24/alpha**4) == 0
    assert sp.simplify(neutral_K - 24/alpha**4) == 0
    print('Frolov q>0: f=1+r^2/alpha^2+..., R(0)=-12/alpha^2, K(0)=24/alpha^4')
    print('Frolov q=0: f=1-r^2/alpha^2+..., R(0)=+12/alpha^2, K(0)=24/alpha^4')

def verify_bounce_algebra():
    x0, m, x, sigma = sp.symbols('x0 m x sigma', positive=True)
    H2 = (x0+2*m)/x**3 - 2*m*x0/x**4 + 8*sp.pi*sigma/x + 16*sp.pi**2*sigma**2
    assert sp.simplify(H2.subs({x:x0,sigma:-1/(4*sp.pi*x0)})) == 0
    print('Belo bounce: H^2(x0)=0 at sigma_B=-1/(4*pi*x0); acceleration positive if 2m>x0')

if __name__ == '__main__':
    verify_scattering()
    verify_frolov_core()
    verify_bounce_algebra()
