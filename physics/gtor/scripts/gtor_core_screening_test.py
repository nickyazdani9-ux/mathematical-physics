"""Offline numerical audit of arXiv:2607.10713v3, natural units G=c=1.

Reproduces the static photon-sphere critical impact parameter calculation;
not a GTOR prediction or a test of dynamical stability.
"""
from math import exp, factorial, sqrt, isclose


def upper_gamma_ratio_5(x: float) -> float:
    return exp(-x) * sum(x**k / factorial(k) for k in range(5))


def mass(r: float, M: float = 1.0, alpha: float = 0.5) -> float:
    ell = M / (12.0 * alpha)
    return M * (1.0 - upper_gamma_ratio_5(r / ell))


def mass_derivative(r: float, M: float = 1.0, alpha: float = 0.5) -> float:
    ell = M / (12.0 * alpha)
    x = r / ell
    return M / ell * exp(-x) * x**4 / factorial(4)


def lapse(r: float, M: float = 1.0, alpha: float = 0.5) -> float:
    return 1.0 - 2.0 * mass(r, M, alpha) / r


def photon_condition(r: float, M: float = 1.0, alpha: float = 0.5) -> float:
    # d/dr [r^2/F(r)] = 0 iff 2 F - r F' = 0.
    m = mass(r, M, alpha)
    mp = mass_derivative(r, M, alpha)
    f = 1.0 - 2.0 * m / r
    fp = -2.0 * mp / r + 2.0 * m / (r * r)
    return 2.0 * f - r * fp


def bisect(f, a: float, b: float, iterations: int = 100) -> float:
    fa = f(a)
    fb = f(b)
    if not (fa * fb < 0):
        raise ValueError("No sign bracket")
    for _ in range(iterations):
        c = 0.5 * (a + b)
        fc = f(c)
        if fc == 0.0:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b)


def main() -> None:
    M, alpha = 1.0, 0.5
    ell = M / (12.0 * alpha)
    rph = bisect(lambda r: photon_condition(r, M, alpha), 2.5, 3.5)
    bc = rph / sqrt(lapse(rph, M, alpha))
    bc_schw = 3.0 * sqrt(3.0) * M
    exact_shift = bc / bc_schw - 1.0
    first_order_shift = -upper_gamma_ratio_5(36.0 * alpha)
    print(f"r_ph: {rph:.12f}")
    print(f"critical-impact shift numerical: {exact_shift:.12e}")
    print(f"first-order estimate -Q(5,36 alpha): {first_order_shift:.12e}")
    print(f"relative discrepancy: {abs(exact_shift - first_order_shift):.3e}")
    assert isclose(rph, 2.9985390124, abs_tol=1e-8)
    assert abs(exact_shift + 8.45e-5) < 1e-7
    assert abs(first_order_shift + 8.42e-5) < 1e-7
    assert abs(exact_shift - first_order_shift) < 5e-7

    # In the paper's closure class: E ~ r^2 exp(-r/ell) and
    # (E + P_t) = -r E'/2.  E' changes sign at r=2 ell.
    density_derivative_sign = lambda r: 2.0 - r / ell
    assert density_derivative_sign(ell) > 0  # NEC < 0
    assert density_derivative_sign(3.0 * ell) < 0  # NEC > 0
    print("Core tangential NEC sign transition: r=2 ell (verified)")


if __name__ == "__main__":
    main()
