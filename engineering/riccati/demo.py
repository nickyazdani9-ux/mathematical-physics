"""
riccati/demo.py — test all modes
=================================

    python -m riccati.demo
"""

import sympy as sp
from sympy import Symbol, Function, ln, pi
from riccati import riccati_solve


def main():
    x = Symbol('x', positive=True)
    u = Symbol('u', positive=True)
    y = Function('y')
    F = Function('F')
    omega = Symbol('omega', positive=True)
    ell = Symbol('ell', positive=True)
    n = Symbol('n', positive=True, integer=True)
    k = Symbol('k', positive=True, integer=True)

    print("\n" + "#" * 50)
    print("  RICCATI ODE SOLVER v0.3")
    print("#" * 50)

    # --- SYMBOLIC (field='real') ---

    print("\n" + "=" * 50)
    print("  field='real' (symbolic)")
    print("=" * 50)

    # 1. Constant coefficients
    print("\n>>> y'' - 3y' + 2y = 0")
    r = riccati_solve(
        y(x).diff(x, 2) - 3*y(x).diff(x) + 2*y(x), x
    )
    r.show()

    # 2. Harmonic oscillator
    print("\n>>> y'' + omega^2 y = 0")
    r = riccati_solve(
        y(x).diff(x, 2) + omega**2 * y(x), x
    )
    r.show()

    # 3. Euler with coords
    print("\n>>> (1+u)^2 F'' + (1+u)F' - 2 ell^2 F = 0")
    r = riccati_solve(
        (1+u)**2 * F(u).diff(u, 2)
        + (1+u) * F(u).diff(u)
        - 2*ell**2 * F(u),
        u, F(u),
        coords=ln(1+u)
    )
    r.show()

    # --- COMPLEX (field='complex') ---

    print("\n" + "=" * 50)
    print("  field='complex' (numerical, in C)")
    print("=" * 50)

    # 4. Bessel in C
    print("\n>>> Bessel: y'' + (1/x)y' + (1 - 4/x^2)y = 0")
    print("    n=2, field='complex'")
    r = riccati_solve(
        y(x).diff(x, 2)
        + y(x).diff(x) / x
        + (1 - 4 / x**2) * y(x),
        x,
        field='complex',
        x_range=(0.5, 30.0),
        z0=1j,
        hint={'level': 2}
    )
    r.show()

    # 5. Airy in C
    print("\n>>> Airy: y'' - x*y = 0")
    print("    field='complex', x_range=(-15, 10)")
    r = riccati_solve(
        y(x).diff(x, 2) - x * y(x), x,
        field='complex',
        x_range=(-15.0, 10.0),
        z0=1j,
        hint={'level': 2}
    )
    r.show()

    # 6. Hermite in C
    print("\n>>> Hermite: y'' - 2x*y' + 4y = 0")
    print("    field='complex'")
    r = riccati_solve(
        y(x).diff(x, 2) - 2*x*y(x).diff(x) + 4*y(x), x,
        field='complex',
        x_range=(-5.0, 5.0),
        z0=0.5j,
        hint={'level': 2}
    )
    r.show()

    # --- COUPLED ---

    print("\n" + "=" * 50)
    print("  coupled=True (coupled manifold)")
    print("=" * 50)

    # 7. Euler on coupled manifold
    print("\n>>> Euler on coupled manifold (g^uv = -pi*k)")
    r = riccati_solve(
        (1+u)**2 * F(u).diff(u, 2)
        + (1+u) * F(u).diff(u)
        - 2*k**2 * F(u),
        u, F(u),
        coords=ln(1+u),
        coupled=True,
        hint={'level': 1, 'coupling': -pi*k}
    )
    r.show()


if __name__ == "__main__":
    main()
