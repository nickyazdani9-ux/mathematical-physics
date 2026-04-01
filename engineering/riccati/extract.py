"""
riccati/extract.py — extract P, Q and find autonomising coordinates
"""

import sympy as sp
from sympy import simplify, expand, diff, ln, solve, Symbol


def extract_P_Q(ode_expr, x, y_func=None):
    """
    Extract P(x), Q(x) from y'' + P*y' + Q*y = 0.

    ode_expr: LHS of ODE = 0
    x: independent variable
    y_func: e.g. y(x) or F(u). Auto-detected if None.
    """
    if y_func is None:
        applied = set()
        for sub in ode_expr.atoms(sp.Derivative):
            applied.add(sub.args[0])
        for sub in ode_expr.atoms(sp.Function):
            if hasattr(sub, 'args') and x in sub.free_symbols:
                applied.add(sub)
        if not applied:
            raise ValueError("Could not find dependent function in ODE")
        y_func = applied.pop()

    yp = diff(y_func, x)
    ypp = diff(y_func, x, 2)

    expr = expand(ode_expr)

    coeff_ypp = expr.coeff(ypp)
    if coeff_ypp == 0:
        raise ValueError("No y'' term found")

    expr_norm = expand(expr / coeff_ypp)
    remainder = expand(expr_norm - ypp)

    P = simplify(remainder.coeff(yp))
    remainder2 = expand(remainder - P * yp)
    Q = simplify(remainder2 / y_func)

    return P, Q


def transform_to_coords(P, Q, x, s_expr):
    """
    Transform P(x), Q(x) into s-coordinates where s = s_expr(x).

    Given y'' + P*y' + Q*y = 0 in x, compute P_s, Q_s such that
    Y'' + P_s*Y' + Q_s*Y = 0 in s, where primes are d/ds.

    The chain rule gives:
        dy/dx = (ds/dx) * dY/ds
        d2y/dx2 = (ds/dx)^2 * d2Y/ds2 + (d2s/dx2) * dY/ds

    So:
        P_s = (P * ds/dx + d2s/dx2) / (ds/dx)^2
        Q_s = Q / (ds/dx)^2

    Returns (P_s, Q_s, ds_dx) all as functions of x.
    To get them purely in s, the caller inverts s(x).
    """
    ds_dx = diff(s_expr, x)
    d2s_dx2 = diff(s_expr, x, 2)

    P_s = simplify((P * ds_dx + d2s_dx2) / ds_dx**2)
    Q_s = simplify(Q / ds_dx**2)

    return P_s, Q_s, ds_dx


def find_autonomising(P, Q, x, coords=None):
    """
    Find or verify the autonomising coordinate s(x).

    coords: user-supplied s(x). If None, tries standard candidates.
    Returns the coordinate expression or None.
    """
    if coords is not None:
        return coords

    # s = x (already autonomous)
    if not P.has(x) and not Q.has(x):
        return x

    # try s = ln(w) for w in {1+x, x}
    for w in [1 + x, x]:
        Q_test = simplify(Q * w**2)
        P_test = simplify(P * w)
        if not Q_test.has(x):
            if not P_test.has(x) or simplify(P_test - 1) == 0:
                return ln(w)

    return None
