"""
riccati/classify.py — ODE and level classification
"""

import sympy as sp
from sympy import simplify, expand, limit, oo, diff, zoo, nan
from enum import Enum


class ODEClass(Enum):
    CONST_COEFF = "constant_coefficient"
    EULER = "euler"
    BESSEL = "bessel"
    AIRY = "airy"
    HERMITE = "hermite"
    LEGENDRE = "legendre"
    LAGUERRE = "laguerre"
    UNKNOWN = "unknown"


class Level(Enum):
    L0 = 0   # globally autonomous
    L1 = 1   # single autonomising coordinate
    L2 = 2   # two incompatible patches


def classify_ode(P, Q, x):
    """
    Identify the ODE class from P(x) and Q(x)
    in y'' + P*y' + Q*y = 0.
    """
    has_P = P.has(x)
    has_Q = Q.has(x)

    if not has_P and not has_Q:
        return ODEClass.CONST_COEFF

    # Euler: P ~ 1/w, Q ~ c/w^2 for w = x or 1+x
    for w in [x, 1 + x]:
        P_test = simplify(P * w)
        Q_test = simplify(Q * w**2)
        if not P_test.has(x) and not Q_test.has(x):
            return ODEClass.EULER

    # Bessel: P = c/x, Q has both constant and 1/x^2 terms
    P_test = simplify(P * x)
    if not P_test.has(x):
        Q_at_inf = limit(Q, x, oo)
        Q_x2 = limit(Q * x**2, x, 0)
        if Q_at_inf not in (0, oo, -oo, zoo, nan):
            if Q_x2 not in (0, oo, -oo, zoo, nan):
                return ODEClass.BESSEL

    # Airy: P = 0, Q linear in x
    if P == 0:
        dQ = diff(Q, x)
        if not dQ.has(x) and dQ != 0:
            return ODEClass.AIRY

    # Hermite: P = -2x, Q = constant
    if simplify(P + 2*x) == 0 and not Q.has(x):
        return ODEClass.HERMITE

    # Legendre: P = -2x/(1-x^2), Q = c/(1-x^2)
    P_test = simplify(P * (1 - x**2))
    Q_test = simplify(Q * (1 - x**2))
    if simplify(P_test + 2*x) == 0 and not Q_test.has(x):
        return ODEClass.LEGENDRE

    return ODEClass.UNKNOWN


def classify_level(P, Q, fixed_points, x, hint=None):
    """
    Determine the autonomy level of the Riccati flow.

    Level 0: fixed points are constant
    Level 1: single regime (one autonomising coordinate)
    Level 2: two incompatible regimes (special function)

    hint: dict with optional 'level' key to override.
    """
    if hint and 'level' in hint:
        return Level(hint['level'])

    if not any(fp.has(x) for fp in fixed_points):
        return Level.L0

    # discriminant sign change => regime change => Level 2
    disc = simplify(P**2 - 4*Q)
    if disc.has(x):
        try:
            d_zero = disc.subs(x, 0)
            d_large = limit(disc, x, oo)
            if d_zero.is_real and d_large.is_real:
                if (d_zero > 0 and d_large < 0) or \
                   (d_zero < 0 and d_large > 0):
                    return Level.L2
        except Exception:
            pass

    # competing constant at infinity => Level 2
    for fp in fixed_points:
        lim_inf = limit(fp, x, oo)
        if lim_inf not in (0, oo, -oo, zoo, nan, sp.zoo):
            if lim_inf.is_number and lim_inf != 0:
                lim_zero = limit(fp * x, x, 0)
                if lim_zero not in (0, oo, -oo, zoo, nan):
                    return Level.L2

    return Level.L1
