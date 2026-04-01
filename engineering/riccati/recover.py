"""
riccati/recover.py — recover y(x) from Riccati fixed points
=============================================================

Two symbolic paths:
  field='real'    — solutions on R
  field='complex' — solutions on C, Re(y) and Im(y) as closed forms

One optional verification tool:
  numerical_verify() — scipy integration to check the symbolic answer
"""

import sympy as sp
from sympy import (
    simplify, exp, sqrt, integrate, limit, oo, zoo, nan, diff,
    Symbol, I, re, im, Piecewise
)
from .classify import Level


def recover_real(fixed_points, auto_coord, P, Q, x, level):
    """
    Symbolic recovery on R.

    L0: y = exp(z* * x)
    L1: y = exp(z_s * s(x)) where z_s is constant in s-coords
    L2: WKB asymptotics from both regimes
    """
    solutions = []
    asymptotics = {}

    if level == Level.L0:
        for fp in fixed_points:
            solutions.append(exp(fp * x))

    elif level == Level.L1:
        for fp in fixed_points:
            if not fp.has(x):
                if auto_coord is not None and auto_coord != x:
                    solutions.append(simplify(exp(fp * auto_coord)))
                else:
                    solutions.append(exp(fp * x))
            else:
                integral = integrate(fp, x)
                if integral is not None:
                    solutions.append(simplify(exp(integral)))

    elif level == Level.L2:
        near_origin = []
        at_infinity = []

        for fp in fixed_points:
            try:
                c = simplify(limit(fp * x, x, 0))
                if c.is_finite and c != 0:
                    near_origin.append(x**c)
            except Exception:
                pass
            try:
                c = limit(fp, x, oo)
                if c.is_finite and c != 0:
                    at_infinity.append(exp(c * x) / sqrt(x))
            except Exception:
                pass

        # try WKB integration, keep only clean results
        for fp in fixed_points:
            try:
                integral = integrate(fp, x)
                if integral is not None and integral != sp.S.Zero:
                    sol = exp(integral)
                    if not sol.has(Piecewise) and not sol.has(sp.acosh):
                        solutions.append(sol)
            except Exception:
                pass

        if near_origin:
            asymptotics['near origin'] = near_origin
        if at_infinity:
            asymptotics['at infinity'] = at_infinity

    return solutions, asymptotics


def recover_complex(fixed_points, auto_coord, P, Q, x, level):
    """
    Symbolic recovery on C.

    For Level 0/1: fixed points are constant or have cleanly
    separable Re/Im parts. Integrate, exponentiate, done.

    For Level 2: the Re/Im split hangs on SymPy because the
    discriminant changes sign. Fall back to real asymptotics.
    The complex path only genuinely helps Level 2 when the
    ODE comes from a coupled manifold (which makes it Level 1).
    """
    solutions = []
    asymptotics = {}

    # Level 2 without coupling: complex field alone doesn't help.
    # The discriminant sqrt(P^2 - 4Q) changes sign, so SymPy
    # can't split re/im. Fall back to real recovery.
    if level == Level.L2:
        return recover_real(fixed_points, auto_coord, P, Q, x, level)

    for fp in fixed_points:
        fp = simplify(fp)

        if not fp.has(x):
            # constant complex fixed point
            if auto_coord is not None and auto_coord != x:
                solutions.append(simplify(exp(fp * auto_coord)))
            else:
                solutions.append(exp(fp * x))
        else:
            # x-dependent: try the Re/Im split with a guard
            try:
                fp_re = sp.re(fp)
                fp_im = sp.im(fp)

                # check these actually evaluated (not left symbolic)
                if fp_re.has(sp.re) or fp_im.has(sp.im):
                    # SymPy couldn't determine re/im, integrate whole
                    integral = integrate(fp, x)
                    if integral is not None:
                        solutions.append(simplify(exp(integral)))
                elif not fp_im.has(x):
                    # Im is constant, Re integrable: the good case
                    int_re = integrate(fp_re, x)
                    if int_re is not None:
                        sol = simplify(exp(int_re)) * exp(I * fp_im * x)
                        solutions.append(simplify(sol))
                else:
                    # both depend on x, integrate together
                    integral = integrate(fp, x)
                    if integral is not None:
                        solutions.append(simplify(exp(integral)))
            except Exception:
                # any failure: try direct integration
                try:
                    integral = integrate(fp, x)
                    if integral is not None:
                        solutions.append(simplify(exp(integral)))
                except Exception:
                    pass

    return solutions, asymptotics


def verify_solution(sol, P, Q, x):
    """Plug y back into y'' + P*y' + Q*y and simplify."""
    yp = diff(sol, x)
    ypp = diff(sol, x, 2)
    return simplify(ypp + P * yp + Q * sol)


def numerical_verify(P, Q, x, x_range, z0=None, n_points=2000):
    """
    Optional numerical verification via scipy.

    Integrates the Riccati dz/dx = -z^2 - Pz - Q on C,
    recovers y = exp(integral z dx), and checks the Riccati
    residual |dz/dx - RHS|.

    Returns a dict with residual stats and the numerical arrays.
    Only import scipy when this is actually called.
    """
    import numpy as np
    from scipy.integrate import solve_ivp

    P_fn = sp.lambdify(x, P, 'numpy')
    Q_fn = sp.lambdify(x, Q, 'numpy')

    x_start, x_end = x_range

    # auto z0 from fixed points if not given
    if z0 is None:
        z_sym = Symbol('z')
        fps = sp.solve(z_sym**2 + P*z_sym + Q, z_sym)
        candidates = []
        for fp in fps:
            try:
                candidates.append(complex(fp.subs(x, x_start)))
            except (TypeError, ValueError):
                pass
        if len(candidates) >= 2:
            z0 = (candidates[0] + candidates[1]) / 2
        elif candidates:
            z0 = candidates[0]
        else:
            z0 = 1j
        if abs(z0.imag) < 1e-10:
            z0 = z0 + 1j

    def rhs(t, z_ri):
        z = z_ri[0] + 1j * z_ri[1]
        dz = -z**2 - complex(P_fn(t)) * z - complex(Q_fn(t))
        return [dz.real, dz.imag]

    x_eval = np.linspace(x_start, x_end, n_points)
    sol = solve_ivp(rhs, (x_start, x_end),
                    [z0.real, z0.imag],
                    t_eval=x_eval, rtol=1e-12, atol=1e-14)

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    z_vals = sol.y[0] + 1j * sol.y[1]
    x_vals = sol.t

    # recover y
    dx = np.diff(x_vals)
    z_mid = (z_vals[:-1] + z_vals[1:]) / 2
    integral = np.concatenate([[0], np.cumsum(z_mid * dx)])
    y_vals = np.exp(integral)

    # Riccati residual via central differences
    n = len(x_vals)
    dz_dx = np.zeros(n - 2, dtype=complex)
    for i in range(1, n - 1):
        dz_dx[i-1] = (z_vals[i+1] - z_vals[i-1]) / \
                     (x_vals[i+1] - x_vals[i-1])

    x_int = x_vals[1:-1]
    z_int = z_vals[1:-1]
    P_int = np.array([complex(P_fn(xi)) for xi in x_int])
    Q_int = np.array([complex(Q_fn(xi)) for xi in x_int])
    rhs_vals = -z_int**2 - P_int * z_int - Q_int
    residual = np.abs(dz_dx - rhs_vals)

    # trim edges
    trim = max(1, len(residual) // 50)
    core = residual[trim:-trim]

    return {
        'max_residual': float(np.max(core)),
        'median_residual': float(np.median(core)),
        'p95_residual': float(np.percentile(core, 95)),
        'x': x_vals,
        'z': z_vals,
        'y': y_vals,
        'y_re': y_vals.real,
        'y_im': y_vals.imag,
        'z0_used': z0,
    }
