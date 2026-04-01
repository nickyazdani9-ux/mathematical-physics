"""
riccati/solver.py — main entry point
======================================

    riccati_solve(ode, x)                     # symbolic on R
    riccati_solve(ode, x, field='complex')    # symbolic on C
    riccati_solve(ode, x, numerical=True)     # scipy verification
"""

import sympy as sp
from sympy import Symbol, simplify, solve, pprint, re, im
from dataclasses import dataclass, field
from typing import Optional, List, Any

from .classify import ODEClass, Level, classify_ode, classify_level
from .extract import extract_P_Q, find_autonomising, transform_to_coords
from .recover import (
    recover_real, recover_complex, verify_solution, numerical_verify
)


@dataclass
class RiccatiResult:
    """Everything the solver produces."""
    P: Any = None
    Q: Any = None
    P_s: Any = None
    Q_s: Any = None
    riccati: Any = None
    fixed_points: List = field(default_factory=list)
    fixed_points_s: List = field(default_factory=list)
    level: Optional[Level] = None
    ode_class: ODEClass = ODEClass.UNKNOWN
    solutions: List = field(default_factory=list)
    asymptotics: dict = field(default_factory=dict)
    autonomising: Any = None
    coords: Any = None
    residuals: List = field(default_factory=list)
    notes: List = field(default_factory=list)
    field_type: str = 'real'
    numerical: Optional[dict] = None

    def to_exprs(self):
        """
        Return solutions as a tuple of sympy expressions.

        L0/L1: exact closed-form solutions.
        L2 (real): WKB approximations if clean, else asymptotics.
        L2 (complex): exact via coupled-coordinate structure.
        """
        if self.solutions:
            return tuple(self.solutions)
        out = []
        for forms in self.asymptotics.values():
            out.extend(forms)
        if out:
            return tuple(out)
        return ()

    def show(self):
        print("\n" + "=" * 50)
        print("  RICCATI FLOW ANALYSIS")
        print("=" * 50)

        print(f"\n  ODE class: {self.ode_class.value}")
        print(f"  Level:     {self.level.value if self.level else '?'}")
        print(f"  Field:     {self.field_type}")

        print(f"\n  P = ", end=""); pprint(self.P)
        print(f"  Q = ", end=""); pprint(self.Q)

        if self.P_s is not None:
            print(f"\n  P (s-coords) = ", end=""); pprint(self.P_s)
            print(f"  Q (s-coords) = ", end=""); pprint(self.Q_s)

        print(f"\n  Riccati: dz/dx = ", end=""); pprint(self.riccati)

        print(f"\n  Fixed points:")
        for i, fp in enumerate(self.fixed_points):
            print(f"    z*_{i+1} = ", end=""); pprint(fp)

        if self.fixed_points_s:
            print(f"\n  Fixed points (s-coords):")
            for i, fp in enumerate(self.fixed_points_s):
                print(f"    z_s*_{i+1} = ", end=""); pprint(fp)

        if self.autonomising is not None:
            print(f"\n  Autonomising coordinate: s = ", end="")
            pprint(self.autonomising)

        if self.solutions:
            label = "Solutions"
            if self.level == Level.L2 and self.field_type == 'real':
                label = "Solutions (WKB, approximate)"
            elif self.field_type == 'complex':
                label = "Solutions (exact, on C)"
            print(f"\n  {label}:")
            for i, sol in enumerate(self.solutions):
                print(f"    y_{i+1} = ", end=""); pprint(sol)

        if self.asymptotics:
            print(f"\n  Asymptotics:")
            for regime, forms in self.asymptotics.items():
                print(f"    {regime}:")
                for f in forms:
                    print(f"      ", end=""); pprint(f)

        if self.residuals:
            print(f"\n  Symbolic verification:")
            for i, r in enumerate(self.residuals):
                status = "PASS" if r == 0 else f"FAIL ({r})"
                print(f"    y_{i+1}: {status}")

        if self.numerical is not None:
            n = self.numerical
            print(f"\n  Numerical verification (scipy):")
            print(f"    z0 = {n['z0_used']:.4f}")
            print(f"    median |dz/dx - RHS|: {n['median_residual']:.2e}")
            print(f"    p95 |dz/dx - RHS|:    {n['p95_residual']:.2e}")

        if self.notes:
            print(f"\n  Notes:")
            for note in self.notes:
                print(f"    {note}")
        print()


def riccati_solve(ode_expr, x, y_func=None, coords=None,
                  field='real', coupled=False,
                  hint=None, verify=True,
                  numerical=False, x_range=None, z0=None):
    """
    Solve a second-order linear ODE via Riccati flow.

    Parameters
    ----------
    ode_expr : sympy expression
        LHS of the ODE = 0.

    x : Symbol
        Independent variable.

    y_func : optional
        Dependent function. Auto-detected if None.

    coords : optional sympy expression
        Autonomising coordinate, e.g. sp.ln(1+u).

    field : 'real' or 'complex'
        'real'    — symbolic solutions on R (default)
        'complex' — symbolic solutions on C.
                    Fixed points are complex; integrate them,
                    exponentiate, split into Re(y) and Im(y).
                    No scipy involved.

    coupled : bool
        If True, the ODE comes from a coupled-coordinate
        metric. Implies field='complex' behaviour.

    hint : optional dict
        'level': int, 'coupling': expr, 'ode_class': str

    verify : bool
        Symbolically verify solutions (default True).

    numerical : bool
        Run scipy Riccati integration as a secondary check.
        Purely a verification tool. Default False.

    x_range : (float, float)
        Domain for numerical verification. Required if numerical=True.

    z0 : complex
        Initial z for numerical verification. Auto-picked if None.
    """
    if hint is None:
        hint = {}

    if coupled:
        field = 'complex'

    result = RiccatiResult()
    result.field_type = field
    z = Symbol('z')

    # 1. Extract P, Q
    P, Q = extract_P_Q(ode_expr, x, y_func)
    result.P = P
    result.Q = Q

    # 2. Classify
    if 'ode_class' in hint:
        result.ode_class = ODEClass(hint['ode_class'])
    else:
        result.ode_class = classify_ode(P, Q, x)

    # 3. Riccati
    result.riccati = -z**2 - P*z - Q

    # 4. Fixed points in original coords
    fps = solve(z**2 + P*z + Q, z)
    result.fixed_points = [simplify(fp) for fp in fps]

    # 5. Autonomising coordinate
    auto = find_autonomising(P, Q, x, coords)
    result.autonomising = auto
    result.coords = coords

    # 6. Transform to s-coords if available
    fps_for_recovery = result.fixed_points
    if auto is not None and auto != x:
        P_s, Q_s, ds_dx = transform_to_coords(P, Q, x, auto)
        result.P_s = P_s
        result.Q_s = Q_s
        fps_s = solve(z**2 + P_s*z + Q_s, z)
        fps_s = [simplify(fp) for fp in fps_s]
        result.fixed_points_s = fps_s
        fps_for_recovery = fps_s

    # 7. Level
    result.level = classify_level(P, Q, result.fixed_points, x, hint)

    # 8. Solve (symbolic)
    if field == 'complex':
        result.solutions, result.asymptotics = recover_complex(
            fps_for_recovery, auto, P, Q, x, result.level
        )
    else:
        result.solutions, result.asymptotics = recover_real(
            fps_for_recovery, auto, P, Q, x, result.level
        )

    # 9. Symbolic verification
    if verify and result.solutions:
        for i, sol in enumerate(result.solutions):
            try:
                res = verify_solution(sol, P, Q, x)
                result.residuals.append(res)
                if res == 0:
                    result.notes.append(
                        f"y_{i+1}: verified (residual = 0)")
                else:
                    result.notes.append(f"y_{i+1}: nonzero residual")
            except Exception as e:
                result.residuals.append(None)
                result.notes.append(
                    f"y_{i+1}: verification failed ({e})")

    # 10. Numerical verification (optional, scipy)
    if numerical:
        if x_range is None:
            result.notes.append(
                "numerical=True but no x_range given. Skipped.")
        else:
            try:
                nv = numerical_verify(P, Q, x, x_range, z0)
                result.numerical = nv
            except Exception as e:
                result.notes.append(f"Numerical verification failed: {e}")

    # 11. Notes
    level_desc = {
        Level.L0: "Globally autonomous. Elementary solutions.",
        Level.L1: "Single autonomising coordinate.",
        Level.L2: "Two incompatible regimes on R." if field == 'real'
                  else "Solved on C via complex fixed points.",
    }
    if result.level in level_desc:
        result.notes.append(level_desc[result.level])

    if 'coupling' in hint:
        result.notes.append(
            f"Coupled manifold: g^uv = {hint['coupling']}. "
            f"Im(z*) pinned constant."
        )

    return result
