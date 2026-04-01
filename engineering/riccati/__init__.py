"""
riccati — ODE solver via Riccati flow
======================================

Two symbolic modes:

    # on R (default)
    r = riccati_solve(ode, x)
    y1, y2 = r.to_exprs()

    # on C (exact closed forms via complex fixed points)
    r = riccati_solve(ode, x, field='complex')
    y1, y2 = r.to_exprs()

Optional numerical verification:

    r = riccati_solve(ode, x, numerical=True, x_range=(0.5, 30))
    print(r.numerical['median_residual'])
"""

from .solver import riccati_solve, RiccatiResult
from .classify import ODEClass, Level

__all__ = ['riccati_solve', 'RiccatiResult', 'ODEClass', 'Level']
__version__ = '0.4.0'
