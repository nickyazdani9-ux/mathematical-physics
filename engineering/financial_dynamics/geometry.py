"""Geometric primitives; exact frozen generators, not exact nonlinear solvers."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class PhaseState:
    phase: float
    turns: int = 0

    def __post_init__(self):
        if not math.isfinite(self.phase) or not 0 <= self.phase < math.tau:
            raise ValueError('phase must be finite in [0, 2*pi)')
        if not isinstance(self.turns, int):
            raise ValueError('turns must be an integer')

    def advance(self, increment):
        if not math.isfinite(increment):
            raise ValueError('finite increment required')
        # Keep the large integer count separate from the fractional phase.
        extra, phase = divmod(self.phase + increment, math.tau)
        return PhaseState(float(phase), self.turns + int(extra))

    def unit(self):
        return complex(math.cos(self.phase), math.sin(self.phase))


def _unit(h):
    h = np.asarray(h, dtype=complex)
    if h.shape != (2,) or not np.isfinite(h).all() or np.max(np.abs(h)) == 0:
        raise ValueError('nonzero finite two-component homogeneous state required')
    h = h / np.max(np.abs(h))
    return h / np.linalg.norm(h)


def sphere_readout(h):
    u, v = _unit(h)
    cross = np.conj(u)*v
    return np.array([2*cross.real, 2*cross.imag, abs(u)**2-abs(v)**2])


def projective_step(h, a, b, c, dt):
    """Advance z=v/u with z'=a+b*z+c*z**2, through poles in CP^1.

    exp(dt*M) is exact only when a,b,c are constant on this interval.
    State-dependent coefficients frozen each step give a first-order scheme.
    """
    h = _unit(h)
    if not np.isfinite([a,b,c,dt]).all() or dt < 0:
        raise ValueError('finite coefficients and nonnegative dt required')
    generator = np.array([[-b/2,-c],[a,b/2]], dtype=complex)
    if np.linalg.norm(generator)*dt > 100:
        raise ValueError('generator step too large; subdivide and test convergence')
    return _unit(expm(dt*generator) @ h)


def selective_step(phase, h, observation, dt, base_frequency=.13):
    """Fixed-parameter research cell; no learned dynamics or trading semantics.

    Selection and cross-state coupling use the already available observation.
    Bounded sphere features avoid division at projective chart poles.
    """
    if not np.isfinite([observation, dt, base_frequency]).all() or dt < 0:
        raise ValueError('invalid observation or interval')
    x = np.tanh(observation)
    old = sphere_readout(h)
    gate = 1/(1+np.exp(-2*x))
    velocity = base_frequency*(1+.25*x) + .03*old[0]
    phase = phase.advance(float(velocity*dt))
    a = .1*gate*x + .05j*phase.unit()
    b = -.2 - .1*gate
    c = .08*np.sin(phase.phase) + .03j*old[1]
    h = projective_step(h,a,b,c,dt)
    features = np.r_[phase.unit().real,phase.unit().imag,sphere_readout(h),gate,x]
    return phase,h,features
